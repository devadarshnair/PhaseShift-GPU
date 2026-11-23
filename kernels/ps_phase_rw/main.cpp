// main.cpp -- ps_phase_rw host app: writes [dcr_phase] configs between launches.
//
// Usage: ps_phase_rw [-L launches] [-n elements] [-c configs|default|none] [-w queue|sync|alt] [-k kernel.vxbin]
//   -L  number of launches (default 5)
//   -c  configs (default): launch 0 gets the all-zero default, launches 1..L-2 get distinct non-default patterns that
//       exercise every field (incl. the HI words of the 64-bit ranges), and the last launch (L >= 3) the default
//       again, so a return to zero is checked too; with L = 5 that is 4 distinct configs over 5 launches.
//       default: the all-zero block before every launch (DCR write traffic, stock config).
//       none: nothing is written (the app is then a plain kernel app; device config stays at reset = default).
//   -w  how the block is written (kernels/common/ps_phase.h): queue = ps::phase_enqueue on the launch queue with no
//       host wait (the strongest ordering check: the writes for launch k+1 are enqueued before launch k has run);
//       sync = ps::phase_write (vx_dcr_write) after waiting for every earlier launch; alt (default) = queue for even
//       launches, sync for odd ones.
// For every launch k the app prints "PSPHASE-HOST launch=k <config>": the config that must be in force during launch
// k. With PS_DUMP_PHASE=1, the ps-tree SimX prints "PSPHASE-DUMP launch=k consumer=<name> <config>" for every
// consumer at the first cycle after launch k's reset; tests/test_ps_phase.py checks that they are equal. Every launch
// computes y = x * a_k + b_k (uint32) with launch-specific a_k, b_k and is checked on the host.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <vortex2.h>

#include "common.h"
#include "ps_common.h"
#include "ps_phase.h"

namespace {

struct Ctx {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  vx_buffer_h x_buf = nullptr, y_buf = nullptr;
  std::vector<vx_event_h> events;
  ~Ctx() {
    if (!device) return;
    for (auto e : events) vx_event_release(e);
    if (x_buf) vx_buffer_release(x_buf);
    if (y_buf) vx_buffer_release(y_buf);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);  // PERF: lines (cumulative over all launches)
    vx_device_release(device);
  }
};

#define RT_CHECK(_expr)                                                       \
  do {                                                                        \
    int _ret = (int)(_expr);                                                  \
    if (_ret != 0) {                                                          \
      std::printf("Error: '%s' returned %d!\nFAILED!\n", #_expr, _ret);       \
      return 1;                                                               \
    }                                                                         \
  } while (false)

// Non-default pattern p >= 1: every field distinct per p, 64-bit ranges with non-zero HI words.
ps::PhaseCfg pattern(uint32_t p) {
  ps::PhaseCfg c;
  c.mode = p;
  c.flags = p << 16;  // reserved bits only: bit 0 (persist_l2) stays 0
  c.l1_waymask = (0x1111u * p) & 0xffffu;
  c.l2_waymask = 0xA0u + p;
  c.insert_policy = 0x10u + p;
  for (uint32_t i = 0; i < 8; ++i) c.dxa_depth[i] = uint8_t(16 * p + i + 1);
  c.dram_sched = p;
  c.dram_aging = 1000u * p;
  c.kv_base[0] = (uint64_t(p) << 32) | (0x80000000u + 0x40u * p);
  c.kv_limit[0] = c.kv_base[0] + 0x10000u * p;
  c.kv_base[1] = 0x20000000u + 0x1000u * p;
  c.kv_limit[1] = c.kv_base[1] + 0x2000u;
  c.wgt_base = (uint64_t(0xffffu) << 32) + p;
  c.wgt_limit = ~uint64_t(0) - p;
  return c;
}

void usage(const char* app) {
  std::printf("Usage: %s [-L launches] [-n elements] [-c configs|default|none] [-w queue|sync|alt] [-k kernel.vxbin]\n",
              app);
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t launches = 5, n = 256;
  std::string cmode = "configs", wmode = "alt", kernel_file = "kernel.vxbin";
  int opt;
  while ((opt = getopt(argc, argv, "L:n:c:w:k:h")) != -1) {
    switch (opt) {
      case 'L': launches = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': n = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'c': cmode = optarg; break;
      case 'w': wmode = optarg; break;
      case 'k': kernel_file = optarg; break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  if (launches < 1 || n < 1 || (cmode != "configs" && cmode != "default" && cmode != "none") ||
      (wmode != "queue" && wmode != "sync" && wmode != "alt")) {
    usage(argv[0]);
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("ps_phase_rw: launches=%u n=%u configs=%s write=%s\n", launches, n, cmode.c_str(), wmode.c_str());

  // ---- per-launch configs and kernel arguments
  std::vector<ps::PhaseCfg> cfg(launches);
  if (cmode == "configs")
    for (uint32_t k = 1; k < launches; ++k)
      if (!(launches >= 3 && k == launches - 1)) cfg[k] = pattern(k);
  std::vector<uint32_t> x(n);
  ps::Rng rng(7);
  for (auto& v : x) v = rng.next();

  Ctx ctx;
  RT_CHECK(vx_device_open(0, &ctx.device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  RT_CHECK(vx_queue_create(ctx.device, &qi, &ctx.queue));
  const uint64_t bytes = (uint64_t)n * sizeof(uint32_t);
  std::vector<ps_phase_rw_arg_t> args(launches);
  RT_CHECK(vx_buffer_create(ctx.device, bytes, VX_MEM_READ, &ctx.x_buf));
  RT_CHECK(vx_buffer_create(ctx.device, bytes, VX_MEM_READ_WRITE, &ctx.y_buf));
  uint64_t x_addr = 0, y_addr = 0;
  RT_CHECK(vx_buffer_address(ctx.x_buf, &x_addr));
  RT_CHECK(vx_buffer_address(ctx.y_buf, &y_addr));
  for (uint32_t k = 0; k < launches; ++k)
    args[k] = ps_phase_rw_arg_t{x_addr, y_addr, n, 2654435761u + 2 * k, 0x1000u * (k + 1)};
  vx_event_h wev = nullptr;
  RT_CHECK(vx_enqueue_write(ctx.queue, ctx.x_buf, 0, x.data(), bytes, 0, nullptr, &wev));
  ctx.events.push_back(wev);
  RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
  RT_CHECK(vx_module_get_kernel(ctx.module_, "phase_rw", &ctx.kernel));
  uint32_t global = n, grid_dim[1], block_dim[1];
  RT_CHECK(vx_device_max_occupancy_grid(ctx.device, 1, &global, grid_dim, block_dim));
  std::printf("launch: grid=%u block=%u\n", grid_dim[0], block_dim[0]);

  // ---- launches; nothing waits between queue-mode launches
  std::vector<std::vector<uint32_t>> y(launches, std::vector<uint32_t>(n, 0));
  std::vector<vx_event_h> read_ev(launches, nullptr);
  ps::Timer timer;
  for (uint32_t k = 0; k < launches; ++k) {
    if (cmode != "none") {
      bool sync = (wmode == "sync") || (wmode == "alt" && (k % 2) == 1);
      if (sync) {
        for (auto e : ctx.events) RT_CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE));
        RT_CHECK(ps::phase_write(ctx.device, cfg[k]));
      } else {
        RT_CHECK(ps::phase_enqueue(ctx.queue, cfg[k]));
      }
    }
    ps::phase_print(k, cfg[k]);
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = ctx.kernel;
    li.args_host = &args[k];
    li.args_size = sizeof(args[k]);
    li.ndim = 1;
    li.grid_dim[0] = grid_dim[0];
    li.block_dim[0] = block_dim[0];
    vx_event_h lev = nullptr;
    RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &lev));
    ctx.events.push_back(lev);
    RT_CHECK(vx_enqueue_read(ctx.queue, y[k].data(), ctx.y_buf, 0, bytes, 1, &lev, &read_ev[k]));
    ctx.events.push_back(read_ev[k]);
  }
  for (auto e : ctx.events) RT_CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE));

  // ---- check every launch's output
  uint32_t errors = 0;
  for (uint32_t k = 0; k < launches; ++k) {
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; ++i)
      if (y[k][i] != x[i] * args[k].a + args[k].b) ++bad;
    std::printf("PSCHECK: launch%u errors=%u n=%u\n", k, bad, n);
    errors += bad;
  }
  std::printf("Elapsed: %.0f ms\n", timer.ms());
  if (errors != 0) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
