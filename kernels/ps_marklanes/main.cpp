// main.cpp -- ps_marklanes host app (carry-over 3; see kernel.cpp).
//
// Usage: ps_marklanes [-g grid] [-w warps per CTA] [-r reps] [-m mode] [-k kernel.vxbin]
// Launches `grid` CTAs of `w * NUM_THREADS` threads (full warps), so the device runs grid * w warps, each executing
// `reps` mark instructions. Prints "PSMARKLANES expect=<grid*w*reps> type=<mark type> ..." -- the count SimX must
// report for that mark type (VX_CSR_PS_MARK, one unit per warp instruction; MPM class 8 marks_* with --perf=8, or the
// PSMARK-TOTAL line with PS_DEBUG_CLASS=1). The checked result (out[i] = i + 1) makes the run PASS independently of
// the counts; tests/test_pstrace.py compares the counts.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <vortex2.h>

#include "common.h"
#include "ps_common.h"
#include "ps_hint.h"

namespace {

struct Ctx {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  vx_buffer_h out_buf = nullptr;
  ~Ctx() {
    if (!device) return;
    if (out_buf) vx_buffer_release(out_buf);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
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

}  // namespace

int main(int argc, char** argv) {
  uint32_t grid = 4, warps = 2, reps = 3, mode = 0;
  std::string kernel_file = "kernel.vxbin";
  int opt;
  while ((opt = getopt(argc, argv, "g:w:r:m:k:h")) != -1) {
    switch (opt) {
      case 'g': grid = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'w': warps = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'r': reps = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'm': mode = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      default:
        std::printf("Usage: %s [-g grid] [-w warps] [-r reps] [-m 0|1|2] [-k kernel.vxbin]\n", argv[0]);
        return opt == 'h' ? 0 : 1;
    }
  }
  if (grid < 1 || warps < 1 || mode > 2) {
    std::printf("bad arguments\nFAILED!\n");
    return 1;
  }
  Ctx ctx;
  RT_CHECK(vx_device_open(0, &ctx.device));
  uint64_t nt = 0, nw = 0, ncores = 0;
  RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_WARPS, &nw));
  RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_CORES, &ncores));
  if (warps > nw) {
    std::printf("warps per CTA %u > NUM_WARPS %u\nFAILED!\n", warps, (unsigned)nw);
    return 1;
  }
  const uint32_t block = warps * (uint32_t)nt, n = grid * block;
  const uint32_t type = (mode == 0) ? PS_MARK_OTHER : (mode == 1) ? PS_MARK_PREFILL_TILE : PS_MARK_DECODE_TILE;
  std::printf("PSMARKLANES expect=%u type=%u mode=%u grid=%u warps=%u reps=%u nt=%u cores=%u\n", grid * warps * reps,
              type, mode, grid, warps, reps, (unsigned)nt, (unsigned)ncores);
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  RT_CHECK(vx_queue_create(ctx.device, &qi, &ctx.queue));
  const uint64_t bytes = (uint64_t)n * sizeof(uint32_t);
  RT_CHECK(vx_buffer_create(ctx.device, bytes, VX_MEM_READ_WRITE, &ctx.out_buf));
  uint64_t out_addr = 0;
  RT_CHECK(vx_buffer_address(ctx.out_buf, &out_addr));
  ps_marklanes_arg_t arg{out_addr, reps, mode};
  RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
  RT_CHECK(vx_module_get_kernel(ctx.module_, "marklanes", &ctx.kernel));
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = ctx.kernel;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = grid;
  li.block_dim[0] = block;
  vx_event_h lev = nullptr, rev = nullptr;
  std::vector<uint32_t> out(n, 0);
  RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &lev));
  RT_CHECK(vx_enqueue_read(ctx.queue, out.data(), ctx.out_buf, 0, bytes, 1, &lev, &rev));
  RT_CHECK(vx_event_wait_value(rev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(lev);
  vx_event_release(rev);
  uint32_t bad = 0;
  for (uint32_t i = 0; i < n; ++i)
    if (out[i] != i + 1) ++bad;
  std::printf("PSCHECK: errors=%u n=%u\n", bad, n);
  if (bad != 0) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
