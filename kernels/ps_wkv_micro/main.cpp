// main.cpp -- ps_wkv_micro host app (mechanism D microbenchmark; design in kernel.cpp).
//
// Usage: ps_wkv_micro [-w <weight KiB>] [-c <KV KiB per round>] [-r <rounds>] [-R none|ranges] [-I <insert word>]
//                     [-s <seed>] [-g <grid>] [-k kernel.vxbin]
//   -w   weight block size in KiB (default 512 = 1/2 of the PS-BASE-S1-4c 1 MiB L2)
//   -c   KV bytes read per round, KiB (default 1024; total KV = rounds x this, each byte read once)
//   -r   rounds (default 16): the weight block is read `rounds` times, so an ideal L2 hit rate of the weight
//        region is (rounds - 1) / rounds (0.9375 at 16; the first round always misses, caches are cold, §1)
//   -R   ranges (default): program [dcr_phase] wgt = the weight buffer and kv[0] = the KV buffer, so SimX
//        classifies the loads WEIGHT / KV_STREAM (methodology §12.3); none: no [dcr_phase] write at all
//   -I   [dcr_phase] insert policy word written with the ranges (PhaseConfig::INS_*, methodology §14.D; default 0)
//   -g   override the number of blocks (default: max occupancy for one thread per hardware thread)
//   -W   <l1word>:<l2word> [dcr_phase] way-mask words (mechanism A2, ps patch waymask, encoding in
//        ps_waymask.h / methodology §14.A), written with the ranges; absent = no change to the default path
//   -X   (with -W) also exercise the host helper's refusal: inside a declared schedule, a second mask change must
//        be refused (prints PSWAYMASK-REFUSED rc=<code>; the masks of -W stay in force)
// Exactly one launch. Check: every thread's 32-bit wrap-around sum equals the host's (exact). Prints PSRANGE (the
// buffer addresses, for PS_SCHEDULE files), PSWKV (sizes), PASSED!/FAILED!.
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
#include "ps_waymask.h"

#define RT_CHECK(_expr)                                                     \
  do {                                                                      \
    int _ret = (int)(_expr);                                                \
    if (_ret != 0) {                                                        \
      std::printf("Error: '%s' returned %d!\nFAILED!\n", #_expr, _ret);     \
      return 1;                                                             \
    }                                                                       \
  } while (false)

int main(int argc, char** argv) {
  uint32_t w_kib = 512, kv_kib = 1024, rounds = 16, seed = 1, grid_override = 0;
  uint32_t insert = 0;
  std::string range_mode = "ranges", kernel_file = "kernel.vxbin";
  bool have_wm = false, wm_refusal_demo = false;
  uint32_t wm_l1 = 0, wm_l2 = 0;
  int c;
  while ((c = getopt(argc, argv, "w:c:r:R:I:s:g:k:W:Xh")) != -1) {
    switch (c) {
      case 'w': w_kib = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'c': kv_kib = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'r': rounds = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'R': range_mode = optarg; break;
      case 'I': insert = (uint32_t)std::strtoul(optarg, nullptr, 0); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'g': grid_override = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      case 'W':
        if (!ps::parse_waymask_arg(optarg, &wm_l1, &wm_l2)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        have_wm = true;
        break;
      case 'X': wm_refusal_demo = true; break;
      default: std::printf("usage: see main.cpp\n"); return c == 'h' ? 0 : 1;
    }
  }
  if (range_mode != "ranges" && range_mode != "none") { std::printf("bad -R\nFAILED!\n"); return 1; }
  if (w_kib == 0 || kv_kib == 0 || rounds == 0) { std::printf("sizes must be > 0\nFAILED!\n"); return 1; }
  if (range_mode == "none" && insert != 0) { std::printf("-I needs -R ranges\nFAILED!\n"); return 1; }
  if (have_wm && range_mode == "none") { std::printf("-W needs -R ranges\nFAILED!\n"); return 1; }
  if (wm_refusal_demo && !have_wm) { std::printf("-X needs -W\nFAILED!\n"); return 1; }
  const uint32_t w_vec = w_kib * 64, kv_vec = kv_kib * 64;   // 16-byte vectors per KiB = 64
  const uint64_t w_words = 4ull * w_vec, kv_words = 4ull * kv_vec * rounds;
  std::vector<uint32_t> W(w_words), KV(kv_words);
  ps::Rng rng(seed);
  for (auto& v : W) v = rng.next();
  for (auto& v : KV) v = rng.next();

  vx_device_h dev = nullptr;
  vx_queue_h q = nullptr;
  RT_CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  RT_CHECK(vx_queue_create(dev, &qi, &q));
  uint64_t nt = 0;
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));

  // One resident thread per hardware thread (cores x warps x NT), so every round sweeps the WHOLE weight block and
  // KV chunk across the device before the next round starts: the weight block's reuse distance is then global
  // (weight + KV chunk bytes), not the few KiB of one CTA (which the L1 would absorb).
  uint64_t ncores = 0, nwarps = 0;
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &ncores));
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nwarps));
  uint32_t global = (uint32_t)(ncores * nwarps * nt);
  uint32_t grid_dim[1], block_dim[1];
  RT_CHECK(vx_device_max_occupancy_grid(dev, 1, &global, grid_dim, block_dim));
  if (grid_override > 0) grid_dim[0] = grid_override;
  const uint32_t nthr = grid_dim[0] * block_dim[0];

  vx_buffer_h wb = nullptr, kb = nullptr, ob = nullptr;
  ps_wkv_arg_t arg = {};
  arg.w_vec = w_vec; arg.kv_vec = kv_vec; arg.rounds = rounds;
  RT_CHECK(vx_buffer_create(dev, w_words * 4, VX_MEM_READ, &wb));
  RT_CHECK(vx_buffer_address(wb, &arg.w_addr));
  RT_CHECK(vx_buffer_create(dev, kv_words * 4, VX_MEM_READ, &kb));
  RT_CHECK(vx_buffer_address(kb, &arg.kv_addr));
  RT_CHECK(vx_buffer_create(dev, (uint64_t)nthr * 4, VX_MEM_READ_WRITE, &ob));
  RT_CHECK(vx_buffer_address(ob, &arg.out_addr));
  RT_CHECK(vx_enqueue_write(q, wb, 0, W.data(), w_words * 4, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(q, kb, 0, KV.data(), kv_words * 4, 0, nullptr, nullptr));
  std::printf("PSWKV: w_bytes=%llu kv_bytes_per_round=%llu rounds=%u kv_bytes=%llu grid=%u block=%u NT=%u\n",
              (unsigned long long)(w_words * 4), (unsigned long long)kv_vec * 16, rounds,
              (unsigned long long)(kv_words * 4), grid_dim[0], block_dim[0], (unsigned)nt);
  if (range_mode == "ranges") {
    ps::PhaseCfg pc;
    pc.wgt_base = arg.w_addr;
    pc.wgt_limit = arg.w_addr + w_words * 4;
    pc.kv_base[0] = arg.kv_addr;
    pc.kv_limit[0] = arg.kv_addr + kv_words * 4;
    pc.insert_policy = insert;
    if (have_wm) {   // PhaseShift (patch waymask): masks for this (the only) launch, set at its boundary
      ps::WayMaskCtl wmc;
      RT_CHECK(wmc.set(pc, wm_l1, wm_l2));
      if (wm_refusal_demo) {
        wmc.begin_schedule();
        const int rr = wmc.set(pc, wm_l1 ^ 1u, wm_l2 ^ 1u);   // any different mask: must be refused
        std::printf("PSWAYMASK-REFUSED rc=%d l1=0x%x l2=0x%x\n", rr, pc.l1_waymask, pc.l2_waymask);
        if (rr != ps::kWaymaskRefused || pc.l1_waymask != wm_l1 || pc.l2_waymask != wm_l2) {
          std::printf("helper accepted a mask change inside a schedule\nFAILED!\n");
          return 1;
        }
        wmc.end_schedule();
      }
      ps::waymask_print(wmc);
    }
    RT_CHECK(ps::phase_enqueue(q, pc));
    std::printf("PSRANGE: wgt=0x%llx-0x%llx kv0=0x%llx-0x%llx insert=0x%x\n", (unsigned long long)pc.wgt_base,
                (unsigned long long)pc.wgt_limit, (unsigned long long)pc.kv_base[0],
                (unsigned long long)pc.kv_limit[0], insert);
  }

  vx_module_h mod = nullptr;
  vx_kernel_h ker = nullptr;
  RT_CHECK(vx_module_load_file(dev, kernel_file.c_str(), &mod));
  RT_CHECK(vx_module_get_kernel(mod, "wkv", &ker));
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = ker;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = grid_dim[0];
  li.block_dim[0] = block_dim[0];
  std::vector<uint32_t> out(nthr, 0);
  vx_event_h lev = nullptr, rev = nullptr;
  RT_CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &lev));
  RT_CHECK(vx_enqueue_read(q, out.data(), ob, 0, (uint64_t)nthr * 4, 1, &lev, &rev));
  RT_CHECK(vx_event_wait_value(rev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(rev);
  vx_event_release(lev);

  // host reference: the same per-thread wrap-around sums
  uint32_t bad = 0;
  for (uint32_t t = 0; t < nthr; ++t) {
    uint32_t acc = 0;
    for (uint32_t r = 0; r < rounds; ++r) {
      for (uint32_t i = t; i < w_vec; i += nthr)
        acc += W[4 * (size_t)i] + W[4 * (size_t)i + 1] + W[4 * (size_t)i + 2] + W[4 * (size_t)i + 3];
      const size_t k0 = (size_t)r * kv_vec;
      for (uint32_t i = t; i < kv_vec; i += nthr) {
        const size_t j = 4 * (k0 + i);
        acc += KV[j] + KV[j + 1] + KV[j + 2] + KV[j + 3];
      }
    }
    if (acc != out[t]) ++bad;
  }
  std::printf("PSCHECK: wkv_sums mismatches=%u of %u\n", bad, nthr);
  vx_buffer_release(wb);
  vx_buffer_release(kb);
  vx_buffer_release(ob);
  vx_kernel_release(ker);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);
  if (bad) { std::printf("FAILED!\n"); return 1; }
  std::printf("PASSED!\n");
  return 0;
}
