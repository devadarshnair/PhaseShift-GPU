// main.cpp -- ps_memprobe host app: machine-parameter microbenchmarks of a PS-BASE configuration.
//
// Usage: ps_memprobe -m chase -s <footprint bytes> [-w warm_hops] [-n timed_hops] [-o rand|seq] [-r seed]
//        ps_memprobe -m stream -s <bytes> [-x threads_per_core_multiplier]
//        ps_memprobe -m fma -n <iterations> [-x multiplier]
//        [-k kernel.vxbin]
//
// chase  The footprint is split into 64 B nodes (one cache line each, VX_CFG_MEM_BLOCK_SIZE = 64). `rand` links them
//        in one random cycle (Sattolo's algorithm, seed -r), so consecutive hops land on unrelated lines and DRAM
//        rows (row conflicts); `seq` links node i -> i+1 (open-row hits, no prefetcher exists in Vortex). One thread
//        does `warm` untimed-for-the-result hops (they fill the caches; their cycles are reported separately) and
//        then `timed` hops. Interpretation (modeling choice, explained in docs/methodology.md §11):
//          * footprint >> L2 and warm = 0: every timed hop is the first touch of its line in this launch (caches are
//            cold at every launch, errata 12/24), so cycles/hop = full DRAM-miss load-to-use latency L0;
//          * L1 < footprint < L2 with warm = footprint lines: the warm pass misses to DRAM (cold) and the timed pass
//            hits in L2 (except for the share that still hits in L1, footprint-dependent), so cycles/hop = L2 hit
//            latency L_L2, and warm cycles/hop = L0 again;
//          * footprint << L1: cycles/hop = L1 hit latency.
//        Every figure includes one integer add per hop (the address computation), i.e. it is a load-to-use latency.
//        The host replays the chain and checks the final offset (PASS) so a broken chain cannot report numbers.
// stream T = cores x warps x NT x multiplier threads read the array (uint32, values = index * 2654435761) with a grid
//        stride and sum it; the host checks every thread's sum. Bytes/cycle = bytes / global PERF cycles.
// fma    T threads x 8 independent fused multiply-add chains x n iterations; host recomputes every thread's result
//        with std::fma (bit-identical: one rounding per step in both). FLOP/cycle = 16 n T / global PERF cycles.
// Output: one `PSPROBE: key=value ...` line, then PASSED!/FAILED!.
#include <cinttypes>

#include "common.h"
#include "elt_host.h"

static void usage(const char* app) {
  std::printf("Usage: %s -m chase|stream|fma [-s bytes] [-w warm] [-n steps] [-o rand|seq] [-r seed] [-x mult] "
              "[-k kernel.vxbin]\n", app);
}

static int launch(ps_elt::Dev& dev, const ps_memprobe_arg_t& arg, uint32_t grid, uint32_t block, vx_event_h* ev) {
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = dev.kernel;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = grid;
  li.block_dim[0] = block;
  std::printf("launch: grid=%u block=%u NT=%u\n", grid, block, (unsigned)dev.nt);
  PS_RT_CHECK(vx_enqueue_launch(dev.queue, &li, 0, nullptr, ev));
  return 0;
}

int main(int argc, char** argv) {
  std::string mode = "chase", order = "rand", kernel_file = "kernel.vxbin";
  uint64_t bytes = 1 << 20;
  uint32_t warm = 0, steps = 4096, seed = 1, mult = 1;
  int c;
  while ((c = getopt(argc, argv, "m:s:w:n:o:r:x:k:h")) != -1) {
    switch (c) {
      case 'm': mode = optarg; break;
      case 's': bytes = std::strtoull(optarg, nullptr, 10); break;
      case 'w': warm = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': steps = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'o': order = optarg; break;
      case 'r': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'x': mult = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  if (mode != "chase" && mode != "stream" && mode != "fma") { usage(argv[0]); return 1; }
  ps_elt::Dev dev;
  if (dev.open(kernel_file, mode.c_str())) return 1;
  uint64_t cores = 0, warps = 0;
  PS_RT_CHECK(vx_device_query(dev.device, VX_CAPS_NUM_CORES, &cores));
  PS_RT_CHECK(vx_device_query(dev.device, VX_CAPS_NUM_WARPS, &warps));
  ps::Timer timer;
  bool ok = true;
  ps_memprobe_arg_t arg = {};
  vx_event_h ev = nullptr;

  if (mode == "chase") {
    const uint32_t line = 64;
    const uint64_t nodes = bytes / line;
    if (nodes < 2 || bytes > (1ull << 31)) { std::printf("bad footprint\nFAILED!\n"); return 1; }
    std::vector<uint32_t> perm(nodes), mem(bytes / 4, 0);
    for (uint64_t i = 0; i < nodes; ++i) perm[i] = (uint32_t)i;
    if (order == "rand") {  // Sattolo: a uniformly random single cycle
      uint64_t s = 0x9E3779B97F4A7C15ull ^ seed;
      for (uint64_t i = nodes - 1; i > 0; --i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        uint64_t j = s % i;
        std::swap(perm[i], perm[j]);
      }
      // perm is now a cyclic permutation: node k -> perm[k]
      for (uint64_t k = 0; k < nodes; ++k) mem[k * (line / 4)] = perm[k] * line;
    } else if (order == "seq") {
      for (uint64_t k = 0; k < nodes; ++k) mem[k * (line / 4)] = (uint32_t)(((k + 1) % nodes) * line);
    } else { usage(argv[0]); return 1; }
    uint32_t p = 0;
    for (uint64_t i = 0; i < (uint64_t)warm + steps; ++i) p = mem[p / 4];
    ps_elt::Buf bm, bo;
    if (dev.alloc(bm, bytes, VX_MEM_READ, mem.data())) return 1;
    const std::vector<uint32_t> zero(4, 0);
    if (dev.alloc(bo, 16, VX_MEM_READ_WRITE, zero.data())) return 1;
    arg.buf_addr = bm.addr; arg.out_addr = bo.addr; arg.start = 0; arg.warm = warm; arg.steps = steps;
    if (launch(dev, arg, 1, 1, &ev)) return 1;
    uint32_t out[4];
    if (dev.read(out, bo, ev)) return 1;
    vx_event_release(ev);
    ok = (out[0] == p);
    std::printf("PSPROBE: mode=chase order=%s footprint=%" PRIu64 " warm=%u steps=%u timed_cycles=%u "
                "cyc_per_hop=%.3f warm_cycles=%u warm_cyc_per_hop=%.3f final=%u expect=%u\n",
                order.c_str(), bytes, warm, steps, out[1], steps ? (double)out[1] / steps : 0.0, out[2],
                warm ? (double)out[2] / warm : 0.0, out[0], p);
  } else {
    const uint32_t T = (uint32_t)(cores * warps * dev.nt * mult);
    uint32_t global = T, grid = 0, block = 0;
    PS_RT_CHECK(vx_device_max_occupancy_grid(dev.device, 1, &global, &grid, &block));
    const uint32_t threads = grid * block;
    ps_elt::Buf bs, bo;
    std::vector<uint32_t> out(threads, 0xFFFFFFFFu);
    if (mode == "stream") {
      const uint32_t n = (uint32_t)(bytes / 4);
      std::vector<uint32_t> src(n);
      for (uint32_t i = 0; i < n; ++i) src[i] = i * 2654435761u;
      if (dev.alloc(bs, (uint64_t)n * 4, VX_MEM_READ, src.data())) return 1;
      if (dev.alloc(bo, (uint64_t)threads * 4, VX_MEM_READ_WRITE, out.data())) return 1;
      arg.buf_addr = bs.addr; arg.out_addr = bo.addr; arg.steps = n;
      if (launch(dev, arg, grid, block, &ev)) return 1;
      if (dev.read(out.data(), bo, ev)) return 1;
      vx_event_release(ev);
      size_t bad = 0;
      for (uint32_t g = 0; g < threads; ++g) {
        uint32_t s = 0;
        for (uint32_t j = g; j < n; j += threads) s += src[j];
        bad += (s != out[g]);
      }
      ok = (bad == 0);
      std::printf("PSPROBE: mode=stream bytes=%" PRIu64 " threads=%u cores=%" PRIu64 " warps=%" PRIu64
                  " bad=%zu\n", (uint64_t)n * 4, threads, cores, warps, bad);
    } else {
      if (dev.alloc(bo, (uint64_t)threads * 4, VX_MEM_READ_WRITE, out.data())) return 1;
      arg.out_addr = bo.addr; arg.steps = steps;
      if (launch(dev, arg, grid, block, &ev)) return 1;
      std::vector<float> y(threads);
      if (dev.read(y.data(), bo, ev)) return 1;
      vx_event_release(ev);
      size_t bad = 0;
      const float b = 0.999f, cc = 1.0e-3f;
      for (uint32_t g = 0; g < threads; ++g) {
        float a[8];
        a[0] = 1.0f + (float)(g & 1023u) * 0.0009765625f;  // exact, as on the device
        a[1] = a[0] + 0.1f; a[2] = a[0] + 0.2f; a[3] = a[0] + 0.3f; a[4] = a[0] + 0.4f;
        a[5] = a[0] + 0.5f; a[6] = a[0] + 0.6f; a[7] = a[0] + 0.7f;
        for (uint32_t i = 0; i < steps; ++i)
          for (int k = 0; k < 8; ++k) a[k] = std::fma(a[k], b, cc);
        float r = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
        bad += (std::memcmp(&r, &y[g], 4) != 0);
      }
      ok = (bad == 0);
      std::printf("PSPROBE: mode=fma iters=%u threads=%u cores=%" PRIu64 " warps=%" PRIu64 " flop=%.0f bad=%zu\n",
                  steps, threads, cores, warps, 16.0 * steps * threads, bad);
    }
  }
  return ps_elt::finish(ok, timer);
}
