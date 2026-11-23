// main.cpp -- ps_mix_micro host app (mechanism B microbenchmark; design in kernel.cpp).
//
// Usage: ps_mix_micro [-r <rows>] [-c <cols>] [-b <bulk KiB>] [-t <tile words>] [-d <depth>] [-m mix|gemv|bulk]
//                     [-s <seed>] [-k kernel.vxbin]
//   -r/-c GEMV shape (default 1024 x 1024 uint32 = 4 MiB of DEMAND reads, read once; > the 1 MiB PS-BASE L2)
//   -b    bulk buffer size in KiB (default 4096 = 4 MiB of BULK_DXA reads, each byte copied once)
//   -t    words per DXA tile (default 256 = 1 KiB, the dxa_copy tile); -d tiles in flight per CTA (default 2)
//   -m    mix (default): both; gemv: no bulk traffic (DEMAND alone); bulk: the GEMV warps do nothing
//   -C    PhaseShift DXA stream class of the bulk descriptor (ps patch dxa-class, VX_PS_CLASS_* value in META bits
//         15:13; default 0 = unset = BULK_DXA, no extra DCR write); -P makes the bulk descriptor L2-prefetch-only (META
//         bit 16): no LMEM write, the bulk warp sums each tile from global memory after its barrier (same exact check)
// Exactly one launch, grid = one CTA per core, block = NW x NT. Check: every GEMV lane partial and every bulk-warp
// lane sum equals the host's (exact, uint32 wrap-around). Prints PSMIX (sizes), PSMIXCYC (per-class completion
// spans from the warps' mcycle stamps), PSCHECK, PASSED!/FAILED!.
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <vortex2.h>
#include <dxa.h>

#include "common.h"
#include "ps_common.h"

#define RT_CHECK(_expr)                                                     \
  do {                                                                      \
    int _ret = (int)(_expr);                                                \
    if (_ret != 0) {                                                        \
      std::printf("Error: '%s' returned %d!\nFAILED!\n", #_expr, _ret);     \
      return 1;                                                             \
    }                                                                       \
  } while (false)

int main(int argc, char** argv) {
  uint32_t rows = 1024, cols = 1024, bulk_kib = 4096, tw = 256, depth = 2, seed = 1;
  std::string mode = "mix", kernel_file = "kernel.vxbin";
  uint32_t ps_dxa_class = 0;   // PhaseShift (patch dxa-class): -C, -P
  bool ps_l2pf = false;
  int c;
  while ((c = getopt(argc, argv, "r:c:b:t:d:m:s:C:Pk:h")) != -1) {
    switch (c) {
      case 'r': rows = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'c': cols = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'b': bulk_kib = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 't': tw = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'd': depth = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'm': mode = optarg; break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      case 'C': ps_dxa_class = (uint32_t)std::strtoul(optarg, nullptr, 0); break;
      case 'P': ps_l2pf = true; break;
      default: std::printf("usage: see main.cpp\n"); return c == 'h' ? 0 : 1;
    }
  }
  if (mode != "mix" && mode != "gemv" && mode != "bulk") { std::printf("bad -m\nFAILED!\n"); return 1; }
  if (rows == 0 || cols == 0 || cols % 4 != 0) { std::printf("bad GEMV shape\nFAILED!\n"); return 1; }
  if (depth < 1 || depth > PS_MIX_MAX_DEPTH) { std::printf("bad -d\nFAILED!\n"); return 1; }
  if (ps_dxa_class >= VX_PS_NUM_CLASSES) { std::printf("bad -C\nFAILED!\n"); return 1; }

  vx_device_h dev = nullptr;
  vx_queue_h q = nullptr;
  RT_CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  RT_CHECK(vx_queue_create(dev, &qi, &q));
  uint64_t nt = 0, nw = 0, ncores = 0, isa = 0, lmem_cap = 0;
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &ncores));
  RT_CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));
  RT_CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  if (!(isa & VX_ISA_EXT_DXA) || nw < 2) { std::printf("device lacks DXA or has < 2 warps\nFAILED!\n"); return 1; }
  if (tw == 0 || tw % nt != 0 || (uint64_t)depth * tw * 4 > lmem_cap) {
    std::printf("tile words must be a multiple of NT and depth x tile must fit LMEM (%llu B)\nFAILED!\n",
                (unsigned long long)lmem_cap);
    return 1;
  }
  const uint32_t grid = (uint32_t)ncores, block = (uint32_t)(nw * nt);
  const uint32_t bulk_words_req = (mode == "gemv") ? 0 : bulk_kib * 256;
  const uint32_t bulk_tiles = bulk_words_req / tw, bulk_words = bulk_tiles * tw;
  const bool gemv_on = (mode != "bulk");

  std::vector<uint32_t> A((size_t)rows * cols), X(cols), B(std::max<uint32_t>(bulk_words, 1));
  ps::Rng rng(seed);
  for (auto& v : A) v = rng.next();
  for (auto& v : X) v = rng.next();
  for (auto& v : B) v = rng.next();

  vx_buffer_h ab = nullptr, xb = nullptr, pb = nullptr, bb = nullptr, sb = nullptr, cb = nullptr;
  ps_mix_arg_t arg = {};
  arg.rows = rows; arg.cols = cols; arg.bulk_tiles = bulk_tiles; arg.tile_words = tw; arg.depth = depth;
  arg.gemv_on = gemv_on ? 1 : 0;
  const uint64_t part_n = (uint64_t)rows * nt, bsum_n = (uint64_t)grid * nt, cyc_n = (uint64_t)grid * nw * 4;
  RT_CHECK(vx_buffer_create(dev, A.size() * 4, VX_MEM_READ, &ab));
  RT_CHECK(vx_buffer_address(ab, &arg.a_addr));
  RT_CHECK(vx_buffer_create(dev, X.size() * 4, VX_MEM_READ, &xb));
  RT_CHECK(vx_buffer_address(xb, &arg.x_addr));
  RT_CHECK(vx_buffer_create(dev, part_n * 4, VX_MEM_READ_WRITE, &pb));
  RT_CHECK(vx_buffer_address(pb, &arg.part_addr));
  RT_CHECK(vx_buffer_create(dev, B.size() * 4, VX_MEM_READ, &bb));
  RT_CHECK(vx_buffer_address(bb, &arg.bulk_addr));
  RT_CHECK(vx_buffer_create(dev, bsum_n * 4, VX_MEM_READ_WRITE, &sb));
  RT_CHECK(vx_buffer_address(sb, &arg.bsum_addr));
  RT_CHECK(vx_buffer_create(dev, cyc_n * 4, VX_MEM_READ_WRITE, &cb));
  RT_CHECK(vx_buffer_address(cb, &arg.cyc_addr));
  RT_CHECK(vx_enqueue_write(q, ab, 0, A.data(), A.size() * 4, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(q, xb, 0, X.data(), X.size() * 4, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(q, bb, 0, B.data(), B.size() * 4, 0, nullptr, nullptr));
  if (bulk_tiles > 0)
    RT_CHECK(vortex::dxa::program_1d(dev, PS_MIX_DXA_DESC, arg.bulk_addr, bulk_words, tw, 4));
  if (bulk_tiles > 0 && (ps_dxa_class != 0 || ps_l2pf)) {   // PhaseShift (patch dxa-class): re-pack META (rank 1)
    RT_CHECK(vortex::dxa::ps_set_stream(dev, PS_MIX_DXA_DESC, 1, 4, vortex::dxa::Layout::RowMajor, ps_dxa_class, ps_l2pf));
    std::printf("PSDXASTREAM: bulk_class=%u l2_only=%d\n", ps_dxa_class, ps_l2pf ? 1 : 0);
  }
  std::printf("PSMIX: mode=%s rows=%u cols=%u gemv_bytes=%llu bulk_bytes=%llu tile_bytes=%u depth=%u grid=%u block=%u"
              " NT=%u NW=%u\n", mode.c_str(), rows, cols, gemv_on ? (unsigned long long)A.size() * 4 : 0ull,
              (unsigned long long)bulk_words * 4, tw * 4, depth, grid, block, (unsigned)nt, (unsigned)nw);
  std::printf("PSRANGE: a=0x%llx-0x%llx bulk=0x%llx-0x%llx\n", (unsigned long long)arg.a_addr,
              (unsigned long long)(arg.a_addr + A.size() * 4), (unsigned long long)arg.bulk_addr,
              (unsigned long long)(arg.bulk_addr + (uint64_t)bulk_words * 4));

  vx_module_h mod = nullptr;
  vx_kernel_h ker = nullptr;
  RT_CHECK(vx_module_load_file(dev, kernel_file.c_str(), &mod));
  RT_CHECK(vx_module_get_kernel(mod, ps_l2pf ? "mix_l2pf" : "mix", &ker));   // PhaseShift (patch dxa-class): -P entry
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = ker;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = grid;
  li.block_dim[0] = block;
  li.lmem_size = depth * tw * 4;
  std::vector<uint32_t> part(part_n, 0), bsum(bsum_n, 0), cyc(cyc_n, 0);
  vx_event_h lev = nullptr, e1 = nullptr, e2 = nullptr, e3 = nullptr;
  RT_CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &lev));
  RT_CHECK(vx_enqueue_read(q, part.data(), pb, 0, part_n * 4, 1, &lev, &e1));
  RT_CHECK(vx_enqueue_read(q, bsum.data(), sb, 0, bsum_n * 4, 1, &lev, &e2));
  RT_CHECK(vx_enqueue_read(q, cyc.data(), cb, 0, cyc_n * 4, 1, &lev, &e3));
  RT_CHECK(vx_event_wait_value(e3, 1, VX_TIMEOUT_INFINITE));
  RT_CHECK(vx_event_wait_value(e2, 1, VX_TIMEOUT_INFINITE));
  RT_CHECK(vx_event_wait_value(e1, 1, VX_TIMEOUT_INFINITE));
  for (auto e : {e1, e2, e3, lev}) vx_event_release(e);

  // host reference
  uint32_t bad_g = 0, bad_b = 0;
  const uint32_t cv = cols / 4;
  if (gemv_on) {
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t l = 0; l < nt; ++l) {
        uint32_t acc = 0;
        for (uint32_t j = l; j < cv; j += (uint32_t)nt)
          for (uint32_t e = 0; e < 4; ++e) acc += A[(size_t)r * cols + 4 * j + e] * X[4 * j + e];
        if (acc != part[(size_t)r * nt + l]) ++bad_g;
      }
  }
  for (uint32_t b = 0; b < grid; ++b)
    for (uint32_t l = 0; l < nt; ++l) {
      uint32_t acc = 0;
      for (uint32_t t = b; t < bulk_tiles; t += grid)
        for (uint32_t i = l; i < tw; i += (uint32_t)nt) acc += B[(size_t)t * tw + i];
      if (acc != bsum[(size_t)b * nt + l]) ++bad_b;
    }
  // completion spans from the warps' mcycle stamps: DEMAND = GEMV warps (1..NW-1), BULK = warp 0 of every CTA
  auto st = [&](uint32_t b, uint32_t w, uint32_t k) {
    const size_t i = ((size_t)b * nw + w) * 4 + k;
    return (uint64_t)cyc[i] | ((uint64_t)cyc[i + 1] << 32);
  };
  uint64_t d0 = ~0ull, d1 = 0, b0 = ~0ull, b1 = 0;
  for (uint32_t b = 0; b < grid; ++b)
    for (uint32_t w = 0; w < nw; ++w) {
      uint64_t s = st(b, w, 0), e = st(b, w, 2);
      if (w == 0) { b0 = std::min(b0, s); b1 = std::max(b1, e); }
      else { d0 = std::min(d0, s); d1 = std::max(d1, e); }
    }
  std::printf("PSMIXCYC: demand_start=%llu demand_end=%llu demand_span=%llu bulk_start=%llu bulk_end=%llu"
              " bulk_span=%llu\n", (unsigned long long)d0, (unsigned long long)d1, (unsigned long long)(d1 - d0),
              (unsigned long long)b0, (unsigned long long)b1, (unsigned long long)(b1 - b0));
  std::printf("PSCHECK: gemv_mismatches=%u of %llu bulk_mismatches=%u of %llu\n", bad_g,
              gemv_on ? (unsigned long long)part_n : 0ull, bad_b, (unsigned long long)bsum_n);
  for (auto bf : {ab, xb, pb, bb, sb, cb}) vx_buffer_release(bf);
  vx_kernel_release(ker);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);
  if (bad_g || bad_b) { std::printf("FAILED!\n"); return 1; }
  std::printf("PASSED!\n");
  return 0;
}
