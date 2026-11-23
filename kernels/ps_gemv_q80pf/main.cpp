// main.cpp -- ps_gemv_q80pf (copy of ps_gemv_q80dq, + pf<DL>_b / pf<DL>_b2 software touch-ahead arms of kernel_pf.cpp; copy of kernels/gemv_q80/main.cpp + dq_* variants, see kernel_dq.cpp); original header: ps_gemv_q80 host app: Y[B][M] = W @ x_b with Q8_0 weights and activations,
// checked against the runq.c arithmetic (kernels/common/ps_q80.h, model/ref_q80.py). Methodology §18.3.
//
// Usage: ps_gemv_q80 -m M -n N [-s seed] [-G GS] [-B B] [-v auto|rows|lanes] [-a host|dev|quant] [-x x.f32] [-L launches]
//                    [-t tol] [-g grid] [-k kernel.vxbin] [-P] [-W l1:l2] [-A]
//   -m/-n/-s  synthetic case, the SAME inputs as the fp32 twins (ps_gemv_f32 / ps_twolaunch): ps::Rng(seed) draws W
//             (M x N uniform in [-1, 1)) first, then the B activation vectors of N values each; timing is data-independent
//            so synthetic weights give the cycles of real ones. W is quantized on the host (ps_q80.h).
//   -G    group size, default export.py's rule for N (64 backed off by halving until it divides N). stories15M's model
//         GS is 32 because its DIM (288) is not a multiple of 64: its shapes pass -G 32 explicitly (also w2, K = 768).
//   -B    activation vectors per launch, 1..4 (small-batch decode); the weight stream is read once for all B.
//   -v    rows (thread per row) | lanes (warp per row) | auto (default): the layout rule applied to the int8 row stride
//         N bytes: rows iff for every warp-aligned first row and 16-byte offset at most DCACHE_NUM_WAYS of the NT rows
//         of a warp share one L1 set (rows_set_pressure in runtime/llama_vx/main.cpp, methodology §14.5).
//         adds the software variants of kernel_sw.cpp, always explicit (never chosen by `auto`): rows_inl, rows_ilp2,
//         rows_ilp4 (thread per row, bit-exact to runq.c like `rows`) and the diagnostics touch_rows, touch_rows4, touch_lanes
//         (weight streaming without arithmetic; y[row] = wrapped uint32 sum of the row's weight words, checked exactly).
//         `-v shfl_probe` runs no GEMV: it checks the segmented butterfly / shfl_idx semantics gemv_dxa relies on (device q80_shfl_probe vs the
//         model of sim/simx/alu_unit.cpp; E1 only, NT = 32) and prints `PSSHFL: ... mismatches=N`.
//         adds rows_nsp and rows_nsp_ilp2 (kernel_nsp.cpp: register-fenced, spill-free; thread per row, so runq-exact like `rows`;
//         rows_nsp_tile is the B = 4 tiling fallback (kernel_tile.cpp, B = 4 only: other B FAIL on purpose); rows_nsp_nu is the plain-C sensitivity (word loop #pragma nounroll, no asm); rows_nsp_ilp2 is dispatched for B = 1, 2 only, B = 3, 4 FAIL on purpose).
//   -a    activation path: host (default; the host quantizes x, so the PERF line is the GEMV alone), dev (the device
//         kernel quantize_q80 runs first; its q and s are compared with the host quantizer, bit for bit, and the GEMV
//         then reads the device's tensors), quant (only the quantizer kernel runs: its cycles/instructions are the PERF
//         line; used by the data-dependence check of the roundf cost).
//   -x    read the B x N activation floats from a raw little-endian float32 file instead of the RNG (real stories15M
//         activations for the quantizer check).
//   -L    back-to-back launches on the same buffers (default 1). Before every launch after the first the output (and in
//         `quant` mode the q / s tensors) is overwritten with NaN / 0x55 poison from the host (host DMA writes go
//         straight to device RAM, they touch no cache), so each launch's result is checked on its own. -P sets the
//         persist_l2 DCR (ps tree, needs -DPS_ALLOW_PERSIST_L2=1) BEFORE the first host write, as gemv_host.h does, so
//         host-write invalidation is active for every write. Resident-control protocol: C_res = C(-L 2 -P) - C(-L 1 -P)
//         (results/<run>/README.md), launch-2 L2 read misses from perf class 5.
//
//   -A    (diagnosis) label address ranges (classification only, no policy is on, cycles unchanged): Wq -> perf class wgt, Ws -> kvp, the per-thread stack window -> kvs,
//         everything else (code, x, y, arguments) stays ord, so the ps counters split L1->L2 bytes, L2 accesses/misses and DRAM bytes by source.
//
// Checks per launch (all must pass for PASSED!):
//   vs_f64     device y vs the exact Q8_0 product (int32 group sums exact; float64 products and sum), metric
//              rel_i = |y_i - ref_i| / S_i, S_i = sum_g |t_g|, tol = max(1e-4, 2 gamma_{G+2}) (model/ref_q80.py error model:
//              two products + G adds per term; the int32 group sums are exact because |ival| <= GS * 127^2 < 2^24).
//   runq_exact bit comparison with the host runq.c-order float32 result (ps_q80.h matmul_row_q80). REQUIRED to be exact
//              for the `rows` variant (same operation order, fp-contract off); informational for `lanes` (tree order).
// Every launch also prints PSDIGEST (FNV-1a of the output bits; errata 37: passing checks are necessary but not sufficient
// against SimX's timing-dependent corruption, so two runs are compared by digest), and
//   PSQ80: <tensor> n=.. gs=.. fnv1a64=..   digests of the quantized W (q, s) and x (xq, xs) for the Python cross-check.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <vortex2.h>

#include "common.h"
#include "ps_common.h"
#include "ps_phase.h"
#include "ps_q80.h"
#include "ps_waymask.h"

#ifndef VX_CFG_DCACHE_SIZE
#define VX_CFG_DCACHE_SIZE 16384
#endif
#ifndef VX_CFG_DCACHE_NUM_WAYS
#define VX_CFG_DCACHE_NUM_WAYS 4
#endif
#ifndef VX_CFG_DCACHE_LINE_SIZE
#define VX_CFG_DCACHE_LINE_SIZE 64
#endif

namespace {

struct Ctx {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h k_gemv = nullptr, k_quant = nullptr;
  std::vector<vx_buffer_h> bufs;
  ~Ctx() {
    if (!device) return;
    for (auto b : bufs) vx_buffer_release(b);
    if (k_gemv) vx_kernel_release(k_gemv);
    if (k_quant) vx_kernel_release(k_quant);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);  // PERF: lines (cumulative over every launch of this run)
    vx_device_release(device);
  }
  int buffer(uint64_t size, int flags, uint64_t* addr, vx_buffer_h* h = nullptr) {
    vx_buffer_h b = nullptr;
    int r = vx_buffer_create(device, size, flags, &b);
    if (r) return r;
    bufs.push_back(b);
    if (h) *h = b;
    return vx_buffer_address(b, addr);
  }
};

#define PS_RT_CHECK(_expr)                                                    \
  do {                                                                        \
    int _ret = (int)(_expr);                                                  \
    if (_ret != 0) {                                                          \
      std::printf("Error: '%s' returned %d!\nFAILED!\n", #_expr, _ret);       \
      return 1;                                                               \
    }                                                                         \
  } while (false)

void usage(const char* app) {
  std::printf("Usage: %s -m M -n N [-s seed] [-G GS] [-B B] [-v auto|rows|lanes] [-a host|dev|quant] [-x x.f32] "
              "[-L launches] [-t tol] [-g grid] [-k kernel.vxbin] [-P] [-W l1:l2] [-A]\n", app);
}

// layout rule (runtime/llama_vx/main.cpp rows_set_pressure), here for the int8 row stride of N bytes
uint32_t rows_set_pressure(uint64_t stride_bytes, uint32_t nt) {
  const uint32_t line = VX_CFG_DCACHE_LINE_SIZE;
  const uint32_t sets = VX_CFG_DCACHE_SIZE / (VX_CFG_DCACHE_LINE_SIZE * VX_CFG_DCACHE_NUM_WAYS);
  const uint64_t period = (uint64_t)line * sets;
  uint32_t worst = 0;
  std::vector<uint32_t> cnt(sets);
  for (uint64_t r0 = 0; r0 < period; r0 += nt) {
    for (uint32_t a = 0; a < line; a += 16) {
      std::fill(cnt.begin(), cnt.end(), 0u);
      for (uint32_t j = 0; j < nt; ++j) {
        const uint64_t byte = a + (r0 + j) * stride_bytes;
        worst = std::max(worst, ++cnt[(byte / line) % sets]);
      }
    }
  }
  return worst;
}

void print_q80_digest(const char* what, size_t n, uint32_t gs, uint64_t h) {
  std::printf("PSQ80: %s n=%zu gs=%u fnv1a64=%016llx\n", what, n, gs, (unsigned long long)h);
}

}  // namespace

int main(int argc, char** argv) {
  std::string variant = "auto", act = "host", xfile, kernel_file = "kernel.vxbin";
  uint32_t M = 64, N = 64, seed = 1, GS = 0, B = 1, launches = 1, grid_override = 0;
  double tol = -1.0;
  bool persist = false, have_wm = false, acct = false;
  uint32_t wm_l1 = 0, wm_l2 = 0;
  int c;
  while ((c = getopt(argc, argv, "m:n:s:G:B:v:a:x:L:t:g:k:PW:Ah")) != -1) {
    switch (c) {
      case 'm': M = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': N = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'G': GS = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'B': B = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'v': variant = optarg; break;
      case 'a': act = optarg; break;
      case 'x': xfile = optarg; break;
      case 'L': launches = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 't': tol = std::strtod(optarg, nullptr); break;
      case 'g': grid_override = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      case 'P': persist = true; break;
      case 'A': acct = true; break;
      case 'W':
        if (!ps::parse_waymask_arg(optarg, &wm_l1, &wm_l2)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        have_wm = true;
        break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  const bool known_variant = variant == "auto" || variant == "rows" || variant == "lanes" || variant == "rows_inl" ||
                             variant == "rows_ilp2" || variant == "rows_ilp4" || variant == "rows_nsp" || variant == "rows_nsp_ilp2" || variant == "rows_nsp_nu" || variant == "rows_nsp_tile" || variant == "touch_rows" ||
                             variant == "touch_rows4" || variant == "touch_lanes" || variant == "shfl_probe" ||
                             variant == "dq_a" || variant == "dq_b" || variant == "dq_ab" || variant == "dq_a2" || variant == "dq_b2" || variant == "dq_ab2" ||
                             variant == "pf1_b" || variant == "pf2_b" || variant == "pf4_b" || variant == "pf1_b2" || variant == "pf2_b2" || variant == "pf4_b2";
  // dq_a*, dq_ab* are NON-BIT-EXACT timing proxies (kernel_dq.cpp): no numeric check, PSDQ line instead
  const bool dq_proxy = variant == "dq_a" || variant == "dq_ab" || variant == "dq_a2" || variant == "dq_ab2";
  if (!known_variant) { std::printf("bad -v %s\nFAILED!\n", variant.c_str()); return 1; }
  if (act != "host" && act != "dev" && act != "quant") { std::printf("bad -a %s\nFAILED!\n", act.c_str()); return 1; }
  if (launches < 1) { std::printf("-L must be >= 1\nFAILED!\n"); return 1; }
  if (B < 1 || B > 4) { std::printf("-B must be in 1..4\nFAILED!\n"); return 1; }
  if (GS == 0) { GS = 64; while (GS > 4 && N % GS) GS >>= 1; }   // export.py's back-off rule
  if (N % GS || GS % 4 || N % 4 || M == 0) { std::printf("need GS %% 4 == 0 and N %% GS == 0 (N=%u GS=%u)\nFAILED!\n", N, GS); return 1; }
  const uint32_t G = N / GS;
  if (tol < 0) { const double m = G + 2, u = std::ldexp(1.0, -24); tol = std::max(1e-4, 2.0 * m * u / (1.0 - m * u)); }

  // ---- inputs: W (M x N) and B activation vectors, quantized on the host (runq.c quantize)
  ps::Rng rng(seed);
  std::vector<float> Wf((size_t)M * N), xf((size_t)B * N);
  for (auto& v : Wf) v = rng.uniform();
  std::string label = "synthetic:s" + std::to_string(seed);
  if (!xfile.empty()) {
    std::ifstream f(xfile, std::ios::binary | std::ios::ate);
    if (!f || (size_t)f.tellg() < xf.size() * 4) { std::printf("cannot read %zu floats from %s\nFAILED!\n", xf.size(), xfile.c_str()); return 1; }
    f.seekg(0);
    f.read(reinterpret_cast<char*>(xf.data()), (std::streamsize)(xf.size() * 4));
    label += "+x:" + xfile;
  } else {
    for (auto& v : xf) v = rng.uniform();
  }
  std::vector<int8_t> Wq((size_t)M * N), xq((size_t)B * N);
  std::vector<float> Ws((size_t)M * G), xs((size_t)B * G);
  ps::quantize_q80(Wf.data(), Wf.size(), GS, Wq.data(), Ws.data());
  for (uint32_t b = 0; b < B; ++b) ps::quantize_q80(xf.data() + (size_t)b * N, N, GS, xq.data() + (size_t)b * N, xs.data() + (size_t)b * G);
  print_q80_digest("Wq", Wq.size(), GS, ps::fnv1a64_bytes(Wq.data(), Wq.size()));
  print_q80_digest("Ws", Ws.size(), GS, ps::fnv1a64_bytes(Ws.data(), Ws.size() * 4));
  print_q80_digest("xq", xq.size(), GS, ps::fnv1a64_bytes(xq.data(), xq.size()));
  print_q80_digest("xs", xs.size(), GS, ps::fnv1a64_bytes(xs.data(), xs.size() * 4));

  // ---- host references: the runq.c-order float32 result and the exact (float64) value with S_i
  std::vector<float> ref32((size_t)B * M);
  std::vector<double> ref64((size_t)B * M), scale((size_t)B * M);
  for (uint32_t b = 0; b < B; ++b)
    for (uint32_t i = 0; i < M; ++i)
      ref32[(size_t)b * M + i] = ps::matmul_row_q80(Wq.data() + (size_t)i * N, Ws.data() + (size_t)i * G,
                                                    xq.data() + (size_t)b * N, xs.data() + (size_t)b * G, N, GS,
                                                    &ref64[(size_t)b * M + i], &scale[(size_t)b * M + i]);

  // ---- device
  Ctx ctx;
  PS_RT_CHECK(vx_device_open(0, &ctx.device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  PS_RT_CHECK(vx_queue_create(ctx.device, &qi, &ctx.queue));
  uint64_t nt = 0;
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  if (variant == "shfl_probe") {   // semantics of the segmented butterfly and shfl_idx (device q80_shfl_probe vs the model of alu_unit.cpp)
    uint64_t nw = 0;
    PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_WARPS, &nw));
    if (nt != 32) { std::printf("shfl_probe needs NT = 32 (E1)\nFAILED!\n"); return 1; }
    constexpr uint32_t SLOTS = 12;
    const uint32_t nthr = (uint32_t)(nt * nw);
    vx_buffer_h hout = nullptr;
    ps_gemv_q80_arg_t parg = {};
    PS_RT_CHECK(ctx.buffer((uint64_t)nthr * SLOTS * 4, VX_MEM_READ_WRITE, &parg.y_addr, &hout));
    PS_RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
    PS_RT_CHECK(vx_module_get_kernel(ctx.module_, "q80_shfl_probe", &ctx.k_gemv));
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = ctx.k_gemv;
    li.args_host = &parg;
    li.args_size = sizeof(parg);
    li.ndim = 1;
    li.grid_dim[0] = 1;
    li.block_dim[0] = nthr;
    vx_event_h ev = nullptr, rd = nullptr;
    std::vector<uint32_t> got((size_t)nthr * SLOTS);
    PS_RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &ev));
    PS_RT_CHECK(vx_enqueue_read(ctx.queue, got.data(), hout, 0, got.size() * 4, 1, &ev, &rd));
    PS_RT_CHECK(vx_event_wait_value(rd, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(rd);
    vx_event_release(ev);
    size_t bad = 0;
    for (uint32_t tid = 0; tid < nthr; ++tid) {
      const uint32_t w = tid / 32, l = tid % 32, base = w * 32;
      auto v = [&](uint32_t lane) { const uint32_t t = base + lane; return t * 2654435761u ^ 0x9e3779b9u; };
      uint32_t exp[SLOTS];
      const uint32_t segs[4] = {4, 8, 16, 32};
      for (int k = 0; k < 4; ++k) { uint32_t a = 0; for (uint32_t j = 0; j < segs[k]; ++j) a += v((l & ~(segs[k] - 1)) + j); exp[k] = a; }
      exp[4] = v(l ^ 3);
      exp[5] = v(l ^ 16);
      for (int j = 0; j < 4; ++j) exp[6 + j] = v(8 * j);
      exp[10] = v((l & ~7u) + 3);
      exp[11] = v((l & ~3u) + 2);
      for (uint32_t k = 0; k < SLOTS; ++k)
        if (got[(size_t)tid * SLOTS + k] != exp[k]) {
          if (bad < 8) std::printf("PSSHFL mismatch tid=%u slot=%u got=%08x exp=%08x\n", tid, k, got[(size_t)tid * SLOTS + k], exp[k]);
          ++bad;
        }
    }
    std::printf("PSSHFL: threads=%u slots=%u mismatches=%zu (segmented butterfly sums S=4/8/16/32, bfly xor 3 in 8, xor 16, idx 0/8/16/24 whole warp, idx 3 in 8, idx 2 in 4)\n", nthr, SLOTS, bad);
    if (bad) { std::printf("FAILED!\n"); return 1; }
    std::printf("PASSED!\n");
    return 0;
  }
  if (variant == "auto") {
    const uint32_t pressure = rows_set_pressure(N, (uint32_t)nt);
    variant = pressure <= VX_CFG_DCACHE_NUM_WAYS ? "rows" : "lanes";
    std::printf("PSGEMV: M=%u N=%u stride_bytes=%u rows_set_pressure=%u ways=%u variant=%s\n", M, N, N, pressure,
                VX_CFG_DCACHE_NUM_WAYS, variant.c_str());
  }
  std::printf("ps_gemv_q80: %s M=%u N=%u GS=%u G=%u B=%u variant=%s act=%s launches=%u tol=%.3e\n", label.c_str(), M, N,
              GS, G, B, variant.c_str(), act.c_str(), launches, tol);
  if (persist && !have_wm) {   // persist_l2 BEFORE the first host write (gemv_host.h:180-184 order; labelled resident control)
    ps::PhaseCfg pc;
    pc.flags = ps::phase_dcr::FLAG_PERSIST_L2;
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
    std::printf("PSPERSIST-HOST: persist_l2=1 (resident control)\n");
  } else if (have_wm) {
    ps::PhaseCfg pc;
    if (persist) pc.flags = ps::phase_dcr::FLAG_PERSIST_L2;
    ps::WayMaskCtl wmc;
    PS_RT_CHECK(wmc.set(pc, wm_l1, wm_l2));
    ps::waymask_print(wmc);
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
  }

  ps_gemv_q80_arg_t arg = {};
  arg.M = M; arg.N = N; arg.GS = GS; arg.B = B;
  ps_quant_q80_arg_t qarg = {};
  qarg.B = B; qarg.n = N; qarg.GS = GS;
  vx_buffer_h hWq, hWs, hxf, hxq, hxs, hy;
  const uint64_t y_size = (uint64_t)B * M * 4;
  PS_RT_CHECK(ctx.buffer((uint64_t)M * N, VX_MEM_READ, &arg.Wq_addr, &hWq));
  PS_RT_CHECK(ctx.buffer((uint64_t)M * G * 4, VX_MEM_READ, &arg.Ws_addr, &hWs));
  PS_RT_CHECK(ctx.buffer((uint64_t)B * N, VX_MEM_READ_WRITE, &arg.xq_addr, &hxq));
  PS_RT_CHECK(ctx.buffer((uint64_t)B * G * 4, VX_MEM_READ_WRITE, &arg.xs_addr, &hxs));
  PS_RT_CHECK(ctx.buffer(y_size, VX_MEM_READ_WRITE, &arg.y_addr, &hy));
  PS_RT_CHECK(ctx.buffer((uint64_t)B * N * 4, VX_MEM_READ, &qarg.x_addr, &hxf));
  qarg.q_addr = arg.xq_addr;
  qarg.s_addr = arg.xs_addr;
  if (arg.Wq_addr % 4 || arg.xq_addr % 4 || arg.Ws_addr % 4) { std::printf("buffers not 4-byte aligned\nFAILED!\n"); return 1; }
  if (act == "quant") {
    // the weights are not needed
  } else {
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hWq, 0, Wq.data(), Wq.size(), 0, nullptr, nullptr));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hWs, 0, Ws.data(), Ws.size() * 4, 0, nullptr, nullptr));
  }
  if (act == "host") {
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hxq, 0, xq.data(), xq.size(), 0, nullptr, nullptr));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hxs, 0, xs.data(), xs.size() * 4, 0, nullptr, nullptr));
  } else {
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hxf, 0, xf.data(), xf.size() * 4, 0, nullptr, nullptr));
  }
  if (acct) {   // over-fetch diagnosis: label address ranges so the ps counters split by traffic source (classification only: every policy is off)
    ps::PhaseCfg pc;
    pc.flags = 0;   // ranges classify (the counters are per request class); every policy is off, so the timing is that of the stock run (checked: cycles equal)
    pc.wgt_base = arg.Wq_addr;  pc.wgt_limit = arg.Wq_addr + (uint64_t)M * N;                    // class wgt: the int8 weights
    pc.kv_base[1] = arg.Ws_addr; pc.kv_limit[1] = arg.Ws_addr + (uint64_t)M * G * 4;             // class kvp: the weight scales
    uint64_t nw = 0, ncores = 0;
    PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_WARPS, &nw));
    PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_CORES, &ncores));
    const uint64_t stack_top = 0xFFFF0000ull, stack_span = ncores * nw * nt << 13;                // VX_MEM_STACK_LOG2_SIZE = 13 per thread
    pc.kv_base[0] = stack_top - stack_span; pc.kv_limit[0] = stack_top;                          // class kvs: the per-thread stacks
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
    std::printf("PSACCT: wgt(Wq)=0x%llx-0x%llx kvp(Ws)=0x%llx-0x%llx kvs(stack)=0x%llx-0x%llx ord=everything else (code, x, y, args)\n",
                (unsigned long long)pc.wgt_base, (unsigned long long)pc.wgt_limit, (unsigned long long)pc.kv_base[1],
                (unsigned long long)pc.kv_limit[1], (unsigned long long)pc.kv_base[0], (unsigned long long)pc.kv_limit[0]);
  }
  PS_RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
  const bool is_touch = variant.rfind("touch_", 0) == 0;
  const bool lanes_type = variant == "lanes" || variant == "touch_lanes";   // warp per row (global = M x NT threads)
  const bool exact_required = !lanes_type && !is_touch;                    // thread-per-row compute variants are runq-order exact
  const std::string kname = variant == "rows" ? "gemv_q80_rows" : variant == "lanes" ? "gemv_q80_lanes"
                            : is_touch ? "q80_" + variant : "gemv_q80_" + variant;
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, kname.c_str(), &ctx.k_gemv));
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, "quantize_q80", &ctx.k_quant));

  auto geometry = [&](uint32_t global, uint32_t* grid, uint32_t* block) -> int {
    uint32_t g[1], b[1];
    int r = vx_device_max_occupancy_grid(ctx.device, 1, &global, g, b);
    if (r) return r;
    *grid = grid_override > 0 ? grid_override : g[0];
    *block = b[0];
    return 0;
  };
  uint32_t grid_g = 0, block_g = 0, grid_q = 0, block_q = 0;
  PS_RT_CHECK(geometry(lanes_type ? (uint32_t)(M * nt) : M, &grid_g, &block_g));
  PS_RT_CHECK(geometry(B * G, &grid_q, &block_q));
  if (block_g % nt != 0) { std::printf("block %u not a multiple of NT %u\nFAILED!\n", block_g, (unsigned)nt); return 1; }
  std::printf("launch: gemv grid=%u block=%u quant grid=%u block=%u NT=%u%s\n", grid_g, block_g, grid_q, block_q,
              (unsigned)nt, grid_override > 0 ? " (grid overridden by -g)" : "");

  auto run = [&](vx_kernel_h k, void* a, size_t asz, uint32_t grid, uint32_t block) -> int {
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = k;
    li.args_host = a;
    li.args_size = asz;
    li.ndim = 1;
    li.grid_dim[0] = grid;
    li.block_dim[0] = block;
    vx_event_h ev = nullptr;
    int r = vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &ev);
    if (r) return r;
    r = vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
    vx_event_release(ev);
    return r;
  };
  auto read_back = [&](void* host, vx_buffer_h b, uint64_t bytes) -> int {
    vx_event_h ev = nullptr;
    int r = vx_enqueue_read(ctx.queue, host, b, 0, bytes, 0, nullptr, &ev);
    if (r) return r;
    r = vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
    vx_event_release(ev);
    return r;
  };

  bool ok = true;
  std::vector<float> y((size_t)B * M);
  const std::vector<float> poison((size_t)B * M, std::nanf(""));
  const std::vector<int8_t> qpoison(xq.size(), (int8_t)0x55);
  const std::vector<float> spoison(xs.size(), std::nanf(""));
  ps::Timer timer;
  for (uint32_t l = 0; l < launches; ++l) {
    const std::string pre = "launch" + std::to_string(l);
    if (l > 0) {   // make every launch's output observable on its own
      if (act == "quant") {
        PS_RT_CHECK(vx_enqueue_write(ctx.queue, hxq, 0, qpoison.data(), qpoison.size(), 0, nullptr, nullptr));
        PS_RT_CHECK(vx_enqueue_write(ctx.queue, hxs, 0, spoison.data(), spoison.size() * 4, 0, nullptr, nullptr));
      } else {
        PS_RT_CHECK(vx_enqueue_write(ctx.queue, hy, 0, poison.data(), y_size, 0, nullptr, nullptr));
        if (act == "dev") {
          PS_RT_CHECK(vx_enqueue_write(ctx.queue, hxq, 0, qpoison.data(), qpoison.size(), 0, nullptr, nullptr));
          PS_RT_CHECK(vx_enqueue_write(ctx.queue, hxs, 0, spoison.data(), spoison.size() * 4, 0, nullptr, nullptr));
        }
      }
    }
    if (act != "host") {
      PS_RT_CHECK(run(ctx.k_quant, &qarg, sizeof qarg, grid_q, block_q));
      std::vector<int8_t> dq(xq.size());
      std::vector<float> ds(xs.size());
      PS_RT_CHECK(read_back(dq.data(), hxq, dq.size()));
      PS_RT_CHECK(read_back(ds.data(), hxs, ds.size() * 4));
      size_t qbad = 0, sbad = 0;
      for (size_t i = 0; i < dq.size(); ++i) qbad += dq[i] != xq[i];
      for (size_t i = 0; i < ds.size(); ++i) sbad += std::memcmp(&ds[i], &xs[i], 4) != 0;
      std::printf("PSQUANT: %s nq=%zu ns=%zu q_mismatch=%zu s_mismatch=%zu\n", pre.c_str(), dq.size(), ds.size(), qbad, sbad);
      print_q80_digest("xq_device", dq.size(), GS, ps::fnv1a64_bytes(dq.data(), dq.size()));
      print_q80_digest("xs_device", ds.size(), GS, ps::fnv1a64_bytes(ds.data(), ds.size() * 4));
      ok = ok && qbad == 0 && sbad == 0;
    }
    if (act == "quant") continue;
    PS_RT_CHECK(run(ctx.k_gemv, &arg, sizeof arg, grid_g, block_g));
    PS_RT_CHECK(read_back(y.data(), hy, y_size));
    if (dq_proxy) {   // timing proxy: y is not the dot product; cycles are data independent, so only cycles are valid
      std::printf("PSDQ: %s NON-BIT-EXACT timing proxy variant=%s: numeric checks skipped, cycles valid\n", pre.c_str(), variant.c_str());
      continue;
    }
    if (is_touch) {   // diagnostic: y[0..M) must be the wrapped uint32 sum of each row's weight words (exact)
      size_t mism = 0;
      for (uint32_t i = 0; i < M; ++i) {
        uint32_t s = 0, w;
        for (uint32_t k = 0; k < N; k += 4) { std::memcpy(&w, &Wq[(size_t)i * N + k], 4); s += w; }
        uint32_t got;
        std::memcpy(&got, &y[i], 4);
        mism += got != s;
      }
      std::printf("PSTOUCH: %s n=%u mismatches=%zu variant=%s\n", pre.c_str(), M, mism, variant.c_str());
      std::printf("PSDIGEST: %s:touch n=%u fnv1a64=%016llx\n", pre.c_str(), M, (unsigned long long)ps::fnv1a64_f32(y.data(), M));
      ok = ok && mism == 0;
      continue;
    }
    auto s64 = ps::compare(y.data(), ref64.data(), scale.data(), y.size(), tol);
    ps::print_check((pre + ":vs_f64").c_str(), s64);
    ok = ok && s64.ok();
    size_t mism = 0;
    for (size_t i = 0; i < y.size(); ++i) mism += std::memcmp(&y[i], &ref32[i], 4) != 0;
    std::printf("PSBITEXACT: %s:runq_exact n=%zu mismatches=%zu variant=%s\n", pre.c_str(), y.size(), mism, variant.c_str());
    if (exact_required) ok = ok && mism == 0;
  }
  std::printf("Elapsed: %.0f ms\n", timer.ms());
  if (!ok) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
