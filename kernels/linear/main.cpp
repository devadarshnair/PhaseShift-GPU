// main.cpp -- ps_linear host app: the linear layers Y[M][N] = X[M][K] @ W^T of a llama2.c model.
//
// Usage: ps_linear [-f <fixture dir> | -m M -n N -k K [-s seed]] [-v small|tcu] [-t tol] [-T tol_fixture] [-g grid]
//                  [-L launches] [-P] [-W l1:l2]
//   -f   ps-fixture-v1 directory (absolute path) with tensors w (N, K), x (M, K) or (K,), out (M, N) or (N,);
//        sha256 of every tensor is verified (kernels/common/ps_fixture.h). Generator: model/linear_fixtures.py.
//   -m/-n/-k/-s  synthetic case: W, X uniform in [-1, 1) (ps::Rng, seed -s). Timing is data-independent, so the
//        benchmark uses synthetic data.
//   -v   small (default): FPU fp32 kernel gemm_small (any config; the earlier `ps_gemm_small`, M = B in 2..16).
//        tcu: fp16-in/fp32-out WGMMA + DXA kernel tcu_gemm (= stock sgemm_tcu_wg_dxa device code; needs a build with
//        -DVX_CFG_TCU_WGMMA_ENABLE, NT = 32 -- the smoke-wgmma-dxa / TIER-CFG config sets of experiments/configs.yaml).
//        M, N, K are zero-padded to the tile constraints (M to a multiple of cta_M = ISSUE_WIDTH x xtileM, N of
//        xtileN, K of tileK; stock main.cpp:334-347); the useful FLOPs are 2MNK, the executed ones 2·Mp·Np·Kp.
//        (P8T track): built with -DITYPE=int8 -DOTYPE=int32 (config EP-SMOKE-S0-tcuint8 / the persist control with those
//        two macros added), `-v tcu` runs the SAME stock WGMMA + DXA device code with int8 inputs and int32 accumulation: W and X are
//        quantized to int8 on the host (Q8_0 quantizer of kernels/common/ps_q80.h, GS = 64 backed off to divide K), C is the EXACT
//        integer product sum_k Wq[n][k] * Xq[m][k] (int32, no overflow for K <= 131,000), checked against an int64 host reference with
//        zero tolerance (PSCHECK vs_int_exact, PSDIGEST over the C bits). The Q8_0 group scales are NOT applied: the stock path has no
//        per-group epilogue, so P8T is a lower bound of a scaled path (labelled so in every table; P8T).
//   -t   tolerance of the main check (default: see "Tolerances" below)
//   -T   tcu only: tolerance of the vs-fixture check (default 1e-2)
//   -g   small only: override the number of blocks (exercises the grid-stride loop)
//   -L   number of back-to-back launches of the same kernel on the same buffers (default 1;: two-launch
//        persist mode, as ps_twolaunch / ps_gemv_q80). Before every launch after the first the whole output buffer
//        (tcu: the padded C) is overwritten with NaN from the host (host DMA writes go straight to device RAM, they touch
//        no cache), so each launch's result is checked independently; with L > 1 the PSCHECK / PSDIGEST labels are
//        prefixed `launch<l>:`. With -L 1 the labels and the device work are exactly as before this option existed.
//   -P   persist_l2 DCR (ps tree, host built with -DPS_ALLOW_PERSIST_L2=1, config EP-CTRL-S0-L2big-P) set BEFORE the first
//        host write (gemv_host.h:180-184 order), so host-write invalidation is active for every write. Resident control:
//        C_res = C(-L 2 -P) - C(-L 1 -P), launch-2 L2 read misses from perf class 5 (results/<run>/README.md).
//
// Checks (all must pass for PASSED!; metric rel_i = |y_i - ref_i| / S_i of kernels/common/ps_common.h):
//   small:  vs_f64      device Y vs a float64 recomputation from the same fp32 W, X
//           vs_fixture  device Y vs the fixture's NumPy float32 `out` (fixture runs)
//           fixture_vs_f64  the fixture's own `out` vs float64 (input sanity)
//   tcu:    vs_f64_fp16in  device C vs float64 sum over the fp16-ROUNDED W, X (each product is exact in double)
//           vs_fixture     device C vs NumPy float32 `out` from the UNROUNDED inputs (fixture runs)
//           pad_zero       every padded element of C (row >= M or col >= N) must be exactly 0
//
// Tolerances (modeling choice; explained here and in docs/methodology.md §2 consequence 6, extension). u = 2^-24
// (fp32 unit roundoff), u_h = 2^-11 (fp16), gamma_K = K u / (1 - K u).
//   * fp32 accumulation, any K (both paths' main check): two correct fp32 K-long dot products with different
//     summation orders differ by <= 2 gamma_K S_i (ps_common.h). Default tol = max(1e-4, 2 gamma_K): the earlier 1e-4 for
//     K <= 838, the bound itself above (K = 1792: 2.14e-4; K = 3584: 4.27e-4). Every product passes through at most
//     K roundings in each order used here: gemm_small (= gemv_rows' order) 1 mul + 3 in-chunk adds + K/4 accumulation
//     adds <= K for K >= 8; SimX's TCU FEDP (sim/simx/tcu/tcu_unit.cpp:176-201: exact fp16 products, one fp32 FMA per
//     packed pair, one fp32 add per word, one add into the accumulator per k-step) <= K; BLAS/NumPy <= K.
//     For the tcu path the reference is computed from the same fp16-rounded inputs, so input rounding cancels and only
//     accumulation error remains: the same 2 gamma_K bound applies (the TCU vs exact needs only gamma_K; 2x margin).
//   * fp16 inputs vs the fp32 model (tcu vs_fixture): rounding w and x to fp16 changes each product by at most
//     (1+u_h)^2 - 1 = 2 u_h + u_h^2 = 9.77e-4 relative (normal fp16 range); the fixture's own fp32 accumulation adds
//     <= gamma_K, the TCU's <= gamma_K, so rel_i <= 9.77e-4 + 2 gamma_K (1.19e-3 at K = 1792) plus an absolute
//     2^-25 |x_j| per fp16-subnormal weight and 2^-25 |w_j| per fp16-subnormal activation (|v| < 2^-14). the earlier 1e-2 is ~8x above that bound and is kept: it is a
//     plausibility check against the real model, while vs_f64_fp16in (2 gamma_K) is the strict kernel check. For
//     bf16 inputs (u_bf = 2^-8, not tested by this step) the same bound gives 7.8e-3 + 2 gamma_K, still < 1e-2 for
//     K <= ~9.000; any |value| > 65504 overflows fp16 and is rejected before simulation.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <type_traits>
#include <string>
#include <vector>

#include <vortex2.h>

#include "common.h"
#include "fp16.h"
#include "ps_common.h"
#include "ps_fixture.h"
#include "ps_phase.h"     // -W: the [dcr_phase] block write
#include "ps_q80.h"       // P8T: host quantizer (int8 build only)
#include "ps_waymask.h"   // -W: mechanism A2 way masks (encoding, methodology §14.A)

#ifdef VX_CFG_TCU_WGMMA_ENABLE
#include <dxa.h>
#include <tensor_cfg.h>
namespace vt = vortex::tensor;
using tcfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, WGMMA_NRC>;
constexpr uint32_t kIEB = sizeof(typename vt::ITYPE::dtype);                          // input element bytes (fp16: 2, int8: 1)
constexpr bool kI8 = std::is_same<vt::ITYPE, vt::int8>::value;                          // P8T build
#endif

namespace {

struct Ctx {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  std::vector<vx_buffer_h> bufs;
  ~Ctx() {
    if (!device) return;
    for (auto b : bufs) vx_buffer_release(b);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);  // PERF: lines
    vx_device_release(device);
  }
  int buffer(uint64_t size, int flags, uint64_t* addr) {
    vx_buffer_h b = nullptr;
    int r = vx_buffer_create(device, size, flags, &b);
    if (r) return r;
    bufs.push_back(b);
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

double gamma_k(uint32_t K) {
  const double u = std::ldexp(1.0, -24);
  return K * u / (1.0 - K * u);
}

void usage(const char* app) {
  std::printf("Usage: %s [-f fixture_dir | -m M -n N -k K [-s seed]] [-v small|tcu] [-t tol] [-T tol_fixture] "
              "[-g grid] [-L launches] [-P] [-W l1:l2]\n", app);
}

[[maybe_unused]] uint32_t round_up(uint32_t v, uint32_t q) { return (v + q - 1) / q * q; }

// float64 reference over the given (already rounded or not) operands; S_i = sum |w x|
void reference(const std::vector<float>& W, const std::vector<float>& X, uint32_t M, uint32_t N, uint32_t K,
               std::vector<double>& ref, std::vector<double>& scale) {
  ref.assign((size_t)M * N, 0.0);
  scale.assign((size_t)M * N, 0.0);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n) {
      double s = 0.0, a = 0.0;
      const float* w = &W[(size_t)n * K];
      const float* x = &X[(size_t)m * K];
      for (uint32_t k = 0; k < K; ++k) {
        double p = (double)w[k] * (double)x[k];
        s += p;
        a += std::fabs(p);
      }
      ref[(size_t)m * N + n] = s;
      scale[(size_t)m * N + n] = a;
    }
}

// Per-launch checker: called with the launch index and the output read back after that launch; returns false on error.
using LaunchCheck = std::function<void(uint32_t, const std::vector<float>&)>;

int run_small(Ctx& ctx, const std::vector<float>& W, const std::vector<float>& X, uint32_t M, uint32_t N,
              uint32_t K, uint32_t grid_override, uint32_t launches, const LaunchCheck& check) {
  uint64_t nt = 0;
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  ps_gemm_small_arg_t arg = {};
  arg.M = M;
  arg.N = N;
  arg.K = K;
  const uint64_t W_size = (uint64_t)N * K * 4, X_size = (uint64_t)M * K * 4, Y_size = (uint64_t)M * N * 4;
  PS_RT_CHECK(ctx.buffer(W_size, VX_MEM_READ, &arg.W_addr));
  PS_RT_CHECK(ctx.buffer(X_size, VX_MEM_READ, &arg.X_addr));
  PS_RT_CHECK(ctx.buffer(Y_size, VX_MEM_READ_WRITE, &arg.Y_addr));
  vx_buffer_h Wb = ctx.bufs[0], Xb = ctx.bufs[1], Yb = ctx.bufs[2];
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, Wb, 0, W.data(), W_size, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, Xb, 0, X.data(), X_size, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_module_load_file(ctx.device, "kernel.vxbin", &ctx.module_));
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, "gemm_small", &ctx.kernel));
  uint32_t global = N, grid_dim[1], block_dim[1];
  PS_RT_CHECK(vx_device_max_occupancy_grid(ctx.device, 1, &global, grid_dim, block_dim));
  if (grid_override > 0) grid_dim[0] = grid_override;
  std::printf("launch: kernel=gemm_small global=%u grid=%u block=%u NT=%u%s\n", global, grid_dim[0], block_dim[0],
              (unsigned)nt, grid_override > 0 ? " (grid overridden by -g)" : "");
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = ctx.kernel;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = grid_dim[0];
  li.block_dim[0] = block_dim[0];
  std::vector<float> Y((size_t)M * N, 0.0f);
  const std::vector<float> poison((size_t)M * N, std::nanf(""));
  for (uint32_t l = 0; l < launches; ++l) {
    if (l > 0)   // make every launch's output observable on its own
      PS_RT_CHECK(vx_enqueue_write(ctx.queue, Yb, 0, poison.data(), Y_size, 0, nullptr, nullptr));
    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    PS_RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &launch_ev));
    PS_RT_CHECK(vx_enqueue_read(ctx.queue, Y.data(), Yb, 0, Y_size, 1, &launch_ev, &read_ev));
    PS_RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    check(l, Y);
  }
  return 0;
}

#ifdef VX_CFG_TCU_WGMMA_ENABLE
// Mirrors the stock sgemm_tcu_wg_dxa host (main.cpp:283-466): capability checks, padding constraints, DXA descriptors
// A (row-major, tile tileK x cta_M) and B (BlockMajor, tile xtileN x tileK), grid {Np/xtileN, Mp/cta_M},
// block {warps*NT, 1}, lmem = A tile + B tile. Returns C (Mp x Np fp32).
int run_tcu(Ctx& ctx, const void* A, const void* B, uint32_t Mp, uint32_t Np,
            uint32_t Kp, uint32_t warps, uint32_t nt, uint32_t launches, const LaunchCheck& check) {
  ps_tcu_arg_t arg = {};
  arg.M = Mp;
  arg.N = Np;
  arg.K = Kp;
  const uint64_t A_size = (uint64_t)Mp * Kp * kIEB, B_size = (uint64_t)Kp * Np * kIEB, C_size = (uint64_t)Mp * Np * 4;
  PS_RT_CHECK(ctx.buffer(A_size, VX_MEM_READ, &arg.A_addr));
  PS_RT_CHECK(ctx.buffer(B_size, VX_MEM_READ, &arg.B_addr));
  PS_RT_CHECK(ctx.buffer(C_size, VX_MEM_READ_WRITE, &arg.C_addr));   // READ_WRITE: -L > 1 poisons C from the host
  vx_buffer_h Ab = ctx.bufs[0], Bb = ctx.bufs[1], Cb = ctx.bufs[2];
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, Ab, 0, A, A_size, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, Bb, 0, B, B_size, 0, nullptr, nullptr));
  const uint32_t cta_M = warps * tcfg::xtileM;
  PS_RT_CHECK(vortex::dxa::program_2d(ctx.device, 0, arg.A_addr, Kp, Mp, Kp * kIEB, tcfg::tileK, cta_M, kIEB));
  PS_RT_CHECK(vortex::dxa::program_2d(ctx.device, 1, arg.B_addr, Np, Kp, Np * kIEB, tcfg::xtileN, tcfg::tileK, kIEB));
  PS_RT_CHECK(vortex::dxa::set_layout(ctx.device, 1, vortex::dxa::Layout::BlockMajor, 2, kIEB));
  PS_RT_CHECK(vortex::dxa::set_tile_geometry(ctx.device, 1, tcfg::tcN, tcfg::b_blk_k));
  PS_RT_CHECK(vx_module_load_file(ctx.device, "kernel.vxbin", &ctx.module_));
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, "tcu_gemm", &ctx.kernel));
  const uint32_t smem = (cta_M * tcfg::tileK + tcfg::tileK * tcfg::xtileN) * kIEB;
  std::printf("launch: kernel=tcu_gemm tile M=%u N=%u K=%u cta_M=%u warps=%u grid=%ux%u block=%u smem=%u\n",
              tcfg::xtileM, tcfg::xtileN, tcfg::tileK, cta_M, warps, Np / tcfg::xtileN, Mp / cta_M, warps * nt,
              smem);
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = ctx.kernel;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 2;
  li.grid_dim[0] = Np / tcfg::xtileN;
  li.grid_dim[1] = Mp / cta_M;
  li.block_dim[0] = warps * nt;
  li.block_dim[1] = 1;
  li.lmem_size = smem;
  std::vector<float> C((size_t)Mp * Np, std::nanf(""));
  const std::vector<float> poison((size_t)Mp * Np, std::nanf(""));
  for (uint32_t l = 0; l < launches; ++l) {
    if (l > 0)   // poison the whole padded C: the pad_zero check then also proves the pad was rewritten by this launch
      PS_RT_CHECK(vx_enqueue_write(ctx.queue, Cb, 0, poison.data(), C_size, 0, nullptr, nullptr));
    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    PS_RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &launch_ev));
    PS_RT_CHECK(vx_enqueue_read(ctx.queue, C.data(), Cb, 0, C_size, 1, &launch_ev, &read_ev));
    PS_RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    check(l, C);
  }
  return 0;
}

// P8T: the int8 build of the stock TCU path (header comment of this file). Returns 0, or 1 on a runtime error; clears `ok` on a
// wrong result. W (N x K) and X (M x K) are quantized to int8 with the Q8_0 quantizer; C must equal the exact integer product.
int tcu_i8_case(Ctx& ctx, const std::vector<float>& W, const std::vector<float>& X, uint32_t M, uint32_t N, uint32_t K,
                uint32_t launches, bool& ok, const std::function<std::string(uint32_t, const char*)>& tag) {
  uint64_t isa = 0, nt = 0, nw = 0, iw = 0;
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_ISA_FLAGS, &isa));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_WARPS, &nw));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_ISSUE_WIDTH, &iw));
  if (!(isa & VX_ISA_EXT_TCU) || !(isa & VX_ISA_EXT_DXA) || nt != VX_CFG_NUM_THREADS || iw > nw) {
    std::printf("device lacks TCU/DXA or NT/ISSUE_WIDTH mismatch (NT=%u, IW=%u, NW=%u)\nFAILED!\n", (unsigned)nt, (unsigned)iw, (unsigned)nw);
    return 1;
  }
  const uint32_t warps = (uint32_t)iw, cta_M = warps * tcfg::xtileM;
  const uint32_t Mp = round_up(M, cta_M), Np = round_up(N, tcfg::xtileN), Kp = round_up(K, tcfg::tileK);
  uint32_t GS = 64;
  while (GS > 4 && K % GS) GS >>= 1;
  std::vector<int8_t> Wq((size_t)N * K), Xq((size_t)M * K);
  std::vector<float> Ws((size_t)N * (K / GS)), Xs((size_t)M * (K / GS));
  ps::quantize_q80(W.data(), W.size(), GS, Wq.data(), Ws.data());
  ps::quantize_q80(X.data(), X.size(), GS, Xq.data(), Xs.data());
  std::vector<int8_t> A((size_t)Mp * Kp, 0), B((size_t)Kp * Np, 0);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t k = 0; k < K; ++k) A[(size_t)m * Kp + k] = Xq[(size_t)m * K + k];
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t k = 0; k < K; ++k) B[(size_t)k * Np + n] = Wq[(size_t)n * K + k];
  std::vector<int64_t> ref((size_t)M * N, 0);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n) {
      int64_t s = 0;
      for (uint32_t k = 0; k < K; ++k) s += (int64_t)Wq[(size_t)n * K + k] * (int64_t)Xq[(size_t)m * K + k];
      ref[(size_t)m * N + n] = s;
    }
  std::printf("tcu_padding: M=%u->%u N=%u->%u K=%u->%u padded_ops=%.0f pad_ratio=%.4f int8 GS=%u\n", M, Mp, N, Np, K, Kp,
              2.0 * Mp * Np * Kp, (double)Mp * Np * Kp / ((double)M * N * K), GS);
  std::printf("PSQ80: Wq n=%zu gs=%u fnv1a64=%016llx\n", Wq.size(), GS, (unsigned long long)ps::fnv1a64_bytes(Wq.data(), Wq.size()));
  std::printf("PSQ80: xq n=%zu gs=%u fnv1a64=%016llx\n", Xq.size(), GS, (unsigned long long)ps::fnv1a64_bytes(Xq.data(), Xq.size()));
  auto check = [&](uint32_t l, const std::vector<float>& C) {
    size_t pad_bad = 0, bad = 0;
    std::vector<int32_t> Y((size_t)M * N);
    for (uint32_t m = 0; m < Mp; ++m)
      for (uint32_t n = 0; n < Np; ++n) {
        int32_t v;
        std::memcpy(&v, &C[(size_t)m * Np + n], 4);
        if (m < M && n < N) { Y[(size_t)m * N + n] = v; bad += (int64_t)v != ref[(size_t)m * N + n]; }
        else if (v != 0) ++pad_bad;
      }
    std::printf("PSCHECK: %s n=%zu errors=%zu\n", tag(l, "pad_zero").c_str(), (size_t)Mp * Np - (size_t)M * N, pad_bad);
    std::printf("PSCHECK: %s n=%zu errors=%zu\n", tag(l, "vs_int_exact").c_str(), (size_t)M * N, bad);
    std::printf("PSDIGEST: %s n=%zu fnv1a64=%016llx\n", tag(l, "vs_int_exact").c_str(), (size_t)M * N,
                (unsigned long long)ps::fnv1a64_bytes(Y.data(), Y.size() * 4));
    ok = ok && pad_bad == 0 && bad == 0;
  };
  return run_tcu(ctx, A.data(), B.data(), Mp, Np, Kp, warps, (uint32_t)nt, launches, check);
}
#endif

}  // namespace

int main(int argc, char** argv) {
  std::string fixture, variant = "small";
  uint32_t M = 4, N = 64, K = 64, seed = 1, grid_override = 0, launches = 1;
  bool persist = false;
  double tol = -1.0;
  [[maybe_unused]] double tol_fx = 1e-2;  // used by -v tcu only
  bool ps_have_wm = false;   // PhaseShift: -W way masks (absent = no DCR write, the old host path)
  uint32_t ps_wm_l1 = 0, ps_wm_l2 = 0;
  int c;
  while ((c = getopt(argc, argv, "f:m:n:k:s:v:t:T:g:L:PW:h")) != -1) {
    switch (c) {
      case 'f': fixture = optarg; break;
      case 'm': M = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': N = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': K = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'v': variant = optarg; break;
      case 't': tol = std::strtod(optarg, nullptr); break;
      case 'T': tol_fx = std::strtod(optarg, nullptr); break;
      case 'g': grid_override = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'L': launches = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'P': persist = true; break;
      case 'W':   // PhaseShift: [dcr_phase] way masks "<l1word>:<l2word>" (ps_waymask.h, methodology §14.A)
        if (!ps::parse_waymask_arg(optarg, &ps_wm_l1, &ps_wm_l2)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        ps_have_wm = true;
        break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  if (variant != "small" && variant != "tcu") { std::printf("bad -v %s\nFAILED!\n", variant.c_str()); return 1; }
  if (launches < 1) { std::printf("-L must be >= 1\nFAILED!\n"); return 1; }
#ifndef VX_CFG_TCU_WGMMA_ENABLE
  if (variant == "tcu") {
    std::printf("-v tcu needs a build with -DVX_CFG_TCU_WGMMA_ENABLE (config smoke-wgmma-dxa or TIER-CFG)\nFAILED!\n");
    return 1;
  }
#endif

  // ---- inputs: W (N x K), X (M x K), optional fixture out (M x N)
  std::vector<float> W, X, out_fx;
  std::string label;
  try {
    if (!fixture.empty()) {
      ps::Fixture fx(fixture);
      ps::Tensor tw = fx.tensor("w"), tx = fx.tensor("x"), to = fx.tensor("out");
      if (tw.shape.size() != 2) throw ps::FixtureError("w must be (N, K)");
      N = tw.dim(0);
      K = tw.dim(1);
      if (tx.shape.size() == 1 && to.shape.size() == 1) {
        M = 1;
        if (tx.dim(0) != K || to.dim(0) != N) throw ps::FixtureError("shape mismatch: need x (K), out (N)");
      } else if (tx.shape.size() == 2 && to.shape.size() == 2) {
        M = tx.dim(0);
        if (tx.dim(1) != K || to.dim(0) != M || to.dim(1) != N)
          throw ps::FixtureError("shape mismatch: need x (M, K), out (M, N)");
      } else {
        throw ps::FixtureError("x/out must both be 1-D or both 2-D");
      }
      W = std::move(tw.data);
      X = std::move(tx.data);
      out_fx = std::move(to.data);
      label = fx.model() + ":" + fx.op();
    } else {
      ps::Rng rng(seed);
      W.resize((size_t)N * K);
      X.resize((size_t)M * K);
      for (auto& v : W) v = rng.uniform();
      for (auto& v : X) v = rng.uniform();
      label = "synthetic:s" + std::to_string(seed);
    }
  } catch (const std::exception& e) {
    std::printf("fixture error: %s\nFAILED!\n", e.what());
    return 1;
  }
  if (M == 0 || N == 0 || K == 0) { std::printf("empty shape\nFAILED!\n"); return 1; }
  if (tol < 0) tol = std::max(1e-4, 2.0 * gamma_k(K));
  std::printf("ps_linear: %s M=%u N=%u K=%u variant=%s tol=%.3e flops=%.0f\n", label.c_str(), M, N, K,
              variant.c_str(), tol, 2.0 * M * N * K);

  bool ok = true;
  std::vector<double> ref64, scale64;
  reference(W, X, M, N, K, ref64, scale64);
  if (!out_fx.empty()) {
    auto st = ps::compare(out_fx.data(), ref64.data(), scale64.data(), (size_t)M * N, tol);
    ps::print_check("fixture_vs_f64", st);
    ok = ok && st.ok();
  }

  Ctx ctx;
  PS_RT_CHECK(vx_device_open(0, &ctx.device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  PS_RT_CHECK(vx_queue_create(ctx.device, &qi, &ctx.queue));
  if (persist && !ps_have_wm) {   // persist_l2 BEFORE the first host write (gemv_host.h:180-184 order; resident control)
    ps::PhaseCfg pc;
    pc.flags = ps::phase_dcr::FLAG_PERSIST_L2;
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
    std::printf("PSPERSIST-HOST: persist_l2=1 (resident control)\n");
  } else if (ps_have_wm) {   // PhaseShift (-W): masks only (every other [dcr_phase] field at reset), before any launch
    ps::PhaseCfg pc;
    if (persist) pc.flags = ps::phase_dcr::FLAG_PERSIST_L2;
    ps::WayMaskCtl wmc;
    PS_RT_CHECK(wmc.set(pc, ps_wm_l1, ps_wm_l2));
    ps::waymask_print(wmc);
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
  }
  ps::Timer timer;

  auto tag = [&](uint32_t l, const char* base) {   // -L 1: the historical labels; -L > 1: launch<l>: prefix
    return launches > 1 ? "launch" + std::to_string(l) + ":" + base : std::string(base);
  };
  if (variant == "small") {
    auto check = [&](uint32_t l, const std::vector<float>& Y) {
      auto s = ps::compare(Y.data(), ref64.data(), scale64.data(), (size_t)M * N, tol);
      ps::print_check(tag(l, "vs_f64").c_str(), s);
      ok = ok && s.ok();
      if (!out_fx.empty()) {
        std::vector<double> rfx(out_fx.begin(), out_fx.end());
        auto sf = ps::compare(Y.data(), rfx.data(), scale64.data(), (size_t)M * N, tol);
        ps::print_check(tag(l, "vs_fixture").c_str(), sf);
        ok = ok && sf.ok();
      }
    };
    if (run_small(ctx, W, X, M, N, K, grid_override, launches, check)) return 1;
  }
#ifdef VX_CFG_TCU_WGMMA_ENABLE
  else if (kI8) {   // P8T: int8 inputs, int32 accumulation, exact integer check
    if (tcu_i8_case(ctx, W, X, M, N, K, launches, ok, tag)) return 1;
  } else {
    if (int bad = ps16::fp16_selftest()) { std::printf("fp16_selftest: %d failures\nFAILED!\n", bad); return 1; }
    std::printf("fp16_selftest: ok\n");
    uint64_t isa = 0, nt = 0, nw = 0, iw = 0;
    PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_ISA_FLAGS, &isa));
    PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
    PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_WARPS, &nw));
    PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_ISSUE_WIDTH, &iw));
    if (!(isa & VX_ISA_EXT_TCU) || !(isa & VX_ISA_EXT_DXA) || nt != VX_CFG_NUM_THREADS || iw > nw) {
      std::printf("device lacks TCU/DXA or NT/ISSUE_WIDTH mismatch (NT=%u, IW=%u, NW=%u)\nFAILED!\n", (unsigned)nt,
                  (unsigned)iw, (unsigned)nw);
      return 1;
    }
    const uint32_t warps = (uint32_t)iw;  // WGMMA group size = ISSUE_WIDTH (stock main.cpp:315-320)
    const uint32_t cta_M = warps * tcfg::xtileM;
    const uint32_t Mp = round_up(M, cta_M), Np = round_up(N, tcfg::xtileN), Kp = round_up(K, tcfg::tileK);
    // offline weight layout: B = W^T (Kp x Np) fp16, zero padded; activations A = X (Mp x Kp) fp16, zero padded
    std::vector<uint16_t> A((size_t)Mp * Kp, 0), B((size_t)Kp * Np, 0);
    std::vector<float> W16((size_t)N * K), X16((size_t)M * K);
    size_t overflow = 0;
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k) {
        float v = W[(size_t)n * K + k];
        if (!(std::fabs(v) <= 65504.0f)) ++overflow;
        uint16_t h = ps16::f32_to_f16(v);
        B[(size_t)k * Np + n] = h;
        W16[(size_t)n * K + k] = ps16::f16_to_f32(h);
      }
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t k = 0; k < K; ++k) {
        float v = X[(size_t)m * K + k];
        if (!(std::fabs(v) <= 65504.0f)) ++overflow;
        uint16_t h = ps16::f32_to_f16(v);
        A[(size_t)m * Kp + k] = h;
        X16[(size_t)m * K + k] = ps16::f16_to_f32(h);
      }
    if (overflow) { std::printf("%zu inputs outside the fp16 range\nFAILED!\n", overflow); return 1; }
    const double pad_ratio = (double)Mp * Np * Kp / ((double)M * N * K);
    std::printf("tcu_padding: M=%u->%u N=%u->%u K=%u->%u padded_flops=%.0f pad_ratio=%.4f\n", M, Mp, N, Np, K, Kp,
                2.0 * Mp * Np * Kp, pad_ratio);
    std::vector<double> ref16, scale16;
    reference(W16, X16, M, N, K, ref16, scale16);
    auto check = [&](uint32_t l, const std::vector<float>& C) {
      std::vector<float> Y((size_t)M * N);
      size_t pad_bad = 0;
      for (uint32_t m = 0; m < Mp; ++m)
        for (uint32_t n = 0; n < Np; ++n) {
          float v = C[(size_t)m * Np + n];
          if (m < M && n < N) Y[(size_t)m * N + n] = v;
          else if (v != 0.0f) ++pad_bad;  // NaN included
        }
      std::printf("PSCHECK: %s n=%zu errors=%zu\n", tag(l, "pad_zero").c_str(), (size_t)Mp * Np - (size_t)M * N, pad_bad);
      ok = ok && pad_bad == 0;
      auto s16 = ps::compare(Y.data(), ref16.data(), scale16.data(), (size_t)M * N, tol);
      ps::print_check(tag(l, "vs_f64_fp16in").c_str(), s16);
      ok = ok && s16.ok();
      if (!out_fx.empty()) {
        std::vector<double> rfx(out_fx.begin(), out_fx.end());
        auto sf = ps::compare(Y.data(), rfx.data(), scale64.data(), (size_t)M * N, tol_fx);
        ps::print_check(tag(l, "vs_fixture").c_str(), sf);
        ok = ok && sf.ok();
      }
    };
    if (run_tcu(ctx, A.data(), B.data(), Mp, Np, Kp, warps, (uint32_t)nt, launches, check)) return 1;
  }
#endif
  std::printf("Elapsed: %.0f ms\n", timer.ms());
  if (!ok) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
