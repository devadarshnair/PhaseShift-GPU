// main.cpp -- ps_softmax_fixed host app: row-wise softmax on Vortex, checked against a float64 host
// softmax (and against the NumPy float32 `probs` of a attention fixture with -f); also an exp probe (-p).
//
// Usage: ps_softmax_fixed [-f <attention fixture dir> | -r <rows> -n <cols> [-a <lo> -b <hi>] [-s <seed>]]
//                         [-v warp|mufu|libm|stock|stock_expf] [-t <tol>] [-g <grid>] [-x] [-k <kernel.vxbin>]
//        ps_softmax_fixed -p sw|mufu|libm [-n <points>] [-a <lo> -b <hi>]
//   -f   ps-fixture-v1 `attention` fixture: input = `scores` (H, T), NumPy reference = `probs` (H, T)
//   -r/-n/-a/-b/-s  synthetic case: rows x cols values uniform in [lo, hi) (default [-1, 1)) from the deterministic
//        xorshift RNG (ps_common.h), seed -s (default 1)
//   -v   kernel (kernel.cpp): warp (default; ps_expf), mufu (MUFU EX2; refused unless the device has the MUFU
//        extension), libm (musl expf), stock (verbatim stock regression kernel), stock_expf (stock with ps_expf)
//   -t   override the tolerance (default: derived, see "Error bound")
//   -g   override the number of blocks (exercises the grid-stride loop over rows)
//   -x   report-only: print the checks and a PSVERDICT line, but exit 0 / PASSED! even if the result is wrong (used
//        to document the stock kernel's behaviour on the bug input without recording a failed run)
//   -p   exp probe: evaluate one exp implementation at -n points (default 65536) evenly spaced over [lo, hi]
//        (default [-104, 89]) plus the special points -110, -87.5, -1e-7, 0, 1e-7, and check it against float64 exp
//        (see "Exp probe")
// Exactly one kernel launch per app run (methodology §2 consequence 5: per-kernel cycles from the first launch).
//
// Error metric (pass criterion, ps_common.h): rel_j = |p_j - ref_j| / max(ref_j, tau), tau = 2^-100. Elements with
// ref_j >= tau are checked relatively; smaller ones (which also cover MUFU's flush-to-zero of subnormal exp values,
// and everything below float32's normal range 2^-126) only absolutely, |p_j - ref_j| <= tol * tau.
//
// Error bound (modeling choice; the softmax analogue of 2*gamma_K bound, methodology §2 consequence 6 and
// §14.1). With u = 2^-24, d_j = x_j - m <= 0, e_j = exp(d_j), S = sum_k e_j, w_k = e_k / S, write the computed
// e^_k = e_k (1 + eps_k). To first order the computed p^_j = e^_j / S^ satisfies
//     p^_j / p_j - 1 = eps_j - sum_k w_k eps_k - eps_sum + eps_rcp + eps_mul
// (the sum_k w_k eps_k term -- the other elements' exp errors reaching S -- was missing;
//). The parts:
//     |eps_k| <= (1 + c) |d_k| u + E_exp
//                      (1 + c)|d_k| u: rounding of the subtraction d_k = x_k - m, which shifts the exponent by
//                      |d_k| u; c = 0 except for MUFU (below)
//                      E_exp: the exp itself: ps_expf <= 2 ulp -> 4u (asserted by the exp probe test); musl expf
//                      <= 1 ulp -> 2u; MUFU: ex2 <= 2 ulp -> 4u, and c = c_L = 1.224 for the prescale
//                      y = d_k * fl(log2 e): its rounding (relative u on y, i.e. |d_k| u on the result) plus the
//                      constant's own representation error (fl(log2 e) = 0x3fb8aa3b is 0.224 u below log2 e)
//                      (computed in kMufuPrescale; the first MUFU probe run, which allowed only |x| u, exceeded it by
//                      up to 1.12x at 187 of 65,541 points, which is how c_L was found)
//     |sum_k w_k eps_k| <= (1 + c) W(n) u + E_exp, with W(n) = max sum_k w_k |d_k| over rows of length n (one
//                      d = 0, the rest <= 0). Setting d/dt_k = 0 gives t_k = 1 + W for every non-max k, so the
//                      maximum has all n - 1 others at one common t: W(n) = max_t (n-1) t e^-t / (1 + (n-1) e^-t)
//                      (W_bound below; 2.31, 3.34, 4.44, 5.60 at n = 64, 256, 1024, 4096; 0 at n = 1)
//     |eps_sum| <= h u the summation of S: every term passes through at most h roundings (kernel.cpp "Summation":
//                      h = (B-1) + ceil(ceil(n/NT)/B) + log2 NT for the warp kernels, h = n - 1 for the stock-shaped
//                      kernels)
//     |eps_rcp| + |eps_mul| <= 2u   reciprocal 1/S and the multiply e_j * (1/S)
// Over the checked elements (ref_j >= tau): p_j >= tau and S >= 1 imply e^{d_j} = p_j S >= tau, so
// |d_j| <= 100 ln 2 = 69.3. Hence (gamma(k) = k u / (1 - k u) adds the higher-order terms)
//     bound = gamma((1 + c) 69.3 + E_exp + (1 + c) W(n) + E_exp + h + 2).
// At n = 4096, NT = 4 (h = 65, W = 5.60): bound_sw = 8.94e-6 <= 1e-5 (the earlier tolerance holds for every tested
// length; 6.75e-6 at n = 1), bound_mufu = 1.44e-5 > 1e-5 (1.18e-5 at n = 1). The tolerance used is
// tol = max(1e-5, bound), printed in every PSCHECK line, so the software-exp kernel is held to the 1e-5 and the
// MUFU kernel to its own derived bound (a deviation recorded). The comparison against the
// fixture's NumPy float32 probs adds NumPy's own error, gamma(69.3 + 8 + 5.6 + 8 + n + 1) (exp <= 4 ulp assumed,
// W <= 5.6, sequential-sum worst case n), to the device bound.
// Typical errors are far below these worst cases (roundings are not aligned); the observed maxima are reported.
//
// Exp probe: allowed relative error per point (vs float64 exp of the same float input), for outputs in float32's
// normal range: sw 4u (2 ulp), libm 2u (1 ulp), mufu (c_L |x| + 4) u (2 ulp of ex2 plus the prescale, the same
// terms as the bound above). Subnormal outputs: absolute error <= 2^-149 (sw, libm: one subnormal ulp) or <= 2^-126
// (mufu: flush to zero). The PSCHECK line reports max(err / allowed), so tol = 1 (pass iff every point is within its
// allowed error); a PSEXP line reports the largest error in ulps of the float64 result.
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <vortex2.h>

#include "common.h"
#include "ps_common.h"
#include "ps_fixture.h"
#include "ps_phase.h"     // -W: the [dcr_phase] block write
#include "ps_waymask.h"   // -W: mechanism A2 way masks (encoding, methodology §14.A)

namespace {

constexpr double kU = 1.0 / 16777216.0;            // 2^-24
constexpr double kTau = 7.888609052210118e-31;     // 2^-100
constexpr double kDLim = 69.31471805599453;        // 100 ln 2
constexpr double kPlanTol = 1e-5;
constexpr double kMinNormal = 1.1754943508222875e-38;   // 2^-126
constexpr double kMinSub = 1.401298464324817e-45;       // 2^-149
constexpr double kLog2e = 1.4426950408889634074;
// c_L: exponent error factor of MUFU's prescale x * fl(log2 e) in units of |x| u (header comment, E_exp)
const double kMufuPrescale = 1.0 + std::fabs((double)(float)kLog2e - kLog2e) / (kLog2e * kU);

double gamma_k(double k) { return k * kU / (1.0 - k * kU); }

uint32_t ilog2(uint32_t v) { uint32_t r = 0; while (v > 1) { v >>= 1; ++r; } return r; }

bool is_stock(const std::string& v) { return v == "stock" || v == "stock_expf"; }

// chain length h of the device summation (see the header comment and kernel.cpp)
double chain_len(const std::string& variant, uint32_t n, uint32_t nt) {
  if (is_stock(variant)) return n > 0 ? n - 1 : 0;
  const uint32_t per_lane = (n + nt - 1) / nt;
  const uint32_t blocks = (per_lane + PS_SOFTMAX_SUM_BLOCK - 1) / PS_SOFTMAX_SUM_BLOCK;
  return (PS_SOFTMAX_SUM_BLOCK - 1) + blocks + ilog2(nt);
}

// W(n) = max over rows of sum_k w_k |d_k| (header comment): (n-1) t e^-t / (1 + (n-1) e^-t) maximised over t by a
// 1e-3 grid on [0, 60] (the maximiser t* = 1 + W is < 12 for n <= 2^20; the function is smooth, so the grid error is
// < 1e-5 in W, negligible next to the other terms)
double W_bound(uint32_t n) {
  if (n <= 1) return 0.0;
  double best = 0.0;
  for (int i = 1; i <= 60000; ++i) {
    const double t = 1e-3 * i, e = (double)(n - 1) * std::exp(-t);
    best = std::fmax(best, t * e / (1.0 + e));
  }
  return best;
}

double device_bound(const std::string& variant, uint32_t n, uint32_t nt) {
  const double h = chain_len(variant, n, nt), W = W_bound(n);
  // gamma((1 + c) d_lim + E_exp   [own element]  + (1 + c) W + E_exp   [the others, through S]  + h + 2)
  if (variant == "mufu") return gamma_k((1.0 + kMufuPrescale) * (kDLim + W) + 2 * 4 + h + 2);
  if (variant == "libm") return gamma_k(kDLim + W + 2 * 2 + h + 2);
  return gamma_k(kDLim + W + 2 * 4 + h + 2);   // warp / stock_expf (ps_expf), and what a correct stock kernel would meet
}

struct Ctx {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  vx_buffer_h src_buf = nullptr, dst_buf = nullptr;
  ~Ctx() {
    if (!device) return;
    if (src_buf) vx_buffer_release(src_buf);
    if (dst_buf) vx_buffer_release(dst_buf);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
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
  std::printf("Usage: %s [-f fixture | -r rows -n cols [-a lo -b hi] [-s seed]] [-v warp|mufu|libm|stock|stock_expf] "
              "[-t tol] [-g grid] [-x] [-W l1:l2] [-k kernel.vxbin]\n       %s -p sw|mufu|libm [-n points] [-a lo -b hi]\n",
              app, app);
}

// float64 softmax of one row (the reference): exact max, exp(x - m) in double, sum, divide
void softmax_f64(const float* x, uint32_t n, double* out) {
  double m = -INFINITY;
  for (uint32_t j = 0; j < n; ++j) m = std::fmax(m, (double)x[j]);
  double s = 0.0;
  for (uint32_t j = 0; j < n; ++j) { out[j] = std::exp((double)x[j] - m); s += out[j]; }
  for (uint32_t j = 0; j < n; ++j) out[j] /= s;
}

size_t count_nonfinite(const std::vector<float>& v) {
  size_t k = 0;
  for (float f : v) k += !std::isfinite(f);
  return k;
}

// launch one kernel on (src -> dst), N elements, `global` threads; dst is poisoned with NaN first
int launch(Ctx& ctx, const char* kname, ps_softmax_arg_t& arg, const std::vector<float>& X, std::vector<float>& Y,
           uint32_t global, uint32_t grid_override, bool need_whole_warps, uint64_t nt) {
  const uint64_t bytes = (uint64_t)X.size() * sizeof(float);
  PS_RT_CHECK(vx_buffer_create(ctx.device, bytes, VX_MEM_READ_WRITE, &ctx.src_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.src_buf, &arg.src_addr));
  PS_RT_CHECK(vx_buffer_create(ctx.device, bytes, VX_MEM_READ_WRITE, &ctx.dst_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.dst_buf, &arg.dst_addr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.src_buf, 0, X.data(), bytes, 0, nullptr, nullptr));
  const std::vector<float> poison(X.size(), std::nanf(""));   // every element must be written by the kernel
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.dst_buf, 0, poison.data(), bytes, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, kname, &ctx.kernel));
  uint32_t grid_dim[1], block_dim[1];
  PS_RT_CHECK(vx_device_max_occupancy_grid(ctx.device, 1, &global, grid_dim, block_dim));
  if (need_whole_warps && block_dim[0] % nt != 0) {
    std::printf("block %u not a multiple of NT %u\nFAILED!\n", block_dim[0], (unsigned)nt);
    return 1;
  }
  if (grid_override > 0) grid_dim[0] = grid_override;
  std::printf("launch: kernel=%s global=%u grid=%u block=%u%s\n", kname, global, grid_dim[0], block_dim[0],
              grid_override > 0 ? " (grid overridden by -g)" : "");
  Y.assign(X.size(), 0.0f);
  ps::Timer timer;
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = ctx.kernel;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = grid_dim[0];
  li.block_dim[0] = block_dim[0];
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  PS_RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &launch_ev));
  PS_RT_CHECK(vx_enqueue_read(ctx.queue, Y.data(), ctx.dst_buf, 0, bytes, 1, &launch_ev, &read_ev));
  PS_RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  std::printf("Elapsed: %.0f ms\n", timer.ms());
  return 0;
}

int exp_probe(Ctx& ctx, const std::string& impl, uint32_t npts, double lo, double hi) {
  std::vector<float> X;
  for (float s : {-110.0f, -87.5f, -1e-7f, 0.0f, 1e-7f}) X.push_back(s);
  for (uint32_t i = 0; i < npts; ++i) X.push_back((float)(lo + (hi - lo) * (npts > 1 ? (double)i / (npts - 1) : 0.0)));
  const uint32_t N = (uint32_t)X.size();
  std::printf("ps_softmax exp probe: impl=%s points=%u range=[%g,%g] (+5 specials)\n", impl.c_str(), N - 5, lo, hi);
  ps_softmax_arg_t arg = {};
  arg.rows = 1;
  arg.cols = N;
  std::vector<float> Y;
  const std::string kname = "exp_probe_" + impl;
  if (launch(ctx, kname.c_str(), arg, X, Y, N, 0, false, 1)) return 1;
  // per point: ratio = err / allowed (PSCHECK with scale = allowed, tol = 1)
  std::vector<double> ref(N), allowed(N);
  double max_ulp = 0.0, max_rel_normal = 0.0;
  size_t worst_ulp = 0;
  for (uint32_t i = 0; i < N; ++i) {
    const double r = std::exp((double)X[i]);
    ref[i] = r;
    if (r >= kMinNormal && r <= 3.4028234663852886e38) {
      const double rel_allow = (impl == "sw") ? 4 * kU : (impl == "libm") ? 2 * kU : (kMufuPrescale * std::fabs((double)X[i]) + 4) * kU;
      allowed[i] = rel_allow * r;
      const double ulp = std::ldexp(1.0, std::ilogb(r) - 23);
      const double e = std::fabs((double)Y[i] - r);
      if (std::isfinite((double)Y[i]) && e / ulp > max_ulp) { max_ulp = e / ulp; worst_ulp = i; }
      if (std::isfinite((double)Y[i])) max_rel_normal = std::fmax(max_rel_normal, e / r);
    } else if (r < kMinNormal) {
      allowed[i] = (impl == "mufu") ? kMinNormal : kMinSub;
    } else {
      allowed[i] = 0.0;   // overflow region: must be +inf exactly; compare() needs finite y, so handled below
    }
  }
  // overflow points (none in the default range) must give +inf
  size_t ovf_bad = 0;
  std::vector<float> Yc(Y);
  for (uint32_t i = 0; i < N; ++i)
    if (allowed[i] == 0.0) { ovf_bad += !(std::isinf(Y[i]) && Y[i] > 0); Yc[i] = 0.0f; ref[i] = 0.0; allowed[i] = 1.0; }
  auto st = ps::compare(Yc.data(), ref.data(), allowed.data(), N, 1.0);
  st.errors += ovf_bad;
  const std::string lab = "exp_" + impl + ":err_over_allowed";
  ps::print_check(lab.c_str(), st);
  std::printf("PSEXP: impl=%s max_ulp=%.3f at_x=%.9g max_rel_normal=%.3e worst_ratio_x=%.9g y=%.9g ref=%.9g "
              "overflow_bad=%zu\n", impl.c_str(), max_ulp, (double)X[worst_ulp], max_rel_normal, (double)X[st.worst],
              (double)Y[st.worst], std::exp((double)X[st.worst]), ovf_bad);
  if (!st.ok()) { std::printf("FAILED!\n"); return 1; }
  std::printf("PASSED!\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixture, variant = "warp", kernel_file = "kernel.vxbin", probe;
  uint32_t R = 4, n = 0, seed = 1, grid_override = 0;
  double lo = NAN, hi = NAN, tol_override = 0.0;
  bool report_only = false;
  bool ps_have_wm = false;   // PhaseShift: -W way masks (absent = no DCR write, the old host path)
  uint32_t ps_wm_l1 = 0, ps_wm_l2 = 0;
  int c;
  while ((c = getopt(argc, argv, "f:r:n:a:b:s:v:t:g:xp:k:W:h")) != -1) {
    switch (c) {
      case 'f': fixture = optarg; break;
      case 'r': R = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': n = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'a': lo = std::strtod(optarg, nullptr); break;
      case 'b': hi = std::strtod(optarg, nullptr); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'v': variant = optarg; break;
      case 't': tol_override = std::strtod(optarg, nullptr); break;
      case 'g': grid_override = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'x': report_only = true; break;
      case 'p': probe = optarg; break;
      case 'k': kernel_file = optarg; break;
      case 'W':   // PhaseShift: [dcr_phase] way masks "<l1word>:<l2word>" (ps_waymask.h, methodology §14.A)
        if (!ps::parse_waymask_arg(optarg, &ps_wm_l1, &ps_wm_l2)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        ps_have_wm = true;
        break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  const bool warp_kernel = (variant == "warp" || variant == "mufu" || variant == "libm");
  if (probe.empty() && !warp_kernel && !is_stock(variant)) { std::printf("bad -v %s\nFAILED!\n", variant.c_str()); return 1; }
  if (!probe.empty() && probe != "sw" && probe != "mufu" && probe != "libm") {
    std::printf("bad -p %s\nFAILED!\n", probe.c_str());
    return 1;
  }

  // ---- device
  Ctx ctx;
  PS_RT_CHECK(vx_device_open(0, &ctx.device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  PS_RT_CHECK(vx_queue_create(ctx.device, &qi, &ctx.queue));
  if (ps_have_wm) {   // PhaseShift (-W): masks only (every other [dcr_phase] field at reset), before any launch
    ps::PhaseCfg pc;
    ps::WayMaskCtl wmc;
    PS_RT_CHECK(wmc.set(pc, ps_wm_l1, ps_wm_l2));
    ps::waymask_print(wmc);
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
  }
  uint64_t nt = 0, isa = 0;
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_ISA_FLAGS, &isa));
  const bool has_mufu = (isa & VX_ISA_EXT_MUFU) != 0;
  if ((variant == "mufu" || probe == "mufu") && !has_mufu) {
    std::printf("MUFU kernels need a device built with -DVX_CFG_EXT_MUFU_ENABLE\nFAILED!\n");
    return 1;
  }
  PS_RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
  if (!probe.empty())
    return exp_probe(ctx, probe, n ? n : 65536, std::isnan(lo) ? -104.0 : lo, std::isnan(hi) ? 89.0 : hi);
  if (std::isnan(lo)) lo = -1.0;
  if (std::isnan(hi)) hi = 1.0;
  if (n == 0) n = 64;

  // ---- inputs
  std::vector<float> X, probs_fx;
  std::string label;
  try {
    if (!fixture.empty()) {
      ps::Fixture fx(fixture);
      if (fx.op() != "attention") throw ps::FixtureError("need an attention fixture (scores, probs)");
      ps::Tensor ts = fx.tensor("scores"), tp = fx.tensor("probs");
      if (ts.shape.size() != 2 || ts.shape != tp.shape) throw ps::FixtureError("scores/probs shape mismatch");
      R = ts.dim(0);
      n = ts.dim(1);
      X = std::move(ts.data);
      probs_fx = std::move(tp.data);
      label = fx.model() + ":" + fx.op() + ":L" + std::to_string((int)fx.meta().at("layer").as_num()) + "_P" +
              std::to_string((int)fx.meta().at("pos").as_num());
    } else {
      if (R == 0 || n == 0 || !(hi > lo)) { std::printf("need rows, cols >= 1 and hi > lo\nFAILED!\n"); return 1; }
      ps::Rng rng(seed);
      X.resize((size_t)R * n);
      for (auto& v : X) v = (float)(lo + (hi - lo) * 0.5 * ((double)rng.uniform() + 1.0));
      char buf[96];
      std::snprintf(buf, sizeof(buf), "synthetic:U[%g,%g):s%u", lo, hi, seed);
      label = buf;
    }
  } catch (const std::exception& e) {
    std::printf("fixture error: %s\nFAILED!\n", e.what());
    return 1;
  }
  const size_t N = (size_t)R * n;

  const double bound = device_bound(variant, n, (uint32_t)nt);
  const double tol = tol_override > 0 ? tol_override : std::fmax(kPlanTol, bound);
  std::printf("ps_softmax: %s rows=%u cols=%u variant=%s NT=%u mufu_hw=%d sum_chain_h=%.0f bound=%.3e tol=%.3e%s\n",
              label.c_str(), R, n, variant.c_str(), (unsigned)nt, (int)has_mufu, chain_len(variant, n, (uint32_t)nt),
              bound, tol, report_only ? " report-only" : "");

  // ---- float64 reference, scale = max(ref, tau)
  std::vector<double> ref(N), scale(N);
  for (uint32_t r = 0; r < R; ++r) softmax_f64(&X[(size_t)r * n], n, &ref[(size_t)r * n]);
  for (size_t i = 0; i < N; ++i) scale[i] = std::fmax(ref[i], kTau);
  bool ok = true;
  double tol_fx = 0.0;
  if (!probs_fx.empty()) {
    // sanity of the input: NumPy float32 probs vs float64, NumPy's own bound
    const double np_bound = gamma_k(kDLim + 8 + W_bound(n) + 8 + n + 1);
    auto st = ps::compare(probs_fx.data(), ref.data(), scale.data(), N, std::fmax(kPlanTol, np_bound));
    ps::print_check("fixture_vs_f64", st);
    ok = ok && st.ok();
    tol_fx = tol_override > 0 ? tol_override : std::fmax(kPlanTol, bound + np_bound);
  }

  ps_softmax_arg_t arg = {};
  arg.rows = R;
  arg.cols = n;
  const std::string kname = (variant == "mufu") ? "softmax_warp_mufu" : (variant == "libm") ? "softmax_warp_libm"
                                                                     : "softmax_" + variant;
  // warp kernels: one warp (nt threads) per row; stock-shaped kernels: one thread per row (as the stock app)
  std::vector<float> Y;
  if (launch(ctx, kname.c_str(), arg, X, Y, warp_kernel ? (uint32_t)(R * nt) : R, grid_override, warp_kernel, nt))
    return 1;

  auto s64 = ps::compare(Y.data(), ref.data(), scale.data(), N, tol);
  ps::print_check("launch0:vs_f64", s64);
  bool dev_ok = s64.ok();
  if (!probs_fx.empty()) {
    std::vector<double> rfx(probs_fx.begin(), probs_fx.end()), sfx(N);
    for (size_t i = 0; i < N; ++i) sfx[i] = std::fmax(rfx[i], kTau);
    auto st = ps::compare(Y.data(), rfx.data(), sfx.data(), N, tol_fx);
    ps::print_check("launch0:vs_fixture", st);
    dev_ok = dev_ok && st.ok();
  }
  // informational: does the result also meet the 1e-5 (only differs from the check for the MUFU kernel)?
  // nonfinite = NaN/Inf outputs (e.g. 0/0 when every exp of a row underflows)
  std::printf("PSINFO: within_plan_tol=%d plan_tol=%.1e max_rel_err=%.6e nonfinite=%zu worst_row=%zu worst_col=%zu "
              "worst_x=%.9g worst_ref=%.9g worst_y=%.9g\n",
              (int)(s64.max_rel <= kPlanTol), kPlanTol, s64.max_rel, count_nonfinite(Y), s64.worst / n, s64.worst % n,
              (double)X[s64.worst], ref[s64.worst], (double)Y[s64.worst]);
  std::printf("PSVERDICT: %s %s\n", variant.c_str(), dev_ok ? "correct" : "wrong");
  ok = ok && dev_ok;
  if (!ok && !report_only) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
