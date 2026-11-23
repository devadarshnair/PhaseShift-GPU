// main.cpp -- ps_silu_gate host app: batched SwiGLU gate on Vortex vs a ps-fixture-v1 `silu_gate`
// fixture or a synthetic case.
//
// Usage: ps_silu_gate [-f <fixture dir> | -b B -n hidden [-s seed]] [-i] [-t tol] [-g grid] [-k kernel.vxbin] [-E musl|poly|mufu]
//   -f  fixture with hb, hb2, out of shape (B, hidden) or (hidden,)
//   synthetic: hb uniform in [-12, 12) (covers both saturated tails of the sigmoid), hb2 uniform in [-1, 1)
//   -i  in place (out = hb buffer, as run.c); default: separate out buffer poisoned with NaN
//   -g  override the block count (grid-stride path)
// Checks: dev_vs_f64, dev_vs_fixture, fixture_vs_f64 (all must pass).
//   -a  synthetic input amplitude: hb uniform in [-amp, amp) (default 12, the range; uses 80 to exercise the |x|-dependent
//       MUFU bound; |hb| must stay < ~88 or expf(-hb) overflows, see below)
//   -E  (methodology §18.7) exp implementation: musl (default: entry silu_gate, behaviour and output unchanged) | poly (entry
//       silu_gate_expf_poly: float-only polynomial exp, ps_expf) | mufu (entry silu_gate_expf_mufu: MUFU ex2; refused unless the device has
//       VX_ISA_EXT_MUFU). poly and mufu are NUMERICS-CHANGING repairs: the output differs from musl's in the last bits, so
//       they are checked against their own derived bound (below) instead of 1e-5, and ps_llama accepts them only
//       token rule (near-tie accepted) and labels the run so. The line `PSEXP:` states variant, bounds and the input range.
//
// Tolerance 1e-5 (chosen here; names none for SiLU-gate), metric rel_i = |y_i - ref_i| / max(|ref_i|, FLT_MIN)
// (elt_host.h). Derivation, u = 2^-24: the formula is a product of factors, so relative errors add (no
// cancellation). expf: the device libm's error e_exp is taken as <= 1 ulp; MEASURED on SimX
//: the device expf (musl in libc32, evaluated in soft-float double, methodology
// §14.0) is correctly rounded, max 0.500 ulp on a 4,096-point grid of [-20, 20] and 2,048 random points of [-87, 87);
// NumPy's float32 exp reached 2.25 ulp on the fixture inputs. A relative error e in E = expf(-v) changes 1 + E by
// e * E/(1+E) <= e; the add, the reciprocal and the two multiplies are correctly rounded (<= u/2 each). Device vs
// float64: <= e_exp + 4 * 0.5 = 3 ulp ~= 1.8e-7 relative (2.5 ulp with the measured 0.5); the 1e-5 tolerance (168 ulp)
// holds for any e_exp up to ~160 ulp. The NumPy float32 fixture uses NumPy's float32 exp with the same 4 roundings,
// so device vs fixture <= ~10 ulp ~= 6e-7 (~7 ulp with the measured errors). 1e-5 leaves a >15x margin and still
// catches any
// wrong formula (e.g. a missing hb2 factor or sigmoid(+v)), which is off by O(1). Inputs with |hb| > ~88 overflow
// expf(-hb) to +inf in float32 (out = -0 instead of a tiny value) in run.c as well; fixtures stay far below that
// (stories models: |hb| < 20) and the synthetic range is [-12, 12).
//
// Derived bound of the numerics-changing variants (u = 2^-24, 1 ulp <= 2u relative). out = v * (1/(1+E)) * hb2 with E = exp(-v):
// a relative error e of the computed E changes 1 + E by e E/(1+E) relative, hence the result by e E/(1+E) (the other factors are exact
// copies); the add, the reciprocal and the two products add 4 x u/2 = 2u (first order; they do not amplify). So per element
//     bound_i = e(x_i) * E_i / (1 + E_i) + 2u,   x_i = -v_i,   E_i = exp(x_i),   e = relative error of the exp implementation.
//   * poly (ps_expf): e = 4u (<= 2 ulp: Cody-Waite reduction exact in its k*ln2 split, degree-7 Taylor truncation 5.2e-9 = 0.09u,
//     Horner roundings; measured <= 2 ulp over 65,541 points of [-104, 89]) -> bound <= 6u = 3.6e-7 for every v.
//   * mufu (vx_expf_fast = ex2(x * fl(log2 e)), errata 27): e(x) = (c_L |x| + 4) u with c_L = 1 + |fl(log2 e) - log2 e| / (log2 e u)
//     (the prescale's constant and product error moved into the exponent: ex2's argument y = x log2 e carries relative error
//     <= u + delta_L, and d(2^y)/2^y = ln2 dy, ln2 log2 e = 1) plus ex2's 2 ulp = 4u -> grows with |x|: at |v| = 12 (the synthetic range)
//     (c_L 12 + 4) u = 1.2e-6; at |v| = 20 (stories activations) 2.0e-6; at |v| = 88, 6.8e-6 (still below 1e-5).
// tol of a check = max_i bound_i over the inputs actually used (a single scalar for ps::compare); dev_vs_fixture adds the NumPy
// fixture's own bound (NumPy float32 exp measured <= 2.25 ulp <= 4.5u, taken as 5u: 5u E/(1+E) + 2u <= 7u).
// For musl the tolerance stays 1e-5 (unchanged behaviour). `-t` overrides every tolerance (the liveness tests).
#include "common.h"
#include "elt_host.h"

static void usage(const char* app) {
  std::printf("Usage: %s [-f fixture_dir | -b B -n hidden [-s seed]] [-i] [-t tol] [-g grid] [-k kernel.vxbin] [-W l1:l2] [-E musl|poly|mufu] [-a amp]\n", app);
}

int main(int argc, char** argv) {
  std::string fixture, kernel_file = "kernel.vxbin";
  uint32_t B = 4, n = 768, seed = 1, grid_override = 0;
  bool inplace = false, tol_given = false;
  std::string expv = "musl";
  float amp = 12.0f;   // -a: synthetic hb amplitude (hb uniform in [-amp, amp)); default 12 = the range
  double tol = 1e-5;
  int c;
  while ((c = getopt(argc, argv, "f:b:n:s:it:g:k:W:E:a:h")) != -1) {
    switch (c) {
      case 'f': fixture = optarg; break;
      case 'b': B = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': n = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'i': inplace = true; break;
      case 't': tol = std::strtod(optarg, nullptr); tol_given = true; break;
      case 'g': grid_override = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      case 'W':   // PhaseShift: [dcr_phase] way masks "<l1word>:<l2word>" (elt_host.h, methodology §14.A)
        if (!ps_elt::waymask_opt(optarg)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        break;
      case 'E': expv = optarg; break;
      case 'a': amp = (float)std::strtod(optarg, nullptr); break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  std::vector<float> hb, hb2, out_fx;
  std::string label;
  try {
    if (!fixture.empty()) {
      ps::Fixture fx(fixture);
      if (fx.op() != "silu_gate") throw ps::FixtureError("op " + fx.op() + " is not silu_gate");
      ps_elt::Rows a = ps_elt::load_rows(fx, "hb"), b = ps_elt::load_rows(fx, "hb2"), o = ps_elt::load_rows(fx, "out");
      ps_elt::expect_same_shape(a, b, "hb vs hb2");
      ps_elt::expect_same_shape(a, o, "hb vs out");
      B = a.B; n = a.n;
      hb = std::move(a.data); hb2 = std::move(b.data); out_fx = std::move(o.data);
      label = ps_elt::fixture_label(fx, B);
    } else {
      ps::Rng rng(seed);
      hb.resize((size_t)B * n); hb2.resize((size_t)B * n);
      for (auto& v : hb) v = amp * rng.uniform();
      for (auto& v : hb2) v = rng.uniform();
      label = "synthetic:s" + std::to_string(seed);
    }
  } catch (const std::exception& e) {
    std::printf("fixture error: %s\nFAILED!\n", e.what());
    return 1;
  }
  const size_t N = (size_t)B * n;
  if (N == 0) { std::printf("empty problem\nFAILED!\n"); return 1; }
  if (expv != "musl" && expv != "poly" && expv != "mufu") {
    std::printf("bad -E %s (musl|poly|mufu)\nFAILED!\n", expv.c_str());
    return 1;
  }
  std::printf("ps_silu_gate: %s B=%u hidden=%u inplace=%d tol=%.1e exp=%s\n", label.c_str(), B, n, (int)inplace, tol, expv.c_str());

  std::vector<double> ref(N);
  for (size_t j = 0; j < N; ++j) {
    double v = hb[j];
    ref[j] = v * (1.0 / (1.0 + std::exp(-v))) * (double)hb2[j];
  }
  std::vector<double> scale = ps_elt::elem_scale(ref);
  // derived bounds of the numerics-changing variants (header comment); musl keeps the tolerance
  const double kU = std::ldexp(1.0, -24);
  const double kL2e = 1.44269504088896340736, cL = 1.0 + std::fabs((double)(float)kL2e - kL2e) / (kL2e * kU);
  double bound_dev = 0.0, bound_fx = 0.0, max_abs_v = 0.0;
  if (expv != "musl") {
    for (size_t j = 0; j < N; ++j) {
      const double x = -(double)hb[j], E = std::exp(x), w = E / (1.0 + E);
      const double e = (expv == "poly") ? 4.0 * kU : (cL * std::fabs(x) + 4.0) * kU;
      bound_dev = std::fmax(bound_dev, e * w + 2.0 * kU);
      bound_fx = std::fmax(bound_fx, 5.0 * kU * w + 2.0 * kU);
      max_abs_v = std::fmax(max_abs_v, std::fabs(x));
    }
    std::printf("PSEXP: variant=%s numerics_changing=1 max_abs_v=%.4g c_L=%.4f bound_dev=%.3e bound_fixture=%.3e (u=2^-24)\n",
                expv.c_str(), max_abs_v, cL, bound_dev, bound_fx);
  } else {
    std::printf("PSEXP: variant=musl numerics_changing=0 tol=%.1e\n", tol);
  }
  const double tol_dev = tol_given || expv == "musl" ? tol : bound_dev;
  const double tol_fx = tol_given || expv == "musl" ? tol : bound_dev + bound_fx;
  bool ok = true;
  if (!out_fx.empty()) ok = ps_elt::check("fixture_vs_f64", out_fx.data(), ref, scale, tol) && ok;

  ps_elt::Dev dev;
  const char* kname = expv == "poly" ? "silu_gate_expf_poly" : expv == "mufu" ? "silu_gate_expf_mufu" : "silu_gate";
  if (dev.open(kernel_file, kname)) return 1;
  if (expv == "mufu") {
    uint64_t isa = 0;
    PS_RT_CHECK(vx_device_query(dev.device, VX_CAPS_ISA_FLAGS, &isa));
    if (!(isa & VX_ISA_EXT_MUFU)) {
      std::printf("-E mufu needs a device built with -DVX_CFG_EXT_MUFU_ENABLE\nFAILED!\n");
      return 1;
    }
  }
  const std::vector<float> poison(N, std::nanf(""));
  ps_elt::Buf ba, bb, bo;
  if (dev.alloc(ba, N * 4, VX_MEM_READ_WRITE, hb.data())) return 1;
  if (dev.alloc(bb, N * 4, VX_MEM_READ, hb2.data())) return 1;
  if (!inplace && dev.alloc(bo, N * 4, VX_MEM_READ_WRITE, poison.data())) return 1;
  ps_silu_gate_arg_t arg = {};
  arg.hb_addr = ba.addr; arg.hb2_addr = bb.addr; arg.out_addr = inplace ? ba.addr : bo.addr; arg.n = (uint32_t)N;

  ps::Timer timer;
  vx_event_h ev = nullptr;
  if (dev.launch(&arg, sizeof(arg), (uint32_t)N, grid_override, &ev)) return 1;
  std::vector<float> y(N);
  if (dev.read(y.data(), inplace ? ba : bo, ev)) return 1;
  vx_event_release(ev);
  ok = ps_elt::check("dev_vs_f64", y.data(), ref, scale, tol_dev) && ok;
  if (!out_fx.empty()) ok = ps_elt::check("dev_vs_fixture", y.data(), ps_elt::to_f64(out_fx), scale, tol_fx) && ok;
  return ps_elt::finish(ok, timer);
}
