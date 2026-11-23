// main.cpp -- ps_rmsnorm host app: batched RMSNorm on Vortex vs a ps-fixture-v1 fixture or a
// synthetic case.
//
// Usage: ps_rmsnorm [-f <fixture dir> | -b <B> -n <n> [-s <seed>]] [-i] [-t <tol>] [-g <grid>] [-k <kernel.vxbin>]
//   -f  fixture with x (B, n) or (n,), weight (n,), out (like x): op rmsnorm_att / rmsnorm_ffn / rmsnorm_final,
//       from model/make_fixtures.py (`batch` subcommand: B in {1, 4}; per-position fixtures: B = 1)
//   -b/-n/-s  synthetic: x uniform in [-2, 2), w uniform in [-1, 1) (xorshift RNG, ps_common.h)
//   -i  in place (out = x buffer, as run.c's final rmsnorm(x, x, ...)); default: separate out buffer poisoned with
//       NaN before the launch, so an element the kernel never writes fails the check
//   -g  override the block count (grid-stride path: a warp handles several rows)
// Checks (all must pass): dev_vs_f64 (device out vs a float64 evaluation of the same formula from the same float32
// inputs), dev_vs_fixture (vs the fixture's NumPy float32 out), fixture_vs_f64 (sanity of the fixture itself).
//
// Tolerance 1e-5, metric rel_i = |y_i - ref_i| / max(|ref_i|, FLT_MIN) (elt_host.h). Derivation (u =
// 2^-24): sum of squares has only non-negative terms, so its relative error is <= gamma_m ~= m u with m the summation
// depth (kernel.cpp: m = ceil(n/NT) + log2 NT; FMA contraction only lowers it). /n and +eps add <= 2u, sqrtf halves the
// argument's relative error and adds u, then 1/x, ss*x and w*(.) add u each (all correctly rounded IEEE ops on the
// F extension). Device vs float64:  rel <= (m/2 + 5) u.  stories15M (n = 288, NT = 4): m = 74 -> 42u = 2.5e-6;
// stories260K (n = 64): m = 18 -> 14u = 8.3e-7; both well below 1e-5. The fixture (NumPy float32, BLAS sdot for the
// sum) obeys the same form with its own m <= n: worst case (288/2 + 5) u = 8.9e-6, so device vs fixture is bounded
// by the sum of the two (<= 1.1e-5 for n = 288 in the worst case, typically ~1e-7 because rounding errors do not all
// align); fixture_vs_f64 is printed, so a dev_vs_fixture failure can be attributed. For n with m/2 + 5 > 167 (i.e.
// ceil(n/NT) > ~320) the 1e-5 tolerance is no longer guaranteed and a step must state its own.
// Tolerance used (the worst-case device-vs-fixture bound 1.1e-5 at n = 288 exceeded the
// fixed 1e-5, a paper gap -- observed <= 3.2e-7): each check uses tol = max(1e-5, its worst-case bound), as:
//   dev_vs_f64      max(1e-5, (m/2 + 5) u),  m = ceil(n/NT) + log2 NT  (device depth, kernel.cpp)
//   fixture_vs_f64  max(1e-5, (n/2 + 5) u)                             (NumPy depth taken as its worst case m <= n:
//                                                                       BLAS sdot's partial-sum order is not fixed)
//   dev_vs_fixture  max(1e-5, (m/2 + 5) u + (n/2 + 5) u)               (triangle inequality through float64)
// so 1e-5 stays the tolerance wherever it is provably sufficient (n = 64: 8.3e-7 + 2.2e-6) and is widened only to
// the derived bound (n = 288, NT = 4: dev_vs_fixture 1.14e-5). -t forces one tolerance for every check (tests use it
// to show the checks are live).
#include "common.h"
#include "elt_host.h"

static void usage(const char* app) {
  std::printf("Usage: %s [-f fixture_dir | -b B -n n [-s seed]] [-i] [-t tol] [-g grid] [-k kernel.vxbin] [-W l1:l2]\n", app);
}

int main(int argc, char** argv) {
  std::string fixture, kernel_file = "kernel.vxbin";
  uint32_t B = 4, n = 288, seed = 1, grid_override = 0;
  bool inplace = false;
  double tol = 1e-5;
  bool tol_forced = false;
  float eps = 1e-5f;  // run.c rmsnorm: ss += 1e-5f
  int c;
  while ((c = getopt(argc, argv, "f:b:n:s:it:g:k:W:h")) != -1) {
    switch (c) {
      case 'f': fixture = optarg; break;
      case 'b': B = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': n = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'i': inplace = true; break;
      case 't': tol = std::strtod(optarg, nullptr); tol_forced = true; break;
      case 'g': grid_override = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      case 'W':   // PhaseShift: [dcr_phase] way masks "<l1word>:<l2word>" (elt_host.h, methodology §14.A)
        if (!ps_elt::waymask_opt(optarg)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  std::vector<float> x, w, out_fx;
  std::string label;
  try {
    if (!fixture.empty()) {
      ps::Fixture fx(fixture);
      if (fx.op().rfind("rmsnorm", 0) != 0) throw ps::FixtureError("op " + fx.op() + " is not rmsnorm_*");
      ps_elt::Rows rx = ps_elt::load_rows(fx, "x"), ro = ps_elt::load_rows(fx, "out");
      ps::Tensor tw = fx.tensor("weight");
      ps_elt::expect_same_shape(rx, ro, "x vs out");
      if (tw.shape.size() != 1 || tw.dim(0) != rx.n) throw ps::FixtureError("weight must be (n,)");
      if (std::fabs(ps_elt::param_num(fx, "eps") - 1e-5) > 1e-12) throw ps::FixtureError("eps != 1e-5");
      B = rx.B; n = rx.n;
      x = std::move(rx.data); w = std::move(tw.data); out_fx = std::move(ro.data);
      label = ps_elt::fixture_label(fx, B);
    } else {
      ps::Rng rng(seed);
      x.resize((size_t)B * n); w.resize(n);
      for (auto& v : x) v = 2.0f * rng.uniform();
      for (auto& v : w) v = rng.uniform();
      label = "synthetic:s" + std::to_string(seed);
    }
  } catch (const std::exception& e) {
    std::printf("fixture error: %s\nFAILED!\n", e.what());
    return 1;
  }
  if (B == 0 || n == 0) { std::printf("empty problem\nFAILED!\n"); return 1; }
  std::printf("ps_rmsnorm: %s B=%u n=%u inplace=%d tol=%.1e\n", label.c_str(), B, n, (int)inplace, tol);

  // float64 reference of run.c's formula from the same float32 inputs (eps = the float 1e-5f, as on the device)
  std::vector<double> ref((size_t)B * n);
  for (uint32_t b = 0; b < B; ++b) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; ++i) ss += (double)x[(size_t)b * n + i] * (double)x[(size_t)b * n + i];
    ss = 1.0 / std::sqrt(ss / n + (double)eps);
    for (uint32_t i = 0; i < n; ++i) ref[(size_t)b * n + i] = (double)w[i] * (ss * (double)x[(size_t)b * n + i]);
  }
  std::vector<double> scale = ps_elt::elem_scale(ref);
  const double u = std::ldexp(1.0, -24);
  const double bound_fx = (n / 2.0 + 5.0) * u;  // NumPy float32 fixture vs float64 (depth <= n)
  bool ok = true;
  std::vector<double> ref_fx;
  if (!out_fx.empty()) {
    ok = ps_elt::check("fixture_vs_f64", out_fx.data(), ref, scale, tol_forced ? tol : std::fmax(tol, bound_fx)) && ok;
    ref_fx = ps_elt::to_f64(out_fx);
  }

  ps_elt::Dev dev;
  if (dev.open(kernel_file, "rmsnorm")) return 1;
  const uint64_t xb = (uint64_t)B * n * sizeof(float);
  const std::vector<float> poison((size_t)B * n, std::nanf(""));
  ps_elt::Buf bx, bw, bo;
  if (dev.alloc(bx, xb, VX_MEM_READ_WRITE, x.data())) return 1;
  if (dev.alloc(bw, (uint64_t)n * sizeof(float), VX_MEM_READ, w.data())) return 1;
  if (!inplace && dev.alloc(bo, xb, VX_MEM_READ_WRITE, poison.data())) return 1;
  ps_rmsnorm_arg_t arg = {};
  arg.x_addr = bx.addr; arg.w_addr = bw.addr; arg.out_addr = inplace ? bx.addr : bo.addr;
  arg.B = B; arg.n = n; arg.eps = eps;

  ps::Timer timer;
  vx_event_h ev = nullptr;
  if (dev.launch(&arg, sizeof(arg), (uint32_t)(B * dev.nt), grid_override, &ev)) return 1;  // one warp per row
  std::vector<float> y((size_t)B * n);
  if (dev.read(y.data(), inplace ? bx : bo, ev)) return 1;
  vx_event_release(ev);
  // device summation depth m = ceil(n/NT) + log2 NT (kernel.cpp)
  uint32_t lg = 0;
  while ((1ull << lg) < dev.nt) ++lg;
  const double bound_dev = ((std::ceil((double)n / (double)dev.nt) + lg) / 2.0 + 5.0) * u;
  std::printf("tolerances: dev_vs_f64 max(%.1e, %.3e), fixture_vs_f64 max(%.1e, %.3e), dev_vs_fixture max(%.1e, %.3e)%s\n",
              tol, bound_dev, tol, bound_fx, tol, bound_dev + bound_fx, tol_forced ? " -- overridden by -t" : "");
  ok = ps_elt::check("dev_vs_f64", y.data(), ref, scale, tol_forced ? tol : std::fmax(tol, bound_dev)) && ok;
  if (!ref_fx.empty())
    ok = ps_elt::check("dev_vs_fixture", y.data(), ref_fx, scale,
                       tol_forced ? tol : std::fmax(tol, bound_dev + bound_fx)) && ok;
  return ps_elt::finish(ok, timer);
}
