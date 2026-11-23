// main.cpp -- ps_residual host app: batched residual add on Vortex vs a ps-fixture-v1
// `residual_att` / `residual_ffn` fixture or a synthetic case.
//
// Usage: ps_residual [-f <fixture dir> | -b B -n dim [-s seed]] [-i] [-t tol] [-g grid] [-k kernel.vxbin]
//   -f  fixture with x, branch, out of shape (B, dim) or (dim,)
//   synthetic: x uniform in [-1, 1); branch = -x on every 7th element (exact cancellation -> out must be exactly 0),
//       uniform in [-1, 1) elsewhere
//   -i  in place (out = x buffer, as run.c's x[i] += xb[i]); default: separate out buffer poisoned with NaN
//   -g  override the block count (grid-stride path)
// Checks: dev_vs_f64, dev_vs_fixture, fixture_vs_f64, and dev_bits_vs_fixture (the number of elements whose bits
// differ from the fixture's out; must be 0).
//
// Tolerance 1e-5, metric rel_i = |y_i - ref_i| / max(|ref_i|, FLT_MIN) (elt_host.h). One correctly
// rounded float32 add: |fl(x + b) - (x + b)| <= (u/2) |x + b| (exact when x + b underflows or cancels within a factor
// of 2, Sterbenz), and the float64 sum of two float32 values is exact unless their exponents differ by > 29, so
// device vs float64 <= u/2 = 3e-8. Device vs fixture (NumPy does the same single IEEE add) must be bit-identical,
// which the extra bit check enforces; 1e-5 is the plan's tolerance for the float64 comparison.
#include "common.h"
#include "elt_host.h"

static void usage(const char* app) {
  std::printf("Usage: %s [-f fixture_dir | -b B -n dim [-s seed]] [-i] [-t tol] [-g grid] [-k kernel.vxbin] [-W l1:l2]\n", app);
}

int main(int argc, char** argv) {
  std::string fixture, kernel_file = "kernel.vxbin";
  uint32_t B = 4, n = 288, seed = 1, grid_override = 0;
  bool inplace = false;
  double tol = 1e-5;
  int c;
  while ((c = getopt(argc, argv, "f:b:n:s:it:g:k:W:h")) != -1) {
    switch (c) {
      case 'f': fixture = optarg; break;
      case 'b': B = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': n = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'i': inplace = true; break;
      case 't': tol = std::strtod(optarg, nullptr); break;
      case 'g': grid_override = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      case 'W':   // PhaseShift: [dcr_phase] way masks "<l1word>:<l2word>" (elt_host.h, methodology §14.A)
        if (!ps_elt::waymask_opt(optarg)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  std::vector<float> x, br, out_fx;
  std::string label;
  try {
    if (!fixture.empty()) {
      ps::Fixture fx(fixture);
      if (fx.op().rfind("residual", 0) != 0) throw ps::FixtureError("op " + fx.op() + " is not residual_*");
      ps_elt::Rows a = ps_elt::load_rows(fx, "x"), b = ps_elt::load_rows(fx, "branch"), o = ps_elt::load_rows(fx, "out");
      ps_elt::expect_same_shape(a, b, "x vs branch");
      ps_elt::expect_same_shape(a, o, "x vs out");
      B = a.B; n = a.n;
      x = std::move(a.data); br = std::move(b.data); out_fx = std::move(o.data);
      label = ps_elt::fixture_label(fx, B);
    } else {
      ps::Rng rng(seed);
      x.resize((size_t)B * n); br.resize((size_t)B * n);
      for (auto& v : x) v = rng.uniform();
      for (size_t j = 0; j < br.size(); ++j) br[j] = (j % 7 == 3) ? -x[j] : rng.uniform();
      label = "synthetic:s" + std::to_string(seed);
    }
  } catch (const std::exception& e) {
    std::printf("fixture error: %s\nFAILED!\n", e.what());
    return 1;
  }
  const size_t N = (size_t)B * n;
  if (N == 0) { std::printf("empty problem\nFAILED!\n"); return 1; }
  std::printf("ps_residual: %s B=%u dim=%u inplace=%d tol=%.1e\n", label.c_str(), B, n, (int)inplace, tol);

  std::vector<double> ref(N);
  for (size_t j = 0; j < N; ++j) ref[j] = (double)x[j] + (double)br[j];
  std::vector<double> scale = ps_elt::elem_scale(ref);
  bool ok = true;
  if (!out_fx.empty()) ok = ps_elt::check("fixture_vs_f64", out_fx.data(), ref, scale, tol) && ok;

  ps_elt::Dev dev;
  if (dev.open(kernel_file, "residual")) return 1;
  const std::vector<float> poison(N, std::nanf(""));
  ps_elt::Buf bx, bb, bo;
  if (dev.alloc(bx, N * 4, VX_MEM_READ_WRITE, x.data())) return 1;
  if (dev.alloc(bb, N * 4, VX_MEM_READ, br.data())) return 1;
  if (!inplace && dev.alloc(bo, N * 4, VX_MEM_READ_WRITE, poison.data())) return 1;
  ps_residual_arg_t arg = {};
  arg.x_addr = bx.addr; arg.branch_addr = bb.addr; arg.out_addr = inplace ? bx.addr : bo.addr; arg.n = (uint32_t)N;

  ps::Timer timer;
  vx_event_h ev = nullptr;
  if (dev.launch(&arg, sizeof(arg), (uint32_t)N, grid_override, &ev)) return 1;
  std::vector<float> y(N);
  if (dev.read(y.data(), inplace ? bx : bo, ev)) return 1;
  vx_event_release(ev);
  ok = ps_elt::check("dev_vs_f64", y.data(), ref, scale, tol) && ok;
  if (!out_fx.empty()) {
    ok = ps_elt::check("dev_vs_fixture", y.data(), ps_elt::to_f64(out_fx), scale, tol) && ok;
    ps::ErrStats bits;
    bits.n = N;
    for (size_t j = 0; j < N; ++j)
      if (std::memcmp(&y[j], &out_fx[j], 4) != 0) { ++bits.errors; bits.worst = j; }
    if (bits.errors) bits.max_rel = INFINITY;
    ps::print_check("dev_bits_vs_fixture", bits);
    ok = ok && bits.ok();
  }
  return ps_elt::finish(ok, timer);
}
