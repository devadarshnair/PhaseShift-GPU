// main.cpp -- ps_rope_diag host (diagnosis of the ps_rope seed-1 failure). A copy of kernels/rope/main.cpp
// (same inputs, same float64 reference, same checks, same launch) with three diagnostic additions; the device kernel is
// ps_rope's rope_kv, unchanged (kernel.cpp includes it). kernels/rope/ is not modified, so ps_rope's vxrun app hash and
// every stored ps_rope run key are unchanged.
//   -o <file>  write the raw result: "PSRD" u32 B,dim,kv_dim,hs, then pos[B] (i32), q_in[B*dim], q_dev[B*dim],
//              k_in[B*kv_dim], kslot_dev[B*kv_dim] (f32, little endian). vxrun treats -o values as outputs.
//   -r <row>   generate the inputs exactly as without -r (same seed, same B), then launch ONLY row <row> (B = 1, that
//              row's q/k/v and position). Same data and angle, one row instead of B: separates indexing from libm.
//   -P <list>  comma-separated positions replacing the generated ones (B entries); for angle-targeted reruns.
//   every q/k element outside tol is listed on a PSERR line (at most 64), with (row, pair i, pos, head_dim), the device
//   value, the float64 reference and the pair norm.
// ---- original ps_rope header follows ----
// main.cpp -- ps_rope host app: batched RoPE fused with the KV-cache write, vs a ps-fixture-v1
// `rope_kv` fixture (model/make_fixtures.py `batch`) or a synthetic case.
//
// Usage: ps_rope [-f <fixture dir> | -b B -d dim -m kv_dim -H head_size -S seq_len -N n_layers -l layer [-s seed]]
//                [-c b_cap] [-i] [-t tol] [-g grid] [-k kernel.vxbin]
//   -f  fixture with q_in (B, dim), k_in, v (B, kv_dim), q_out, k_out; params positions[B], n_layers, seq_len, ...
//   synthetic: q, k, v uniform in [-1, 1); positions: row 0 at seq_len - 1 (largest angle), others pseudo-random
//   -i  rotate q in place (q_out = q buffer, as run.c); default: separate q_out buffer poisoned with NaN
//   -g  override the block count (grid-stride path)
//   -c  batch capacity b_cap of the cache layout (default B; must be >= B). b_cap > B is the driver's case (cache
//       allocated for more sequences than one launch writes): row b's slot is still ((layer*b_cap + b)*seq_len +
//       pos)*kv_dim, rows B..b_cap-1 of every layer must stay untouched (checked by kv_untouched). Added (
//       this capacity-stride path had no test).
// The device KV caches (K and V, n_layers x b_cap x seq_len x kv_dim) are pre-filled with a sentinel bit
// pattern; after the launch the whole caches are read back and
//   q_vs_f64 / q_vs_fixture        rotated q
//   kcache_vs_f64 / kcache_vs_fixture   row b's K slot at (layer, b, pos_b) vs the rotated k
//   vcache_exact                   row b's V slot equals v bit for bit (a copy; tol 0)
//   kv_untouched                   every other cache element still holds the sentinel (no stray or misplaced write)
// plus fixture_q_vs_f64 / fixture_k_vs_f64 (sanity of the fixture). All must pass.
//
// Metric (modeling choice): RoPE rotates each pair (v0, v1) by angle val, so an angle error d moves both outputs by at
// most |d| * r with r = hypot(v0, v1), the pair's norm; plain |ref_i| is not a valid scale because a component of a
// rotated pair can be ~0 (cancellation in v0*cos - v1*sin). So rel_i = |y_i - ref_i| / max(r_pair(i), FLT_MIN).
// Tolerance 1e-4 (sin/cos come from different libms on host and device). Derivation, u = 2^-24, against the
// float64 reference (same formula evaluated in double from the same float32 inputs):
//   * the exponent head_dim/(float)head_size is rounded (abs err <= e u/2), amplified by ln(theta) = 9.21 in
//     theta^e: relative freq error <= 9.21 e u/2; with the 1/x rounding (u/2) and the device libm's powf error
//     (<= 1 ulp used below; MEASURED on SimX: musl powf is correctly rounded, max 0.481 ulp over
//     all 24 exponents hd/(float)hs the kernels use) the float32 freq has
//     relative error <= (4.6 e + 1.5) u;
//   * val = pos * freq: abs error <= pos * freq * (4.6 e + 1.5) u + ulp(val)/2. Largest terms: stories15M (head_size 48,
//     pos <= 255): head_dim 0 gives freq = 1 exactly and val = pos exactly (0 error); head_dim 2 gives freq =
//     10000^(-1/24) = 0.681, val <= 173.7, error <= 173.7 * 1.7u + 2^-17 = 1.8e-5 + 7.6e-6 = 2.6e-5; max over e of
//     pos * 10000^-e * 4.6 e u is at e = 1/ln(10000) and adds <= 2.8e-6. stories260K (head_size 8, pos <= 511):
//     freq in {1, 0.1, 0.01, 0.001}, error <= 51.1 * 1.5u + 2^-19 = 6.5e-6;
//   * cosf/sinf (<= 1 ulp each used below; MEASURED: max 0.500 / 0.500 ulp, i.e. correctly rounded, on every float32
//     angle the fixtures produce and on 2,048 points of [0, 600)) and the rotation (2 products +
//     1 add/sub, an FMA possibly) add <= ~4u = 2.4e-7 of r.
//   => device vs float64 <= ~2.7e-5 (15M) / ~7e-6 (260K) of r, 3.7x / 14x below 1e-4. Device vs the NumPy float32
//   fixture: both compute the identical float32 exponent; only powf/cos/sin implementations differ (<= ~2 ulp of freq
//   -> <= 173.7 * 2u * 2 = 4.1e-5 worst case), also below 1e-4. Margin: the 15M bound reaches 1e-4 only if powf is
//   off by ~8 ulp (173.7 * (e_pow + 0.7) u + 1.0e-5 = 1e-4 at e_pow = 7.9); the device libm (musl in libc32, soft-float
//   double evaluation, methodology §14.0) measured <= 0.5 ulp, NumPy float32 <= 2.25 ulp on the same inputs
//   (independent check), so the derivation's 1-ulp inputs hold with margin. Synthetic shapes must keep pos * freq_max < ~1000.
#include <cstdint>
#include "common.h"
#include "elt_host.h"

static void usage(const char* app) {
  std::printf("Usage: %s [-o dump] [-r row] [-P p0,p1,..] [-f fixture_dir | -b B -d dim -m kv_dim -H head_size -S seq_len -N n_layers -l layer "
              "[-s seed]] [-c b_cap] [-i] [-t tol] [-g grid] [-k kernel.vxbin] [-W l1:l2]\n", app);
}

static uint32_t ps_bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }

static const uint32_t SENTINEL = 0xDEADBEEFu;  // a finite float (-6.3e18) no correct kernel writes by accident

int main(int argc, char** argv) {
  std::string fixture, kernel_file = "kernel.vxbin";
  uint32_t B = 4, dim = 64, kv_dim = 32, hs = 8, seq_len = 512, n_layers = 5, layer = 4, seed = 1, grid_override = 0,
           cap_opt = 0;
  bool inplace = false;
  std::string dump_file, pos_list;
  int row_sel = -1;
  double tol = 1e-4;
  float theta = 10000.0f;  // run.c: powf(10000.0f, ...)
  int c;
  while ((c = getopt(argc, argv, "f:b:d:m:H:S:N:l:s:c:it:g:k:W:o:r:P:h")) != -1) {
    switch (c) {
      case 'f': fixture = optarg; break;
      case 'b': B = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'd': dim = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'm': kv_dim = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'H': hs = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'S': seq_len = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'N': n_layers = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'l': layer = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'c': cap_opt = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'i': inplace = true; break;
      case 't': tol = std::strtod(optarg, nullptr); break;
      case 'g': grid_override = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      case 'W':   // PhaseShift: [dcr_phase] way masks "<l1word>:<l2word>" (elt_host.h, methodology §14.A)
        if (!ps_elt::waymask_opt(optarg)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        break;
      case 'o': dump_file = optarg; break;
      case 'r': row_sel = (int)std::strtol(optarg, nullptr, 10); break;
      case 'P': pos_list = optarg; break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  std::vector<float> q, k, v, qo_fx, ko_fx;
  std::vector<int32_t> pos;
  std::string label;
  try {
    if (!fixture.empty()) {
      ps::Fixture fx(fixture);
      if (fx.op() != "rope_kv") throw ps::FixtureError("op " + fx.op() + " is not rope_kv");
      ps_elt::Rows rq = ps_elt::load_rows(fx, "q_in"), rk = ps_elt::load_rows(fx, "k_in"),
                   rv = ps_elt::load_rows(fx, "v"), rqo = ps_elt::load_rows(fx, "q_out"),
                   rko = ps_elt::load_rows(fx, "k_out");
      ps_elt::expect_same_shape(rq, rqo, "q_in vs q_out");
      ps_elt::expect_same_shape(rk, rko, "k_in vs k_out");
      ps_elt::expect_same_shape(rk, rv, "k_in vs v");
      if (rq.B != rk.B) throw ps::FixtureError("q and k batch differ");
      B = rq.B; dim = rq.n; kv_dim = rk.n;
      hs = ps_elt::param_u32(fx, "head_size");
      seq_len = ps_elt::param_u32(fx, "seq_len");
      n_layers = ps_elt::param_u32(fx, "n_layers");
      if (ps_elt::param_u32(fx, "dim") != dim || ps_elt::param_u32(fx, "kv_dim") != kv_dim)
        throw ps::FixtureError("dim/kv_dim params disagree with tensor shapes");
      if (ps_elt::param_num(fx, "rope_theta") != 10000.0) throw ps::FixtureError("rope_theta != 10000");
      int l = ps_elt::layer_of(fx);
      if (l < 0) throw ps::FixtureError("rope_kv fixture needs a layer");
      layer = (uint32_t)l;
      const ps::Json& jp = fx.meta().at("params").at("positions");
      if (jp.kind != ps::Json::Array || jp.arr.size() != B) throw ps::FixtureError("positions must have B entries");
      for (auto& e : jp.arr) {
        double p = e.as_num();
        if (!(p >= 0.0 && p < (double)seq_len) || p != (double)(int32_t)p) throw ps::FixtureError("bad position");
        pos.push_back((int32_t)p);
      }
      q = std::move(rq.data); k = std::move(rk.data); v = std::move(rv.data);
      qo_fx = std::move(rqo.data); ko_fx = std::move(rko.data);
      label = ps_elt::fixture_label(fx, B);
    } else {
      ps::Rng rng(seed);
      q.resize((size_t)B * dim); k.resize((size_t)B * kv_dim); v.resize((size_t)B * kv_dim);
      for (auto& x : q) x = rng.uniform();
      for (auto& x : k) x = rng.uniform();
      for (auto& x : v) x = rng.uniform();
      for (uint32_t b = 0; b < B; ++b) pos.push_back(b == 0 ? (int32_t)(seq_len - 1) : (int32_t)(rng.next() % seq_len));
      label = "diag:synthetic:s" + std::to_string(seed);
    }
  } catch (const std::exception& e) {
    std::printf("fixture error: %s\nFAILED!\n", e.what());
    return 1;
  }
  if (!pos_list.empty()) {   // diag: explicit positions
    std::vector<int32_t> pp;
    const char* c0 = pos_list.c_str();
    while (*c0) { char* e; pp.push_back((int32_t)std::strtol(c0, &e, 10)); c0 = (*e == ',') ? e + 1 : e; if (e == c0 && *e) break; }
    if (pp.size() != B) { std::printf("bad -P: %zu positions for B=%u\nFAILED!\n", pp.size(), B); return 1; }
    pos = pp;
    label += ":P";
  }
  if (row_sel >= 0) {        // diag: one row of the generated inputs, launched as B = 1
    if ((uint32_t)row_sel >= B) { std::printf("bad -r %d (B=%u)\nFAILED!\n", row_sel, B); return 1; }
    const size_t r = (size_t)row_sel;
    q.assign(q.begin() + r * dim, q.begin() + (r + 1) * dim);
    k.assign(k.begin() + r * kv_dim, k.begin() + (r + 1) * kv_dim);
    v.assign(v.begin() + r * kv_dim, v.begin() + (r + 1) * kv_dim);
    if (!qo_fx.empty()) {
      qo_fx.assign(qo_fx.begin() + r * dim, qo_fx.begin() + (r + 1) * dim);
      ko_fx.assign(ko_fx.begin() + r * kv_dim, ko_fx.begin() + (r + 1) * kv_dim);
    }
    pos = {pos[r]};
    B = 1;
    label += ":row" + std::to_string(row_sel);
  }
  if (B == 0 || dim == 0 || (dim & 1) || (kv_dim & 1) || kv_dim > dim || hs == 0 || (hs & 1) || dim % hs ||
      kv_dim % hs || layer >= n_layers || seq_len == 0) {
    std::printf("bad shape: B=%u dim=%u kv_dim=%u head_size=%u layer=%u n_layers=%u\nFAILED!\n", B, dim, kv_dim, hs,
                layer, n_layers);
    return 1;
  }
  const uint32_t b_cap = cap_opt ? cap_opt : B;
  if (b_cap < B) {
    std::printf("bad -c: b_cap=%u < B=%u\nFAILED!\n", b_cap, B);
    return 1;
  }
  std::printf("ps_rope: %s B=%u dim=%u kv_dim=%u head_size=%u seq_len=%u layer=%u/%u b_cap=%u inplace=%d tol=%.1e pos=",
              label.c_str(), B, dim, kv_dim, hs, seq_len, layer, n_layers, b_cap, (int)inplace, tol);
  for (uint32_t b = 0; b < B; ++b) std::printf("%s%d", b ? "," : "", pos[b]);
  std::printf("\n");

  // float64 reference (run.c's formula in double from the same float32 inputs) and the pair-norm scale
  std::vector<double> q_ref((size_t)B * dim), q_sc((size_t)B * dim), k_ref((size_t)B * kv_dim), k_sc((size_t)B * kv_dim);
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t i = 0; i < dim; i += 2) {
      double freq = 1.0 / std::pow((double)theta, (double)(i % hs) / (double)hs);
      double val = (double)pos[b] * freq, cr = std::cos(val), ci = std::sin(val);
      auto rot = [&](const float* src, std::vector<double>& ref, std::vector<double>& sc, size_t o) {
        double a = src[o + i], bb = src[o + i + 1];
        ref[o + i] = a * cr - bb * ci;
        ref[o + i + 1] = a * ci + bb * cr;
        sc[o + i] = sc[o + i + 1] = std::fmax(std::hypot(a, bb), (double)FLT_MIN);
      };
      rot(q.data(), q_ref, q_sc, (size_t)b * dim);
      if (i < kv_dim) rot(k.data(), k_ref, k_sc, (size_t)b * kv_dim);
    }
  }
  bool ok = true;
  if (!qo_fx.empty()) {
    ok = ps_elt::check("fixture_q_vs_f64", qo_fx.data(), q_ref, q_sc, tol) && ok;
    ok = ps_elt::check("fixture_k_vs_f64", ko_fx.data(), k_ref, k_sc, tol) && ok;
  }

  ps_elt::Dev dev;
  if (dev.open(kernel_file, "rope_kv")) return 1;
  const size_t cache_n = (size_t)n_layers * b_cap * seq_len * kv_dim;
  std::vector<float> sentinel(cache_n);
  { float s; std::memcpy(&s, &SENTINEL, 4); std::fill(sentinel.begin(), sentinel.end(), s); }
  const std::vector<float> poison((size_t)B * dim, std::nanf(""));
  ps_elt::Buf bq, bk, bv, bp, bqo, bkc, bvc;
  if (dev.alloc(bq, (uint64_t)B * dim * 4, VX_MEM_READ_WRITE, q.data())) return 1;
  if (dev.alloc(bk, (uint64_t)B * kv_dim * 4, VX_MEM_READ, k.data())) return 1;
  if (dev.alloc(bv, (uint64_t)B * kv_dim * 4, VX_MEM_READ, v.data())) return 1;
  if (dev.alloc(bp, (uint64_t)B * 4, VX_MEM_READ, pos.data())) return 1;
  if (!inplace && dev.alloc(bqo, (uint64_t)B * dim * 4, VX_MEM_READ_WRITE, poison.data())) return 1;
  if (dev.alloc(bkc, (uint64_t)cache_n * 4, VX_MEM_READ_WRITE, sentinel.data())) return 1;
  if (dev.alloc(bvc, (uint64_t)cache_n * 4, VX_MEM_READ_WRITE, sentinel.data())) return 1;
  ps_rope_arg_t arg = {};
  arg.q_addr = bq.addr; arg.k_addr = bk.addr; arg.v_addr = bv.addr; arg.pos_addr = bp.addr;
  arg.q_out_addr = inplace ? bq.addr : bqo.addr; arg.kc_addr = bkc.addr; arg.vc_addr = bvc.addr;
  arg.B = B; arg.b_cap = b_cap; arg.dim = dim; arg.kv_dim = kv_dim; arg.head_size = hs; arg.seq_len = seq_len;
  arg.layer = layer; arg.theta = theta;
  std::printf("kv cache: 2 x %zu floats (%.2f MiB each), layout [layer][b][pos][kv_head][head_dim]\n", cache_n,
              cache_n * 4.0 / 1048576.0);

  ps::Timer timer;
  vx_event_h ev = nullptr;
  if (dev.launch(&arg, sizeof(arg), B * (dim / 2), grid_override, &ev)) return 1;  // one thread per pair
  std::vector<float> qy((size_t)B * dim), kc(cache_n), vc(cache_n);
  if (dev.read(qy.data(), inplace ? bq : bqo, ev)) return 1;
  if (dev.read(kc.data(), bkc, ev)) return 1;
  if (dev.read(vc.data(), bvc, ev)) return 1;
  vx_event_release(ev);

  ok = ps_elt::check("q_vs_f64", qy.data(), q_ref, q_sc, tol) && ok;
  // diag: list every out-of-tolerance q element
  auto list_errs = [&](const char* what, const float* y, const std::vector<double>& ref, const std::vector<double>& sc,
                       const float* in, uint32_t n, int& budget) {
    for (uint32_t b = 0; b < B; ++b)
      for (uint32_t i = 0; i < n; ++i) {
        size_t j = (size_t)b * n + i;
        double rel = std::fabs((double)y[j] - ref[j]) / sc[j];
        if (!(rel <= tol) && budget-- > 0) {
          uint32_t i0 = i & ~1u;
          std::printf("PSERR: %s idx=%zu row=%u i=%u pair=%u pos=%d head=%u head_dim=%u in0=%.9g in1=%.9g dev=%.9g "
                      "dev_bits=%08x ref=%.12g norm=%.9g rel=%.6e\n", what, j, b, i, i0, pos[b], i / hs, i0 % hs,
                      in[(size_t)b * n + i0], in[(size_t)b * n + i0 + 1], y[j], ps_bits(y[j]), ref[j], sc[j], rel);
        }
      }
  };
  int qbudget = 64;
  list_errs("q", qy.data(), q_ref, q_sc, q.data(), dim, qbudget);
  if (!qo_fx.empty()) ok = ps_elt::check("q_vs_fixture", qy.data(), ps_elt::to_f64(qo_fx), q_sc, tol) && ok;
  // gather the slots, then mark them so the untouched check can skip them
  std::vector<float> kslot((size_t)B * kv_dim), vslot((size_t)B * kv_dim);
  std::vector<uint8_t> in_slot(cache_n, 0);
  for (uint32_t b = 0; b < B; ++b) {
    size_t slot = ((size_t)(layer * b_cap + b) * seq_len + (uint32_t)pos[b]) * kv_dim;
    for (uint32_t i = 0; i < kv_dim; ++i) {
      kslot[(size_t)b * kv_dim + i] = kc[slot + i];
      vslot[(size_t)b * kv_dim + i] = vc[slot + i];
      in_slot[slot + i] = 1;
    }
  }
  ok = ps_elt::check("kcache_vs_f64", kslot.data(), k_ref, k_sc, tol) && ok;
  int kbudget = 64;
  list_errs("k", kslot.data(), k_ref, k_sc, k.data(), kv_dim, kbudget);
  if (!dump_file.empty()) {  // diag: raw dump
    FILE* f = std::fopen(dump_file.c_str(), "wb");
    if (!f) { std::printf("cannot write %s\nFAILED!\n", dump_file.c_str()); return 1; }
    const uint32_t h[5] = {0x44525350u, B, dim, kv_dim, hs};  // "PSRD"
    std::fwrite(h, 4, 5, f);
    std::fwrite(pos.data(), 4, B, f);
    std::fwrite(q.data(), 4, (size_t)B * dim, f);
    std::fwrite(qy.data(), 4, (size_t)B * dim, f);
    std::fwrite(k.data(), 4, (size_t)B * kv_dim, f);
    std::fwrite(kslot.data(), 4, (size_t)B * kv_dim, f);
    std::fclose(f);
    std::printf("dump: %s\n", dump_file.c_str());
  }
  if (!ko_fx.empty()) ok = ps_elt::check("kcache_vs_fixture", kslot.data(), ps_elt::to_f64(ko_fx), k_sc, tol) && ok;
  ok = ps_elt::check("vcache_exact", vslot.data(), ps_elt::to_f64(v), ps_elt::elem_scale(ps_elt::to_f64(v)), 0.0) && ok;
  ps::ErrStats un;
  un.n = 2 * cache_n;
  for (size_t j = 0; j < cache_n; ++j) {
    if (in_slot[j]) continue;
    uint32_t a, bb;
    std::memcpy(&a, &kc[j], 4);
    std::memcpy(&bb, &vc[j], 4);
    if (a != SENTINEL) { ++un.errors; un.worst = j; }
    if (bb != SENTINEL) { ++un.errors; un.worst = j; }
  }
  if (un.errors) un.max_rel = INFINITY;
  ps::print_check("kv_untouched", un);
  ok = ok && un.ok();
  return ps_elt::finish(ok, timer);
}
