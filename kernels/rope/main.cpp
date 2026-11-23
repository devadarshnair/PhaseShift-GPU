// main.cpp -- ps_rope host app: batched RoPE fused with the KV-cache write, vs a ps-fixture-v1
// `rope_kv` fixture (model/make_fixtures.py `batch`) or a synthetic case.
//
// Usage: ps_rope [-f <fixture dir> | -b B -d dim -m kv_dim -H head_size -S seq_len -N n_layers -l layer [-s seed]]
//                [-c b_cap] [-i] [-t tol] [-g grid] [-k kernel.vxbin] [-R perlaunch|table|setup] [-A] [-Q pos0] [-Z npos]
//   -f  fixture with q_in (B, dim), k_in, v (B, kv_dim), q_out, k_out; params positions[B], n_layers, seq_len, ...
//   synthetic: q, k, v uniform in [-1, 1); positions: row 0 at seq_len - 1 (largest angle), others pseudo-random
//   -i  rotate q in place (q_out = q buffer, as run.c); default: separate q_out buffer poisoned with NaN
//   -g  override the block count (grid-stride path)
//   -R  (methodology §18.7) perlaunch (default: rope_kv, behaviour and output unchanged) | table: launch 1 = rope_table (the
//       one-time angle-table setup), launch 2 = rope_kv_tab (angles loaded from the table); every check below applies unchanged
//       to the rope_kv_tab outputs, plus the table is read back and compared ENTRY BY ENTRY with a correctly-rounded host
//       recomputation (kernels/rope/rope_table_ref.h: PSCHECK table_rope_table_vs_host, PSTABLE). PERF is cumulative over both
//       launches (setup + kernel). | setup: only launch 1 and the table check. The table covers positions 0..seq_len-1.
//       Bit-identity of rope_kv_tab to rope_kv is shown by running both modes on the same inputs and comparing the PSDIGEST
//       lines (q_vs_f64, kcache_vs_f64, vcache_exact, kcache_full, vcache_full): tests/test_ep_elementwise.py.
//   -Q pos0, -Z npos  the table covers positions pos0 .. pos0+npos-1 (default 0 .. seq_len-1) in -R setup and -A (probe positions pos0+b);
//       with -R table only -Z (a fixture needs the table only up to its largest position: -Z max_pos+1 saves simulation time; positions
//       beyond it are never read), pos0 must be 0.
//   -K kv8  (methodology §18.9) with -R table: launch 2 is `rope_kv_tab_q8` instead of rope_kv_tab: q is rotated as before, K (rotated) and V
//       are written as INT8 with one fp32 scale per (position, KV head) (common_kv8.h layout). Checks: q as usual; K8 / V8 bytes and K / V scales
//       BIT-IDENTICAL to a host recomputation from the device's own table (non-fused float32 rotation, runq.c's quantize with group = head size:
//       PSCHECK k8_exact / v8_exact / ks_exact / vs_exact, 0 errors required) and every other cache byte / scale untouched (kv8_untouched).
//       -J <file> dumps the inputs and the device slots so that tests/test_ep_kv8.py can recompute them with the NumPy port (model/ref_q80.py).
//   -A  device probe of the PER-LAUNCH angles: synthetic only; dim := head_size, kv_dim := 0 (no cache), q rows = (1, 0, 1, 0, ...)
//       at positions 0..B-1, so q_out = (cos, sin) of rope_kv's own per-token evaluation, exactly (1*c - 0*s = c, 1*s + 0*c = s).
//       Same host check (PSCHECK probe_rope_table_vs_host) and a PSDIGEST that must equal the table's for the same (head_size, positions).
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
#include "rope_table_ref.h"
#include "common_kv8.h"
#include "ps_q80.h"

static void usage(const char* app) {
  std::printf("Usage: %s [-f fixture_dir | -b B -d dim -m kv_dim -H head_size -S seq_len -N n_layers -l layer "
              "[-s seed]] [-c b_cap] [-i] [-t tol] [-g grid] [-k kernel.vxbin] [-W l1:l2] [-R perlaunch|table|setup] [-A] [-Q pos0] [-Z npos] [-K kv8] [-J dumpfile]\n", app);
}

static const uint32_t SENTINEL = 0xDEADBEEFu;  // a finite float (-6.3e18) no correct kernel writes by accident

int main(int argc, char** argv) {
  std::string fixture, kernel_file = "kernel.vxbin";
  uint32_t B = 4, dim = 64, kv_dim = 32, hs = 8, seq_len = 512, n_layers = 5, layer = 4, seed = 1, grid_override = 0,
           cap_opt = 0;
  bool inplace = false, angle_probe = false;
  std::string rmode = "perlaunch", kmode = "fp32", dump_path;
  uint32_t tab_pos0 = 0, tab_npos = 0;   // -Q / -Z: window of the angle table / probe (default: all of 0..seq_len-1)
  double tol = 1e-4;
  float theta = 10000.0f;  // run.c: powf(10000.0f, ...)
  int c;
  while ((c = getopt(argc, argv, "f:b:d:m:H:S:N:l:s:c:it:g:k:W:R:AQ:Z:K:J:h")) != -1) {
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
      case 'R': rmode = optarg; break;
      case 'A': angle_probe = true; break;
      case 'K': kmode = optarg; break;
      case 'J': dump_path = optarg; break;
      case 'Q': tab_pos0 = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'Z': tab_npos = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
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
      label = "synthetic:s" + std::to_string(seed);
    }
  } catch (const std::exception& e) {
    std::printf("fixture error: %s\nFAILED!\n", e.what());
    return 1;
  }
  if (rmode != "perlaunch" && rmode != "table" && rmode != "setup") {
    std::printf("bad -R %s (perlaunch|table|setup)\nFAILED!\n", rmode.c_str());
    return 1;
  }
  if (kmode != "fp32" && kmode != "kv8") { std::printf("bad -K %s (kv8)\nFAILED!\n", kmode.c_str()); return 1; }
  const bool kv8 = kmode == "kv8";
  if (kv8 && (rmode != "table" || angle_probe)) { std::printf("-K kv8 needs -R table and no -A\nFAILED!\n"); return 1; }
  if (angle_probe) {
    if (!fixture.empty() || rmode != "perlaunch") {
      std::printf("-A is a synthetic per-launch probe (no -f, no -R)\nFAILED!\n");
      return 1;
    }
    dim = hs; kv_dim = 0;   // q row = one head; no KV write (so the cache need not hold B x seq_len slots)
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
  if (angle_probe) {   // rope_kv's own per-launch angles at positions 0..B-1 (see the header comment of this file)
    if (tab_pos0 + B > seq_len) {
      std::printf("-A needs pos0 + B <= seq_len\nFAILED!\n");
      return 1;
    }
    std::printf("ps_rope: probe B=%u head_size=%u (q = (1,0) pairs, positions %u..%u, kv_dim=0)\n", B, hs, tab_pos0, tab_pos0 + B - 1);
    std::vector<float> qu((size_t)B * hs), ku(4, 0.0f);
    for (size_t t = 0; t < qu.size(); ++t) qu[t] = (t & 1) ? 0.0f : 1.0f;
    std::vector<int32_t> pp(B);
    for (uint32_t b = 0; b < B; ++b) pp[b] = (int32_t)(tab_pos0 + b);
    const std::vector<float> pois((size_t)B * hs, std::nanf(""));
    ps_elt::Dev dev;
    if (dev.open(kernel_file, "rope_kv")) return 1;
    ps_elt::Buf bq, bk, bp, bqo;
    if (dev.alloc(bq, (uint64_t)B * hs * 4, VX_MEM_READ, qu.data())) return 1;
    if (dev.alloc(bk, 16, VX_MEM_READ_WRITE, ku.data())) return 1;
    if (dev.alloc(bp, (uint64_t)B * 4, VX_MEM_READ, pp.data())) return 1;
    if (dev.alloc(bqo, (uint64_t)B * hs * 4, VX_MEM_READ_WRITE, pois.data())) return 1;
    ps_rope_arg_t pa = {};
    pa.q_addr = bq.addr; pa.k_addr = bk.addr; pa.v_addr = bk.addr; pa.pos_addr = bp.addr; pa.q_out_addr = bqo.addr;
    pa.kc_addr = bk.addr; pa.vc_addr = bk.addr; pa.B = B; pa.b_cap = B; pa.dim = hs; pa.kv_dim = 0; pa.head_size = hs;
    pa.seq_len = seq_len; pa.layer = 0; pa.theta = theta;
    ps::Timer tm;
    vx_event_h ev = nullptr;
    if (dev.launch(&pa, sizeof(pa), B * (hs / 2), grid_override, &ev)) return 1;
    std::vector<float> qy((size_t)B * hs);
    if (dev.read(qy.data(), bqo, ev)) return 1;
    vx_event_release(ev);
    bool pok = ps_rope::check_table("probe_", qy.data(), hs, B, theta, tab_pos0);
    return ps_elt::finish(pok, tm);
  }
  std::printf("ps_rope: %s B=%u dim=%u kv_dim=%u head_size=%u seq_len=%u layer=%u/%u b_cap=%u inplace=%d tol=%.1e rope=%s pos=",
              label.c_str(), B, dim, kv_dim, hs, seq_len, layer, n_layers, b_cap, (int)inplace, tol, rmode.c_str());
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

  const bool tabmode = rmode != "perlaunch";
  // validate the table window and the positions BEFORE any device object exists (an early return after the first enqueue_write
  // released buffers with writes still pending and crashed in the Dev destructor, losing the buffered message)
  const uint32_t npos_t = tab_npos ? tab_npos : (rmode == "setup" ? seq_len - tab_pos0 : seq_len);
  if (tabmode) {
    if (tab_pos0 + npos_t > seq_len || (rmode == "table" && tab_pos0 != 0)) {
      std::printf("bad table window: pos0=%u npos=%u seq_len=%u (-R table needs pos0 = 0)\nFAILED!\n", tab_pos0, npos_t, seq_len);
      return 1;
    }
    if (rmode == "table")
      for (uint32_t b = 0; b < B; ++b)
        if ((uint32_t)pos[b] >= npos_t) {
          std::printf("position %d is outside the table (npos=%u)\nFAILED!\n", pos[b], npos_t);
          return 1;
        }
  }
  ps_elt::Dev dev;
  if (dev.open(kernel_file, tabmode ? "rope_table" : "rope_kv")) return 1;   // (kv8: launch 2 is rope_kv_tab_q8, below)
  vx_kernel_h k_setup = tabmode ? dev.kernel : nullptr;   // rope_table; dev.kernel then becomes rope_kv_tab for launch 2
  if (tabmode && rmode == "table") PS_RT_CHECK(vx_module_get_kernel(dev.module_, kv8 ? "rope_kv_tab_q8" : "rope_kv_tab", &dev.kernel));
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
  const uint32_t Hkv8 = hs ? kv_dim / hs : 0;
  const size_t sc_n = (size_t)n_layers * b_cap * Hkv8 * seq_len;   // kv8: scales [layer][b][kv_head][pos]
  ps_elt::Buf bks, bvs;
  // the queue reads the host buffers when it executes, not at enqueue: they must outlive the writes, hence function scope
  const std::vector<uint8_t> sent8(kv8 ? cache_n : 0, 0xEF);
  const std::vector<float> sent_s(kv8 ? sc_n : 0, sentinel.empty() ? 0.0f : sentinel[0]);
  if (kv8) {   // INT8 caches (sentinel byte 0xEF) and scale arrays (sentinel float bit pattern)
    if (dev.alloc(bkc, (uint64_t)cache_n, VX_MEM_READ_WRITE, sent8.data())) return 1;
    if (dev.alloc(bvc, (uint64_t)cache_n, VX_MEM_READ_WRITE, sent8.data())) return 1;
    if (dev.alloc(bks, (uint64_t)sc_n * 4, VX_MEM_READ_WRITE, sent_s.data())) return 1;
    if (dev.alloc(bvs, (uint64_t)sc_n * 4, VX_MEM_READ_WRITE, sent_s.data())) return 1;
  } else {
    if (dev.alloc(bkc, (uint64_t)cache_n * 4, VX_MEM_READ_WRITE, sentinel.data())) return 1;
    if (dev.alloc(bvc, (uint64_t)cache_n * 4, VX_MEM_READ_WRITE, sentinel.data())) return 1;
  }
  ps_rope_arg_t arg = {};
  arg.q_addr = bq.addr; arg.k_addr = bk.addr; arg.v_addr = bv.addr; arg.pos_addr = bp.addr;
  arg.q_out_addr = inplace ? bq.addr : bqo.addr; arg.kc_addr = bkc.addr; arg.vc_addr = bvc.addr;
  arg.B = B; arg.b_cap = b_cap; arg.dim = dim; arg.kv_dim = kv_dim; arg.head_size = hs; arg.seq_len = seq_len;
  arg.layer = layer; arg.theta = theta;
  std::printf("kv cache: 2 x %zu floats (%.2f MiB each), layout [layer][b][pos][kv_head][head_dim]\n", cache_n,
              cache_n * 4.0 / 1048576.0);

  ps::Timer timer;
  vx_event_h ev = nullptr;
  ps_elt::Buf btab;
  std::vector<float> tab;
  if (tabmode) {   // launch 1 = the one-time angle table (default: seq_len positions), read back and checked entry by entry
    const size_t tab_n = (size_t)npos_t * (hs / 2) * 2;
    const std::vector<float> tpois(tab_n, std::nanf(""));
    if (dev.alloc(btab, (uint64_t)tab_n * 4, VX_MEM_READ_WRITE, tpois.data())) return 1;
    ps_rope_table_arg_t ta = {};
    ta.tab_addr = btab.addr; ta.npos = npos_t; ta.head_size = hs; ta.theta = theta; ta.pos0 = tab_pos0;
    vx_kernel_h k_main = dev.kernel;
    dev.kernel = k_setup;
    if (dev.launch(&ta, sizeof(ta), (uint32_t)(npos_t * (hs / 2)), grid_override, &ev)) return 1;
    dev.kernel = k_main ? k_main : k_setup;
    tab.resize(tab_n);
    if (dev.read(tab.data(), btab, ev)) return 1;
    vx_event_release(ev);
    ev = nullptr;
    ok = ps_rope::check_table("table_", tab.data(), hs, npos_t, theta, tab_pos0) && ok;
    if (rmode == "setup") {
      vx_kernel_release(k_setup);   // dev.kernel == k_setup in setup mode; the destructor must not release it twice
      dev.kernel = nullptr;
      return ps_elt::finish(ok, timer);
    }
    vx_kernel_release(k_setup);
  }
  if (kv8) {   // rope_kv_tab_q8 (q rotation work items + one work item per (b, KV head, K or V))
    ps_rope_q8_arg_t qa = {};
    qa.t.a = arg; qa.t.tab_addr = btab.addr; qa.ks_addr = bks.addr; qa.vs_addr = bvs.addr; qa.H_kv = Hkv8;
    if (dev.launch(&qa, sizeof(qa), B * (dim / 2) + 2 * B * Hkv8, grid_override, &ev)) return 1;
  } else if (tabmode) {   // rope_kv_tab: same arguments as rope_kv plus the table
    ps_rope_tab_arg_t targ = {};
    targ.a = arg; targ.tab_addr = btab.addr;
    if (dev.launch(&targ, sizeof(targ), B * (dim / 2), grid_override, &ev)) return 1;
  } else {
    if (dev.launch(&arg, sizeof(arg), B * (dim / 2), grid_override, &ev)) return 1;  // one thread per pair
  }
  if (kv8) {
    std::vector<float> qy8((size_t)B * dim), ks8(sc_n), vs8(sc_n);
    std::vector<int8_t> kc8(cache_n), vc8(cache_n);
    if (dev.read(qy8.data(), inplace ? bq : bqo, ev)) return 1;
    if (dev.read(kc8.data(), bkc, ev)) return 1;
    if (dev.read(vc8.data(), bvc, ev)) return 1;
    if (dev.read(ks8.data(), bks, ev)) return 1;
    if (dev.read(vs8.data(), bvs, ev)) return 1;
    vx_event_release(ev);
    ok = ps_elt::check("q_vs_f64", qy8.data(), q_ref, q_sc, tol) && ok;
    // host expectation from the DEVICE's own table (read back above): non-fused float32 rotation, then runq.c's quantize with group = head
    const uint32_t hh = hs / 2;
    std::vector<uint8_t> in_slot8(cache_n, 0), in_sc(sc_n, 0);
    ps::ErrStats ek, ev8, eks, evs, un;
    std::vector<int8_t> dk_all, dv_all, ek_all, ev_all;
    std::vector<float> dks_all, dvs_all;
    std::vector<float> xk(hs), xv(hs);
    std::vector<int8_t> qk(hs), qv(hs);
    std::vector<int32_t> pos_d(pos.begin(), pos.end());
    for (uint32_t b = 0; b < B; ++b) {
      const uint32_t p = (uint32_t)pos[b];
      for (uint32_t kvh = 0; kvh < Hkv8; ++kvh) {
        for (uint32_t jj = 0; jj < hh; ++jj) {
          const float x0 = k[(size_t)b * kv_dim + kvh * hs + 2 * jj], x1 = k[(size_t)b * kv_dim + kvh * hs + 2 * jj + 1];
          const float fcr = tab[((size_t)p * hh + jj) * 2], fci = tab[((size_t)p * hh + jj) * 2 + 1];
          volatile float m00 = x0 * fcr, m11 = x1 * fci, m01 = x0 * fci, m10 = x1 * fcr;   // separate roundings (no FMA contraction)
          volatile float r0 = m00 - m11, r1 = m01 + m10;
          xk[2 * jj] = r0; xk[2 * jj + 1] = r1;
          xv[2 * jj] = v[(size_t)b * kv_dim + kvh * hs + 2 * jj]; xv[2 * jj + 1] = v[(size_t)b * kv_dim + kvh * hs + 2 * jj + 1];
        }
        float sk = 0.0f, sv = 0.0f;
        ps::quantize_q80(xk.data(), hs, hs, qk.data(), &sk);
        ps::quantize_q80(xv.data(), hs, hs, qv.data(), &sv);
        const size_t slot = ((size_t)(layer * b_cap + b) * seq_len + p) * kv_dim + (size_t)kvh * hs;
        const size_t sci = ((size_t)(layer * b_cap + b) * Hkv8 + kvh) * seq_len + p;
        for (uint32_t i = 0; i < hs; ++i) {
          in_slot8[slot + i] = 1;
          if (kc8[slot + i] != qk[i]) ++ek.errors;
          if (vc8[slot + i] != qv[i]) ++ev8.errors;
          dk_all.push_back(kc8[slot + i]); dv_all.push_back(vc8[slot + i]);
          ek_all.push_back(qk[i]); ev_all.push_back(qv[i]);
        }
        in_sc[sci] = 1;
        uint32_t a, bb, c2, d2;
        std::memcpy(&a, &ks8[sci], 4); std::memcpy(&bb, &sk, 4);
        std::memcpy(&c2, &vs8[sci], 4); std::memcpy(&d2, &sv, 4);
        if (a != bb) ++eks.errors;
        if (c2 != d2) ++evs.errors;
        dks_all.push_back(ks8[sci]); dvs_all.push_back(vs8[sci]);
      }
    }
    ek.n = ev8.n = dk_all.size(); eks.n = evs.n = dks_all.size();
    ek.has_digest = ev8.has_digest = eks.has_digest = evs.has_digest = true;
    ek.digest = ps::fnv1a64_bytes(dk_all.data(), dk_all.size());
    ev8.digest = ps::fnv1a64_bytes(dv_all.data(), dv_all.size());
    eks.digest = ps::fnv1a64_f32(dks_all.data(), dks_all.size());
    evs.digest = ps::fnv1a64_f32(dvs_all.data(), dvs_all.size());
    ps::print_check("k8_exact", ek);
    ps::print_check("v8_exact", ev8);
    ps::print_check("ks_exact", eks);
    ps::print_check("vs_exact", evs);
    un.n = 2 * (cache_n + sc_n);
    for (size_t j = 0; j < cache_n; ++j)
      if (!in_slot8[j]) { if ((uint8_t)kc8[j] != 0xEF) ++un.errors; if ((uint8_t)vc8[j] != 0xEF) ++un.errors; }
    for (size_t j = 0; j < sc_n; ++j)
      if (!in_sc[j]) {
        uint32_t a, bb;
        std::memcpy(&a, &ks8[j], 4); std::memcpy(&bb, &vs8[j], 4);
        if (a != SENTINEL) ++un.errors;
        if (bb != SENTINEL) ++un.errors;
      }
    if (un.errors) un.max_rel = INFINITY;
    ps::print_check("kv8_untouched", un);
    // bytes accounting of this step's KV write (E5: the scales are counted): per (b, KV head) 2 x (hs int8 + 4 B scale)
    std::printf("PSINFO: kv8_write_bytes=%zu kv8_scale_bytes=%zu fp32_kv_write_bytes=%zu\n", (size_t)B * Hkv8 * 2 * (hs + 4),
                (size_t)B * Hkv8 * 2 * 4, (size_t)B * Hkv8 * 2 * hs * 4);
    ok = ok && ek.errors == 0 && ev8.errors == 0 && eks.errors == 0 && evs.errors == 0 && un.errors == 0;
    if (!dump_path.empty()) {   // inputs and device slots for the NumPy-port cross-check (tests/test_ep_kv8.py)
      std::FILE* f = std::fopen(dump_path.c_str(), "wb");
      if (!f) { std::printf("cannot write %s\nFAILED!\n", dump_path.c_str()); return 1; }
      const uint32_t hdr[6] = {hs, B, kv_dim, Hkv8, npos_t, 0};
      std::fwrite(hdr, 4, 6, f);
      std::fwrite(tab.data(), 4, tab.size(), f);
      std::fwrite(pos_d.data(), 4, B, f);
      std::fwrite(k.data(), 4, (size_t)B * kv_dim, f);
      std::fwrite(v.data(), 4, (size_t)B * kv_dim, f);
      std::fwrite(dk_all.data(), 1, dk_all.size(), f);
      std::fwrite(dv_all.data(), 1, dv_all.size(), f);
      std::fwrite(dks_all.data(), 4, dks_all.size(), f);
      std::fwrite(dvs_all.data(), 4, dvs_all.size(), f);
      std::fclose(f);
    }
    return ps_elt::finish(ok, timer);
  }
  std::vector<float> qy((size_t)B * dim), kc(cache_n), vc(cache_n);
  if (dev.read(qy.data(), inplace ? bq : bqo, ev)) return 1;
  if (dev.read(kc.data(), bkc, ev)) return 1;
  if (dev.read(vc.data(), bvc, ev)) return 1;
  vx_event_release(ev);

  ok = ps_elt::check("q_vs_f64", qy.data(), q_ref, q_sc, tol) && ok;
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
  if (!ko_fx.empty()) ok = ps_elt::check("kcache_vs_fixture", kslot.data(), ps_elt::to_f64(ko_fx), k_sc, tol) && ok;
  ok = ps_elt::check("vcache_exact", vslot.data(), ps_elt::to_f64(v), ps_elt::elem_scale(ps_elt::to_f64(v)), 0.0) && ok;
  ps::ErrStats un;
  un.n = 2 * cache_n;
  std::vector<float> untouched;   // (k_j, v_j) pairs, written slots replaced by 0.0f: n = 2 * cache_n; the bit-exact digest of this check
  untouched.reserve(2 * cache_n);
  for (size_t j = 0; j < cache_n; ++j) {
    if (in_slot[j]) { untouched.push_back(0.0f); untouched.push_back(0.0f); continue; }
    uint32_t a, bb;
    std::memcpy(&a, &kc[j], 4);
    std::memcpy(&bb, &vc[j], 4);
    if (a != SENTINEL) { ++un.errors; un.worst = j; }
    if (bb != SENTINEL) { ++un.errors; un.worst = j; }
    untouched.push_back(kc[j]);
    untouched.push_back(vc[j]);
  }
  if (un.errors) un.max_rel = INFINITY;
  un.digest = ps::fnv1a64_f32(untouched.data(), untouched.size());
  un.has_digest = true;
  ps::print_check("kv_untouched", un);
  ok = ok && un.ok();
  // bit-exact digests of the WHOLE cache arrays (sentinel included), so a stray or misplaced write also changes them
  std::printf("PSDIGEST: kcache_full n=%zu fnv1a64=%016llx\n", cache_n, (unsigned long long)ps::fnv1a64_f32(kc.data(), cache_n));
  std::printf("PSDIGEST: vcache_full n=%zu fnv1a64=%016llx\n", cache_n, (unsigned long long)ps::fnv1a64_f32(vc.data(), cache_n));
  return ps_elt::finish(ok, timer);
}
