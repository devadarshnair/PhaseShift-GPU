// main.cpp -- ps_mixed_attn host app: fused mixed-batch attention on Vortex (one launch: a prefill chunk
// plus a batch of decode queries, kernel.cpp), checked against a float64 host attention per part and, with -f, against
// the NumPy float32 outputs of a mixed-attention fixture (model/make_mixed_fixtures.py).
//
// Usage: ps_mixed_attn [-f <attention_mixed fixture dir> | -H <H_q> -K <H_kv> -D <head_dim> -c <chunk> [-o <s0>]
//                       -T <ctx>[,<ctx>...] [-n <n_seq>] [-M <Tmax_d>] [-d uniform|neg] [-q <qscale>] [-s <seed_p>]
//                       [-u <seed_d>]] [-Q <Qt>] [-t <Tk>] [-w <Tt>] [-r] [-a 0|1] [-e <tol>] [-k <kernel.vxbin>]
//   synthetic: prefill chunk = c query tokens at positions s0 .. s0+c-1 of ONE sequence over keys 0 .. s0+c-1; decode
//        batch = n_seq sequences, sequence b with context ctx[b] (-T a single value with -n n_seq: all equal; or a comma
//        list, one value per sequence, ragged). c = 0 (decode only) or n_seq = 0 (prefill only; -T 0) are allowed.
//        Inputs are drawn EXACTLY as the separate apps draw them, so the same flags reproduce their inputs bit for bit:
//          prefill part = ps_prefill_attn -B 1 -H -K -D -S c -o s0 -d -q -s <seed_p> (q [c][H_q][D] first, then K and
//            V interleaved per element for t < s0 + c; Tmax_p = s0 + c + 3, rows >= s0 + c are NaN);
//          decode part = ps_decode_attn_gqa -B n_seq -H -K -D -T ctx -d -q -s <seed_d> (q [n_seq][H_q][D] first, then
//            for b, t < ctx[b]: K and V interleaved; Tmax_d = max ctx + 3 unless -M; rows >= ctx[b] are NaN).
//        With equal ctx and equal tiles the fused launch must therefore produce the separate launches' outputs bit for
//        bit (PSDIGEST lines; tested). Default seeds 1 and 1 (both apps' default).
//   -f   ps-fixture-v1 `attention_mixed` fixture (model/make_mixed_fixtures.py): q_p (c, dim), k_p / v_p (s0+c, kv_dim),
//        out_p (c, dim), q_d (n_seq, dim), k_d / v_d (n_seq, Tmax_d, kv_dim), out_d (n_seq, dim); params n_heads,
//        n_kv_heads, head_size, chunk, s0, ctx (list). The app overwrites cache rows a part must not read with NaN.
//   -Q, -t  prefill query tokens per CTA (Qt) and key positions per key tile (Tk); default: the rule (below)
//   -w   decode key positions per tile (Tt; default 32, the gqa default)
//   -r   decode CTAs first (CTAs [0, n_d) decode, [n_d, n_d + n_p) prefill): ordering sensitivity, not the layout
//   -a   per-CTA timestamps (default 1; 0 = off, to measure their cost)
//   -e   override both tolerances (default: derived per part, "Error bound")
//   -W   PhaseShift [dcr_phase] way-mask words <l1word>:<l2word> (mechanism A2, ps patch waymask, encoding in
//        kernels/common/ps_waymask.h, methodology §14.A); absent = no change. Written with -R's block, or alone.
//   -R   PhaseShift [dcr_phase] KV ranges (ps patch mem-class; default none = no DCR write, the validated
//        behaviour): kv = kv[0] = the decode K..V span (classified KV_STREAM) and kv[1] = the prefill K..V span
//        (KV_PERSIST); with -R the K and V caches of each part share one allocation (V at the next 4 KiB boundary)
//        so each range covers exactly them (the allocator would otherwise put small buffers into the holes between
//        separate allocations); this moves K/V, so -R runs are compared only with each other. kvacct = the same
//        ranges with VX_PHASE_FLAG_RANGE_ACCT_ONLY (accounting only, PS_DEBUG_CLASS). Refused if any other buffer
//        lies inside a range.
// PS_HINTS=1 builds (CONFIGS -DPS_HINTS=1; kernels/common/ps_hint.h): prefill CTAs tag their K/V copy loads KV_PERSIST,
// decode CTAs their key-tile loop KV_STREAM, and every (CTA, key tile) is marked PREFILL_TILE or DECODE_TILE; the
// PSHINT line prints the expected per-type mark counts (tiles_p, tiles_d). A different binary from PS_HINTS=0: compare
// mechanisms only within one binary.
// Exactly one kernel launch per app run (methodology §2 consequence 5). Prints PSCHECK/PSDIGEST lines per part
// (ps_common.h), one PSCTA line per CTA and PSREGION/PSCORE summaries (timestamps), a PSINFO line, then PASSED!/FAILED!.
//
// LMEM and tiles: one CTA = NT*NW threads = all warps of a core, so one CTA is resident per core and owns the core's
// LMEM (VX_CAPS_LOCAL_MEM_SIZE, 16 KiB by default). The launch has ONE lmem size for all CTAs: max(prefill footprint
// ps_pattn_lmem(G, D, Qt, Tk, 1), decode footprint ps_attn_lmem(G, D, Tt, 1, 0)). Prefill tiles follow the rule
// unchanged (kernels/prefill_attn/main.cpp "LMEM and tiles"; methodology §14.4), INCLUDING its last clause (halve Qt
// while H_kv * ceil(c / Qt) < cores), computed on the prefill CTAs alone, so the prefill part has the separate launch's
// tiles and work (modeling choice for equal-work comparisons, methodology §14.6). Decode tiles: Tt = 32.
//
// Error metric and bound (methodology §14.6 = the bound of §14.2 bound of §14.4, applied per part):
// rel_i = |o_i - ref_i| / S_i, S_i = sum_t p_t |v_t[d]| over the keys the row sees; per row (u = 2^-24,
// gamma(k) = k u / (1 - k u)):
//   rel_i <= 2 E_s + 2 (s_max - s_min) u + gamma(n_acc + n_l + 8 (n_tiles - 1) + 10),
//   E_s = max_t [ gamma_D * scale * sum_d |q_d k_td| + 3 u |s_t| ],  n_acc = T_row + n_tiles,
//   prefill rows: T_row = p + 1 (position p), n_l = Tk + n_tiles, n_tiles = key tiles of the row's CTA;
//   decode rows:  T_row = ctx[b],             n_l = Tt + n_tiles, n_tiles = ceil(ctx[b] / Tt).
// (Derivation of each term: kernels/prefill_attn/main.cpp and kernels/decode_attn_gqa/main.cpp header comments; the
// fused kernel executes the same floating-point operations in the same order per row, so the bound carries over
// unchanged; the decode bound is now per sequence because ctx differs.) tol_part = max(1e-4, max over the part's
// rows of the bound); every run must also meet 1e-4 itself (PSINFO within_plan_tol). Against the fixture's NumPy
// float32 outputs, NumPy's own bound (same formula with n_acc + n_l = 2 T_row: einsum over t and the softmax sum) is
// added, as.
//
// PSINFO counts: useful_flops_p = 4 D H_q sum_i (s0 + i + 1), exec_flops_p = 4 D Rp nt over visited (CTA, tile)
// pairs, useful_flops_d = 4 D H_q sum_b ctx[b] (QK^T and PV), kv_bytes_p = 2 (s0+c) H_kv D 4,
// kv_bytes_d = 2 sum_b ctx[b] H_kv D 4 (E3: K and V of every sequence once), q_out_bytes = 2 (c + n_seq) H_q D 4.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <vortex.h>   // vortex2.h + vx_dcr_write / vx_check_occupancy

#include "common.h"
#include "ps_common.h"
#include "ps_fixture.h"
#include "ps_hint.h"
#include "ps_phase.h"
#include "ps_waymask.h"

namespace {

constexpr double kU = 1.0 / 16777216.0;   // 2^-24
constexpr double kPlanTol = 1e-4;

double gamma_k(double k) { return k * kU / (1.0 - k * kU); }

struct Ctx {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  std::vector<vx_buffer_h> bufs;
  ~Ctx() {
    if (!device) return;
    for (auto b : bufs)
      if (b) vx_buffer_release(b);
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
  std::printf("Usage: %s [-f fixture | -H Hq -K Hkv -D d -c chunk [-o s0] -T ctx[,ctx...] [-n nseq] [-M tmax_d] "
              "[-d uniform|neg] [-q qscale] [-s seed_p] [-u seed_d]] [-Q Qt] [-t Tk] [-w Tt] [-r] [-a 0|1] [-e tol] "
              "[-R none|kv|kvacct] [-W l1:l2] [-k kernel.vxbin]\n", app);
}

// largest Tk <= 32 whose prefill LMEM footprint fits `cap` bytes (0 if none); kernels/prefill_attn/main.cpp max_tk
uint32_t max_tk(uint32_t G, uint32_t D, uint32_t Qt, uint64_t cap) {
  for (uint32_t tk = 32; tk >= 1; --tk)
    if (ps_pattn_lmem(G, D, Qt, tk, 1).total * 4ull <= cap) return tk;
  return 0;
}

// One attention row (one query, one head) in float64: out/scale for D columns and the bound ingredients.
struct RowRef {
  double es = 0.0, smax = -INFINITY, smin = INFINITY;
};
RowRef ref_row(const float* qh, const float* K, const float* V, size_t row_stride, uint32_t T, uint32_t D,
               double* out, double* scale, std::vector<double>& s, std::vector<double>& p) {
  RowRef r;
  const double sc = 1.0 / std::sqrt((double)D);
  s.resize(T);
  p.resize(T);
  for (uint32_t t = 0; t < T; ++t) {
    const float* kt = K + (size_t)t * row_stride;
    double dot = 0.0, mag = 0.0;
    for (uint32_t d = 0; d < D; ++d) { dot += (double)qh[d] * kt[d]; mag += std::fabs((double)qh[d] * kt[d]); }
    s[t] = dot * sc;
    r.es = std::fmax(r.es, gamma_k(D) * mag * sc + 3 * kU * std::fabs(s[t]));
    r.smax = std::fmax(r.smax, s[t]);
    r.smin = std::fmin(r.smin, s[t]);
  }
  double l = 0.0;
  for (uint32_t t = 0; t < T; ++t) { p[t] = std::exp(s[t] - r.smax); l += p[t]; }
  for (uint32_t t = 0; t < T; ++t) p[t] /= l;
  for (uint32_t d = 0; d < D; ++d) {
    double o = 0.0, a = 0.0;
    for (uint32_t t = 0; t < T; ++t) {
      const double vt = V[(size_t)t * row_stride + d];
      o += p[t] * vt;
      a += p[t] * std::fabs(vt);
    }
    out[d] = o;
    scale[d] = a;
  }
  return r;
}
double row_bound(const RowRef& r, double chain) { return 2 * r.es + 2 * (r.smax - r.smin) * kU + gamma_k(chain + 10); }

struct PartRef {
  std::vector<double> out, scale;
  double bound = 0.0, bound_np = 0.0;   // device-chain bound and NumPy-chain bound (max over rows)
  double smin = INFINITY, smax = -INFINITY;
};

}  // namespace

int main(int argc, char** argv) {
  std::string fixture, dist = "uniform", kernel_file = "kernel.vxbin", ctx_arg, range_mode = "none";
  uint32_t H_q = 8, H_kv = 4, D = 8, C = 16, s0 = 0, n_seq = 0, Tmax_d = 0, seed_p = 1, seed_d = 1;
  bool have_wm = false;   // PhaseShift (patch waymask)
  uint32_t wm_l1 = 0, wm_l2 = 0;
  uint32_t Qt = 0, Tk = 0, Tt = 32, order = 0, use_ts = 1;
  bool n_given = false;
  double qscale = 1.0, tol_override = 0.0;
  int c;
  while ((c = getopt(argc, argv, "f:H:K:D:c:o:T:n:M:d:q:s:u:Q:t:w:ra:e:R:k:W:h")) != -1) {
    switch (c) {
      case 'f': fixture = optarg; break;
      case 'H': H_q = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'K': H_kv = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'D': D = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'c': C = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'o': s0 = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'T': ctx_arg = optarg; break;
      case 'n': n_seq = (uint32_t)std::strtoul(optarg, nullptr, 10); n_given = true; break;
      case 'M': Tmax_d = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'd': dist = optarg; break;
      case 'q': qscale = std::strtod(optarg, nullptr); break;
      case 's': seed_p = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'u': seed_d = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'Q': Qt = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 't': Tk = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'w': Tt = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'r': order = 1; break;
      case 'a': use_ts = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'e': tol_override = std::strtod(optarg, nullptr); break;
      case 'R': range_mode = optarg; break;
      case 'W':   // PhaseShift (patch waymask): [dcr_phase] way masks "<l1word>:<l2word>" (ps_waymask.h)
        if (!ps::parse_waymask_arg(optarg, &wm_l1, &wm_l2)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        have_wm = true;
        break;
      case 'k': kernel_file = optarg; break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  if (dist != "uniform" && dist != "neg") { std::printf("bad -d %s\nFAILED!\n", dist.c_str()); return 1; }
  if (Tt == 0) { std::printf("-w must be >= 1\nFAILED!\n"); return 1; }

  // ---- inputs
  std::vector<float> qp, Kp, Vp, qd, Kd, Vd, outp_fx, outd_fx;
  std::vector<uint32_t> ctxv;
  uint32_t Tmax_p = 0;
  std::string label;
  try {
    if (!fixture.empty()) {
      ps::Fixture fx(fixture);
      if (fx.op() != "attention_mixed") throw ps::FixtureError("need an attention_mixed fixture");
      const ps::Json& p = fx.meta().at("params");
      H_q = (uint32_t)p.at("n_heads").as_num();
      H_kv = (uint32_t)p.at("n_kv_heads").as_num();
      D = (uint32_t)p.at("head_size").as_num();
      C = (uint32_t)p.at("chunk").as_num();
      s0 = (uint32_t)p.at("s0").as_num();
      for (auto& v : p.at("ctx").arr) ctxv.push_back((uint32_t)v.as_num());
      n_seq = (uint32_t)ctxv.size();
      const size_t qd_dim = (size_t)H_q * D, kv_dim = (size_t)H_kv * D;
      ps::Tensor tqp = fx.tensor("q_p"), tkp = fx.tensor("k_p"), tvp = fx.tensor("v_p"), top = fx.tensor("out_p");
      ps::Tensor tqd = fx.tensor("q_d"), tkd = fx.tensor("k_d"), tvd = fx.tensor("v_d"), tod = fx.tensor("out_d");
      Tmax_d = tkd.shape.size() == 3 ? tkd.dim(1) : 0;
      if (tqp.numel() != C * qd_dim || top.numel() != C * qd_dim || tkp.numel() != (s0 + C) * kv_dim ||
          tvp.shape != tkp.shape || tqd.numel() != n_seq * qd_dim || tod.numel() != n_seq * qd_dim ||
          tkd.numel() != (size_t)n_seq * Tmax_d * kv_dim || tvd.shape != tkd.shape)
        throw ps::FixtureError("attention_mixed fixture shapes do not match its params");
      for (auto T : ctxv)
        if (T == 0 || T > Tmax_d) throw ps::FixtureError("ctx entries must be in [1, Tmax_d]");
      Tmax_p = s0 + C + 3;
      qp = std::move(tqp.data);
      outp_fx = std::move(top.data);
      Kp.assign((size_t)Tmax_p * kv_dim, std::nanf(""));
      Vp.assign(Kp.size(), std::nanf(""));
      std::copy(tkp.data.begin(), tkp.data.end(), Kp.begin());
      std::copy(tvp.data.begin(), tvp.data.end(), Vp.begin());
      qd = std::move(tqd.data);
      outd_fx = std::move(tod.data);
      Kd = std::move(tkd.data);
      Vd = std::move(tvd.data);
      for (uint32_t b = 0; b < n_seq; ++b)   // rows a decode CTA must not read
        for (size_t i = ((size_t)b * Tmax_d + ctxv[b]) * kv_dim; i < (size_t)(b + 1) * Tmax_d * kv_dim; ++i)
          Kd[i] = Vd[i] = std::nanf("");
      label = fx.model() + ":attention_mixed:" + (fx.meta().has("name") ? fx.meta().at("name").as_str() : "");
    } else {
      // ctx list
      if (!ctx_arg.empty()) {
        size_t pos = 0;
        while (pos <= ctx_arg.size()) {
          size_t e = ctx_arg.find(',', pos);
          if (e == std::string::npos) e = ctx_arg.size();
          ctxv.push_back((uint32_t)std::strtoul(ctx_arg.substr(pos, e - pos).c_str(), nullptr, 10));
          pos = e + 1;
        }
      }
      if (ctxv.size() == 1 && ctxv[0] == 0) ctxv.clear();               // -T 0: no decode part
      if (ctxv.size() == 1 && n_given) ctxv.assign(n_seq, ctxv[0]);      // -T t -n n: n equal sequences
      if (n_given && ctxv.size() != n_seq) throw ps::FixtureError("-n disagrees with the -T list");
      n_seq = (uint32_t)ctxv.size();
      uint32_t tmax = 0;
      for (auto T : ctxv) {
        if (T == 0) throw ps::FixtureError("every ctx must be >= 1");
        tmax = std::max(tmax, T);
      }
      if (Tmax_d == 0) Tmax_d = tmax + 3;
      if (Tmax_d < tmax) throw ps::FixtureError("Tmax_d < max ctx");
      if (H_q == 0 || H_kv == 0 || D == 0 || H_q % H_kv != 0) throw ps::FixtureError("need H_q, H_kv, D >= 1, H_kv | H_q");
      const bool neg = (dist == "neg");
      const size_t kv_dim = (size_t)H_kv * D;
      // prefill part: kernels/prefill_attn/main.cpp synthetic generator with B = 1, S = c
      Tmax_p = s0 + C + 3;
      {
        ps::Rng rng(seed_p);
        qp.resize((size_t)C * H_q * D);
        Kp.assign((size_t)Tmax_p * kv_dim, std::nanf(""));
        Vp.assign(Kp.size(), std::nanf(""));
        for (auto& x : qp) x = (float)(qscale * (neg ? 0.75 + 0.25 * rng.uniform() : rng.uniform()));
        if (C > 0)
          for (uint32_t t = 0; t < s0 + C; ++t)
            for (uint32_t i = 0; i < kv_dim; ++i) {
              const size_t idx = (size_t)t * kv_dim + i;
              Kp[idx] = neg ? -0.75f + 0.25f * rng.uniform() : rng.uniform();
              Vp[idx] = rng.uniform();
            }
      }
      // decode part: kernels/decode_attn_gqa/main.cpp synthetic generator with B = n_seq, per-sequence T
      {
        ps::Rng rng(seed_d);
        qd.resize((size_t)n_seq * H_q * D);
        Kd.assign((size_t)n_seq * Tmax_d * kv_dim, std::nanf(""));
        Vd.assign(Kd.size(), std::nanf(""));
        for (auto& x : qd) x = (float)(qscale * (neg ? 0.75 + 0.25 * rng.uniform() : rng.uniform()));
        for (uint32_t b = 0; b < n_seq; ++b)
          for (uint32_t t = 0; t < ctxv[b]; ++t)
            for (uint32_t i = 0; i < kv_dim; ++i) {
              const size_t idx = ((size_t)b * Tmax_d + t) * kv_dim + i;
              Kd[idx] = neg ? -0.75f + 0.25f * rng.uniform() : rng.uniform();
              Vd[idx] = rng.uniform();
            }
      }
      char buf[160];
      std::snprintf(buf, sizeof(buf), "synthetic:%s:q%g:s%u:u%u", dist.c_str(), qscale, seed_p, seed_d);
      label = buf;
    }
  } catch (const std::exception& e) {
    std::printf("input error: %s\nFAILED!\n", e.what());
    return 1;
  }
  const uint32_t G = H_q / H_kv;
  if (C == 0 && n_seq == 0) { std::printf("empty launch (c = 0 and no decode sequences)\nFAILED!\n"); return 1; }
  if (range_mode != "none" && range_mode != "kv" && range_mode != "kvacct") {
    std::printf("bad -R %s\nFAILED!\n", range_mode.c_str());
    return 1;
  }
  if (C > 0 && D % 4 != 0) { std::printf("head_dim %u must be a multiple of 4 (prefill 4 x 4 blocks)\nFAILED!\n", D); return 1; }
  if (n_seq > 0 && G != 1 && G != 2 && G != 4 && G != 8) {
    std::printf("group size %u not supported by the decode part (1, 2, 4, 8)\nFAILED!\n", G);
    return 1;
  }

  // ---- device
  Ctx ctx;
  PS_RT_CHECK(vx_device_open(0, &ctx.device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  PS_RT_CHECK(vx_queue_create(ctx.device, &qi, &ctx.queue));
  uint64_t nt = 0, nw = 0, lmem_cap = 0, ncores = 0;
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_WARPS, &nw));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_CORES, &ncores));
  const uint32_t nth = (uint32_t)(nt * nw);   // one CTA = all warps of a core

  // ---- prefill tiles: the rule (kernels/prefill_attn/main.cpp), B = 1, S = c
  uint32_t nqt = 0, n_p = 0;
  ps_pattn_lmem_t Lp = {};
  if (C > 0) {
    if (Qt == 0) {
      uint32_t best = 0, best_tk = 0;
      for (uint32_t cand = 64; cand >= 1; cand >>= 1) {
        if (cand > C && cand > 1) continue;
        if (G * cand > 128) continue;
        const uint32_t tk = max_tk(G, D, cand, lmem_cap);
        if (tk >= 8) { best = cand; best_tk = tk; break; }
      }
      if (best == 0) { best = 1; best_tk = max_tk(G, D, 1, lmem_cap); }
      while (best > 1 && (uint64_t)H_kv * ((C + best - 1) / best) < ncores) {
        best >>= 1;
        best_tk = max_tk(G, D, best, lmem_cap);
      }
      Qt = best;
      if (Tk == 0) Tk = best_tk;
    }
    if (Tk == 0) Tk = std::min<uint32_t>(32, max_tk(G, D, Qt, lmem_cap));
    if (Qt == 0 || Tk == 0) { std::printf("no prefill tile fits the LMEM (%llu B)\nFAILED!\n", (unsigned long long)lmem_cap); return 1; }
    Lp = ps_pattn_lmem(G, D, Qt, Tk, 1);
    nqt = (C + Qt - 1) / Qt;
    n_p = H_kv * nqt;
  }
  const uint32_t n_d = n_seq * H_kv;
  const ps_attn_lmem_t Ld = ps_attn_lmem(G, D, Tt, 1, 0);
  const uint32_t lmem_bytes = std::max(C > 0 ? Lp.total * 4 : 0u, n_seq > 0 ? Ld.total * 4 : 0u);
  if (lmem_bytes > lmem_cap) {
    std::printf("LMEM footprint %u B > %llu B per core (G=%u D=%u Qt=%u Tk=%u Tt=%u)\nFAILED!\n", lmem_bytes,
                (unsigned long long)lmem_cap, G, D, Qt, Tk, Tt);
    return 1;
  }
  const uint32_t grid = n_p + n_d;
  auto p_kend = [&](uint32_t qt) { const uint32_t q0 = qt * Qt, nq = std::min(Qt, C - q0); return s0 + q0 + nq; };
  auto p_ntiles = [&](uint32_t qt) { return (p_kend(qt) + Tk - 1) / Tk; };

  // ---- references, bounds, work counts
  const size_t kv_dim = (size_t)H_kv * D;
  PartRef rp, rd;
  std::vector<double> sbuf, pbuf;
  double useful_p = 0, exec_p = 0, useful_d = 0;
  uint64_t tiles_p = 0, sum_ctx = 0, tiles_d = 0;   // tiles_d: (decode CTA, key tile) pairs (mark expectation)
  uint32_t ntiles_p_max = 0, ntiles_d_max = 0;
  rp.out.assign((size_t)C * H_q * D, 0.0);
  rp.scale.assign(rp.out.size(), 0.0);
  for (uint32_t i = 0; i < C; ++i) {
    const uint32_t pos = s0 + i, T = pos + 1, ntl = p_ntiles(i / Qt);
    const double chain = (double)(T + ntl) + (double)(Tk + ntl) + 8.0 * (ntl - 1.0);
    for (uint32_t h = 0; h < H_q; ++h) {
      const size_t ob = ((size_t)i * H_q + h) * D;
      RowRef r = ref_row(&qp[ob], &Kp[(h / G) * D], &Vp[(h / G) * D], kv_dim, T, D, &rp.out[ob], &rp.scale[ob], sbuf, pbuf);
      rp.bound = std::fmax(rp.bound, row_bound(r, chain));
      rp.bound_np = std::fmax(rp.bound_np, row_bound(r, 2.0 * T));
      rp.smin = std::fmin(rp.smin, r.smin);
      rp.smax = std::fmax(rp.smax, r.smax);
    }
    useful_p += 4.0 * D * H_q * T;
  }
  for (uint32_t qt = 0; qt < nqt; ++qt) {
    const uint32_t ke = p_kend(qt), ntl = p_ntiles(qt);
    for (uint32_t j = 0; j < ntl; ++j) exec_p += 4.0 * Lp.rows * std::min(Tk, ke - j * Tk) * D;
    tiles_p += ntl;
    ntiles_p_max = std::max(ntiles_p_max, ntl);
  }
  exec_p *= H_kv;
  tiles_p *= H_kv;
  rd.out.assign((size_t)n_seq * H_q * D, 0.0);
  rd.scale.assign(rd.out.size(), 0.0);
  for (uint32_t b = 0; b < n_seq; ++b) {
    const uint32_t T = ctxv[b], ntl = (T + Tt - 1) / Tt;
    const double chain = (double)(T + ntl) + (double)(Tt + ntl) + 8.0 * (ntl - 1.0);
    for (uint32_t h = 0; h < H_q; ++h) {
      const size_t ob = ((size_t)b * H_q + h) * D;
      const size_t kb = (size_t)b * Tmax_d * kv_dim + (h / G) * D;
      RowRef r = ref_row(&qd[ob], &Kd[kb], &Vd[kb], kv_dim, T, D, &rd.out[ob], &rd.scale[ob], sbuf, pbuf);
      rd.bound = std::fmax(rd.bound, row_bound(r, chain));
      rd.bound_np = std::fmax(rd.bound_np, row_bound(r, 2.0 * T));
      rd.smin = std::fmin(rd.smin, r.smin);
      rd.smax = std::fmax(rd.smax, r.smax);
    }
    useful_d += 4.0 * D * H_q * T;
    sum_ctx += T;
    tiles_d += (uint64_t)ntl * (H_q / G);   // H_q / G = H_kv decode CTAs per sequence
    ntiles_d_max = std::max(ntiles_d_max, ntl);
  }
  const double tol_p = tol_override > 0 ? tol_override : std::fmax(kPlanTol, rp.bound);
  const double tol_d = tol_override > 0 ? tol_override : std::fmax(kPlanTol, rd.bound);
  uint32_t ctx_min = ~0u, ctx_max = 0;
  for (auto T : ctxv) { ctx_min = std::min(ctx_min, T); ctx_max = std::max(ctx_max, T); }
  if (n_seq == 0) ctx_min = 0;
  std::printf("ps_mixed_attn: %s H_q=%u H_kv=%u D=%u G=%u c=%u s0=%u Tmax_p=%u n_seq=%u ctx=[%u,%u] sum_ctx=%llu "
              "Tmax_d=%u Qt=%u Tk=%u rows=%u nqt=%u ntiles_p_max=%u Tt=%u ntiles_d_max=%u n_p=%u n_d=%u grid=%u order=%s "
              "ts=%u NT=%u NW=%u cores=%u lmem=%u/%llu logits_p=[%.1f,%.1f] logits_d=[%.1f,%.1f] bound_p=%.3e tol_p=%.3e "
              "bound_d=%.3e tol_d=%.3e\n",
              label.c_str(), H_q, H_kv, D, G, C, s0, Tmax_p, n_seq, ctx_min, ctx_max, (unsigned long long)sum_ctx,
              Tmax_d, Qt, Tk, C > 0 ? Lp.rows : 0, nqt, ntiles_p_max, Tt, ntiles_d_max, n_p, n_d, grid,
              order ? "decode_first" : "prefill_first", use_ts, (unsigned)nt, (unsigned)nw, (unsigned)ncores,
              lmem_bytes, (unsigned long long)lmem_cap, C > 0 ? rp.smin : 0.0, C > 0 ? rp.smax : 0.0,
              n_seq > 0 ? rd.smin : 0.0, n_seq > 0 ? rd.smax : 0.0, rp.bound, tol_p, rd.bound, tol_d);
  bool ok = true;
  double tolfx_p = 0.0, tolfx_d = 0.0;
  const bool have_fx = !fixture.empty();
  if (have_fx) {
    // NumPy float32 outputs vs float64 within NumPy's own bound
    if (C > 0) {
      auto st = ps::compare(outp_fx.data(), rp.out.data(), rp.scale.data(), rp.out.size(), std::fmax(kPlanTol, rp.bound_np));
      ps::print_check("prefill:fixture_vs_f64", st);
      ok = ok && st.ok();
    }
    if (n_seq > 0) {
      auto st = ps::compare(outd_fx.data(), rd.out.data(), rd.scale.data(), rd.out.size(), std::fmax(kPlanTol, rd.bound_np));
      ps::print_check("decode:fixture_vs_f64", st);
      ok = ok && st.ok();
    }
    tolfx_p = tol_override > 0 ? tol_override : std::fmax(kPlanTol, rp.bound + rp.bound_np);
    tolfx_d = tol_override > 0 ? tol_override : std::fmax(kPlanTol, rd.bound + rd.bound_np);
  }

  // ---- buffers (an empty part gets a 64 B dummy so every address is valid; the kernel never touches it)
  ps_mattn_arg_t arg = {};
  arg.H_q = H_q; arg.H_kv = H_kv; arg.D = D; arg.G = G;
  arg.S = C; arg.s0 = s0; arg.Tmax_p = Tmax_p; arg.Qt = Qt; arg.Tk = Tk; arg.nqt = nqt; arg.n_p = n_p;
  arg.n_seq = n_seq; arg.Tmax_d = Tmax_d; arg.Tt = Tt; arg.n_d = n_d; arg.order = order;
  arg.scale = 1.0f / sqrtf((float)D);
  auto mkbuf = [&](uint64_t bytes, int flags, uint64_t* addr, vx_buffer_h* out) -> int {
    vx_buffer_h b = nullptr;
    int r = vx_buffer_create(ctx.device, std::max<uint64_t>(bytes, 64), flags, &b);
    if (r != 0) return r;
    ctx.bufs.push_back(b);
    *out = b;
    return vx_buffer_address(b, addr);
  };
  const size_t NOp = (size_t)C * H_q * D, NOd = (size_t)n_seq * H_q * D;
  vx_buffer_h qp_b, kp_b, vp_b, op_b, qd_b, kd_b, vd_b, od_b, ctx_b, ts_b = nullptr;
  PS_RT_CHECK(mkbuf(qp.size() * 4, VX_MEM_READ, &arg.qp_addr, &qp_b));
  // -R kv/kvacct: K and V of a part share ONE allocation (V at the next 4 KiB boundary after K) so that one
  // [dcr_phase] range covers exactly them; the runtime allocator places small buffers in the holes between separate
  // allocations (measured: q_d and ctx landed between kp and vp), which a span would mislabel. This changes the K/V
  // addresses, so -R runs are compared only with each other, never with -R none (the validated layout, unchanged).
  const bool packed = (range_mode != "none");
  uint64_t voff_p = 0, voff_d = 0;
  auto round_page = [](uint64_t b) { return (std::max<uint64_t>(b, 64) + 4095) & ~uint64_t(4095); };
  if (packed) {
    voff_p = round_page(Kp.size() * 4);
    PS_RT_CHECK(mkbuf(voff_p + Vp.size() * 4, VX_MEM_READ, &arg.kp_addr, &kp_b));
    vp_b = kp_b;
    arg.vp_addr = arg.kp_addr + voff_p;
  } else {
    PS_RT_CHECK(mkbuf(Kp.size() * 4, VX_MEM_READ, &arg.kp_addr, &kp_b));
    PS_RT_CHECK(mkbuf(Vp.size() * 4, VX_MEM_READ, &arg.vp_addr, &vp_b));
  }
  PS_RT_CHECK(mkbuf(NOp * 4, VX_MEM_READ_WRITE, &arg.op_addr, &op_b));
  PS_RT_CHECK(mkbuf(qd.size() * 4, VX_MEM_READ, &arg.qd_addr, &qd_b));
  if (packed) {
    voff_d = round_page(Kd.size() * 4);
    PS_RT_CHECK(mkbuf(voff_d + Vd.size() * 4, VX_MEM_READ, &arg.kd_addr, &kd_b));
    vd_b = kd_b;
    arg.vd_addr = arg.kd_addr + voff_d;
  } else {
    PS_RT_CHECK(mkbuf(Kd.size() * 4, VX_MEM_READ, &arg.kd_addr, &kd_b));
    PS_RT_CHECK(mkbuf(Vd.size() * 4, VX_MEM_READ, &arg.vd_addr, &vd_b));
  }
  PS_RT_CHECK(mkbuf(NOd * 4, VX_MEM_READ_WRITE, &arg.od_addr, &od_b));
  PS_RT_CHECK(mkbuf(ctxv.size() * 4, VX_MEM_READ, &arg.ctx_addr, &ctx_b));
  const uint64_t ts_bytes = (uint64_t)grid * PS_MATTN_TS_WORDS * 4;
  if (use_ts) PS_RT_CHECK(mkbuf(ts_bytes, VX_MEM_READ_WRITE, &arg.ts_addr, &ts_b));
  auto put = [&](vx_buffer_h b, const void* src, uint64_t bytes, uint64_t off = 0) -> int {
    return bytes ? vx_enqueue_write(ctx.queue, b, off, src, bytes, 0, nullptr, nullptr) : 0;
  };
  PS_RT_CHECK(put(qp_b, qp.data(), qp.size() * 4));
  PS_RT_CHECK(put(kp_b, Kp.data(), Kp.size() * 4));
  PS_RT_CHECK(put(vp_b, Vp.data(), Vp.size() * 4, voff_p));
  PS_RT_CHECK(put(qd_b, qd.data(), qd.size() * 4));
  PS_RT_CHECK(put(kd_b, Kd.data(), Kd.size() * 4));
  PS_RT_CHECK(put(vd_b, Vd.data(), Vd.size() * 4, voff_d));
  PS_RT_CHECK(put(ctx_b, ctxv.data(), ctxv.size() * 4));
  const std::vector<float> poison_p(NOp, std::nanf("")), poison_d(NOd, std::nanf(""));
  PS_RT_CHECK(put(op_b, poison_p.data(), NOp * 4));
  PS_RT_CHECK(put(od_b, poison_d.data(), NOd * 4));
  const std::vector<uint32_t> ts_zero(use_ts ? grid * PS_MATTN_TS_WORDS : 0, 0u);
  if (use_ts) PS_RT_CHECK(put(ts_b, ts_zero.data(), ts_bytes));
  if (range_mode != "none") {
    // PhaseShift [dcr_phase] ranges: kv[0] = decode K..V span, kv[1] = prefill K..V span (each covers the allocator
    // gap between its two buffers); no other buffer may lie inside either span (checked).
    struct Buf { const char* name; uint64_t addr, bytes; };
    const uint64_t kpb = std::max<uint64_t>(Kp.size() * 4, 64), kdb = std::max<uint64_t>(Kd.size() * 4, 64);
    const uint64_t lo_d = std::min(arg.kd_addr, arg.vd_addr), hi_d = std::max(arg.kd_addr, arg.vd_addr) + kdb;
    const uint64_t lo_p = std::min(arg.kp_addr, arg.vp_addr), hi_p = std::max(arg.kp_addr, arg.vp_addr) + kpb;
    const Buf others[] = {{"q_p", arg.qp_addr, std::max<uint64_t>(qp.size() * 4, 64)},
                          {"out_p", arg.op_addr, std::max<uint64_t>(NOp * 4, 64)},
                          {"q_d", arg.qd_addr, std::max<uint64_t>(qd.size() * 4, 64)},
                          {"out_d", arg.od_addr, std::max<uint64_t>(NOd * 4, 64)},
                          {"ctx", arg.ctx_addr, std::max<uint64_t>(ctxv.size() * 4, 64)},
                          {"ts", arg.ts_addr, use_ts ? ts_bytes : 0}};
    auto hits = [](uint64_t lo, uint64_t hi, const Buf& b) { return b.bytes && b.addr < hi && b.addr + b.bytes > lo; };
    bool bad = (lo_d < hi_p && hi_d > lo_p);   // the two spans must not overlap
    for (const auto& b : others) {
      if (hits(lo_d, hi_d, b) || hits(lo_p, hi_p, b)) {
        std::printf("-R %s: buffer %s [0x%llx,+%llu) lies inside a K..V span\n", range_mode.c_str(), b.name,
                    (unsigned long long)b.addr, (unsigned long long)b.bytes);
        bad = true;
      }
    }
    if (bad) {
      std::printf("-R %s: K..V spans decode [0x%llx,0x%llx) (kd 0x%llx vd 0x%llx) prefill [0x%llx,0x%llx) (kp 0x%llx "
                  "vp 0x%llx) not usable\nFAILED!\n", range_mode.c_str(), (unsigned long long)lo_d,
                  (unsigned long long)hi_d, (unsigned long long)arg.kd_addr, (unsigned long long)arg.vd_addr,
                  (unsigned long long)lo_p, (unsigned long long)hi_p, (unsigned long long)arg.kp_addr,
                  (unsigned long long)arg.vp_addr);
      return 1;
    }
    ps::PhaseCfg pc;
    pc.kv_base[0] = lo_d;
    pc.kv_limit[0] = hi_d;
    pc.kv_base[1] = lo_p;
    pc.kv_limit[1] = hi_p;
    if (range_mode == "kvacct") pc.flags |= ps::phase_dcr::FLAG_RANGE_ACCT_ONLY;
    if (have_wm) {
      ps::WayMaskCtl wmc;
      PS_RT_CHECK(wmc.set(pc, wm_l1, wm_l2));
      ps::waymask_print(wmc);
    }
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
    std::printf("PSRANGE: mode=%s kv0=0x%llx-0x%llx kv1=0x%llx-0x%llx flags=0x%x\n", range_mode.c_str(),
                (unsigned long long)lo_d, (unsigned long long)hi_d, (unsigned long long)lo_p, (unsigned long long)hi_p,
                pc.flags);
  } else if (have_wm) {   // PhaseShift (patch waymask): masks only, every other [dcr_phase] field at reset (0)
    ps::PhaseCfg pc;
    ps::WayMaskCtl wmc;
    PS_RT_CHECK(wmc.set(pc, wm_l1, wm_l2));
    ps::waymask_print(wmc);
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
  }
  PS_RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, "attn_mixed", &ctx.kernel));
  PS_RT_CHECK(vx_check_occupancy(ctx.device, nth, lmem_bytes));

  std::printf("launch: kernel=attn_mixed grid=%u block=%u lmem_bytes=%u\n", grid, nth, lmem_bytes);
  // work units the kernel marks (PS_HINTS=1 only), per type other,prefill_tile,decode_tile (VX_PS_MARK_*);
  // compared with SimX's PSMARK-TOTAL line by tests/test_mem_class.py
  std::printf("PSHINT: ps_hints=%d expect_marks=0,%llu,%llu\n", (int)PS_HINTS,
              (unsigned long long)(PS_HINTS ? tiles_p : 0), (unsigned long long)(PS_HINTS ? tiles_d : 0));
  std::vector<float> Op(NOp), Od(NOd);
  std::vector<uint32_t> ts(use_ts ? grid * PS_MATTN_TS_WORDS : 0);
  ps::Timer timer;
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = ctx.kernel;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = grid;
  li.block_dim[0] = nth;
  li.lmem_size = lmem_bytes;
  vx_event_h launch_ev = nullptr;
  PS_RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &launch_ev));
  std::vector<vx_event_h> evs;
  auto get = [&](void* dst, vx_buffer_h b, uint64_t bytes) -> int {
    if (!bytes) return 0;
    vx_event_h e = nullptr;
    int r = vx_enqueue_read(ctx.queue, dst, b, 0, bytes, 1, &launch_ev, &e);
    if (r == 0) evs.push_back(e);
    return r;
  };
  PS_RT_CHECK(get(Op.data(), op_b, NOp * 4));
  PS_RT_CHECK(get(Od.data(), od_b, NOd * 4));
  if (use_ts) PS_RT_CHECK(get(ts.data(), ts_b, ts_bytes));
  for (auto e : evs) {
    PS_RT_CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e);
  }
  vx_event_release(launch_ev);
  std::printf("Elapsed: %.0f ms\n", timer.ms());

  // ---- correctness per part
  double max_rel_p = 0.0, max_rel_d = 0.0;
  if (C > 0) {
    auto s = ps::compare(Op.data(), rp.out.data(), rp.scale.data(), NOp, tol_p);
    ps::print_check("prefill:vs_f64", s);
    ok = ok && s.ok();
    max_rel_p = s.max_rel;
    if (have_fx) {
      std::vector<double> r(outp_fx.begin(), outp_fx.end());
      auto st = ps::compare(Op.data(), r.data(), rp.scale.data(), NOp, tolfx_p);
      ps::print_check("prefill:vs_fixture", st);
      ok = ok && st.ok();
    }
  }
  if (n_seq > 0) {
    auto s = ps::compare(Od.data(), rd.out.data(), rd.scale.data(), NOd, tol_d);
    ps::print_check("decode:vs_f64", s);
    ok = ok && s.ok();
    max_rel_d = s.max_rel;
    if (have_fx) {
      std::vector<double> r(outd_fx.begin(), outd_fx.end());
      auto st = ps::compare(Od.data(), r.data(), rd.scale.data(), NOd, tolfx_d);
      ps::print_check("decode:vs_fixture", st);
      ok = ok && st.ok();
    }
  }

  // ---- per-CTA timestamps -> per-region attribution
  if (use_ts) {
    bool ts_ok = true;
    uint64_t t0 = ~0ull, t1 = 0;
    struct Reg { uint64_t ctas = 0, busy = 0, first = ~0ull, last = 0; } reg[2];
    std::vector<uint64_t> core_busy[2], core_last(ncores, 0);
    std::vector<uint32_t> core_ctas[2];
    for (int r = 0; r < 2; ++r) { core_busy[r].assign(ncores, 0); core_ctas[r].assign(ncores, 0); }
    for (uint32_t i = 0; i < grid; ++i) {
      const uint32_t* w = &ts[(size_t)i * PS_MATTN_TS_WORDS];
      const uint64_t st = ((uint64_t)w[1] << 32) | w[0], en = ((uint64_t)w[3] << 32) | w[2];
      const uint32_t core = w[4], region = w[5], idx = w[6];
      const bool first = i < (order ? n_d : n_p);
      const uint32_t want_region = order ? (first ? 1u : 0u) : (first ? 0u : 1u);
      const uint32_t want_idx = first ? i : i - (order ? n_d : n_p);
      if (w[7] != PS_MATTN_TS_MAGIC || region != want_region || idx != want_idx || en < st || core >= ncores) {
        std::printf("PSCTA: cta=%u BAD record magic=%x region=%u idx=%u core=%u start=%llu end=%llu\n", i, w[7], region,
                    idx, core, (unsigned long long)st, (unsigned long long)en);
        ts_ok = false;
        continue;
      }
      std::printf("PSCTA: cta=%u region=%c idx=%u core=%u start=%llu end=%llu\n", i, region ? 'd' : 'p', idx, core,
                  (unsigned long long)st, (unsigned long long)en);
      Reg& R = reg[region];
      ++R.ctas;
      R.busy += en - st;
      R.first = std::min(R.first, st);
      R.last = std::max(R.last, en);
      core_busy[region][core] += en - st;
      ++core_ctas[region][core];
      core_last[core] = std::max(core_last[core], en);
      t0 = std::min(t0, st);
      t1 = std::max(t1, en);
    }
    for (int r = 0; r < 2; ++r)
      if (reg[r].ctas)
        std::printf("PSREGION: region=%s ctas=%llu busy=%llu first_start=%llu last_end=%llu\n", r ? "decode" : "prefill",
                    (unsigned long long)reg[r].ctas, (unsigned long long)reg[r].busy, (unsigned long long)reg[r].first,
                    (unsigned long long)reg[r].last);
    for (uint32_t k = 0; k < ncores; ++k)
      std::printf("PSCORE: core=%u ctas_p=%u busy_p=%llu ctas_d=%u busy_d=%llu last_end=%llu\n", k, core_ctas[0][k],
                  (unsigned long long)core_busy[0][k], core_ctas[1][k], (unsigned long long)core_busy[1][k],
                  (unsigned long long)core_last[k]);
    std::printf("PSSPAN: first_start=%llu last_end=%llu ts_ok=%d\n", (unsigned long long)t0, (unsigned long long)t1,
                (int)ts_ok);
    ok = ok && ts_ok;
  }

  const bool within = (C == 0 || max_rel_p <= kPlanTol) && (n_seq == 0 || max_rel_d <= kPlanTol);
  std::printf("PSINFO: within_plan_tol=%d useful_flops_p=%.0f exec_flops_p=%.0f useful_flops_d=%.0f kv_bytes_p=%llu "
              "kv_bytes_d=%llu q_out_bytes=%llu tiles_p=%llu\n",
              (int)within, useful_p, exec_p, useful_d, (unsigned long long)(2ull * (s0 + C) * kv_dim * 4 * (C > 0)),
              (unsigned long long)(2ull * sum_ctx * kv_dim * 4), (unsigned long long)(2ull * (C + n_seq) * H_q * D * 4),
              (unsigned long long)tiles_p);
  if (!ok) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
