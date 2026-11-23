// main.cpp -- ps_prefill_attn host app: tiled causal prefill attention on Vortex, checked against a
// float64 host attention (and against the NumPy float32 `out` of a prefill attention fixture with -f).
//
// Usage: ps_prefill_attn [-f <attention_prefill fixture dir> [-S n] | -B <b> -H <H_q> -K <H_kv> -D <head_dim>
//                         -S <tokens> [-M <Tmax>] [-d uniform|neg] [-q <qscale>] [-s <seed>]] [-o <s0>]
//                        [-v lmem|dxa] [-p <depth>] [-Q <Qt>] [-t <Tk>] [-e <tol>] [-k <kernel.vxbin>]
//   synthetic: S query tokens at absolute positions s0 .. s0+S-1 (-o, default 0) attend causally to keys 0 .. s0+S-1.
//        q [B][S][H_q][D], K, V [B][Tmax][H_kv][D] from the deterministic xorshift RNG (ps_common.h), seed -s;
//        -d uniform: q, K, V uniform in [-1, 1); -d neg: q in [0.5, 1), K in [-1, -0.5), V in [-1, 1), so every logit
//        is negative (errata 7); -q multiplies q (logit magnitude). Cache rows s0+S .. Tmax-1 are NaN, so a kernel
//        that reads a key it must not see fails. Default Tmax = s0 + S + 3.
//   -f   ps-fixture-v1 `attention_prefill` fixture (model/make_prefill_fixtures.py; B = 1): q (N, dim), k_cache /
//        v_cache (N, kv_dim), NumPy out (N, dim) of N tokens of a real model. -S n uses the first n tokens (causal
//        attention of a prefix is the prefix of the output); -o s0 then launches only the queries s0 .. n-1 against
//        keys 0 .. n-1 (a chunked-prefill chunk).
//   -v   kernel (kernel.cpp): lmem (lmem_tiled, default) or dxa (dxa_dbuf; needs -DVX_CFG_EXT_DXA_ENABLE)
//   -p   DXA stages (dxa; 1, 2 or 4; default 2)
//   -C   PhaseShift DXA stream class of the K/V descriptors (dxa; ps patch dxa-class): VX_PS_CLASS_* value
//        written into META bits 15:13 (default 0 = unset: no extra DCR write, the reads keep the stock BULK_DXA tag)
//   -Q, -t  query tokens per CTA (Qt) and key positions per tile (Tk); default: see "Tiles"
//   -e   override the tolerance (default: derived per case, see "Error bound")
// Exactly one kernel launch per app run (methodology §2 consequence 5). Prints PSCHECK lines (ps_common.h), a
// PSINFO line (work and traffic counts, below), then PASSED!/FAILED!.
//
// LMEM and tiles: a CTA is NT*NW threads (all warps of one core), so one CTA is resident per core and its LMEM budget
// is the core's LMEM (VX_CAPS_LOCAL_MEM_SIZE, 16 KiB by default). Footprint (common.h ps_pattn_lmem), Rp = G*Qt
// rounded up to 4: 2 Rp D + Rp Tk + 3 Rp + 2 nbuf Tk D floats (nbuf = 1 lmem, depth dxa; regions rounded to 16).
// Default tiles (modeling choice, methodology §14.4): K/V reuse in LMEM is R = G*Qt (FLOPs per K/V element loaded),
// the per-tile fixed cost (5 barriers, the M1/M2 passes) is amortised over Tk keys. Rule: the largest Qt in
// {64, ..., 2, 1} with G*Qt <= 128 and Qt <= S whose largest fitting Tk (<= 32) is >= 8, with that Tk; if none
// reaches 8, Qt = 1 with its largest Tk; then Qt is halved while B*H_kv*ceil(S/Qt) < cores (keep every core busy).
//
// Error metric (methodology §2 consequence 6): rel_i = |o_i - ref_i| / S_i, S_i = sum_t p_t |v_t[d]| over the keys
// t <= p the row sees (the natural magnitude of o[i][h][d] = sum_t p_t v_t[d], p >= 0), as (§14.2).
// Error bound (modeling choice, methodology §14.4; the bound, independently re-derived, with the prefill
// chain lengths and, the rescale term): per output row (token at position p, head h; u = 2^-24,
// gamma(k) = k u / (1 - k u)),
//   rel_i <= 2 E_s + 2 (s_max - s_min) u + gamma(n_acc + n_l + 8 (n_tiles - 1) + 10),
//   E_s = max_{t<=p} [ gamma_D * scale * sum_d |q_d k_td| + 3 u |s_t| ],  n_acc = (p + 1) + n_tiles,  n_l = Tk + n_tiles,
// n_tiles = key tiles the row's CTA visits. E_s bounds each logit's absolute error (a D-long sequential FMA chain,
// plus fl(sqrtf(D)), its reciprocal and the multiply); a weight error eps_t moves o_i by at most
// sum_t p_t |eps_t| |v_t| + |o_i| |sum_t w_t eps_t| <= 2 max|eps| S_i, hence the factor 2 on E_s, on the rounding of
// s_t - m and on ps_expf's 4u (inside the +10). acc is ONE sequential FMA chain across all visited tiles (a = acc c,
// then nt FMAs per tile): p + 1 visible keys plus one rescale per tile; masked keys have p = 0 exactly and
// fma(0, v, a) = a adds no rounding. l sums each tile from 0 and folds it in: Tk + n_tiles. The +10 covers the exp
// terms and the final division. Rescale term (implemented): c_j multiplies acc and l alike
// but only the contributions of tiles < j, so its error (ps_expf, <= 4u) does not cancel in acc / l beyond one tile;
// it costs <= 2 * 4u per max-raising tile, 8u (n_tiles - 1) in total (methodology §14.4). Up to the host
// omitted it and this comment said the error cancels; the published bound was at most stricter, never looser.
// tol = max(1e-4, max over rows of this bound); every run must also meet the 1e-4 itself (PSINFO
// within_plan_tol, asserted by the tests). Against the fixture's NumPy float32 out, NumPy's own bound (same formula
// with n_acc + n_l = 2 (p + 1): einsum over t and the softmax sum) is added.
//
// PSINFO counts (for results/<run>): useful_flops = 4 D sum over (b, query, head) of (p + 1) (QK^T and PV of the
// visible keys: E4's causal 2 S^2 d for s0 = 0, up to the diagonal term); exec_flops = 4 D Rp nt summed over every
// visited (CTA, tile) (what the kernel computes: masked entries of diagonal tiles and padding rows included);
// kv_bytes = 2 B (s0+S) H_kv D 4 (K and V of the sequence once, the compulsory traffic); kv_tile_bytes = bytes of K/V
// copied into LMEM over all CTAs (lmem: nt rows per tile; dxa: min(Tk, s0+S-t0) rows); q_out_bytes = 2 B S H_q D 4;
// tiles = visited (CTA, tile) pairs, tiles_noskip = the count without causal tile skipping (every CTA visits
// ceil((s0+S)/Tk) tiles).
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <vortex.h>   // vortex2.h + vx_dcr_write / vx_check_occupancy
#include <dxa.h>

#include "common.h"
#include "ps_common.h"
#include "ps_fixture.h"
#include "ps_phase.h"     // -W: the [dcr_phase] block write
#include "ps_waymask.h"   // -W: mechanism A2 way masks (encoding, methodology §14.A)

namespace {

constexpr double kU = 1.0 / 16777216.0;   // 2^-24
constexpr double kPlanTol = 1e-4;

double gamma_k(double k) { return k * kU / (1.0 - k * kU); }

struct Ctx {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  vx_buffer_h q_buf = nullptr, k_buf = nullptr, v_buf = nullptr, o_buf = nullptr;
  ~Ctx() {
    if (!device) return;
    for (auto b : {q_buf, k_buf, v_buf, o_buf})
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
  std::printf("Usage: %s [-f fixture [-S n] | -B b -H Hq -K Hkv -D d -S s [-M tmax] [-d uniform|neg] [-q qscale] "
              "[-s seed]] [-o s0] [-v lmem|dxa] [-p depth] [-Q Qt] [-t Tk] [-e tol] [-C class] [-W l1:l2] [-k kernel.vxbin]\n", app);
}

// largest Tk <= 32 whose LMEM footprint fits `cap` bytes (0 if none)
uint32_t max_tk(uint32_t G, uint32_t D, uint32_t Qt, uint32_t nbuf, uint64_t cap) {
  for (uint32_t tk = 32; tk >= 1; --tk)
    if (ps_pattn_lmem(G, D, Qt, tk, nbuf).total * 4ull <= cap) return tk;
  return 0;
}

struct Geo {   // tile geometry of a launch (the kernel's own arithmetic, for the bound and the PSINFO counts)
  uint32_t S, s0, Qt, Tk, nqt;
  uint32_t kend(uint32_t qt) const { uint32_t q0 = qt * Qt, nq = std::min(Qt, S - q0); return s0 + q0 + nq; }
  uint32_t ntiles(uint32_t qt) const { return (kend(qt) + Tk - 1) / Tk; }
};

struct Ref {
  std::vector<double> out, scale;   // [B][S][H_q][D]
  double bound = 0.0;               // max over rows of the error bound (header comment)
  double smin = INFINITY, smax = -INFINITY;   // range of the visible logits (printed: the deep-negative test checks it)
};

// float64 causal attention + S_i + the per-row error bound. numpy = true: NumPy's chains 2 (p + 1) instead of the
// kernel's (p + 1 + n_tiles) + (Tk + n_tiles) + 8 (n_tiles - 1) (the last term, header comment).
Ref reference(const std::vector<float>& q, const std::vector<float>& K, const std::vector<float>& V, uint32_t B,
              uint32_t H_q, uint32_t H_kv, uint32_t D, uint32_t Tmax, const Geo& g, bool numpy) {
  Ref r;
  const size_t NO = (size_t)B * g.S * H_q * D;
  r.out.assign(NO, 0.0);
  r.scale.assign(NO, 0.0);
  const uint32_t G = H_q / H_kv;
  const double sc = 1.0 / std::sqrt((double)D);
  const uint32_t T_all = g.s0 + g.S;
  std::vector<double> s(T_all), p(T_all);
  for (uint32_t b = 0; b < B; ++b)
    for (uint32_t i = 0; i < g.S; ++i) {
      const uint32_t pos = g.s0 + i, T = pos + 1, nt = g.ntiles(i / g.Qt);
      const double chain = numpy ? 2.0 * T
                                 : (double)(T + nt) + (double)(g.Tk + nt) + 8.0 * (nt > 0 ? nt - 1.0 : 0.0);
      for (uint32_t h = 0; h < H_q; ++h) {
        const uint32_t kvh = h / G;
        const float* qh = &q[(((size_t)b * g.S + i) * H_q + h) * D];
        double es = 0.0, smax = -INFINITY, smin = INFINITY;
        for (uint32_t t = 0; t < T; ++t) {
          const float* kt = &K[(((size_t)b * Tmax + t) * H_kv + kvh) * D];
          double dot = 0.0, mag = 0.0;
          for (uint32_t d = 0; d < D; ++d) { dot += (double)qh[d] * kt[d]; mag += std::fabs((double)qh[d] * kt[d]); }
          s[t] = dot * sc;
          es = std::fmax(es, gamma_k(D) * mag * sc + 3 * kU * std::fabs(s[t]));
          smax = std::fmax(smax, s[t]);
          smin = std::fmin(smin, s[t]);
        }
        double l = 0.0;
        for (uint32_t t = 0; t < T; ++t) { p[t] = std::exp(s[t] - smax); l += p[t]; }
        for (uint32_t t = 0; t < T; ++t) p[t] /= l;
        const size_t ob = (((size_t)b * g.S + i) * H_q + h) * D;
        for (uint32_t d = 0; d < D; ++d) {
          double o = 0.0, a = 0.0;
          for (uint32_t t = 0; t < T; ++t) {
            const double vt = V[(((size_t)b * Tmax + t) * H_kv + kvh) * D + d];
            o += p[t] * vt;
            a += p[t] * std::fabs(vt);
          }
          r.out[ob + d] = o;
          r.scale[ob + d] = a;
        }
        r.bound = std::fmax(r.bound, 2 * es + 2 * (smax - smin) * kU + gamma_k(chain + 10));
        r.smin = std::fmin(r.smin, smin);
        r.smax = std::fmax(r.smax, smax);
      }
    }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixture, variant = "lmem", dist = "uniform", kernel_file = "kernel.vxbin";
  uint32_t B = 1, H_q = 8, H_kv = 4, D = 8, S = 16, s0 = 0, Tmax = 0, seed = 1, depth = 2, Qt = 0, Tk = 0;
  bool S_given = false;
  double qscale = 1.0, tol_override = 0.0;
  bool ps_have_wm = false;   // PhaseShift: -W way masks (absent = no DCR write, the old host path)
  uint32_t ps_wm_l1 = 0, ps_wm_l2 = 0;
  int c;
  uint32_t ps_dxa_class = 0;   // PhaseShift (patch dxa-class): -C
  while ((c = getopt(argc, argv, "f:B:H:K:D:S:o:M:d:q:s:v:p:Q:t:e:C:k:W:h")) != -1) {
    switch (c) {
      case 'f': fixture = optarg; break;
      case 'B': B = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'H': H_q = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'K': H_kv = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'D': D = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'S': S = (uint32_t)std::strtoul(optarg, nullptr, 10); S_given = true; break;
      case 'o': s0 = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'M': Tmax = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'd': dist = optarg; break;
      case 'q': qscale = std::strtod(optarg, nullptr); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'v': variant = optarg; break;
      case 'p': depth = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'Q': Qt = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 't': Tk = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'e': tol_override = std::strtod(optarg, nullptr); break;
      case 'k': kernel_file = optarg; break;
      case 'C': ps_dxa_class = (uint32_t)std::strtoul(optarg, nullptr, 0); break;
      case 'W':   // PhaseShift: [dcr_phase] way masks "<l1word>:<l2word>" (ps_waymask.h, methodology §14.A)
        if (!ps::parse_waymask_arg(optarg, &ps_wm_l1, &ps_wm_l2)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        ps_have_wm = true;
        break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  if (variant != "lmem" && variant != "dxa") { std::printf("bad -v %s\nFAILED!\n", variant.c_str()); return 1; }
  if (dist != "uniform" && dist != "neg") { std::printf("bad -d %s\nFAILED!\n", dist.c_str()); return 1; }
  if (depth != 1 && depth != 2 && depth != 4) { std::printf("-p must be 1, 2 or 4\nFAILED!\n"); return 1; }
  const bool dxa = (variant == "dxa");

  // ---- inputs
  std::vector<float> q, K, V, out_fx;
  std::string label;
  try {
    if (!fixture.empty()) {
      ps::Fixture fx(fixture);
      if (fx.op() != "attention_prefill") throw ps::FixtureError("need an attention_prefill fixture");
      const ps::Json& p = fx.meta().at("params");
      H_q = (uint32_t)p.at("n_heads").as_num();
      H_kv = (uint32_t)p.at("n_kv_heads").as_num();
      D = (uint32_t)p.at("head_size").as_num();
      ps::Tensor tq = fx.tensor("q"), tk = fx.tensor("k_cache"), tv = fx.tensor("v_cache"), to = fx.tensor("out");
      const uint32_t N = tq.shape.size() == 2 ? tq.dim(0) : 0;
      if (N == 0 || tq.dim(1) != H_q * D || tk.shape.size() != 2 || tk.dim(0) != N || tk.dim(1) != H_kv * D ||
          tk.shape != tv.shape || to.shape != tq.shape)
        throw ps::FixtureError("attention_prefill fixture shapes do not match its params");
      const uint32_t n = S_given ? S : N;   // prefix of n tokens
      if (n == 0 || n > N || s0 >= n) throw ps::FixtureError("need 0 <= s0 < S <= fixture tokens");
      B = 1;
      S = n - s0;
      Tmax = n;
      const size_t qd = (size_t)H_q * D, kd = (size_t)H_kv * D;
      q.assign(tq.data.begin() + s0 * qd, tq.data.begin() + n * qd);
      out_fx.assign(to.data.begin() + s0 * qd, to.data.begin() + n * qd);
      K.assign(tk.data.begin(), tk.data.begin() + n * kd);
      V.assign(tv.data.begin(), tv.data.begin() + n * kd);
      label = fx.model() + ":attention_prefill:L" + std::to_string((int)fx.meta().at("layer").as_num()) + "_N" +
              std::to_string(N) + "_S" + std::to_string(n);
    } else {
      if (Tmax == 0) Tmax = s0 + S + 3;
      if (B == 0 || H_q == 0 || H_kv == 0 || D == 0 || S == 0 || Tmax < s0 + S || H_q % H_kv != 0)
        throw ps::FixtureError("need B, H_q, H_kv, D, S >= 1, Tmax >= s0 + S, H_kv | H_q");
      ps::Rng rng(seed);
      q.resize((size_t)B * S * H_q * D);
      K.assign((size_t)B * Tmax * H_kv * D, std::nanf(""));
      V.assign(K.size(), std::nanf(""));
      const bool neg = (dist == "neg");
      for (auto& x : q) x = (float)(qscale * (neg ? 0.75 + 0.25 * rng.uniform() : rng.uniform()));
      for (uint32_t b = 0; b < B; ++b)
        for (uint32_t t = 0; t < s0 + S; ++t)
          for (uint32_t i = 0; i < H_kv * D; ++i) {
            const size_t idx = ((size_t)b * Tmax + t) * H_kv * D + i;
            K[idx] = neg ? -0.75f + 0.25f * rng.uniform() : rng.uniform();
            V[idx] = rng.uniform();
          }
      char buf[128];
      std::snprintf(buf, sizeof(buf), "synthetic:%s:q%g:s%u", dist.c_str(), qscale, seed);
      label = buf;
    }
  } catch (const std::exception& e) {
    std::printf("input error: %s\nFAILED!\n", e.what());
    return 1;
  }
  const uint32_t G = H_q / H_kv;
  if (D % 4 != 0) { std::printf("head_dim %u must be a multiple of 4 (4 x 4 register blocks)\nFAILED!\n", D); return 1; }

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
  uint64_t nt = 0, nw = 0, isa = 0, lmem_cap = 0, ncores = 0;
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_WARPS, &nw));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_ISA_FLAGS, &isa));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_CORES, &ncores));
  if (dxa && !(isa & VX_ISA_EXT_DXA)) {
    std::printf("variant dxa needs a device built with -DVX_CFG_EXT_DXA_ENABLE\nFAILED!\n");
    return 1;
  }
  const uint32_t nth = (uint32_t)(nt * nw);   // one CTA = all warps of a core
  const uint32_t nbuf = dxa ? depth : 1;

  // ---- tiles (header comment "LMEM and tiles")
  if (Qt == 0) {
    uint32_t best = 0, best_tk = 0;
    for (uint32_t cand = 64; cand >= 1; cand >>= 1) {
      if (cand > S && cand > 1) continue;
      if (G * cand > 128) continue;
      const uint32_t tk = max_tk(G, D, cand, nbuf, lmem_cap);
      if (tk >= 8) { best = cand; best_tk = tk; break; }
    }
    if (best == 0) { best = 1; best_tk = max_tk(G, D, 1, nbuf, lmem_cap); }
    while (best > 1 && (uint64_t)B * H_kv * ((S + best - 1) / best) < ncores) {
      best >>= 1;
      best_tk = max_tk(G, D, best, nbuf, lmem_cap);
    }
    Qt = best;
    if (Tk == 0) Tk = best_tk;
  }
  if (Tk == 0) Tk = std::min<uint32_t>(32, max_tk(G, D, Qt, nbuf, lmem_cap));
  if (Qt == 0 || Tk == 0) { std::printf("no tile fits the LMEM (%llu B)\nFAILED!\n", (unsigned long long)lmem_cap); return 1; }
  const ps_pattn_lmem_t L = ps_pattn_lmem(G, D, Qt, Tk, nbuf);
  const uint32_t lmem_bytes = L.total * 4;
  if (lmem_bytes > lmem_cap) {
    std::printf("LMEM footprint %u B > %llu B per core (G=%u D=%u Qt=%u Tk=%u nbuf=%u)\nFAILED!\n", lmem_bytes,
                (unsigned long long)lmem_cap, G, D, Qt, Tk, nbuf);
    return 1;
  }
  Geo geo{S, s0, Qt, Tk, (S + Qt - 1) / Qt};

  // ---- work / traffic counts (PSINFO)
  double useful_flops = 0, exec_flops = 0, kv_tile_bytes = 0;
  uint64_t tiles = 0, tiles_noskip = 0;
  for (uint32_t i = 0; i < S; ++i) useful_flops += 4.0 * D * H_q * B * (double)(s0 + i + 1);
  const uint32_t Tkv = s0 + S;
  for (uint32_t qt = 0; qt < geo.nqt; ++qt) {
    const uint32_t ke = geo.kend(qt), ntl = geo.ntiles(qt);
    for (uint32_t j = 0; j < ntl; ++j) {
      const uint32_t t0 = j * Tk, ntv = std::min(Tk, ke - t0);
      exec_flops += 4.0 * L.rows * ntv * D;
      kv_tile_bytes += 2.0 * (dxa ? std::min(Tk, Tkv - t0) : ntv) * D * 4;
    }
    tiles += ntl;
    tiles_noskip += (Tkv + Tk - 1) / Tk;
  }
  const double heads = (double)B * H_kv;
  exec_flops *= heads; kv_tile_bytes *= heads; tiles *= (uint64_t)heads; tiles_noskip *= (uint64_t)heads;

  // ---- reference and tolerance
  Ref ref = reference(q, K, V, B, H_q, H_kv, D, Tmax, geo, false);
  const double tol = tol_override > 0 ? tol_override : std::fmax(kPlanTol, ref.bound);
  const size_t NO = (size_t)B * S * H_q * D;
  const uint32_t grid = B * H_kv * geo.nqt;
  uint32_t ntiles_max = 0;
  for (uint32_t qt = 0; qt < geo.nqt; ++qt) ntiles_max = std::max(ntiles_max, geo.ntiles(qt));
  std::printf("ps_prefill_attn: %s B=%u H_q=%u H_kv=%u D=%u S=%u s0=%u Tmax=%u variant=%s G=%u Qt=%u Tk=%u depth=%u "
              "rows=%u nqt=%u ntiles_max=%u grid=%u NT=%u NW=%u cores=%u lmem=%u/%llu logits=[%.1f,%.1f] bound=%.3e tol=%.3e\n",
              label.c_str(), B, H_q, H_kv, D, S, s0, Tmax, variant.c_str(), G, Qt, Tk, dxa ? depth : 0, L.rows,
              geo.nqt, ntiles_max, grid, (unsigned)nt, (unsigned)nw, (unsigned)ncores, lmem_bytes,
              (unsigned long long)lmem_cap, ref.smin, ref.smax, ref.bound, tol);
  bool ok = true;
  double tol_fx = 0.0;
  if (!out_fx.empty()) {
    Ref rnp = reference(q, K, V, B, H_q, H_kv, D, Tmax, geo, true);
    auto st = ps::compare(out_fx.data(), ref.out.data(), ref.scale.data(), NO, std::fmax(kPlanTol, rnp.bound));
    ps::print_check("fixture_vs_f64", st);
    ok = ok && st.ok();
    tol_fx = tol_override > 0 ? tol_override : std::fmax(kPlanTol, ref.bound + rnp.bound);
  }

  // ---- buffers
  ps_pattn_arg_t arg = {};
  arg.B = B; arg.H_q = H_q; arg.H_kv = H_kv; arg.D = D; arg.S = S; arg.s0 = s0; arg.Tmax = Tmax;
  arg.G = G; arg.Qt = Qt; arg.Tk = Tk; arg.depth = depth; arg.nqt = geo.nqt;
  arg.scale = 1.0f / sqrtf((float)D);
  const uint64_t qb = (uint64_t)q.size() * 4, kb = (uint64_t)K.size() * 4, ob = NO * 4;
  PS_RT_CHECK(vx_buffer_create(ctx.device, qb, VX_MEM_READ, &ctx.q_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.q_buf, &arg.q_addr));
  PS_RT_CHECK(vx_buffer_create(ctx.device, kb, VX_MEM_READ, &ctx.k_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.k_buf, &arg.k_addr));
  PS_RT_CHECK(vx_buffer_create(ctx.device, kb, VX_MEM_READ, &ctx.v_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.v_buf, &arg.v_addr));
  PS_RT_CHECK(vx_buffer_create(ctx.device, ob, VX_MEM_READ_WRITE, &ctx.o_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.o_buf, &arg.o_addr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.q_buf, 0, q.data(), qb, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.k_buf, 0, K.data(), kb, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.v_buf, 0, V.data(), kb, 0, nullptr, nullptr));
  const std::vector<float> poison(NO, std::nanf(""));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.o_buf, 0, poison.data(), ob, 0, nullptr, nullptr));
  if (dxa) {
    // 4-D descriptors over [B][Tmax][H_kv][D]: dim0 = d, dim1 = kv head, dim2 = position (extent s0 + S: rows
    // beyond are out of range and zero-filled), dim3 = sequence; tile = one head's D x Tk block -> LMEM rows [Tk][D]
    for (auto [slot, base] : {std::pair<uint32_t, uint64_t>{PS_PATTN_DESC_K, arg.k_addr}, {PS_PATTN_DESC_V, arg.v_addr}})
      PS_RT_CHECK(vortex::dxa::program_4d(ctx.device, slot, base, D, H_kv, Tkv, B, D * 4, H_kv * D * 4,
                                          Tmax * H_kv * D * 4, D, 1, Tk, 1, 4));
    if (ps_dxa_class != 0) {   // PhaseShift (patch dxa-class): re-pack META with the stream class (row-major, rank 4)
#ifndef VX_PS_NUM_CLASSES   // the base tree has no class field / ps_set_stream (host builds there must compile)
      std::printf("-C needs the ps tree (patch dxa-class)\nFAILED!\n");
      return 1;
#else
      if (ps_dxa_class >= VX_PS_NUM_CLASSES) { std::printf("bad -C %u\nFAILED!\n", ps_dxa_class); return 1; }
      for (uint32_t slot : {(uint32_t)PS_PATTN_DESC_K, (uint32_t)PS_PATTN_DESC_V})
        PS_RT_CHECK(vortex::dxa::ps_set_stream(ctx.device, slot, 4, 4, vortex::dxa::Layout::RowMajor, ps_dxa_class, false));
      std::printf("PSDXASTREAM: kv_class=%u\n", ps_dxa_class);
#endif
    }
  }
  PS_RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
  const char* kname = dxa ? "attn_prefill_dxa" : "attn_prefill_lmem";
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, kname, &ctx.kernel));
  PS_RT_CHECK(vx_check_occupancy(ctx.device, nth, lmem_bytes));

  std::printf("launch: kernel=%s grid=%u block=%u lmem_bytes=%u\n", kname, grid, nth, lmem_bytes);
  std::vector<float> O(NO);
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
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  PS_RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &launch_ev));
  PS_RT_CHECK(vx_enqueue_read(ctx.queue, O.data(), ctx.o_buf, 0, ob, 1, &launch_ev, &read_ev));
  PS_RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  std::printf("Elapsed: %.0f ms\n", timer.ms());

  auto s64 = ps::compare(O.data(), ref.out.data(), ref.scale.data(), NO, tol);
  ps::print_check("launch0:vs_f64", s64);
  ok = ok && s64.ok();
  if (!out_fx.empty()) {
    std::vector<double> rfx(out_fx.begin(), out_fx.end());
    auto st = ps::compare(O.data(), rfx.data(), ref.scale.data(), NO, tol_fx);
    ps::print_check("launch0:vs_fixture", st);
    ok = ok && st.ok();
  }
  std::printf("PSINFO: within_plan_tol=%d useful_flops=%.0f exec_flops=%.0f kv_bytes=%llu kv_tile_bytes=%.0f "
              "q_out_bytes=%llu tiles=%llu tiles_noskip=%llu\n",
              (int)(s64.max_rel <= kPlanTol), useful_flops, exec_flops,
              (unsigned long long)(2ull * B * Tkv * H_kv * D * 4), kv_tile_bytes, (unsigned long long)(qb + ob),
              (unsigned long long)tiles, (unsigned long long)tiles_noskip);
  if (!ok) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
