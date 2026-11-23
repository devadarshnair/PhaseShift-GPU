// main.cpp -- ps_decode_attn_gqa host app: fused GQA decode attention on Vortex, checked against a
// float64 host attention (and against the NumPy float32 `out` of a attention fixture with -f).
//
// Usage: ps_decode_attn_gqa [-f <attention fixture dir> | -B <b> -H <H_q> -K <H_kv> -D <head_dim> -T <ctx>
//                            [-M <Tmax>] [-d uniform|neg] [-q <qscale>] [-s <seed>]]
//                           [-v naive|gqa|dxa] [-p <depth>] [-t <Tt>] [-e <tol>] [-k <kernel.vxbin>]
//   -f   ps-fixture-v1 `attention` fixture (B = 1): q (dim), k_cache / v_cache (T, kv_dim), NumPy out (dim)
//   synthetic: q [B][H_q][D], K, V [B][Tmax][H_kv][D] from the deterministic xorshift RNG (ps_common.h), seed -s;
//        -d uniform: q, K, V uniform in [-1, 1); -d neg: q in [0.5, 1), K in [-1, -0.5), V in [-1, 1), so every
//        logit is negative (the softmax max-initialisation bug case, errata 7 /); -q multiplies q (logit
//        magnitude). Cache rows T..Tmax-1 are NaN, so a kernel that reads past T fails. Default Tmax = T + 3.
//   -v   kernel (kernel.cpp): naive (naive_per_qhead), gqa (gqa_shared_kv, default), dxa (gqa_dxa_dbuf; needs a
//        device built with -DVX_CFG_EXT_DXA_ENABLE); (kernel_kv.cpp, methodology §18.9): gqa_kv8, dxa_kv8 (int8 K / V + one fp32
//        scale per (position, KV head); the host quantizes the fp32 K / V it generated, runq.c's quantize with group = D) and
//        gqa_kv16, dxa_kv16 (fp16 K / V, host round-to-nearest-even; needs a device built with -DVX_CFG_EXT_ZFHMIN_ENABLE, ISA flag
//        VX_ISA_EXT_ZFHMIN). The reference and the error bound are those of the fp32 kernels applied to the DEQUANTIZED operands the
//        kernel reads (device operand == host operand bit for bit, header comment "Error bound"); the error of the cache itself against the
//        fp32-KV attention is printed as PSKVQ (information, not pass / fail). PSKV prints the KV bytes of the step incl. the scales.
//   -p   DXA stages (dxa; 1, 2 or 4; default 2)
//   -C   PhaseShift DXA stream class of the K/V descriptors (dxa; ps patch dxa-class): VX_PS_CLASS_* value
//        written into META bits 15:13 (default 0 = unset: no extra DCR write, the reads keep the stock BULK_DXA tag)
//   -t   key positions per tile (default: 32 for naive/gqa; for dxa the largest Tt <= 32 whose LMEM footprint fits
//        the per-CTA LMEM budget, see "LMEM")
//   -e   override the tolerance (default: derived per case, see "Error bound")
//   -W   PhaseShift [dcr_phase] way-mask words <l1word>:<l2word> (mechanism A2, ps patch waymask, encoding in
//        kernels/common/ps_waymask.h, methodology §14.A); absent = no change. Written with -R's block, or alone.
//   -R   PhaseShift [dcr_phase] KV range (ps patch mem-class; default none = no DCR write, the validated
//        behaviour): kv = program kv[0] = the K..V span (one range from the lower to the upper end of the two cache
//        buffers; refused if q or out lies inside it) so SimX classifies unhinted K/V traffic KV_STREAM; kvacct = the
//        same range with VX_PHASE_FLAG_RANGE_ACCT_ONLY (the range only labels DRAM bytes for PS_DEBUG_CLASS, so a
//        PS_HINTS=1 build is measured on its hints alone). Written with ps::phase_enqueue on the launch queue. The
//        base tree ignores the DCR writes.
// PS_HINTS=1 builds (CONFIGS -DPS_HINTS=1; kernels/common/ps_hint.h): the kernel tags its K/V loads KV_STREAM and marks
// one DECODE_TILE per (CTA, key tile); the PSHINT line prints the expected per-type mark counts (grid * ntiles).
// Exactly one kernel launch per app run (methodology §2 consequence 5). Prints PSCHECK lines (ps_common.h),
// a PSINFO line with the E3 KV traffic of the step, then PASSED!/FAILED!.
//
// LMEM: CTA = NT*NW threads (all warps of one core), so one CTA is resident per core and its LMEM budget is the
// whole core LMEM (VX_CAPS_LOCAL_MEM_SIZE, 16 KiB by default). Footprint (common.h ps_attn_lmem): 2 G D + G Tt + 3 G
// floats, plus 2 depth Tt D for dxa (each region rounded to 16 floats).
//
// Error metric (methodology §2 consequence 6, ps_common.h): rel_i = |o_i - ref_i| / S_i with
// S_i = sum_t p_t |v_t[d]| (the natural magnitude of the output element o[h][d] = sum_t p_t v_t[d], p >= 0).
// Error bound (modeling choice, methodology §14.2; corrected, and, the
// the earlier rescale term): per query head, with u = 2^-24 and gamma(k) = k u / (1 - k u),
//   rel_i <= 2 E_s + 2 (s_max - s_min) u + gamma(n_acc + n_l + 8 (n_tiles - 1) + 10),
//   E_s = max_t [ gamma_D * scale * sum_d |q_d k_td| + 3 u |s_t| ],  n_acc = T + n_tiles,  n_l = Tt + n_tiles.
// E_s bounds the absolute error of each logit s_t (a D-long float32 dot product: gamma_D times the sum of |terms|;
// plus fl(sqrtf(D)), its reciprocal and the multiply, 3u |s_t|). An error eps_t in the weight p_t (exponent error,
// or the exp's own error) moves o_i by at most sum_t p_t |eps_t| |v_t| + |o_i| |sum_t w_t eps_t| <= 2 max|eps| S_i
// (numerator and denominator; |o_i| <= S_i), hence the factor 2 on E_s, on the rounding of s_t - m (|s_t - m| u, at
// most (s_max - s_min) u) and on ps_expf's 4u (inside the +10). The summation chains: acc is ONE sequential FMA chain
// across all tiles (kernel.cpp phase V: a = acc * c, then nt FMAs per tile), so the first key's term passes through
// n_acc = T + n_tiles roundings; l sums each tile from 0 and then folds it in (l = l c + sum): n_l = Tt + n_tiles.
// The +10 also covers the final division (used gamma(2 (Tt + 2 n_tiles) + 10), which undercounted the acc
// chain, and 1x (s_max - s_min) u). Rescale term: the factor c_j = exp(m_j - m_{j+1}) of tile j multiplies
// acc and l alike, but only the contributions of tiles < j, so its error (ps_expf, <= 4u) does NOT cancel in acc / l
// for more than one tile: it perturbs a subset of the weights, which by the weight argument above costs <= 2 * 4u per
// max-raising tile, <= 8u (n_tiles - 1) in total (the rounding of c's argument telescopes into the 2 (s_max - s_min) u
// term, and ps_expf(0) = 1 exactly when the max does not change). Up to the host omitted this term (and this
// comment claimed the error cancels), so the published bound was at most stricter than intended, never looser.
// tol = max(1e-4, max over heads of this bound), and every run must also meet the 1e-4 itself
// (PSINFO within_plan_tol, asserted by the tests). The comparison against the fixture's NumPy float32 out adds
// NumPy's own bound (same formula with n_acc + n_l = 2 T: einsum and the softmax sum over t) to it.
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
#include "common_kv.h"
#include "f16_host.h"
#include "ps_common.h"
#include "ps_q80.h"
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
  vx_buffer_h q_buf = nullptr, k_buf = nullptr, v_buf = nullptr, o_buf = nullptr, ks_buf = nullptr, vs_buf = nullptr;
  ~Ctx() {
    if (!device) return;
    for (auto b : {q_buf, k_buf, v_buf, o_buf, ks_buf, vs_buf})
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
  std::printf("Usage: %s [-f fixture | -B b -H Hq -K Hkv -D d -T t [-M tmax] [-d uniform|neg] [-q qscale] [-s seed]] "
              "[-v naive|gqa|dxa] [-p depth] [-t Tt] [-e tol] [-R none|kv|kvacct] [-C class] [-W l1:l2] [-k kernel.vxbin]\n", app);
}

struct Ref {
  std::vector<double> out, scale;   // [B][H_q][D]
  double bound = 0.0;               // max over heads of the error bound (header comment)
};

// float64 attention + S_i + the per-head error bound for a device kernel with tile Tt
Ref reference(const std::vector<float>& q, const std::vector<float>& K, const std::vector<float>& V, uint32_t B,
              uint32_t H_q, uint32_t H_kv, uint32_t D, uint32_t T, uint32_t Tmax,
              uint32_t chain_terms /* n_acc + n_l + 8 (n_tiles - 1) */) {
  Ref r;
  r.out.assign((size_t)B * H_q * D, 0.0);
  r.scale.assign((size_t)B * H_q * D, 0.0);
  const uint32_t kv_mul = H_q / H_kv;
  const double sc = 1.0 / std::sqrt((double)D);
  std::vector<double> s(T), p(T);
  for (uint32_t b = 0; b < B; ++b)
    for (uint32_t h = 0; h < H_q; ++h) {
      const uint32_t kvh = h / kv_mul;
      const float* qh = &q[((size_t)b * H_q + h) * D];
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
      for (uint32_t d = 0; d < D; ++d) {
        double o = 0.0, a = 0.0;
        for (uint32_t t = 0; t < T; ++t) {
          const double vt = V[(((size_t)b * Tmax + t) * H_kv + kvh) * D + d];
          o += p[t] * vt;
          a += p[t] * std::fabs(vt);
        }
        r.out[((size_t)b * H_q + h) * D + d] = o;
        r.scale[((size_t)b * H_q + h) * D + d] = a;
      }
      r.bound = std::fmax(r.bound, 2 * es + 2 * (smax - smin) * kU + gamma_k((double)chain_terms + 10));
    }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixture, variant = "gqa", dist = "uniform", kernel_file = "kernel.vxbin", range_mode = "none";
  uint32_t B = 1, H_q = 8, H_kv = 4, D = 8, T = 17, Tmax = 0, seed = 1, depth = 2, Tt = 0;
  bool have_wm = false;   // PhaseShift (patch waymask)
  uint32_t wm_l1 = 0, wm_l2 = 0;
  double qscale = 1.0, tol_override = 0.0;
  int c;
  uint32_t ps_dxa_class = 0;   // PhaseShift (patch dxa-class): -C
  while ((c = getopt(argc, argv, "f:B:H:K:D:T:M:d:q:s:v:p:t:e:R:C:k:W:h")) != -1) {
    switch (c) {
      case 'f': fixture = optarg; break;
      case 'B': B = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'H': H_q = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'K': H_kv = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'D': D = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'T': T = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'M': Tmax = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'd': dist = optarg; break;
      case 'q': qscale = std::strtod(optarg, nullptr); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'v': variant = optarg; break;
      case 'p': depth = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 't': Tt = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'W':   // PhaseShift (patch waymask): [dcr_phase] way masks "<l1word>:<l2word>" (ps_waymask.h)
        if (!ps::parse_waymask_arg(optarg, &wm_l1, &wm_l2)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        have_wm = true;
        break;
      case 'e': tol_override = std::strtod(optarg, nullptr); break;
      case 'R': range_mode = optarg; break;
      case 'k': kernel_file = optarg; break;
      case 'C': ps_dxa_class = (uint32_t)std::strtoul(optarg, nullptr, 0); break;
      case 'h': usage(argv[0]); return 0;
      default: usage(argv[0]); return 1;
    }
  }
  if (variant != "naive" && variant != "gqa" && variant != "dxa" && variant != "gqa_kv8" && variant != "dxa_kv8" &&
      variant != "gqa_kv16" && variant != "dxa_kv16") { std::printf("bad -v %s\nFAILED!\n", variant.c_str()); return 1; }
  if (dist != "uniform" && dist != "neg") { std::printf("bad -d %s\nFAILED!\n", dist.c_str()); return 1; }
  if (depth != 1 && depth != 2 && depth != 4) { std::printf("-p must be 1, 2 or 4\nFAILED!\n"); return 1; }
  if (range_mode != "none" && range_mode != "kv" && range_mode != "kvacct") {
    std::printf("bad -R %s\nFAILED!\n", range_mode.c_str());
    return 1;
  }

  // ---- inputs
  std::vector<float> q, K, V, out_fx;
  std::vector<float> Kd, Vd;           // the dequantized operands the kv8 / kv16 kernels read (== K, V for the fp32 kernels)
  std::vector<int8_t> Kq, Vq;          // kv8 caches
  std::vector<uint16_t> Kh, Vh;        // kv16 caches (fp16 bits)
  std::vector<float> Ks, Vs;           // kv8 scales [B][H_kv][Tmax]
  std::string label;
  try {
    if (!fixture.empty()) {
      ps::Fixture fx(fixture);
      if (fx.op() != "attention") throw ps::FixtureError("need an attention fixture");
      const ps::Json& p = fx.meta().at("params");
      H_q = (uint32_t)p.at("n_heads").as_num();
      H_kv = (uint32_t)p.at("n_kv_heads").as_num();
      D = (uint32_t)p.at("head_size").as_num();
      ps::Tensor tq = fx.tensor("q"), tk = fx.tensor("k_cache"), tv = fx.tensor("v_cache"), to = fx.tensor("out");
      if (tq.numel() != (size_t)H_q * D || tk.shape.size() != 2 || tk.dim(1) != H_kv * D || tk.shape != tv.shape ||
          to.numel() != (size_t)H_q * D)
        throw ps::FixtureError("attention fixture shapes do not match its params");
      B = 1;
      T = tk.dim(0);
      Tmax = T;
      q = std::move(tq.data);
      K = std::move(tk.data);
      V = std::move(tv.data);
      out_fx = std::move(to.data);
      label = fx.model() + ":attention:L" + std::to_string((int)fx.meta().at("layer").as_num()) + "_P" +
              std::to_string((int)fx.meta().at("pos").as_num());
    } else {
      if (Tmax == 0) Tmax = T + 3;
      if (B == 0 || H_q == 0 || H_kv == 0 || D == 0 || T == 0 || Tmax < T || H_q % H_kv != 0)
        throw ps::FixtureError("need B, H_q, H_kv, D, T >= 1, Tmax >= T, H_kv | H_q");
      ps::Rng rng(seed);
      q.resize((size_t)B * H_q * D);
      K.assign((size_t)B * Tmax * H_kv * D, std::nanf(""));
      V.assign(K.size(), std::nanf(""));
      const bool neg = (dist == "neg");
      for (auto& x : q) x = (float)(qscale * (neg ? 0.75 + 0.25 * rng.uniform() : rng.uniform()));
      for (uint32_t b = 0; b < B; ++b)
        for (uint32_t t = 0; t < T; ++t)
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
  const uint32_t kv_mul = H_q / H_kv;
  const uint32_t G = (variant == "naive") ? 1 : kv_mul;
  if (G != 1 && G != 2 && G != 4 && G != 8) { std::printf("group size %u not supported (1, 2, 4, 8)\nFAILED!\n", G); return 1; }
  const bool dxa = (variant == "dxa" || variant == "dxa_kv8" || variant == "dxa_kv16");
  const uint32_t kvbits = (variant == "gqa_kv8" || variant == "dxa_kv8") ? 8u : (variant == "gqa_kv16" || variant == "dxa_kv16") ? 16u : 32u;
  const bool kvq = kvbits != 32u;
  const uint32_t esz = kvbits / 8u;   // bytes per cached element (4 for the fp32 kernels)
  if (kvq && (D % 4 != 0)) { std::printf("kv8 / kv16 kernels need D %% 4 == 0\nFAILED!\n"); return 1; }

  // ---- device
  Ctx ctx;
  PS_RT_CHECK(vx_device_open(0, &ctx.device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  PS_RT_CHECK(vx_queue_create(ctx.device, &qi, &ctx.queue));
  uint64_t nt = 0, nw = 0, isa = 0, lmem_cap = 0;
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_WARPS, &nw));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_ISA_FLAGS, &isa));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  if (kvbits == 16 && !(isa & VX_ISA_EXT_ZFHMIN)) {
    std::printf("variant %s needs a device built with -DVX_CFG_EXT_ZFHMIN_ENABLE\nFAILED!\n", variant.c_str());
    return 1;
  }
  if (dxa && !(isa & VX_ISA_EXT_DXA)) {
    std::printf("variant dxa needs a device built with -DVX_CFG_EXT_DXA_ENABLE\nFAILED!\n");
    return 1;
  }
  const uint32_t nth = (uint32_t)(nt * nw);   // one CTA = all warps of a core
  if (Tt == 0) {
    Tt = 32;
    if (dxa)
      while (Tt > 1 && (kvq ? ps_attn_kv_lmem(G, D, Tt, depth, 1, esz).total : ps_attn_lmem(G, D, Tt, depth, 1).total) * 4ull > lmem_cap) --Tt;
  }
  const uint32_t lmem_words = kvq ? ps_attn_kv_lmem(G, D, Tt, depth, dxa, esz).total : ps_attn_lmem(G, D, Tt, depth, dxa).total;
  const uint32_t lmem_bytes = lmem_words * 4;
  if (lmem_bytes > lmem_cap) {
    std::printf("LMEM footprint %u B > %llu B per core (G=%u D=%u Tt=%u depth=%u)\nFAILED!\n", lmem_bytes,
                (unsigned long long)lmem_cap, G, D, Tt, depth);
    return 1;
  }
  const uint32_t ntiles = (T + Tt - 1) / Tt;

  // ---- the reduced-precision caches and the dequantized operands (rows t >= T stay NaN / poison, so a read past T fails)
  if (kvbits == 8) {
    Kq.assign(K.size(), (int8_t)0x55); Vq.assign(K.size(), (int8_t)0x55);
    Ks.assign((size_t)B * H_kv * Tmax, std::nanf("")); Vs = Ks;
    Kd.assign(K.size(), std::nanf("")); Vd = Kd;
    for (uint32_t b = 0; b < B; ++b)
      for (uint32_t t = 0; t < T; ++t)
        for (uint32_t kvh = 0; kvh < H_kv; ++kvh) {
          const size_t off = (((size_t)b * Tmax + t) * H_kv + kvh) * D, so = ((size_t)b * H_kv + kvh) * Tmax + t;
          ps::quantize_q80(&K[off], D, D, &Kq[off], &Ks[so]);
          ps::quantize_q80(&V[off], D, D, &Vq[off], &Vs[so]);
          for (uint32_t d = 0; d < D; ++d) {
            Kd[off + d] = (float)Kq[off + d] * Ks[so];   // float32 product, as the device
            Vd[off + d] = (float)Vq[off + d] * Vs[so];
          }
        }
  } else if (kvbits == 16) {
    Kh.assign(K.size(), 0x5555u); Vh = Kh;
    Kd.assign(K.size(), std::nanf("")); Vd = Kd;
    for (uint32_t b = 0; b < B; ++b)
      for (uint32_t t = 0; t < T; ++t)
        for (size_t i = ((size_t)b * Tmax + t) * H_kv * D, e = i + (size_t)H_kv * D; i < e; ++i) {
          Kh[i] = ps_f16::f32_to_f16_rne(K[i]); Vh[i] = ps_f16::f32_to_f16_rne(V[i]);
          Kd[i] = ps_f16::f16_to_f32(Kh[i]); Vd[i] = ps_f16::f16_to_f32(Vh[i]);
        }
  }
  const std::vector<float>& Kr = kvq ? Kd : K;
  const std::vector<float>& Vr = kvq ? Vd : V;

  // ---- reference and tolerance
  // n_acc + n_l + the rescale term 8 (n_tiles - 1) (header comment)
  Ref ref = reference(q, Kr, Vr, B, H_q, H_kv, D, T, Tmax, (T + ntiles) + (Tt + ntiles) + 8 * (ntiles - 1));
  const double tol = tol_override > 0 ? tol_override : std::fmax(kPlanTol, ref.bound);
  const size_t NO = (size_t)B * H_q * D;
  std::printf("ps_decode_attn: %s B=%u H_q=%u H_kv=%u D=%u T=%u Tmax=%u variant=%s G=%u Tt=%u depth=%u ntiles=%u "
              "NT=%u NW=%u lmem=%u/%llu bound=%.3e tol=%.3e\n", label.c_str(), B, H_q, H_kv, D, T, Tmax,
              variant.c_str(), G, Tt, dxa ? depth : 0, ntiles, (unsigned)nt, (unsigned)nw, lmem_bytes,
              (unsigned long long)lmem_cap, ref.bound, tol);
  bool ok = true;
  double tol_fx = 0.0;
  if (!out_fx.empty() && !kvq) {
    // NumPy float32 reference: its own bound (same formula, chains 2 T: einsum over t and the softmax sum)
    Ref rnp = reference(q, K, V, B, H_q, H_kv, D, T, Tmax, 2 * T);
    auto st = ps::compare(out_fx.data(), ref.out.data(), ref.scale.data(), NO, std::fmax(kPlanTol, rnp.bound));
    ps::print_check("fixture_vs_f64", st);
    ok = ok && st.ok();
    tol_fx = tol_override > 0 ? tol_override : std::fmax(kPlanTol, ref.bound + rnp.bound);
  }

  // ---- buffers
  ps_attn_arg_t arg = {};
  arg.B = B; arg.H_q = H_q; arg.H_kv = H_kv; arg.D = D; arg.T = T; arg.Tmax = Tmax;
  arg.kv_mul = kv_mul; arg.G = G; arg.Tt = Tt; arg.depth = depth;
  arg.scale = 1.0f / sqrtf((float)D);
  const uint64_t qb = (uint64_t)q.size() * 4, kb = (uint64_t)K.size() * esz, ob = NO * 4,
                 sb = (uint64_t)B * H_kv * Tmax * 4;   // scale bytes per array (kv8)
  PS_RT_CHECK(vx_buffer_create(ctx.device, qb, VX_MEM_READ, &ctx.q_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.q_buf, &arg.q_addr));
  PS_RT_CHECK(vx_buffer_create(ctx.device, kb, VX_MEM_READ, &ctx.k_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.k_buf, &arg.k_addr));
  PS_RT_CHECK(vx_buffer_create(ctx.device, kb, VX_MEM_READ, &ctx.v_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.v_buf, &arg.v_addr));
  PS_RT_CHECK(vx_buffer_create(ctx.device, ob, VX_MEM_READ_WRITE, &ctx.o_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.o_buf, &arg.o_addr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.q_buf, 0, q.data(), qb, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.k_buf, 0, kvbits == 8 ? (const void*)Kq.data() : kvbits == 16 ? (const void*)Kh.data() : (const void*)K.data(), kb, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.v_buf, 0, kvbits == 8 ? (const void*)Vq.data() : kvbits == 16 ? (const void*)Vh.data() : (const void*)V.data(), kb, 0, nullptr, nullptr));
  ps_attn_kv_arg_t karg = {};
  if (kvbits == 8) {
    PS_RT_CHECK(vx_buffer_create(ctx.device, sb, VX_MEM_READ, &ctx.ks_buf));
    PS_RT_CHECK(vx_buffer_address(ctx.ks_buf, &karg.ks_addr));
    PS_RT_CHECK(vx_buffer_create(ctx.device, sb, VX_MEM_READ, &ctx.vs_buf));
    PS_RT_CHECK(vx_buffer_address(ctx.vs_buf, &karg.vs_addr));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.ks_buf, 0, Ks.data(), sb, 0, nullptr, nullptr));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.vs_buf, 0, Vs.data(), sb, 0, nullptr, nullptr));
  }
  const std::vector<float> poison(NO, std::nanf(""));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.o_buf, 0, poison.data(), ob, 0, nullptr, nullptr));
  if (dxa) {
    // 4-D descriptors over [B][Tmax][H_kv][D]: dim0 = d, dim1 = kv head, dim2 = position (size T: rows >= T are
    // out of range and zero-filled), dim3 = sequence; tile = one head's D x Tt block -> LMEM rows [Tt][D]
    for (auto [slot, base] : {std::pair<uint32_t, uint64_t>{PS_DXA_DESC_K, arg.k_addr}, {PS_DXA_DESC_V, arg.v_addr}})
      PS_RT_CHECK(vortex::dxa::program_4d(ctx.device, slot, base, D, H_kv, T, B, D * esz, H_kv * D * esz,
                                          Tmax * H_kv * D * esz, D, 1, Tt, 1, esz));
    if (kvbits == 8)   // scales [B][H_kv][Tmax]: dims (position, kv head, sequence), tile Tt x 1 x 1 (kernel_kv.cpp: 3-D issue)
      for (auto [slot, base] : {std::pair<uint32_t, uint64_t>{PS_DXA_DESC_KS, karg.ks_addr}, {PS_DXA_DESC_VS, karg.vs_addr}})
        PS_RT_CHECK(vortex::dxa::program_3d(ctx.device, slot, base, T, H_kv, B, Tmax * 4, H_kv * Tmax * 4, Tt, 1, 1, 4));
    if (ps_dxa_class != 0) {   // PhaseShift (patch dxa-class): re-pack META with the stream class (row-major, rank 4)
#ifndef VX_PS_NUM_CLASSES   // the base tree has no class field / ps_set_stream (host builds there must compile)
      std::printf("-C needs the ps tree (patch dxa-class)\nFAILED!\n");
      return 1;
#else
      if (ps_dxa_class >= VX_PS_NUM_CLASSES) { std::printf("bad -C %u\nFAILED!\n", ps_dxa_class); return 1; }
      for (uint32_t slot : {(uint32_t)PS_DXA_DESC_K, (uint32_t)PS_DXA_DESC_V})
        PS_RT_CHECK(vortex::dxa::ps_set_stream(ctx.device, slot, 4, esz, vortex::dxa::Layout::RowMajor, ps_dxa_class, false));
      std::printf("PSDXASTREAM: kv_class=%u\n", ps_dxa_class);
#endif
    }
  }
  if (range_mode != "none") {
    // PhaseShift [dcr_phase] kv[0] = [min(K, V) base, max(K, V) end): K and V are two separate allocations, so the
    // span also covers the allocator gap between them; it must not contain q or out (checked).
    const uint64_t lo = std::min(arg.k_addr, arg.v_addr), hi = std::max(arg.k_addr, arg.v_addr) + kb;
    auto inside = [&](uint64_t a, uint64_t n) { return a < hi && a + n > lo; };
    if (inside(arg.q_addr, qb) || inside(arg.o_addr, ob)) {
      std::printf("-R %s: q or out lies inside the K..V span [0x%llx, 0x%llx)\nFAILED!\n", range_mode.c_str(),
                  (unsigned long long)lo, (unsigned long long)hi);
      return 1;
    }
    ps::PhaseCfg pc;
    pc.kv_base[0] = lo;
    pc.kv_limit[0] = hi;
    if (range_mode == "kvacct") pc.flags |= ps::phase_dcr::FLAG_RANGE_ACCT_ONLY;
    if (have_wm) {
      ps::WayMaskCtl wmc;
      PS_RT_CHECK(wmc.set(pc, wm_l1, wm_l2));
      ps::waymask_print(wmc);
    }
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
    std::printf("PSRANGE: mode=%s kv0=0x%llx-0x%llx kv_bytes=%llu gap_bytes=%llu flags=0x%x\n", range_mode.c_str(),
                (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)(2 * kb),
                (unsigned long long)(hi - lo - 2 * kb), pc.flags);
  } else if (have_wm) {   // PhaseShift (patch waymask): masks only, every other [dcr_phase] field at reset (0)
    ps::PhaseCfg pc;
    ps::WayMaskCtl wmc;
    PS_RT_CHECK(wmc.set(pc, wm_l1, wm_l2));
    ps::waymask_print(wmc);
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
  }
  PS_RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
  const char* kname = (variant == "naive") ? "attn_naive" : variant == "gqa_kv8" ? "attn_gqa_kv8" : variant == "dxa_kv8" ? "attn_dxa_kv8" :
                      variant == "gqa_kv16" ? "attn_gqa_kv16" : variant == "dxa_kv16" ? "attn_dxa_kv16" : dxa ? "attn_dxa" : "attn_gqa";
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, kname, &ctx.kernel));
  PS_RT_CHECK(vx_check_occupancy(ctx.device, nth, lmem_bytes));

  const uint32_t grid = B * (H_q / G);
  std::printf("launch: kernel=%s grid=%u block=%u lmem_bytes=%u\n", kname, grid, nth, lmem_bytes);
  // work units the kernel marks (PS_HINTS=1 only): one DECODE_TILE per (CTA, key tile); counts per type
  // other,prefill_tile,decode_tile (VX_PS_MARK_*), compared with SimX's PSMARK-TOTAL line by tests/test_mem_class.py
  std::printf("PSHINT: ps_hints=%d expect_marks=0,0,%llu\n", (int)PS_HINTS,
              (unsigned long long)(PS_HINTS ? (uint64_t)grid * ntiles : 0));
  std::vector<float> O(NO);
  ps::Timer timer;
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = ctx.kernel;
  karg.a = arg; karg.esz = esz;
  li.args_host = kvq ? (const void*)&karg : (const void*)&arg;
  li.args_size = kvq ? sizeof(karg) : sizeof(arg);
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
  if (!out_fx.empty() && !kvq) {
    std::vector<double> rfx(out_fx.begin(), out_fx.end());
    auto st = ps::compare(O.data(), rfx.data(), ref.scale.data(), NO, tol_fx);
    ps::print_check("launch0:vs_fixture", st);
    ok = ok && st.ok();
  }
  // E3 traffic of this step (one layer): K and V of every sequence once = 2 B T H_kv D q bytes (q = 4), plus q, out
  const uint64_t e3_kv = 2ull * B * T * H_kv * D * esz;   // data bytes of the cache at the stored element width (4 for the fp32 kernels)
  std::printf("PSINFO: within_plan_tol=%d e3_kv_bytes=%llu q_out_bytes=%llu kv_reads_per_step=%u\n",
              (int)(s64.max_rel <= kPlanTol), (unsigned long long)e3_kv, (unsigned long long)(2 * ob),
              variant == "naive" ? kv_mul : 1u);
  // bytes of the step's KV reads (measured bytes count the scales, the analytic region thresholds of §2.3 ignore them)
  const uint64_t kv_scale_bytes = (kvbits == 8) ? 2ull * B * T * H_kv * 4 : 0ull;
  std::printf("PSKV: mode=%s esz=%u data_bytes=%llu scale_bytes=%llu total_bytes=%llu fp32_bytes=%llu ratio_vs_fp32=%.4f\n",
              kvbits == 8 ? "kv8" : kvbits == 16 ? "kv16" : "fp32", esz, (unsigned long long)e3_kv, (unsigned long long)kv_scale_bytes,
              (unsigned long long)(e3_kv + kv_scale_bytes), (unsigned long long)(2ull * B * T * H_kv * D * 4),
              (double)(e3_kv + kv_scale_bytes) / (double)(2ull * B * T * H_kv * D * 4));
  if (kvq) {
    // information: the error of the REDUCED CACHE itself, device output vs the float64 attention over the original fp32 K / V (never pass / fail)
    Ref ref0 = reference(q, K, V, B, H_q, H_kv, D, T, Tmax, (T + ntiles) + (Tt + ntiles) + 8 * (ntiles - 1));
    auto sq = ps::compare(O.data(), ref0.out.data(), ref0.scale.data(), NO, 1.0);
    std::printf("PSKVQ: cache_error_vs_fp32kv max_rel=%.6e max_abs=%.6e (information; scale S_i = sum_t p_t |v_t[d]|)\n", sq.max_rel, sq.max_abs);
  }
  if (!ok) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
