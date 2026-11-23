// kernel.cpp -- ps_decode_attn_gqa device kernels: fused decode attention (QK^T -> softmax -> PV in one
// kernel; scores never leave the CTA), float32. Design choices are documented in docs/methodology.md §14.2.
//
// Work decomposition: one CTA (block of NT*NW threads = all warps of one core) per (sequence b, group of G query
// heads that share one KV head). Three entry points share one body:
//   attn_naive  (`naive_per_qhead`)  G = 1: one CTA per (b, query head); the kv_mul CTAs of a KV head each stream
//                                    that head's K/V, so every K/V line is fetched kv_mul times per step
//   attn_gqa    (`gqa_shared_kv`)    G = kv_mul: one CTA per (b, KV head) serving its kv_mul query heads, so each
//                                    K/V element is loaded once per step (the E5 / E3 traffic assumption)
//   attn_dxa    (`gqa_dxa_dbuf`)     attn_gqa with K/V tiles staged into LMEM by the DXA engine with `depth` stages
//                                    (depth 1 = load-then-compute; 2 = double buffering; 4 = three tiles in flight):
//                                    the software KV-prefetch baseline (R§4) and mechanism C's zero-hardware
//                                    layer. Needs VX_CFG_EXT_DXA_ENABLE.
//
// Online (single-pass) softmax over key tiles of Tt positions (running max and running sum, rescaled per tile; here per
// query head): with running max m, running sum l and running output acc[D],
//     tile scores s_t = (q . k_t) * scale,  m' = max(m, max_t s_t),  c = exp(m - m'),
//     p_t = exp(s_t - m'),  l' = l c + sum_t p_t,  acc' = acc c + sum_t p_t v_t,   out = acc / l  after the last tile.
// m starts at -inf (exp(-inf - m') = 0, the fix), so all-negative logits are handled. LMEM holds
// O(G*D + G*Tt) floats independent of T, so T is not limited by the 16 KiB LMEM default (VX_config.toml:279-288).
//
// Per tile, four phases separated by CTA barriers:
//   S  thread per key position t: s[h] = q_h . k_t for all G heads at once (each K element is loaded once and used
//      G times; q_h from LMEM), written to ps[h][t]
//   M1 thread per head: tile max, m', c
//   M2 thread per (head, position): p = exp(s - m') in place (ps_expf, kernels/softmax_fixed/warp_ops.h)
//   V  thread per output column d: acc[h][d] = acc[h][d] c_h + sum_t p[h][t] v_t[d] for all G heads (each V element
//      loaded once, G accumulators in registers); threads h < G also update l_h
// Keys beyond T are never touched (loops stop at the valid count; DXA zero-fills out-of-range rows).
//
// PhaseShift hints (kernels/common/ps_hint.h; compiled only with -DPS_HINTS=1, a distinct binary): every warp
// tags its loads inside the key-tile loop (the K and V rows; everything else in the loop is LMEM) as KV_STREAM and
// returns to ORDINARY before the output store; thread 0 of the CTA marks each finished key tile as one DECODE_TILE
// work unit (host expectation: grid * ntiles). In the dxa variant the K/V reads are DXA's (BULK_DXA by SimX).
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#include "common.h"
#include "warp_ops.h"
#include "ps_hint.h"

namespace {

enum class Mode { Plain, Dxa };

struct Ctx {
  const float* q;
  const float* k;
  const float* v;
  float* o;
  uint32_t b, h0, kvh;
};

template <uint32_t G, Mode MODE>
static inline __attribute__((always_inline)) void attn_cta(const ps_attn_arg_t* arg) {
  const uint32_t D = arg->D, T = arg->T, Tt = arg->Tt, H_kv = arg->H_kv;
  const uint32_t groups = arg->H_q / G;                 // CTAs per sequence
  const uint32_t cta = blockIdx.x;
  const uint32_t b = cta / groups, h0 = (cta % groups) * G, kvh = h0 / arg->kv_mul;
  const uint32_t tid = threadIdx.x, nth = blockDim.x;
  const float scale = arg->scale;
  const float* q = reinterpret_cast<const float*>(arg->q_addr) + ((size_t)b * arg->H_q + h0) * D;
  const float* kbase = reinterpret_cast<const float*>(arg->k_addr);
  const float* vbase = reinterpret_cast<const float*>(arg->v_addr);
  float* out = reinterpret_cast<float*>(arg->o_addr) + ((size_t)b * arg->H_q + h0) * D;
  const size_t row_stride = (size_t)H_kv * D;                                   // floats between positions
  const size_t seq_off = ((size_t)b * arg->Tmax * H_kv + kvh) * D;              // K[b][0][kvh][0]

  const ps_attn_lmem_t L = ps_attn_lmem(G, D, Tt, arg->depth, MODE == Mode::Dxa);
  float* lm = reinterpret_cast<float*>(__local_mem());
  float* qs = lm + L.qs;
  float* acc = lm + L.acc;
  float* ps = lm + L.ps;
  float* mst = lm + L.stat;
  float* lst = mst + G;
  float* cst = lst + G;

  vortex::barrier sync(5);   // phase barrier (ids 1..4 are the DXA stage barriers; id 0 is __syncthreads' slot)

  for (uint32_t i = tid; i < G * D; i += nth) { qs[i] = q[i]; acc[i] = 0.0f; }
  for (uint32_t h = tid; h < G; h += nth) { mst[h] = psk::neg_inf(); lst[h] = 0.0f; }
  sync.arrive_and_wait();

  const uint32_t ntiles = (T + Tt - 1) / Tt;
  const uint32_t depth = arg->depth;
  vortex::barrier stage_bar[4] = {vortex::barrier(1), vortex::barrier(2), vortex::barrier(3), vortex::barrier(4)};
  const bool issuer = (MODE == Mode::Dxa) && (get_sub_group_id() == 0);
  const uint32_t stage_floats = ps_attn_round16(Tt * D);
  auto issue = [&](uint32_t j) {   // DXA K and V tile j -> stage j % depth (whole issuer warp executes it)
    const uint32_t s = j % depth;
    stage_bar[s].expect_tx(2);
    vx_dxa_issue_4d_wg(PS_DXA_DESC_K, stage_bar[s].id(), lm + L.ks + s * stage_floats, 0, kvh, j * Tt, b);
    vx_dxa_issue_4d_wg(PS_DXA_DESC_V, stage_bar[s].id(), lm + L.vs + s * stage_floats, 0, kvh, j * Tt, b);
  };
  if (MODE == Mode::Dxa && issuer)
    for (uint32_t j = 0; j + 1 < depth && j < ntiles; ++j) issue(j);

  PS_MEM_HINT(PS_CLASS_KV_STREAM);
  for (uint32_t j = 0; j < ntiles; ++j) {
    const uint32_t t0 = j * Tt;
    const uint32_t nt = (T - t0 < Tt) ? T - t0 : Tt;
    const float* krows;   // row tt of the tile: krows + tt * kstride
    const float* vrows;
    size_t kstride;
    uint32_t s = 0;
    if (MODE == Mode::Dxa) {
      s = j % depth;
      if (issuer && j + depth - 1 < ntiles) issue(j + depth - 1);
      stage_bar[s].arrive_and_wait();            // tile j landed in stage s (tx events) and every warp arrived
      krows = lm + L.ks + s * stage_floats;
      vrows = lm + L.vs + s * stage_floats;
      kstride = D;
    } else {
      krows = kbase + seq_off + (size_t)t0 * row_stride;
      vrows = vbase + seq_off + (size_t)t0 * row_stride;
      kstride = row_stride;
    }

    // S: scores of the tile, thread per position, all G heads
    for (uint32_t tt = tid; tt < nt; tt += nth) {
      const float* kr = krows + tt * kstride;
      float sc[G];
#pragma unroll
      for (uint32_t h = 0; h < G; ++h) sc[h] = 0.0f;
      for (uint32_t d = 0; d < D; ++d) {
        const float kd = kr[d];
#pragma unroll
        for (uint32_t h = 0; h < G; ++h) sc[h] = __builtin_fmaf(qs[h * D + d], kd, sc[h]);
      }
#pragma unroll
      for (uint32_t h = 0; h < G; ++h) ps[h * Tt + tt] = sc[h] * scale;
    }
    sync.arrive_and_wait();

    // M1: running max and rescale factor, thread per head
    for (uint32_t h = tid; h < G; h += nth) {
      const float mo = mst[h];
      float mx = mo;
      for (uint32_t tt = 0; tt < nt; ++tt) mx = __builtin_fmaxf(mx, ps[h * Tt + tt]);
      cst[h] = psk::ps_expf(mo - mx);           // mo = -inf on the first tile -> 0
      mst[h] = mx;
    }
    sync.arrive_and_wait();

    // M2: p = exp(s - m'), thread per (head, position)
    for (uint32_t i = tid; i < G * nt; i += nth) {
      const uint32_t h = i / nt, tt = i - h * nt;
      ps[h * Tt + tt] = psk::ps_expf(ps[h * Tt + tt] - mst[h]);
    }
    sync.arrive_and_wait();

    // V: running sums and outputs
    for (uint32_t h = tid; h < G; h += nth) {
      float sum = 0.0f;
      for (uint32_t tt = 0; tt < nt; ++tt) sum += ps[h * Tt + tt];
      lst[h] = __builtin_fmaf(lst[h], cst[h], sum);
    }
    for (uint32_t d = tid; d < D; d += nth) {
      float a[G];
#pragma unroll
      for (uint32_t h = 0; h < G; ++h) a[h] = acc[h * D + d] * cst[h];
      for (uint32_t tt = 0; tt < nt; ++tt) {
        const float vd = vrows[tt * kstride + d];
#pragma unroll
        for (uint32_t h = 0; h < G; ++h) a[h] = __builtin_fmaf(ps[h * Tt + tt], vd, a[h]);
      }
#pragma unroll
      for (uint32_t h = 0; h < G; ++h) acc[h * D + d] = a[h];
    }
    if (MODE == Mode::Dxa)
      stage_bar[s].arrive_and_wait();            // every warp is done with stage s and with ps
    else
      sync.arrive_and_wait();                    // ps is rewritten by the next tile
    if (tid == 0) PS_MARK(PS_MARK_DECODE_TILE);
  }
  PS_MEM_HINT(PS_CLASS_ORDINARY);

  for (uint32_t i = tid; i < G * D; i += nth) out[i] = acc[i] / lst[i / D];
}

template <Mode MODE>
static inline __attribute__((always_inline)) void dispatch(const ps_attn_arg_t* arg) {
  switch (arg->G) {   // the host only launches G in {1, 2, 4, 8}
    case 1: attn_cta<1, MODE>(arg); break;
    case 2: attn_cta<2, MODE>(arg); break;
    case 4: attn_cta<4, MODE>(arg); break;
    case 8: attn_cta<8, MODE>(arg); break;
    default: break;
  }
}

}  // namespace

__kernel void attn_naive(ps_attn_arg_t* __UNIFORM__ arg) { attn_cta<1, Mode::Plain>(arg); }
__kernel void attn_gqa(ps_attn_arg_t* __UNIFORM__ arg) { dispatch<Mode::Plain>(arg); }
__kernel void attn_dxa(ps_attn_arg_t* __UNIFORM__ arg) { dispatch<Mode::Dxa>(arg); }
