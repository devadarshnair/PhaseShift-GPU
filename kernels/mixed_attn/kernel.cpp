// kernel.cpp -- ps_mixed_attn device kernel: fused mixed-batch attention. ONE launch serves a
// Sarathi-style mixed iteration [cand:A5] (a prefill chunk plus a batch of decode queries) with a CTA-level split
// (POD-Attention-like [cand:B10]): CTAs [0, n_p) run the prefill chunk, CTAs [n_p, n_p + n_d) the decode queries
// (common.h; arg->order = 1 puts the decode range first). This is the kernel in which a per-kernel static
// configuration cannot specialise: prefill-like and decode-like CTAs share one launch, one configuration and the
// machine at the same time. Design choices: docs/methodology.md §14.6. float32.
//
// Reuse, not re-design: the two CTA bodies are the validated kernels' bodies, adapted only where the fused launch
// requires it, so that each part computes exactly what the separate launch computes (same tiles, same instruction
// order of every floating-point operation; bit-identical outputs are TESTED, tests/test_mixed_attn.py):
//   prefill_cta  = kernels/prefill_attn/kernel.cpp prefill_cta<Mode::Lmem> (`lmem_tiled`): tiled causal
//                  online-softmax attention with K/V tiles copied into LMEM by the threads, causal tile skipping, the
//                  chunk offset s0. Changes: the CTA index is a parameter (blockIdx.x - offset), the arguments come
//                  from ps_mattn_arg_t (Tmax -> Tmax_p, *_addr -> *p_addr), B = 1, and the DXA path is not compiled.
//   decode_cta   = kernels/decode_attn_gqa/kernel.cpp attn_cta<G, Mode::Plain> (`gqa_shared_kv`): one CTA per
//                  (sequence, KV head) serving its G query heads, K/V read straight from global memory, online softmax
//                  over key tiles of Tt. Changes: the CTA index is a parameter, the context length is per sequence
//                  (T = ctx[b], read from global memory) instead of one T for the batch, Tmax -> Tmax_d, and the DXA
//                  path is not compiled.
// Both bodies use the shared ps_expf (warp_ops.h, symlinked so vxrun's app-source hash covers it) and the inline
// FMA macro below. The DXA variants are not fused here (does not ask for them;
// measured that DXA does not pay at the default 16 KiB tiles).
//
// LMEM: every CTA of the launch gets the same LMEM size (the host passes max(prefill footprint, decode footprint));
// each body uses its own layout (ps_pattn_lmem / ps_attn_lmem) from offset 0. One CTA = NT*NW threads = all warps of
// a core, so one CTA is resident per core.
//
// PhaseShift hints (kernels/common/ps_hint.h; compiled only with -DPS_HINTS=1, a distinct binary --:
// compare mechanisms only between configurations running the same binary): prefill CTAs tag their K/V tile copy
// loads KV_PERSIST (every query tile of the chunk re-reads the same K/V rows within the launch) and decode CTAs tag
// their key-tile loop KV_STREAM (each K/V row read once per step); both return to ORDINARY before the output store.
// Thread 0 of each CTA marks every finished key tile: PREFILL_TILE (host expectation tiles_p) or DECODE_TILE
// (sum over decode CTAs of ceil(ctx[b] / Tt)).
//
// Per-CTA timestamps (per-region attribution; arg->ts_addr != 0): thread 0 reads the core's cycle counter (mcycle,
// vx_rdcycle; every SimX core ticks once per machine cycle, so the counters of different cores are one clock,
// sim/simx/core.cpp:343-352) at CTA entry and, after a CTA barrier, at CTA exit, and writes one 64 B record per CTA
// (common.h). Cost: two counter reads, one barrier and eight stores per CTA (measured: ts-off control rows).
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include "common.h"
// FMA codegen (errata 30): the generated Makefile adds -fno-math-errno, so __builtin_fmaf is the inline
// fmadd.s; the macro (as in kernels/prefill_attn/kernel.cpp) keeps this kernel libcall-free even with stock flags.
// Same IEEE fused multiply-add either way, so results are identical to the separate kernels.
#define __builtin_fmaf(a, b, c) __builtin_elementwise_fma((float)(a), (float)(b), (float)(c))
#include "warp_ops.h"
#include "ps_hint.h"

namespace {

// ------------------------------------------------------------------------------------------------ prefill CTA
// kernels/prefill_attn/kernel.cpp prefill_cta<Mode::Lmem> with cta = prefill CTA index (see the header comment)
static inline __attribute__((always_inline)) void prefill_cta(const ps_mattn_arg_t* arg, uint32_t cta) {
  const uint32_t D = arg->D, S = arg->S, s0 = arg->s0, G = arg->G, Qt = arg->Qt, Tk = arg->Tk, H_kv = arg->H_kv;
  const uint32_t H_q = arg->H_q, nqt = arg->nqt;
  const uint32_t qt = cta % nqt, bk = cta / nqt;              // query tile innermost (numbering)
  const uint32_t kvh = bk % H_kv, b = bk / H_kv;              // b = 0: one prefill sequence
  const uint32_t q0 = qt * Qt;
  const uint32_t nq = (S - q0 < Qt) ? S - q0 : Qt;            // valid query tokens of this tile
  const uint32_t pos0 = s0 + q0;                              // absolute position of the tile's first token
  const uint32_t kend = pos0 + nq;                            // keys 0 .. kend-1 are visible to some row
  const uint32_t ntiles = (kend + Tk - 1) / Tk;
  const uint32_t tid = threadIdx.x, nth = blockDim.x;
  const float scale = arg->scale;
  const uint32_t h0 = kvh * G;
  const size_t row_stride = (size_t)H_kv * D;                                   // floats between key positions
  const size_t seq_off = ((size_t)b * arg->Tmax_p * H_kv + kvh) * D;            // K[b][0][kvh][0]
  const float* kbase = reinterpret_cast<const float*>(arg->kp_addr);
  const float* vbase = reinterpret_cast<const float*>(arg->vp_addr);
  const float* qg = reinterpret_cast<const float*>(arg->qp_addr);
  float* og = reinterpret_cast<float*>(arg->op_addr);

  const ps_pattn_lmem_t L = ps_pattn_lmem(G, D, Qt, Tk, 1);
  const uint32_t Rp = L.rows, nrb = Rp / 4, ncb = D / 4, Rv = G * nq;   // padded rows, row/col blocks, valid rows
  float* lm = reinterpret_cast<float*>(__local_mem());
  float* qs = lm + L.qs;
  float* acc = lm + L.acc;
  float* ps = lm + L.ps;
  float* mst = lm + L.stat;
  float* lst = mst + Rp;
  float* cst = lst + Rp;

  vortex::barrier sync(5);   // phase barrier

  // Q tile -> LMEM (rows >= Rv: q = 0), acc = 0, m = -inf, l = 0
  for (uint32_t i = tid; i < Rp * D; i += nth) {
    const uint32_t r = i / D, d = i - r * D;
    float v = 0.0f;
    if (r < Rv) {
      const uint32_t tok = r / G, h = r - tok * G;
      v = qg[(((size_t)b * S + q0 + tok) * H_q + h0 + h) * D + d];
    }
    qs[i] = v;
    acc[i] = 0.0f;
  }
  for (uint32_t r = tid; r < Rp; r += nth) { mst[r] = psk::neg_inf(); lst[r] = 0.0f; }
  sync.arrive_and_wait();

  for (uint32_t j = 0; j < ntiles; ++j) {
    const uint32_t t0 = j * Tk;
    const uint32_t nt = (kend - t0 < Tk) ? kend - t0 : Tk;   // keys of this tile that any row can see
    float* ks = lm + L.ks;
    float* vs = lm + L.vs;
    {
      // cooperative copy of rows t0 .. t0+nt-1 (element i = tt * D + d; consecutive threads take consecutive d)
      uint32_t tt = tid / D, d = tid - tt * D;
      PS_MEM_HINT(PS_CLASS_KV_PERSIST);
      for (uint32_t i = tid; i < nt * D; i += nth) {
        const size_t g = seq_off + (size_t)(t0 + tt) * row_stride + d;
        ks[i] = kbase[g];
        vs[i] = vbase[g];
        d += nth;
        while (d >= D) { d -= D; ++tt; }
      }
      PS_MEM_HINT(PS_CLASS_ORDINARY);
      sync.arrive_and_wait();
    }

    // S: scores, item = (4-row block rb, key t); masked entries -> -inf
    for (uint32_t it = tid; it < nrb * nt; it += nth) {
      const uint32_t rb = it / nt, t = it - rb * nt;
      const float* qr = qs + rb * 4 * D;
      const float* kr = ks + t * D;
      float s0v = 0.0f, s1v = 0.0f, s2v = 0.0f, s3v = 0.0f;
      for (uint32_t d = 0; d < D; ++d) {
        const float kd = kr[d];
        s0v = __builtin_fmaf(qr[d], kd, s0v);
        s1v = __builtin_fmaf(qr[D + d], kd, s1v);
        s2v = __builtin_fmaf(qr[2 * D + d], kd, s2v);
        s3v = __builtin_fmaf(qr[3 * D + d], kd, s3v);
      }
      const float sv[4] = {s0v, s1v, s2v, s3v};
      const uint32_t kp = t0 + t;
#pragma unroll
      for (uint32_t u = 0; u < 4; ++u) {
        const uint32_t r = rb * 4 + u;
        uint32_t tok = r / G;
        if (tok >= nq) tok = nq - 1;            // padding rows take the last valid token's position
        ps[r * Tk + t] = (kp <= pos0 + tok) ? sv[u] * scale : psk::neg_inf();
      }
    }
    sync.arrive_and_wait();

    // M1: running max and rescale factor, thread per row
    for (uint32_t r = tid; r < Rp; r += nth) {
      const float mo = mst[r];
      float mx = mo;
      for (uint32_t t = 0; t < nt; ++t) mx = __builtin_fmaxf(mx, ps[r * Tk + t]);
      cst[r] = psk::ps_expf(mo - mx);           // mo = -inf on the first tile -> 0; no visible key -> exp(0) = 1
      mst[r] = mx;
    }
    sync.arrive_and_wait();

    // M2: p = exp(s - m'), masked -> 0, thread per (row, key)
    for (uint32_t it = tid; it < Rp * nt; it += nth) {
      const uint32_t r = it / nt, t = it - r * nt;
      const float sv = ps[r * Tk + t];
      ps[r * Tk + t] = (sv == psk::neg_inf()) ? 0.0f : psk::ps_expf(sv - mst[r]);
    }
    sync.arrive_and_wait();

    // V: running sums (thread per row) and outputs (4 x 4 register blocks)
    for (uint32_t r = tid; r < Rp; r += nth) {
      float sum = 0.0f;
      for (uint32_t t = 0; t < nt; ++t) sum += ps[r * Tk + t];
      lst[r] = __builtin_fmaf(lst[r], cst[r], sum);
    }
    for (uint32_t it = tid; it < nrb * ncb; it += nth) {
      const uint32_t rb = it / ncb, cb = it - rb * ncb;
      const uint32_t r0 = rb * 4, d0 = cb * 4;
      float a[4][4];
#pragma unroll
      for (uint32_t u = 0; u < 4; ++u) {
        const float c = cst[r0 + u];
#pragma unroll
        for (uint32_t w = 0; w < 4; ++w) a[u][w] = acc[(r0 + u) * D + d0 + w] * c;
      }
      for (uint32_t t = 0; t < nt; ++t) {
        float p[4], v[4];
#pragma unroll
        for (uint32_t u = 0; u < 4; ++u) p[u] = ps[(r0 + u) * Tk + t];
#pragma unroll
        for (uint32_t w = 0; w < 4; ++w) v[w] = vs[t * D + d0 + w];
#pragma unroll
        for (uint32_t u = 0; u < 4; ++u)
#pragma unroll
          for (uint32_t w = 0; w < 4; ++w) a[u][w] = __builtin_fmaf(p[u], v[w], a[u][w]);
      }
#pragma unroll
      for (uint32_t u = 0; u < 4; ++u)
#pragma unroll
        for (uint32_t w = 0; w < 4; ++w) acc[(r0 + u) * D + d0 + w] = a[u][w];
    }
    sync.arrive_and_wait();                      // Ks/Vs and ps are rewritten by the next tile
    if (tid == 0) PS_MARK(PS_MARK_PREFILL_TILE);
  }

  // out = acc / l for the valid rows
  for (uint32_t i = tid; i < Rv * D; i += nth) {
    const uint32_t r = i / D, d = i - r * D;
    const uint32_t tok = r / G, h = r - tok * G;
    og[(((size_t)b * S + q0 + tok) * H_q + h0 + h) * D + d] = acc[i] / lst[r];
  }
}

// ------------------------------------------------------------------------------------------------ decode CTA
// kernels/decode_attn_gqa/kernel.cpp attn_cta<G, Mode::Plain> with cta = decode CTA index and T = ctx[b]
template <uint32_t G>
static inline __attribute__((always_inline)) void decode_cta(const ps_mattn_arg_t* arg, uint32_t cta) {
  const uint32_t D = arg->D, Tt = arg->Tt, H_kv = arg->H_kv;
  const uint32_t groups = arg->H_q / G;                 // CTAs per sequence (= H_kv: G = kv_mul)
  const uint32_t b = cta / groups, h0 = (cta % groups) * G, kvh = h0 / arg->G;
  const uint32_t T = reinterpret_cast<const uint32_t*>(arg->ctx_addr)[b];   // this sequence's context length
  const uint32_t tid = threadIdx.x, nth = blockDim.x;
  const float scale = arg->scale;
  const float* q = reinterpret_cast<const float*>(arg->qd_addr) + ((size_t)b * arg->H_q + h0) * D;
  const float* kbase = reinterpret_cast<const float*>(arg->kd_addr);
  const float* vbase = reinterpret_cast<const float*>(arg->vd_addr);
  float* out = reinterpret_cast<float*>(arg->od_addr) + ((size_t)b * arg->H_q + h0) * D;
  const size_t row_stride = (size_t)H_kv * D;                                   // floats between positions
  const size_t seq_off = ((size_t)b * arg->Tmax_d * H_kv + kvh) * D;            // K[b][0][kvh][0]

  const ps_attn_lmem_t L = ps_attn_lmem(G, D, Tt, 1, 0);
  float* lm = reinterpret_cast<float*>(__local_mem());
  float* qs = lm + L.qs;
  float* acc = lm + L.acc;
  float* ps = lm + L.ps;
  float* mst = lm + L.stat;
  float* lst = mst + G;
  float* cst = lst + G;

  vortex::barrier sync(5);

  for (uint32_t i = tid; i < G * D; i += nth) { qs[i] = q[i]; acc[i] = 0.0f; }
  for (uint32_t h = tid; h < G; h += nth) { mst[h] = psk::neg_inf(); lst[h] = 0.0f; }
  sync.arrive_and_wait();

  const uint32_t ntiles = (T + Tt - 1) / Tt;
  PS_MEM_HINT(PS_CLASS_KV_STREAM);
  for (uint32_t j = 0; j < ntiles; ++j) {
    const uint32_t t0 = j * Tt;
    const uint32_t nt = (T - t0 < Tt) ? T - t0 : Tt;
    const float* krows = kbase + seq_off + (size_t)t0 * row_stride;   // row tt of the tile: krows + tt * kstride
    const float* vrows = vbase + seq_off + (size_t)t0 * row_stride;
    const size_t kstride = row_stride;

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
    sync.arrive_and_wait();                      // ps is rewritten by the next tile
    if (tid == 0) PS_MARK(PS_MARK_DECODE_TILE);
  }
  PS_MEM_HINT(PS_CLASS_ORDINARY);

  for (uint32_t i = tid; i < G * D; i += nth) out[i] = acc[i] / lst[i / D];
}

static inline __attribute__((always_inline)) void decode_dispatch(const ps_mattn_arg_t* arg, uint32_t cta) {
  switch (arg->G) {   // the host only launches G in {1, 2, 4, 8}
    case 1: decode_cta<1>(arg, cta); break;
    case 2: decode_cta<2>(arg, cta); break;
    case 4: decode_cta<4>(arg, cta); break;
    case 8: decode_cta<8>(arg, cta); break;
    default: break;
  }
}

}  // namespace

__kernel void attn_mixed(ps_mattn_arg_t* __UNIFORM__ arg) {
  const uint32_t cta = blockIdx.x;
  const uint32_t first = arg->order ? arg->n_d : arg->n_p;   // CTAs of the first range
  const bool in_first = cta < first;
  const bool decode = arg->order ? in_first : !in_first;
  const uint32_t idx = in_first ? cta : cta - first;          // region-local CTA index
  uint32_t* ts = arg->ts_addr ? reinterpret_cast<uint32_t*>(arg->ts_addr) + cta * PS_MATTN_TS_WORDS : nullptr;
  uint64_t t_start = 0;
  if (ts && threadIdx.x == 0) t_start = vx_rdcycle();
  if (decode)
    decode_dispatch(arg, idx);
  else
    prefill_cta(arg, idx);
  if (ts) {
    vortex::barrier(5).arrive_and_wait();     // every warp of the CTA has finished its part
    if (threadIdx.x == 0) {
      const uint64_t t_end = vx_rdcycle();
      ts[0] = (uint32_t)t_start;
      ts[1] = (uint32_t)(t_start >> 32);
      ts[2] = (uint32_t)t_end;
      ts[3] = (uint32_t)(t_end >> 32);
      ts[4] = (uint32_t)vx_core_id();
      ts[5] = decode ? 1u : 0u;
      ts[6] = idx;
      ts[7] = PS_MATTN_TS_MAGIC;
    }
  }
}
