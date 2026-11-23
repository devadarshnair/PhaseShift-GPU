// kernel_k8.cpp -- (decode attention with an int8 K cache and QK^T by DOT4ACC (attn_k8_dot) or scalar int8 (attn_k8_sw).
// Structure and numerics of the V / softmax side are kernels/decode_attn_gqa/kernel.cpp's attn_gqa (one CTA per (sequence, KV head) serving its G query heads,
// phases S / M1 / M2 / V separated by CTA barriers, online softmax over key tiles of Tt positions, ps_expf, V fp32, thread per output column, same accumulation
// orders); what changes:
//   * prologue: q is quantized per head (sQ = max|q| / 127), and the fp32 K row of position quant_pos is quantized per row into the int8 cache + scale
//     (scheme in common.h); two extra CTA barriers.
//   * S phase (thread per key position, all G heads): the D/4 int8 K words of the row are loaded as 32-bit words (4 independent loads in flight per chunk) and
//     combined with the packed int8 q words from LMEM:  dot variant: one DOT4ACC per (word, head);  sw variant: sign-extend the 4 K bytes with shifts and
//     do 4 scalar multiplies with the pre-unpacked int32 q (the cheapest straightforward scalar int8 path; no SIMD-within-a-register tricks).
//     The int32 sum is exact, so the two variants give BIT-IDENTICAL scores; the score is (float)isum * (sQ * sK_t) * scale.
// Compiled with fp contract off (the fp32 side uses explicit __builtin_fmaf like the stock kernel).
#include <math.h>
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include "common.h"
#include "../decode_attn_gqa/warp_ops.h"

#pragma clang fp contract(off)

namespace {

#define PSK8_INL inline __attribute__((always_inline))
#define PSK8_FENCE(v) asm volatile("" : "+r"(v))

PSK8_INL void ps_dot4acc(int32_t& acc, uint32_t a, uint32_t b) {
  asm(".insn r 0x7b, 0, 1, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b));
}
PSK8_INL int32_t sext8(uint32_t w, int k) { return (int32_t)(w << (24 - 8 * k)) >> 24; }

template <uint32_t G, bool DOT>
PSK8_INL void attn_k8_cta(const ps_attn_k8_arg_t* karg) {
  const ps_attn_arg_t* arg = &karg->a;
  const uint32_t D = arg->D, T = arg->T, Tt = arg->Tt, H_kv = arg->H_kv, Tmax = arg->Tmax;
  const uint32_t groups = arg->H_q / G;
  const uint32_t cta = blockIdx.x;
  const uint32_t b = cta / groups, h0 = (cta % groups) * G, kvh = h0 / arg->kv_mul;
  const uint32_t tid = threadIdx.x, nth = blockDim.x;
  const float scale = arg->scale;
  const float* q = reinterpret_cast<const float*>(arg->q_addr) + ((size_t)b * arg->H_q + h0) * D;
  int8_t* kq = reinterpret_cast<int8_t*>(arg->k_addr);
  const float* vbase = reinterpret_cast<const float*>(arg->v_addr);
  const float* kf = reinterpret_cast<const float*>(karg->kf_addr);
  float* out = reinterpret_cast<float*>(arg->o_addr) + ((size_t)b * arg->H_q + h0) * D;
  const size_t row_stride = (size_t)H_kv * D;                          // elements between positions (bytes of kq, floats of V / kf)
  const size_t seq_off = ((size_t)b * Tmax * H_kv + kvh) * D;          // [b][0][kvh][0]
  float* ks_g = reinterpret_cast<float*>(karg->ks_addr) + ((size_t)b * H_kv + kvh) * Tmax;
  const uint32_t qp = karg->quant_pos;
  const bool doq = qp != 0xFFFFFFFFu;
  const uint32_t DW = D / 4;

  const ps_attn_k8_lmem_t L = ps_attn_k8_lmem(G, D, Tt);
  float* lm = reinterpret_cast<float*>(__local_mem());
  float* qs = lm + L.qs;
  float* acc = lm + L.acc;
  float* ps = lm + L.ps;
  float* mst = lm + L.stat;
  float* lst = mst + G;
  float* cst = lst + G;
  float* qsc = cst + G;                       // sQ per head
  float* ksc0 = qsc + G;                      // sK of the quantized new row
  float* kt = lm + L.kt;
  int8_t* qb = reinterpret_cast<int8_t*>(lm + L.qb);
  const uint32_t* qw = reinterpret_cast<const uint32_t*>(lm + L.qb);
  int32_t* qi = reinterpret_cast<int32_t*>(lm + L.qi);

  vortex::barrier sync(5);

  const size_t qrow = seq_off + (size_t)qp * row_stride;                 // new K row (valid only if doq)
  for (uint32_t i = tid; i < G * D; i += nth) { qs[i] = q[i]; acc[i] = 0.0f; }
  for (uint32_t h = tid; h < G; h += nth) { mst[h] = psk::neg_inf(); lst[h] = 0.0f; }
  if (doq)
    for (uint32_t i = tid; i < D; i += nth) kt[i] = fabsf(kf[qrow + i]);
  sync.arrive_and_wait();

  // per-head q scale (thread h < G) and per-row K scale (thread G): max|x| / 127
  for (uint32_t u = tid; u < G + 1; u += nth) {
    if (u < G) {
      float mx = 0.0f;
      for (uint32_t d = 0; d < D; ++d) mx = __builtin_fmaxf(mx, fabsf(qs[u * D + d]));
      qsc[u] = mx / 127.0f;
    } else if (doq) {
      float mx = 0.0f;
      for (uint32_t d = 0; d < D; ++d) mx = __builtin_fmaxf(mx, kt[d]);
      ksc0[0] = mx / 127.0f;
    }
  }
  sync.arrive_and_wait();

  for (uint32_t i = tid; i < G * D; i += nth) {
    const float s = qsc[i / D];
    int32_t v = 0;
    if (s != 0.0f) v = (int32_t)roundf(qs[i] / s);
    qb[i] = (int8_t)v;
    qi[i] = v;
  }
  if (doq) {
    const float s = ksc0[0];
    for (uint32_t i = tid; i < D; i += nth) {
      int32_t v = 0;
      if (s != 0.0f) v = (int32_t)roundf(kf[qrow + i] / s);
      kq[qrow + i] = (int8_t)v;
    }
    if (tid == 0) ks_g[qp] = s;
  }
  sync.arrive_and_wait();

  const uint32_t ntiles = (T + Tt - 1) / Tt;
  for (uint32_t j = 0; j < ntiles; ++j) {
    const uint32_t t0 = j * Tt;
    const uint32_t nt = (T - t0 < Tt) ? T - t0 : Tt;
    const int8_t* krows = kq + seq_off + (size_t)t0 * row_stride;
    const float* vrows = vbase + seq_off + (size_t)t0 * row_stride;

    // S: scores of the tile, thread per position, all G heads (int8 dot product)
    for (uint32_t tt = tid; tt < nt; tt += nth) {
      const uint32_t* kw = reinterpret_cast<const uint32_t*>(krows + (size_t)tt * row_stride);
      int32_t is[G];
#pragma unroll
      for (uint32_t h = 0; h < G; ++h) is[h] = 0;
      uint32_t d4 = 0;
      for (; d4 + 4 <= DW; d4 += 4) {
        uint32_t w0 = kw[d4], w1 = kw[d4 + 1], w2 = kw[d4 + 2], w3 = kw[d4 + 3];
        PSK8_FENCE(w0); PSK8_FENCE(w1); PSK8_FENCE(w2); PSK8_FENCE(w3);   // all four loads issued before the first use
        const uint32_t wv[4] = {w0, w1, w2, w3};
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
          for (uint32_t h = 0; h < G; ++h) {
            if (DOT) {
              ps_dot4acc(is[h], wv[u], qw[h * DW + d4 + u]);
            } else {
#pragma unroll
              for (int k = 0; k < 4; ++k) is[h] += sext8(wv[u], k) * qi[h * D + 4 * (d4 + u) + k];
            }
          }
        }
      }
      for (; d4 < DW; ++d4) {
        const uint32_t w = kw[d4];
#pragma unroll
        for (uint32_t h = 0; h < G; ++h) {
          if (DOT) {
            ps_dot4acc(is[h], w, qw[h * DW + d4]);
          } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) is[h] += sext8(w, k) * qi[h * D + 4 * d4 + k];
          }
        }
      }
      const float sk = ks_g[t0 + tt];
#pragma unroll
      for (uint32_t h = 0; h < G; ++h) ps[h * Tt + tt] = ((float)is[h] * (qsc[h] * sk)) * scale;
    }
    sync.arrive_and_wait();

    // M1: running max and rescale factor, thread per head (as the stock kernel)
    for (uint32_t h = tid; h < G; h += nth) {
      const float mo = mst[h];
      float mx = mo;
      for (uint32_t tt = 0; tt < nt; ++tt) mx = __builtin_fmaxf(mx, ps[h * Tt + tt]);
      cst[h] = psk::ps_expf(mo - mx);
      mst[h] = mx;
    }
    sync.arrive_and_wait();

    // M2: p = exp(s - m'), thread per (head, position)
    for (uint32_t i = tid; i < G * nt; i += nth) {
      const uint32_t h = i / nt, tt = i - h * nt;
      ps[h * Tt + tt] = psk::ps_expf(ps[h * Tt + tt] - mst[h]);
    }
    sync.arrive_and_wait();

    // V: running sums and outputs (fp32 V)
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
        const float vd = vrows[tt * row_stride + d];
#pragma unroll
        for (uint32_t h = 0; h < G; ++h) a[h] = __builtin_fmaf(ps[h * Tt + tt], vd, a[h]);
      }
#pragma unroll
      for (uint32_t h = 0; h < G; ++h) acc[h * D + d] = a[h];
    }
    sync.arrive_and_wait();
  }

  for (uint32_t i = tid; i < G * D; i += nth) out[i] = acc[i] / lst[i / D];
}

template <bool DOT>
PSK8_INL void dispatch_k8(const ps_attn_k8_arg_t* arg) {
  switch (arg->a.G) {   // the host only launches G in {1, 2, 4, 8}
    case 1: attn_k8_cta<1, DOT>(arg); break;
    case 2: attn_k8_cta<2, DOT>(arg); break;
    case 4: attn_k8_cta<4, DOT>(arg); break;
    case 8: attn_k8_cta<8, DOT>(arg); break;
    default: break;
  }
}

}  // namespace

__kernel void attn_k8_dot(ps_attn_k8_arg_t* __UNIFORM__ arg) { dispatch_k8<true>(arg); }
__kernel void attn_k8_sw(ps_attn_k8_arg_t* __UNIFORM__ arg) { dispatch_k8<false>(arg); }
