// common_kv.h -- (methodology §18.9): arguments and LMEM layout of the reduced-precision-KV decode attention entries
// attn_gqa_kv8, attn_dxa_kv8 (int8 KV, one fp32 scale per (position, KV head)) and attn_gqa_kv16, attn_dxa_kv16 (fp16 KV, Zfhmin; no scales),
// kernel_kv.cpp. Shared by the host main.cpp and the device kernels; ps_attn_arg_t of common.h (q, o, B, H_q, H_kv, D, T, Tmax, kv_mul, G, Tt,
// depth, scale) is reused unchanged as the first member, so the stock entries and their argument layout are untouched.
//
// Cache layouts (the stock layout with a narrower element type; the K / V base addresses a.k_addr / a.v_addr point at them):
//   K8, V8  int8   [B][Tmax][H_kv][D]            (K16, V16: fp16, same index)
//   Ks, Vs  float  [B][H_kv][Tmax]               scales, kv8 only: scale of (b, t, kvh) at ((b * H_kv + kvh) * Tmax + t). Why this layout and not
//                                                position-major: the scales of the Tt positions of a key tile are then CONTIGUOUS (one 4 B x Tt run
//                                                per CTA and tile: coalesced loads, one 1-D DXA run), instead of Tt strided 4-byte reads.
// Quantization (design): symmetric, one group = the D elements of one (position, KV head); scale = max|v| / 127.0f; q = (int8) round(v /
// scale), exactly runq.c's `quantize` (ps_q80.h / model/ref_q80.py); dequantization = (float) q * scale, a float32 product, so the device operand
// of the attention equals the host / NumPy dequantized operand bit for bit.
#ifndef PS_DECODE_ATTN_COMMON_KV_H
#define PS_DECODE_ATTN_COMMON_KV_H

#include "common.h"

typedef struct {
  ps_attn_arg_t a;       // a.k_addr / a.v_addr: the int8 (esz = 1) or fp16 (esz = 2) caches
  uint64_t ks_addr;      // float32 K scales [B][H_kv][Tmax]   (esz = 1 only)
  uint64_t vs_addr;      // float32 V scales
  uint32_t esz;          // element bytes: 1 (int8 + scales) or 2 (fp16)
  uint32_t pad;
} ps_attn_kv_arg_t;

// DXA descriptor slots: K, V (4-D, element esz bytes), Ks, Vs (3-D over [B][H_kv][Tmax], 4-byte elements; kv8 only)
#define PS_DXA_DESC_KS 2u
#define PS_DXA_DESC_VS 3u

// LMEM layout of one CTA, in 32-bit words, each region rounded up to 16 words (64 B):
//   qs[G][D] | acc[G][D] | ps[G][Tt] | stat[3][G] | ksc[nst][Tt] | vsc[nst][Tt] (kv8 only) | Kb[depth][Tt*D*esz B] | Vb[depth][..] (dxa only)
// nst = depth for dxa (a scale tile per stage), 1 for the plain kernel (the CTA's threads copy the tile's scales to LMEM themselves).
static inline uint32_t ps_kv_round16(uint32_t n) { return (n + 15u) & ~15u; }
typedef struct {
  uint32_t qs, acc, ps, stat, ksc, vsc, kb, vb, total;   // word offsets and total words
  uint32_t sc_stage, data_stage;                         // words per stage of the scales / of the K (or V) data tile
} ps_attn_kv_lmem_t;
static inline ps_attn_kv_lmem_t ps_attn_kv_lmem(uint32_t G, uint32_t D, uint32_t Tt, uint32_t depth, int dxa, uint32_t esz) {
  ps_attn_kv_lmem_t L;
  const uint32_t nst = dxa ? depth : 1u;
  L.sc_stage = (esz == 1) ? ps_kv_round16(Tt) : 0u;
  L.data_stage = ps_kv_round16((Tt * D * esz + 3u) / 4u);
  L.qs = 0;
  L.acc = L.qs + ps_kv_round16(G * D);
  L.ps = L.acc + ps_kv_round16(G * D);
  L.stat = L.ps + ps_kv_round16(G * Tt);
  L.ksc = L.stat + ps_kv_round16(3 * G);
  L.vsc = L.ksc + nst * L.sc_stage;
  L.kb = L.vsc + nst * L.sc_stage;
  L.vb = L.kb + (dxa ? depth * L.data_stage : 0u);
  L.total = L.vb + (dxa ? depth * L.data_stage : 0u);
  return L;
}

#endif
