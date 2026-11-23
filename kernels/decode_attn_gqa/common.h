// common.h -- ps_decode_attn_gqa kernel arguments and LMEM layout (shared by host main.cpp and device kernel.cpp).
//. One decode step of one layer: for every sequence b and query head h,
//     out[b][h] = softmax_t( q[b][h] . K[b][t][h / kv_mul] * scale ) @ V[b][t][h / kv_mul],   t < T,
// float32. Layouts (run.c's per-layer key_cache/value_cache generalised to batch, the KV layout without the
// layer index): q, out [B][H_q][D]; K, V [B][Tmax][H_kv][D] (Tmax = cache capacity >= T).
#ifndef PS_DECODE_ATTN_COMMON_H
#define PS_DECODE_ATTN_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t q_addr;
  uint64_t k_addr;
  uint64_t v_addr;
  uint64_t o_addr;
  uint32_t B, H_q, H_kv, D;
  uint32_t T;       // context length (keys 0..T-1 are read)
  uint32_t Tmax;    // cache capacity (position stride of the sequence dimension)
  uint32_t kv_mul;  // H_q / H_kv (run.c's kv_mul): query head h reads KV head h / kv_mul
  uint32_t G;       // query heads per CTA: kv_mul (gqa, dxa) or 1 (naive)
  uint32_t Tt;      // key positions per tile
  uint32_t depth;   // DXA stages (dxa only; 1 = no overlap)
  float scale;      // 1 / sqrtf(D), as run.c divides by sqrtf(head_size)
  uint32_t pad;
} ps_attn_arg_t;

// DXA descriptor slots (programmed by the host, PS_DXA_DESC_K/V)
#define PS_DXA_DESC_K 0u
#define PS_DXA_DESC_V 1u

// LMEM layout of one CTA, in floats, each region rounded up to 16 floats (64 B):
//   qs[G][D] | acc[G][D] | ps[G][Tt] | stat[3][G] (m, l, c) | Ks[depth][Tt][D] | Vs[depth][Tt][D] (dxa only)
static inline uint32_t ps_attn_round16(uint32_t n) { return (n + 15u) & ~15u; }
typedef struct {
  uint32_t qs, acc, ps, stat, ks, vs, total;   // float offsets and total floats
} ps_attn_lmem_t;
static inline ps_attn_lmem_t ps_attn_lmem(uint32_t G, uint32_t D, uint32_t Tt, uint32_t depth, int dxa) {
  ps_attn_lmem_t L;
  L.qs = 0;
  L.acc = L.qs + ps_attn_round16(G * D);
  L.ps = L.acc + ps_attn_round16(G * D);
  L.stat = L.ps + ps_attn_round16(G * Tt);
  L.ks = L.stat + ps_attn_round16(3 * G);
  L.vs = L.ks + (dxa ? depth * ps_attn_round16(Tt * D) : 0);
  L.total = L.vs + (dxa ? depth * ps_attn_round16(Tt * D) : 0);
  return L;
}

#endif
