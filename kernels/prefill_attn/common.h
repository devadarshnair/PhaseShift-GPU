// common.h -- ps_prefill_attn kernel arguments and LMEM layout (shared by host main.cpp and device kernel.cpp).
//. Causal (prefill) attention of one layer: for sequence b, query token i (absolute position
// p = s0 + i) and query head h,
//     out[b][i][h] = softmax_{t <= p}( q[b][i][h] . K[b][t][h / G] * scale ) @ V[b][t][h / G],
// float32. Layouts: q, out [B][S][H_q][D] (token-major: one token's H_q heads are contiguous, as run.c's q / xb);
// K, V [B][Tmax][H_kv][D] (one layer device KV cache, methodology §14.0; Tmax = capacity). Keys
// 0 .. s0 + S - 1 must be present (the RoPE kernel wrote this chunk's K/V; s0 > 0 = chunked prefill: the earlier
// chunks' keys are already in the cache).
#ifndef PS_PREFILL_ATTN_COMMON_H
#define PS_PREFILL_ATTN_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t q_addr;
  uint64_t k_addr;
  uint64_t v_addr;
  uint64_t o_addr;
  uint32_t B, H_q, H_kv, D;
  uint32_t S;       // query tokens of this launch
  uint32_t s0;      // absolute position of query token 0 (0 = whole-prompt prefill)
  uint32_t Tmax;    // cache capacity (position stride of the sequence dimension)
  uint32_t G;       // H_q / H_kv (run.c's kv_mul): query head h reads KV head h / G
  uint32_t Qt;      // query tokens per CTA (query tile)
  uint32_t Tk;      // key positions per key tile
  uint32_t depth;   // DXA stages (dxa only; 1 = copy, then compute)
  uint32_t nqt;     // query tiles per (b, KV head) = ceil(S / Qt)
  float scale;      // 1 / sqrtf(D), as run.c divides by sqrtf(head_size)
  uint32_t pad[3];
} ps_pattn_arg_t;

// DXA descriptor slots (programmed by the host)
#define PS_PATTN_DESC_K 0u
#define PS_PATTN_DESC_V 1u

// Rows of a CTA: R = G * Qt (query token i, head h -> row r = i * G + h), padded to a multiple of the 4-row
// register block of the S and V phases (padding rows compute with q = 0 and are never stored).
static inline uint32_t ps_pattn_rows(uint32_t G, uint32_t Qt) { return (G * Qt + 3u) & ~3u; }

// LMEM layout of one CTA, in floats, each region rounded up to 16 floats (64 B):
//   qs[Rp][D] | acc[Rp][D] | ps[Rp][Tk] | stat[3][Rp] (m, l, c) | Ks[nbuf][Tk][D] | Vs[nbuf][Tk][D]
// nbuf = 1 for lmem_tiled (threads copy one K and one V tile, then compute), depth for dxa (ring of stages).
static inline uint32_t ps_pattn_round16(uint32_t n) { return (n + 15u) & ~15u; }
typedef struct {
  uint32_t rows, qs, acc, ps, stat, ks, vs, total;   // padded row count, float offsets, total floats
} ps_pattn_lmem_t;
static inline ps_pattn_lmem_t ps_pattn_lmem(uint32_t G, uint32_t D, uint32_t Qt, uint32_t Tk, uint32_t nbuf) {
  ps_pattn_lmem_t L;
  L.rows = ps_pattn_rows(G, Qt);
  L.qs = 0;
  L.acc = L.qs + ps_pattn_round16(L.rows * D);
  L.ps = L.acc + ps_pattn_round16(L.rows * D);
  L.stat = L.ps + ps_pattn_round16(L.rows * Tk);
  L.ks = L.stat + ps_pattn_round16(3 * L.rows);
  L.vs = L.ks + nbuf * ps_pattn_round16(Tk * D);
  L.total = L.vs + nbuf * ps_pattn_round16(Tk * D);
  return L;
}

#endif
