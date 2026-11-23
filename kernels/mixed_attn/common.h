// common.h -- ps_mixed_attn kernel arguments (shared by host main.cpp and device kernel.cpp).
// Fused mixed-batch attention of one layer: ONE launch serves a Sarathi-style mixed iteration [cand:A5]:
//   * a prefill chunk of one sequence: query tokens at positions s0 .. s0+c-1 attend causally to keys 0 .. s0+c-1
//     (the operation, prefill_common.h; layouts q_p, o_p [c][H_q][D], K_p, V_p [Tmax_p][H_kv][D]);
//   * n_seq decode sequences: sequence b's one query attends to its keys 0 .. ctx[b]-1 (the operation,
//     decode_common.h; layouts q_d, o_d [n_seq][H_q][D], K_d, V_d [n_seq][Tmax_d][H_kv][D]; ctx[b] may differ per
//     sequence, as in a continuous batch).
// CTA split (POD-Attention-like CTA-level split [cand:B10]): CTAs [0, n_p) are prefill CTAs, one per
// (KV head, query tile of Qt tokens), numbered exactly as numbers them; CTAs [n_p, n_p + n_d) are decode CTAs,
// one per (decode sequence, KV head), numbered as the earlier gqa_shared_kv. order = 1 swaps the two ranges (decode CTAs
// first; a measured sensitivity, not the layout). Design: docs/methodology.md §14.6.
#ifndef PS_MIXED_ATTN_COMMON_H
#define PS_MIXED_ATTN_COMMON_H

#include <stdint.h>

#include "prefill_common.h"   // ps_pattn_lmem: prefill CTA LMEM layout (symlink to kernels/prefill_attn/common.h)
#include "decode_common.h"    // ps_attn_lmem: decode CTA LMEM layout (symlink to kernels/decode_attn_gqa/common.h)

typedef struct {
  // prefill chunk (B = 1 sequence)
  uint64_t qp_addr, kp_addr, vp_addr, op_addr;
  // decode sequences
  uint64_t qd_addr, kd_addr, vd_addr, od_addr;
  uint64_t ctx_addr;   // uint32_t ctx[n_seq]: context length (keys) of each decode sequence, >= 1
  uint64_t ts_addr;    // per-CTA timestamp records (PS_MATTN_TS_WORDS uint32 each), 0 = not recorded
  uint32_t H_q, H_kv, D, G;   // G = H_q / H_kv (run.c's kv_mul), shared by both parts
  // prefill geometry (names)
  uint32_t S;          // chunk length c (query tokens of the chunk)
  uint32_t s0;         // absolute position of the chunk's first token
  uint32_t Tmax_p;     // prefill cache capacity (position stride)
  uint32_t Qt, Tk;     // query tokens per CTA, key positions per key tile
  uint32_t nqt;        // query tiles = ceil(S / Qt)
  uint32_t n_p;        // prefill CTAs = H_kv * nqt (0 if S == 0)
  // decode geometry (names)
  uint32_t n_seq;      // decode sequences
  uint32_t Tmax_d;     // decode cache capacity (position stride per sequence)
  uint32_t Tt;         // key positions per decode tile
  uint32_t n_d;        // decode CTAs = n_seq * H_kv
  uint32_t order;      // 0: prefill CTAs first (layout), 1: decode CTAs first
  float scale;         // 1 / sqrtf(D), as run.c divides by sqrtf(head_size)
  uint32_t pad[3];
} ps_mattn_arg_t;

// Timestamp record of CTA i at ts_addr + i * PS_MATTN_TS_WORDS * 4 (one 64 B line per CTA, so records written by
// different cores never share a cache line): [0..1] start cycle (lo, hi), [2..3] end cycle (lo, hi), [4] core id,
// [5] region (0 = prefill, 1 = decode), [6] region-local CTA index, [7] = 0x5053 ("PS", written last).
#define PS_MATTN_TS_WORDS 16u
#define PS_MATTN_TS_MAGIC 0x5053u

#endif
