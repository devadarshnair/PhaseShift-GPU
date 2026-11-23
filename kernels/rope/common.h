// common.h -- ps_rope kernel arguments (host main.cpp and device kernel.cpp).
// RoPE (run.c forward, lines 264-279) fused with the KV-cache write (run.c lines 254-257 point k, v into the cache):
//   for each row b and pair i = 0, 2, ..., dim-2:
//     head_dim = i % head_size;  freq = 1 / powf(theta, head_dim / (float)head_size);  val = pos_b * freq
//     (q[i], q[i+1]) rotated by val -> q_out;  if i < kv_dim: rotated (k[i], k[i+1]) and (v[i], v[i+1]) are written
//     to row b's cache slot.
// Device KV cache layout (methodology §14 prelude): [layer][b][pos][kv_head][head_dim], i.e. run.c's
// key_cache[layer][pos][kv_dim] with a batch axis of capacity b_cap between layer and pos; row b's slot at position
// p of layer l starts at element ((l * b_cap + b) * seq_len + p) * kv_dim. K and V are two such arrays.
#ifndef PS_ROPE_COMMON_H
#define PS_ROPE_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t q_addr;      // B x dim (query projection)
  uint64_t k_addr;      // B x kv_dim (key projection, un-rotated)
  uint64_t v_addr;      // B x kv_dim (value projection)
  uint64_t pos_addr;    // B int32 positions
  uint64_t q_out_addr;  // B x dim (may equal q_addr: a thread reads its pair before writing it)
  uint64_t kc_addr;     // key cache base   [n_layers][b_cap][seq_len][kv_dim]
  uint64_t vc_addr;     // value cache base [n_layers][b_cap][seq_len][kv_dim]
  uint32_t B;           // rows in this launch (b = 0..B-1 use cache rows 0..B-1)
  uint32_t b_cap;       // batch capacity of the cache layout (>= B)
  uint32_t dim, kv_dim, head_size, seq_len;
  uint32_t layer;
  float theta;          // 10000.0f in run.c
} ps_rope_arg_t;


// ---- (methodology §18.7): the angle table variant. rope_table is a one-time setup launch that evaluates
// the same musl expressions as rope_kv for every (pos0 <= pos < pos0 + npos, pair j < head_size/2) and stores them as interleaved
// (cos, sin) float32 at tab[((pos - pos0) * (head_size / 2) + j) * 2]; rope_kv_tab then only loads them. New structs only: the
// layout of ps_rope_arg_t (and so rope_kv's code and arguments) is unchanged.
typedef struct {
  uint64_t tab_addr;    // npos x head_size/2 x {cos, sin} float32 (written by rope_table)
  uint32_t npos;        // positions pos0 .. pos0 + npos - 1, stored from index 0 (ps_llama: pos0 = 0, npos = seq_len)
  uint32_t head_size;
  float theta;          // 10000.0f
  uint32_t pos0;        // first position of the table (0 except in ps_rope's -Q window checks)
} ps_rope_table_arg_t;

typedef struct {
  ps_rope_arg_t a;      // exactly rope_kv's arguments (a.theta unused)
  uint64_t tab_addr;    // the table of rope_table (same head_size as a.head_size; positions in a.pos_addr < npos)
} ps_rope_tab_arg_t;

#endif
