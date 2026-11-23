// common_kv8.h -- (methodology §18.9): arguments of rope_kv_tab_q8 (kernel_kv8.cpp): RoPE angle table, q rotated as
// rope_kv_tab, and the rotated K and the V of the step written to an INT8 cache with one fp32 scale per (position, KV head). New struct only: the
// layouts of ps_rope_arg_t / ps_rope_tab_arg_t (and so rope_kv's and rope_kv_tab's code and arguments) are unchanged.
#ifndef PS_ROPE_COMMON_KV8_H
#define PS_ROPE_COMMON_KV8_H

#include "common.h"

typedef struct {
  ps_rope_tab_arg_t t;  // t.a: as rope_kv's arguments, EXCEPT t.a.kc_addr / vc_addr are the INT8 caches [n_layers][b_cap][seq_len][kv_dim] (bytes);
                        // t.tab_addr: the angle table of rope_table
  uint64_t ks_addr;     // float32 K scales [n_layers][b_cap][H_kv][seq_len]: scale of (layer, b, kvh, pos) at ((layer * b_cap + b) * H_kv + kvh) * seq_len + pos
  uint64_t vs_addr;     // float32 V scales, same layout
  uint32_t H_kv;        // kv_dim / head_size
  uint32_t pad;
} ps_rope_q8_arg_t;

#endif
