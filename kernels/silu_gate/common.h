// common.h -- ps_silu_gate kernel arguments (host main.cpp and device kernel.cpp).
// out[j] = hb[j] * (1 / (1 + expf(-hb[j]))) * hb2[j]   over all B x hidden_dim elements (run.c SwiGLU, lines 338-345)
#ifndef PS_SILU_GATE_COMMON_H
#define PS_SILU_GATE_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t hb_addr;   // B x hidden (w1 @ x)
  uint64_t hb2_addr;  // B x hidden (w3 @ x)
  uint64_t out_addr;  // B x hidden (may equal hb_addr, as run.c writes s->hb in place)
  uint32_t n;         // total elements = B * hidden_dim (rows are contiguous; the op is element-wise)
  uint32_t pad;
} ps_silu_gate_arg_t;

#endif
