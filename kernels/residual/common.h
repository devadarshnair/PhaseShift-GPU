// common.h -- ps_residual kernel arguments (host main.cpp and device kernel.cpp).
// out[j] = x[j] + branch[j]   over all B x dim elements (run.c residual connections, lines 325-327 and 351-353:
// `x[i] += xb[i]`)
#ifndef PS_RESIDUAL_COMMON_H
#define PS_RESIDUAL_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t x_addr;       // B x dim (residual stream)
  uint64_t branch_addr;  // B x dim (attention or FFN output)
  uint64_t out_addr;     // B x dim (may equal x_addr: run.c updates x in place)
  uint32_t n;            // total elements = B * dim
  uint32_t pad;
} ps_residual_arg_t;

#endif
