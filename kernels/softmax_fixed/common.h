// common.h -- ps_softmax_fixed kernel arguments (shared by host main.cpp and device kernel.cpp).
// dst[r][c] = softmax(src[r][:])[c] over rows of length `cols`, float32, row-major, row stride = cols.
#ifndef PS_SOFTMAX_COMMON_H
#define PS_SOFTMAX_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t src_addr;  // rows x cols float32 (the stock-copy kernels also write exp() into it, as the stock kernel does)
  uint64_t dst_addr;  // rows x cols float32
  uint32_t rows;
  uint32_t cols;
} ps_softmax_arg_t;

// Two-level per-lane summation block (see kernel.cpp "Summation"): each lane adds PS_SOFTMAX_SUM_BLOCK of its
// strided elements into a block partial before adding the partial to its running total. The host's error bound
// (main.cpp) uses the same constant, so it lives here.
#define PS_SOFTMAX_SUM_BLOCK 32u

#endif
