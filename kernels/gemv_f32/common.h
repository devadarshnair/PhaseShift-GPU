// common.h -- ps_gemv_f32 kernel arguments (shared by host main.cpp and device kernel.cpp).
// y[M] = W[M][N] @ x[N], float32, W row-major with row stride N (llama2.c / run.c `matmul` layout: W is (d, n),
// out = W @ x; fixture format).
#ifndef PS_GEMV_COMMON_H
#define PS_GEMV_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t W_addr;  // M x N float32, row-major
  uint64_t x_addr;  // N float32
  uint64_t y_addr;  // M float32
  uint32_t M;       // rows (outputs)
  uint32_t N;       // columns (reduction length K)
} ps_gemv_arg_t;

#endif
