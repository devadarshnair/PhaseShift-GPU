// common.h -- ps_rmsnorm kernel arguments (shared by host main.cpp and device kernel.cpp).
// out[b][i] = w[i] * (ss_b * x[b][i]),  ss_b = 1 / sqrtf(sum_i x[b][i]^2 / n + eps)   (run.c rmsnorm, lines 182-195)
// x, out: B rows of n float32 (row-major, row stride n); w: n float32 (one layer's RMSNorm weight, shared by rows).
#ifndef PS_RMSNORM_COMMON_H
#define PS_RMSNORM_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t x_addr;    // B x n
  uint64_t w_addr;    // n
  uint64_t out_addr;  // B x n (may equal x_addr: each warp reads its whole row before writing it)
  uint32_t B;         // rows (batched decode: one row per sequence)
  uint32_t n;         // row length (dim)
  float eps;          // 1e-5f in run.c
  uint32_t pad;
} ps_rmsnorm_arg_t;

#endif
