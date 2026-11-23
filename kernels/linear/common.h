// common.h -- ps_linear kernel arguments (shared by host main.cpp and device kernel.cpp).
//
// A linear layer of a llama2.c-style model computes, for M token rows at once,
//     Y[M][N] = X[M][K] @ W^T,      W = (N, K) row-major (llama2.c / run.c `matmul` layout: out = W @ x)
// i.e. N = d_out, K = d_in. M = 1 is a decode GEMV, M = B in 2..16 is batched decode (small-M), M = prefill tokens.
#ifndef PS_LINEAR_COMMON_H
#define PS_LINEAR_COMMON_H

#include <stdint.h>

// FPU fp32 small-M GEMM (kernel `gemm_small`): one thread per output feature n, all M rows per thread.
typedef struct {
  uint64_t W_addr;  // N x K float32, row-major (llama2.c layout, no transpose)
  uint64_t X_addr;  // M x K float32, row-major (token rows)
  uint64_t Y_addr;  // M x N float32, row-major
  uint32_t M;       // token rows
  uint32_t N;       // output features
  uint32_t K;       // reduction length
  uint32_t pad_;
} ps_gemm_small_arg_t;

// TCU fp16-in / fp32-out GEMM (kernel `tcu_gemm`). Layout and field order identical to the stock
// tests/regression/sgemm_tcu_wg_dxa `kernel_arg_t`, so the device code is the stock code byte for byte:
//     C[M][N] = A[M][K] @ B[K][N], A = X (fp16, row-major), B = W^T (fp16, row-major K x N), C fp32 row-major.
// M, N, K are the PADDED sizes (multiples of cta_M, xtileN, tileK; see main.cpp).
typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} ps_tcu_arg_t;

#ifndef WGMMA_NRC
#define WGMMA_NRC 8   // same default as the stock sgemm_tcu_wg_dxa/common.h; the CI/TIER configs pass 32
#endif
#ifndef ITYPE
#define ITYPE fp16
#endif
#ifndef OTYPE
#define OTYPE fp32
#endif

#endif
