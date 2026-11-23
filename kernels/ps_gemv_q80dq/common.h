// common.h -- Q8_0 GEMV / activation-quantizer kernel arguments (host main.cpp and device kernel.cpp).
//
// Q8_0 as llama2.c runq.c [cand:D15]: a tensor of n values is split into groups of GS consecutive values; each group is
// stored as GS int8 values q and one float32 scale s, x ~ q * s. A weight matrix W (M, N) (N = reduction length, N % GS
// == 0, N % 4 == 0) is two arrays: Wq M x N int8 row-major and Ws M x (N / GS) float32 row-major (group g of row i has
// scale Ws[i * (N / GS) + g]), exactly the order runq.c's `init_quantized_tensors` / `matmul` use. Activations: xq B x N
// int8, xs B x (N / GS) float32 (one quantized vector per batch row). Output y: B x M float32 (row b = W @ x_b).
#ifndef PS_GEMV_Q80_COMMON_H
#define PS_GEMV_Q80_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t Wq_addr;  // M x N int8
  uint64_t Ws_addr;  // M x (N / GS) float32
  uint64_t xq_addr;  // B x N int8
  uint64_t xs_addr;  // B x (N / GS) float32
  uint64_t y_addr;   // B x M float32
  uint32_t M;        // output rows
  uint32_t N;        // reduction length
  uint32_t GS;       // group size (32 for stories15M, 64 for llama8b_s8)
  uint32_t B;        // activation vectors (1..4)
} ps_gemv_q80_arg_t;

// Activation quantizer (runq.c `quantize`): x B x n float32 -> q B x n int8, s B x (n / GS) float32.
typedef struct {
  uint64_t x_addr;
  uint64_t q_addr;
  uint64_t s_addr;
  uint32_t B;
  uint32_t n;
  uint32_t GS;
  uint32_t pad_;
} ps_quant_q80_arg_t;

#endif
