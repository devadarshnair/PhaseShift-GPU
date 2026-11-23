// kernel.cpp -- Q8_0 device kernels (design and error model: docs/methodology.md §18.3).
//
//  gemv_q80_rows   thread per output row (grid-stride over rows), the stock thread-per-row layout of gemv_rows. A thread
//                  streams its row's int8 values as 32-bit words (4 weights per load), unpacks each byte with a left
//                  shift (byte to the top) and an arithmetic right shift by 24, multiplies with the activation byte
//                  unpacked the same way and accumulates in int32 (exact: GS * 127^2 < 2^24), per GROUP. After each group
//                  of GS weights: acc += ((float)ival * ws[g]) * xs[g] -- runq.c matmul's operation order and rounding,
//                  with the group loop sequential. Compiled with fp-contract off (below), so the result is BIT-identical
//                  to runq.c's matmul (and to model/ref_q80.py matmul_q80). All B activation vectors (B = 1..4) share each
//                  weight word: it is loaded and unpacked once and multiplied with B x words (B accumulators per thread),
//                  i.e. the weight stream is read once for the whole small batch (the batched-decode layout of gemm_small).
//  gemv_q80_lanes  warp per output row, lane l handles groups l, l + NT, ... (whole groups, so no sub-group reduction),
//                  then the NT lane partials are combined with the vx_shfl_bfly butterfly of gemv_lanes. The summation order
//                  differs from runq.c (lane partial sums, tree combine), so it is checked with the 2 gamma_{G+2} rule, not
//                  bit-exactly. It exists layout rule (rows when the NT rows a warp loads at once do not
//                  overflow an L1 set, else lanes); with int8 rows (stride N bytes) the rule picks rows for every shape in
//                  this study, so lanes is measured for completeness only.
//  quantize_q80    activation quantizer, thread per (row, group): wmax = max |x|, scale = wmax / 127.0f, q = (int8) roundf
//                  (x / scale); runq.c `quantize` with musl roundf (round half away from zero) in place of the double
//                  `round` (identical results: the double round of a float is exact). A zero group gets q = 0.
//
// The GS template parameter fixes the group size at compile time for GS = 32 and 64 (the two sizes this study uses; the
// word loop then unrolls fully); any other GS multiple of 4 takes the generic runtime-GS path.
#include <math.h>
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#pragma clang fp contract(off)   // no fused multiply-add: runq.c's separate roundings, bit-exact `rows` results

#ifndef VX_CFG_NUM_THREADS
#define VX_CFG_NUM_THREADS 4
#endif

static inline size_t bits_from_f32(float v) { uint32_t u; __builtin_memcpy(&u, &v, 4); return u; }
static inline float f32_from_bits(size_t u) { uint32_t b = (uint32_t)u; float v; __builtin_memcpy(&v, &b, 4); return v; }

// byte k (0..3) of a 32-bit word as a sign-extended int: shift the byte to the top, then arithmetic shift down
#define BYTE_S32(w, k) ((int32_t)((uint32_t)(w) << (24 - 8 * (k))) >> 24)

// int32 dot product of one weight word with one activation word (4 int8 x 4 int8)
static inline int32_t dot4(uint32_t wv, uint32_t xv) {
  return BYTE_S32(wv, 0) * BYTE_S32(xv, 0) + BYTE_S32(wv, 1) * BYTE_S32(xv, 1) +
         BYTE_S32(wv, 2) * BYTE_S32(xv, 2) + BYTE_S32(wv, 3) * BYTE_S32(xv, 3);
}

// one group g of row-pointer w (words) against the B activation vectors: acc[b] += (float(ival_b) * ws) * xs_b[g]
template <int B, int GSC>
static inline void group_step(const uint32_t* w, const float* ws, const uint32_t* const* xw, const float* const* xs,
                              uint32_t g, uint32_t gs_rt, float* acc) {
  const uint32_t GW = (GSC ? GSC : gs_rt) >> 2;   // words per group
  int32_t iv[B];
#pragma unroll
  for (int b = 0; b < B; ++b) iv[b] = 0;
  const uint32_t* wg = w + g * GW;
  if (GSC) {
#pragma unroll
    for (uint32_t k = 0; k < (uint32_t)(GSC >> 2); ++k) {
      const uint32_t wv = wg[k];
#pragma unroll
      for (int b = 0; b < B; ++b) iv[b] += dot4(wv, xw[b][g * GW + k]);
    }
  } else {
    for (uint32_t k = 0; k < GW; ++k) {
      const uint32_t wv = wg[k];
#pragma unroll
      for (int b = 0; b < B; ++b) iv[b] += dot4(wv, xw[b][g * GW + k]);
    }
  }
  const float s = ws[g];
#pragma unroll
  for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[b] * s) * xs[b][g];
}

template <int B, int GSC>
static void rows_body(const ps_gemv_q80_arg_t* arg) {
  const int8_t* Wq = reinterpret_cast<const int8_t*>(arg->Wq_addr);
  const float* Ws = reinterpret_cast<const float*>(arg->Ws_addr);
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N, gs = arg->GS, G = N / (GSC ? GSC : gs);
  const uint32_t* xw[B];
  const float* xs[B];
#pragma unroll
  for (int b = 0; b < B; ++b) {
    xw[b] = reinterpret_cast<const uint32_t*>(arg->xq_addr + (uint64_t)b * N);
    xs[b] = reinterpret_cast<const float*>(arg->xs_addr) + (size_t)b * G;
  }
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < M; row += stride) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(Wq + (size_t)row * N);
    const float* ws = Ws + (size_t)row * G;
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;
    for (uint32_t g = 0; g < G; ++g) group_step<B, GSC>(w, ws, xw, xs, g, gs, acc);
#pragma unroll
    for (int b = 0; b < B; ++b) y[(size_t)b * M + row] = acc[b];
  }
}

template <int B, int GSC>
static void lanes_body(const ps_gemv_q80_arg_t* arg) {
  const int8_t* Wq = reinterpret_cast<const int8_t*>(arg->Wq_addr);
  const float* Ws = reinterpret_cast<const float*>(arg->Ws_addr);
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N, gs = arg->GS, G = N / (GSC ? GSC : gs);
  const uint32_t* xw[B];
  const float* xs[B];
#pragma unroll
  for (int b = 0; b < B; ++b) {
    xw[b] = reinterpret_cast<const uint32_t*>(arg->xq_addr + (uint64_t)b * N);
    xs[b] = reinterpret_cast<const float*>(arg->xs_addr) + (size_t)b * G;
  }
  constexpr uint32_t NT = VX_CFG_NUM_THREADS;
  constexpr int clamp = NT - 1;
  constexpr int segmask = ~clamp & 0x3f;
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = gtid % NT;
  const uint32_t nwarps = (gridDim.x * blockDim.x) / NT;
  for (uint32_t row = gtid / NT; row < M; row += nwarps) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(Wq + (size_t)row * N);
    const float* ws = Ws + (size_t)row * G;
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;
    for (uint32_t g = lane; g < G; g += NT) group_step<B, GSC>(w, ws, xw, xs, g, gs, acc);
#pragma unroll
    for (int b = 0; b < B; ++b) {
#pragma unroll
      for (uint32_t off = NT >> 1; off > 0; off >>= 1)
        acc[b] += f32_from_bits(vx_shfl_bfly(bits_from_f32(acc[b]), (int)off, clamp, segmask));
    }
    if (lane == 0) {
#pragma unroll
      for (int b = 0; b < B; ++b) y[(size_t)b * M + row] = acc[b];
    }
  }
}

#define PS_Q80_DISPATCH(BODY, arg)                                                        \
  do {                                                                                    \
    const uint32_t gs_ = (arg)->GS;                                                       \
    switch ((arg)->B) {                                                                   \
      case 1: if (gs_ == 32) BODY<1, 32>(arg); else if (gs_ == 64) BODY<1, 64>(arg); else BODY<1, 0>(arg); break; \
      case 2: if (gs_ == 32) BODY<2, 32>(arg); else if (gs_ == 64) BODY<2, 64>(arg); else BODY<2, 0>(arg); break; \
      case 3: if (gs_ == 32) BODY<3, 32>(arg); else if (gs_ == 64) BODY<3, 64>(arg); else BODY<3, 0>(arg); break; \
      default: if (gs_ == 32) BODY<4, 32>(arg); else if (gs_ == 64) BODY<4, 64>(arg); else BODY<4, 0>(arg); break; \
    }                                                                                     \
  } while (0)

__kernel void gemv_q80_rows(ps_gemv_q80_arg_t* __UNIFORM__ arg) { PS_Q80_DISPATCH(rows_body, arg); }

__kernel void gemv_q80_lanes(ps_gemv_q80_arg_t* __UNIFORM__ arg) { PS_Q80_DISPATCH(lanes_body, arg); }

// runq.c `quantize`, one thread per (row b, group g); the group's GS values are contiguous in x (rows are multiples of GS
// long), so the flat group index gg = b * (n / GS) + g addresses x + gg * GS, q + gg * GS and s + gg.
__kernel void quantize_q80(ps_quant_q80_arg_t* __UNIFORM__ arg) {
  const float* x = reinterpret_cast<const float*>(arg->x_addr);
  int8_t* q = reinterpret_cast<int8_t*>(arg->q_addr);
  float* s = reinterpret_cast<float*>(arg->s_addr);
  const uint32_t GS = arg->GS, total = arg->B * (arg->n / GS);
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t gg = blockIdx.x * blockDim.x + threadIdx.x; gg < total; gg += stride) {
    const float* xp = x + (size_t)gg * GS;
    float wmax = 0.0f;
    for (uint32_t i = 0; i < GS; ++i) {
      const float v = fabsf(xp[i]);
      if (v > wmax) wmax = v;
    }
    const float scale = wmax / 127.0f;
    s[gg] = scale;
    uint32_t* qw = reinterpret_cast<uint32_t*>(q + (size_t)gg * GS);
    for (uint32_t i = 0; i < GS; i += 4) {
      uint32_t word = 0;
      for (uint32_t k = 0; k < 4; ++k) {
        int32_t qi = 0;
        if (scale != 0.0f) qi = (int32_t)roundf(xp[i + k] / scale);
        word |= ((uint32_t)qi & 0xffu) << (8 * k);
      }
      qw[i >> 2] = word;
    }
  }
}
