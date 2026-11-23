// NT-generic copy (warp width PSW = VX_CFG_NUM_THREADS, was 32). Original header follows.
// kernel.cpp -- WARP-COOPERATIVE, deep-MLP dot4 Q8_0 GEMV on the psdot/psdotw tree
// (real DOT4ACC instruction of patches/vortex/dot4/0001-dot4-simx.patch: custom-3 0x7B, funct3 0, funct7 1, rd += sum_k int8*int8).
// One warp (block of 32 threads) computes a tile of R = 32 / S weight rows for all B <= 4 activation vectors:
//   * S lanes per row split the K words of every Q8_0 group (lane = s * R + r); with the host weight re-layout of common.h one
//     warp-wide lw reads 128 contiguous bytes (adjacent words in adjacent lanes) instead of 32 rows' lines (thread-per-row kernel);
//   * U groups per step: all U * GW / S weight words of a lane are loaded first (independent lw, register-fenced so they are all in
//     flight before the first use: 8-16 loads per lane), then per word B activation lw (same word for the R lanes of a row: warp-
//     shared, L1 hits) and B DOT4ACC;
//   * per group the S partial int32 sums of a row are combined by log2(S) SHFL.BFLY (Vortex warp shuffle, vx_intrinsics.h; whole
//     warp: cval 31, mask 0) + add. int32 addition is exact (|ival| <= GS * 127^2 < 2^31), so the order does not matter;
//   * epilogue acc = acc + ((float)ival * ws) * xs per group in group order with fp-contract off, exactly runq.c: BIT-EXACT.
//     All S lanes of a row compute it (no divergence); lane s = 0 stores.
// Variants gemv_q80_w<S>u<U>.
#include <math.h>
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"
#ifndef VX_CFG_NUM_THREADS
#error need VX_CFG_NUM_THREADS
#endif
constexpr int PSW = VX_CFG_NUM_THREADS;  // warp width (was the literal 32)

#pragma clang fp contract(off)

namespace {
#define PSD_INL inline __attribute__((always_inline))
#define PSD_FENCE(v) asm volatile("" : "+r"(v))

PSD_INL void ps_dot4acc(int32_t& acc, uint32_t a, uint32_t b) {
  asm(".insn r 0x7b, 0, 1, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b));
}

template <int B, int GSC, int S, int U>
PSD_INL void step(const uint32_t* wl, const uint32_t* const* xw, uint32_t g0, uint32_t s, int32_t (&iv)[U][B]) {
  constexpr int GW = GSC / 4, WL = GW / S, R = PSW / S;
  uint32_t w[U][WL];
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int i = 0; i < WL; ++i) w[u][i] = wl[((size_t)u * WL + i) * PSW];
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int i = 0; i < WL; ++i) PSD_FENCE(w[u][i]);   // every weight load issued before any use
#pragma unroll
  for (int u = 0; u < U; ++u) {
#pragma unroll
    for (int b = 0; b < B; ++b) iv[u][b] = 0;
#pragma unroll
    for (int i = 0; i < WL; ++i) {
      const uint32_t xi = (g0 + u) * GW + i * S + s;
#pragma unroll
      for (int b = 0; b < B; ++b) ps_dot4acc(iv[u][b], w[u][i], xw[b][xi]);
    }
#pragma unroll
    for (int b = 0; b < B; ++b)
#pragma unroll
      for (int d = R; d < PSW; d <<= 1) iv[u][b] += (int32_t)vx_shfl_bfly((size_t)iv[u][b], d, PSW - 1, 0);
  }
}

template <int B, int GSC, int S, int U>
PSD_INL void body(const ps_gemv_q80dot4wn_arg_t* arg) {
  constexpr int GW = GSC / 4, WL = GW / S, R = PSW / S;
  const uint32_t* Wl = reinterpret_cast<const uint32_t*>(arg->Wl_addr);
  const float* Wsl = reinterpret_cast<const float*>(arg->Wsl_addr);
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N, G = arg->G;
  const uint32_t lane = threadIdx.x, s = lane / R, r = lane % R;
  const uint32_t t = blockIdx.x, row = t * R + r;
  const uint32_t* xw[B];
  const float* xs[B];
#pragma unroll
  for (int b = 0; b < B; ++b) {
    xw[b] = reinterpret_cast<const uint32_t*>(arg->xq_addr + (uint64_t)b * N);
    xs[b] = reinterpret_cast<const float*>(arg->xs_addr) + (size_t)b * G;
  }
  const uint32_t* wl = Wl + (size_t)t * G * WL * PSW + lane;
  const float* ws = Wsl + (size_t)t * G * R + r;
  float acc[B];
#pragma unroll
  for (int b = 0; b < B; ++b) acc[b] = 0.0f;
  uint32_t g = 0;
  for (; g + U <= G; g += U) {
    int32_t iv[U][B];
    step<B, GSC, S, U>(wl + (size_t)g * WL * PSW, xw, g, s, iv);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const float sc = ws[(size_t)(g + u) * R];
#pragma unroll
      for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[u][b] * sc) * xs[b][g + u];
    }
  }
  for (; g < G; ++g) {
    int32_t iv[1][B];
    step<B, GSC, S, 1>(wl + (size_t)g * WL * PSW, xw, g, s, iv);
    const float sc = ws[(size_t)g * R];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[0][b] * sc) * xs[b][g];
  }
  if (s == 0 && row < M) {
#pragma unroll
    for (int b = 0; b < B; ++b) y[(size_t)b * M + row] = acc[b];
  }
}
}  // namespace

#define PSW_PATH(NAME, B, GSC, S, U) \
  extern "C" __attribute__((noinline, used)) void NAME(const ps_gemv_q80dot4wn_arg_t* arg) { body<B, GSC, S, U>(arg); }
#define PSW_B(PFX, S, U, B) PSW_PATH(PFX##_B##B##_GS32, B, 32, S, U) PSW_PATH(PFX##_B##B##_GS64, B, 64, S, U)
#define PSW_ARM(PFX, S, U)                                                                      \
  PSW_B(PFX, S, U, 1) PSW_B(PFX, S, U, 2) PSW_B(PFX, S, U, 3) PSW_B(PFX, S, U, 4)                \
  __kernel void gemv_q80_##PFX(ps_gemv_q80dot4wn_arg_t* __UNIFORM__ arg) {                       \
    const uint32_t gs_ = arg->GS;                                                               \
    switch (arg->B) {                                                                           \
      case 1: if (gs_ == 32) PFX##_B1_GS32(arg); else PFX##_B1_GS64(arg); break;                \
      case 2: if (gs_ == 32) PFX##_B2_GS32(arg); else PFX##_B2_GS64(arg); break;                \
      case 3: if (gs_ == 32) PFX##_B3_GS32(arg); else PFX##_B3_GS64(arg); break;                \
      default: if (gs_ == 32) PFX##_B4_GS32(arg); else PFX##_B4_GS64(arg); break;               \
    }                                                                                           \
  }
PSW_ARM(w1u1, 1, 1)
PSW_ARM(w1u2, 1, 2)
PSW_ARM(w2u1, 2, 1)
PSW_ARM(w2u2, 2, 2)
PSW_ARM(w2u4, 2, 4)
PSW_ARM(w4u2, 4, 2)
PSW_ARM(w4u4, 4, 4)
