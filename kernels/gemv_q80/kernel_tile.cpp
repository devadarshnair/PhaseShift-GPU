// kernel_tile.cpp -- fallback F4 (docs/ addendum §2.2, §5): B-tiling of the batched Q8_0 GEMV. A translation unit of its own,
// linked after kernel_nsp.cpp (kernels/gemv_q80/app.conf), so every earlier entry keeps its code.
//
//  gemv_q80_rows_nsp_tile  B = 4 as two B = 2 passes PER GROUP: the integer sums of activation vectors 0, 1 are computed over the group's weight words,
//                          then the same weight words are loaded AGAIN (from L1) for vectors 2, 3. The live set per word is that of B = 2 (the frame of the
//                          B = 2 path), at the price of 2x the weight-word loads per group (the plan's "weight L1->L2 traffic x2" for a row that does not stay in L1).
//                          The register fence is kept (the tile is the fallback AFTER the fence); the float steps are unchanged and in group order, so the result is
//                          bit-identical to runq.c. Dispatched for B = 4 only (other B do nothing: the host check FAILS on purpose).
#include <math.h>
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#pragma clang fp contract(off)

namespace {

#define PST_INL inline __attribute__((always_inline))
PST_INL int32_t t_byte(uint32_t w, int k) { return (int32_t)(w << (24 - 8 * k)) >> 24; }
PST_INL int32_t t_dot4(uint32_t wv, uint32_t xv) {
  return t_byte(wv, 0) * t_byte(xv, 0) + t_byte(wv, 1) * t_byte(xv, 1) + t_byte(wv, 2) * t_byte(xv, 2) + t_byte(wv, 3) * t_byte(xv, 3);
}

// words of one group of one row against the 2 activation vectors xw[0], xw[1]; fence after each add
template <int GSC>
PST_INL void t_sums2(const uint32_t* w, const uint32_t* const* xw, uint32_t g, uint32_t GW, int32_t& i0, int32_t& i1) {
  i0 = 0;
  i1 = 0;
  const uint32_t gw0 = GSC ? (uint32_t)(GSC >> 2) : GW;
  if (GSC) {
#pragma unroll
    for (uint32_t k = 0; k < (uint32_t)(GSC >> 2); ++k) {
      const uint32_t wv = w[g * gw0 + k];
      i0 += t_dot4(wv, xw[0][g * gw0 + k]);
      asm volatile("" : "+r"(i0));
      i1 += t_dot4(wv, xw[1][g * gw0 + k]);
      asm volatile("" : "+r"(i1));
    }
  } else {
    for (uint32_t k = 0; k < gw0; ++k) {
      const uint32_t wv = w[g * gw0 + k];
      i0 += t_dot4(wv, xw[0][g * gw0 + k]);
      asm volatile("" : "+r"(i0));
      i1 += t_dot4(wv, xw[1][g * gw0 + k]);
      asm volatile("" : "+r"(i1));
    }
  }
}

template <int GSC>
PST_INL void tile_body(const ps_gemv_q80_arg_t* arg) {
  constexpr int B = 4;
  const int8_t* Wq = reinterpret_cast<const int8_t*>(arg->Wq_addr);
  const float* Ws = reinterpret_cast<const float*>(arg->Ws_addr);
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N, gs = arg->GS, G = N / (GSC ? GSC : gs), GW = gs >> 2;
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
    for (uint32_t g = 0; g < G; ++g) {
      int32_t iv[B];
      t_sums2<GSC>(w, xw, g, GW, iv[0], iv[1]);        // tile 0: vectors 0, 1
      t_sums2<GSC>(w, xw + 2, g, GW, iv[2], iv[3]);    // tile 1: vectors 2, 3 (weight words loaded again)
      const float s = ws[g];
#pragma unroll
      for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[b] * s) * xs[b][g];
    }
#pragma unroll
    for (int b = 0; b < B; ++b) y[(size_t)b * M + row] = acc[b];
  }
}

}  // namespace

extern "C" __attribute__((noinline, used)) void nsp_tile_B4_GS32(const ps_gemv_q80_arg_t* arg) { tile_body<32>(arg); }
extern "C" __attribute__((noinline, used)) void nsp_tile_B4_GS64(const ps_gemv_q80_arg_t* arg) { tile_body<64>(arg); }
extern "C" __attribute__((noinline, used)) void nsp_tile_B4_GSG(const ps_gemv_q80_arg_t* arg) { tile_body<0>(arg); }

__kernel void gemv_q80_rows_nsp_tile(ps_gemv_q80_arg_t* __UNIFORM__ arg) {
  const uint32_t gs_ = arg->GS;
  if (arg->B != 4) return;                 // not dispatched on purpose: output keeps the host's poison, the host check FAILS
  if (gs_ == 32) nsp_tile_B4_GS32(arg); else if (gs_ == 64) nsp_tile_B4_GS64(arg); else nsp_tile_B4_GSG(arg);
}
