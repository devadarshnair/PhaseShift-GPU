// kernel_pf.cpp -- (a): software TOUCH-AHEAD control on the bit-exact 4-load hoist (dq_b / dq_b2, kernel_dq.cpp).
// The cheap control for the hardware stream prefetcher: at the first group of every 64 B line of its weight row, each lane issues
// ONE extra load, `lw zero, off(addr)`, to the line DL lines further along the SAME row (guarded: only inside the row). A load to x0
// reserves no scoreboard slot (core.cpp: wb is false for an integer x0 destination), so it never stalls the warp and returns no value
// (it is the RISC-V "touch" idiom; Vortex has no prefetch instruction); the line is allocated in L1 and L2 like any load. Values are
// untouched, so the kernel is bit-exact (the host still requires 0 mismatches). Cost per touch: one LSU request slot and ~5 ALU
// instructions per line. Arms: pf<DL>_b (U = 1 group in flight) and pf<DL>_b2 (U = 2, B <= 2), DL = 1, 2, 4 lines ahead.
#include <math.h>
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#pragma clang fp contract(off)

namespace {

#define PSD_INL inline __attribute__((always_inline))
PSD_INL int32_t dq_byte(uint32_t w, int k) { return (int32_t)(w << (24 - 8 * k)) >> 24; }
PSD_INL int32_t dq_dot4(uint32_t wv, uint32_t xv) {
  return dq_byte(wv, 0) * dq_byte(xv, 0) + dq_byte(wv, 1) * dq_byte(xv, 1) + dq_byte(wv, 2) * dq_byte(xv, 2) +
         dq_byte(wv, 3) * dq_byte(xv, 3);
}
#define PSD_FENCE(v) asm volatile("" : "+r"(v))

template <int B>
PSD_INL void dq_word(uint32_t wv, const uint32_t* const* xw, uint32_t idx, int32_t* iv) {
#pragma unroll
  for (int b = 0; b < B; ++b) {
    iv[b] += dq_dot4(wv, xw[b][idx]);
    PSD_FENCE(iv[b]);
  }
}

// touch the line DL lines ahead of word index g * GW of the row, once per 64 B line (words per line = 16)
template <int GSC, int DL>
PSD_INL void pf_touch(const uint32_t* w, uint32_t g, uint32_t N4) {
  constexpr uint32_t GW = GSC >> 2;
  const uint32_t cur = g * GW;
  if ((cur & 15u) == 0u && cur + 16u * DL < N4) asm volatile("lw zero, 0(%0)" ::"r"(w + cur + 16u * DL));
}

template <int B, int GSC, int U>
PSD_INL void dq_sums(const uint32_t* w, const uint32_t* const* xw, uint32_t g0, int32_t (&iv)[U][B]) {
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int b = 0; b < B; ++b) iv[u][b] = 0;
  constexpr uint32_t GW = GSC >> 2;
#pragma unroll
  for (uint32_t k = 0; k < GW; k += 4) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      uint32_t w0 = w[(g0 + u) * GW + k], w1 = w[(g0 + u) * GW + k + 1], w2 = w[(g0 + u) * GW + k + 2], w3 = w[(g0 + u) * GW + k + 3];
      asm volatile("" : "+r"(w0), "+r"(w1), "+r"(w2), "+r"(w3));
      dq_word<B>(w0, xw, (g0 + u) * GW + k, iv[u]);
      dq_word<B>(w1, xw, (g0 + u) * GW + k + 1, iv[u]);
      dq_word<B>(w2, xw, (g0 + u) * GW + k + 2, iv[u]);
      dq_word<B>(w3, xw, (g0 + u) * GW + k + 3, iv[u]);
    }
  }
}

template <int B, int GSC, int U, int DL>
PSD_INL void pf_body(const ps_gemv_q80_arg_t* arg) {
  const int8_t* Wq = reinterpret_cast<const int8_t*>(arg->Wq_addr);
  const float* Ws = reinterpret_cast<const float*>(arg->Ws_addr);
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N, G = N / GSC, N4 = N >> 2;
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
    uint32_t g = 0;
    for (; g + U <= G; g += U) {
      pf_touch<GSC, DL>(w, g, N4);
      if (U == 2) pf_touch<GSC, DL>(w, g + 1, N4);
      int32_t iv[U][B];
      dq_sums<B, GSC, U>(w, xw, g, iv);
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const float s = ws[g + u];
#pragma unroll
        for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[u][b] * s) * xs[b][g + u];
      }
    }
    for (; g < G; ++g) {
      pf_touch<GSC, DL>(w, g, N4);
      int32_t iv[1][B];
      dq_sums<B, GSC, 1>(w, xw, g, iv);
      const float s = ws[g];
#pragma unroll
      for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[0][b] * s) * xs[b][g];
    }
#pragma unroll
    for (int b = 0; b < B; ++b) y[(size_t)b * M + row] = acc[b];
  }
}

}  // namespace

#define PSF_PATH(NAME, U, B, GSC, DL) \
  extern "C" __attribute__((noinline, used)) void NAME(const ps_gemv_q80_arg_t* arg) { pf_body<B, GSC, U, DL>(arg); }
#define PSF_ARM(PFX, U, DL, B) PSF_PATH(PFX##_B##B##_GS32, U, B, 32, DL) PSF_PATH(PFX##_B##B##_GS64, U, B, 64, DL)
#define PSF_ARM4(PFX, U, DL) PSF_ARM(PFX, U, DL, 1) PSF_ARM(PFX, U, DL, 2) PSF_ARM(PFX, U, DL, 3) PSF_ARM(PFX, U, DL, 4)
#define PSF_ARM2(PFX, U, DL) PSF_ARM(PFX, U, DL, 1) PSF_ARM(PFX, U, DL, 2)
PSF_ARM4(pf1b, 1, 1) PSF_ARM4(pf2b, 1, 2) PSF_ARM4(pf4b, 1, 4)
PSF_ARM2(pf1b2, 2, 1) PSF_ARM2(pf2b2, 2, 2) PSF_ARM2(pf4b2, 2, 4)

#define PSF_ENTRY4(ENTRY, PFX)                                                                  \
  __kernel void ENTRY(ps_gemv_q80_arg_t* __UNIFORM__ arg) {                                     \
    const uint32_t gs_ = arg->GS;                                                               \
    switch (arg->B) {                                                                           \
      case 1: if (gs_ == 32) PFX##_B1_GS32(arg); else if (gs_ == 64) PFX##_B1_GS64(arg); break; \
      case 2: if (gs_ == 32) PFX##_B2_GS32(arg); else if (gs_ == 64) PFX##_B2_GS64(arg); break; \
      case 3: if (gs_ == 32) PFX##_B3_GS32(arg); else if (gs_ == 64) PFX##_B3_GS64(arg); break; \
      default: if (gs_ == 32) PFX##_B4_GS32(arg); else if (gs_ == 64) PFX##_B4_GS64(arg); break; \
    }                                                                                           \
  }
#define PSF_ENTRY2(ENTRY, PFX)                                                                  \
  __kernel void ENTRY(ps_gemv_q80_arg_t* __UNIFORM__ arg) {                                     \
    const uint32_t gs_ = arg->GS;                                                               \
    switch (arg->B) {                                                                           \
      case 1: if (gs_ == 32) PFX##_B1_GS32(arg); else if (gs_ == 64) PFX##_B1_GS64(arg); break; \
      case 2: if (gs_ == 32) PFX##_B2_GS32(arg); else if (gs_ == 64) PFX##_B2_GS64(arg); break; \
      default: break;                                                                           \
    }                                                                                           \
  }
PSF_ENTRY4(gemv_q80_pf1_b, pf1b)
PSF_ENTRY4(gemv_q80_pf2_b, pf2b)
PSF_ENTRY4(gemv_q80_pf4_b, pf4b)
PSF_ENTRY2(gemv_q80_pf1_b2, pf1b2)
PSF_ENTRY2(gemv_q80_pf2_b2, pf2b2)
PSF_ENTRY2(gemv_q80_pf4_b2, pf4b2)
