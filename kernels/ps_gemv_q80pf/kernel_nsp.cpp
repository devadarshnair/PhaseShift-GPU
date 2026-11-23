// kernel_nsp.cpp -- spill-free Q8_0 batched GEMV (docs/, addendum §1.3, §2.2; methodology §18.11).
// A translation unit of its own, linked LAST (kernels/gemv_q80/app.conf), so the code and addresses of every earlier entry (kernel.cpp,
// kernel_sw.cpp, and ps_llama's reused copy of kernel.cpp) are unchanged.
//
//  gemv_q80_rows_nsp       F1: `rows_inl` (group step inlined, thread per row, runq.c order) plus a REGISTER FENCE after each weight word.
//  gemv_q80_rows_nsp_nu    F3 (sensitivity: "plain C" alternative to the asm fence): `rows_inl` with the word loop NOT unrolled
//                          (`#pragma nounroll`), no asm. The word loop then keeps iv[b] in loop-carried registers by construction. +5-16 % instructions
//                          vs F1 (loop overhead), reported as a sensitivity and as the planned fallback if F1 failed S1/D1.
//  gemv_q80_rows_nsp_ilp2  F2: F1 with U = 2 groups in flight per thread (fenced). Spill-free only while U * B <= 4, so it is dispatched for
//                          B = 1, 2; B = 3, 4 do nothing on purpose (the output keeps the host's NaN poison, the host check prints FAILED!).
//
// Why the fence (root cause, §1): the int32 sum iv[b] of a group is exact in any order, so -O3 re-associates the chain
// `iv[b] += dot4(...)` over the fully unrolled group into one add tree, hoists every load and product of the group above the adds, and then
// the live set (GS*B products plus the loaded words, 32..256 values) exceeds rv32's ~27 allocatable integer registers: each value is stored to
// and reloaded from the stack once per group. Stack interleaving makes that a per-core footprint of 256 x frame bytes, which exceeds the
// 256 KiB L2 at B = 4 / GS = 64. `asm volatile("" : "+r"(iv[b]))` emits NO instruction; it tells the compiler iv[b] must be in a register
// here and may have changed, so the adds cannot be deferred past the word and the live set stays at ~1 weight word + B activation words + B
// sums. Values are unchanged: the fence only constrains the ORDER of exact integer additions; the float steps are untouched and stay in
// group order under fp-contract(off), so the result is bit-identical to runq.c's matmul (checked by PSBITEXACT: runq_exact mismatches=0).
//
// Each (B, GS) path is a noinline function of its own with its own symbol (nsp_rows_B<b>_GS<32|64|G>, nsp_rows2_B<b>_GS...), so the
// static check (results/<run>/static_spill.py; names nsp_rows_*, nsp_rows2_*, nsp_nu_*) can read each path in the linked kernel.elf. Cost of the call: once per thread per launch.
#include <math.h>
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#pragma clang fp contract(off)   // runq.c's separate roundings, as kernel.cpp

namespace {

#define PSN_INL inline __attribute__((always_inline))

PSN_INL int32_t nsp_byte(uint32_t w, int k) { return (int32_t)(w << (24 - 8 * k)) >> 24; }
PSN_INL int32_t nsp_dot4(uint32_t wv, uint32_t xv) {
  return nsp_byte(wv, 0) * nsp_byte(xv, 0) + nsp_byte(wv, 1) * nsp_byte(xv, 1) + nsp_byte(wv, 2) * nsp_byte(xv, 2) +
         nsp_byte(wv, 3) * nsp_byte(xv, 3);
}
// register fence: zero instructions, forces `v` into a register here
#define PSN_FENCE(v) asm volatile("" : "+r"(v))

// U consecutive groups g0 .. g0 + U - 1 of one row against B activation vectors, one fence per weight word
// FENCE: emit the register fence after each add; NU: word loop not unrolled (#pragma nounroll), F3
template <int B, int GSC, int U, bool FENCE, bool NU>
PSN_INL void nsp_sums(const uint32_t* w, const uint32_t* const* xw, uint32_t g0, uint32_t GW, int32_t (&iv)[U][B]) {
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int b = 0; b < B; ++b) iv[u][b] = 0;
  const uint32_t gw0 = GSC ? (uint32_t)(GSC >> 2) : GW;
  if (GSC && !NU) {
#pragma unroll
    for (uint32_t k = 0; k < (uint32_t)(GSC >> 2); ++k) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint32_t wv = w[(g0 + u) * gw0 + k];
#pragma unroll
        for (int b = 0; b < B; ++b) {
          iv[u][b] += nsp_dot4(wv, xw[b][(g0 + u) * gw0 + k]);
          if (FENCE) PSN_FENCE(iv[u][b]);
        }
      }
    }
  } else {
#pragma nounroll
    for (uint32_t k = 0; k < gw0; ++k) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint32_t wv = w[(g0 + u) * gw0 + k];
#pragma unroll
        for (int b = 0; b < B; ++b) {
          iv[u][b] += nsp_dot4(wv, xw[b][(g0 + u) * gw0 + k]);
          if (FENCE) PSN_FENCE(iv[u][b]);
        }
      }
    }
  }
}

template <int B, int GSC, int U, bool FENCE, bool NU>
PSN_INL void nsp_body(const ps_gemv_q80_arg_t* arg) {
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
    uint32_t g = 0;
    for (; g + U <= G; g += U) {
      int32_t iv[U][B];
      nsp_sums<B, GSC, U, FENCE, NU>(w, xw, g, GW, iv);
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const float s = ws[g + u];
#pragma unroll
        for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[u][b] * s) * xs[b][g + u];
      }
    }
    for (; g < G; ++g) {   // tail (U = 2 only): one group at a time
      int32_t iv[1][B];
      nsp_sums<B, GSC, 1, FENCE, NU>(w, xw, g, GW, iv);
      const float s = ws[g];
#pragma unroll
      for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[0][b] * s) * xs[b][g];
    }
#pragma unroll
    for (int b = 0; b < B; ++b) y[(size_t)b * M + row] = acc[b];
  }
}

}  // namespace

// one noinline function per (variant, B, GS) path, with its own symbol
#define PSN_PATH(NAME, U, B, GSC, FENCE, NU) \
  extern "C" __attribute__((noinline, used)) void NAME(const ps_gemv_q80_arg_t* arg) { nsp_body<B, GSC, U, FENCE, NU>(arg); }
#define PSN_F1(B) PSN_PATH(nsp_rows_B##B##_GS32, 1, B, 32, true, false) PSN_PATH(nsp_rows_B##B##_GS64, 1, B, 64, true, false) PSN_PATH(nsp_rows_B##B##_GSG, 1, B, 0, true, false)
#define PSN_F2(B) PSN_PATH(nsp_rows2_B##B##_GS32, 2, B, 32, true, false) PSN_PATH(nsp_rows2_B##B##_GS64, 2, B, 64, true, false) PSN_PATH(nsp_rows2_B##B##_GSG, 2, B, 0, true, false)
#define PSN_F3(B) PSN_PATH(nsp_nu_B##B##_GS32, 1, B, 32, false, true) PSN_PATH(nsp_nu_B##B##_GS64, 1, B, 64, false, true) PSN_PATH(nsp_nu_B##B##_GSG, 1, B, 0, false, true)
PSN_F1(1) PSN_F1(2) PSN_F1(3) PSN_F1(4)
PSN_F2(1) PSN_F2(2)
PSN_F3(1) PSN_F3(2) PSN_F3(3) PSN_F3(4)

#define PSN_DISPATCH(PFX, arg)                                                                               \
  do {                                                                                                             \
    const uint32_t gs_ = (arg)->GS;                                                                                \
    switch ((arg)->B) {                                                                                            \
      case 1: if (gs_ == 32) PFX##_B1_GS32(arg); else if (gs_ == 64) PFX##_B1_GS64(arg); else PFX##_B1_GSG(arg); break; \
      case 2: if (gs_ == 32) PFX##_B2_GS32(arg); else if (gs_ == 64) PFX##_B2_GS64(arg); else PFX##_B2_GSG(arg); break; \
      default: break;                                                                                              \
    }                                                                                                              \
  } while (0)

#define PSN_ENTRY(ENTRY, PFX)                                                                                            \
  __kernel void ENTRY(ps_gemv_q80_arg_t* __UNIFORM__ arg) {                                                              \
    const uint32_t gs_ = arg->GS;                                                                                        \
    switch (arg->B) {                                                                                                    \
      case 1: if (gs_ == 32) PFX##_B1_GS32(arg); else if (gs_ == 64) PFX##_B1_GS64(arg); else PFX##_B1_GSG(arg); break; \
      case 2: if (gs_ == 32) PFX##_B2_GS32(arg); else if (gs_ == 64) PFX##_B2_GS64(arg); else PFX##_B2_GSG(arg); break; \
      case 3: if (gs_ == 32) PFX##_B3_GS32(arg); else if (gs_ == 64) PFX##_B3_GS64(arg); else PFX##_B3_GSG(arg); break; \
      default: if (gs_ == 32) PFX##_B4_GS32(arg); else if (gs_ == 64) PFX##_B4_GS64(arg); else PFX##_B4_GSG(arg); break; \
    }                                                                                                                    \
  }
PSN_ENTRY(gemv_q80_rows_nsp, nsp_rows)
PSN_ENTRY(gemv_q80_rows_nsp_nu, nsp_nu)

// B >= 3 is not dispatched on purpose (spills again at U * B > 4): the output stays NaN-poisoned and the host check FAILS
__kernel void gemv_q80_rows_nsp_ilp2(ps_gemv_q80_arg_t* __UNIFORM__ arg) { PSN_DISPATCH(nsp_rows2, arg); }
