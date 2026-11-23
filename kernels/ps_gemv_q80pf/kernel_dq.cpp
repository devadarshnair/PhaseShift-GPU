// kernel_dq.cpp -- dequant-removal ablation arms on the SW*' kernel (/.md #2).
// A copy of gemv_q80's sources (kernel*.cpp unchanged, so every earlier entry has the code and addresses it has)
// plus this translation unit, linked LAST (app.conf). The arms are the register-fenced kernel_nsp.cpp (F1 = U 1 group in
// flight, F2 = U 2 groups) with ONE change each:
//   (A) dq_a / dq_a2   : "dot4 timing proxy". The per-word unpack (4 byte extracts of w, 4 of x, 4 mul, 3 add, 1 add)
//                        is replaced by `iv += wv * xv` (one mul + one add per 32-bit word). A real dot4 op would be ONE
//                        instruction, so the proxy has one instruction more than the ISA op: it is conservative (it
//                        underestimates the gain of a real op). The result is NOT the dot product: NON-BIT-EXACT, the
//                        host skips every numeric check (main.cpp, PSDQ line). Control flow and every address are data
//                        independent, so the cycle count is valid.
//   (B) dq_b / dq_b2   : "4-load hoist" (software same-line merge). The four weight words of each 16 B segment are loaded
//                        into registers and fenced (`asm volatile("" : "+r"(w0), "+r"(w1), "+r"(w2), "+r"(w3))`, zero
//                        instructions) BEFORE the first use, so the lane issues four loads to ONE line back to back
//                        (like the earlier touch_rows4) instead of one per word interleaved with ~30 ALU instructions. Values are
//                        unchanged: bit-exact (the host still requires runq_exact == 0 mismatches).
//   (A+B) dq_ab / dq_ab2 : both.
// Spill safety: the live set per segment is 4 weight words + B x 1 activation words + U x B sums; U x B <= 4 only (as
// kernel_nsp.cpp); the static spill check is repeated in results/<run>/static_spill.txt.
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

// one weight word against B activation words, arm A or the exact dot
template <int B, bool PROXY>
PSD_INL void dq_word(uint32_t wv, const uint32_t* const* xw, uint32_t idx, int32_t* iv) {
#pragma unroll
  for (int b = 0; b < B; ++b) {
    if (PROXY) iv[b] += (int32_t)(wv * xw[b][idx]);   // mul + add (timing proxy for dot4)
    else iv[b] += dq_dot4(wv, xw[b][idx]);
    PSD_FENCE(iv[b]);
  }
}

template <int B, int GSC, int U, bool PROXY, bool HOIST>
PSD_INL void dq_sums(const uint32_t* w, const uint32_t* const* xw, uint32_t g0, int32_t (&iv)[U][B]) {
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int b = 0; b < B; ++b) iv[u][b] = 0;
  constexpr uint32_t GW = GSC >> 2;
  if (HOIST) {
#pragma unroll
    for (uint32_t k = 0; k < GW; k += 4) {          // one 16 B segment (4 words) at a time
#pragma unroll
      for (int u = 0; u < U; ++u) {
        uint32_t w0 = w[(g0 + u) * GW + k], w1 = w[(g0 + u) * GW + k + 1], w2 = w[(g0 + u) * GW + k + 2],
                 w3 = w[(g0 + u) * GW + k + 3];
        asm volatile("" : "+r"(w0), "+r"(w1), "+r"(w2), "+r"(w3));   // all four loads issued before any use
        dq_word<B, PROXY>(w0, xw, (g0 + u) * GW + k, iv[u]);
        dq_word<B, PROXY>(w1, xw, (g0 + u) * GW + k + 1, iv[u]);
        dq_word<B, PROXY>(w2, xw, (g0 + u) * GW + k + 2, iv[u]);
        dq_word<B, PROXY>(w3, xw, (g0 + u) * GW + k + 3, iv[u]);
      }
    }
  } else {
#pragma unroll
    for (uint32_t k = 0; k < GW; ++k) {
#pragma unroll
      for (int u = 0; u < U; ++u) dq_word<B, PROXY>(w[(g0 + u) * GW + k], xw, (g0 + u) * GW + k, iv[u]);
    }
  }
}

template <int B, int GSC, int U, bool PROXY, bool HOIST>
PSD_INL void dq_body(const ps_gemv_q80_arg_t* arg) {
  const int8_t* Wq = reinterpret_cast<const int8_t*>(arg->Wq_addr);
  const float* Ws = reinterpret_cast<const float*>(arg->Ws_addr);
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N, G = N / GSC;
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
      dq_sums<B, GSC, U, PROXY, HOIST>(w, xw, g, iv);
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const float s = ws[g + u];
#pragma unroll
        for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[u][b] * s) * xs[b][g + u];
      }
    }
    for (; g < G; ++g) {   // tail (U = 2, odd G)
      int32_t iv[1][B];
      dq_sums<B, GSC, 1, PROXY, HOIST>(w, xw, g, iv);
      const float s = ws[g];
#pragma unroll
      for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[0][b] * s) * xs[b][g];
    }
#pragma unroll
    for (int b = 0; b < B; ++b) y[(size_t)b * M + row] = acc[b];
  }
}

}  // namespace

#define PSD_PATH(NAME, U, B, GSC, PROXY, HOIST) \
  extern "C" __attribute__((noinline, used)) void NAME(const ps_gemv_q80_arg_t* arg) { dq_body<B, GSC, U, PROXY, HOIST>(arg); }
#define PSD_ARM(PFX, U, PROXY, HOIST, B) \
  PSD_PATH(PFX##_B##B##_GS32, U, B, 32, PROXY, HOIST) PSD_PATH(PFX##_B##B##_GS64, U, B, 64, PROXY, HOIST)
#define PSD_ARM4(PFX, U, PROXY, HOIST) PSD_ARM(PFX, U, PROXY, HOIST, 1) PSD_ARM(PFX, U, PROXY, HOIST, 2) PSD_ARM(PFX, U, PROXY, HOIST, 3) PSD_ARM(PFX, U, PROXY, HOIST, 4)
#define PSD_ARM2(PFX, U, PROXY, HOIST) PSD_ARM(PFX, U, PROXY, HOIST, 1) PSD_ARM(PFX, U, PROXY, HOIST, 2)
PSD_ARM4(dqa, 1, true, false)
PSD_ARM4(dqb, 1, false, true)
PSD_ARM4(dqab, 1, true, true)
PSD_ARM2(dqa2, 2, true, false)
PSD_ARM2(dqb2, 2, false, true)
PSD_ARM2(dqab2, 2, true, true)

// GS is 32 (stories15M) or 64 (llama8b_s8) in every class of this step; any other GS is not dispatched (NaN-poisoned output)
#define PSD_ENTRY4(ENTRY, PFX)                                                                  \
  __kernel void ENTRY(ps_gemv_q80_arg_t* __UNIFORM__ arg) {                                     \
    const uint32_t gs_ = arg->GS;                                                               \
    switch (arg->B) {                                                                           \
      case 1: if (gs_ == 32) PFX##_B1_GS32(arg); else if (gs_ == 64) PFX##_B1_GS64(arg); break; \
      case 2: if (gs_ == 32) PFX##_B2_GS32(arg); else if (gs_ == 64) PFX##_B2_GS64(arg); break; \
      case 3: if (gs_ == 32) PFX##_B3_GS32(arg); else if (gs_ == 64) PFX##_B3_GS64(arg); break; \
      default: if (gs_ == 32) PFX##_B4_GS32(arg); else if (gs_ == 64) PFX##_B4_GS64(arg); break; \
    }                                                                                           \
  }
#define PSD_ENTRY2(ENTRY, PFX)                                                                  \
  __kernel void ENTRY(ps_gemv_q80_arg_t* __UNIFORM__ arg) {                                     \
    const uint32_t gs_ = arg->GS;                                                               \
    switch (arg->B) {                                                                           \
      case 1: if (gs_ == 32) PFX##_B1_GS32(arg); else if (gs_ == 64) PFX##_B1_GS64(arg); break; \
      case 2: if (gs_ == 32) PFX##_B2_GS32(arg); else if (gs_ == 64) PFX##_B2_GS64(arg); break; \
      default: break;                                                                           \
    }                                                                                           \
  }
PSD_ENTRY4(gemv_q80_dq_a, dqa)
PSD_ENTRY4(gemv_q80_dq_b, dqb)
PSD_ENTRY4(gemv_q80_dq_ab, dqab)
PSD_ENTRY2(gemv_q80_dq_a2, dqa2)
PSD_ENTRY2(gemv_q80_dq_b2, dqb2)
PSD_ENTRY2(gemv_q80_dq_ab2, dqab2)
