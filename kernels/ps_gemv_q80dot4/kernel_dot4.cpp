// kernel_dot4.cpp -- REAL int8 dot4 instruction arms on the SW*' Q8 GEMV (T2 "next decisive
// experiment" 1). Copy of kernel_dq.cpp (same register-fenced kernel_nsp.cpp structure, F1 = 1 group in flight,
// F2 = 2 groups) with the dot4 PROXY replaced by a real SimX instruction (patch patches/vortex/dot4/, tree psdot):
//   custom-3 opcode 0x7B, R-type, funct3 = 0:
//     funct7 = 0  DOT4    rd  = sum_{k=0..3} int8(rs1[8k+7:8k]) * int8(rs2[8k+7:8k])           (2 integer sources)
//     funct7 = 1  DOT4ACC rd += sum_{k=0..3} int8(rs1[k]) * int8(rs2[k])  (accumulating, like NVIDIA DP4A; rd is also a source,
//                                                                          i.e. 3 integer register reads)
//   Both are exact int32 arithmetic (|sum| <= 4 * 128 * 128 < 2^31): the result equals the unpacked 4 mul + 3 add of the
//   exact path, so the kernel is BIT-EXACT against the host runq reference (PSBITEXACT mismatches=0 is required).
// Arms (the entry is gemv_q80_<variant>; "2" = U 2 groups in flight):
//   dot_d  / dot_d2  : DOT4 then an ordinary `add` (instruction count = the proxy: mul + add, but a real op)
//   dot_dh / dot_dh2 : dot_d plus the 4-load hoist (arm B)
//   dot_c  / dot_c2  : DOT4ACC (one instruction per word: lw w, lw x, dot4acc)
//   dot_ch / dot_ch2 : dot_c plus the 4-load hoist
//   dq_ab / dq_ab2   : the mul+add PROXY with hoist (NON-BIT-EXACT; control: its cycles on psdot must equal 's)
// The latency of the op is a SimX build parameter (VX_CFG_PS_DOT4_LATENCY, 3 = the MDV pipe's kMulDivLatency, 4 = one
// extra adder-tree stage); the kernel binary is the same for both.
#include <math.h>
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#pragma clang fp contract(off)

namespace {

#define PSD_INL inline __attribute__((always_inline))
#define PSD_FENCE(v) asm volatile("" : "+r"(v))

// real instructions (funct7 0 / 1), see the header comment
PSD_INL int32_t ps_dot4(uint32_t a, uint32_t b) {
  int32_t r;
  asm(".insn r 0x7b, 0, 0, %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
PSD_INL void ps_dot4acc(int32_t& acc, uint32_t a, uint32_t b) {
  asm(".insn r 0x7b, 0, 1, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b));
}

// MODE: 1 = mul+add proxy, 2 = DOT4 + add, 3 = DOT4ACC
// one weight word against B activation words
template <int B, int MODE>
PSD_INL void dq_word(uint32_t wv, const uint32_t* const* xw, uint32_t idx, int32_t* iv) {
#pragma unroll
  for (int b = 0; b < B; ++b) {
    if (MODE == 1) iv[b] += (int32_t)(wv * xw[b][idx]);   // mul + add (timing proxy for dot4)
    else if (MODE == 2) iv[b] += ps_dot4(wv, xw[b][idx]);
    else ps_dot4acc(iv[b], wv, xw[b][idx]);
    PSD_FENCE(iv[b]);
  }
}

template <int B, int GSC, int U, int MODE, bool HOIST>
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
        dq_word<B, MODE>(w0, xw, (g0 + u) * GW + k, iv[u]);
        dq_word<B, MODE>(w1, xw, (g0 + u) * GW + k + 1, iv[u]);
        dq_word<B, MODE>(w2, xw, (g0 + u) * GW + k + 2, iv[u]);
        dq_word<B, MODE>(w3, xw, (g0 + u) * GW + k + 3, iv[u]);
      }
    }
  } else {
#pragma unroll
    for (uint32_t k = 0; k < GW; ++k) {
#pragma unroll
      for (int u = 0; u < U; ++u) dq_word<B, MODE>(w[(g0 + u) * GW + k], xw, (g0 + u) * GW + k, iv[u]);
    }
  }
}

template <int B, int GSC, int U, int MODE, bool HOIST>
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
      dq_sums<B, GSC, U, MODE, HOIST>(w, xw, g, iv);
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const float s = ws[g + u];
#pragma unroll
        for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[u][b] * s) * xs[b][g + u];
      }
    }
    for (; g < G; ++g) {   // tail (U = 2, odd G)
      int32_t iv[1][B];
      dq_sums<B, GSC, 1, MODE, HOIST>(w, xw, g, iv);
      const float s = ws[g];
#pragma unroll
      for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[0][b] * s) * xs[b][g];
    }
#pragma unroll
    for (int b = 0; b < B; ++b) y[(size_t)b * M + row] = acc[b];
  }
}

}  // namespace

#define PSD_PATH(NAME, U, B, GSC, MODE, HOIST) \
  extern "C" __attribute__((noinline, used)) void NAME(const ps_gemv_q80_arg_t* arg) { dq_body<B, GSC, U, MODE, HOIST>(arg); }
#define PSD_ARM(PFX, U, MODE, HOIST, B) \
  PSD_PATH(PFX##_B##B##_GS32, U, B, 32, MODE, HOIST) PSD_PATH(PFX##_B##B##_GS64, U, B, 64, MODE, HOIST)
#define PSD_ARM4(PFX, U, MODE, HOIST) PSD_ARM(PFX, U, MODE, HOIST, 1) PSD_ARM(PFX, U, MODE, HOIST, 2) PSD_ARM(PFX, U, MODE, HOIST, 3) PSD_ARM(PFX, U, MODE, HOIST, 4)
#define PSD_ARM2(PFX, U, MODE, HOIST) PSD_ARM(PFX, U, MODE, HOIST, 1) PSD_ARM(PFX, U, MODE, HOIST, 2)
PSD_ARM4(dqab, 1, 1, true)
PSD_ARM2(dqab2, 2, 1, true)
PSD_ARM4(dotd, 1, 2, false)
PSD_ARM4(dotdh, 1, 2, true)
PSD_ARM4(dotc, 1, 3, false)
PSD_ARM4(dotch, 1, 3, true)
PSD_ARM2(dotd2, 2, 2, false)
PSD_ARM2(dotdh2, 2, 2, true)
PSD_ARM2(dotc2, 2, 3, false)
PSD_ARM2(dotch2, 2, 3, true)

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
PSD_ENTRY4(gemv_q80_dq_ab, dqab)
PSD_ENTRY2(gemv_q80_dq_ab2, dqab2)
PSD_ENTRY4(gemv_q80_dot_d, dotd)
PSD_ENTRY4(gemv_q80_dot_dh, dotdh)
PSD_ENTRY4(gemv_q80_dot_c, dotc)
PSD_ENTRY4(gemv_q80_dot_ch, dotch)
PSD_ENTRY2(gemv_q80_dot_d2, dotd2)
PSD_ENTRY2(gemv_q80_dot_dh2, dotdh2)
PSD_ENTRY2(gemv_q80_dot_c2, dotc2)
PSD_ENTRY2(gemv_q80_dot_ch2, dotch2)
