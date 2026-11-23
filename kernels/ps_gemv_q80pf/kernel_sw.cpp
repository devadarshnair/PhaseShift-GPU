// kernel_sw.cpp -- software variants of the Q8_0 GEMV (methodology §18.4) and the over-fetch diagnostic
// kernels. A translation unit of its own, linked AFTER kernel.cpp (kernels/gemv_q80/app.conf), so that the stock entries
// (gemv_q80_rows / gemv_q80_lanes / quantize_q80) and ps_llama's reused copy of kernel.cpp are untouched by this step.
//
// Every entry here takes the stock argument struct (common.h), so the host driver and the launch geometry are shared.
//
//  gemv_q80_rows_inl     = gemv_q80_rows with the group step forced inline. The stock kernel's group_step is an OUTLINED function
//                          (clang does not inline it at B >= 2 / GS = 32), so every group pays a call, a prologue/epilogue that
//                          saves and restores callee-saved registers (the "spills" of instruction mix) and an argument
//                          setup. Inlining removes them; the arithmetic (and so the result, bit for bit) is the same.
//  gemv_q80_rows_ilp2/4  U = 2 / 4 groups in flight per thread: the integer group sums iv[u][b] of U consecutive groups are
//                          computed from loads issued together (U x GS/4 weight words in flight instead of GS/4), and the float
//                          scale steps then run in group order, so the result is still bit-identical to runq.c (int32 sums are
//                          exact and independent of the order; only the float chain is ordered). The tail (G % U groups) takes
//                          the one-group path. This is the "independent chains" of the §14.3 diagnosis applied to Q8_0.
//  q80_touch_rows        diagnostic: thread per row, the SAME 4-byte weight-word loads as gemv_q80_rows but no arithmetic except
//                          one integer add per word; y[row] = bits of the wrapped uint32 sum of the row's words. Measures the
//                          DRAM traffic of the access pattern alone.
//  q80_touch_rows4       as q80_touch_rows with 4 independent partial sums (4 loads issued before the adds): same pattern, more in flight.
//  q80_touch_lanes       warp per row, lane l reads words l, l + NT, ...: a warp's load covers consecutive bytes (coalesced).
//                          The lane partial sums are combined with an integer butterfly. Same y as the others.
#include <math.h>
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#pragma clang fp contract(off)   // as kernel.cpp: runq.c's separate roundings

#ifndef VX_CFG_NUM_THREADS
#define VX_CFG_NUM_THREADS 4
#endif

namespace {

#define PSW_INL inline __attribute__((always_inline))

PSW_INL int32_t byte_s32(uint32_t w, int k) { return (int32_t)(w << (24 - 8 * k)) >> 24; }
PSW_INL int32_t dot4w(uint32_t wv, uint32_t xv) {
  return byte_s32(wv, 0) * byte_s32(xv, 0) + byte_s32(wv, 1) * byte_s32(xv, 1) + byte_s32(wv, 2) * byte_s32(xv, 2) +
         byte_s32(wv, 3) * byte_s32(xv, 3);
}

// U consecutive groups g0 .. g0 + U - 1 of one row against B activation vectors: iv[u][b] (int32, exact)
template <int B, int GSC, int U>
PSW_INL void int_sums(const uint32_t* w, const uint32_t* const* xw, uint32_t g0, uint32_t GW, int32_t (&iv)[U][B]) {
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int b = 0; b < B; ++b) iv[u][b] = 0;
  const uint32_t gw0 = GSC ? (uint32_t)(GSC >> 2) : GW;
  if (GSC) {
#pragma unroll
    for (uint32_t k = 0; k < (uint32_t)(GSC >> 2); ++k) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint32_t wv = w[(g0 + u) * gw0 + k];
#pragma unroll
        for (int b = 0; b < B; ++b) iv[u][b] += dot4w(wv, xw[b][(g0 + u) * gw0 + k]);
      }
    }
  } else {
    for (uint32_t k = 0; k < gw0; ++k) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint32_t wv = w[(g0 + u) * gw0 + k];
#pragma unroll
        for (int b = 0; b < B; ++b) iv[u][b] += dot4w(wv, xw[b][(g0 + u) * gw0 + k]);
      }
    }
  }
}

template <int B, int GSC, int U>
void rows_u_body(const ps_gemv_q80_arg_t* arg) {
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
      int_sums<B, GSC, U>(w, xw, g, GW, iv);
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const float s = ws[g + u];
#pragma unroll
        for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[u][b] * s) * xs[b][g + u];
      }
    }
    for (; g < G; ++g) {   // tail: one group at a time
      int32_t iv[1][B];
      int_sums<B, GSC, 1>(w, xw, g, GW, iv);
      const float s = ws[g];
#pragma unroll
      for (int b = 0; b < B; ++b) acc[b] = acc[b] + ((float)iv[0][b] * s) * xs[b][g];
    }
#pragma unroll
    for (int b = 0; b < B; ++b) y[(size_t)b * M + row] = acc[b];
  }
}

#define PSW_DISPATCH(U, arg)                                                                                          \
  do {                                                                                                                \
    const uint32_t gs_ = (arg)->GS;                                                                                   \
    switch ((arg)->B) {                                                                                               \
      case 1: if (gs_ == 32) rows_u_body<1, 32, U>(arg); else if (gs_ == 64) rows_u_body<1, 64, U>(arg); else rows_u_body<1, 0, U>(arg); break; \
      case 2: if (gs_ == 32) rows_u_body<2, 32, U>(arg); else if (gs_ == 64) rows_u_body<2, 64, U>(arg); else rows_u_body<2, 0, U>(arg); break; \
      case 3: if (gs_ == 32) rows_u_body<3, 32, U>(arg); else if (gs_ == 64) rows_u_body<3, 64, U>(arg); else rows_u_body<3, 0, U>(arg); break; \
      default: if (gs_ == 32) rows_u_body<4, 32, U>(arg); else if (gs_ == 64) rows_u_body<4, 64, U>(arg); else rows_u_body<4, 0, U>(arg); break; \
    }                                                                                                                 \
  } while (0)

}  // namespace

__kernel void gemv_q80_rows_inl(ps_gemv_q80_arg_t* __UNIFORM__ arg) { PSW_DISPATCH(1, arg); }
__kernel void gemv_q80_rows_ilp2(ps_gemv_q80_arg_t* __UNIFORM__ arg) { PSW_DISPATCH(2, arg); }
__kernel void gemv_q80_rows_ilp4(ps_gemv_q80_arg_t* __UNIFORM__ arg) { PSW_DISPATCH(4, arg); }

// ---- diagnostics: weight streaming without arithmetic
__kernel void q80_touch_rows(ps_gemv_q80_arg_t* __UNIFORM__ arg) {
  const uint8_t* Wq = reinterpret_cast<const uint8_t*>(arg->Wq_addr);
  uint32_t* y = reinterpret_cast<uint32_t*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N, nw = N >> 2, stride = gridDim.x * blockDim.x;
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < M; row += stride) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(Wq + (size_t)row * N);
    uint32_t s = 0;
    for (uint32_t k = 0; k < nw; ++k) s += w[k];
    y[row] = s;
  }
}

__kernel void q80_touch_rows4(ps_gemv_q80_arg_t* __UNIFORM__ arg) {
  const uint8_t* Wq = reinterpret_cast<const uint8_t*>(arg->Wq_addr);
  uint32_t* y = reinterpret_cast<uint32_t*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N, nw = N >> 2, stride = gridDim.x * blockDim.x;
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < M; row += stride) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(Wq + (size_t)row * N);
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
    for (; k + 4 <= nw; k += 4) {
      const uint32_t a = w[k], b = w[k + 1], c = w[k + 2], d = w[k + 3];
      s0 += a; s1 += b; s2 += c; s3 += d;
    }
    for (; k < nw; ++k) s0 += w[k];
    y[row] = s0 + s1 + s2 + s3;
  }
}

__kernel void q80_touch_lanes(ps_gemv_q80_arg_t* __UNIFORM__ arg) {
  const uint8_t* Wq = reinterpret_cast<const uint8_t*>(arg->Wq_addr);
  uint32_t* y = reinterpret_cast<uint32_t*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N, nw = N >> 2;
  constexpr uint32_t NT = VX_CFG_NUM_THREADS;
  constexpr int clamp = NT - 1;
  constexpr int segmask = ~clamp & 0x3f;
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = gtid % NT;
  const uint32_t nwarps = (gridDim.x * blockDim.x) / NT;
  for (uint32_t row = gtid / NT; row < M; row += nwarps) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(Wq + (size_t)row * N);
    uint32_t s = 0;
    for (uint32_t k = lane; k < nw; k += NT) s += w[k];
#pragma unroll
    for (uint32_t off = NT >> 1; off > 0; off >>= 1) s += (uint32_t)vx_shfl_bfly(s, (int)off, clamp, segmask);
    if (lane == 0) y[row] = s;
  }
}

// ---- shuffle semantics probe (the segmented butterfly and shfl_idx that kernels/gemv_dxa relies on; host check: main.cpp -v shfl_probe)
// Thread tid of one CTA (all warps) holds v(tid) = tid * 2654435761 ^ 0x9e3779b9 (uint32, distinct per thread) and writes PS_SHFL_SLOTS words
// out[tid * PS_SHFL_SLOTS + k]:
//   k 0..3   segmented butterfly all-reduce (uint32 add) over segments of S = 4, 8, 16, 32 lanes: for off = S/2 .. 1: a += bfly(a, off, S-1, ~(S-1) & 0x3f)
//   k 4      bfly(v, 3, 7, 0x38)        one exchange, segments of 8: lane l reads lane l ^ 3
//   k 5      bfly(v, 16, 31, 0x20)      one exchange, whole warp (the kernel's kSegAll operand): lane l reads lane l ^ 16
//   k 6..9   idx(v, 8 j, 31, 0x20)      j = 0..3, the whole-warp broadcast of lane 8 j (gemv_dxa's group-order accumulation)
//   k 10     idx(v, 3, 7, 0x38)         segments of 8: every lane reads lane (l & ~7) + 3
//   k 11     idx(v, 2, 3, 0x3c)         segments of 4 (VX_QUAD_CVAL / MASK): every lane reads lane (l & ~3) + 2
#define PS_SHFL_SLOTS 12
__kernel void q80_shfl_probe(ps_gemv_q80_arg_t* __UNIFORM__ arg) {
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->y_addr);
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t v = tid * 2654435761u ^ 0x9e3779b9u;
  uint32_t* o = out + (size_t)tid * PS_SHFL_SLOTS;
#define PSW_SEG_SUM(K, S)                                                                                               \
  {                                                                                                                     \
    uint32_t a = v;                                                                                                     \
    for (int off = (S) >> 1; off > 0; off >>= 1) a += (uint32_t)vx_shfl_bfly(a, off, (S) - 1, ~((S) - 1) & 0x3f);        \
    o[K] = a;                                                                                                           \
  }
  PSW_SEG_SUM(0, 4)
  PSW_SEG_SUM(1, 8)
  PSW_SEG_SUM(2, 16)
  PSW_SEG_SUM(3, 32)
#undef PSW_SEG_SUM
  o[4] = (uint32_t)vx_shfl_bfly(v, 3, 7, 0x38);
  o[5] = (uint32_t)vx_shfl_bfly(v, 16, 31, 0x20);
  o[6] = (uint32_t)vx_shfl_idx(v, 0, 31, 0x20);
  o[7] = (uint32_t)vx_shfl_idx(v, 8, 31, 0x20);
  o[8] = (uint32_t)vx_shfl_idx(v, 16, 31, 0x20);
  o[9] = (uint32_t)vx_shfl_idx(v, 24, 31, 0x20);
  o[10] = (uint32_t)vx_shfl_idx(v, 3, 7, 0x38);
  o[11] = (uint32_t)vx_shfl_idx(v, 2, 3, 0x3c);
}
