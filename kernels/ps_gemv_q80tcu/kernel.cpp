// kernel.cpp -- Q8_0 GEMV on the stock Vortex int8 TCU (wmma_context<NT, int8, int32>, vx_tensor.h; non-WGMMA path,
// the device idiom of tests/regression/sgemm_tcu). One warp (block of NT threads) computes a 16-weight-row tile of y for all
// B <= 16 batch vectors:   C[16 (batch, padded)][16 (weight rows)] = A[16][64] . B[64][16]   per mma_sync (int8 x int8 -> int32).
//   B operand = the 16 weight rows' int8 bytes (col_major load = W row-major, ldm = N), A operand = activation bytes.
// Per-block scales AFTER the MMA: the int32 accumulator holds sum over exactly ONE Q8_0 group, so fragC is zeroed per group and
// acc += ((float)ival * ws) * xs as in runq.c (fp-contract off), group order sequential: bit-exact to the host reference
// (the int32 sum is exact, the TCU adds int32 exactly: |ival| <= GS*127^2 < 2^24).
// K tile = 64 int8. GS = 64: one MMA per group. GS = 32: a 64-tile covers TWO groups, so two MMAs per tile with the same B
// fragment; the A operand of the first/second MMA is the activation with the other group's 32 bytes zeroed (host-prepared
// xp layout, step g = group g). Half of the MACs of a GS = 32 tile are therefore multiplications by zero (counted, README).
// Variant tcu_pk (B <= 8 and GS = 32 only; GS = 64 and B > 8 run the plain path): the 16 A rows of a 64-tile hold the batch rows TWICE, the upper 8
// rows (padding for B <= 8) carry the SECOND group of the tile: row b = x_b with only group 2t kept (bytes 0..31), row 8 + b = x_b with only
// group 2t + 1 kept (bytes 32..63). One MMA then yields both groups' int32 sums (rows b and 8 + b), so the GS = 32 MMA count is halved, with
// the same B fragment; the epilogue adds group 2t then group 2t + 1 (sequential group order kept, bit-exact).
// Lane geometry of the accumulator (NT = 32: tcM = 8, tcN = 4; store_matrix_sync of vx_tensor.h): register r = mb * 4 + j holds
// batch row mb * 8 + (lane >> 2), weight column j * 4 + (lane & 3). For B <= 8 only mb = 0 is read (the rest is padding rows).
#include <math.h>
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include "common.h"

#pragma clang fp contract(off)

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::int8, vt::int32>;
static_assert(VX_CFG_NUM_THREADS == 32, "kernel written for NT = 32 (E1)");
static_assert(ctx::tileM == 16 && ctx::tileN == 16 && ctx::tileK == 64, "unexpected WMMA tile");

#define PSD_INL inline __attribute__((always_inline))

PSD_INL int32_t bits_i32(float v) { int32_t u; __builtin_memcpy(&u, &v, 4); return u; }

template <int MB, int GSC, int PB, int PK = 0>
PSD_INL void tcu_body(const ps_gemv_q80tcu_arg_t* arg) {
  const int8_t* Wq = reinterpret_cast<const int8_t*>(arg->Wq_addr);
  const float* Ws = reinterpret_cast<const float*>(arg->Ws_addr);
  const int8_t* xp = reinterpret_cast<const int8_t*>(arg->xp_addr);
  const float* xs = reinterpret_cast<const float*>(arg->xs_addr);
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N, G = arg->G, B = arg->B;
  const uint32_t lane = vx_thread_id();
  const uint32_t lc = lane & 3u, lr = lane >> 2;
  const uint32_t n0 = blockIdx.x * 16u;
  const int8_t* pW = Wq + (size_t)n0 * N;
  // per-lane scale pointers: weight columns n0 + 4 j + lc, batch rows mb * 8 + lr
  const float* wsp = Ws + (size_t)(n0 + lc) * G;
  const float* xsp = xs + (size_t)lr * G;
  float acc[MB][4];
#pragma unroll
  for (int mb = 0; mb < MB; ++mb)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[mb][j] = 0.0f;

  auto epilogue = [&](const ctx::fragment_acc& c, uint32_t g) {
    float w[4], x[MB];
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j] = wsp[(size_t)j * 4 * G + g];
#pragma unroll
    for (int mb = 0; mb < MB; ++mb) x[mb] = xsp[(size_t)mb * 8 * G + g];
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float f1 = (float)bits_i32(c.data[mb * 4 + j]) * w[j];
        const float f2 = f1 * x[mb];
        acc[mb][j] = acc[mb][j] + f2;
      }
  };

  if (PK) {   // packed GS = 32 tiles (MB == 1, B <= 8)
    ctx::fragment_a fragA;
    ctx::fragment_b fragB;
    ctx::fragment_acc fragC;
    for (uint32_t g = 0; g < G; g += 2) {
      ctx::load_matrix_sync<vt::col_major>(fragB, pW + (size_t)g * 32, N);
      ctx::load_matrix_sync(fragA, xp + (size_t)(g / 2) * (16 * 64), 64);
      ctx::fill_fragment(fragC, 0);
      ctx::mma_sync(fragC, fragA, fragB, fragC);
      float w0[4], w1[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) { w0[j] = wsp[(size_t)j * 4 * G + g]; w1[j] = wsp[(size_t)j * 4 * G + g + 1]; }
      const float x0 = xsp[g], x1 = xsp[g + 1];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float f1 = (float)bits_i32(fragC.data[j]) * w0[j];
        acc[0][j] = acc[0][j] + f1 * x0;
      }
      if (g + 1 < G) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float f1 = (float)bits_i32(fragC.data[4 + j]) * w1[j];
          acc[0][j] = acc[0][j] + f1 * x1;
        }
      }
    }
  } else {
  ctx::fragment_a fragA;
  ctx::fragment_b fragB, fragBn;
  ctx::fragment_acc fragC;
  // GS = 64: tile t = group t. GS = 32: tile t = groups 2t, 2t+1. PB = 1: the B fragment (weights) of tile t + 1 is loaded before the
  // MMA / epilogue of tile t (one tile of weight prefetch; the last tile re-reads itself, inside the padded buffer).
  constexpr uint32_t GPT = (GSC == 64) ? 1 : 2;          // groups per tile
  const uint32_t T = (G + GPT - 1) / GPT;
  if (PB) ctx::load_matrix_sync<vt::col_major>(fragBn, pW, N);
  for (uint32_t t = 0; t < T; ++t) {
    if (PB) {
      fragB = fragBn;
      const uint32_t tn = (t + 1 < T) ? t + 1 : t;
      ctx::load_matrix_sync<vt::col_major>(fragBn, pW + (size_t)tn * 64, N);
    } else {
      ctx::load_matrix_sync<vt::col_major>(fragB, pW + (size_t)t * 64, N);
    }
    const uint32_t g = t * GPT;
    ctx::load_matrix_sync(fragA, xp + (size_t)g * (16 * 64), 64);
    ctx::fill_fragment(fragC, 0);
    ctx::mma_sync(fragC, fragA, fragB, fragC);
    epilogue(fragC, g);
    if (GPT == 2 && g + 1 < G) {
      ctx::load_matrix_sync(fragA, xp + (size_t)(g + 1) * (16 * 64), 64);
      ctx::fill_fragment(fragC, 0);
      ctx::mma_sync(fragC, fragA, fragB, fragC);
      epilogue(fragC, g + 1);
    }
  }
  }
#pragma unroll
  for (int mb = 0; mb < MB; ++mb)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t m = mb * 8 + lr, n = n0 + j * 4 + lc;
      if (m < B && n < M) y[(size_t)m * M + n] = acc[mb][j];
    }
}

template <int PB>
PSD_INL void dispatch(ps_gemv_q80tcu_arg_t* arg) {
  if (arg->B <= 8) {
    if (arg->GS == 64) tcu_body<1, 64, PB>(arg); else tcu_body<1, 32, PB>(arg);
  } else {
    if (arg->GS == 64) tcu_body<2, 64, PB>(arg); else tcu_body<2, 32, PB>(arg);
  }
}
template <>
PSD_INL void dispatch<2>(ps_gemv_q80tcu_arg_t* arg) {   // tcu_pk
  if (arg->GS == 32 && arg->B <= 8) tcu_body<1, 32, 0, 1>(arg); else dispatch<0>(arg);
}
__kernel void gemv_q80_tcu(ps_gemv_q80tcu_arg_t* __UNIFORM__ arg) { dispatch<0>(arg); }
__kernel void gemv_q80_tcu_pb(ps_gemv_q80tcu_arg_t* __UNIFORM__ arg) { dispatch<1>(arg); }
__kernel void gemv_q80_tcu_pk(ps_gemv_q80tcu_arg_t* __UNIFORM__ arg) { dispatch<2>(arg); }
