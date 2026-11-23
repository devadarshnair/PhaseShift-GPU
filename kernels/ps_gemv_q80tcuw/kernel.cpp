// kernel.cpp -- the stock int8 TCU Q8_0 GEMV of kernels/ps_gemv_q80tcu (same WMMA
// tile, same MMA/epilogue math and order: bit-exact) given dot4w's tuning liberties:
//   * pre-laid-out weight fragments (common.h): each of the 8 fragment_b registers is ONE coalesced 128 B warp load (stock: col_major
//     load_matrix_sync from row-major W, 16 rows x 4 B per register = 16 distinct lines), and pre-laid scales (stock: 4 strided rows);
//   * deep weight load pipelining: U K-tiles' B fragments (8 U words per lane) are loaded and register-fenced before the first MMA
//     (the analogue of dot4w's U groups of fenced weight loads); variant pb = software-pipelined next-tile prefetch instead.
// Variants (7, the dot4w tuning budget):  lay_u1, lay_u2, lay_u4, lay_pb  (plain path: one MMA per group)
//                                        pk_u1, pk_u2, pk_u4              (packed GS = 32 path of tcu_pk: one MMA per 2 groups;
//                                                                          GS = 64 or B > 8 run the lay_u<U> plain path)
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
#define PSD_FENCEF(v) asm volatile("" : "+f"(v))

PSD_INL int32_t bits_i32(float v) { int32_t u; __builtin_memcpy(&u, &v, 4); return u; }

// fragment_b of tile t from the pre-laid weight stream (lane-offset pointer wf = Wf + blk*T*256 + lane)
PSD_INL void load_b(ctx::fragment_b& f, const float* wf, uint32_t t) {
#pragma unroll
  for (int r = 0; r < 8; ++r) f.data[r] = wf[((size_t)t * 8 + r) * 32];
}
PSD_INL void fence_b(ctx::fragment_b& f) {
#pragma unroll
  for (int r = 0; r < 8; ++r) PSD_FENCEF(f.data[r]);
}

template <int MB, int GSC, int U, int PK, int PB>
PSD_INL void body(const ps_gemv_q80tcuw_arg_t* arg) {
  const float* Wsl = reinterpret_cast<const float*>(arg->Wsl_addr);
  const int8_t* xp = reinterpret_cast<const int8_t*>(arg->xp_addr);
  const float* xs = reinterpret_cast<const float*>(arg->xs_addr);
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const uint32_t M = arg->M, G = arg->G, B = arg->B, T = arg->T;
  const uint32_t lane = vx_thread_id();
  const uint32_t lc = lane & 3u, lr = lane >> 2;
  const uint32_t blk = blockIdx.x, n0 = blk * 16u;
  const float* wf = reinterpret_cast<const float*>(arg->Wf_addr) + (size_t)blk * T * 256 + lane;
  const float* wsp = Wsl + (size_t)blk * G * 16 + lc;     // + g * 16 + 4 j
  const float* xsp = xs + (size_t)lr * G;
  float acc[MB][4];
#pragma unroll
  for (int mb = 0; mb < MB; ++mb)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[mb][j] = 0.0f;

  auto epilogue = [&](const ctx::fragment_acc& c, uint32_t g, int half) __attribute__((always_inline)) {   // half: accumulator rows 0..3 (+4*half) for pk
    float w[4], x[MB];
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j] = wsp[(size_t)g * 16 + 4 * j];
#pragma unroll
    for (int mb = 0; mb < MB; ++mb) x[mb] = xsp[(size_t)mb * 8 * G + g];
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float f1 = (float)bits_i32(c.data[(PK ? half * 4 : mb * 4) + j]) * w[j];
        const float f2 = f1 * x[mb];
        acc[mb][j] = acc[mb][j] + f2;
      }
  };
  // one K tile (fragB loaded): plain = 1 or 2 MMAs (GS 64 / GS 32), packed = 1 MMA for both groups
  auto tile = [&](const ctx::fragment_b& fb, uint32_t t) __attribute__((always_inline)) {
    ctx::fragment_a fa;
    ctx::fragment_acc fc;
    if (PK) {
      const uint32_t g = 2 * t;
      ctx::load_matrix_sync(fa, xp + (size_t)t * (16 * 64), 64);
      ctx::fill_fragment(fc, 0);
      ctx::mma_sync(fc, fa, fb, fc);
      epilogue(fc, g, 0);
      if (g + 1 < G) epilogue(fc, g + 1, 1);
    } else {
      constexpr uint32_t GPT = (GSC == 64) ? 1 : 2;
      const uint32_t g = t * GPT;
      ctx::load_matrix_sync(fa, xp + (size_t)g * (16 * 64), 64);
      ctx::fill_fragment(fc, 0);
      ctx::mma_sync(fc, fa, fb, fc);
      epilogue(fc, g, 0);
      if (GPT == 2 && g + 1 < G) {
        ctx::load_matrix_sync(fa, xp + (size_t)(g + 1) * (16 * 64), 64);
        ctx::fill_fragment(fc, 0);
        ctx::mma_sync(fc, fa, fb, fc);
        epilogue(fc, g + 1, 0);
      }
    }
  };

  uint32_t t = 0;
  if (PB) {
    ctx::fragment_b fb, fbn;
    load_b(fbn, wf, 0);
    for (; t < T; ++t) {
      fb = fbn;
      load_b(fbn, wf, (t + 1 < T) ? t + 1 : t);
      tile(fb, t);
    }
  } else {
    for (; t + U <= T; t += U) {
      ctx::fragment_b fb[U];
#pragma unroll
      for (int u = 0; u < U; ++u) load_b(fb[u], wf, t + u);
#pragma unroll
      for (int u = 0; u < U; ++u) fence_b(fb[u]);   // all U*8 weight loads in flight before the first MMA
#pragma unroll
      for (int u = 0; u < U; ++u) tile(fb[u], t + u);
    }
    for (; t < T; ++t) {
      ctx::fragment_b fb;
      load_b(fb, wf, t);
      tile(fb, t);
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

template <int U, int PK, int PB>
PSD_INL void dispatch(ps_gemv_q80tcuw_arg_t* arg) {
  if (PK && arg->GS == 32 && arg->B <= 8) { body<1, 32, U, 1, 0>(arg); return; }
  if (arg->B <= 8) {
    if (arg->GS == 64) body<1, 64, U, 0, PB>(arg); else body<1, 32, U, 0, PB>(arg);
  } else {
    if (arg->GS == 64) body<2, 64, U, 0, PB>(arg); else body<2, 32, U, 0, PB>(arg);
  }
}
__kernel void gemv_q80_lay_u1(ps_gemv_q80tcuw_arg_t* __UNIFORM__ arg) { dispatch<1, 0, 0>(arg); }
__kernel void gemv_q80_lay_u2(ps_gemv_q80tcuw_arg_t* __UNIFORM__ arg) { dispatch<2, 0, 0>(arg); }
__kernel void gemv_q80_lay_u4(ps_gemv_q80tcuw_arg_t* __UNIFORM__ arg) { dispatch<4, 0, 0>(arg); }
__kernel void gemv_q80_lay_pb(ps_gemv_q80tcuw_arg_t* __UNIFORM__ arg) { dispatch<1, 0, 1>(arg); }
__kernel void gemv_q80_pk_u1(ps_gemv_q80tcuw_arg_t* __UNIFORM__ arg) { dispatch<1, 1, 0>(arg); }
__kernel void gemv_q80_pk_u2(ps_gemv_q80tcuw_arg_t* __UNIFORM__ arg) { dispatch<2, 1, 0>(arg); }
__kernel void gemv_q80_pk_u4(ps_gemv_q80tcuw_arg_t* __UNIFORM__ arg) { dispatch<4, 1, 0>(arg); }
