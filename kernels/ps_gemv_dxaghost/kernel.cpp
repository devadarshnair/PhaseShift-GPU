// kernel.cpp -- ps_gemv_dxa device kernels (methodology §18.4): weight-streaming GEMV with DXA -> LMEM staging.
//
//  gemv_dxa   one entry for both flavours (arg->q80): f32 weights or Q8_0 weights, B = 1..4 vectors.
//
// Structure (the earlier `gqa_dxa_dbuf` pattern): one CTA = one core, all warps. The CTA loops over its row groups (NW rows, one per
// warp) and, inside a row group, over K-tiles of KT columns; the flattened tile sequence t = 0, 1, ... is streamed through a ring of
// `depth` LMEM stages:
//   prologue   warp 0 issues tiles 0 .. depth-2 (expect_tx + one DXA transfer for W, one more for the Q8_0 scales);
//   iteration  warp 0 issues tile t + depth - 1 into the stage that held tile t - 1 (every warp finished it at the end of the previous
//              iteration: the CTA barrier below), all warps wait on the barrier of tile t's stage (tx events + arrivals), compute,
//              then meet at the phase barrier.
// depth = 1 therefore has no overlap (issue, wait, compute, barrier), depth 2 and 4 overlap the next tiles' transfers with compute.
// x (and the Q8_0 activation scales) are copied once into LMEM by all threads before the first tile.
//
// Compute (warp w = row w of the row group; all lanes of a warp execute the same control flow, as the shuffles need all lanes):
//  f32  the gemv_lanes pattern on LMEM: lane l accumulates float4 chunks l, l + NT, ... of the tile row (same per-lane order and the
//       same butterfly as `gemv_lanes`, so when KT >= N (one tile per row) the result is BIT-identical to gemv_lanes; with several
//       K-tiles a lane's chunk set changes, the 2 gamma_N rule applies). B accumulators per lane, one weight chunk shared by all B.
//  q80  lane l takes word l of every 32-word step; GS/4 consecutive lanes hold one group. Each lane forms the int32 dot of its
//       weight word with the B activation words, the lanes of a group are reduced with an all-reduce butterfly restricted to the
//       GS/4-lane segment (exact: int32), and then the groups of the step are added to the fp32 accumulator IN GROUP ORDER (a
//       shfl_idx broadcast of each group's scaled sum, acc += ((float)ival * ws) * xs as runq.c): the result is BIT-identical to
//       runq.c's matmul and to gemv_q80_rows, for every B, KT and depth.
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#include "common.h"

#pragma clang fp contract(off)   // no fused multiply-add: runq.c's separate roundings (the f32 loop re-enables contraction locally)

#ifndef VX_CFG_NUM_THREADS
#define VX_CFG_NUM_THREADS 4
#endif

namespace {
constexpr uint32_t NT = VX_CFG_NUM_THREADS;
constexpr int kClamp = NT - 1;
constexpr int kSegAll = ~kClamp & 0x3f;

struct f4 { float x, y, z, w; };

static inline size_t bits_from_f32(float v) { uint32_t u; __builtin_memcpy(&u, &v, 4); return u; }
static inline float f32_from_bits(size_t u) { uint32_t b = (uint32_t)u; float v; __builtin_memcpy(&v, &b, 4); return v; }
static inline int32_t byte_s32(uint32_t w, int k) { return (int32_t)(w << (24 - 8 * k)) >> 24; }
static inline int32_t dot4w(uint32_t wv, uint32_t xv) {
  return byte_s32(wv, 0) * byte_s32(xv, 0) + byte_s32(wv, 1) * byte_s32(xv, 1) + byte_s32(wv, 2) * byte_s32(xv, 2) +
         byte_s32(wv, 3) * byte_s32(xv, 3);
}

template <int B>
static void body(const ps_gemv_dxa_arg_t* arg) {
  const uint32_t M = arg->M, N = arg->N, GS = arg->GS, KT = arg->KT, depth = arg->depth, nw = arg->nw;
  const bool q80 = arg->q80 != 0;
  const uint32_t G = q80 ? N / GS : 0;
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const psgd_lmem_t L = psgd_lmem(N, GS, B, KT, depth, q80, nw);
  uint8_t* lm = reinterpret_cast<uint8_t*>(__local_mem());
  const uint32_t tid = threadIdx.x, nth = blockDim.x;
  const uint32_t warp = get_sub_group_id();
  const uint32_t lane = tid % NT;

  vortex::barrier sync(5);   // phase barrier (ids 1..4 are the stage barriers, id 0 is __syncthreads' slot)
  vortex::barrier stage_bar[4] = {vortex::barrier(1), vortex::barrier(2), vortex::barrier(3), vortex::barrier(4)};

  // ---- x into LMEM (cooperative, once)
  if (q80) {
    const uint32_t* xg = reinterpret_cast<const uint32_t*>(arg->xq_addr);
    uint32_t* xl = reinterpret_cast<uint32_t*>(lm + L.x_off);
    for (uint32_t i = tid; i < B * N / 4; i += nth) xl[i] = xg[i];
    const uint32_t* sg = reinterpret_cast<const uint32_t*>(arg->xs_addr);
    uint32_t* sl = reinterpret_cast<uint32_t*>(lm + L.xs_off);
    for (uint32_t i = tid; i < B * G; i += nth) sl[i] = sg[i];
  } else {
    const uint32_t* xg = reinterpret_cast<const uint32_t*>(arg->xq_addr);
    uint32_t* xl = reinterpret_cast<uint32_t*>(lm + L.x_off);
    for (uint32_t i = tid; i < B * N; i += nth) xl[i] = xg[i];
  }
  sync.arrive_and_wait();

  const uint32_t ngroups = (M + nw - 1) / nw;                  // row groups in the whole GEMV
  const uint32_t nkt = (N + KT - 1) / KT;                      // K-tiles per row group
  const uint32_t my_groups = (ngroups > blockIdx.x) ? (ngroups - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
  const uint32_t total = my_groups * nkt;                      // tiles of this CTA
  const bool issuer = (warp == 0);
  const uint32_t KTg = q80 ? KT / GS : 0;

  auto issue = [&](uint32_t t) {   // tile t -> stage t % depth (the whole issuer warp executes it)
    const uint32_t s = t % depth, kt = t % nkt, rg = blockIdx.x + (t / nkt) * gridDim.x;
    uint8_t* st = lm + L.stage0 + s * L.stage_bytes;
    stage_bar[s].expect_tx(q80 ? 2 : 1);
    vx_dxa_issue_2d_wg(PS_GEMVDXA_DESC_W, stage_bar[s].id(), st, kt * KT, rg * nw);
    if (q80) vx_dxa_issue_2d_wg(PS_GEMVDXA_DESC_S, stage_bar[s].id(), st + L.ws_off, kt * KTg, rg * nw);
  };
  if (issuer)
    for (uint32_t t = 0; t + 1 < depth && t < total; ++t) issue(t);

  float acc[B];
#pragma unroll
  for (int b = 0; b < B; ++b) acc[b] = 0.0f;

  for (uint32_t t = 0; t < total; ++t) {
    const uint32_t s = t % depth, kt = t % nkt, rg = blockIdx.x + (t / nkt) * gridDim.x;
    const uint32_t row = rg * nw + warp;
    if (issuer && t + depth - 1 < total) issue(t + depth - 1);
    stage_bar[s].arrive_and_wait();                            // tile t landed in stage s and every warp arrived
    const uint8_t* st = lm + L.stage0 + s * L.stage_bytes;
    const uint32_t col0 = kt * KT;
    const uint32_t kcols = (N - col0 < KT) ? N - col0 : KT;    // columns of this tile inside the matrix
    if (kt == 0) {
#pragma unroll
      for (int b = 0; b < B; ++b) acc[b] = 0.0f;
    }
    if (!q80) {
      const float* wt = reinterpret_cast<const float*>(st) + (size_t)warp * KT;
      const float* xl = reinterpret_cast<const float*>(lm + L.x_off) + col0;
      {
#pragma clang fp contract(on)   // the stock gemv_lanes expression compiled with the default contraction, so the two agree bit for bit
        for (uint32_t c = 4 * lane; c < kcols; c += 4 * NT) {
          const f4 a = *reinterpret_cast<const f4*>(wt + c);
#pragma unroll
          for (int b = 0; b < B; ++b) {
            const f4 xv = *reinterpret_cast<const f4*>(xl + (size_t)b * N + c);
            acc[b] += a.x * xv.x + a.y * xv.y + a.z * xv.z + a.w * xv.w;
          }
        }
      }
      if (kt == nkt - 1) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
          float sum = acc[b];
#pragma unroll
          for (uint32_t off = NT >> 1; off > 0; off >>= 1)
            sum += f32_from_bits(vx_shfl_bfly(bits_from_f32(sum), (int)off, kClamp, kSegAll));
          if (lane == 0 && row < M) y[(size_t)b * M + row] = sum;
        }
      }
    } else {
      const uint32_t* wt = reinterpret_cast<const uint32_t*>(st) + (size_t)warp * (KT / 4);
      const float* wst = reinterpret_cast<const float*>(st + L.ws_off) + (size_t)warp * KTg;
      const uint32_t* xl = reinterpret_cast<const uint32_t*>(lm + L.x_off) + col0 / 4;
      const float* xsl = reinterpret_cast<const float*>(lm + L.xs_off) + col0 / GS;
      const uint32_t GW = GS / 4;                              // words (= lanes) per group, a power of two <= NT
      const int segclamp = (int)GW - 1, segmask = ~segclamp & 0x3f;
      const uint32_t KTw = kcols / 4;
      for (uint32_t base = 0; base < KTw; base += NT) {
        const uint32_t wi = base + lane;
        const bool valid = wi < KTw;
        int32_t iv[B];
        const uint32_t wv = valid ? wt[wi] : 0u;
#pragma unroll
        for (int b = 0; b < B; ++b) iv[b] = valid ? dot4w(wv, xl[(size_t)b * (N / 4) + wi]) : 0;
#pragma unroll
        for (int b = 0; b < B; ++b)
          for (uint32_t off = GW >> 1; off > 0; off >>= 1)
            iv[b] += (int32_t)vx_shfl_bfly((size_t)iv[b], (int)off, segclamp, segmask);
        // every lane of a group now holds the group's exact int32 sum; form the fp32 scaled term (runq.c: (ival * ws) * xs)
        float f[B];
        const uint32_t gl = wi / GW;                           // group inside the tile
        const float ws = valid ? wst[gl] : 0.0f;
#pragma unroll
        for (int b = 0; b < B; ++b) f[b] = valid ? ((float)iv[b] * ws) * xsl[(size_t)b * G + gl] : 0.0f;
        // add the groups of this step in group order: lane j * GW holds group j of the step
        for (uint32_t j = 0; j * GW < NT && base + j * GW < KTw; ++j) {
#pragma unroll
          for (int b = 0; b < B; ++b) acc[b] = acc[b] + f32_from_bits(vx_shfl_idx(bits_from_f32(f[b]), (int)(j * GW), kClamp, kSegAll));
        }
      }
      if (kt == nkt - 1 && lane == 0 && row < M) {
#pragma unroll
        for (int b = 0; b < B; ++b) y[(size_t)b * M + row] = acc[b];
      }
    }
    sync.arrive_and_wait();                                    // stage s may be restaged
  }
}
}  // namespace

__kernel void gemv_dxa(ps_gemv_dxa_arg_t* __UNIFORM__ arg) {
  switch (arg->B) {
    case 1: body<1>(arg); break;
    case 2: body<2>(arg); break;
    case 3: body<3>(arg); break;
    default: body<4>(arg); break;
  }
}
