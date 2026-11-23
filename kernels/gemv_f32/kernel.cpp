// kernel.cpp -- ps_gemv_f32 device kernels. y = W @ x, float32.
//
// Two entry points in one module (the host picks one with -v):
//
//  gemv_rows  ("rows", default): one thread per output row, grid-stride over rows. Each thread streams its own
//             row with 16-byte float4 loads, like the stock tests/regression/sgemv kernel, but generalised:
//             any N (scalar tail when N % 4 != 0 or the buffers are not 16-byte aligned) and any launch geometry
//             (grid-stride loop instead of one-row-per-thread-or-return). Same per-row summation as sgemv
//             (sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w), so its instruction count is directly comparable.
//             Weakness: lanes of a warp read NT different rows (stride N*4 bytes), so one warp load touches NT
//             cache lines.
//
//  gemv_lanes ("lanes"): one warp per output row, grid-stride over rows by warp. Lane l reads float4 columns
//             4l, 4l + 4NT, ... so a warp load touches consecutive bytes (NT x 16 B = one 64-byte line at NT=4,
//             i.e. coalesced); each lane keeps a partial sum and the NT partials are combined with a butterfly
//             shuffle reduction (vx_shfl_bfly, as dotproduct2's warp_reduce_sum uses vx_shfl_down). Row index
//             is uniform across the warp, so every lane of a warp takes the same branches (the shuffles need all
//             NT lanes active; the host launches a multiple of NT threads per block).
//
// Both are exact float32 dot products in a different summation order than NumPy; the host checks against the
// fixture with the S_i-scaled tolerance explained in kernels/common/ps_common.h.
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#ifndef VX_CFG_NUM_THREADS
#define VX_CFG_NUM_THREADS 4
#endif

struct f4 { float x, y, z, w; };

static inline size_t bits_from_f32(float v) { uint32_t u; __builtin_memcpy(&u, &v, 4); return u; }
static inline float f32_from_bits(size_t u) { uint32_t b = (uint32_t)u; float v; __builtin_memcpy(&v, &b, 4); return v; }

__kernel void gemv_rows(ps_gemv_arg_t* __UNIFORM__ arg) {
  const float* W = reinterpret_cast<const float*>(arg->W_addr);
  const float* x = reinterpret_cast<const float*>(arg->x_addr);
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N;
  const uint32_t stride = gridDim.x * blockDim.x;
  // float4 path only if every row start and x are 16-byte aligned
  const bool vec = ((N & 3u) == 0) && ((arg->W_addr & 15u) == 0) && ((arg->x_addr & 15u) == 0);
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < M; row += stride) {
    const float* a = W + (size_t)row * N;
    float sum = 0.0f;
    if (vec) {
      for (uint32_t c = 0; c < N; c += 4) {
        f4 av = *reinterpret_cast<const f4*>(a + c);
        f4 bv = *reinterpret_cast<const f4*>(x + c);
        sum += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
      }
    } else {
      for (uint32_t c = 0; c < N; ++c)
        sum += a[c] * x[c];
    }
    y[row] = sum;
  }
}

__kernel void gemv_lanes(ps_gemv_arg_t* __UNIFORM__ arg) {
  const float* W = reinterpret_cast<const float*>(arg->W_addr);
  const float* x = reinterpret_cast<const float*>(arg->x_addr);
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N;
  constexpr uint32_t NT = VX_CFG_NUM_THREADS;
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = gtid % NT;
  const uint32_t nwarps = (gridDim.x * blockDim.x) / NT;
  const bool vec = ((N & 3u) == 0) && ((arg->W_addr & 15u) == 0) && ((arg->x_addr & 15u) == 0);
  constexpr int clamp = NT - 1;
  constexpr int segmask = ~clamp & 0x3f;
  for (uint32_t row = gtid / NT; row < M; row += nwarps) {
    const float* a = W + (size_t)row * N;
    float sum = 0.0f;
    if (vec) {
      for (uint32_t c = 4 * lane; c < N; c += 4 * NT) {
        f4 av = *reinterpret_cast<const f4*>(a + c);
        f4 bv = *reinterpret_cast<const f4*>(x + c);
        sum += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
      }
    } else {
      for (uint32_t c = lane; c < N; c += NT)
        sum += a[c] * x[c];
    }
#pragma unroll
    for (uint32_t off = NT >> 1; off > 0; off >>= 1)
      sum += f32_from_bits(vx_shfl_bfly(bits_from_f32(sum), (int)off, clamp, segmask));
    if (lane == 0)
      y[row] = sum;
  }
}
