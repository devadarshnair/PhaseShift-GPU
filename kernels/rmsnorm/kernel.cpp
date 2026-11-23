// kernel.cpp -- ps_rmsnorm device kernel: batched RMSNorm, float32, run.c's formula.
//
// Design: one warp per row, grid-stride over rows by warp (as gemv_lanes, kernels/gemv_f32/kernel.cpp). Lane l sums
// the squares of x[l], x[l+NT], ... sequentially (strided, so a warp load touches consecutive words), the NT partials
// are combined with a butterfly shuffle (vx_shfl_bfly), which leaves the bit-identical total in every lane
// (floating-point addition is commutative, and lane l and lane l^off add the same two values), so every lane applies
// the same scale. Then  ss = 1/sqrtf(ss/n + eps)  in run.c's order (rmsnorm, run.c lines 182-195: `ss /= size;
// ss += 1e-5f; ss = 1.0f / sqrtf(ss);`) and  out[i] = w[i] * (ss * x[i])  (run.c line 193). The row index is uniform
// across a warp, so the shuffles always see all NT lanes (the host launches whole warps).
// Summation depth (rounding steps a square passes through) = ceil(n/NT) + log2(NT); the tolerance derivation in
// main.cpp uses it.
#include <math.h>
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#ifndef VX_CFG_NUM_THREADS
#define VX_CFG_NUM_THREADS 4
#endif

static inline size_t bits_from_f32(float v) { uint32_t u; __builtin_memcpy(&u, &v, 4); return u; }
static inline float f32_from_bits(size_t u) { uint32_t b = (uint32_t)u; float v; __builtin_memcpy(&v, &b, 4); return v; }

__kernel void rmsnorm(ps_rmsnorm_arg_t* __UNIFORM__ arg) {
  const float* x = reinterpret_cast<const float*>(arg->x_addr);
  const float* w = reinterpret_cast<const float*>(arg->w_addr);
  float* out = reinterpret_cast<float*>(arg->out_addr);
  const uint32_t B = arg->B, n = arg->n;
  const float eps = arg->eps;
  constexpr uint32_t NT = VX_CFG_NUM_THREADS;
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = gtid % NT;
  const uint32_t nwarps = (gridDim.x * blockDim.x) / NT;
  constexpr int clamp = NT - 1;
  constexpr int segmask = ~clamp & 0x3f;
  for (uint32_t row = gtid / NT; row < B; row += nwarps) {
    const float* xr = x + (size_t)row * n;
    float ss = 0.0f;
    for (uint32_t i = lane; i < n; i += NT)
      ss += xr[i] * xr[i];
#pragma unroll
    for (uint32_t off = NT >> 1; off > 0; off >>= 1)
      ss += f32_from_bits(vx_shfl_bfly(bits_from_f32(ss), (int)off, clamp, segmask));
    ss /= (float)n;
    ss += eps;
    ss = 1.0f / sqrtf(ss);
    float* o = out + (size_t)row * n;
    for (uint32_t i = lane; i < n; i += NT)
      o[i] = w[i] * (ss * xr[i]);
  }
}
