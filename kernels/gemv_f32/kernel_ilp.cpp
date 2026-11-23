// kernel_ilp.cpp -- ILP variants of the fp32 GEMV (methodology §18.4). A translation unit of its own, linked
// AFTER kernel.cpp (kernels/gemv_f32/app.conf), so the stock entries gemv_rows / gemv_lanes and ps_llama's reused copy of
// kernel.cpp are untouched. Same argument struct and launch geometry as the stock kernels.
//
// Why: the stock `gemv_rows` loop is one serial dependence chain per thread (sum += a.x*b.x + ... over float4 chunks) with its
// loads issued one chunk ahead of use; on Vortex's in-order warps a warp stalls on the first consumer of a missing load, so the
// number of loads in flight per warp is what hides DRAM latency (methodology §14.3). The variants below keep U independent
// accumulator chains per thread and issue the loads of U chunks together before any of them is consumed:
//   gemv_rows_ilp2 / gemv_rows_ilp4   thread per row, U = 2 / 4 float4 chunks (8 / 16 weights) per iteration, U partial sums
//   gemv_lanes_ilp2                   warp per row (lane l: float4 chunks 4l, 4l + 4NT, ...), U = 2 chunks per iteration
// The partial sums are added at the end ((s0 + s1) [+ (s2 + s3)]), so the summation order differs from the stock kernels: the
// result is within the 2 gamma_K S_i rule (kernels/common/ps_common.h) but not bit-identical to them. Tails: N % (4U) != 0 leaves
// whole chunks (handled one at a time on chain 0) and the scalar path (N % 4 != 0 or unaligned buffers) is the stock scalar loop.
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#ifndef VX_CFG_NUM_THREADS
#define VX_CFG_NUM_THREADS 4
#endif

namespace {
struct f4 { float x, y, z, w; };

static inline size_t bits_from_f32(float v) { uint32_t u; __builtin_memcpy(&u, &v, 4); return u; }
static inline float f32_from_bits(size_t u) { uint32_t b = (uint32_t)u; float v; __builtin_memcpy(&v, &b, 4); return v; }

static inline __attribute__((always_inline)) float dot4(const f4& a, const f4& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// one thread-per-row dot product with U chains
template <int U>
static inline __attribute__((always_inline)) float row_dot(const float* a, const float* x, uint32_t N, bool vec) {
  float s[U];
#pragma unroll
  for (int u = 0; u < U; ++u) s[u] = 0.0f;
  if (vec) {
    uint32_t c = 0;
    for (; c + 4 * U <= N; c += 4 * U) {
      f4 av[U], bv[U];
#pragma unroll
      for (int u = 0; u < U; ++u) av[u] = *reinterpret_cast<const f4*>(a + c + 4 * u);   // all weight loads first
#pragma unroll
      for (int u = 0; u < U; ++u) bv[u] = *reinterpret_cast<const f4*>(x + c + 4 * u);
#pragma unroll
      for (int u = 0; u < U; ++u) s[u] += dot4(av[u], bv[u]);
    }
    for (; c < N; c += 4) {
      f4 av = *reinterpret_cast<const f4*>(a + c);
      f4 bv = *reinterpret_cast<const f4*>(x + c);
      s[0] += dot4(av, bv);
    }
  } else {
    for (uint32_t c = 0; c < N; ++c) s[0] += a[c] * x[c];
  }
  if (U == 2) return s[0] + s[1];
  if (U == 4) return (s[0] + s[1]) + (s[2 % U] + s[3 % U]);
  return s[0];
}

template <int U>
static inline __attribute__((always_inline)) void rows_ilp(ps_gemv_arg_t* arg) {
  const float* W = reinterpret_cast<const float*>(arg->W_addr);
  const float* x = reinterpret_cast<const float*>(arg->x_addr);
  float* y = reinterpret_cast<float*>(arg->y_addr);
  const uint32_t M = arg->M, N = arg->N;
  const uint32_t stride = gridDim.x * blockDim.x;
  const bool vec = ((N & 3u) == 0) && ((arg->W_addr & 15u) == 0) && ((arg->x_addr & 15u) == 0);
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < M; row += stride)
    y[row] = row_dot<U>(W + (size_t)row * N, x, N, vec);
}
}  // namespace

__kernel void gemv_rows_ilp2(ps_gemv_arg_t* __UNIFORM__ arg) { rows_ilp<2>(arg); }
__kernel void gemv_rows_ilp4(ps_gemv_arg_t* __UNIFORM__ arg) { rows_ilp<4>(arg); }

__kernel void gemv_lanes_ilp2(ps_gemv_arg_t* __UNIFORM__ arg) {
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
    float s0 = 0.0f, s1 = 0.0f;
    if (vec) {
      uint32_t c = 4 * lane;
      for (; c + 4 * NT < N; c += 8 * NT) {   // two chunks (this lane's chunk c and c + 4 NT) per iteration
        f4 a0 = *reinterpret_cast<const f4*>(a + c), a1 = *reinterpret_cast<const f4*>(a + c + 4 * NT);
        f4 b0 = *reinterpret_cast<const f4*>(x + c), b1 = *reinterpret_cast<const f4*>(x + c + 4 * NT);
        s0 += dot4(a0, b0);
        s1 += dot4(a1, b1);
      }
      for (; c < N; c += 4 * NT) {
        f4 a0 = *reinterpret_cast<const f4*>(a + c), b0 = *reinterpret_cast<const f4*>(x + c);
        s0 += dot4(a0, b0);
      }
    } else {
      for (uint32_t c = lane; c < N; c += NT) s0 += a[c] * x[c];
    }
    float sum = s0 + s1;
#pragma unroll
    for (uint32_t off = NT >> 1; off > 0; off >>= 1)
      sum += f32_from_bits(vx_shfl_bfly(bits_from_f32(sum), (int)off, clamp, segmask));
    if (lane == 0) y[row] = sum;
  }
}
