// kernel.cpp -- ps_softmax_fixed device kernels. Row-wise softmax, float32. Design and results:
// docs/methodology.md §14.1.
//
// Semantics (= run.c `softmax` and model/ref_llama2c.py `Llama2cRef.softmax`): for each row x of length n,
//     m = max_j x_j  (exact maximum of the row; the running maximum starts at -inf),
//     e_j = exp(x_j - m),  S = sum_j e_j,  p_j = e_j / S.
// Subtracting the row maximum is the standard overflow guard: every exponent is <= 0, so e_j <= 1 and S >= 1 (the
// maximum contributes exp(0) = 1); the result is mathematically unchanged because softmax is shift-invariant. The
// stock tests/regression/softmax kernel starts its running maximum at 0.0 (kernel.cpp:16; errata 7), so for an
// all-negative row it subtracts 0 instead of the maximum; `stock` / `stock_expf` below reproduce it
// comparison. p_j is computed as e_j * (1/S) (one division per row instead of one per element; one extra rounding,
// accounted for in the host's error bound).
//
// Entry points (the host picks one with -v):
//   softmax_warp       (`warp`, default)  one warp per row, 3 passes (max, exp+sum, scale), exp = ps_expf (float-only
//                                         software exp, warp_ops.h)
//   softmax_warp_mufu  (`mufu`)           the same with exp = MUFU EX2 (vx_expf_fast); needs VX_CFG_EXT_MUFU_ENABLE
//   softmax_warp_libm  (`libm`)           the same with musl expf (soft-float double on rv32imaf; cost reference)
//   softmax_stock      (`stock`)          verbatim algorithm of tests/regression/softmax/kernel.cpp (one thread per
//                                         row, running max initialised to 0.0, the stock 4-term polynomial "exp" of
//                                         tests/regression/softmax/common.h, exp values written back into src)
//   softmax_stock_expf (`stock_expf`)     the stock kernel with an accurate exp (ps_expf) instead of the polynomial:
//                                         isolates the max = 0.0 initialisation from the polynomial's own error
//   exp_probe_{sw,mufu,libm} (`-p`)       dst[i] = exp(src[i]) element-wise (accuracy and cost of each exp)
//
// Warp-per-row layout: lane l of the warp handles columns l, l+NT, l+2NT, ... (consecutive lanes read consecutive
// floats, so a warp load touches one or two cache lines); row index = global warp id, grid-stride over rows, so the
// row is uniform across the warp and the shuffle reductions (warp_ops.h) see all NT lanes. Lanes whose strided
// range is empty (n < NT) keep the identities -inf / 0.
//
// Summation (accuracy, see main.cpp "Error bound"): each lane sums its strided exp values in blocks of
// PS_SOFTMAX_SUM_BLOCK (32) into a block partial, and adds each block partial to its running total; then the NT
// lane totals are combined by the butterfly. Every term therefore passes through at most
//     h = (B - 1) + ceil(ceil(n/NT)/B) + log2(NT)       (B = 32)
// additions instead of ceil(n/NT) - 1 + log2(NT) for a plain running sum (h <= 65 vs 1025 at n = 4096, NT = 4), which
// keeps the worst-case relative error of S within the 1e-5 budget at n = 4096. Cost: one extra add and loop
// level per 32 elements.
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"
#include "warp_ops.h"

using psk::NT;

template <typename Exp>
static inline __attribute__((always_inline)) void softmax_rows_warp(const ps_softmax_arg_t* arg) {
  const float* src = reinterpret_cast<const float*>(arg->src_addr);
  float* dst = reinterpret_cast<float*>(arg->dst_addr);
  const uint32_t R = arg->rows, n = arg->cols;
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = gtid % NT;
  const uint32_t nwarps = (gridDim.x * blockDim.x) / NT;
  constexpr uint32_t BLK = PS_SOFTMAX_SUM_BLOCK * NT;   // columns covered by one block partial of every lane
  for (uint32_t row = gtid / NT; row < R; row += nwarps) {
    const float* x = src + (size_t)row * n;
    float* y = dst + (size_t)row * n;
    // pass 1: row maximum, running max starts at -inf (the fix)
    float m = psk::neg_inf();
    for (uint32_t c = lane; c < n; c += NT)
      m = __builtin_fmaxf(m, x[c]);
    m = psk::warp_max(m);
    // pass 2: e = exp(x - m), stored into y, two-level sum
    float total = 0.0f;
    for (uint32_t c0 = lane; c0 < n; c0 += BLK) {
      const uint32_t c1 = (n - (c0 - lane) < BLK) ? n : (c0 - lane) + BLK;   // end of this block's columns
      float part = 0.0f;
      for (uint32_t c = c0; c < c1; c += NT) {
        float e = Exp::eval(x[c] - m);
        y[c] = e;
        part += e;
      }
      total += part;
    }
    total = psk::warp_sum(total);
    // pass 3: scale by 1/S
    const float inv = 1.0f / total;
    for (uint32_t c = lane; c < n; c += NT)
      y[c] *= inv;
  }
}

__kernel void softmax_warp(ps_softmax_arg_t* __UNIFORM__ arg) { softmax_rows_warp<psk::ExpSw>(arg); }
__kernel void softmax_warp_mufu(ps_softmax_arg_t* __UNIFORM__ arg) { softmax_rows_warp<psk::ExpMufu>(arg); }
__kernel void softmax_warp_libm(ps_softmax_arg_t* __UNIFORM__ arg) { softmax_rows_warp<psk::ExpLibm>(arg); }

// ---- exp probe: one thread per element, grid-stride; rows * cols elements
template <typename Exp>
static inline __attribute__((always_inline)) void exp_elems(const ps_softmax_arg_t* arg) {
  const float* src = reinterpret_cast<const float*>(arg->src_addr);
  float* dst = reinterpret_cast<float*>(arg->dst_addr);
  const uint32_t total = arg->rows * arg->cols;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride)
    dst[i] = Exp::eval(src[i]);
}
__kernel void exp_probe_sw(ps_softmax_arg_t* __UNIFORM__ arg) { exp_elems<psk::ExpSw>(arg); }
__kernel void exp_probe_mufu(ps_softmax_arg_t* __UNIFORM__ arg) { exp_elems<psk::ExpMufu>(arg); }
__kernel void exp_probe_libm(ps_softmax_arg_t* __UNIFORM__ arg) { exp_elems<psk::ExpLibm>(arg); }

// ---- stock reproduction (tests/regression/softmax/{kernel.cpp,common.h} at the pinned HEAD, Apache-2.0)
// The stock polynomial "exp" (common.h): Horner form of 1 + x + x^2/2 + x^3/6 + x^4/24 with 0.333 for 1/3.
static inline float stock_poly_exp(float x) {
  float y = 1 + x * 0.25;
  y = 1 + x * y * 0.333;
  y = 1 + x * y * 0.5;
  y = 1 + x * y;
  return y;
}
struct ExpStockPoly {
  static inline __attribute__((always_inline)) float eval(float x) { return stock_poly_exp(x); }
};

template <typename Exp>
static inline __attribute__((always_inline)) void softmax_rows_stock(const ps_softmax_arg_t* arg) {
  float* src0_ptr = reinterpret_cast<float*>(arg->src_addr);
  float* dst_ptr = reinterpret_cast<float*>(arg->dst_addr);
  size_t num_cols = arg->cols;
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= arg->rows)
    return;
  size_t offset = row * num_cols;
  float max = 0.0;                      // the stock initialisation (the bug under test)
  for (size_t i = 0; i < num_cols; i++) {
    float curr = src0_ptr[offset + i];
    if (curr > max) max = curr;
  }
  float sum = 0.0;
  for (size_t i = 0; i < num_cols; i++) {
    float computed = Exp::eval(src0_ptr[offset + i] - max);
    src0_ptr[offset + i] = computed;
    sum += computed;
  }
  for (size_t i = 0; i < num_cols; i++)
    dst_ptr[offset + i] = src0_ptr[offset + i] / sum;
}

__kernel void softmax_stock(ps_softmax_arg_t* __UNIFORM__ arg) { softmax_rows_stock<ExpStockPoly>(arg); }
__kernel void softmax_stock_expf(ps_softmax_arg_t* __UNIFORM__ arg) { softmax_rows_stock<psk::ExpSw>(arg); }
