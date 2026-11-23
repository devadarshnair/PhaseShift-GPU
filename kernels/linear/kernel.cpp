// kernel.cpp -- ps_linear device kernels: the linear layers Y[M][N] = X[M][K] @ W^T.
//
//  gemm_small  (FPU, fp32; any config): small-M / batched-decode GEMM for M = B in 2..16 (works for any M >= 1).
//              One thread per output feature n (grid-stride over n). The thread streams its weight row W[n][:]
//              ONCE per chunk of up to 8 token rows and keeps one fp32 accumulator per row of the chunk, so for
//              M <= 8 every weight byte is read once per launch (weight-stationary per thread); for 8 < M <= 16 the
//              row is streamed twice (second pass mostly L1/L2 hits). Modeling choice: 8 accumulators bound the
//              register footprint (8 acc + 4 W + 4 X floats; no spills at -O3), and the chunk is dispatched to a
//              compile-time MB in 1..8 so the inner loop is fully unrolled. Per token row the summation order is
//              exactly gemv_rows' (kernels/gemv_f32/kernel.cpp: sum += w.x*x.x + w.y*x.y + w.z*x.z + w.w*x.w over
//              float4 chunks), so gemm_small with M = 1 computes bit-identical results to ps_gemv_f32 -v rows.
//              The X rows are shared by all threads (warp-uniform addresses -> one line per warp load).
//
//  tcu_gemm    (TCU WGMMA + DXA, fp16 in / fp32 out; only built when -DVX_CFG_TCU_WGMMA_ENABLE): the device code
//              of the stock tests/regression/sgemm_tcu_wg_dxa/kernel.cpp (Vortex, Apache-2.0), copied token for token
//              except the entry name (kernel_main -> tcu_gemm) and the argument type name (kernel_arg_t ->
//              ps_tcu_arg_t, same layout). Copied, not re-written, so that TCU timings are the stock app's
//              timings ("Prefill uses the stock sgemm_tcu_wg_dxa for timing") while ps_linear can feed
//              real weights (fixtures) for the correctness test; the host checks instrs/cycles equality against the
//              stock app at the same shape (tests/test_linear.py, results/<run>). C = A @ B with A = X (M x K fp16
//              row-major) and B = W^T (K x N fp16 row-major, transposed once on the host when the fp32 weights are
//              converted to fp16 -- an offline weight-layout step, see main.cpp).
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

namespace {
struct f4 { float x, y, z, w; };

template <int MB>
inline void rows_chunk(const float* __restrict__ w, const float* __restrict__ X, float* __restrict__ Y, uint32_t n,
                       uint32_t N, uint32_t K, uint32_t t0, bool vec) {
  float acc[MB];
#pragma unroll
  for (int m = 0; m < MB; ++m) acc[m] = 0.0f;
  const float* x0 = X + (size_t)t0 * K;
  if (vec) {
    for (uint32_t c = 0; c < K; c += 4) {
      f4 wv = *reinterpret_cast<const f4*>(w + c);
#pragma unroll
      for (int m = 0; m < MB; ++m) {
        f4 xv = *reinterpret_cast<const f4*>(x0 + (size_t)m * K + c);
        acc[m] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
      }
    }
  } else {
    for (uint32_t c = 0; c < K; ++c) {
      float wc = w[c];
#pragma unroll
      for (int m = 0; m < MB; ++m) acc[m] += wc * x0[(size_t)m * K + c];
    }
  }
#pragma unroll
  for (int m = 0; m < MB; ++m) Y[(size_t)(t0 + m) * N + n] = acc[m];
}
}  // namespace

__kernel void gemm_small(ps_gemm_small_arg_t* __UNIFORM__ arg) {
  const float* W = reinterpret_cast<const float*>(arg->W_addr);
  const float* X = reinterpret_cast<const float*>(arg->X_addr);
  float* Y = reinterpret_cast<float*>(arg->Y_addr);
  const uint32_t M = arg->M, N = arg->N, K = arg->K;
  const uint32_t stride = gridDim.x * blockDim.x;
  // float4 path only if every W row start, every X row start and both bases are 16-byte aligned
  const bool vec = ((K & 3u) == 0) && ((arg->W_addr & 15u) == 0) && ((arg->X_addr & 15u) == 0);
  for (uint32_t n = blockIdx.x * blockDim.x + threadIdx.x; n < N; n += stride) {
    const float* w = W + (size_t)n * K;
    for (uint32_t t0 = 0; t0 < M; t0 += 8) {
      const uint32_t mb = (M - t0 < 8) ? (M - t0) : 8;
      switch (mb) {  // uniform across the warp (M is a kernel argument)
        case 1: rows_chunk<1>(w, X, Y, n, N, K, t0, vec); break;
        case 2: rows_chunk<2>(w, X, Y, n, N, K, t0, vec); break;
        case 3: rows_chunk<3>(w, X, Y, n, N, K, t0, vec); break;
        case 4: rows_chunk<4>(w, X, Y, n, N, K, t0, vec); break;
        case 5: rows_chunk<5>(w, X, Y, n, N, K, t0, vec); break;
        case 6: rows_chunk<6>(w, X, Y, n, N, K, t0, vec); break;
        case 7: rows_chunk<7>(w, X, Y, n, N, K, t0, vec); break;
        default: rows_chunk<8>(w, X, Y, n, N, K, t0, vec); break;
      }
    }
  }
}

#ifdef VX_CFG_TCU_WGMMA_ENABLE
// ---------------------------------------------------------------------------------------------------------------
// BEGIN copy of third_party/vortex/tests/regression/sgemm_tcu_wg_dxa/kernel.cpp (HEAD 50cb174), Apache-2.0,
// Copyright (c) 2019-2023 the Vortex authors. Code tokens unchanged except entry and argument type names; some comments abridged (test_tcu_kernel_is_stock_copy).
#include <vx_tensor.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

// DXA descriptor slots (programmed by host in main.cpp).
[[maybe_unused]] constexpr uint32_t kDescA = 0;
[[maybe_unused]] constexpr uint32_t kDescB = 1;

__kernel void tcu_gemm(ps_tcu_arg_t* __UNIFORM__ arg) {
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
#ifdef SW_LOAD_B
  auto pB = reinterpret_cast<const ctx::input_t *>(arg->B_addr);
#endif
#ifdef SW_LOAD_A
  auto pA = reinterpret_cast<const ctx::input_t *>(arg->A_addr);
#endif

  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x;
  uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS;

  // CTA tile dimensions
  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // Shared-memory tile buffers (see the stock file for the DXA_DOUBLE_BUFFER register-residency discussion).
  //   buffer: A [cta_M x tileK] then B [tileK x xtileN]
  uint32_t a_elems   = cta_M * ctx::tileK;
  auto smem = reinterpret_cast<ctx::input_t *>(__local_mem());
  ctx::input_t* A_cur = smem;
  ctx::input_t* B_cur = smem + a_elems;

  // Barrier id: slot 0/1 selected by bit 8, CTA slot in the low byte (same
  // packing as vortex::barrier). Raw intrinsics are used instead of barrier
  // objects so the id lives in one register.
  uint32_t bar_cur = get_local_group_id();
  const uint32_t num_bar_warps = get_num_sub_groups();
#ifdef DXA_DOUBLE_BUFFER
  uint32_t buf_elems = a_elems + ctx::tileK * ctx::xtileN;
  const uintptr_t a_tog = reinterpret_cast<uintptr_t>(smem) ^
                          reinterpret_cast<uintptr_t>(smem + buf_elems);
  const uintptr_t b_tog = reinterpret_cast<uintptr_t>(smem + a_elems) ^
                          reinterpret_cast<uintptr_t>(smem + buf_elems + a_elems);
  constexpr uint32_t kBarTog = 1u << 8;
#endif

  // Initialize accumulator tile to zero.
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // Only the first warp in the CTA issues DXA commands.
  [[maybe_unused]] const bool is_dxa_warp = (get_sub_group_id() == 0);

  // Stage the tile at K-offset k into the {A_smem, B_smem} buffer guarded
  // by barrier id `bar`. All state is passed by value in registers.
  // SW_LOAD_B replaces B's DXA with a cooperative SW load (K-major);
  // SW_LOAD_A replaces A's DXA with a cooperative SW load (row-major).
  auto load_tile = [&](ctx::input_t* A_smem, ctx::input_t* B_smem,
                       uint32_t bar, uint32_t k) {
    #if defined(SW_LOAD_A) && defined(SW_LOAD_B)
      // both via SW — no DXA needed
    #elif defined(SW_LOAD_A)
      if (is_dxa_warp) { vx_barrier_expect_tx(bar, 1); vx_dxa_issue_2d_wg(kDescB, bar, B_smem, tile_col, k); }
    #elif defined(SW_LOAD_B)
      if (is_dxa_warp) { vx_barrier_expect_tx(bar, 1); vx_dxa_issue_2d_wg(kDescA, bar, A_smem, k, tile_row); }
    #else
      if (is_dxa_warp) {
        vx_barrier_expect_tx(bar, 2);  // Two pending transactions: A + B
        vx_dxa_issue_2d_wg(kDescA, bar, A_smem, k, tile_row);
        vx_dxa_issue_2d_wg(kDescB, bar, B_smem, tile_col, k);
      }
    #endif
    #ifdef SW_LOAD_A
      // Cooperative load A row-major (matches DXA row-major layout).
      uint32_t a_size = cta_M * ctx::tileK;
      for (size_t i = 0; i < a_size; i += num_threads) {
        size_t idx = i + tid;
        size_t r = idx / ctx::tileK;
        size_t c = idx % ctx::tileK;
        A_smem[r * ctx::tileK + c] = pA[(tile_row + r) * K + (k + c)];
      }
    #endif
    #ifdef SW_LOAD_B
      // Cooperative load B block-major (matches DXA BlockMajor LAYOUT / bbuf).
      uint32_t b_size = ctx::tileK * ctx::xtileN;
      for (size_t i = 0; i < b_size; i += num_threads) {
        size_t idx = i + tid;
        size_t r = idx / ctx::xtileN;
        size_t c = idx % ctx::xtileN;
        B_smem[ctx::b_blockmajor_idx(r, c)] = pB[(k + r) * N + (tile_col + c)];
      }
    #endif
    (void)A_smem; (void)B_smem; (void)bar; (void)k;
  };

  // One WGMMA accumulation step out of the given buffer pair.
  auto compute_tile = [&](ctx::input_t* A_smem, ctx::input_t* B_smem) {
    auto A_warp = A_smem + warp_rank * ctx::xtileM * ctx::tileK;
    // B layout in SMEM: block-major (bbuf-native); stride field unused.
    auto desc_b = vt::vx_make_smem_desc(B_smem, 0);
  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    // RS: A from registers, B from smem (NRC <= 16 only)
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    // SS: both from smem
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif
  };

  uint32_t num_k_tiles = K / ctx::tileK;

#ifdef DXA_DOUBLE_BUFFER
  // Software pipeline over the two buffers: stage tile i+1 into the OTHER
  // buffer, wait for tile i, compute it, release its buffer, toggle.
  load_tile(A_cur, B_cur, bar_cur, 0);
  for (uint32_t i = 0; i < num_k_tiles; ++i) {
    if (i + 1 < num_k_tiles) {
      load_tile(reinterpret_cast<ctx::input_t*>(
                    reinterpret_cast<uintptr_t>(A_cur) ^ a_tog),
                reinterpret_cast<ctx::input_t*>(
                    reinterpret_cast<uintptr_t>(B_cur) ^ b_tog),
                bar_cur ^ kBarTog, (i + 1) * ctx::tileK);
    }
    vx_barrier(bar_cur, num_bar_warps);   // tile i staged
    compute_tile(A_cur, B_cur);
    if (i + 2 < num_k_tiles) {
      vx_barrier(bar_cur, num_bar_warps); // readers done: buffer may restage
    }
    A_cur = reinterpret_cast<ctx::input_t*>(
                reinterpret_cast<uintptr_t>(A_cur) ^ a_tog);
    B_cur = reinterpret_cast<ctx::input_t*>(
                reinterpret_cast<uintptr_t>(B_cur) ^ b_tog);
    bar_cur ^= kBarTog;
  }
#else
  // Serial single-buffer schedule: stage and compute alternately in buf0.
  load_tile(A_cur, B_cur, bar_cur, 0);
  for (uint32_t i = 0; i < num_k_tiles; ++i) {
    if (i != 0) {
      load_tile(A_cur, B_cur, bar_cur, i * ctx::tileK);
    }
    vx_barrier(bar_cur, num_bar_warps);   // tile i staged
    compute_tile(A_cur, B_cur);
    if (i + 1 < num_k_tiles) {
      vx_barrier(bar_cur, num_bar_warps); // release buf0 for tile i+1
    }
  }
#endif

  // Store the computed C tile to global memory.
  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
// END copy of sgemm_tcu_wg_dxa/kernel.cpp
// ---------------------------------------------------------------------------------------------------------------
#endif  // VX_CFG_TCU_WGMMA_ENABLE
