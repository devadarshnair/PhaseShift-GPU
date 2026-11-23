// kernel.cpp -- ps_mix_micro device kernel (mechanism B microbenchmark: decode GEMV demand traffic
// concurrent with a DXA bulk copy, on every core at the same time).
//
// One CTA per core (block = NW x NT threads, so the KMU places exactly one CTA on each core). Inside a CTA:
//   warp 0       = the BULK warp: copies its share of the bulk buffer (tiles b, b + grid, b + 2 grid, ...; the CTAs'
//                  streams interleave in address) into LMEM with 1D DXA transfers, `depth` tiles in flight (a ring of
//                  `depth` LMEM slots, one single-warp barrier per slot), and sums every word it finds in LMEM (so
//                  the copy is checked exactly and the LMEM data is consumed);
//   warps 1..NW-1 = the DEMAND warps: a decode-style GEMV y = A x over uint32 (wrap-around products, exact host
//                  check), row r handled by GEMV warp r mod G (G = grid x (NW - 1)), lanes strided over 16-byte
//                  column vectors, so every warp load covers 512 contiguous bytes of A; A is read once (no reuse,
//                  like a weight stream at batch 1) and x stays L1/L2-resident.
// DXA reads are tagged BULK_DXA where DxaCore builds them (patch mem-class); the GEMV loads are ORDINARY (no ranges
// programmed), i.e. DEMAND for mechanism B. Each warp records its start and end mcycle (lane 0), so the host can
// report the DEMAND completion span and the BULK completion span separately from the kernel cycles.
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"

struct u4 { uint32_t x, y, z, w; };

static inline void stamp(uint32_t* p, uint64_t c) {
  p[0] = (uint32_t)c;
  p[1] = (uint32_t)(c >> 32);
}

__kernel void mix(ps_mix_arg_t* __UNIFORM__ arg) {
  const uint32_t warp = get_sub_group_id();
  const uint32_t lane = (uint32_t)vx_thread_id();
  const uint32_t nw = get_num_sub_groups();
  const uint32_t NT = (uint32_t)vx_num_threads();
  uint32_t* cyc = reinterpret_cast<uint32_t*>(arg->cyc_addr) + ((size_t)blockIdx.x * nw + warp) * 4;
  const uint64_t t0 = vx_rdcycle();
  if (warp == 0) {
    uint32_t* lm = reinterpret_cast<uint32_t*>(__local_mem());
    const uint32_t tw = arg->tile_words, depth = arg->depth, grid = gridDim.x;
    const uint32_t ntiles = (arg->bulk_tiles > blockIdx.x) ? (arg->bulk_tiles - blockIdx.x + grid - 1) / grid : 0;
    vortex::barrier bar[PS_MIX_MAX_DEPTH] = {vortex::barrier(1, 1), vortex::barrier(2, 1), vortex::barrier(3, 1),
                                             vortex::barrier(4, 1)};
    uint32_t acc = 0;
    for (uint32_t k = 0; k < depth && k < ntiles; ++k) {
      bar[k].expect_tx(1);   // whole warp, as the stock dxa_copy kernel issues
      vx_dxa_issue_1d_wg(PS_MIX_DXA_DESC, bar[k].id(), lm + k * tw, (blockIdx.x + k * grid) * tw);
    }
    for (uint32_t k = 0; k < ntiles; ++k) {
      const uint32_t s = k % depth;
      bar[s].arrive_and_wait();
      const uint32_t* t = lm + s * tw;
      for (uint32_t i = lane; i < tw; i += NT)
        acc += t[i];
      // In-order issue: the adds above consumed every load of slot s before the warp re-arms it below.
      const uint32_t kn = k + depth;
      if (kn < ntiles) {
        bar[s].expect_tx(1);
        vx_dxa_issue_1d_wg(PS_MIX_DXA_DESC, bar[s].id(), t, (blockIdx.x + kn * grid) * tw);
      }
    }
    reinterpret_cast<uint32_t*>(arg->bsum_addr)[(size_t)blockIdx.x * NT + lane] = acc;
  } else if (arg->gemv_on) {
    const u4* A = reinterpret_cast<const u4*>(arg->a_addr);
    const u4* X = reinterpret_cast<const u4*>(arg->x_addr);
    uint32_t* part = reinterpret_cast<uint32_t*>(arg->part_addr);
    const uint32_t G = gridDim.x * (nw - 1);
    const uint32_t gw = blockIdx.x * (nw - 1) + (warp - 1);
    const uint32_t cv = arg->cols / 4;
    for (uint32_t r = gw; r < arg->rows; r += G) {
      const u4* row = A + (size_t)r * cv;
      uint32_t acc = 0;
      for (uint32_t j = lane; j < cv; j += NT) {
        const u4 a = row[j], x = X[j];
        acc += a.x * x.x + a.y * x.y + a.z * x.z + a.w * x.w;
      }
      part[(size_t)r * NT + lane] = acc;
    }
  }
  const uint64_t t1 = vx_rdcycle();
  if (lane == 0) {
    stamp(cyc, t0);
    stamp(cyc + 2, t1);
  }
}

// PhaseShift (patch dxa-class): the -P variant (L2-prefetch-only bulk descriptor). A separate entry point, so the
// stock `mix` kernel above keeps its exact instruction stream.
__kernel void mix_l2pf(ps_mix_arg_t* __UNIFORM__ arg) {
  const uint32_t warp = get_sub_group_id();
  const uint32_t lane = (uint32_t)vx_thread_id();
  const uint32_t nw = get_num_sub_groups();
  const uint32_t NT = (uint32_t)vx_num_threads();
  uint32_t* cyc = reinterpret_cast<uint32_t*>(arg->cyc_addr) + ((size_t)blockIdx.x * nw + warp) * 4;
  const uint64_t t0 = vx_rdcycle();
  if (warp == 0) {
    uint32_t* lm = reinterpret_cast<uint32_t*>(__local_mem());
    const uint32_t tw = arg->tile_words, depth = arg->depth, grid = gridDim.x;
    const uint32_t ntiles = (arg->bulk_tiles > blockIdx.x) ? (arg->bulk_tiles - blockIdx.x + grid - 1) / grid : 0;
    vortex::barrier bar[PS_MIX_MAX_DEPTH] = {vortex::barrier(1, 1), vortex::barrier(2, 1), vortex::barrier(3, 1),
                                             vortex::barrier(4, 1)};
    uint32_t acc = 0;
    for (uint32_t k = 0; k < depth && k < ntiles; ++k) {
      bar[k].expect_tx(1);   // whole warp, as the stock dxa_copy kernel issues
      vx_dxa_issue_1d_wg(PS_MIX_DXA_DESC, bar[k].id(), lm + k * tw, (blockIdx.x + k * grid) * tw);
    }
    for (uint32_t k = 0; k < ntiles; ++k) {
      const uint32_t s = k % depth;
      bar[s].arrive_and_wait();
      const uint32_t* t = lm + s * tw;
      // PhaseShift (patch dxa-class): L2-prefetch-only stream; LMEM slot s holds nothing, read the tile from GMEM
      const uint32_t* g = reinterpret_cast<const uint32_t*>(arg->bulk_addr) + (size_t)(blockIdx.x + k * grid) * tw;
      for (uint32_t i = lane; i < tw; i += NT)
        acc += g[i];
      const uint32_t kn = k + depth;
      if (kn < ntiles) {
        bar[s].expect_tx(1);
        vx_dxa_issue_1d_wg(PS_MIX_DXA_DESC, bar[s].id(), t, (blockIdx.x + kn * grid) * tw);
      }
    }
    reinterpret_cast<uint32_t*>(arg->bsum_addr)[(size_t)blockIdx.x * NT + lane] = acc;
  } else if (arg->gemv_on) {
    const u4* A = reinterpret_cast<const u4*>(arg->a_addr);
    const u4* X = reinterpret_cast<const u4*>(arg->x_addr);
    uint32_t* part = reinterpret_cast<uint32_t*>(arg->part_addr);
    const uint32_t G = gridDim.x * (nw - 1);
    const uint32_t gw = blockIdx.x * (nw - 1) + (warp - 1);
    const uint32_t cv = arg->cols / 4;
    for (uint32_t r = gw; r < arg->rows; r += G) {
      const u4* row = A + (size_t)r * cv;
      uint32_t acc = 0;
      for (uint32_t j = lane; j < cv; j += NT) {
        const u4 a = row[j], x = X[j];
        acc += a.x * x.x + a.y * x.y + a.z * x.z + a.w * x.w;
      }
      part[(size_t)r * NT + lane] = acc;
    }
  }
  const uint64_t t1 = vx_rdcycle();
  if (lane == 0) {
    stamp(cyc, t0);
    stamp(cyc + 2, t1);
  }
}
