// common.h -- ps_gemv_dxa kernel arguments and LMEM layout (host main.cpp and device kernel.cpp).
//
// Weight-streaming GEMV with the weights staged GMEM -> LMEM by the DXA engine. Two flavours (the P32 and P8 tracks):
//   f32  W (M, N) float32 row-major, x (B, N) float32;                 y[b][m] = sum_k W[m][k] x[b][k]
//   q80  Wq (M, N) int8 + Ws (M, N/GS) float32, xq (B, N) int8 + xs (B, N/GS) float32 (Q8_0 as kernels/gemv_q80/common.h)
// Output y (B, M) float32, B = 1..4 vectors share every staged weight tile.
//
// Tiling: a CTA is one core (all warps). A "row group" is NW rows (one per warp). The weights of a row group are staged as K-tiles
// of KT columns (f32: KT floats; q80: KT bytes, a multiple of GS) by one 2-D DXA transfer per tile (+ one for the Q8_0 scales),
// into a ring of `depth` LMEM stages. The accumulators of a warp's row stay in registers across the K-tiles of its row group.
#ifndef PS_GEMV_DXA_COMMON_H
#define PS_GEMV_DXA_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t Wq_addr;   // f32: M x N float32; q80: M x N int8
  uint64_t Ws_addr;   // q80: M x G float32 (unused for f32)
  uint64_t xq_addr;   // f32: B x N float32; q80: B x N int8
  uint64_t xs_addr;   // q80: B x G float32 (unused for f32)
  uint64_t y_addr;    // B x M float32
  uint32_t M, N, GS, B;
  uint32_t KT;        // tile columns (f32: floats, a multiple of 16; q80: bytes, a multiple of GS)
  uint32_t depth;     // DXA stages 1, 2 or 4
  uint32_t q80;       // 0 = f32 flavour, 1 = Q8_0 flavour
  uint32_t nw;        // warps per CTA = rows per row group
} ps_gemv_dxa_arg_t;

// DXA descriptor slots (programmed by the host)
#define PS_GEMVDXA_DESC_W 0u
#define PS_GEMVDXA_DESC_S 1u

static inline uint32_t psgd_r64(uint32_t n) { return (n + 63u) & ~63u; }

// LMEM layout in bytes (every region rounded up to 64 B):  x region | stage 0 | ... | stage depth-1
//   f32: x = B*N floats;                            stage = NW rows x KT floats
//   q80: x = xq (B*N bytes) | xs (B*G floats);     stage = Wq tile (NW rows x KT bytes) | Ws tile (NW rows x KT/GS floats)
typedef struct {
  uint32_t x_off, xs_off, stage0, stage_bytes, ws_off /* within a stage */, total;
} psgd_lmem_t;

static inline psgd_lmem_t psgd_lmem(uint32_t N, uint32_t GS, uint32_t B, uint32_t KT, uint32_t depth, uint32_t q80, uint32_t nw) {
  psgd_lmem_t L;
  L.x_off = 0;
  if (q80) {
    L.xs_off = psgd_r64(B * N);
    L.stage0 = L.xs_off + psgd_r64(B * (N / GS) * 4);
    L.ws_off = psgd_r64(nw * KT);
    L.stage_bytes = L.ws_off + psgd_r64(nw * (KT / GS) * 4);
  } else {
    L.xs_off = 0;
    L.stage0 = psgd_r64(B * N * 4);
    L.ws_off = 0;
    L.stage_bytes = psgd_r64(nw * KT * 4);
  }
  L.total = L.stage0 + depth * L.stage_bytes;
  return L;
}

#endif
