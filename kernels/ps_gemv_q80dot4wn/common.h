// NT-generic copy of ps_gemv_q80dot4w (warp width W = NUM_THREADS instead of 32: tile R = W / S rows, lane stride W in Wl). Identical at W = 32.
// common.h -- ps_gemv_q80dot4w: warp-cooperative dot4 Q8_0 GEMV arguments.
// Same Q8_0 problem as kernels/ps_gemv_q80dot4 (W: M x N int8 + M x G float32 scales; x: B x N int8 + B x G scales; y: B x M).
// Host-side WEIGHT re-layout (not counted in cycles; the same liberty the TCU app got for its activation layout xp), variant
// with S lanes per row and R = 32 / S rows per warp tile, GW = GS / 4 words per group, WL = GW / S words per lane per group:
//   Wl  uint32 words, index ((t * G + g) * WL + i) * 32 + lane, lane = s * R + r, holds word (i * S + s) of group g of row t * R + r
//       (zero for padding rows r >= M): one warp-wide lw reads 128 contiguous bytes.
//   Wsl float32, index (t * G + g) * R + r = Ws[row][g] (pad rows 0).
// Activations xq / xs are the ORIGINAL row-major arrays (no activation re-layout, no batch padding).
#ifndef PS_GEMV_Q80DOT4WN_COMMON_H
#define PS_GEMV_Q80DOT4WN_COMMON_H
#include <stdint.h>
typedef struct {
  uint64_t Wl_addr;
  uint64_t Wsl_addr;
  uint64_t xq_addr;   // B x N int8
  uint64_t xs_addr;   // B x G float32
  uint64_t y_addr;    // B x M float32
  uint32_t M, N, GS, B;
  uint32_t G;
  uint32_t pad_;
} ps_gemv_q80dot4wn_arg_t;
#endif
