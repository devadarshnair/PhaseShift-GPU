// common.h -- ps_gemv_q80tcuw: the stock int8 TCU Q8_0 GEMV (kernels/ps_gemv_q80tcu) given
// the same host-side WEIGHT re-layout liberty as ps_gemv_q80dot4w (one-time model-load transform, not counted in cycles):
//   Wf   uint32 words, index ((blk * T + t) * 8 + r) * 32 + lane = fragment_b register r of lane `lane` for weight-row block blk
//        (16 rows) and K tile t (64 int8), i.e. exactly the word load_matrix_sync<col_major>(fragB, W + blk*16*N + t*64, N) loads
//        (vx_tensor.h, NT 32 int8: tcN 4, tcK 4, b_sub_blocks 2, b_sub_steps 2, NRB 8): row n0 + (lane%16)/4 + (lane/16)*4 + (r%2)*8,
//        bytes t*64 + (lane%4)*4 + (r/2)*16 .. +3 (zero beyond N / beyond M). One fragment register load = 128 contiguous bytes.
//   Wsl  float32, index (blk * G + g) * 16 + n = Ws[blk*16 + n][g] (pad rows 0): the 4 per-lane scale loads of a group hit one 64 B span.
//   xp / xs  as ps_gemv_q80tcu (per-token activation layout, B padded to 16; packed layout for the pk variants).
#ifndef PS_GEMV_Q80TCUW_COMMON_H
#define PS_GEMV_Q80TCUW_COMMON_H
#include <stdint.h>
typedef struct {
  uint64_t Wf_addr;
  uint64_t Wsl_addr;
  uint64_t xp_addr;
  uint64_t xs_addr;
  uint64_t y_addr;   // B x M float32
  uint32_t M, N, GS, B;
  uint32_t G;        // N / GS
  uint32_t T;        // K tiles = ceil(N / 64)
} ps_gemv_q80tcuw_arg_t;
#endif
