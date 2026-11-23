// common.h -- ps_gemv_q80tcu: Q8_0 GEMV on Vortex's own int8 TCU (WMMA, vx_tensor.h) kernel arguments.
// Same Q8_0 definition as kernels/ps_gemv_q80dot4/common.h (W: M x N int8 + M x G float32 scales; x: B x N int8 + B x G scales; y: B x M).
// The TCU wants padded, tile-shaped operands, all prepared by the host (counted as padding in README):
//   Wq   M rows padded to Mp = round_up(M, 16) with zero rows (+64 trailing bytes: the last k-tile of a row reads past N)
//   Ws   Mp x G float32 (pad rows 0)
//   xp   S steps x 16 x 64 int8: A operand of every MMA step (see kernel.cpp): batch rows B..15 are zero
//   xs   16 x G float32 (rows B..15 zero)
#ifndef PS_GEMV_Q80TCU_COMMON_H
#define PS_GEMV_Q80TCU_COMMON_H
#include <stdint.h>
typedef struct {
  uint64_t Wq_addr;
  uint64_t Ws_addr;
  uint64_t xp_addr;
  uint64_t xs_addr;
  uint64_t y_addr;   // B x M float32
  uint32_t M, N, GS, B;
  uint32_t G;        // N / GS
  uint32_t pad_;
} ps_gemv_q80tcu_arg_t;
#endif
