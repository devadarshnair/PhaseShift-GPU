// common.h -- ps_libm_probe arguments. One device thread evaluates, for each of n (pos, head_dim, head_size)
// inputs, exactly ps_rope's rope_kv expressions (kernels/rope/kernel.cpp): freq = 1.0f / powf(theta, hd/(float)hs),
// val = pos * freq, cosf(val), sinf(val), and stores the four float32 results. No other thread does any work, so the
// musl/soft-float code runs with a single active lane (no SIMT divergence inside libm).
#ifndef PS_LIBM_PROBE_COMMON_H
#define PS_LIBM_PROBE_COMMON_H
#include <stdint.h>
typedef struct {
  uint64_t in_addr;   // n x {int32 pos, int32 head_dim, int32 head_size}
  uint64_t out_addr;  // n x {float freq, val, cos, sin}
  uint32_t n;
  float theta;
} ps_libm_probe_arg_t;
#endif
