// kernel.cpp -- ps_libm_probe device kernel: a single-thread libm probe of ps_rope's angle path.
// The expressions are copied token for token from kernels/rope/kernel.cpp (rope_kv), so for the same (pos, hd, hs) the
// float32 freq/val/cos/sin bits must equal what a correct rope_kv lane computes.
#include <math.h>
#include <vx_spawn2.h>
#include "common.h"

__kernel void libm_probe(ps_libm_probe_arg_t* __UNIFORM__ arg) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const int32_t* in = reinterpret_cast<const int32_t*>(arg->in_addr);
  float* out = reinterpret_cast<float*>(arg->out_addr);
  const float theta = arg->theta;
  for (uint32_t j = 0; j < arg->n; ++j) {
    const int p = in[3 * j];
    const int head_dim = in[3 * j + 1];
    const uint32_t hs = (uint32_t)in[3 * j + 2];
    const float freq = 1.0f / powf(theta, head_dim / (float)hs);
    const float val = p * freq;
    out[4 * j] = freq;
    out[4 * j + 1] = val;
    out[4 * j + 2] = cosf(val);
    out[4 * j + 3] = sinf(val);
  }
}
