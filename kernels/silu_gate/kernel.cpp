// kernel.cpp -- ps_silu_gate device kernel: SwiGLU gate, float32, run.c's formula and order
// (run.c lines 338-345: `float val = s->hb[i]; val *= (1.0f / (1.0f + expf(-val))); val *= s->hb2[i];`).
// One thread per element, grid-stride over all B * hidden elements (the op is element-wise, so the batch is just a
// longer vector). expf is musl's (libc32 toolchain, -lm), see main.cpp for the tolerance.
#include <math.h>
#include <vx_spawn2.h>
#include "common.h"

__kernel void silu_gate(ps_silu_gate_arg_t* __UNIFORM__ arg) {
  const float* hb = reinterpret_cast<const float*>(arg->hb_addr);
  const float* hb2 = reinterpret_cast<const float*>(arg->hb2_addr);
  float* out = reinterpret_cast<float*>(arg->out_addr);
  const uint32_t n = arg->n;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride) {
    float val = hb[j];
    val *= (1.0f / (1.0f + expf(-val)));
    val *= hb2[j];
    out[j] = val;
  }
}
