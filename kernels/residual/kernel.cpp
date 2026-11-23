// kernel.cpp -- ps_residual device kernel: residual add out = x + branch, float32.
// One thread per element, grid-stride over all B * dim elements. A single IEEE add per element, so a correct kernel
// is bit-identical to run.c / the NumPy fixture (main.cpp checks that too).
#include <vx_spawn2.h>
#include "common.h"

__kernel void residual(ps_residual_arg_t* __UNIFORM__ arg) {
  const float* x = reinterpret_cast<const float*>(arg->x_addr);
  const float* br = reinterpret_cast<const float*>(arg->branch_addr);
  float* out = reinterpret_cast<float*>(arg->out_addr);
  const uint32_t n = arg->n;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride)
    out[j] = x[j] + br[j];
}
