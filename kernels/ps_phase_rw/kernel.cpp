// kernel.cpp -- ps_phase_rw device kernel: y[i] = x[i] * a + b in uint32 (exact, wraps mod 2^32).
// The kernel only gives each launch real work and a checkable result; the test is about the [dcr_phase] config the
// host writes between launches (main.cpp).
#include <vx_spawn2.h>
#include "common.h"

__kernel void phase_rw(ps_phase_rw_arg_t* __UNIFORM__ arg) {
  const uint32_t* x = reinterpret_cast<const uint32_t*>(arg->x_addr);
  uint32_t* y = reinterpret_cast<uint32_t*>(arg->y_addr);
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n; i += stride)
    y[i] = x[i] * arg->a + arg->b;
}
