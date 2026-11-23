// kernel_exp.cpp -- (methodology §18.7): SwiGLU gate with replacement exp implementations. NUMERICS-CHANGING
// repairs of the musl soft-float expf that dominates silu_gate (~760 instructions per element, methodology §14.0); musl stays the
// default (kernel.cpp, entry silu_gate, untouched). Same formula and operation order as kernel.cpp / run.c:
//   val = hb[j]; val *= 1.0f / (1.0f + EXP(-val)); val *= hb2[j];
//   silu_gate_expf_poly   EXP = psk::ExpSw::eval = ps_expf (warp_ops.h): float-only Cody-Waite range reduction + degree-7
//                         polynomial, all FMAs, <= 2 ulp (derivation in warp_ops.h and kernels/softmax_fixed/main.cpp); runs on
//                         any configuration.
//   silu_gate_expf_mufu   EXP = psk::ExpMufu::eval = vx_expf_fast(x) = ex2(x * fl(log2 e)) on the MUFU unit (vx_mufu.h); legal only
//                         on a device built with -DVX_CFG_EXT_MUFU_ENABLE (the host refuses it otherwise, ISA flag
//                         VX_ISA_EXT_MUFU). ex2 <= 2 ulp, subnormal results flushed to zero, plus the prescale error.
// Error bounds of the whole gate (derived in main.cpp, methodology §18.7): relative error of EXP e(x) (units of u = 2^-24) -> out
// error <= e(x) * E/(1+E) + 2u (E = exp(-v); the add, the reciprocal and the two products are correctly rounded, 4 x u/2).
#include <vx_spawn2.h>
#include "common.h"
#include "warp_ops.h"

template <class Exp>
static inline __attribute__((always_inline)) void silu_gate_body(ps_silu_gate_arg_t* arg) {
  const float* hb = reinterpret_cast<const float*>(arg->hb_addr);
  const float* hb2 = reinterpret_cast<const float*>(arg->hb2_addr);
  float* out = reinterpret_cast<float*>(arg->out_addr);
  const uint32_t n = arg->n;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride) {
    float val = hb[j];
    val *= (1.0f / (1.0f + Exp::eval(-val)));
    val *= hb2[j];
    out[j] = val;
  }
}

__kernel void silu_gate_expf_poly(ps_silu_gate_arg_t* __UNIFORM__ arg) { silu_gate_body<psk::ExpSw>(arg); }
__kernel void silu_gate_expf_mufu(ps_silu_gate_arg_t* __UNIFORM__ arg) { silu_gate_body<psk::ExpMufu>(arg); }
