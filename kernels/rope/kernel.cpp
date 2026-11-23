// kernel.cpp -- ps_rope device kernel: batched RoPE + KV-cache write, float32, run.c's formula.
//
// Design: one thread per (row b, pair i/2), grid-stride over B * dim/2 work items. Each thread evaluates its own
// frequency and angle exactly as run.c (lines 264-279):  freq = 1.0f / powf(theta, head_dim / (float)head_size),
// val = pos * freq, fcr = cosf(val), fci = sinf(val)  -- no host-side tables, so the device's libm (musl, linked by
// tests/regression/common.mk: -lm -lc from the libc32 toolchain) is what the test exercises; the host/libm differences
// are covered by the tolerance derivation in main.cpp. Rotation as run.c: v0*fcr - v1*fci, v0*fci + v1*fcr (clang may
// contract one product into an FMA; covered by the same derivation). For i < kv_dim the rotated k pair and the v pair
// are stored into row b's slot of the cache (layout in common.h), which is the "rope (+ KV write)" fusion
// run.c gets the same effect by pointing s->k / s->v into the cache before the matmuls.
#include <math.h>
#include <vx_spawn2.h>
#include "common.h"

__kernel void rope_kv(ps_rope_arg_t* __UNIFORM__ arg) {
  const float* q = reinterpret_cast<const float*>(arg->q_addr);
  const float* k = reinterpret_cast<const float*>(arg->k_addr);
  const float* v = reinterpret_cast<const float*>(arg->v_addr);
  const int32_t* pos = reinterpret_cast<const int32_t*>(arg->pos_addr);
  float* q_out = reinterpret_cast<float*>(arg->q_out_addr);
  float* kc = reinterpret_cast<float*>(arg->kc_addr);
  float* vc = reinterpret_cast<float*>(arg->vc_addr);
  const uint32_t B = arg->B, b_cap = arg->b_cap, dim = arg->dim, kv_dim = arg->kv_dim, hs = arg->head_size,
                 seq_len = arg->seq_len, layer = arg->layer;
  const float theta = arg->theta;
  const uint32_t half = dim >> 1;
  const uint32_t total = B * half;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    const uint32_t b = t / half;
    const uint32_t i = 2 * (t - b * half);
    const int p = pos[b];
    const int head_dim = (int)(i % hs);
    const float freq = 1.0f / powf(theta, head_dim / (float)hs);
    const float val = p * freq;
    const float fcr = cosf(val);
    const float fci = sinf(val);
    const float* qb = q + (size_t)b * dim;
    const float q0 = qb[i], q1 = qb[i + 1];
    float* qo = q_out + (size_t)b * dim;
    qo[i] = q0 * fcr - q1 * fci;
    qo[i + 1] = q0 * fci + q1 * fcr;
    if (i < kv_dim) {
      const size_t slot = ((size_t)(layer * b_cap + b) * seq_len + (uint32_t)p) * kv_dim;
      const float* kb = k + (size_t)b * kv_dim;
      const float* vb = v + (size_t)b * kv_dim;
      const float k0 = kb[i], k1 = kb[i + 1];
      kc[slot + i] = k0 * fcr - k1 * fci;
      kc[slot + i + 1] = k0 * fci + k1 * fcr;
      vc[slot + i] = vb[i];
      vc[slot + i + 1] = vb[i + 1];
    }
  }
}
