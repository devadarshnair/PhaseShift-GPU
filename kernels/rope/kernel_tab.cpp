// kernel_tab.cpp -- (methodology §18.7): RoPE with a precomputed angle table. Two entries:
//
//   rope_table   one-time setup launch: for every position p in [pos0, pos0 + npos) (pos0 = 0 in ps_llama) and pair j < head_size/2 it evaluates, with the SAME source
//                expressions rope_kv (kernel.cpp) uses, freq = 1.0f / powf(theta, head_dim / (float)head_size), val = p * freq,
//                cosf(val), sinf(val), and stores (cos, sin) at tab[((p - pos0) * (head_size / 2) + j) * 2]. head_dim = 2 j is the
//                value rope_kv computes as i % head_size for the even pair index i. The device libm (musl) is the same, so every
//                entry is bit-equal to the value rope_kv computes per token (checked by the host against the device probe
//                and a correctly-rounded long-double recomputation, kernels/rope/rope_table_ref.h).
//   rope_kv_tab  rope_kv with fcr / fci loaded from the table instead of computed; the rotation and the KV write are the
//                same source lines, so q, k and the cache slots are bit-identical to rope_kv's.
//
// Modeling choice: angles depend only on (position, pair), so the per-token powf + cosf + sinf (3,800-6,400 soft-float
// instructions per pair, methodology §14.0) is a model-load cost paid once; it is reported separately and never put inside
// C_tok. Separate file (not kernel.cpp) so that rope_kv's translation unit, and with it the default
// ps_llama module image, is not changed by this step (ps_llama compiles this file as translation unit 10).
#include <math.h>
#include <vx_spawn2.h>
#include "common.h"

__kernel void rope_table(ps_rope_table_arg_t* __UNIFORM__ arg) {
  float* tab = reinterpret_cast<float*>(arg->tab_addr);
  const uint32_t hs = arg->head_size, half = hs >> 1, total = arg->npos * half;
  const float theta = arg->theta;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    const uint32_t pu = t / half;
    const int p = (int)(arg->pos0 + pu);
    const int head_dim = (int)(2 * (t - pu * half));
    const float freq = 1.0f / powf(theta, head_dim / (float)hs);
    const float val = p * freq;
    tab[2 * (size_t)t] = cosf(val);
    tab[2 * (size_t)t + 1] = sinf(val);
  }
}

__kernel void rope_kv_tab(ps_rope_tab_arg_t* __UNIFORM__ targ) {
  const ps_rope_arg_t* arg = &targ->a;
  const float* tab = reinterpret_cast<const float*>(targ->tab_addr);
  const float* q = reinterpret_cast<const float*>(arg->q_addr);
  const float* k = reinterpret_cast<const float*>(arg->k_addr);
  const float* v = reinterpret_cast<const float*>(arg->v_addr);
  const int32_t* pos = reinterpret_cast<const int32_t*>(arg->pos_addr);
  float* q_out = reinterpret_cast<float*>(arg->q_out_addr);
  float* kc = reinterpret_cast<float*>(arg->kc_addr);
  float* vc = reinterpret_cast<float*>(arg->vc_addr);
  const uint32_t B = arg->B, b_cap = arg->b_cap, dim = arg->dim, kv_dim = arg->kv_dim, hs = arg->head_size,
                 seq_len = arg->seq_len, layer = arg->layer;
  const uint32_t half = dim >> 1;
  const uint32_t hhalf = hs >> 1;
  const uint32_t total = B * half;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    const uint32_t b = t / half;
    const uint32_t i = 2 * (t - b * half);
    const int p = pos[b];
    const uint32_t j = (i % hs) >> 1;
    const float* cs = tab + 2 * ((size_t)(uint32_t)p * hhalf + j);
    const float fcr = cs[0];
    const float fci = cs[1];
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
