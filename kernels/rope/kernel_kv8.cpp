// kernel_kv8.cpp -- (methodology §18.9): RoPE (angles table) with an INT8 KV-cache write.
//
//   rope_kv_tab_q8   work items t < B * dim / 2: rotate q exactly as rope_kv_tab (same expressions; this translation unit is compiled with
//                    fp contract off, so every product and sum is a separate float32 operation, as the NumPy port model/ref_q80.py evaluates it);
//                    work items B * dim / 2 + (which * B + b) * H_kv + kvh, which = 0 (K), 1 (V): ONE thread per (sequence b, KV head, K or V)
//                    forms the head's D = head_size values (K: rotated by the table angles, V: copied), computes
//                        scale = max_i |x_i| / 127.0f,    q_i = (int8) roundf(x_i / scale)  (q_i = 0 when scale = 0),
//                    which is runq.c's `quantize` with group = the head (GS = D), and stores the D int8 values at the (layer, b, pos) slot of the
//                    int8 cache and the scale at the position-contiguous scale layout of common_kv8.h.
// Why one thread per head (not per pair): the scale needs max over the head's D values, and one thread keeps the reduction exact and
// order-free (max is order independent anyway) without cross-thread traffic; the cost is B * H_kv * 2 threads doing 2 passes over D values
// (the rotation is recomputed in pass 2 from the same inputs, so pass 1 and pass 2 see bit-identical values), measured in the PSL line of
// op rope_kv (methodology §18.9). roundf is musl's (round half away from zero), equal to runq.c's double `round` of the same float32 quotient.
#include <math.h>
#include <vx_spawn2.h>
#include "common_kv8.h"

#pragma clang fp contract(off)

__kernel void rope_kv_tab_q8(ps_rope_q8_arg_t* __UNIFORM__ qarg) {
  const ps_rope_arg_t* arg = &qarg->t.a;
  const float* tab = reinterpret_cast<const float*>(qarg->t.tab_addr);
  const float* q = reinterpret_cast<const float*>(arg->q_addr);
  const float* k = reinterpret_cast<const float*>(arg->k_addr);
  const float* v = reinterpret_cast<const float*>(arg->v_addr);
  const int32_t* pos = reinterpret_cast<const int32_t*>(arg->pos_addr);
  float* q_out = reinterpret_cast<float*>(arg->q_out_addr);
  int8_t* kc = reinterpret_cast<int8_t*>(arg->kc_addr);
  int8_t* vc = reinterpret_cast<int8_t*>(arg->vc_addr);
  float* ks = reinterpret_cast<float*>(qarg->ks_addr);
  float* vs = reinterpret_cast<float*>(qarg->vs_addr);
  const uint32_t B = arg->B, b_cap = arg->b_cap, dim = arg->dim, kv_dim = arg->kv_dim, hs = arg->head_size,
                 seq_len = arg->seq_len, layer = arg->layer, H_kv = qarg->H_kv;
  const uint32_t half = dim >> 1, hhalf = hs >> 1;
  const uint32_t nq = B * half, nkv = B * H_kv, total = nq + 2 * nkv;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    if (t < nq) {
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
    } else {
      const uint32_t e = t - nq;
      const uint32_t which = e / nkv;
      const uint32_t r = e - which * nkv;
      const uint32_t b = r / H_kv, kvh = r - b * H_kv;
      const uint32_t p = (uint32_t)pos[b];
      const float* src = (which ? v : k) + (size_t)b * kv_dim + (size_t)kvh * hs;
      const float* cs = tab + 2 * (size_t)p * hhalf;
      int8_t* dst = (which ? vc : kc) + ((size_t)(layer * b_cap + b) * seq_len + p) * kv_dim + (size_t)kvh * hs;
      float* sdst = (which ? vs : ks) + ((size_t)(layer * b_cap + b) * H_kv + kvh) * seq_len + p;
      float wmax = 0.0f;
      for (uint32_t jj = 0; jj < hhalf; ++jj) {
        float r0 = src[2 * jj], r1 = src[2 * jj + 1];
        if (!which) {
          const float x0 = r0, x1 = r1, fcr = cs[2 * jj], fci = cs[2 * jj + 1];
          r0 = x0 * fcr - x1 * fci;
          r1 = x0 * fci + x1 * fcr;
        }
        const float a0 = fabsf(r0), a1 = fabsf(r1);
        if (a0 > wmax) wmax = a0;
        if (a1 > wmax) wmax = a1;
      }
      const float scale = wmax / 127.0f;
      *sdst = scale;
      for (uint32_t jj = 0; jj < hhalf; ++jj) {
        float r0 = src[2 * jj], r1 = src[2 * jj + 1];
        if (!which) {
          const float x0 = r0, x1 = r1, fcr = cs[2 * jj], fci = cs[2 * jj + 1];
          r0 = x0 * fcr - x1 * fci;
          r1 = x0 * fci + x1 * fcr;
        }
        int32_t q0 = 0, q1 = 0;
        if (scale != 0.0f) {
          q0 = (int32_t)roundf(r0 / scale);
          q1 = (int32_t)roundf(r1 / scale);
        }
        dst[2 * jj] = (int8_t)q0;
        dst[2 * jj + 1] = (int8_t)q1;
      }
    }
  }
}
