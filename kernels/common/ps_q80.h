// ps_q80.h -- host-side Q8_0 helpers shared by the PhaseShift Q8_0 apps (methodology §18.5):
// a C++ port of llama2.c runq.c [cand:D15] `quantize`, `dequantize` and `matmul`, bit-identical to runq.c on the host
// (and to model/ref_q80.py, which the Tests check through the digests the apps print).
//
//  quantize_q80  per group of GS: wmax = max |x| (float), scale = wmax / 127.0f (float division), q = (int8_t)
//                round(x / scale) with the float division and the C double `round` (half away from zero); the q's range
//                is [-127, 127]. A group with scale 0 gets q = 0 (runq.c would divide by zero; see model/ref_q80.py).
//  matmul_row    one output row in runq.c's order: for each group, an exact int32 sum, then
//                val += ((float)ival * ws) * xs -- three separate float operations (the host is built without FMA
//                contraction on x86-64 baseline, and the expression is split into statements so it stays that way).
//  fnv1a64       FNV-1a 64 over bytes (digest of the quantized tensors, printed by the apps for the Python cross-check).
#ifndef PS_Q80_H
#define PS_Q80_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps {

inline void quantize_q80(const float* x, size_t n, uint32_t gs, int8_t* q, float* s) {
  const size_t groups = n / gs;
  for (size_t g = 0; g < groups; ++g) {
    float wmax = 0.0f;
    for (uint32_t i = 0; i < gs; ++i) {
      const float v = std::fabs(x[g * gs + i]);
      if (v > wmax) wmax = v;
    }
    const float scale = wmax / 127.0f;
    s[g] = scale;
    for (uint32_t i = 0; i < gs; ++i) {
      if (scale == 0.0f) { q[g * gs + i] = 0; continue; }
      const float quant_value = x[g * gs + i] / scale;
      q[g * gs + i] = (int8_t)std::round((double)quant_value);   // C `round` on a double, as runq.c's round(float)
    }
  }
}

// one row of runq.c matmul: wq / xq int8 vectors of length n, ws / xs their per-group scales
inline float matmul_row_q80(const int8_t* wq, const float* ws, const int8_t* xq, const float* xs, uint32_t n,
                            uint32_t gs, double* exact = nullptr, double* abs_sum = nullptr) {
  float val = 0.0f;
  double ex = 0.0, ab = 0.0;
  for (uint32_t j = 0; j + gs <= n; j += gs) {
    int32_t ival = 0;
    for (uint32_t k = 0; k < gs; ++k) ival += (int32_t)xq[j + k] * (int32_t)wq[j + k];
    const float f1 = (float)ival * ws[j / gs];
    const float f2 = f1 * xs[j / gs];
    val = val + f2;
    const double t = (double)ival * (double)ws[j / gs] * (double)xs[j / gs];
    ex += t;
    ab += std::fabs(t);
  }
  if (exact) *exact = ex;
  if (abs_sum) *abs_sum = ab;
  return val;
}

inline uint64_t fnv1a64_bytes(const void* p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
  const unsigned char* b = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ull; }
  return h;
}

}  // namespace ps

#endif
