// rope_table_ref.h -- host recomputation and entry-by-entry check RoPE angle table (methodology §18.7).
// Included by kernels/rope/main.cpp (ps_rope) and runtime/llama_vx/main.cpp (ps_llama, through reused/rope/).
//
// What is checked. The device table (rope_table) holds, for position p and pair j (head_dim = 2j, head size hs):
//     freq = 1.0f / powf(theta, head_dim / (float)hs);  val = p * freq;  cos = cosf(val);  sin = sinf(val)     (float32 each)
// evaluated by the device's musl libm. The host cannot call musl, and glibc's sinf / cosf are different algorithms, so the
// reference is built from CORRECT ROUNDING instead, in 80-bit long double (64-bit significand, error 2^-64 relative, far below
// float32's 2^-24 half-ulp, so the chance that it disagrees with exact rounding is ~2^-40 per entry):
//     pw = (float)powl(theta, e)        e = (float)head_dim / (float)hs  (float division, as the device)
//     freq = 1.0f / pw ; val = (float)p * freq                          (float arithmetic, bit-exact IEEE: identical to the device)
//     C = cosl((long double)val), S = sinl((long double)val)            exact values of the float angle
//     ref_cos = (float)C, ref_sin = (float)S                            the correctly rounded floats
// Why this is sound: the measured musl's powf / cosf / sinf as correctly rounded (max 0.481 / 0.500 / 0.500 ulp on
// every angle and exponent the kernels use, methodology §14.0), and a result with error < 0.5 ulp IS the correctly rounded one.
// Pass criterion per entry (PSCHECK `rope_table_vs_host`, scale = ulp(ref), tol = 0.5 + 2^-9 ulp): |dev - exact| <= 0.5 ulp + 2^-9 ulp.
// Why not exactly 0.5: musl's sinf/cosf are NOT correctly rounded in the last 2^-10 ulp: they evaluate a fixed polynomial in double
// (__cosdf: |cos x - c(x)| < 2^-34.1, __sindf: relative 2^-37.5, fdlibm kernels) and round that to float, so for an exact value within
// ~2^-34.1 / 2^-24 = 2^-10.1 ulp of a rounding boundary the device may return the neighbouring float (first seen on stories260K, entry
// 2398 at 0.50023 ulp). The reduction x - k pi/2 (double, |x| < 2048) adds < 2^-40 ulp. So the sound bound is 0.5 + 2^-9 ulp (one bit of
// slack over 2^-10.1): every entry must be the correctly rounded float, or its neighbour at an exact value that close to a tie.
// `PSTABLE` counts the entries whose bit pattern differs from the correctly rounded float and lists (up to 16) each with its exact error
// in ulps; a one-bit corruption of an entry would show as a mismatch with error ~1 ulp or more, i.e. FAIL, unless the exact value happens
// to sit within 2^-9 ulp of a tie (probability < 0.4 % per corrupted entry; the cross-machine digest compare in tests/test_ep_elementwise.py
// closes that gap: the table digest must be identical at REG-CFG, E1 and PS-BASE-S1-4c-lat0, i.e. independent of memory timing).
// Because the angle val is recomputed on the host from the host's correctly rounded powf, a device powf that is
// NOT correctly rounded for some exponent also fails this check (val differs, cos/sin differ by far more than 0.5 ulp) -- it
// cannot hide as a coincidence. The check is the guard against a silent corruption of the setup launch (SimX could
// corrupt a stack round-trip inside powf under some memory timings; fixed the cache-ordering defect, but a PASSED
// tolerance check is not enough for a table that every later token reuses).
#ifndef PS_ROPE_TABLE_REF_H
#define PS_ROPE_TABLE_REF_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ps_common.h"

namespace ps_rope {

struct TableRef {
  std::vector<float> cs;        // correctly rounded (cos, sin) floats, [pos][j][2]
  std::vector<double> exact;    // the exact values (long double rounded to double: error 2^-53, i.e. 2^-29 ulp), same layout
  std::vector<double> ulp;      // spacing of floats at ref (scale of the check); >= the smallest subnormal
};

inline TableRef host_table(uint32_t hs, uint32_t npos, float theta, uint32_t pos0 = 0) {
  TableRef r;
  const uint32_t half = hs / 2;
  const size_t n = (size_t)npos * half * 2;
  r.cs.resize(n); r.exact.resize(n); r.ulp.resize(n);
  std::vector<float> freq(half);
  for (uint32_t j = 0; j < half; ++j) {
    volatile float e = (float)(int)(2 * j) / (float)hs;                          // head_dim / (float)hs
    const float pw = (float)powl((long double)theta, (long double)e);
    volatile float f = 1.0f / pw;
    freq[j] = f;
  }
  for (uint32_t p = 0; p < npos; ++p)
    for (uint32_t j = 0; j < half; ++j) {
      volatile float vv = (float)(int)(pos0 + p) * freq[j];
      const float val = vv;
      const long double c = cosl((long double)val), s = sinl((long double)val);
      const size_t o = ((size_t)p * half + j) * 2;
      const long double ex[2] = {c, s};
      for (int k = 0; k < 2; ++k) {
        const float f32 = (float)ex[k];
        r.cs[o + k] = f32;
        r.exact[o + k] = (double)ex[k];
        const float a = std::fabs(f32);
        const double u = (double)(std::nextafterf(a, INFINITY) - a);
        r.ulp[o + k] = u > 0.0 ? u : 1.401298464324817e-45;
      }
    }
  return r;
}

// Entry-by-entry comparison of a device table (npos x hs/2 x 2 floats). Prints PSCHECK rope_table_vs_host (+ its PSDIGEST) and
// PSTABLE. Returns true iff every entry is within the criterion. `label` prefixes the check name.
inline bool check_table(const char* label, const float* dev, uint32_t hs, uint32_t npos, float theta, uint32_t pos0 = 0) {
  TableRef ref = host_table(hs, npos, theta, pos0);
  const size_t n = ref.cs.size();
  ps::ErrStats s = ps::compare(dev, ref.exact.data(), ref.ulp.data(), n, 0.5 + 1.0 / 512.0);
  std::string name = std::string(label) + "rope_table_vs_host";
  ps::print_check(name.c_str(), s);
  size_t neq = 0, worst_i = 0;
  double worst_ulp = 0.0;
  for (size_t i = 0; i < n; ++i) {
    uint32_t a, b;
    std::memcpy(&a, &dev[i], 4);
    std::memcpy(&b, &ref.cs[i], 4);
    if (a != b) {
      ++neq;
      const double d = std::fabs((double)dev[i] - (double)ref.cs[i]) / ref.ulp[i];
      if (d > worst_ulp) { worst_ulp = d; worst_i = i; }
      if (neq <= 16)
        std::printf("PSTABLE-MISMATCH: %s index=%zu pos=%zu pair=%zu %s dev=%.9g(0x%08x) correctly_rounded=%.9g(0x%08x) exact_err_ulp=%.6f\n",
                    label, i, pos0 + i / 2 / (hs / 2), (i / 2) % (hs / 2), (i & 1) ? "sin" : "cos", (double)dev[i], a, (double)ref.cs[i], b,
                    std::fabs((double)dev[i] - ref.exact[i]) / ref.ulp[i]);
    }
  }
  std::printf("PSTABLE: %s hs=%u pos0=%u npos=%u entries=%zu bits_equal_to_correct_rounding=%zu mismatches=%zu max_mismatch_ulp=%.3f "
              "worst_index=%zu\n", label, hs, pos0, npos, n, n - neq, neq, worst_ulp, worst_i);
  return s.ok();
}

}  // namespace ps_rope

#endif
