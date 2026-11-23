// fp16.h -- host-side IEEE-754 binary16 conversion for ps_linear's TCU path.
//
// The out-of-tree app Makefile (scripts/install_kernel_apps.sh) compiles only main.cpp, so the stock app's softfloat
// helpers (rvfloats.cpp + softfloat.a) are not linked; these two functions replace rv_ftoh_s / rv_htof_s for the
// one thing the host needs: rounding fp32 weights/activations to fp16 with round-to-nearest-even (the IEEE default,
// = rv_ftoh_s(..., rm = 0) as the stock app calls it, = NumPy's astype(np.float16)) and widening fp16 back to fp32
// (exact). fp16_selftest() checks both exhaustively before every run (all 63,488 finite halves round-trip, and every
// midpoint between two adjacent finite halves rounds to the one with the even mantissa, including the subnormal range,
// the subnormal/normal boundary and the overflow threshold 65,520 -> inf).
#ifndef PS_LINEAR_FP16_H
#define PS_LINEAR_FP16_H

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ps16 {

inline uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

inline uint16_t f32_to_f16(float f) {
  const uint32_t x = bits(f);
  const uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
  const uint32_t a = x & 0x7fffffffu;
  if (a >= 0x7f800000u) return sign | 0x7c00u | (a > 0x7f800000u ? 0x200u : 0u);  // inf / quiet NaN
  if (a >= 0x477ff000u) return sign | 0x7c00u;   // >= 65520 = midpoint(65504, 65536): RNE goes to inf
  if (a < 0x38800000u) {                         // < 2^-14: fp16 subnormal (or zero), unit 2^-24
    float v; std::memcpy(&v, &a, 4);
    // v * 2^24 < 2^10 is exact (power-of-two scaling); nearbyint rounds half to even in the default mode
    return sign | (uint16_t)std::nearbyint(v * 16777216.0f);
  }
  const uint32_t mant = a & 0x7fffffu;
  uint32_t h = ((((a >> 23) - 127u + 15u)) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h += 1u;  // a carry into the exponent is the correct result
  return sign | (uint16_t)h;
}

inline float f16_to_f32(uint16_t h) {
  const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  float v;
  if (e == 0) v = std::ldexp((float)m, -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp((float)(m | 0x400u), (int)e - 25);
  return (h & 0x8000u) ? -v : v;
}

// Returns the number of failures (0 = ok).
inline int fp16_selftest() {
  int bad = 0;
  for (uint32_t s = 0; s < 2; ++s) {
    for (uint32_t h = 0; h < 0x7c00u; ++h) {           // every finite magnitude
      const uint16_t hs = (uint16_t)(h | (s << 15));
      const double v = f16_to_f32(hs);
      if (f32_to_f16((float)v) != hs) ++bad;          // round trip
      // midpoint to the next half (exactly representable in fp32: 12 significant bits)
      const double nxt = (h + 1 < 0x7c00u) ? (double)f16_to_f32((uint16_t)((h + 1) | (s << 15)))
                                           : (s ? -65536.0 : 65536.0);
      const float mid = (float)((v + nxt) / 2.0);
      if ((double)mid != (v + nxt) / 2.0) { ++bad; continue; }
      const uint16_t want = (h + 1 < 0x7c00u) ? (uint16_t)(((h & 1u) ? h + 1 : h) | (s << 15))
                                              : (uint16_t)(0x7c00u | (s << 15));   // 65520 -> inf
      if (f32_to_f16(mid) != want) ++bad;
    }
  }
  if (f32_to_f16(INFINITY) != 0x7c00u || f32_to_f16(-INFINITY) != 0xfc00u) ++bad;
  if ((f32_to_f16(NAN) & 0x7c00u) != 0x7c00u || (f32_to_f16(NAN) & 0x3ffu) == 0) ++bad;
  if (f32_to_f16(1e-9f) != 0 || f32_to_f16(1e6f) != 0x7c00u) ++bad;
  return bad;
}

}  // namespace ps16

#endif  // PS_LINEAR_FP16_H
