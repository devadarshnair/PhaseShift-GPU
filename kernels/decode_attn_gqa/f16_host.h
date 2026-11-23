// f16_host.h -- host-side IEEE binary16 conversions kv16 cache (methodology §18.9): float32 -> binary16 round to nearest even
// (the cache the host builds for gqa_kv16 / dxa_kv16) and binary16 -> float32 (exact; the operand the host reference attends over, equal to what the
// device's Zfhmin `fcvt.s.h` produces). Tested against NumPy's float16 on random and edge values by tests/test_ep_kv8.py.
#ifndef PS_F16_HOST_H
#define PS_F16_HOST_H

#include <cstdint>
#include <cstring>

namespace ps_f16 {

// float32 -> IEEE binary16 bits, round to nearest even (kv16 cache); overflow -> inf, subnormals handled
inline uint16_t f32_to_f16_rne(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t e = (int32_t)((x >> 23) & 0xffu);
  uint32_t m = x & 0x7fffffu;
  if (e == 255) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));   // inf / NaN
  int32_t he = e - 127 + 15;
  if (he >= 31) return (uint16_t)(sign | 0x7c00u);                          // overflow
  if (he <= 0) {                                                            // subnormal half or zero
    if (he < -10) return (uint16_t)sign;
    m |= 0x800000u;
    const uint32_t shift = (uint32_t)(14 - he);                             // 24-bit significand -> 10 bits
    uint32_t h = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u))) ++h;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)he << 10) | (m >> 13);
  const uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;                   // may carry into the exponent (correct)
  return (uint16_t)(sign | h);
}
inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
  if (e == 0) {
    if (m == 0) x = sign;
    else {   // subnormal: normalise
      int sh = 0;
      while (!(m & 0x400u)) { m <<= 1; ++sh; }
      x = sign | ((uint32_t)(127 - 15 + 1 - sh) << 23) | ((m & 0x3ffu) << 13);
    }
  } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
  else x = sign | ((e + 127 - 15) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

}  // namespace ps_f16

#endif
