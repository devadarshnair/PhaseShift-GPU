// warp_ops.h -- device-side warp helpers shared by ps_softmax_fixed and ps_decode_attn_gqa (which
// reaches this file through a symlink so that vxrun's app-source hash covers it).
//
// Warp-wide reductions over the NT lanes of one warp with the butterfly shuffle vx_shfl_bfly (vx_intrinsics.h), the
// pattern ps_gemv_f32's `lanes` kernel uses (kernels/gemv_f32/kernel.cpp): after log2(NT) steps every lane holds
// the reduction of all NT lanes. All NT lanes of the warp must be active when these are called (the callers make
// the row/work index warp-uniform and launch whole warps). clamp = NT-1 and segmask = ~(NT-1) & 0x3f select one
// NT-lane segment (= VX_QUAD_CVAL / VX_QUAD_MASK at NT = 4).
//
// Exp: three implementations, chosen per kernel entry point (not by #ifdef) so one kernel binary serves both
// configurations:
//   ExpSw    ps_expf() below: float-only software exp (default). Why not libm: the device libm is musl
//            (~/tools/libc32/bin/musl-clang; linked by tests/regression/common.mk: VX_LIBS += -lm), whose expf
//            evaluates in double precision; rv32imaf has no D extension, so every call runs soft-float
//            __muldf3/__adddf3 (~750 instructions per call, measured exp probe; found the same for
//            expf/sinf/cosf/powf, methodology §14.0). A GPU's non-SFU exp is a float polynomial, so ExpSw is the
//            fair "without MUFU" cost.
//   ExpLibm  musl expf (kept to measure the cost above).
//   ExpMufu  vx_expf_fast(x) = vx_ex2f(x * log2(e)) (sw/kernel/include/vx_mufu.h): the MUFU EX2 instruction, legal
//            only when SimX is built with -DVX_CFG_EXT_MUFU_ENABLE (VX_config.toml:47, off by default); the host
//            refuses the MUFU entry points on a device without the extension (ISA flag VX_ISA_EXT_MUFU).
//            MUFU results are approximate: ex2 within 2 ulp, subnormal results flushed to zero (vx_mufu.h header,
//            tests/regression/mufu/main.cpp accuracy bounds), plus the prescale x * fl(log2 e) (its rounding and the
//            constant's own error: (1.224 |x| + 4) u relative in total, derived in softmax_fixed/main.cpp).
//
// ps_expf (modeling choice; accuracy measured on the device exp probe, tests/test_softmax_fixed.py):
//   1. clamp x to [-110, 88.8] (exp(-110) = 1.7e-48 rounds to 0 in float32; exp(88.8) overflows to +inf). NaN input
//      is not propagated (fminf/fmaxf drop it); softmax/attention inputs are finite.
//   2. Cody-Waite range reduction: k = round(x log2 e) with the 1.5*2^23 shifter (exact for |x log2 e| < 2^22),
//      r = x - k ln2_hi - k ln2_lo with fdlibm's split ln2_hi = 0x3f317200 (9 trailing zero bits, so k ln2_hi is
//      exact for |k| <= 2^9) and ln2_lo = 0x35bfbe8e, both as explicit FMAs; |r| <= ln2/2 + tiny.
//   3. e^r by its degree-7 Taylor polynomial in Horner form (FMAs): truncation r^8/8! <= 0.3466^8/40320 = 5.2e-9 =
//      0.09 u relative, so the error is dominated by the Horner roundings (<= ~1 ulp).
//   4. scale by 2^k as 2^k1 * 2^k2 with k1 = k >> 1, k2 = k - k1 (each in [-80, 64], a normal float built from its
//      exponent bits), so results in float32's subnormal range (x < -87.34) are rounded once in the last multiply
//      instead of being flushed, and x > 88.72 overflows to +inf.
#ifndef PS_WARP_OPS_H
#define PS_WARP_OPS_H

#include <math.h>
#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_mufu.h>

#ifndef VX_CFG_NUM_THREADS
#define VX_CFG_NUM_THREADS 4
#endif

namespace psk {

constexpr uint32_t NT = VX_CFG_NUM_THREADS;
constexpr int kClamp = (int)NT - 1;
constexpr int kSegMask = ~kClamp & 0x3f;

static inline size_t bits_from_f32(float v) { uint32_t u; __builtin_memcpy(&u, &v, 4); return u; }
static inline float f32_from_bits(size_t u) { uint32_t b = (uint32_t)u; float v; __builtin_memcpy(&v, &b, 4); return v; }

static inline __attribute__((always_inline)) float shfl_bfly_f32(float v, uint32_t off) {
  return f32_from_bits(vx_shfl_bfly(bits_from_f32(v), (int)off, kClamp, kSegMask));
}

// max over the warp; fmaxf (fmax.s) so there is no float select that could lower to a branch, and -inf identities
// of lanes without elements drop out
static inline __attribute__((always_inline)) float warp_max(float v) {
#pragma unroll
  for (uint32_t off = NT >> 1; off > 0; off >>= 1)
    v = __builtin_fmaxf(v, shfl_bfly_f32(v, off));
  return v;
}

static inline __attribute__((always_inline)) float warp_sum(float v) {
#pragma unroll
  for (uint32_t off = NT >> 1; off > 0; off >>= 1)
    v += shfl_bfly_f32(v, off);
  return v;
}

static inline __attribute__((always_inline)) float f32_from_u32(uint32_t b) { float v; __builtin_memcpy(&v, &b, 4); return v; }
static inline __attribute__((always_inline)) uint32_t u32_from_f32(float v) { uint32_t b; __builtin_memcpy(&b, &v, 4); return b; }

static inline __attribute__((always_inline)) float ps_expf(float x) {
  x = __builtin_fminf(__builtin_fmaxf(x, -110.0f), 88.8f);
  const float kShift = 12582912.0f;                       // 1.5 * 2^23
  float kd = __builtin_fmaf(x, 1.44269504088896341f, kShift);
  const int32_t k = (int32_t)(u32_from_f32(kd) - u32_from_f32(kShift));
  kd -= kShift;                                           // k as a float, exact
  float r = __builtin_fmaf(-kd, f32_from_u32(0x3f317200u), x);   // ln2_hi
  r = __builtin_fmaf(-kd, f32_from_u32(0x35bfbe8eu), r);         // ln2_lo
  float p = 1.98412698412698413e-4f;                      // 1/7!
  p = __builtin_fmaf(p, r, 1.38888888888888889e-3f);      // 1/6!
  p = __builtin_fmaf(p, r, 8.33333333333333333e-3f);      // 1/5!
  p = __builtin_fmaf(p, r, 4.16666666666666667e-2f);      // 1/4!
  p = __builtin_fmaf(p, r, 1.66666666666666667e-1f);      // 1/3!
  p = __builtin_fmaf(p, r, 0.5f);
  p = __builtin_fmaf(p, r, 1.0f);
  p = __builtin_fmaf(p, r, 1.0f);
  const int32_t k1 = k >> 1, k2 = k - k1;
  const float s1 = f32_from_u32((uint32_t)(k1 + 127) << 23), s2 = f32_from_u32((uint32_t)(k2 + 127) << 23);
  return (p * s1) * s2;
}

struct ExpSw {
  static inline __attribute__((always_inline)) float eval(float x) { return ps_expf(x); }
};
struct ExpLibm {
  static inline __attribute__((always_inline)) float eval(float x) { return expf(x); }
};
struct ExpMufu {
  static inline __attribute__((always_inline)) float eval(float x) { return vx_expf_fast(x); }
};

static inline float neg_inf() { return -__builtin_inff(); }

}  // namespace psk

#endif  // PS_WARP_OPS_H
