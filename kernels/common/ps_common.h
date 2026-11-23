// ps_common.h -- host-side helpers shared by the PhaseShift out-of-tree Vortex apps.
//
// Every app prints, after the run, one machine-readable line
//   PSCHECK: <label> n=<N> max_abs_err=<a> max_rel_err=<r> max_elem_rel_err=<e> tol=<t> errors=<k>
// followed by "PASSED!" or "FAILED!" (the strings scripts/vxrun.py and ci/blackbox.sh look for).
//
// Error metric (modeling choice, explained here because every kernel test uses it):
//   abs_i      = |y_i - ref_i|
//   rel_i      = abs_i / scale_i          <- the PASS criterion: max_i rel_i <= tol
//   elem_rel_i = abs_i / |ref_i|          <- informational only (undefined / huge where ref_i ~ 0)
// `scale_i` is chosen by the app as the natural magnitude of the computation that produced element i. For a dot
// product y_i = sum_j a_ij x_j it is S_i = sum_j |a_ij x_j|. Why: in float32 every product and every addition
// rounds with relative error <= u = 2^-24; in any summation order (sequential, tree, 4-wide partial sums) each term
// a_ij x_j passes through at most K roundings (1 multiply + <= K-1 adds), so
//   |fl(y_i) - y_i| <= ((1 + u)^K - 1) * S_i = gamma_K * S_i,   gamma_K ~= K u  (for K u << 1).
// Two correct implementations (device vs NumPy/BLAS) therefore differ by at most 2 gamma_K S_i, i.e. rel_i <= 2 K u.
// Plain |ref_i| is not a valid scale when the dot product cancels (ref_i ~ 0 while S_i is large): two correct
// float32 implementations with different summation orders can then differ by many ulps of ref_i.
// For element-wise kernels the app passes scale_i = max(|ref_i|, tiny).
// Elements with scale_i == 0 must match exactly (abs_i == 0).
#ifndef PS_COMMON_H
#define PS_COMMON_H

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ps {

struct ErrStats {
  double max_abs = 0.0;
  double max_rel = 0.0;       // abs / scale (pass criterion)
  double max_elem_rel = 0.0;  // abs / |ref| over ref != 0 (informational)
  size_t worst = 0;           // index of the element with the largest rel
  size_t n = 0;
  size_t errors = 0;          // elements with rel > tol, NaN/Inf, or (scale == 0 and abs != 0)
  double tol = 0.0;
  // FNV-1a 64 over the IEEE-754 bit patterns of y[0..n): identifies the device output exactly, so two runs
  // (e.g. two builds of the same kernel) can be shown bit-identical, not merely equal in their error statistics.
  uint64_t digest = 0;
  bool has_digest = false;
  bool ok() const { return errors == 0 && n > 0; }
};

inline uint64_t fnv1a64_f32(const float* y, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; ++i) {
    uint32_t b;
    std::memcpy(&b, &y[i], sizeof b);
    for (int k = 0; k < 4; ++k) { h ^= (b >> (8 * k)) & 0xffu; h *= 0x100000001b3ull; }
  }
  return h;
}

inline ErrStats compare(const float* y, const double* ref, const double* scale, size_t n, double tol) {
  ErrStats s;
  s.n = n;
  s.tol = tol;
  s.digest = fnv1a64_f32(y, n);
  s.has_digest = true;
  for (size_t i = 0; i < n; ++i) {
    double yi = (double)y[i];
    if (!std::isfinite(yi)) { ++s.errors; s.max_rel = INFINITY; s.max_abs = INFINITY; s.worst = i; continue; }
    double a = std::fabs(yi - ref[i]);
    double r;
    if (scale[i] > 0.0) r = a / scale[i];
    else r = (a == 0.0) ? 0.0 : INFINITY;
    if (a > s.max_abs) s.max_abs = a;
    if (r > s.max_rel) { s.max_rel = r; s.worst = i; }
    if (ref[i] != 0.0) {
      double e = a / std::fabs(ref[i]);
      if (e > s.max_elem_rel) s.max_elem_rel = e;
    }
    if (!(r <= tol)) ++s.errors;
  }
  return s;
}

inline void print_check(const char* label, const ErrStats& s) {
  std::printf("PSCHECK: %s n=%zu max_abs_err=%.6e max_rel_err=%.6e max_elem_rel_err=%.6e tol=%.1e errors=%zu "
              "worst=%zu\n",
              label, s.n, s.max_abs, s.max_rel, s.max_elem_rel, s.tol, s.errors, s.worst);
  // separate line, so every existing PSCHECK parser is unaffected
  if (s.has_digest) std::printf("PSDIGEST: %s n=%zu fnv1a64=%016llx\n", label, s.n, (unsigned long long)s.digest);
}

// Deterministic host RNG (xorshift32) for synthetic inputs: independent of libc rand(), so a synthetic case is the
// same on every host. Values uniform in [-1, 1).
struct Rng {
  uint32_t s;
  explicit Rng(uint32_t seed) : s(seed ? seed : 0x9e3779b9u) {}
  uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
  float uniform() { return (float)((next() >> 8) * (1.0 / 16777216.0)) * 2.0f - 1.0f; }
};

struct Timer {
  std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
  double ms() const {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
  }
};

}  // namespace ps

#endif  // PS_COMMON_H
