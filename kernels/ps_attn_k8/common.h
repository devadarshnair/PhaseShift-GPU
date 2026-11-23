// common.h -- ps_attn_k8 (arguments and LMEM layout of the decode attention with an INT8 K cache whose QK^T is computed with
// the DOT4ACC instruction (patches/vortex/dot4: custom-3 0x7B funct7 1, rd += sum_k int8(rs1.byte[k]) * int8(rs2.byte[k])) or, as the equal-effort
// software baseline, with scalar int8 multiplies. V, the softmax and the output stay float32 exactly as in kernels/decode_attn_gqa/kernel.cpp (attn_gqa).
// Shared by the host (main.cpp of ps_attn_k8 and of ps_llamadot4att) and the device kernels (kernel_k8.cpp).
//
// Quantization scheme (modeling choice, explained in results/<run>/README.md):
//   K: symmetric int8, ONE fp32 scale per (position, KV head) = the D elements of one cache row (per-row scale): sK = max_i |k_i| / 127.0f,
//      kq_i = (int8) roundf(k_i / sK) (kq_i = 0 when sK = 0), exactly the `kv8` K scheme (kernels/rope/kernel_kv8.cpp, runq.c `quantize` with group = D).
//   Q: symmetric int8, ONE fp32 scale per (query head) = the D elements of that head's (RoPE-rotated) q vector (per-head scale), computed in the attention
//      kernel's prologue from the fp32 q (no host help, counted in the attention cycles): sQ = max_i |q_i| / 127.0f, qq_i = (int8) roundf(q_i / sQ).
//   score s_t = (sum_i qq_i kq_{t,i}) * sQ * sK_t * scale   with the int32 sum exact (|sum| <= D * 127^2 < 2^31).
// The int8 K cache of position T-1 (the one the current token just wrote in fp32 through rope_kv) is produced INSIDE the attention kernel's prologue from the
// fp32 K cache row (arg quant_pos); the rows of the earlier positions were produced by the same code at the earlier tokens. So no extra launch is needed
// and the quantization cost is inside the measured attention cycles. (The unchanged fp32 rope_kv keeps writing the fp32 K row; it is never read again.)
#ifndef PS_ATTN_K8_COMMON_H
#define PS_ATTN_K8_COMMON_H

#include "../decode_attn_gqa/common.h"

typedef struct {
  ps_attn_arg_t a;      // a.k_addr: INT8 K cache [B][Tmax][H_kv][D] (bytes); a.v_addr: fp32 V cache (stock layout); q, o, shapes as the stock kernel
  uint64_t ks_addr;     // fp32 K scales [B][H_kv][Tmax] (layout: the scales of a key tile are contiguous)
  uint64_t kf_addr;     // fp32 K cache (stock layout [B][Tmax][H_kv][D]); only row quant_pos is read (to quantize it)
  uint32_t quant_pos;   // position whose fp32 K row is quantized into the int8 cache by the prologue (0xFFFFFFFF = none)
  uint32_t pad;
} ps_attn_k8_arg_t;

static inline uint32_t ps_k8_round16(uint32_t n) { return (n + 15u) & ~15u; }
// LMEM layout of one CTA, in 32-bit words, each region rounded up to 16 words:
//   qs[G][D] f32 | acc[G][D] | ps[G][Tt] | stat[4G+1] (m, l, c, sQ per head; sK of the new row) | kt[D] (|k| of the new row) | qb[G*D bytes] (packed int8 q) | qi[G*D] (int32 q, sw variant)
typedef struct { uint32_t qs, acc, ps, stat, kt, qb, qi, total; } ps_attn_k8_lmem_t;
static inline ps_attn_k8_lmem_t ps_attn_k8_lmem(uint32_t G, uint32_t D, uint32_t Tt) {
  ps_attn_k8_lmem_t L;
  L.qs = 0;
  L.acc = L.qs + ps_k8_round16(G * D);
  L.ps = L.acc + ps_k8_round16(G * D);
  L.stat = L.ps + ps_k8_round16(G * Tt);
  L.kt = L.stat + ps_k8_round16(4 * G + 1);
  L.qb = L.kt + ps_k8_round16(D);
  L.qi = L.qb + ps_k8_round16((G * D + 3u) / 4u);
  L.total = L.qi + ps_k8_round16(G * D);
  return L;
}

#endif
