// kernel_kv.cpp -- ps_decode_attn_gqa device kernels with a reduced-precision KV cache (methodology §18.9). The stock entries
// of kernel.cpp (attn_naive, attn_gqa, attn_dxa) are unchanged; this translation unit (linked after it) adds
//   attn_gqa_kv8  / attn_dxa_kv8    int8 K and V + one fp32 scale per (position, KV head)       (`gqa_kv8`, `dxa_kv8`)
//   attn_gqa_kv16 / attn_dxa_kv16   fp16 K and V, no scales; compiled only with Zfhmin (-DVX_CFG_EXT_ZFHMIN_ENABLE, __riscv_zfhmin)
// Structure and numerics are kernel.cpp's (fused QK^T -> online softmax -> PV, one CTA per (sequence, KV head) serving its G query heads,
// phases S / M1 / M2 / V separated by CTA barriers, ps_expf, same accumulation orders); the ONLY change is where the K / V operands come from:
//   * the element is loaded narrow and dequantized in the core right after the load: kd = (float) k_int8 * ks[t] (kv8: ONE float32 product,
//     so the operand equals the host / NumPy dequantized value bit for bit) or kd = (float) k_fp16 (kv16: exact);
//   * plain kernel: int8 K rows are read as 32-bit words (4 elements per load, sign-extended with shifts) in the S phase (thread per position);
//     V is read element-wise (thread per output column, as the stock kernel; consecutive lanes read consecutive bytes, so a warp's loads
//     coalesce into one line). The scales of the tile (Tt floats each for K and V) are copied to LMEM by the CTA's threads first;
//   * DXA kernel: the K and V tiles are 4-D descriptors over the narrow cache (1- or 2-byte elements, tile D x 1 x Tt x 1), the scales
//     3-D descriptors over [B][H_kv][Tmax] (4-byte elements, tile Tt x 1 x 1); all land in LMEM through the same `depth`-stage ring and
//     the stage barrier counts 4 (kv8) or 2 (kv16) transfers (expect_tx), dequantization happens after the LMEM read;
//   * the error bound of kernel.cpp / main.cpp (methodology §14.2) holds unchanged against the float64 attention over the DEQUANTIZED operands,
//     because the dequantized operand the kernel uses is exactly the one the host reference uses (the quantization error itself is a property
//     of the cache, reported separately, methodology §18.9).
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#include "common_kv.h"
#include "warp_ops.h"
#include "ps_hint.h"

namespace {

enum class Mode { Plain, Dxa };

static inline __attribute__((always_inline)) int32_t byte_s32(uint32_t w, int k) {
  return (int32_t)(w << (24 - 8 * k)) >> 24;
}

template <uint32_t G, Mode MODE, bool I8>
static inline __attribute__((always_inline)) void attn_kv_cta(const ps_attn_kv_arg_t* karg) {
  const ps_attn_arg_t* arg = &karg->a;
  const uint32_t D = arg->D, T = arg->T, Tt = arg->Tt, H_kv = arg->H_kv, Tmax = arg->Tmax;
  const uint32_t groups = arg->H_q / G;
  const uint32_t cta = blockIdx.x;
  const uint32_t b = cta / groups, h0 = (cta % groups) * G, kvh = h0 / arg->kv_mul;
  const uint32_t tid = threadIdx.x, nth = blockDim.x;
  const float scale = arg->scale;
  const float* q = reinterpret_cast<const float*>(arg->q_addr) + ((size_t)b * arg->H_q + h0) * D;
  const uint8_t* kbase = reinterpret_cast<const uint8_t*>(arg->k_addr);
  const uint8_t* vbase = reinterpret_cast<const uint8_t*>(arg->v_addr);
  float* out = reinterpret_cast<float*>(arg->o_addr) + ((size_t)b * arg->H_q + h0) * D;
  constexpr size_t ESZ = I8 ? 1 : 2;
  const size_t row_stride = (size_t)H_kv * D * ESZ;                              // bytes between positions
  const size_t seq_off = ((size_t)b * Tmax * H_kv + kvh) * D * ESZ;              // K[b][0][kvh][0] in bytes
  const size_t sc_off = ((size_t)b * H_kv + kvh) * Tmax;                         // Ks[b][kvh][0] in floats
  const float* ks_g = reinterpret_cast<const float*>(karg->ks_addr) + sc_off;
  const float* vs_g = reinterpret_cast<const float*>(karg->vs_addr) + sc_off;

  const ps_attn_kv_lmem_t L = ps_attn_kv_lmem(G, D, Tt, arg->depth, MODE == Mode::Dxa, (uint32_t)ESZ);
  float* lm = reinterpret_cast<float*>(__local_mem());
  float* qs = lm + L.qs;
  float* acc = lm + L.acc;
  float* ps = lm + L.ps;
  float* mst = lm + L.stat;
  float* lst = mst + G;
  float* cst = lst + G;

  vortex::barrier sync(5);

  for (uint32_t i = tid; i < G * D; i += nth) { qs[i] = q[i]; acc[i] = 0.0f; }
  for (uint32_t h = tid; h < G; h += nth) { mst[h] = psk::neg_inf(); lst[h] = 0.0f; }
  sync.arrive_and_wait();

  const uint32_t ntiles = (T + Tt - 1) / Tt;
  const uint32_t depth = arg->depth;
  vortex::barrier stage_bar[4] = {vortex::barrier(1), vortex::barrier(2), vortex::barrier(3), vortex::barrier(4)};
  const bool issuer = (MODE == Mode::Dxa) && (get_sub_group_id() == 0);
  auto issue = [&](uint32_t j) {   // DXA tile j -> stage j % depth (whole issuer warp executes it)
    const uint32_t s = j % depth;
    stage_bar[s].expect_tx(I8 ? 4 : 2);
    vx_dxa_issue_4d_wg(PS_DXA_DESC_K, stage_bar[s].id(), lm + L.kb + s * L.data_stage, 0, kvh, j * Tt, b);
    vx_dxa_issue_4d_wg(PS_DXA_DESC_V, stage_bar[s].id(), lm + L.vb + s * L.data_stage, 0, kvh, j * Tt, b);
    if (I8) {
      vx_dxa_issue_3d_wg(PS_DXA_DESC_KS, stage_bar[s].id(), lm + L.ksc + s * L.sc_stage, j * Tt, kvh, b);
      vx_dxa_issue_3d_wg(PS_DXA_DESC_VS, stage_bar[s].id(), lm + L.vsc + s * L.sc_stage, j * Tt, kvh, b);
    }
  };
  if (MODE == Mode::Dxa && issuer)
    for (uint32_t j = 0; j + 1 < depth && j < ntiles; ++j) issue(j);

  PS_MEM_HINT(PS_CLASS_KV_STREAM);
  for (uint32_t j = 0; j < ntiles; ++j) {
    const uint32_t t0 = j * Tt;
    const uint32_t nt = (T - t0 < Tt) ? T - t0 : Tt;
    const uint8_t* krows;   // row tt of the tile: krows + tt * kstride bytes
    const uint8_t* vrows;
    const float* ksc;       // scales of the tile (LMEM), kv8
    const float* vsc;
    size_t kstride;
    uint32_t s = 0;
    if (MODE == Mode::Dxa) {
      s = j % depth;
      if (issuer && j + depth - 1 < ntiles) issue(j + depth - 1);
      stage_bar[s].arrive_and_wait();
      krows = reinterpret_cast<const uint8_t*>(lm + L.kb + s * L.data_stage);
      vrows = reinterpret_cast<const uint8_t*>(lm + L.vb + s * L.data_stage);
      ksc = lm + L.ksc + s * L.sc_stage;
      vsc = lm + L.vsc + s * L.sc_stage;
      kstride = (size_t)D * ESZ;
    } else {
      krows = kbase + seq_off + (size_t)t0 * row_stride;
      vrows = vbase + seq_off + (size_t)t0 * row_stride;
      ksc = lm + L.ksc;
      vsc = lm + L.vsc;
      kstride = row_stride;
      if (I8)   // the CTA copies the tile's scales to LMEM (thread tt reads its own K scale in S; V scales are read by every thread in V)
        for (uint32_t tt = tid; tt < nt; tt += nth) {
          lm[L.ksc + tt] = ks_g[t0 + tt];
          lm[L.vsc + tt] = vs_g[t0 + tt];
        }
    }

    // S: scores of the tile, thread per position, all G heads (K dequantized right after the load)
    for (uint32_t tt = tid; tt < nt; tt += nth) {
      const uint8_t* kr = krows + tt * kstride;
      float sc[G];
#pragma unroll
      for (uint32_t h = 0; h < G; ++h) sc[h] = 0.0f;
      if (I8) {
        const uint32_t* kw = reinterpret_cast<const uint32_t*>(kr);
        const float kscale = ksc[tt];
        for (uint32_t d4 = 0; d4 < D / 4; ++d4) {
          const uint32_t w = kw[d4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float kd = (float)byte_s32(w, k) * kscale;
#pragma unroll
            for (uint32_t h = 0; h < G; ++h) sc[h] = __builtin_fmaf(qs[h * D + 4 * d4 + k], kd, sc[h]);
          }
        }
      } else {
#if defined(__riscv_zfhmin)
        const _Float16* kh = reinterpret_cast<const _Float16*>(kr);
        for (uint32_t d = 0; d < D; ++d) {
          const float kd = (float)kh[d];
#pragma unroll
          for (uint32_t h = 0; h < G; ++h) sc[h] = __builtin_fmaf(qs[h * D + d], kd, sc[h]);
        }
#endif
      }
#pragma unroll
      for (uint32_t h = 0; h < G; ++h) ps[h * Tt + tt] = sc[h] * scale;
    }
    sync.arrive_and_wait();

    // M1: running max and rescale factor, thread per head
    for (uint32_t h = tid; h < G; h += nth) {
      const float mo = mst[h];
      float mx = mo;
      for (uint32_t tt = 0; tt < nt; ++tt) mx = __builtin_fmaxf(mx, ps[h * Tt + tt]);
      cst[h] = psk::ps_expf(mo - mx);
      mst[h] = mx;
    }
    sync.arrive_and_wait();

    // M2: p = exp(s - m'), thread per (head, position)
    for (uint32_t i = tid; i < G * nt; i += nth) {
      const uint32_t h = i / nt, tt = i - h * nt;
      ps[h * Tt + tt] = psk::ps_expf(ps[h * Tt + tt] - mst[h]);
    }
    sync.arrive_and_wait();

    // V: running sums and outputs (V dequantized right after the load)
    for (uint32_t h = tid; h < G; h += nth) {
      float sum = 0.0f;
      for (uint32_t tt = 0; tt < nt; ++tt) sum += ps[h * Tt + tt];
      lst[h] = __builtin_fmaf(lst[h], cst[h], sum);
    }
    for (uint32_t d = tid; d < D; d += nth) {
      float a[G];
#pragma unroll
      for (uint32_t h = 0; h < G; ++h) a[h] = acc[h * D + d] * cst[h];
      for (uint32_t tt = 0; tt < nt; ++tt) {
        float vd;
        if (I8) {
          vd = (float)(int32_t)reinterpret_cast<const int8_t*>(vrows)[tt * kstride + d];
          vd = vd * vsc[tt];
        } else {
#if defined(__riscv_zfhmin)
          vd = (float)reinterpret_cast<const _Float16*>(vrows + tt * kstride)[d];
#else
          vd = 0.0f;
#endif
        }
#pragma unroll
        for (uint32_t h = 0; h < G; ++h) a[h] = __builtin_fmaf(ps[h * Tt + tt], vd, a[h]);
      }
#pragma unroll
      for (uint32_t h = 0; h < G; ++h) acc[h * D + d] = a[h];
    }
    if (MODE == Mode::Dxa)
      stage_bar[s].arrive_and_wait();
    else
      sync.arrive_and_wait();
    if (tid == 0) PS_MARK(PS_MARK_DECODE_TILE);
  }
  PS_MEM_HINT(PS_CLASS_ORDINARY);

  for (uint32_t i = tid; i < G * D; i += nth) out[i] = acc[i] / lst[i / D];
}

template <Mode MODE, bool I8>
static inline __attribute__((always_inline)) void dispatch_kv(const ps_attn_kv_arg_t* arg) {
  switch (arg->a.G) {   // the host only launches G in {1, 2, 4, 8}
    case 1: attn_kv_cta<1, MODE, I8>(arg); break;
    case 2: attn_kv_cta<2, MODE, I8>(arg); break;
    case 4: attn_kv_cta<4, MODE, I8>(arg); break;
    case 8: attn_kv_cta<8, MODE, I8>(arg); break;
    default: break;
  }
}

}  // namespace

__kernel void attn_gqa_kv8(ps_attn_kv_arg_t* __UNIFORM__ arg) { dispatch_kv<Mode::Plain, true>(arg); }
__kernel void attn_dxa_kv8(ps_attn_kv_arg_t* __UNIFORM__ arg) { dispatch_kv<Mode::Dxa, true>(arg); }
#if defined(__riscv_zfhmin)
__kernel void attn_gqa_kv16(ps_attn_kv_arg_t* __UNIFORM__ arg) { dispatch_kv<Mode::Plain, false>(arg); }
__kernel void attn_dxa_kv16(ps_attn_kv_arg_t* __UNIFORM__ arg) { dispatch_kv<Mode::Dxa, false>(arg); }
#endif
