// kernel.cpp -- ps_prefill_attn device kernels: tiled causal (prefill) attention on the FPU with LMEM
// tiles: the tiled online-softmax forward pass of FlashAttention [cand:A7] (QK^T -> online softmax -> PV fused; the
// S x S score matrix never exists), float32, written from the algorithm (methodology §14.2 recurrence); no code is
// copied from any implementation. Design choices are documented in docs/methodology.md §14.4.
//
// Work decomposition: one CTA (all NT*NW threads of one core) per (sequence b, KV head, query tile of Qt tokens).
// The CTA serves the G = H_q / H_kv query heads that share the KV head, so its R = G * Qt rows (row r = token i,
// head h, r = i * G + h) all read the SAME key/value tiles: each K/V element brought into LMEM is used R times
// (the decode kernel reuses it G times). Per key tile of Tk positions:
//   load  K and V tile -> LMEM:
//         lmem_tiled (`attn_prefill_lmem`): every thread copies elements (coalesced global loads), then a barrier;
//         dxa_dbuf   (`attn_prefill_dxa`):  the DXA engine copies tile j + depth - 1 while tile j is computed
//                    (`depth` LMEM stages, completion by expect_tx on the stage barrier, one issuer warp; the
//                    gqa_dxa_dbuf pattern). depth 1 = copy, then compute. Needs VX_CFG_EXT_DXA_ENABLE.
//   S     scores: item = (4-row block, key t); 4 dot products over D (q rows and the k row from LMEM, 4 FMAs per
//         k load); causal mask: key position t0 + t > row position -> -inf
//   M1    thread per row: running max m' = max(m, tile max), rescale c = exp(m - m')
//   M2    thread per (row, key): p = exp(s - m'), masked -> exactly 0
//   V     thread per row: l = l c + sum_t p;  item = (4-row block, 4 columns): acc = acc c + P V as a 4 x 4
//         register block (4 p loads + 4 v loads per 16 FMAs)
// Online softmax as / methodology §14.2: m starts at -inf (errata 7 /), first-tile c = exp(-inf) = 0.
// Causal tile skipping: a CTA visits key tiles 0 .. ceil(kend / Tk) - 1 only, kend = 1 + (last query position of
// the tile), so tiles entirely above the diagonal are never loaded nor computed (~half of the S x S tiles at s0 = 0,
// E4's 2 S^2 d); tiles crossing the diagonal are masked per element. Keys >= kend are never read (the DXA
// descriptor's position extent is s0 + S, rows beyond it are zero-filled and never used).
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#include "common.h"
// FMA codegen (measured): the regression build (tests/regression/common.mk VX_CFLAGS) leaves clang's default
// -fmath-errno on for riscv32-unknown-elf, so `__builtin_fmaf` is lowered to a CALL to musl's fmaf (`fmadd.s; ret`)
// with argument moves around it; `__builtin_elementwise_fma` (llvm.fma, which has no errno) is emitted inline as
// fmadd.s. Both are the same IEEE-754 fused multiply-add (one rounding), so results are bit-identical; only the
// instruction count changes. This app maps every __builtin_fmaf, including those of the shared ps_expf in
// warp_ops.h (unchanged), to the inline form. Evidence and the effect on:.
// Since every generated app Makefile adds -fno-math-errno (scripts/install_kernel_apps.sh, errata 30), which
// lowers __builtin_fmaf to the same inline fmadd.s; this macro is then redundant (identical assembly) and is
// kept so that this kernel stays libcall-free even if built with the stock regression flags.
#define __builtin_fmaf(a, b, c) __builtin_elementwise_fma((float)(a), (float)(b), (float)(c))
#include "warp_ops.h"

namespace {

enum class Mode { Lmem, Dxa };

template <Mode MODE>
static inline __attribute__((always_inline)) void prefill_cta(const ps_pattn_arg_t* arg) {
  const uint32_t D = arg->D, S = arg->S, s0 = arg->s0, G = arg->G, Qt = arg->Qt, Tk = arg->Tk, H_kv = arg->H_kv;
  const uint32_t H_q = arg->H_q, nqt = arg->nqt;
  const uint32_t cta = blockIdx.x;
  const uint32_t qt = cta % nqt, bk = cta / nqt;              // query tile innermost: one head's tiles go round-robin
  const uint32_t kvh = bk % H_kv, b = bk / H_kv;              // over the cores
  const uint32_t q0 = qt * Qt;
  const uint32_t nq = (S - q0 < Qt) ? S - q0 : Qt;            // valid query tokens of this tile
  const uint32_t pos0 = s0 + q0;                              // absolute position of the tile's first token
  const uint32_t kend = pos0 + nq;                            // keys 0 .. kend-1 are visible to some row
  const uint32_t ntiles = (kend + Tk - 1) / Tk;
  const uint32_t tid = threadIdx.x, nth = blockDim.x;
  const float scale = arg->scale;
  const uint32_t h0 = kvh * G;
  const size_t row_stride = (size_t)H_kv * D;                                   // floats between key positions
  const size_t seq_off = ((size_t)b * arg->Tmax * H_kv + kvh) * D;              // K[b][0][kvh][0]
  const float* kbase = reinterpret_cast<const float*>(arg->k_addr);
  const float* vbase = reinterpret_cast<const float*>(arg->v_addr);
  const float* qg = reinterpret_cast<const float*>(arg->q_addr);
  float* og = reinterpret_cast<float*>(arg->o_addr);

  const ps_pattn_lmem_t L = ps_pattn_lmem(G, D, Qt, Tk, MODE == Mode::Dxa ? arg->depth : 1);
  const uint32_t Rp = L.rows, nrb = Rp / 4, ncb = D / 4, Rv = G * nq;   // padded rows, row/col blocks, valid rows
  float* lm = reinterpret_cast<float*>(__local_mem());
  float* qs = lm + L.qs;
  float* acc = lm + L.acc;
  float* ps = lm + L.ps;
  float* mst = lm + L.stat;
  float* lst = mst + Rp;
  float* cst = lst + Rp;

  vortex::barrier sync(5);   // phase barrier (ids 1..4 are the DXA stage barriers; id 0 is __syncthreads' slot)

  // Q tile -> LMEM (rows >= Rv: q = 0), acc = 0, m = -inf, l = 0
  for (uint32_t i = tid; i < Rp * D; i += nth) {
    const uint32_t r = i / D, d = i - r * D;
    float v = 0.0f;
    if (r < Rv) {
      const uint32_t tok = r / G, h = r - tok * G;
      v = qg[(((size_t)b * S + q0 + tok) * H_q + h0 + h) * D + d];
    }
    qs[i] = v;
    acc[i] = 0.0f;
  }
  for (uint32_t r = tid; r < Rp; r += nth) { mst[r] = psk::neg_inf(); lst[r] = 0.0f; }
  sync.arrive_and_wait();

  const uint32_t depth = arg->depth;
  vortex::barrier stage_bar[4] = {vortex::barrier(1), vortex::barrier(2), vortex::barrier(3), vortex::barrier(4)};
  const bool issuer = (MODE == Mode::Dxa) && (get_sub_group_id() == 0);
  const uint32_t stage_floats = ps_pattn_round16(Tk * D);
  auto issue = [&](uint32_t j) {   // DXA K and V tile j -> stage j % depth (the whole issuer warp executes it)
    const uint32_t s = j % depth;
    stage_bar[s].expect_tx(2);
    vx_dxa_issue_4d_wg(PS_PATTN_DESC_K, stage_bar[s].id(), lm + L.ks + s * stage_floats, 0, kvh, j * Tk, b);
    vx_dxa_issue_4d_wg(PS_PATTN_DESC_V, stage_bar[s].id(), lm + L.vs + s * stage_floats, 0, kvh, j * Tk, b);
  };
  if (MODE == Mode::Dxa && issuer)
    for (uint32_t j = 0; j + 1 < depth && j < ntiles; ++j) issue(j);

  for (uint32_t j = 0; j < ntiles; ++j) {
    const uint32_t t0 = j * Tk;
    const uint32_t nt = (kend - t0 < Tk) ? kend - t0 : Tk;   // keys of this tile that any row can see
    float* ks;
    float* vs;
    uint32_t s = 0;
    if (MODE == Mode::Dxa) {
      s = j % depth;
      if (issuer && j + depth - 1 < ntiles) issue(j + depth - 1);
      stage_bar[s].arrive_and_wait();            // tile j landed in stage s (tx events) and every warp arrived
      ks = lm + L.ks + s * stage_floats;
      vs = lm + L.vs + s * stage_floats;
    } else {
      ks = lm + L.ks;
      vs = lm + L.vs;
      // cooperative copy of rows t0 .. t0+nt-1 (element i = tt * D + d; consecutive threads take consecutive d)
      uint32_t tt = tid / D, d = tid - tt * D;
      for (uint32_t i = tid; i < nt * D; i += nth) {
        const size_t g = seq_off + (size_t)(t0 + tt) * row_stride + d;
        ks[i] = kbase[g];
        vs[i] = vbase[g];
        d += nth;
        while (d >= D) { d -= D; ++tt; }
      }
      sync.arrive_and_wait();
    }

    // S: scores, item = (4-row block rb, key t); masked entries -> -inf
    for (uint32_t it = tid; it < nrb * nt; it += nth) {
      const uint32_t rb = it / nt, t = it - rb * nt;
      const float* qr = qs + rb * 4 * D;
      const float* kr = ks + t * D;
      float s0v = 0.0f, s1v = 0.0f, s2v = 0.0f, s3v = 0.0f;
      for (uint32_t d = 0; d < D; ++d) {
        const float kd = kr[d];
        s0v = __builtin_fmaf(qr[d], kd, s0v);
        s1v = __builtin_fmaf(qr[D + d], kd, s1v);
        s2v = __builtin_fmaf(qr[2 * D + d], kd, s2v);
        s3v = __builtin_fmaf(qr[3 * D + d], kd, s3v);
      }
      const float sv[4] = {s0v, s1v, s2v, s3v};
      const uint32_t kp = t0 + t;
#pragma unroll
      for (uint32_t u = 0; u < 4; ++u) {
        const uint32_t r = rb * 4 + u;
        uint32_t tok = r / G;
        if (tok >= nq) tok = nq - 1;            // padding rows take the last valid token's position
        ps[r * Tk + t] = (kp <= pos0 + tok) ? sv[u] * scale : psk::neg_inf();
      }
    }
    sync.arrive_and_wait();

    // M1: running max and rescale factor, thread per row
    for (uint32_t r = tid; r < Rp; r += nth) {
      const float mo = mst[r];
      float mx = mo;
      for (uint32_t t = 0; t < nt; ++t) mx = __builtin_fmaxf(mx, ps[r * Tk + t]);
      cst[r] = psk::ps_expf(mo - mx);           // mo = -inf on the first tile -> 0; no visible key -> exp(0) = 1
      mst[r] = mx;
    }
    sync.arrive_and_wait();

    // M2: p = exp(s - m'), masked -> 0, thread per (row, key)
    for (uint32_t it = tid; it < Rp * nt; it += nth) {
      const uint32_t r = it / nt, t = it - r * nt;
      const float sv = ps[r * Tk + t];
      ps[r * Tk + t] = (sv == psk::neg_inf()) ? 0.0f : psk::ps_expf(sv - mst[r]);
    }
    sync.arrive_and_wait();

    // V: running sums (thread per row) and outputs (4 x 4 register blocks)
    for (uint32_t r = tid; r < Rp; r += nth) {
      float sum = 0.0f;
      for (uint32_t t = 0; t < nt; ++t) sum += ps[r * Tk + t];
      lst[r] = __builtin_fmaf(lst[r], cst[r], sum);
    }
    for (uint32_t it = tid; it < nrb * ncb; it += nth) {
      const uint32_t rb = it / ncb, cb = it - rb * ncb;
      const uint32_t r0 = rb * 4, d0 = cb * 4;
      float a[4][4];
#pragma unroll
      for (uint32_t u = 0; u < 4; ++u) {
        const float c = cst[r0 + u];
#pragma unroll
        for (uint32_t w = 0; w < 4; ++w) a[u][w] = acc[(r0 + u) * D + d0 + w] * c;
      }
      for (uint32_t t = 0; t < nt; ++t) {
        float p[4], v[4];
#pragma unroll
        for (uint32_t u = 0; u < 4; ++u) p[u] = ps[(r0 + u) * Tk + t];
#pragma unroll
        for (uint32_t w = 0; w < 4; ++w) v[w] = vs[t * D + d0 + w];
#pragma unroll
        for (uint32_t u = 0; u < 4; ++u)
#pragma unroll
          for (uint32_t w = 0; w < 4; ++w) a[u][w] = __builtin_fmaf(p[u], v[w], a[u][w]);
      }
#pragma unroll
      for (uint32_t u = 0; u < 4; ++u)
#pragma unroll
        for (uint32_t w = 0; w < 4; ++w) acc[(r0 + u) * D + d0 + w] = a[u][w];
    }
    if (MODE == Mode::Dxa)
      stage_bar[s].arrive_and_wait();            // every warp is done with stage s and with ps
    else
      sync.arrive_and_wait();                    // Ks/Vs and ps are rewritten by the next tile
  }

  // out = acc / l for the valid rows
  for (uint32_t i = tid; i < Rv * D; i += nth) {
    const uint32_t r = i / D, d = i - r * D;
    const uint32_t tok = r / G, h = r - tok * G;
    og[(((size_t)b * S + q0 + tok) * H_q + h0 + h) * D + d] = acc[i] / lst[r];
  }
}

}  // namespace

__kernel void attn_prefill_lmem(ps_pattn_arg_t* __UNIFORM__ arg) { prefill_cta<Mode::Lmem>(arg); }
__kernel void attn_prefill_dxa(ps_pattn_arg_t* __UNIFORM__ arg) { prefill_cta<Mode::Dxa>(arg); }
