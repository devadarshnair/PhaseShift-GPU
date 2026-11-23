// main.cpp -- ps_attn_k8 (attention microbench of the int8-K-cache decode attention. ONE kernel launch per run (cold caches), one layer:
//   ps_attn_k8 -T <ctx> [-H 6] [-K 6] [-D 48] [-t Tt=32] [-s seed] [-q qscale] [-v fp32|k8dot|k8sw] [-e tol] [-k kernel.vxbin]
//   -v fp32   the stock attn_gqa of kernels/decode_attn_gqa (fp32 K / V cache), the denominator of the attention-time ratio
//   -v k8dot  attn_k8_dot: int8 K cache, q quantized in the kernel, QK^T with DOT4ACC (needs a dot4 tree: psdot / psall / psall2), V fp32
//   -v k8sw   attn_k8_sw : the same kernel with scalar int8 multiplies (equal-effort software baseline; bit-identical output to k8dot)
// Shapes default to stories15M (6 heads, 6 KV heads, head size 48, G = kv_mul = 1); the cache holds exactly T positions (Tmax = T), T = 128 .. 1024+ is free here
// (the model, not the kernel, limits T; stories15M's seq_len is 256). Synthetic data: q, K, V uniform in [-1, 1) from ps::Rng(seed) (q multiplied by -q).
// Int8 arms: the host quantizes K rows 0..T-2 per row (scheme, common.h); row T-1 is quantized by the kernel's prologue from the fp32 K row (as in the
// token app), so its cost is inside the measured launch. The reference is the float64 attention over the DEQUANTIZED operands the kernel uses (device operand ==
// host operand: same float32 divisions / roundf), error metric rel_i = |o_i - ref_i| / S_i with S_i = sum_t p_t |v_t[d]|, tolerance -e (default 3e-4: the
// attn_gqa bound for these sizes is ~1e-5, the extra slack covers the float32 score product (float)isum * (sQ sK) * scale, ~3 roundings of |s| <= 10). PSKVQ prints the
// error of the int8 K / q path against the float64 attention over the ORIGINAL fp32 data (information, not pass / fail). PSDIGEST prints the output digest, so k8dot and k8sw
// can be compared for bit equality.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <vortex.h>

#include "common.h"
#include "ps_common.h"

namespace {

struct Ctx {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  std::vector<vx_buffer_h> bufs;
  ~Ctx() {
    if (!device) return;
    for (auto b : bufs) vx_buffer_release(b);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
  int buffer(uint64_t size, int flags, uint64_t* addr, vx_buffer_h* h) {
    vx_buffer_h b = nullptr;
    int r = vx_buffer_create(device, size, flags, &b);
    if (r) return r;
    bufs.push_back(b);
    if (h) *h = b;
    return vx_buffer_address(b, addr);
  }
};
#define PS_RT_CHECK(_expr) do { int _ret = (int)(_expr); if (_ret != 0) { std::printf("Error: '%s' returned %d!\nFAILED!\n", #_expr, _ret); return 1; } } while (false)

// float64 attention: scores s[t] (already scaled) -> softmax -> V; out[h][d], S[h][d]
void softmax_v(const std::vector<double>& s, const std::vector<float>& V, uint32_t T, uint32_t H_kv, uint32_t D, uint32_t kvh, double* out, double* sc) {
  double m = -1e300;
  for (uint32_t t = 0; t < T; ++t) m = std::max(m, s[t]);
  std::vector<double> p(T);
  double l = 0;
  for (uint32_t t = 0; t < T; ++t) { p[t] = std::exp(s[t] - m); l += p[t]; }
  for (uint32_t d = 0; d < D; ++d) {
    double o = 0, a = 0;
    for (uint32_t t = 0; t < T; ++t) {
      const double v = V[((size_t)t * H_kv + kvh) * D + d];
      o += p[t] * v; a += p[t] * std::fabs(v);
    }
    out[d] = o / l; sc[d] = std::max(a / l, 1e-30);
  }
}
}  // namespace

int main(int argc, char** argv) {
  std::string kernel_file = "kernel.vxbin", variant = "fp32";
  uint32_t T = 128, H_q = 6, H_kv = 6, D = 48, Tt = 32, seed = 1;
  double qscale = 1.0, tol = 3e-4;
  int c;
  while ((c = getopt(argc, argv, "T:H:K:D:t:s:q:v:e:k:h")) != -1) {
    switch (c) {
      case 'T': T = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'H': H_q = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'K': H_kv = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'D': D = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 't': Tt = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'q': qscale = std::strtod(optarg, nullptr); break;
      case 'v': variant = optarg; break;
      case 'e': tol = std::strtod(optarg, nullptr); break;
      case 'k': kernel_file = optarg; break;
      default: std::printf("Usage: %s -T ctx [-H hq] [-K hkv] [-D d] [-t Tt] [-s seed] [-q qscale] [-v fp32|k8dot|k8sw] [-e tol] [-k kernel.vxbin]\n", argv[0]); return 1;
    }
  }
  const bool i8 = variant != "fp32";
  if (variant != "fp32" && variant != "k8dot" && variant != "k8sw") { std::printf("bad -v %s\nFAILED!\n", variant.c_str()); return 1; }
  if (D % 4 || H_q % H_kv || T < 1 || Tt < 1) { std::printf("need D %% 4 == 0, H_q %% H_kv == 0, T >= 1\nFAILED!\n"); return 1; }
  const uint32_t kv_mul = H_q / H_kv, G = kv_mul, B = 1, Tmax = T;
  if (G != 1 && G != 2 && G != 4 && G != 8) { std::printf("G must be 1, 2, 4 or 8\nFAILED!\n"); return 1; }
  const float scale = 1.0f / std::sqrt((float)D);

  ps::Rng rng(seed);
  std::vector<float> q((size_t)H_q * D), K((size_t)T * H_kv * D), V((size_t)T * H_kv * D);
  for (auto& x : q) x = rng.uniform() * (float)qscale;
  for (auto& x : K) x = rng.uniform();
  for (auto& x : V) x = rng.uniform();

  // quantization (common.h): K per row (T x H_kv rows), q per head
  auto qscale_of = [&](const float* x, uint32_t n) { float mx = 0.0f; for (uint32_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(x[i])); return mx / 127.0f; };
  auto quant = [&](float x, float s) -> int32_t { return s != 0.0f ? (int32_t)std::roundf(x / s) : 0; };
  std::vector<int8_t> Kq((size_t)T * H_kv * D, (int8_t)0x55);
  std::vector<float> Ks((size_t)H_kv * Tmax, std::nanf(""));
  std::vector<int32_t> qq((size_t)H_q * D);
  std::vector<float> qs(H_q);
  for (uint32_t h = 0; h < H_q; ++h) { qs[h] = qscale_of(&q[(size_t)h * D], D); for (uint32_t d = 0; d < D; ++d) qq[(size_t)h * D + d] = quant(q[(size_t)h * D + d], qs[h]); }
  const uint32_t qp = T - 1;
  for (uint32_t t = 0; t < T; ++t)
    for (uint32_t kh = 0; kh < H_kv; ++kh) {
      const float* row = &K[((size_t)t * H_kv + kh) * D];
      const float s = qscale_of(row, D);
      if (t == qp) continue;   // the kernel quantizes the newest row itself (scale and bytes stay poisoned)
      Ks[(size_t)kh * Tmax + t] = s;
      for (uint32_t d = 0; d < D; ++d) Kq[((size_t)t * H_kv + kh) * D + d] = (int8_t)quant(row[d], s);
    }

  // float64 references: over the original fp32 data (ref0) and over the dequantized int8 operands (ref)
  const size_t NO = (size_t)H_q * D;
  std::vector<double> ref0(NO), sc0(NO), ref(NO), sc(NO);
  for (uint32_t h = 0; h < H_q; ++h) {
    const uint32_t kh = h / kv_mul;
    std::vector<double> s0(T), s1(T);
    for (uint32_t t = 0; t < T; ++t) {
      double a = 0; int64_t ia = 0;
      const float* row = &K[((size_t)t * H_kv + kh) * D];
      for (uint32_t d = 0; d < D; ++d) a += (double)q[(size_t)h * D + d] * row[d];
      s0[t] = a * scale;
      const float ksc = qscale_of(row, D);
      for (uint32_t d = 0; d < D; ++d) ia += (int64_t)qq[(size_t)h * D + d] * quant(row[d], ksc);
      s1[t] = (double)ia * ((double)qs[h] * (double)ksc) * scale;
    }
    softmax_v(s0, V, T, H_kv, D, kh, &ref0[(size_t)h * D], &sc0[(size_t)h * D]);
    softmax_v(i8 ? s1 : s0, V, T, H_kv, D, kh, &ref[(size_t)h * D], &sc[(size_t)h * D]);
  }

  Ctx ctx;
  PS_RT_CHECK(vx_device_open(0, &ctx.device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  PS_RT_CHECK(vx_queue_create(ctx.device, &qi, &ctx.queue));
  uint64_t nt = 0, nw = 0, lmem_cap = 0;
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_WARPS, &nw));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  const uint32_t nth = (uint32_t)(nt * nw);
  const uint32_t lmem_words = i8 ? ps_attn_k8_lmem(G, D, Tt).total : ps_attn_lmem(G, D, Tt, 1, 0).total;
  const uint32_t lmem_bytes = lmem_words * 4;
  if (lmem_bytes > lmem_cap) { std::printf("LMEM footprint %u B > %llu B\nFAILED!\n", lmem_bytes, (unsigned long long)lmem_cap); return 1; }
  std::printf("ps_attn_k8: variant=%s B=%u H_q=%u H_kv=%u D=%u T=%u Tmax=%u G=%u Tt=%u ntiles=%u NT=%u NW=%u lmem=%u/%llu tol=%.3e seed=%u qscale=%g\n", variant.c_str(), B, H_q, H_kv, D, T,
              Tmax, G, Tt, (T + Tt - 1) / Tt, (unsigned)nt, (unsigned)nw, lmem_bytes, (unsigned long long)lmem_cap, tol, seed, qscale);

  ps_attn_k8_arg_t karg = {};
  ps_attn_arg_t& arg = karg.a;
  vx_buffer_h hq, hk, hv, ho, hks = nullptr, hkf = nullptr;
  const uint64_t qb = q.size() * 4, kb = K.size() * 4, ob = NO * 4;
  PS_RT_CHECK(ctx.buffer(qb, VX_MEM_READ, &arg.q_addr, &hq));
  PS_RT_CHECK(ctx.buffer(kb, VX_MEM_READ, &arg.v_addr, &hv));
  PS_RT_CHECK(ctx.buffer(ob, VX_MEM_READ_WRITE, &arg.o_addr, &ho));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, hq, 0, q.data(), qb, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, hv, 0, V.data(), kb, 0, nullptr, nullptr));
  if (!i8) {
    PS_RT_CHECK(ctx.buffer(kb, VX_MEM_READ, &arg.k_addr, &hk));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hk, 0, K.data(), kb, 0, nullptr, nullptr));
  } else {
    PS_RT_CHECK(ctx.buffer(Kq.size(), VX_MEM_READ_WRITE, &arg.k_addr, &hk));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hk, 0, Kq.data(), Kq.size(), 0, nullptr, nullptr));
    PS_RT_CHECK(ctx.buffer(Ks.size() * 4, VX_MEM_READ_WRITE, &karg.ks_addr, &hks));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hks, 0, Ks.data(), Ks.size() * 4, 0, nullptr, nullptr));
    PS_RT_CHECK(ctx.buffer(kb, VX_MEM_READ, &karg.kf_addr, &hkf));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hkf, 0, K.data(), kb, 0, nullptr, nullptr));
    karg.quant_pos = qp;
  }
  arg.B = B; arg.H_q = H_q; arg.H_kv = H_kv; arg.D = D; arg.T = T; arg.Tmax = Tmax; arg.kv_mul = kv_mul; arg.G = G; arg.Tt = Tt; arg.depth = 1; arg.scale = scale;
  PS_RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
  const char* kname = variant == "fp32" ? "attn_gqa" : variant == "k8dot" ? "attn_k8_dot" : "attn_k8_sw";
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, kname, &ctx.kernel));
  PS_RT_CHECK(vx_check_occupancy(ctx.device, nth, lmem_bytes));
  const uint32_t grid = B * (H_q / G);
  std::printf("launch: kernel=%s grid=%u block=%u lmem_bytes=%u\n", kname, grid, nth, lmem_bytes);

  std::vector<float> O(NO);
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = ctx.kernel;
  li.args_host = i8 ? (const void*)&karg : (const void*)&arg;
  li.args_size = i8 ? sizeof(karg) : sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = grid;
  li.block_dim[0] = nth;
  li.lmem_size = lmem_bytes;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  PS_RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &launch_ev));
  PS_RT_CHECK(vx_enqueue_read(ctx.queue, O.data(), ho, 0, ob, 1, &launch_ev, &read_ev));
  PS_RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  std::printf("PSDIGEST: out n=%zu fnv1a64=%016llx\n", NO, (unsigned long long)ps::fnv1a64_f32(O.data(), NO));
  auto s64 = ps::compare(O.data(), ref.data(), sc.data(), NO, tol);
  ps::print_check("launch0:vs_f64", s64);
  bool ok = s64.ok();
  auto sq = ps::compare(O.data(), ref0.data(), sc0.data(), NO, 1.0);
  std::printf("PSKVQ: variant=%s error_vs_fp32_attention max_rel=%.6e max_abs=%.6e (information; scale S_i = sum_t p_t |v_t[d]|)\n", variant.c_str(), sq.max_rel, sq.max_abs);
  // bytes of K the kernel streams per step (all heads): fp32 = 4 T H_kv D; int8 = T H_kv D + 4 T H_kv (scales)
  std::printf("PSKBYTES: variant=%s k_bytes=%llu\n", variant.c_str(), (unsigned long long)(i8 ? (uint64_t)T * H_kv * (D + 4) : (uint64_t)T * H_kv * D * 4));
  if (!ok) { std::printf("FAILED!\n"); return 1; }
  std::printf("PASSED!\n");
  return 0;
}
