// main.cpp -- ps_gemv_q80dot4w: host of the warp-cooperative dot4 Q8_0 GEMV (kernel.cpp). New app;
// kernels/ps_gemv_q80dot4 and ps_gemv_q80tcu untouched (this file is a copy of the TCU host with the operand layout replaced).
// Usage: ps_gemv_q80dot4w -m M -n N [-s seed] [-G GS] [-B B (1..4)] -v w<S>u<U> [-L launches] [-t tol] [-g grid] [-k kernel.vxbin]
// Inputs and references are the SAME as ps_gemv_q80dot4 (ps::Rng(seed) draws W (M x N) then the B activation vectors; Q8_0 quantizer and
// runq.c-order reference of kernels/common/ps_q80.h), so a (m, n, G, B, seed) case is the identical problem as the dot4 / software / TCU rows.
// Host-side preparation NOT in the PERF cycles: the WEIGHT re-layout of common.h (a one-time model-load cost, unlike the TCU's per-token
// activation layout) with M padded to a multiple of R = 32 / S (zero rows, executed and counted in dot4wn_padding:). Activations unchanged.
// Checks per launch: vs_f64 (2 gamma_{G+2} rule) and PSBITEXACT vs the runq.c order (REQUIRED to be 0 mismatches).
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <vortex2.h>
#include "common.h"
#include "ps_common.h"
#include "ps_q80.h"

namespace {
struct Ctx {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h k = nullptr;
  std::vector<vx_buffer_h> bufs;
  ~Ctx() {
    if (!device) return;
    for (auto b : bufs) vx_buffer_release(b);
    if (k) vx_kernel_release(k);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
  int buffer(uint64_t size, int flags, uint64_t* addr, vx_buffer_h* h = nullptr) {
    vx_buffer_h b = nullptr;
    int r = vx_buffer_create(device, size, flags, &b);
    if (r) return r;
    bufs.push_back(b);
    if (h) *h = b;
    return vx_buffer_address(b, addr);
  }
};
#define PS_RT_CHECK(_expr) do { int _ret = (int)(_expr); if (_ret != 0) { std::printf("Error: '%s' returned %d!\nFAILED!\n", #_expr, _ret); return 1; } } while (false)
}  // namespace

int main(int argc, char** argv) {
  std::string kernel_file = "kernel.vxbin", variant = "w1u1";
  uint32_t M = 64, N = 64, seed = 1, GS = 0, B = 1, launches = 1, grid_override = 0;
  double tol = -1.0;
  int c;
  while ((c = getopt(argc, argv, "m:n:s:G:B:v:L:t:g:k:h")) != -1) {
    switch (c) {
      case 'm': M = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': N = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'G': GS = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'B': B = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'v': variant = optarg; break;
      case 'L': launches = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 't': tol = std::strtod(optarg, nullptr); break;
      case 'g': grid_override = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      default: std::printf("Usage: %s -m M -n N [-s seed] [-G 32|64] [-B 1..4] [-v w<S>u<U>] [-L launches]\n", argv[0]); return 1;
    }
  }
  static const char* kV[] = {"w1u1", "w1u2", "w2u1", "w2u2", "w2u4", "w4u2", "w4u4"};
  bool vok = false;
  for (auto v : kV) vok = vok || variant == v;
  if (!vok) { std::printf("bad -v %s\nFAILED!\n", variant.c_str()); return 1; }
  Ctx ctx;
  PS_RT_CHECK(vx_device_open(0, &ctx.device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  PS_RT_CHECK(vx_queue_create(ctx.device, &qi, &ctx.queue));
  uint64_t nt = 0, isa = 0;
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_ISA_FLAGS, &isa));
  if (nt < 8 || nt > 32 || (nt & (nt - 1))) { std::printf("kernel needs NT in {8,16,32}\nFAILED!\n"); return 1; }
  const uint32_t S = (uint32_t)(variant[1] - '0'), R = (uint32_t)nt / S;
  if (launches < 1) { std::printf("-L must be >= 1\nFAILED!\n"); return 1; }
  if (B < 1 || B > 4) { std::printf("-B must be in 1..4\nFAILED!\n"); return 1; }
  if (GS == 0) { GS = 64; while (GS > 4 && N % GS) GS >>= 1; }
  if (GS != 32 && GS != 64) { std::printf("kernel supports GS = 32 or 64 (got %u)\nFAILED!\n", GS); return 1; }
  if (N % GS || N % 4 || M == 0) { std::printf("need N %% GS == 0 (N=%u GS=%u)\nFAILED!\n", N, GS); return 1; }
  const uint32_t G = N / GS;
  if (tol < 0) { const double m = G + 2, u = std::ldexp(1.0, -24); tol = std::max(1e-4, 2.0 * m * u / (1.0 - m * u)); }

  ps::Rng rng(seed);
  std::vector<float> Wf((size_t)M * N), xf((size_t)B * N);
  for (auto& v : Wf) v = rng.uniform();
  for (auto& v : xf) v = rng.uniform();
  std::vector<int8_t> Wq((size_t)M * N), xq((size_t)B * N);
  std::vector<float> Ws((size_t)M * G), xs((size_t)B * G);
  ps::quantize_q80(Wf.data(), Wf.size(), GS, Wq.data(), Ws.data());
  for (uint32_t b = 0; b < B; ++b) ps::quantize_q80(xf.data() + (size_t)b * N, N, GS, xq.data() + (size_t)b * N, xs.data() + (size_t)b * G);
  std::printf("PSQ80: Wq n=%zu gs=%u fnv1a64=%016llx\n", Wq.size(), GS, (unsigned long long)ps::fnv1a64_bytes(Wq.data(), Wq.size()));
  std::printf("PSQ80: Ws n=%zu gs=%u fnv1a64=%016llx\n", Ws.size(), GS, (unsigned long long)ps::fnv1a64_bytes(Ws.data(), Ws.size() * 4));
  std::printf("PSQ80: xq n=%zu gs=%u fnv1a64=%016llx\n", xq.size(), GS, (unsigned long long)ps::fnv1a64_bytes(xq.data(), xq.size()));
  std::printf("PSQ80: xs n=%zu gs=%u fnv1a64=%016llx\n", xs.size(), GS, (unsigned long long)ps::fnv1a64_bytes(xs.data(), xs.size() * 4));

  std::vector<float> ref32((size_t)B * M);
  std::vector<double> ref64((size_t)B * M), scale((size_t)B * M);
  for (uint32_t b = 0; b < B; ++b)
    for (uint32_t i = 0; i < M; ++i)
      ref32[(size_t)b * M + i] = ps::matmul_row_q80(Wq.data() + (size_t)i * N, Ws.data() + (size_t)i * G, xq.data() + (size_t)b * N,
                                                    xs.data() + (size_t)b * G, N, GS, &ref64[(size_t)b * M + i], &scale[(size_t)b * M + i]);

  // ---- weight re-layout (common.h), M padded to Mp = round_up(M, R)
  const uint32_t GW = GS / 4, WL = GW / S, Mp = (M + R - 1) / R * R, T = Mp / R;
  std::vector<uint32_t> Wl((size_t)T * G * WL * nt, 0u);
  std::vector<float> Wsl((size_t)T * G * R, 0.0f);
  for (uint32_t t = 0; t < T; ++t)
    for (uint32_t g = 0; g < G; ++g)
      for (uint32_t r = 0; r < R; ++r) {
        const uint32_t row = t * R + r;
        if (row >= M) continue;
        Wsl[((size_t)t * G + g) * R + r] = Ws[(size_t)row * G + g];
        for (uint32_t i = 0; i < WL; ++i)
          for (uint32_t s = 0; s < S; ++s)
            std::memcpy(&Wl[(((size_t)t * G + g) * WL + i) * nt + s * R + r], &Wq[(size_t)row * N + (size_t)g * GS + (size_t)(i * S + s) * 4], 4);
      }
  std::printf("dot4wn_padding: M=%u->%u (S=%u lanes/row, R=%u rows/warp tile) executed_macs/useful_macs=%.3f\n", M, Mp, S, R, (double)Mp / M);

  std::printf("ps_gemv_q80dot4wn: synthetic:s%u M=%u N=%u GS=%u G=%u B=%u variant=%s launches=%u tol=%.3e\n", seed, M, N, GS, G, B, variant.c_str(), launches, tol);

  ps_gemv_q80dot4wn_arg_t arg = {};
  arg.M = M; arg.N = N; arg.GS = GS; arg.B = B; arg.G = G;
  vx_buffer_h hW, hWs, hxq, hxs, hy;
  const uint64_t y_size = (uint64_t)B * M * 4;
  PS_RT_CHECK(ctx.buffer(Wl.size() * 4, VX_MEM_READ, &arg.Wl_addr, &hW));
  PS_RT_CHECK(ctx.buffer(Wsl.size() * 4, VX_MEM_READ, &arg.Wsl_addr, &hWs));
  PS_RT_CHECK(ctx.buffer(xq.size(), VX_MEM_READ, &arg.xq_addr, &hxq));
  PS_RT_CHECK(ctx.buffer(xs.size() * 4, VX_MEM_READ, &arg.xs_addr, &hxs));
  PS_RT_CHECK(ctx.buffer(y_size, VX_MEM_READ_WRITE, &arg.y_addr, &hy));
  if (arg.Wl_addr % 4 || arg.xq_addr % 4 || arg.Wsl_addr % 4) { std::printf("buffers not 4-byte aligned\nFAILED!\n"); return 1; }
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, hW, 0, Wl.data(), Wl.size() * 4, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, hWs, 0, Wsl.data(), Wsl.size() * 4, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, hxq, 0, xq.data(), xq.size(), 0, nullptr, nullptr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, hxs, 0, xs.data(), xs.size() * 4, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, ("gemv_q80_" + variant).c_str(), &ctx.k));
  const uint32_t grid = grid_override ? grid_override : T;
  std::printf("launch: gemv grid=%u block=%u (one warp per R weight rows) NT=%u\n", grid, (unsigned)nt, (unsigned)nt);

  bool ok = true;
  std::vector<float> y((size_t)B * M);
  const std::vector<float> poison((size_t)B * M, std::nanf(""));
  ps::Timer timer;
  for (uint32_t l = 0; l < launches; ++l) {
    const std::string pre = "launch" + std::to_string(l);
    if (l > 0) PS_RT_CHECK(vx_enqueue_write(ctx.queue, hy, 0, poison.data(), y_size, 0, nullptr, nullptr));
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = ctx.k;
    li.args_host = &arg;
    li.args_size = sizeof(arg);
    li.ndim = 1;
    li.grid_dim[0] = grid;
    li.block_dim[0] = (uint32_t)nt;
    vx_event_h ev = nullptr, rd = nullptr;
    PS_RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &ev));
    PS_RT_CHECK(vx_enqueue_read(ctx.queue, y.data(), hy, 0, y_size, 1, &ev, &rd));
    PS_RT_CHECK(vx_event_wait_value(rd, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(rd);
    vx_event_release(ev);
    std::printf("PSDIGEST: %s n=%zu fnv1a64=%016llx\n", pre.c_str(), y.size(), (unsigned long long)ps::fnv1a64_f32(y.data(), y.size()));
    auto s64 = ps::compare(y.data(), ref64.data(), scale.data(), y.size(), tol);
    ps::print_check((pre + ":vs_f64").c_str(), s64);
    ok = ok && s64.ok();
    size_t mism = 0;
    for (size_t i = 0; i < y.size(); ++i) mism += std::memcmp(&y[i], &ref32[i], 4) != 0;
    std::printf("PSBITEXACT: %s:runq_exact n=%zu mismatches=%zu variant=%s\n", pre.c_str(), y.size(), mism, variant.c_str());
    ok = ok && mism == 0;
  }
  std::printf("Elapsed: %.0f ms\n", timer.ms());
  if (!ok) { std::printf("FAILED!\n"); return 1; }
  std::printf("PASSED!\n");
  return 0;
}
