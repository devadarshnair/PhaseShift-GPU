// main.cpp -- ps_gemv_dxa host app (methodology §18.4): software weight streaming, the weights staged GMEM -> LMEM by
// the DXA engine (kernel.cpp), for the P32 (fp32 weights) and P8 (Q8_0 weights) tracks.
//
// Usage: ps_gemv_dxa -m M -n N [-s seed] [-w f32|q80] [-G GS] [-B B] [-d depth] [-T KT] [-g ctas] [-L launches] [-t tol]
//                    [-k kernel.vxbin] [-P]
//   -m/-n/-s  synthetic case, the SAME inputs as the twins (ps_gemv_f32 / ps_twolaunch for f32, ps_gemv_q80 for q80): ps::Rng(seed)
//             draws W (M x N uniform in [-1, 1)) first, then the B activation vectors; timing is data-independent.
//   -w        weight flavour: f32 (P32 track) or q80 (P8 track, default); q80 quantizes W and x on the host (ps_q80.h).
//   -G        Q8_0 group size (default 64 backed off to divide N, as ps_gemv_q80); must be 16, 32, 64 or 128 (GS / 4 lanes per group).
//   -B        activation vectors per launch, 1..4 (the staged tile is shared by all B).
//   -d        DXA stages (1 = no overlap, 2, 4); default 2.
//   -T        K-tile columns KT (f32 floats, a multiple of 16; q80 bytes, a multiple of GS). Default: the largest that fits the LMEM
//             budget (capacity of one core minus x and the stages), at most N rounded up.
//   -g        CTAs (cores used); default min(cores, ceil(M / NW)).
//   -L/-P     two-launch persist mode exactly as ps_gemv_q80 / ps_twolaunch: -L back-to-back launches with the output poisoned from the
//             host before every launch after the first (each launch's output is checked on its own); -P sets the persist_l2 DCR
//             (ps tree, -DPS_ALLOW_PERSIST_L2=1) before the first host write.
//
// Checks per launch (PASSED needs all):
//   f32  vs_f64 (device y vs the float64 product; rel_i = |y - ref| / S_i, tol = max(1e-4, 2 gamma_N): ps_common.h).
//   q80  vs_f64 (tol = max(1e-4, 2 gamma_{G+2}) as ps_gemv_q80) and runq_exact: the output must equal the runq.c-order float32 result
//        BIT for bit (the kernel sums the int32 group sums exactly and adds the fp32 group terms in group order).
// Every launch prints PSDIGEST (FNV-1a of the output bits). Counter check (perf class 16): PSDXA lines give the DXA counters read at
// the end of the run; the host prints the predicted LMEM write beats and gmem reads of the transfers it programmed
// (PSDXAEXP, the enumeration rule of sim/simx/dxa/dxa_core.cpp enumerate_work_list for row-major tiles) so the test can compare them.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <vortex2.h>
#include <dxa.h>

#include "common.h"
#include "ps_common.h"
#include "ps_phase.h"
#include "ps_q80.h"

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
    vx_device_dump_perf(device, stdout);  // PERF: lines (cumulative over every launch of this run)
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

#define PS_RT_CHECK(_expr)                                                    \
  do {                                                                        \
    int _ret = (int)(_expr);                                                  \
    if (_ret != 0) {                                                          \
      std::printf("Error: '%s' returned %d!\nFAILED!\n", #_expr, _ret);       \
      return 1;                                                               \
    }                                                                         \
  } while (false)

// LMEM write beats and GMEM line reads of one 2-D row-major transfer (tile0 x tile1 elements of elem bytes, source rows `stride` bytes
// apart starting at gbase, destination rows contiguous from a 64-byte-aligned smem): the rule of dxa_core.cpp enumerate_work_list
// (a beat = one LMEM word write of at most one GMEM line and one 64-byte LMEM word; consecutive same-line reads count once).
void expected_dxa(uint64_t gbase, uint64_t stride, uint32_t tile0, uint32_t tile1, uint32_t eb, uint64_t* beats, uint64_t* reads) {
  uint64_t prev_cl = ~0ull;
  for (uint32_t row = 0; row < tile1; ++row) {
    const uint64_t rg = gbase + (uint64_t)row * stride;
    const uint64_t sbase = (uint64_t)row * tile0 * eb;
    for (uint32_t e0 = 0; e0 < tile0;) {
      const uint64_t g = rg + (uint64_t)e0 * eb, s = sbase + (uint64_t)e0 * eb;
      const uint32_t cl_room = 64 - (uint32_t)(g & 63), sw_room = 64 - (uint32_t)(s & 63), row_room = (tile0 - e0) * eb;
      uint32_t span = std::min({cl_room, sw_room, row_room, 64u});
      span -= span % eb;
      if (span == 0) break;
      ++*beats;
      const uint64_t cl = g & ~63ull;
      if (cl != prev_cl) { ++*reads; prev_cl = cl; }
      e0 += span / eb;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string flavour = "q80", kernel_file = "kernel.vxbin";
  uint32_t M = 64, N = 64, seed = 1, GS = 0, B = 1, depth = 2, KT = 0, launches = 1, grid_override = 0;
  double tol = -1.0;
  bool persist = false;
  int c;
  while ((c = getopt(argc, argv, "m:n:s:w:G:B:d:T:g:L:t:k:Ph")) != -1) {
    switch (c) {
      case 'm': M = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': N = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'w': flavour = optarg; break;
      case 'G': GS = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'B': B = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'd': depth = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'T': KT = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'g': grid_override = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'L': launches = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 't': tol = std::strtod(optarg, nullptr); break;
      case 'k': kernel_file = optarg; break;
      case 'P': persist = true; break;
      case 'h': std::printf("Usage: %s -m M -n N [-s seed] [-w f32|q80] [-G GS] [-B B] [-d depth] [-T KT] [-g ctas] [-L launches] [-t tol] [-k kernel.vxbin] [-P]\n", argv[0]); return 0;
      default: return 1;
    }
  }
  if (flavour != "f32" && flavour != "q80") { std::printf("bad -w %s\nFAILED!\n", flavour.c_str()); return 1; }
  const bool q80 = flavour == "q80";
  if (launches < 1 || B < 1 || B > 4 || (depth != 1 && depth != 2 && depth != 4) || M == 0 || N == 0 || N % 4) {
    std::printf("bad arguments (-L >= 1, -B 1..4, -d 1|2|4, N %% 4 == 0)\nFAILED!\n");
    return 1;
  }
  if (q80) {
    if (GS == 0) { GS = 64; while (GS > 4 && N % GS) GS >>= 1; }
    if ((GS != 16 && GS != 32 && GS != 64 && GS != 128) || N % GS) { std::printf("need GS in {16,32,64,128} dividing N (N=%u GS=%u)\nFAILED!\n", N, GS); return 1; }
  } else {
    GS = 1;
  }
  const uint32_t G = q80 ? N / GS : 0;
  if (tol < 0) {
    const double u = std::ldexp(1.0, -24);
    if (q80) { const double m = G + 2; tol = std::max(1e-4, 2.0 * m * u / (1.0 - m * u)); }
    else { const double g = N * u / (1.0 - N * u); tol = std::max(1e-4, 2.0 * g); }
  }

  // ---- inputs (same RNG order as the twins: W, then the activation vectors)
  ps::Rng rng(seed);
  std::vector<float> Wf((size_t)M * N), xf((size_t)B * N);
  for (auto& v : Wf) v = rng.uniform();
  for (auto& v : xf) v = rng.uniform();
  std::vector<int8_t> Wq, xq;
  std::vector<float> Ws, xs;
  std::vector<float> ref32((size_t)B * M, 0.0f);
  std::vector<double> ref64((size_t)B * M), scale((size_t)B * M);
  if (q80) {
    Wq.resize((size_t)M * N); xq.resize((size_t)B * N); Ws.resize((size_t)M * G); xs.resize((size_t)B * G);
    ps::quantize_q80(Wf.data(), Wf.size(), GS, Wq.data(), Ws.data());
    for (uint32_t b = 0; b < B; ++b) ps::quantize_q80(xf.data() + (size_t)b * N, N, GS, xq.data() + (size_t)b * N, xs.data() + (size_t)b * G);
    for (uint32_t b = 0; b < B; ++b)
      for (uint32_t i = 0; i < M; ++i)
        ref32[(size_t)b * M + i] = ps::matmul_row_q80(Wq.data() + (size_t)i * N, Ws.data() + (size_t)i * G, xq.data() + (size_t)b * N,
                                                      xs.data() + (size_t)b * G, N, GS, &ref64[(size_t)b * M + i], &scale[(size_t)b * M + i]);
  } else {
    for (uint32_t b = 0; b < B; ++b)
      for (uint32_t i = 0; i < M; ++i) {
        double s = 0.0, a = 0.0;
        for (uint32_t j = 0; j < N; ++j) {
          const double p = (double)Wf[(size_t)i * N + j] * (double)xf[(size_t)b * N + j];
          s += p; a += std::fabs(p);
        }
        ref64[(size_t)b * M + i] = s; scale[(size_t)b * M + i] = a;
      }
  }

  // ---- device
  Ctx ctx;
  PS_RT_CHECK(vx_device_open(0, &ctx.device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  PS_RT_CHECK(vx_queue_create(ctx.device, &qi, &ctx.queue));
  uint64_t nt = 0, nw = 0, ncores = 0, isa = 0, lmem_cap = 0;
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_WARPS, &nw));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_CORES, &ncores));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_ISA_FLAGS, &isa));
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  if (!(isa & VX_ISA_EXT_DXA)) { std::printf("needs a device built with -DVX_CFG_EXT_DXA_ENABLE\nFAILED!\n"); return 1; }
  if (nt != 32 && nt != 16 && nt != 8) { std::printf("NT=%u unsupported (the Q8_0 segments need GS/4 <= NT)\nFAILED!\n", (unsigned)nt); return 1; }
  if (q80 && GS / 4 > nt) { std::printf("GS/4 = %u lanes exceed NT = %u\nFAILED!\n", GS / 4, (unsigned)nt); return 1; }
  const uint32_t NW = (uint32_t)nw;
  // tile columns: the largest multiple of the quantum (f32: 16 floats = one 64 B line; q80: GS) that fits the LMEM budget
  const uint32_t quantum = q80 ? GS : 16;
  const uint32_t kt_max = (N + quantum - 1) / quantum * quantum;
  if (KT == 0) {
    KT = quantum;
    for (uint32_t k = quantum; k <= kt_max; k += quantum)
      if (psgd_lmem(N, GS, B, k, depth, q80, NW).total <= lmem_cap) KT = k; else break;
  }
  if (KT % quantum || KT == 0 || KT > 65535) { std::printf("bad KT %u (multiple of %u, <= 65535)\nFAILED!\n", KT, quantum); return 1; }
  const psgd_lmem_t L = psgd_lmem(N, GS, B, KT, depth, q80, NW);
  if (L.total > lmem_cap) { std::printf("LMEM footprint %u B > %llu B per core (N=%u B=%u KT=%u depth=%u)\nFAILED!\n", L.total, (unsigned long long)lmem_cap, N, B, KT, depth); return 1; }
  const uint32_t ngroups = (M + NW - 1) / NW, nkt = (N + KT - 1) / KT;
  uint32_t grid = grid_override ? grid_override : (uint32_t)std::min<uint64_t>(ncores, ngroups);
  grid = std::max(1u, std::min(grid, ngroups));
  std::printf("ps_gemv_dxa: synthetic:s%u flavour=%s M=%u N=%u GS=%u G=%u B=%u depth=%u KT=%u nkt=%u row_groups=%u grid=%u NT=%u NW=%u lmem=%u/%llu launches=%u tol=%.3e\n",
              seed, flavour.c_str(), M, N, GS, G, B, depth, KT, nkt, ngroups, grid, (unsigned)nt, NW, L.total, (unsigned long long)lmem_cap, launches, tol);
  if (q80) {
    std::printf("PSQ80: Wq n=%zu gs=%u fnv1a64=%016llx\n", Wq.size(), GS, (unsigned long long)ps::fnv1a64_bytes(Wq.data(), Wq.size()));
    std::printf("PSQ80: Ws n=%zu gs=%u fnv1a64=%016llx\n", Ws.size(), GS, (unsigned long long)ps::fnv1a64_bytes(Ws.data(), Ws.size() * 4));
    std::printf("PSQ80: xq n=%zu gs=%u fnv1a64=%016llx\n", xq.size(), GS, (unsigned long long)ps::fnv1a64_bytes(xq.data(), xq.size()));
    std::printf("PSQ80: xs n=%zu gs=%u fnv1a64=%016llx\n", xs.size(), GS, (unsigned long long)ps::fnv1a64_bytes(xs.data(), xs.size() * 4));
  }
  if (persist) {   // persist_l2 BEFORE the first host write (gemv_host.h order; labelled resident control)
    ps::PhaseCfg pc;
    pc.flags = ps::phase_dcr::FLAG_PERSIST_L2;
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
    std::printf("PSPERSIST-HOST: persist_l2=1 (resident control)\n");
  }

  ps_gemv_dxa_arg_t arg = {};
  arg.M = M; arg.N = N; arg.GS = GS; arg.B = B; arg.KT = KT; arg.depth = depth; arg.q80 = q80 ? 1 : 0; arg.nw = NW;
  vx_buffer_h hW, hWs = nullptr, hx, hxs = nullptr, hy;
  const uint64_t y_size = (uint64_t)B * M * 4;
  const uint64_t w_bytes = q80 ? (uint64_t)M * N : (uint64_t)M * N * 4;
  PS_RT_CHECK(ctx.buffer(w_bytes, VX_MEM_READ, &arg.Wq_addr, &hW));
  PS_RT_CHECK(ctx.buffer(q80 ? (uint64_t)B * N : (uint64_t)B * N * 4, VX_MEM_READ, &arg.xq_addr, &hx));
  PS_RT_CHECK(ctx.buffer(y_size, VX_MEM_READ_WRITE, &arg.y_addr, &hy));
  if (q80) {
    PS_RT_CHECK(ctx.buffer((uint64_t)M * G * 4, VX_MEM_READ, &arg.Ws_addr, &hWs));
    PS_RT_CHECK(ctx.buffer((uint64_t)B * G * 4, VX_MEM_READ, &arg.xs_addr, &hxs));
  }
  if (arg.Wq_addr % 64 || arg.xq_addr % 4 || arg.Ws_addr % 4 || arg.xs_addr % 4) { std::printf("buffer alignment\nFAILED!\n"); return 1; }
  if (q80) {
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hW, 0, Wq.data(), Wq.size(), 0, nullptr, nullptr));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hWs, 0, Ws.data(), Ws.size() * 4, 0, nullptr, nullptr));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hx, 0, xq.data(), xq.size(), 0, nullptr, nullptr));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hxs, 0, xs.data(), xs.size() * 4, 0, nullptr, nullptr));
  } else {
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hW, 0, Wf.data(), Wf.size() * 4, 0, nullptr, nullptr));
    PS_RT_CHECK(vx_enqueue_write(ctx.queue, hx, 0, xf.data(), xf.size() * 4, 0, nullptr, nullptr));
  }
  // DXA descriptors: tile = NW rows x KT columns of the (M x N) weight matrix (rows are the outer dimension), landing row-major in LMEM
  if (q80) {
    PS_RT_CHECK(vortex::dxa::program_2d(ctx.device, PS_GEMVDXA_DESC_W, arg.Wq_addr, N, M, N, KT, NW, 1));
    PS_RT_CHECK(vortex::dxa::program_2d(ctx.device, PS_GEMVDXA_DESC_S, arg.Ws_addr, G, M, G * 4, KT / GS, NW, 4));
  } else {
    PS_RT_CHECK(vortex::dxa::program_2d(ctx.device, PS_GEMVDXA_DESC_W, arg.Wq_addr, N, M, N * 4, KT, NW, 4));
  }
  // predicted DXA counters of one launch: every tile of every CTA (OOB rows / columns are enumerated too)
  uint64_t exp_beats = 0, exp_reads = 0, exp_xfers = 0;
  for (uint32_t cta = 0; cta < grid; ++cta) {
    const uint32_t my_groups = (ngroups > cta) ? (ngroups - cta + grid - 1) / grid : 0;
    for (uint32_t t = 0; t < my_groups * nkt; ++t) {
      const uint32_t kt = t % nkt, rg = cta + (t / nkt) * grid;
      uint64_t gb = arg.Wq_addr + (uint64_t)(rg * NW) * (q80 ? N : N * 4ull) + (uint64_t)kt * KT * (q80 ? 1 : 4);
      expected_dxa(gb, q80 ? N : N * 4ull, KT, NW, q80 ? 1 : 4, &exp_beats, &exp_reads);
      ++exp_xfers;
      if (q80) {
        gb = arg.Ws_addr + (uint64_t)(rg * NW) * (G * 4ull) + (uint64_t)kt * (KT / GS) * 4;
        expected_dxa(gb, G * 4ull, KT / GS, NW, 4, &exp_beats, &exp_reads);
        ++exp_xfers;
      }
    }
  }
  std::printf("PSDXAEXP: transfers=%llu lmem_writes=%llu gmem_reads=%llu staged_bytes=%llu per_launch\n", (unsigned long long)exp_xfers,
              (unsigned long long)exp_beats, (unsigned long long)exp_reads,
              (unsigned long long)(exp_xfers / (q80 ? 2 : 1) * (uint64_t)NW * KT * (q80 ? 1 : 4)));
  PS_RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, "gemv_dxa", &ctx.kernel));

  bool ok = true;
  std::vector<float> y((size_t)B * M);
  const std::vector<float> poison((size_t)B * M, std::nanf(""));
  ps::Timer timer;
  for (uint32_t l = 0; l < launches; ++l) {
    const std::string pre = "launch" + std::to_string(l);
    if (l > 0) PS_RT_CHECK(vx_enqueue_write(ctx.queue, hy, 0, poison.data(), y_size, 0, nullptr, nullptr));   // make every launch observable
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = ctx.kernel;
    li.args_host = &arg;
    li.args_size = sizeof(arg);
    li.ndim = 1;
    li.grid_dim[0] = grid;
    li.block_dim[0] = (uint32_t)(nt * nw);
    vx_event_h ev = nullptr, rd = nullptr;
    PS_RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &ev));
    PS_RT_CHECK(vx_enqueue_read(ctx.queue, y.data(), hy, 0, y_size, 1, &ev, &rd));
    PS_RT_CHECK(vx_event_wait_value(rd, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(rd);
    vx_event_release(ev);
    auto s64 = ps::compare(y.data(), ref64.data(), scale.data(), y.size(), tol);
    ps::print_check((pre + ":vs_f64").c_str(), s64);
    ok = ok && s64.ok();
    if (q80) {
      size_t mism = 0;
      for (size_t i = 0; i < y.size(); ++i) mism += std::memcmp(&y[i], &ref32[i], 4) != 0;
      std::printf("PSBITEXACT: %s:runq_exact n=%zu mismatches=%zu variant=dxa\n", pre.c_str(), y.size(), mism);
      ok = ok && mism == 0;
    }
  }
  std::printf("Elapsed: %.0f ms\n", timer.ms());
  if (!ok) { std::printf("FAILED!\n"); return 1; }
  std::printf("PASSED!\n");
  return 0;
}
