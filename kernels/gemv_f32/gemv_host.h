// gemv_host.h -- host driver shared by ps_gemv_f32 (kernels/gemv_f32/main.cpp) and the cold-cache check
// ps_twolaunch (kernels/ps_twolaunch/main.cpp, which reaches this file through a symlink so that vxrun's app-source
// hash covers it).
//
// Usage: <app> [-f <fixture dir> | -m <rows> -n <cols> [-s <seed>]] [-v rows|lanes] [-L <launches>] [-t <tol>]
//              [-g <grid>] [-k <kernel.vxbin>] [-P]
//   -P   PhaseShift persist_l2 sensitivity knob: sets the [dcr_phase] persist_l2 bit before the first launch
//        (ps tree; the L2 then keeps its lines across launches). Refused (FAILED) unless the host is compiled with
//        -DPS_ALLOW_PERSIST_L2=1 (kernels/common/ps_phase.h).
//   -f   ps-fixture-v1 directory with tensors w (M x N), x (N), out (M) (absolute path: blackbox runs the app with
//        cwd = the build app dir); sha256 of every tensor is verified (ps_fixture.h)
//   -m/-n/-s  synthetic case: W, x uniform in [-1, 1) from the deterministic xorshift RNG (ps_common.h), seed -s
//   -v   kernel variant (see kernel.cpp): rows (default) or lanes; adds the ILP variants of kernel_ilp.cpp: rows_ilp2,
//        rows_ilp4 (thread per row, 2 / 4 independent accumulator chains) and lanes_ilp2 (warp per row, 2 chains). Their
//        summation order differs from the stock kernels (partial sums added at the end), so they pass the 2 gamma_K rule
//        (the default tolerance below) but are not bit-identical to rows / lanes; the PSDIGEST line identifies them exactly.
//   -L   number of back-to-back launches of the same kernel on the same buffers (default: app-specific; 1 for
//        ps_gemv_f32, 2 for ps_twolaunch). Before every launch after the first, y is overwritten on the device with
//        a NaN pattern, so each launch's output is checked independently (host DMA writes go straight to device RAM
//        in SimX, sw/runtime/simx/vortex.cpp:155-161, i.e. they do not touch any cache).
//   -t   tolerance on max_rel_err (default max(1e-4, 2 gamma_N) with gamma_N = N u / (1 - N u), u = 2^-24: identical to the
//        original 1e-4 for N <= 838, i.e. for every shape the earlier steps ran except K = 1792 (2.14e-4); states the rule
//        explicitly because two different summation orders of an N-long fp32 dot product differ by <= 2 gamma_N S_i)
//   -g   override the number of blocks (default: enough blocks for one thread (rows) / one warp (lanes) per row,
//        from vx_device_max_occupancy_grid). A smaller grid makes the kernels' grid-stride loops process several
//        rows per thread / warp (used by tests/test_kernels.py to exercise that path).
//
// Checks per launch (both must pass for PASSED!):
//   vs_f64     device y vs a float64 host recomputation from the same W, x
//   vs_fixture device y vs the fixture's NumPy float32 `out` (fixture runs only)
// and once per fixture: the fixture's own `out` vs the float64 recomputation (fixture_vs_f64, sanity of the input).
// Metric: rel_i = |y_i - ref_i| / S_i with S_i = sum_j |W_ij x_j| (definition and derivation in
// kernels/common/ps_common.h). Tolerance 1e-4 ("fp32 tolerance 1e-4 rel"): two correct float32 K-long dot
// products with different summation orders differ by at most 2 gamma_K S_i ~= 2 K u S_i with u = 2^-24, which is
// <= 1e-4 for K <= 838 (2 * 838 * 5.96e-8 = 9.99e-5). That covers every K this step tests (stories15M wq K = 288,
// random K = 512) and also stories15M w2 (K = 768). For larger K the worst-case bound exceeds 1e-4, but rounding
// errors are not all aligned in practice (typical growth ~ sqrt(K) u); a step that tests K > 838 must state its own
// tolerance (e.g. 2 K u) instead of silently reusing 1e-4.
#ifndef PS_GEMV_HOST_H
#define PS_GEMV_HOST_H

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <vortex2.h>

#include "common.h"
#include "ps_common.h"
#include "ps_fixture.h"
#include "ps_phase.h"   // -P: persist_l2 sensitivity knob (refused unless -DPS_ALLOW_PERSIST_L2=1)
#include "ps_waymask.h"   // -W: mechanism A2 way masks (encoding; option added, host side only)

namespace ps_gemv {

struct Ctx {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  vx_buffer_h W_buf = nullptr, x_buf = nullptr, y_buf = nullptr;
  ~Ctx() {
    if (!device) return;
    if (W_buf) vx_buffer_release(W_buf);
    if (x_buf) vx_buffer_release(x_buf);
    if (y_buf) vx_buffer_release(y_buf);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);  // PERF: lines (cumulative over all launches of this app run)
    vx_device_release(device);
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

inline void usage(const char* app) {
  std::printf("Usage: %s [-f fixture_dir | -m rows -n cols [-s seed]] [-v rows|lanes] [-L launches] [-t tol] "
              "[-g grid] [-k kernel.vxbin] [-P] [-W l1:l2]\n", app);
}

inline int run(int argc, char** argv, uint32_t default_launches) {
  std::string fixture, variant = "rows", kernel_file = "kernel.vxbin";
  uint32_t M = 64, N = 64, seed = 1, launches = default_launches, grid_override = 0;
  double tol = -1.0;   // < 0: default max(1e-4, 2 gamma_N), resolved once N is known
  bool persist = false;
  bool have_wm = false;   // PhaseShift: -W way masks (absent = no DCR write, the old path)
  uint32_t wm_l1 = 0, wm_l2 = 0;
  int c;
  while ((c = getopt(argc, argv, "f:m:n:s:v:L:t:g:k:PW:h")) != -1) {
    switch (c) {
      case 'f': fixture = optarg; break;
      case 'm': M = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': N = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 's': seed = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'v': variant = optarg; break;
      case 'L': launches = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 't': tol = std::strtod(optarg, nullptr); break;
      case 'g': grid_override = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      case 'P': persist = true; break;
      case 'W':   // PhaseShift: [dcr_phase] way masks "<l1word>:<l2word>" (ps_waymask.h, methodology §14.A)
        if (!ps::parse_waymask_arg(optarg, &wm_l1, &wm_l2)) { std::printf("bad -W %s\nFAILED!\n", optarg); return 1; }
        have_wm = true;
        break;
      case 'h': usage(argv[0]); std::exit(0);
      default: usage(argv[0]); return 1;
    }
  }
  if (variant != "rows" && variant != "lanes" && variant != "rows_ilp2" && variant != "rows_ilp4" && variant != "lanes_ilp2") {
    std::printf("bad -v %s\nFAILED!\n", variant.c_str());
    return 1;
  }
  const bool lanes_type = variant.rfind("lanes", 0) == 0;   // warp per row
  if (launches < 1) { std::printf("-L must be >= 1\nFAILED!\n"); return 1; }

  // ---- inputs
  std::vector<float> W, x, out_fx;
  std::string label;
  try {
    if (!fixture.empty()) {
      ps::Fixture fx(fixture);
      ps::Tensor tw = fx.tensor("w"), tx = fx.tensor("x"), to = fx.tensor("out");
      if (tw.shape.size() != 2 || tx.shape.size() != 1 || to.shape.size() != 1 || tw.dim(1) != tx.dim(0) ||
          tw.dim(0) != to.dim(0))
        throw ps::FixtureError("shape mismatch: need w (M,N), x (N), out (M)");
      M = tw.dim(0);
      N = tw.dim(1);
      W = std::move(tw.data);
      x = std::move(tx.data);
      out_fx = std::move(to.data);
      label = fx.model() + ":" + fx.op();
    } else {
      ps::Rng rng(seed);
      W.resize((size_t)M * N);
      x.resize(N);
      for (auto& v : W) v = rng.uniform();
      for (auto& v : x) v = rng.uniform();
      label = "synthetic:s" + std::to_string(seed);
    }
  } catch (const std::exception& e) {
    std::printf("fixture error: %s\nFAILED!\n", e.what());
    return 1;
  }
  if (tol < 0) {
    const double u = std::ldexp(1.0, -24), g = N * u / (1.0 - N * u);
    tol = std::fmax(1e-4, 2.0 * g);
  }
  std::printf("ps_gemv: %s M=%u N=%u variant=%s launches=%u tol=%.1e\n", label.c_str(), M, N, variant.c_str(),
              launches, tol);

  // ---- float64 reference and error scale S_i = sum_j |W_ij x_j|
  std::vector<double> ref64(M), scale(M);
  for (uint32_t i = 0; i < M; ++i) {
    double s = 0.0, a = 0.0;
    for (uint32_t j = 0; j < N; ++j) {
      double p = (double)W[(size_t)i * N + j] * (double)x[j];
      s += p;
      a += std::fabs(p);
    }
    ref64[i] = s;
    scale[i] = a;
  }
  bool ok = true;
  std::vector<double> ref_fx;
  if (!out_fx.empty()) {
    ref_fx.assign(out_fx.begin(), out_fx.end());
    auto st = ps::compare(out_fx.data(), ref64.data(), scale.data(), M, tol);
    ps::print_check("fixture_vs_f64", st);
    ok = ok && st.ok();
  }

  // ---- device
  Ctx ctx;
  PS_RT_CHECK(vx_device_open(0, &ctx.device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  PS_RT_CHECK(vx_queue_create(ctx.device, &qi, &ctx.queue));
  uint64_t nt = 0;
  PS_RT_CHECK(vx_device_query(ctx.device, VX_CAPS_NUM_THREADS, &nt));
  if (persist && !have_wm) {   // PhaseShift: persist_l2 for every launch (labelled sensitivity study; ps tree only)
    ps::PhaseCfg pc;
    pc.flags = ps::phase_dcr::FLAG_PERSIST_L2;
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
    std::printf("PSPERSIST-HOST: persist_l2=1 (sensitivity study)\n");
  } else if (have_wm) {   // PhaseShift: way masks (one block; with -P too, which SimX then rejects by design)
    ps::PhaseCfg pc;
    if (persist) pc.flags = ps::phase_dcr::FLAG_PERSIST_L2;
    ps::WayMaskCtl wmc;
    PS_RT_CHECK(wmc.set(pc, wm_l1, wm_l2));
    ps::waymask_print(wmc);
    PS_RT_CHECK(ps::phase_enqueue(ctx.queue, pc));
  }

  const uint64_t W_size = (uint64_t)M * N * sizeof(float), x_size = (uint64_t)N * sizeof(float),
                 y_size = (uint64_t)M * sizeof(float);
  ps_gemv_arg_t arg = {};
  arg.M = M;
  arg.N = N;
  PS_RT_CHECK(vx_buffer_create(ctx.device, W_size, VX_MEM_READ, &ctx.W_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.W_buf, &arg.W_addr));
  PS_RT_CHECK(vx_buffer_create(ctx.device, x_size, VX_MEM_READ, &ctx.x_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.x_buf, &arg.x_addr));
  PS_RT_CHECK(vx_buffer_create(ctx.device, y_size, VX_MEM_READ_WRITE, &ctx.y_buf));
  PS_RT_CHECK(vx_buffer_address(ctx.y_buf, &arg.y_addr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.W_buf, 0, W.data(), W_size, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.x_buf, 0, x.data(), x_size, 0, nullptr, nullptr));
  PS_RT_CHECK(vx_module_load_file(ctx.device, kernel_file.c_str(), &ctx.module_));
  PS_RT_CHECK(vx_module_get_kernel(ctx.module_, ("gemv_" + variant).c_str(), &ctx.kernel));   // gemv_rows, gemv_lanes, gemv_rows_ilp2, ...

  // rows: one thread per row; lanes: one warp (nt threads) per row. Block = one full core (NT x NW lanes), which
  // is a multiple of nt, so warps are never partial (the lanes shuffle needs all lanes active).
  uint32_t global = lanes_type ? (uint32_t)(M * nt) : M;
  uint32_t grid_dim[1], block_dim[1];
  PS_RT_CHECK(vx_device_max_occupancy_grid(ctx.device, 1, &global, grid_dim, block_dim));
  if (block_dim[0] % nt != 0) { std::printf("block %u not a multiple of NT %u\nFAILED!\n", block_dim[0], (unsigned)nt); return 1; }
  if (grid_override > 0) grid_dim[0] = grid_override;
  std::printf("launch: global=%u grid=%u block=%u NT=%u%s\n", global, grid_dim[0], block_dim[0], (unsigned)nt,
              grid_override > 0 ? " (grid overridden by -g)" : "");

  std::vector<float> y(M);
  const std::vector<float> poison(M, std::nanf(""));
  ps::Timer timer;
  for (uint32_t l = 0; l < launches; ++l) {
    if (l > 0)  // make every launch's output observable on its own
      PS_RT_CHECK(vx_enqueue_write(ctx.queue, ctx.y_buf, 0, poison.data(), y_size, 0, nullptr, nullptr));
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = ctx.kernel;
    li.args_host = &arg;
    li.args_size = sizeof(arg);
    li.ndim = 1;
    li.grid_dim[0] = grid_dim[0];
    li.block_dim[0] = block_dim[0];
    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    PS_RT_CHECK(vx_enqueue_launch(ctx.queue, &li, 0, nullptr, &launch_ev));
    PS_RT_CHECK(vx_enqueue_read(ctx.queue, y.data(), ctx.y_buf, 0, y_size, 1, &launch_ev, &read_ev));
    PS_RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    std::string tag = "launch" + std::to_string(l) + ":vs_f64";
    auto s64 = ps::compare(y.data(), ref64.data(), scale.data(), M, tol);
    ps::print_check(tag.c_str(), s64);
    ok = ok && s64.ok();
    if (!ref_fx.empty()) {
      tag = "launch" + std::to_string(l) + ":vs_fixture";
      auto sfx = ps::compare(y.data(), ref_fx.data(), scale.data(), M, tol);
      ps::print_check(tag.c_str(), sfx);
      ok = ok && sfx.ok();
    }
  }
  std::printf("Elapsed: %.0f ms\n", timer.ms());
  if (!ok) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}

}  // namespace ps_gemv

#endif  // PS_GEMV_HOST_H
