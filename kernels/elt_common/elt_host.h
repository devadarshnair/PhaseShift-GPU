// elt_host.h -- host helpers shared element-wise apps ps_rmsnorm, ps_rope, ps_silu_gate, ps_residual
//. Canonical copy: kernels/elt_common/elt_host.h. Each app directory reaches it through a symlink
// (kernels/<app>/elt_host.h -> ../elt_common/elt_host.h), so vxrun's app-source hash (kernels/common + kernels/<app>,
// rglob follows symlinks) covers it, as the earlier ps_twolaunch does for the gemv sources. kernels/elt_common has no
// main.cpp, so scripts/install_kernel_apps.sh does not treat it as an app.
//
// Contents: a device context (open, buffers, one 1-D launch, read back, perf dump at exit), batched-tensor loading
// from ps-fixture-v1 fixtures, and the element-wise error scale.
//
// Error metric for the element-wise kernels (modeling choice; the general metric is in kernels/common/ps_common.h):
// rel_i = |y_i - ref_i| / scale_i with scale_i = max(|ref_i|, FLT_MIN) unless the app says otherwise (RoPE uses the
// norm of the rotated input pair, see kernels/rope/main.cpp). The FLT_MIN floor only matters for results that are
// zero or subnormal: there a relative error is meaningless and the check degenerates to |y_i - ref_i| <= tol*FLT_MIN,
// i.e. practically exact agreement, which is what a correct kernel delivers for exact zeros (0*w, x + (-x)).
#ifndef PS_ELT_HOST_H
#define PS_ELT_HOST_H

#include <unistd.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <vortex2.h>

#include "ps_common.h"
#include "ps_fixture.h"
#include "ps_phase.h"     // -W: the [dcr_phase] block write
#include "ps_waymask.h"   // -W: mechanism A2 way masks (encoding)

namespace ps_elt {

// PhaseShift: the apps' -W <l1word>:<l2word> option (mechanism A2 way masks, methodology §14.A). The main parses it
// with waymask_opt(); Dev::open() then writes one [dcr_phase] block holding only the masks (every other field at its
// reset value 0) on the launch queue before the first launch. Without -W nothing is written: the old host path.
struct WaymaskOpt {
  bool have = false;
  uint32_t l1 = 0, l2 = 0;
};
inline WaymaskOpt& waymask_opt_state() {
  static WaymaskOpt o;
  return o;
}
inline bool waymask_opt(const char* arg) {
  WaymaskOpt& o = waymask_opt_state();
  if (!ps::parse_waymask_arg(arg, &o.l1, &o.l2)) return false;
  o.have = true;
  return true;
}

#define PS_RT_CHECK(_expr)                                                    \
  do {                                                                        \
    int _ret = (int)(_expr);                                                  \
    if (_ret != 0) {                                                          \
      std::printf("Error: '%s' returned %d!\nFAILED!\n", #_expr, _ret);       \
      return 1;                                                               \
    }                                                                         \
  } while (false)

struct Buf {
  vx_buffer_h h = nullptr;
  uint64_t addr = 0;
  uint64_t bytes = 0;
};

struct Dev {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  uint64_t nt = 0;  // threads per warp
  std::vector<Buf> bufs;

  ~Dev() {
    if (!device) return;
    for (auto& b : bufs)
      if (b.h) vx_buffer_release(b.h);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);  // PERF: lines (one launch per app run, methodology §2 consequence 5)
    vx_device_release(device);
  }

  int open(const std::string& kernel_file, const char* kernel_name) {
    PS_RT_CHECK(vx_device_open(0, &device));
    vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
    PS_RT_CHECK(vx_queue_create(device, &qi, &queue));
    if (waymask_opt_state().have) {   // PhaseShift (-W): masks only, before the first launch on this queue
      ps::PhaseCfg pc;
      ps::WayMaskCtl wmc;
      PS_RT_CHECK(wmc.set(pc, waymask_opt_state().l1, waymask_opt_state().l2));
      ps::waymask_print(wmc);
      PS_RT_CHECK(ps::phase_enqueue(queue, pc));
    }
    PS_RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &nt));
    PS_RT_CHECK(vx_module_load_file(device, kernel_file.c_str(), &module_));
    PS_RT_CHECK(vx_module_get_kernel(module_, kernel_name, &kernel));
    return 0;
  }

  // Allocate `bytes` on the device; if `init` is given, upload it (host DMA writes go straight to device RAM in
  // SimX, sw/runtime/simx/vortex.cpp:155-161, so they are outside every kernel's cycle count).
  int alloc(Buf& b, uint64_t bytes, uint32_t flags, const void* init) {
    b.bytes = bytes;
    PS_RT_CHECK(vx_buffer_create(device, bytes, flags, &b.h));
    PS_RT_CHECK(vx_buffer_address(b.h, &b.addr));
    bufs.push_back(b);
    if (init) PS_RT_CHECK(vx_enqueue_write(queue, b.h, 0, init, bytes, 0, nullptr, nullptr));
    return 0;
  }

  // One 1-D launch of `global` work items. warp_rows: the kernel maps one warp per item group and needs whole
  // warps (block % NT == 0). Prints the geometry. grid_override > 0 replaces the block count (grid-stride tests).
  int launch(const void* arg, size_t arg_size, uint32_t global, uint32_t grid_override, vx_event_h* ev) {
    uint32_t grid_dim[1], block_dim[1];
    PS_RT_CHECK(vx_device_max_occupancy_grid(device, 1, &global, grid_dim, block_dim));
    if (block_dim[0] % nt != 0) {
      std::printf("block %u not a multiple of NT %u\nFAILED!\n", block_dim[0], (unsigned)nt);
      return 1;
    }
    if (grid_override > 0) grid_dim[0] = grid_override;
    std::printf("launch: global=%u grid=%u block=%u NT=%u%s\n", global, grid_dim[0], block_dim[0], (unsigned)nt,
                grid_override > 0 ? " (grid overridden by -g)" : "");
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = kernel;
    li.args_host = arg;
    li.args_size = arg_size;
    li.ndim = 1;
    li.grid_dim[0] = grid_dim[0];
    li.block_dim[0] = block_dim[0];
    PS_RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, ev));
    return 0;
  }

  // Blocking read of a whole buffer after event `after` (the launch).
  int read(void* dst, const Buf& b, vx_event_h after) {
    vx_event_h rev = nullptr;
    PS_RT_CHECK(vx_enqueue_read(queue, dst, b.h, 0, b.bytes, after ? 1 : 0, after ? &after : nullptr, &rev));
    PS_RT_CHECK(vx_event_wait_value(rev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(rev);
    return 0;
  }
};

// A batched tensor from a fixture: shape (B, n) (or (n,) read as B = 1). Throws ps::FixtureError on mismatch.
struct Rows {
  uint32_t B = 0, n = 0;
  std::vector<float> data;
};

inline Rows load_rows(const ps::Fixture& fx, const std::string& name) {
  ps::Tensor t = fx.tensor(name);
  Rows r;
  if (t.shape.size() == 2) { r.B = t.dim(0); r.n = t.dim(1); }
  else if (t.shape.size() == 1) { r.B = 1; r.n = t.dim(0); }
  else throw ps::FixtureError(name + ": expected shape (B, n) or (n,)");
  r.data = std::move(t.data);
  return r;
}

inline void expect_same_shape(const Rows& a, const Rows& b, const char* what) {
  if (a.B != b.B || a.n != b.n) throw ps::FixtureError(std::string("shape mismatch: ") + what);
}

// Fixture params helpers
inline double param_num(const ps::Fixture& fx, const char* key) { return fx.meta().at("params").at(key).as_num(); }
inline uint32_t param_u32(const ps::Fixture& fx, const char* key) {
  double v = param_num(fx, key);
  if (!(v >= 0.0 && v < 4294967296.0) || v != (double)(uint32_t)v)
    throw ps::FixtureError(std::string("param ") + key + " is not a uint32");
  return (uint32_t)v;
}
inline int layer_of(const ps::Fixture& fx) { return (int)fx.meta().at("layer").as_num(); }

inline std::string fixture_label(const ps::Fixture& fx, uint32_t B) {
  return fx.model() + ":" + fx.op() + ":L" + std::to_string(layer_of(fx)) + ":B" + std::to_string(B);
}

// scale_i = max(|ref_i|, FLT_MIN) (see the header comment)
inline std::vector<double> elem_scale(const std::vector<double>& ref) {
  std::vector<double> s(ref.size());
  for (size_t i = 0; i < ref.size(); ++i) s[i] = std::fmax(std::fabs(ref[i]), (double)FLT_MIN);
  return s;
}

// compare + print one PSCHECK line; returns ok
inline bool check(const char* label, const float* y, const std::vector<double>& ref, const std::vector<double>& scale,
                  double tol) {
  ps::ErrStats s = ps::compare(y, ref.data(), scale.data(), ref.size(), tol);
  ps::print_check(label, s);
  return s.ok();
}

inline std::vector<double> to_f64(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

inline int finish(bool ok, const ps::Timer& t) {
  std::printf("Elapsed: %.0f ms\n", t.ms());
  if (!ok) { std::printf("FAILED!\n"); return 1; }
  std::printf("PASSED!\n");
  return 0;
}

}  // namespace ps_elt

#endif  // PS_ELT_HOST_H
