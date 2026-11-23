// main.cpp -- ps_issueprobe host app: per-warp issue-cadence microbenchmarks.
//
// Usage: ps_issueprobe -m <variant> -w <warps 1..NW> -n <iterations> [-k kernel.vxbin]
//   variants: alu_ind alu_dep fma_ind4 fma_ind8 fma_ind16 fma_dep fadd_ind fsgnj_ind fcvt_ind mix_fa ld_ind ld_dep br8 br4
// One block of 32*W threads (grid 1) runs on core 0, i.e. exactly W resident warps (NT = 32). Each lane-0 stores the
// mcycle value before and after its warp's timed loop. Output (one line):
//   PSPROBE: variant=<v> warps=<W> iters=<n> bad=<#threads whose result differs from the host reference>
//            cyc_min=.. cyc_mean=.. cyc_max=.. span=<max t1 - min t0> t0_min=.. t1_max=..
// Host reference is bit-exact (integer wrap-around arithmetic; float fma/add/copysign replayed in the same order).
// tau and IPC are computed by results/<run>/analyze.py from these cycles and the dynamic instructions per iteration
// L (U = 32 asm instructions + compiler loop control, read from the kernel disassembly).
#include <cinttypes>
#include <cmath>
#include <cstring>

#include "common.h"
#include "elt_host.h"

static float rtree8(const float* f) { return ((f[0] + f[1]) + (f[2] + f[3])) + ((f[4] + f[5]) + (f[6] + f[7])); }
static float rtree16(const float* f) {
  return (((f[0] + f[1]) + (f[2] + f[3])) + ((f[4] + f[5]) + (f[6] + f[7]))) +
         (((f[8] + f[9]) + (f[10] + f[11])) + ((f[12] + f[13]) + (f[14] + f[15])));
}

int main(int argc, char** argv) {
  std::string mode = "alu_ind", kernel_file = "kernel.vxbin";
  uint32_t W = 1, n = 256;
  int c;
  while ((c = getopt(argc, argv, "m:w:n:k:h")) != -1) {
    switch (c) {
      case 'm': mode = optarg; break;
      case 'w': W = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'n': n = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'k': kernel_file = optarg; break;
      default: std::printf("usage: ps_issueprobe -m <variant> -w <warps> -n <iters>\n"); return 1;
    }
  }
  ps_elt::Dev dev;
  if (dev.open(kernel_file, mode.c_str())) return 1;
  uint64_t warps_hw = 0;
  PS_RT_CHECK(vx_device_query(dev.device, VX_CAPS_NUM_WARPS, &warps_hw));
  if (dev.nt != 32 || W < 1 || W > warps_hw) { std::printf("need NT=32 and 1 <= W <= %" PRIu64 "\nFAILED!\n", warps_hw); return 1; }
  ps::Timer timer;
  const uint32_t T = 32 * W;
  std::vector<uint32_t> bufw(1024);                    // 4 KiB source for the load variants
  for (uint32_t i = 0; i < bufw.size(); ++i) bufw[i] = i * 2654435761u;
  if (mode == "ld_dep") for (uint32_t i = 0; i < bufw.size(); ++i) bufw[i] = 0;   // patched below once the address is known
  std::vector<uint32_t> out(2 * T, 0xFFFFFFFFu), cyc(2 * W, 0);
  ps_elt::Buf bb, bo, bc;
  if (dev.alloc(bb, 4096, VX_MEM_READ_WRITE, bufw.data())) return 1;
  if (mode == "ld_dep") {                              // cell i holds its own device address
    for (uint32_t i = 0; i < bufw.size(); ++i) bufw[i] = (uint32_t)(bb.addr + 4ull * i);
    vx_event_h wev = nullptr; (void)wev;
    PS_RT_CHECK(vx_enqueue_write(dev.queue, bb.h, 0, bufw.data(), 4096, 0, nullptr, nullptr));
  }
  if (dev.alloc(bo, out.size() * 4, VX_MEM_READ_WRITE, out.data())) return 1;
  if (dev.alloc(bc, cyc.size() * 4, VX_MEM_READ_WRITE, cyc.data())) return 1;
  ps_issueprobe_arg_t arg = {};
  arg.out_addr = bo.addr; arg.cyc_addr = bc.addr; arg.buf_addr = bb.addr; arg.iters = n;
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li); li.kernel = dev.kernel; li.args_host = &arg; li.args_size = sizeof(arg);
  li.ndim = 1; li.grid_dim[0] = 1; li.block_dim[0] = T;
  std::printf("launch: grid=1 block=%u NT=%u\n", T, (unsigned)dev.nt);
  vx_event_h ev = nullptr;
  PS_RT_CHECK(vx_enqueue_launch(dev.queue, &li, 0, nullptr, &ev));
  if (dev.read(out.data(), bo, ev)) return 1;
  if (dev.read(cyc.data(), bc, ev)) return 1;
  vx_event_release(ev);

  size_t bad = 0;
  const float fbp = 0.999f, fcp = 1.0e-3f;
  for (uint32_t g = 0; g < T; ++g) {
    const uint32_t b = g | 1u;
    uint32_t wi = 0xFFFFFFFFu, wi2 = 0;
    float wf = 0;
    bool isf = false;
    auto ichains = [&](int nch, uint32_t per_iter) {   // chain k: a_k = g + k, += b, per_iter times per iteration
      uint32_t s = 0;
      for (int k = 0; k < nch; ++k) s += (g + k) + per_iter * n * b;
      return s;
    };
    float x0 = 1.0f + (float)(g & 1023u) * 0.0009765625f, f[16];
    for (int k = 0; k < 16; ++k) f[k] = x0 + (float)(k) * 0.1f;   // device: x0 + 0.1f, x0 + 0.2f ... (constant adds)
    // device constants are the float literals x0 + 0.1f etc.; reproduce them literally
    f[1] = x0 + 0.1f; f[2] = x0 + 0.2f; f[3] = x0 + 0.3f; f[4] = x0 + 0.4f; f[5] = x0 + 0.5f; f[6] = x0 + 0.6f;
    f[7] = x0 + 0.7f; f[8] = x0 + 0.8f; f[9] = x0 + 0.9f; f[10] = x0 + 1.0f; f[11] = x0 + 1.1f; f[12] = x0 + 1.2f;
    f[13] = x0 + 1.3f; f[14] = x0 + 1.4f; f[15] = x0 + 1.5f; f[0] = x0;
    auto fmachain = [&](int nch, uint32_t per_iter) {
      for (int k = 0; k < nch; ++k)
        for (uint32_t i = 0; i < per_iter * n; ++i) f[k] = std::fma(f[k], fbp, fcp);
    };
    if (mode == "alu_ind") wi = ichains(16, 2);
    else if (mode == "alu_dep") wi = g + 32u * n * b;
    else if (mode == "br8") wi = ichains(7, 4);
    else if (mode == "br4") wi = ichains(3, 8);
    else if (mode == "ffl_chk") wi = 0x0Du;   // DZ | OF | NX, see kernel.cpp
    else if (mode == "fma_ind16") { fmachain(16, 2); wf = rtree16(f); isf = true; }
    else if (mode == "fma_ind8") { fmachain(8, 4); wf = rtree8(f); isf = true; }
    else if (mode == "fma_ind4") { fmachain(4, 8); wf = (f[0] + f[1]) + (f[2] + f[3]); isf = true; }
    else if (mode == "fma_dep") { fmachain(1, 32); wf = f[0]; isf = true; }
    else if (mode == "fadd_ind") {
      for (int k = 0; k < 8; ++k) for (uint32_t i = 0; i < 4 * n; ++i) f[k] = f[k] + fbp;
      wf = rtree8(f); isf = true;
    } else if (mode == "fsgnj_ind") {
      for (int k = 0; k < 8; ++k) f[k] = std::copysign(f[k], -fbp);
      wf = rtree8(f); isf = true;
    } else if (mode == "fcvt_ind") {
      float v = (float)(int32_t)b;
      for (int k = 0; k < 8; ++k) f[k] = v;
      wf = rtree8(f); isf = true;
    } else if (mode == "mix_fa") {
      fmachain(8, 2); wf = rtree8(f); isf = true; wi2 = ichains(8, 2);
    } else if (mode == "ld_ind") {
      uint32_t s = 0;
      for (int k = 0; k < 8; ++k) s += 2u * bufw[g + 64u * k];
      wi = s;
    } else if (mode == "ld_dep") wi = (uint32_t)(bb.addr + 4ull * g);
    else { std::printf("unknown variant\nFAILED!\n"); return 1; }
    if (isf) {
      float gotf; std::memcpy(&gotf, &out[2 * g], 4);
      bad += (std::memcmp(&gotf, &wf, 4) != 0) || (out[2 * g + 1] != wi2);
    } else {
      bad += (out[2 * g] != wi);
    }
  }
  uint32_t tmin = 0xFFFFFFFFu, tmax = 0, cmin = 0xFFFFFFFFu, cmax = 0;
  double csum = 0;
  for (uint32_t w = 0; w < W; ++w) {
    uint32_t t0 = cyc[2 * w], t1 = cyc[2 * w + 1], d = t1 - t0;
    tmin = std::min(tmin, t0); tmax = std::max(tmax, t1);
    cmin = std::min(cmin, d); cmax = std::max(cmax, d); csum += d;
  }
  std::printf("PSPROBE: variant=%s warps=%u iters=%u bad=%zu cyc_min=%u cyc_mean=%.1f cyc_max=%u span=%u\n",
              mode.c_str(), W, n, bad, cmin, csum / W, cmax, tmax - tmin);
  return ps_elt::finish(bad == 0, timer);
}
