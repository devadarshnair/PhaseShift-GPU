// main.cpp -- ps_libm_probe host. Usage: ps_libm_probe -L <pos:hd:hs,...> [-k kernel.vxbin]
// For every listed input the device's (one thread, see kernel.cpp) float32 freq, val, cosf(val), sinf(val) are printed
// as PSPROBE lines with their bit patterns, next to the float64 reference evaluated from the DEVICE's float32 val
// (cos/sin error in ulp of the float32 result: |dev - ref| / ulp(ref)). The run PASSES if every cos/sin is within
// 1 ulp (the bound kernels/rope/main.cpp's tolerance derivation assumes) and freq/val equal the host's float32
// recomputation of the same expressions within 1 ulp; otherwise it prints FAILED!.
#include <cstdint>
#include "common.h"
#include "elt_host.h"

static uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
static double ulp_of(double r) {
  float f = (float)std::fabs(r);
  if (f < FLT_MIN) return std::ldexp(1.0, -149);
  return (double)std::nextafter(f, INFINITY) - (double)f;
}

int main(int argc, char** argv) {
  std::string kernel_file = "kernel.vxbin", list;
  int c;
  while ((c = getopt(argc, argv, "L:k:h")) != -1) {
    switch (c) {
      case 'L': list = optarg; break;
      case 'k': kernel_file = optarg; break;
      default: std::printf("Usage: %s -L pos:hd:hs[,...] [-k kernel.vxbin]\n", argv[0]); return c == 'h' ? 0 : 1;
    }
  }
  std::vector<int32_t> in;
  for (const char* s = list.c_str(); *s;) {
    char* e;
    for (int f = 0; f < 3; ++f) {
      in.push_back((int32_t)std::strtol(s, &e, 10));
      if (e == s) { std::printf("bad -L\nFAILED!\n"); return 1; }
      s = (*e == ':' || *e == ',') ? e + 1 : e;
    }
  }
  const uint32_t n = (uint32_t)(in.size() / 3);
  if (n == 0) { std::printf("empty -L\nFAILED!\n"); return 1; }
  const float theta = 10000.0f;
  ps_elt::Dev dev;
  if (dev.open(kernel_file, "libm_probe")) return 1;
  ps_elt::Buf bi, bo;
  const std::vector<float> poison((size_t)4 * n, std::nanf(""));
  if (dev.alloc(bi, (uint64_t)in.size() * 4, VX_MEM_READ, in.data())) return 1;
  if (dev.alloc(bo, (uint64_t)n * 16, VX_MEM_READ_WRITE, poison.data())) return 1;
  ps_libm_probe_arg_t arg = {};
  arg.in_addr = bi.addr; arg.out_addr = bo.addr; arg.n = n; arg.theta = theta;
  ps::Timer timer;
  vx_event_h ev = nullptr;
  if (dev.launch(&arg, sizeof(arg), (uint32_t)dev.nt, 1, &ev)) return 1;  // one block of NT; only thread 0 works
  std::vector<float> out((size_t)4 * n);
  if (dev.read(out.data(), bo, ev)) return 1;
  vx_event_release(ev);
  bool ok = true;
  double worst = 0.0;
  for (uint32_t j = 0; j < n; ++j) {
    const int p = in[3 * j], hd = in[3 * j + 1];
    const uint32_t hs = (uint32_t)in[3 * j + 2];
    const float hfreq = 1.0f / std::pow(theta, hd / (float)hs);   // host float32 recomputation (host libm)
    const float hval = p * hfreq;
    const float freq = out[4 * j], val = out[4 * j + 1], cs = out[4 * j + 2], sn = out[4 * j + 3];
    const double rc = std::cos((double)val), rs = std::sin((double)val);
    const double ec = std::fabs((double)cs - rc) / ulp_of(rc), es = std::fabs((double)sn - rs) / ulp_of(rs);
    const double ef = std::fabs((double)freq - (double)hfreq) / ulp_of(hfreq);
    const double ev2 = std::fabs((double)val - (double)hval) / ulp_of(hval);
    const double ref64 = (double)p / std::pow(10000.0, (double)hd / (double)hs);
    std::printf("PSPROBE: pos=%d hd=%d hs=%u freq=%.9g(%08x) val=%.9g(%08x) cos=%.9g(%08x) sin=%.9g(%08x) "
                "cos_ulp=%.3f sin_ulp=%.3f freq_vs_host_ulp=%.3f val_vs_host_ulp=%.3f val_vs_f64=%.3e\n",
                p, hd, hs, freq, bits(freq), val, bits(val), cs, bits(cs), sn, bits(sn), ec, es, ef, ev2,
                (double)val - ref64);
    worst = std::fmax(worst, std::fmax(ec, es));
    if (!(ec <= 1.0 && es <= 1.0 && ef <= 1.0 && ev2 <= 1.0)) ok = false;
  }
  std::printf("PSPROBE: n=%u worst_cos_sin_ulp=%.3f\n", n, worst);
  return ps_elt::finish(ok, timer);
}
