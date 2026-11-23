// ps_phase_roundtrip.cpp -- unit check (tests/test_ps_phase.py::test_encode_decode_roundtrip): the host encoder
// (kernels/common/ps_phase.h, ps::PhaseCfg::regs()) and the SimX decoder (vortex-trees/ps/sim/common/phase_config.h,
// vortex::PhaseConfig::write()) agree for random configs, and both print the same canonical string. Also: an address
// outside the block is rejected without side effects, and the default config is all zero.
#include <cstdio>
#include <cstring>
#include "phase_config.h"
#include "ps_phase.h"

static uint32_t s = 12345u;
static uint32_t rnd() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
static uint64_t rnd64() { return (uint64_t(rnd()) << 32) | rnd(); }

int main() {
  int bad = 0;
  {
    vortex::PhaseConfig d;
    ps::PhaseCfg h;
    if (d.str() != h.str()) { std::printf("default str differs:\n%s\n%s\n", d.str().c_str(), h.str().c_str()); ++bad; }
    if (d.write(VX_DCR_PHASE_STATE_BEGIN - 1, 7) || d.write(VX_DCR_PHASE_STATE_END, 7) || d.str() != h.str()) {
      std::printf("out-of-block write accepted or changed state\n"); ++bad;
    }
    if (h.regs().size() != (size_t)(VX_DCR_PHASE_STATE_END - VX_DCR_PHASE_STATE_BEGIN)) {
      std::printf("regs() size %zu\n", h.regs().size()); ++bad;
    }
  }
  for (int t = 0; t < 2000; ++t) {
    ps::PhaseCfg h;
    h.mode = rnd(); h.flags = rnd(); h.l1_waymask = rnd(); h.l2_waymask = rnd(); h.insert_policy = rnd();
    for (auto& v : h.dxa_depth) v = uint8_t(rnd());
    h.dram_sched = rnd(); h.dram_aging = rnd();
    h.kv_base[0] = rnd64(); h.kv_limit[0] = rnd64(); h.kv_base[1] = rnd64(); h.kv_limit[1] = rnd64();
    h.wgt_base = rnd64(); h.wgt_limit = rnd64();
    vortex::PhaseConfig d;
    // start from a different non-zero state: a full write must overwrite everything
    for (uint32_t a = VX_DCR_PHASE_STATE_BEGIN; a < VX_DCR_PHASE_STATE_END; ++a) d.write(a, 0xdeadbeefu);
    uint32_t prev = 0;
    for (const auto& rv : h.regs()) {
      if (rv.first < prev || !vortex::PhaseConfig::is_phase_dcr(rv.first)) { std::printf("bad addr %#x\n", rv.first); ++bad; }
      prev = rv.first;
      if (!d.write(rv.first, rv.second)) { std::printf("write %#x rejected\n", rv.first); ++bad; }
    }
    if (d.str() != h.str()) {
      if (bad < 5) std::printf("mismatch:\n simx %s\n host %s\n", d.str().c_str(), h.str().c_str());
      ++bad;
    }
    if (d.persist_l2() != ((h.flags & ps::phase_dcr::FLAG_PERSIST_L2) != 0)) { std::printf("persist bit\n"); ++bad; }
  }
  std::printf("roundtrip: %s (%d errors)\n", bad ? "FAIL" : "OK", bad);
  return bad ? 1 : 0;
}
