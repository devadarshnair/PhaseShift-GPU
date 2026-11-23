// ps_waymask_unit.cpp -- unit check (tests/test_waymask.py::test_waymask_unit): the host helper
// (kernels/common/ps_waymask.h) and the SimX mask decoding (vortex-trees/ps/sim/common/phase_config.h,
// PhaseConfig::wm_effective / waymask_error), against expectations written from docs/methodology.md §14.A.
#include <cstdio>
#include "phase_config.h"
#include "ps_phase.h"
#include "ps_waymask.h"

static int bad = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main() {
  using vortex::PhaseConfig;
  // encoding: 0 = all ways; bits 31:16 apply to WEIGHT fills only, 0 there = same as 15:0
  CHECK(PhaseConfig::wm_effective(0, 4, false) == 0xF);
  CHECK(PhaseConfig::wm_effective(0, 8, true) == 0xFF);
  CHECK(PhaseConfig::wm_effective(0x3, 4, false) == 0x3);
  CHECK(PhaseConfig::wm_effective(0x3, 4, true) == 0x3);
  CHECK(PhaseConfig::wm_effective(0xF0000F, 8, false) == 0x0F);
  CHECK(PhaseConfig::wm_effective(0xF0000F, 8, true) == 0xF0);
  CHECK(PhaseConfig::wm_effective(0xC0000, 4, false) == 0xF);   // weight-only mask: others keep every way
  CHECK(PhaseConfig::wm_effective(0xC0000, 4, true) == 0xC);
  CHECK(PhaseConfig::waymask_error(0xF, 4)[0] == 0);
  CHECK(PhaseConfig::waymask_error(0x10, 4)[0] != 0);
  CHECK(PhaseConfig::waymask_error(0x100000, 4)[0] != 0);
  CHECK(PhaseConfig::waymask_error(0xFF00FF, 8)[0] == 0);
  CHECK(PhaseConfig::waymask_error(0x1000000, 8)[0] != 0);
  CHECK(PhaseConfig::waymask_error(1, 32)[0] != 0);
  // host word builder and parser
  CHECK(ps::waymask_word(0x0F, 0xF0) == 0xF0000Fu);
  uint32_t a = 0, b = 0;
  CHECK(ps::parse_waymask_arg("0x3:0xf0000f", &a, &b) && a == 3 && b == 0xF0000F);
  CHECK(!ps::parse_waymask_arg("3", &a, &b));
  CHECK(!ps::parse_waymask_arg("3:", &a, &b));
  CHECK(!ps::parse_waymask_arg("3:4x", &a, &b));
  // helper: changes outside a schedule are accepted and counted; inside a schedule refused (cfg untouched);
  // re-setting the same masks inside a schedule is not a change
  ps::WayMaskCtl c;
  ps::PhaseCfg cfg;
  CHECK(c.set(cfg, 0x1, 0x3) == 0 && cfg.l1_waymask == 0x1 && cfg.l2_waymask == 0x3 && c.changes() == 1);
  c.begin_schedule();
  CHECK(c.set(cfg, 0x1, 0x3) == 0 && c.changes() == 1);
  CHECK(c.set(cfg, 0x3, 0x3) == ps::kWaymaskRefused && cfg.l1_waymask == 0x1 && c.changes() == 1);
  CHECK(c.set(cfg, 0x1, 0x0) == ps::kWaymaskRefused && cfg.l2_waymask == 0x3);
  c.end_schedule();
  CHECK(c.set(cfg, 0x0, 0x0) == 0 && cfg.l1_waymask == 0 && cfg.l2_waymask == 0 && c.changes() == 2);
  // host struct and SimX decoder agree on the mask registers
  ps::PhaseCfg h;
  h.l1_waymask = 0xC0003; h.l2_waymask = 0xF0000F;
  PhaseConfig d;
  for (const auto& rv : h.regs()) d.write(rv.first, rv.second);
  CHECK(d.l1_waymask == 0xC0003 && d.l2_waymask == 0xF0000F && d.str() == h.str());
  std::printf("waymask_unit: %s (%d errors)\n", bad ? "FAIL" : "OK", bad);
  return bad ? 1 : 0;
}
