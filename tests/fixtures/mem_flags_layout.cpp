// mem_flags_layout.cpp -- fixture (tests/test_mem_class.py::test_mem_flags_layout_dxa_on_off): compiles the ps
// tree's sim/simx/types.h with and without -DVX_CFG_EXT_DXA_ENABLE (its static_asserts must hold in both builds) and
// prints the raw value of MemFlags with each PhaseShift-visible field set, so the test can compare the two builds.
#include "types.h"
#include <cstdio>

int main() {
  using vortex::MemFlags;
  for (uint32_t c = 0; c < VX_PS_NUM_CLASSES; ++c) {
    MemFlags f;
    f.mem_class = c;
    std::printf("mem_class=%u raw=0x%08x\n", c, f.raw);
  }
  MemFlags g;
  g.amo_unsigned = 1;
  g.mem_class = 4;
  std::printf("amo_unsigned+bulk raw=0x%08x shift=%u mask=0x%08x size=%zu\n", g.raw, MemFlags::MEM_CLASS_SHIFT,
              MemFlags::MEM_CLASS_MASK, sizeof(MemFlags));
#ifdef VX_CFG_EXT_DXA_ENABLE
  MemFlags d;
  d.dxa_notify_done = 1;
  d.dxa_notify_bar_id = 0xFFFF;
  d.mem_class = 0;
  std::printf("dxa_sideband raw=0x%08x\n", d.raw);   // bits 4..20 only: must not touch the class bits
#endif
  return 0;
}
