// ps_waymask.h -- PhaseShift host helper for mechanism A2 way masks (ps patch waymask; semantics in
// docs/methodology.md §14.A).
//
// Encoding of [dcr_phase] l1_waymask (L1 dcaches) and l2_waymask (L2), see vortex-trees/ps/sim/common/phase_config.h:
//   bits 15:0  allowed-victim way mask of every non-WEIGHT fill (0 = all ways = stock)
//   bits 31:16 allowed-victim way mask of WEIGHT-class fills  (0 = the same as bits 15:0)
//
// Legality (why a helper): a mask is a LAUNCH-BOUNDARY knob. A mask change is only sound when every way is empty
// when the new mask comes into force, i.e. after the cache flush/reset that ends every launch (the runtime's trailing
// cp_submit_cache_flush + SimX's per-launch reset; flush cost: ~1,040 walk cycles at PS-BASE-S1-4c plus the
// dirty write-backs, results/<run>/csw.csv). PS_SCHEDULE cannot change it at all (SimX's parser and
// scripts/mkschedule.py abort). This helper adds the host-side rule: inside a host "schedule" -- a declared run of
// consecutive launches that must see one cache-partition state (begin_schedule() .. end_schedule(), e.g. the
// launches of one P6 sub-kernel schedule) -- set() refuses any mask change (returns kWaymaskRefused and changes
// nothing); outside a schedule it accepts the change and counts it, so the caller can charge the flush cost.
//
// Header-only and runtime-free (it only edits a ps::PhaseCfg), so tests/fixtures/ps_waymask_unit.cpp can test it
// with plain g++. The caller writes the PhaseCfg with ps::phase_enqueue as usual.
#ifndef PS_WAYMASK_H
#define PS_WAYMASK_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ps {

constexpr int kWaymaskRefused = -78;   // return code of a refused mask change (nothing was changed)
constexpr uint32_t kA2FlushWalkCycles = 1040;   // L1D 16 + L2 1,024 walk cycles at PS-BASE-S1-4c (analytical)

// One way-mask word from a non-weight mask and a weight mask (0 = "same as the non-weight mask").
inline uint32_t waymask_word(uint32_t mask, uint32_t weight_mask = 0) {
  return (mask & 0xffffu) | ((weight_mask & 0xffffu) << 16);
}

// Parse "<l1word>:<l2word>" (strtoul base 0, e.g. "0x3:0xf0f"); false on a malformed string.
inline bool parse_waymask_arg(const char* s, uint32_t* l1, uint32_t* l2) {
  char* end = nullptr;
  const unsigned long a = std::strtoul(s, &end, 0);
  if (end == s || *end != ':') return false;
  const char* t = end + 1;
  const unsigned long b = std::strtoul(t, &end, 0);
  if (end == t || *end != '\0' || a > 0xffffffffUL || b > 0xffffffffUL) return false;
  *l1 = uint32_t(a);
  *l2 = uint32_t(b);
  return true;
}

class WayMaskCtl {
 public:
  void begin_schedule() { in_schedule_ = true; }
  void end_schedule() { in_schedule_ = false; }
  bool in_schedule() const { return in_schedule_; }
  uint32_t changes() const { return changes_; }   // accepted mask changes (each costs one A2 flush)
  uint32_t l1() const { return l1_; }
  uint32_t l2() const { return l2_; }

  // Set the masks for the next launch in `cfg` (any struct with l1_waymask / l2_waymask, i.e. ps::PhaseCfg).
  // Returns 0, or kWaymaskRefused (cfg untouched) for a change inside a schedule.
  template <class Cfg>
  int set(Cfg& cfg, uint32_t l1, uint32_t l2) {
    const bool change = (l1 != l1_) || (l2 != l2_);
    if (change && in_schedule_) {
      std::fprintf(stderr, "ps_waymask: mask change 0x%x:0x%x -> 0x%x:0x%x refused inside a schedule (A2 needs a "
                           "launch boundary + flush outside the schedule)\n", l1_, l2_, l1, l2);
      return kWaymaskRefused;
    }
    if (change) ++changes_;
    l1_ = l1;
    l2_ = l2;
    cfg.l1_waymask = l1;
    cfg.l2_waymask = l2;
    return 0;
  }

 private:
  bool in_schedule_ = false;
  uint32_t changes_ = 0;
  uint32_t l1_ = 0, l2_ = 0;   // the device's reset state: all ways
};

// "PSWAYMASK-HOST l1=0x.. l2=0x.. changes=n flush_walk_cycles=n*1040": what the host set (stdout).
inline void waymask_print(const WayMaskCtl& c) {
  std::printf("PSWAYMASK-HOST l1=0x%x l2=0x%x changes=%u flush_walk_cycles_analytical=%u\n", c.l1(), c.l2(),
              c.changes(), c.changes() * kA2FlushWalkCycles);
  std::fflush(stdout);
}

}  // namespace ps

#endif  // PS_WAYMASK_H
