// ps_phase.h -- PhaseShift host helper for the [dcr_phase] DCR block (ps patch dcr-phase).
//
// Host code fills a ps::PhaseCfg and writes it to the device between launches. SimX (ps tree) latches the block into
// every consumer (L1/L2/L3 caches, the DRAM model, DXA; vortex-trees/ps/sim/common/phase_config.h) and keeps it
// across the per-launch reset, so the config written before launch k is the one in force during launch k. The base
// tree has no such block: SimX ignores unknown DCR writes there (sim/simx/core.cpp Core::dcr_write returns 0), so code
// using this header runs unchanged on both trees.
//
// Writes always cover the WHOLE block (all 21 registers, address order), so the device state equals the struct
// exactly and never depends on what an earlier launch wrote. A default-constructed PhaseCfg is all zero = stock
// behaviour (the no-op regression state).
//
// Ordering (vortex-trees/ps/sw/runtime): every vx_queue has one worker thread that retires its commands in FIFO
// order, and a launch command is followed on the same queue by the runtime's trailing cache flush
// (sw/runtime/common/device.cpp:816 cp_submit_cache_flush). Two ways to place a config between launches:
//   * ps::phase_enqueue(queue, cfg): vx_enqueue_dcr_write on the SAME queue as the launches. The writes retire after
//     the previous launch and its flush and before the next launch, with no host wait. Preferred.
//   * ps::phase_write(device, cfg): vx_dcr_write (sw/runtime/include/vortex.h:72), synchronous, but it goes through the
//     runtime's legacy default queue, which is NOT ordered with respect to other queues; the caller must first wait
//     for every previously enqueued launch (e.g. its read-back event) to complete.
// Commands on different queues are not ordered with respect to each other unless they are linked by event wait lists
// (sw/runtime/common/queue.cpp: a command's wait-list events must complete before its work runs). Independently of
// the host queues, the command processor model is serial and holds a LAUNCH until the device drains
// (sim/common/cmd_processor.cpp tick_engine WaitDone / tick_launch WaitBusy-WaitDrain), so a DCR write can never land
// in the middle of a running launch (errata 25).
#ifndef PS_PHASE_H
#define PS_PHASE_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <vortex.h>
#include <vortex2.h>
#include <VX_types.h>

namespace ps {

// DCR addresses of the [dcr_phase] block (vortex-trees/ps/VX_types.toml). Spelled out so that this header also
// compiles against the base tree's VX_types.h; checked against the generated header when it has the block.
namespace phase_dcr {
constexpr uint32_t STATE_BEGIN   = 0x028;
constexpr uint32_t MODE          = 0x028;
constexpr uint32_t FLAGS         = 0x029;
constexpr uint32_t L1_WAYMASK    = 0x02A;
constexpr uint32_t L2_WAYMASK    = 0x02B;
constexpr uint32_t INSERT_POLICY = 0x02C;
constexpr uint32_t DXA_DEPTH0    = 0x02D;
constexpr uint32_t DXA_DEPTH1    = 0x02E;
constexpr uint32_t DRAM_SCHED    = 0x02F;
constexpr uint32_t DRAM_AGING    = 0x030;
constexpr uint32_t KV0_BASE_LO   = 0x031;   // then KV0_BASE_HI, KV0_LIMIT_LO, KV0_LIMIT_HI
constexpr uint32_t KV1_BASE_LO   = 0x035;   // same 4-word layout
constexpr uint32_t WGT_BASE_LO   = 0x039;   // same 4-word layout
constexpr uint32_t STATE_END     = 0x03D;
constexpr uint32_t FLAG_PERSIST_L2 = 0x1;
constexpr uint32_t FLAG_RANGE_ACCT_ONLY = 0x2;   // ps patch mem-class: KV/weight ranges label, never classify
}  // namespace phase_dcr

#ifdef VX_DCR_PHASE_STATE_BEGIN   // ps tree: the generated header must agree with the constants above
static_assert(phase_dcr::STATE_BEGIN == VX_DCR_PHASE_STATE_BEGIN, "[dcr_phase] map mismatch");
static_assert(phase_dcr::MODE == VX_DCR_PHASE_MODE, "[dcr_phase] map mismatch");
static_assert(phase_dcr::FLAGS == VX_DCR_PHASE_FLAGS, "[dcr_phase] map mismatch");
static_assert(phase_dcr::L1_WAYMASK == VX_DCR_PHASE_L1_WAYMASK, "[dcr_phase] map mismatch");
static_assert(phase_dcr::L2_WAYMASK == VX_DCR_PHASE_L2_WAYMASK, "[dcr_phase] map mismatch");
static_assert(phase_dcr::INSERT_POLICY == VX_DCR_PHASE_INSERT_POLICY, "[dcr_phase] map mismatch");
static_assert(phase_dcr::DXA_DEPTH0 == VX_DCR_PHASE_DXA_DEPTH0, "[dcr_phase] map mismatch");
static_assert(phase_dcr::DXA_DEPTH1 == VX_DCR_PHASE_DXA_DEPTH1, "[dcr_phase] map mismatch");
static_assert(phase_dcr::DRAM_SCHED == VX_DCR_PHASE_DRAM_SCHED, "[dcr_phase] map mismatch");
static_assert(phase_dcr::DRAM_AGING == VX_DCR_PHASE_DRAM_AGING, "[dcr_phase] map mismatch");
static_assert(phase_dcr::KV0_BASE_LO == VX_DCR_PHASE_KV0_BASE_LO && VX_DCR_PHASE_KV0_LIMIT_HI == phase_dcr::KV0_BASE_LO + 3,
              "[dcr_phase] map mismatch");
static_assert(phase_dcr::KV1_BASE_LO == VX_DCR_PHASE_KV1_BASE_LO && VX_DCR_PHASE_KV1_LIMIT_HI == phase_dcr::KV1_BASE_LO + 3,
              "[dcr_phase] map mismatch");
static_assert(phase_dcr::WGT_BASE_LO == VX_DCR_PHASE_WGT_BASE_LO && VX_DCR_PHASE_WGT_LIMIT_HI == phase_dcr::WGT_BASE_LO + 3,
              "[dcr_phase] map mismatch");
static_assert(phase_dcr::STATE_END == VX_DCR_PHASE_STATE_END, "[dcr_phase] map mismatch");
static_assert(phase_dcr::FLAG_PERSIST_L2 == VX_PHASE_FLAG_PERSIST_L2, "[dcr_phase] map mismatch");
#endif
#ifdef VX_PHASE_FLAG_RANGE_ACCT_ONLY   // ps tree from patch mem-class on
static_assert(phase_dcr::FLAG_RANGE_ACCT_ONLY == VX_PHASE_FLAG_RANGE_ACCT_ONLY, "[dcr_phase] map mismatch");
#endif

// One [dcr_phase] configuration. Field meanings: vortex-trees/ps/sim/common/phase_config.h (0 = stock everywhere).
struct PhaseCfg {
  uint32_t mode = 0;
  uint32_t flags = 0;            // bit 0 = persist_l2 (sensitivity knob only); bit 1 = range_acct_only
  uint32_t l1_waymask = 0;
  uint32_t l2_waymask = 0;
  uint32_t insert_policy = 0;
  uint8_t dxa_depth[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint32_t dram_sched = 0;
  uint32_t dram_aging = 0;
  uint64_t kv_base[2] = {0, 0};
  uint64_t kv_limit[2] = {0, 0};
  uint64_t wgt_base = 0;
  uint64_t wgt_limit = 0;

  // The full block as (address, value) pairs in address order.
  std::vector<std::pair<uint32_t, uint32_t>> regs() const {
    namespace d = phase_dcr;
    std::vector<std::pair<uint32_t, uint32_t>> r;
    r.reserve(d::STATE_END - d::STATE_BEGIN);
    r.emplace_back(d::MODE, mode);
    r.emplace_back(d::FLAGS, flags);
    r.emplace_back(d::L1_WAYMASK, l1_waymask);
    r.emplace_back(d::L2_WAYMASK, l2_waymask);
    r.emplace_back(d::INSERT_POLICY, insert_policy);
    for (uint32_t w = 0; w < 2; ++w) {
      uint32_t v = 0;
      for (uint32_t i = 0; i < 4; ++i)
        v |= uint32_t(dxa_depth[4 * w + i]) << (8 * i);
      r.emplace_back(d::DXA_DEPTH0 + w, v);
    }
    r.emplace_back(d::DRAM_SCHED, dram_sched);
    r.emplace_back(d::DRAM_AGING, dram_aging);
    auto range = [&r](uint32_t lo_addr, uint64_t base, uint64_t limit) {
      r.emplace_back(lo_addr + 0, uint32_t(base));
      r.emplace_back(lo_addr + 1, uint32_t(base >> 32));
      r.emplace_back(lo_addr + 2, uint32_t(limit));
      r.emplace_back(lo_addr + 3, uint32_t(limit >> 32));
    };
    range(d::KV0_BASE_LO, kv_base[0], kv_limit[0]);
    range(d::KV1_BASE_LO, kv_base[1], kv_limit[1]);
    range(d::WGT_BASE_LO, wgt_base, wgt_limit);
    return r;
  }

  // Same one-line format as vortex::PhaseConfig::str() in SimX (tests/test_ps_phase.py compares them exactly).
  std::string str() const {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "mode=0x%x flags=0x%x l1_waymask=0x%x l2_waymask=0x%x insert=0x%x"
                  " dxa_depth=%u,%u,%u,%u,%u,%u,%u,%u dram_sched=0x%x dram_aging=%u"
                  " kv0=0x%llx-0x%llx kv1=0x%llx-0x%llx wgt=0x%llx-0x%llx",
                  mode, flags, l1_waymask, l2_waymask, insert_policy,
                  unsigned(dxa_depth[0]), unsigned(dxa_depth[1]), unsigned(dxa_depth[2]), unsigned(dxa_depth[3]),
                  unsigned(dxa_depth[4]), unsigned(dxa_depth[5]), unsigned(dxa_depth[6]), unsigned(dxa_depth[7]),
                  dram_sched, dram_aging,
                  (unsigned long long)kv_base[0], (unsigned long long)kv_limit[0],
                  (unsigned long long)kv_base[1], (unsigned long long)kv_limit[1],
                  (unsigned long long)wgt_base, (unsigned long long)wgt_limit);
    return buf;
  }
};

// Persist mode is a labelled sensitivity study, never a default: the default config must keep it off.
static_assert((PhaseCfg{}.flags & phase_dcr::FLAG_PERSIST_L2) == 0, "persist_l2 must never be enabled by default");

// persist_l2 guard: phase_enqueue / phase_write refuse a config with the persist_l2 bit
// unless the host code is compiled with -DPS_ALLOW_PERSIST_L2=1 (only the labelled persist-L2 sensitivity runs do
// that). The default (macro unset) is 0, so no default build can enable persist mode (static check below).
#ifndef PS_ALLOW_PERSIST_L2
#define PS_ALLOW_PERSIST_L2 0
#endif
static_assert(PS_ALLOW_PERSIST_L2 == 0 || PS_ALLOW_PERSIST_L2 == 1, "PS_ALLOW_PERSIST_L2 must be 0 or 1");
constexpr bool kPersistL2Allowed = (PS_ALLOW_PERSIST_L2 == 1);
constexpr int kPersistRefused = -77;   // return code of a refused persist_l2 write (nothing was written)
inline bool persist_refused(const PhaseCfg& c) {
  if ((c.flags & phase_dcr::FLAG_PERSIST_L2) == 0 || kPersistL2Allowed) return false;
  std::fprintf(stderr, "ps_phase: persist_l2 refused (build with -DPS_ALLOW_PERSIST_L2=1 for the persist_l2 sensitivity "
                       "study)\n");
  return true;
}

// Enqueue the whole block on `q` (the queue that carries the launches); see the ordering note at the top.
inline int phase_enqueue(vx_queue_h q, const PhaseCfg& c) {
  if (persist_refused(c)) return kPersistRefused;
  for (const auto& rv : c.regs()) {
    vx_result_t r = vx_enqueue_dcr_write(q, rv.first, rv.second, 0, nullptr, nullptr);
    if (r != VX_SUCCESS) return (int)r;
  }
  return 0;
}

// Write the whole block synchronously with vx_dcr_write. The caller must have waited for every earlier launch.
inline int phase_write(vx_device_h dev, const PhaseCfg& c) {
  if (persist_refused(c)) return kPersistRefused;
  for (const auto& rv : c.regs()) {
    int r = vx_dcr_write(dev, rv.first, rv.second);
    if (r != 0) return r;
  }
  return 0;
}

// "PSPHASE-HOST launch=<k> <str>": what the host wrote for launch k (k = 0-based launch index since device open).
inline void phase_print(uint32_t launch, const PhaseCfg& c) {
  std::printf("PSPHASE-HOST launch=%u %s\n", launch, c.str().c_str());
  std::fflush(stdout);
}

}  // namespace ps

#endif  // PS_PHASE_H
