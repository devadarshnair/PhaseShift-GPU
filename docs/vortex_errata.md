# Vortex errata (doc/code mismatches and behaviours that affect PhaseShift)

A running list of Vortex documentation errors, code/doc mismatches and simulator behaviours that affect PhaseShift,
with the evidence for each item (newest at the end). All `file:line` pointers are relative to the pristine Vortex clone
`third_party/vortex` at `50cb174546ec69a3869b42b8871e932e4440511f` (v3.0 + 5 commits). "Checked" = re-read in the
clone (read-only). "Unverified" = the claim's source could not be located in the clone. Items 1-7 come from an
internal literature/documentation review (`R`); the rest were found while building and measuring PhaseShift.

## From the documentation review

1. **L2/L3 macro names are wrong in the simulation doc.** `docs/simulation.md:65-80` gives
   `VX_CFG_L2_CACHE_SIZE` / `VX_CFG_L3_CACHE_SIZE` (table rows at `:72-73`, example `CONFIGS` at `:80`); the real keys
   are `VX_CFG_L2_SIZE` / `VX_CFG_L3_SIZE` (`VX_config.toml:250,265`). The doc's example silently changes nothing.
   *Checked; inner line numbers corrected.*
2. **Stale DXA design doc (3 places).** (a) It says the engine is per cluster; the code instantiates DXA per socket
   (`sim/simx/socket.cpp:174-177`, `hw/rtl/VX_socket.sv:689-806`, per R). (b) It points the reserved META fields at
   `VX_types.toml:282-291` (`docs/designs/dxa_async_copy_multicast.md:195`, *checked*); per R those lines hold
   texture constants and the fields are in `hw/rtl/dxa/VX_dxa_pkg.sv:39-50` and `sw/common/dxa_meta.h:24-26`.
   (c) Its `VX_config.toml:40-58` line references are outdated. (a) and (c) as reported by R; not re-read.
3. **DXA is named two ways.** "Data-transfer Acceleration (DXA)" in `CHANGELOG.md:41`; "Direct eXecution
   Accelerator" in `docs/designs/dxa_async_copy_multicast.md:1`. *Checked.*
4. **`VORTEX_USE_CP` exists only in the CHANGELOG** (`CHANGELOG.md:44`); a `grep -rn` of the tree finds no other
   occurrence. *Checked.*
5. **Verilator version.** `CHANGELOG.md:108` says "Verilator pinned to 5.028"; the toolchain installer's prebuilt is
   5.050 (`ci/toolchain_install.sh.in:146`, *checked*; installed `~/tools/verilator` reports 5.050). R's pointer `ci/toolchain_install.sh.in:145-154` for "5.052 was rolled back after a 204 GB
   elaboration peak" does not contain that statement, and no file in the clone mentions 5.052 or 204 GB:
   **unverified** (the provenance is probably upstream history, not the tree).
6. **Cache instantiation line numbers in `docs/designs/cache_subsystem.md` are stale** (R§2; which lines was not
   recorded). **Unverified**; whoever relies on a line number in that doc re-reads the source.
7. **Regression softmax initializes `max = 0.0`** (`tests/regression/softmax/kernel.cpp:16`), which is wrong for
   all-negative attention logits; PhaseShift's `softmax_fixed` initializes with −∞. *Checked.*

## Items found during planning

8. **Out-of-tree apps must live under the build dir.** `tests/regression/common.mk:1` sets
   `ROOT_DIR := $(realpath ../../..)` unconditionally. *Checked.*
9. **Ramulator's CMake fetches its dependencies at configure time** (FetchContent, not vendored):
   yaml-cpp `yaml-cpp-0.7.0`, spdlog `v1.11.0`, argparse `v2.9` from GitHub, into `ext/`
   (`third_party/ramulator/CMakeLists.txt:27-63`). The audit's "vendored" guess is wrong; with cmake 4.x the old
   yaml-cpp needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` (risk K6). *Checked.*
10. **`third_party/Makefile` builds Ramulator in-source with `make -j4`** (`third_party/Makefile:9`:
    `cd ramulator && mkdir -p build && cd build && cmake .. && make -j4`), ignoring `MAKEFLAGS=-j1`. Never run it in
    the pristine clone or `build/vortex-upstream`. *Checked.*
11. **The "free" DCR range starts at `VX_DCR_KMU_STATE_END = 0x024`** (`VX_types.toml:117`), which SimX treats as
    exclusive (`sim/simx/processor.cpp:357`: `addr >= VX_DCR_KMU_STATE_BEGIN && addr < VX_DCR_KMU_STATE_END`).
    RTL behaviour **unverified**. *SimX checked.*
12. **SimX resets every cache at each launch; its flush only clears dirty bits.** `ProcessorImpl::run()` calls
    `reset()` (`sim/simx/processor.cpp:274-275`), which calls `SimPlatform::instance().reset()` (`:332-333`); the cache
    reset loops over all sets (`sim/simx/mem/cache.cpp:696-703`; the line reset sets `lru_ctr = 0` and resets every sector, `:166-169`). The flush is
    triggered by the host's `dcr_read(VX_DCR_BASE_CACHE_FLUSH)` after `run()` (`sw/runtime/simx/vortex.cpp:165-170`,
    `processor.cpp:391-398`: `if (addr == VX_DCR_BASE_CACHE_FLUSH) … this->flush_caches();`), is not cycle-timed per line (`cache.cpp:1657-1700`) and runs outside any
    kernel's cycle count. The RTL flush walk invalidates (`docs/designs/cache_subsystem.md:646-692`, §6.6 "What flush
    does to the arrays"). Consequence: no cross-launch reuse on Vortex (methodology §2 consequence 4). *run/reset/cache
    reset/line reset/host hook/flush DCR checked; the untimed flush walk (`cache.cpp:1657-1700`) per *
13. **The DXA GMEM read that reaches L2/DRAM is built at `sim/simx/dxa/dxa_core.cpp:579-586`** (`MemReq mreq;
    mreq.addr = lw.gmem_cl_addr; … gmem_arb_->ReqIn`); `:718-764` is the LMEM **write** (`req.op = MemOp::ST`). R§5 and
    the internals notes (Q3) point at `:719`, which is the wrong request for class tagging (item 13).
    *Checked (`:579-586`, `:718-722`).*
14. **BLISS is not a drop-in scheduler for Vortex's DRAM model.** In the pinned Ramulator, `BLISSScheduler` registers
    as an `IBHScheduler` (`third_party/ramulator/src/dram_controller/impl/scheduler/bliss_scheduler.cpp:10-11`) and
    throws unless the BLISS plugin is active (`:31-39`), while `sim/common/dram_sim.cpp:191-193` hard-codes the
    `Generic` controller with the `FRFCFS` scheduler. *Checked.*
15. **SimX and RTL insertion semantics already differ.** RTL PLRU state advances on lookup hits only
    (`hw/rtl/cache/VX_cache_repl.sv:111`: "state advances on lookup hits only"; `:132`:
    `.write (init || (lookup_valid && lookup_hit))`; `:131` is `.read (~stall)`), whereas the SimX fill sets `line.lru_ctr = 0` (MRU insert,
    `sim/simx/mem/cache.cpp:1075`). Any SimX-vs-RTL comparison of mechanism D needs the `rtl_like_fill` baseline
   . *Checked; an earlier pointer `:111,131` was corrected to `:111,132`.*
16. **Custom CSRs: use 0x800–0x8FF.** By the RISC-V convention CSR addresses 0xC00–0xFFF (bits[11:10] = 11) are
    read-only; custom user read/write CSRs belong in 0x800–0x8FF, where no `VX_CSR_*` is defined in `VX_types.toml`
    (planner-checked). SimX aborts on unknown CSR reads (`sim/simx/csr_unit.cpp:328-331`) and writes
    (`:385-389`), so hint-writing apps cannot run on the base tree; unknown DCR writes are ignored
    (`sim/simx/core.cpp:1234-1238`). *csr_unit.cpp and core.cpp checked; the 0x8xx scan.*
17. **The open Yosys flow supports NanGate45.** R§2's "NanGate is a DC-only `.db`" holds only for the Synopsys flow:
    `hw/syn/yosys/Makefile:54-55` selects the vendored `NangateOpenCellLibrary_typical.lib` for `PDK=nangate45`.
    *Checked.*
18. **WGMMA K limit: comment vs code disagree.** `ci/testcases/dxa.yaml:504-506` keeps K at 64 because K = 128
    "drifts past the 10-ULP gate", but `tests/regression/sgemm_tcu_wg_dxa/main.cpp:29-31` now scales the bound as
    `max(8, K/5)` (K = 128 → 25 ULP). Which one holds is **unverified** try K = 128 once. *Both lines
    checked.*

## Appended by later steps

_(append as `N. **Title.** description, file:line, step ID, checked/unverified`)_

19. **cmake 4.2.3 honours `CMAKE_POLICY_VERSION_MINIMUM` as an environment variable**. A scratch project with
    `cmake_minimum_required(VERSION 3.1)` fails to configure without it and configures with
    `CMAKE_POLICY_VERSION_MINIMUM=3.5` in the environment; infra patch 0000 passes it as `-D` regardless
    (`third_party/Makefile`, PhaseShift trees only). With it, the pinned yaml-cpp 0.7.0 (`ext/yaml-cpp/CMakeLists.txt:2`)
    only prints a deprecation warning. Checked.
20. **No GCC 15 fix was needed for third_party or sw.** Ramulator2 (incl. FetchContent yaml-cpp 0.7.0, spdlog v1.11.0,
    argparse v2.9), the GDDR6 ext, softfloat, cocogfx and `sw/runtime/stub` (`sw/runtime/stub/Makefile:12`, `-Werror`)
    built with host GCC 15.2 unmodified; `sw/kernel` (`sw/kernel/Makefile:32`, `-Werror`) built unmodified with the
    **cross** compiler `~/tools/riscv32-gnu-toolchain/bin/riscv32-unknown-elf-gcc` (`sw/kernel/Makefile:26`), not host GCC
    (corrected). The planned `-include cstdint` workaround was not applied.
    Checked (logs (build logs)).
21. **Machine note: `/tmp` is a 3.8 GB RAM-backed tmpfs shared by every user process.** GCC writes its `cc*.s`
    temporaries there by default; the first Ramulator build failed at
    `third_party/ramulator/src/dram_controller/impl/scheduler/generic_scheduler.cpp:58` with `fatal error: error writing
    to /tmp/cc….s: No space left on device` while /tmp was full (an I/O failure, not a source problem). `scripts/build_vortex.sh` now exports `TMPDIR=$PS/build/tmp` (disk).
    Not a Vortex bug; recorded because every later build or simulation that spills temporaries should do the same.
22. **`third_party/Makefile`'s `all` also builds cocogfx** (`third_party/Makefile:1`: `all: cvfpu softfloat ramulator
    cocogfx`), so `make third_party` produces `cocogfx/libcocogfx.a` in addition to the artefacts lists. Checked.
23. **GCC 15 breaks the SimX build under `-Werror` through Ramulator's bundled fmt**. `sim/simx/Makefile:19` builds
    with `-Wall -Wextra -Wfatal-errors -Werror` and adds `ramulator/ext/spdlog/include` and `…/yaml-cpp/include` with
    plain `-I` (`:23-24`); compiling `sim/common/dram_sim.cpp` with host GCC 15.2 fails in spdlog v1.11.0's bundled fmt v9:
    `spdlog/fmt/bundled/core.h:1735:15: error: possibly dangling reference to a temporary [-Werror=dangling-reference]`
    (a GCC ≥ 13 warning; false positive in fmt v9). Fix = **infra patch 0001** (`patches/vortex/infra/
    0001-gcc15-thirdparty-isystem.patch`): those two FetchContent include dirs become `-isystem`, so warnings inside the
    vendored headers no longer fail the build while Vortex's own sources keep `-Werror` (no warning flag disabled, no
    simulated-behaviour change). Only `sim/simx/Makefile` is patched; `sim/{rtlsim,opaesim,xrtsim,avedsim}/Makefile` carry
    the same `-I` lines (`rtlsim/Makefile:16-17`) and will need the same treatment when builds rtlsim (a new infra
    patch). The test apps (host `-pedantic -Werror`, `tests/regression/common.mk`) and the kernel libraries built
    cleanly. Checked.
    ** addendum (rtlsim/opaesim/xrtsim, measured):** compiling `sim/common/dram_sim.cpp` with the exact C++ flags
    rtlsim's Makefile hands to Verilator (`make -n` of `build/vortex-ps/sim/rtlsim`, no Verilator run) fails on GCC 15.2
    with the same `-Werror=dangling-reference` in fmt v9 (`results/<run>/logs/simcompile-pre-rtlsim-dram_sim.log`).
    **Infra patch 0002** (`patches/vortex/infra/0002-gcc15-simulators-isystem.patch`) makes the two FetchContent include
    dirs `-isystem` in `sim/{rtlsim,opaesim,xrtsim,avedsim}/Makefile` (identical fix to 0001; applied to both trees,
    base re-verified from patches alone). After it, `dram_sim.cpp` compiles and the DramSim API probe passes with each
    simulator's flags (`results/<run>/sim_compile_check.sh`; rtlsim, opaesim, xrtsim, avedsim — which builds
    `xrtsim/xrt_sim.cpp` and `dram_sim.cpp` too —, `results/<run>/sim_compile_check_avedsim.out`). The simulator top files themselves include
    Verilator-generated headers and are first compiled when the RTL simulator verilates the GPU. Status: checked (compile only).
24. **Cold caches at every launch: confirmed by measurement (two-launch check); launches are not
    timing-identical.** `ps_twolaunch` launches the same GEMV 1, 2 or 3 times on the same device buffers in one app run
    (REG-CFG, `--perf=5`; MPM counters accumulate from device reset, `sim/simx/mem/cache.cpp:691`,
    `sim/simx/processor.cpp:334`, so a k-launch run prints the sum over k launches). stories15M layer-0 wq (288×288 fp32,
    324 KiB, fits the 1 MiB L2): L2 read misses 5,262 / 10,529 / 15,796 for 1 / 2 / 3 launches = 1.0000 / 1.0005 /
    1.0006 × k·misses(1); L2 requests exactly 5,694·k; a warm L2 would have given ≈ 5,262 + small for every k. 512×512
    random (1 MiB): 16,474 / 32,935 (0.9996 × 2). **§1's claim (no cross-launch reuse in SimX) holds; memhier keeps
    `persist_across_launch = False` for Vortex.** Side finding: launch k ≥ 2 is not a cycle-exact replay of launch 1
    even though every cache is reset — wq: +5 L2 read misses and +625 cycles per later launch (+0.7 %); 512×512: L2
    requests 185,088 vs 2 × 100,208 (−7.6 %), core-0 L1 dcache read misses 38,370 in launch 2 vs 50,157 in launch 1
    (`--perf=4`), −1,657 cycles (−0.6 %). Instructions are exactly k × instrs(1). The L1 miss count at this shape is
    timing/interleaving-dependent even within one launch: in the same 1-launch run (`--perf=4`, log
    `results/<run>/logs/P2.2_ps_twolaunch_86a399fa_r0.log`, second-pass rows;: `…_f7f6fd17_r0.log`) cores 0/1/3 have 50,157 / 50,276 / 50,375 L1 read misses
    and core 2 has 39,232, for identical per-core work (98,688 requests each). (The `evicts` counter is no evidence
    either way: it counts dirty write-backs only, `sim/simx/mem/cache.cpp:1065,1334,1686`, and the L1 is write-through
    at REG-CFG, `VX_config.toml:226`.) Hypothesis — **code-supported, not isolated**: DRAM-model state survives the
    per-launch reset. `DramSim::Impl` builds each channel's Ramulator frontend and memory system once, in its
    constructor (`sim/common/dram_sim.cpp:279-295`), and `reset()` (`:308-317`, called from `sim/simx/mem/memory.cpp:82`)
    only clears the wrapper's cycle counters and queues; Ramulator's bank/row-buffer/refresh state and clock carry
    over, so DRAM latencies and hence warp interleaving and MSHR merging differ in later launches. Other non-reset state
    was not audited. Consequence (rule in `docs/methodology.md` §2, consequence 5): SimX is deterministic for a fixed
    launch sequence; per-kernel cycles come from the first launch of an app run; k-launch totals ÷ k only when stated;
    A/B comparisons only on identical launch sequences. Data: `results/<run>/coldcache.csv`, rows
    `kind = coldcache[-l1]`. Status: measured.
    ** addendum (flush write-backs are dropped from the DRAM model's timing).** The per-launch reset
    clears the DramSim WRAPPER's queues — `DramSim::Impl::reset()` empties every channel's `pending_txs`, `read_rsps`
    and `write_rsps` (`sim/common/dram_sim.cpp:308-316`; call chain `ProcessorImpl::run()` → `reset()`
    `sim/simx/processor.cpp:274-275` → `SimPlatform::instance().reset()` `:332-333` → `Memory::on_reset()`
    `sim/simx/mem/memory.cpp:182` → `dram_sim_.reset()` `:82`) — while Ramulator keeps what it already accepted. The
    cache-flush loops end on `SimPlatform::instance().idle()` (`processor.cpp:248,262,269`), and `idle()` counts only
    SimChannel packets and cross-domain calls (`sim/common/simobject.h:255-256`, ps tree), not DramSim's internal
    transaction queues. So the write-backs of every flush still waiting in `pending_txs` when the flush loop ends are
    discarded at the next launch's reset (after the last launch they are still queued when the host reads the
    counters): they never reach Ramulator and never contend with the next launch; entries already accepted by Ramulator
    but not yet drained from `write_rsps` are dropped from the wrapper only (their completion is never reported).
    Measured with patch ps-counters (`results/<run>/traces/`, completions = `dram_wr_done`): `ps_phase_rw -L 5` 16
    write-backs per flush, 11 completed, identical in each of
    the 5 launches; `ps_twolaunch -L 2` 18 / 11 per launch; decode gqa REG-CFG 208 / 105; sgemm -n64 256 / 153.
    Functional results are unaffected: `Memory` writes the bytes into RAM when the request arrives, before it is timed
    (`memory.cpp:105-113`). Timing consequence: the DRAM-side cost of write-backs is under-modelled at every launch
    boundary (base and ps trees alike; stock behaviour, not changed by any PhaseShift patch). Counter consequence
    (methodology §12.9): `dram_wr_done` / `dram_wr_wait` undercount, Little's law cannot be applied to classes that
    write, and later steps use `dram_wr_bytes` (issued) for write traffic. Line numbers pristine unless marked. Status: checked (read + measured).
25. **[dcr_phase] DCR block placed at 0x028–0x03C; free-range check (resolves item 11's open RTL point).** Every
    DCR block in `VX_types.toml` (ps tree): BASE [0x000, 0x002), MMU [0x004, 0x009), KMU [0x010, 0x024), TEX
    [0x040, 0x060), RASTER [0x060, 0x06A), OM [0x080, 0x098), RTU [0x0A0, 0x0A7), DXA [0x100, 0x280); no `VX_DCR_*`
    value lies in 0x024–0x03F. Line numbers below are the pristine clone's. Range tests in SimX are all half-open `[BEGIN, END)` (`sim/simx/processor.cpp:357` KMU,
    `cluster.cpp:424,457` OM/RASTER, `socket.cpp:385,390` DXA/TEX); the RTL decodes DCRs by exact address only
    (`hw/rtl/VX_kmu.sv:114-148` `case (dcr_req_addr)` with `default: ;`, `hw/rtl/core/VX_dcr_data.sv:51,81` equality
    tests), so 0x024 is not captured by the KMU in RTL either (*read, not simulated*). The CP's AXI-lite register file
    also uses offsets 0x020–0x02C (`sim/common/cmd_processor.cpp:78-82,135-138`, `sw/runtime/common/device.cpp:76-77`), but that is
    a separate address space (CP registers, not DCRs). The block nevertheless starts at **0x028**, leaving 0x024–0x027
    as a guard after KMU, which grew upstream (CLUSTER_DIM_*) and may grow again. Unknown DCR writes are ignored by
    SimX (`Core::dcr_write`, `sim/simx/core.cpp:1234-1238`, returns 0), so host code writing the block runs unchanged
    on the base tree (measured: `results/<run>/`). Ordering: a `vx_queue` retires its commands in FIFO order on one
    worker thread (`sw/runtime/common/queue.cpp`), the launch command is followed on the same queue by the trailing
    cache flush (`sw/runtime/common/device.cpp:816`), and `vx_dcr_write` (`sw/runtime/common/legacy_runtime.cpp:253`)
    goes through the runtime's separate legacy default queue, which is not ordered against other queues; hence the
    host helper `kernels/common/ps_phase.h` offers `phase_enqueue` (same queue as the launches) and `phase_write`
    (synchronous; caller waits first). The latched config survives the per-launch reset because every consumer keeps
    it in a member its `on_reset()` does not touch (verified per consumer and launch with `PS_DUMP_PHASE=1`,
    `tests/test_ps_phase.py`). Status: checked.
    ** addendum (minor 5).** Precisely: commands on *different* queues are not ordered with respect to
    each other **unless they are linked by event wait lists** — every enqueue takes `n_wait_events, wait_events`
    (`sw/runtime/include/vortex2.h:349-350`), and a queue's worker thread waits for each of a command's wait-list events
    before running it (`sw/runtime/common/queue.cpp:63-64,80-85`). Independently of the host queues there is a stronger,
    model-level guarantee: the command processor model executes its ring serially and holds a LAUNCH command in
    `EngState::WaitDone` until the launch FSM returns to Idle (`sim/common/cmd_processor.cpp:557-561`), which happens only
    after the device reported busy and then drained (`tick_launch`, `:377-394`, `WaitBusy` → `WaitDrain` on
    `vortex_busy()`). So no CP-ring DCR write can take effect while a launch is running, the same guarantee the stock KMU
    DCRs rely on. `kernels/common/ps_phase.h`'s header comment says the same. Status: read (not simulated).

26. **Stock `softmax` regression kernel: measured wrong on all-negative rows, and on its own inputs (extends
    item 7).** Reproduced verbatim as `softmax_stock` in `kernels/softmax_fixed/kernel.cpp` (instructions within 0.01 % of
    the stock app at its shape: 1,790,015 vs 1,789,845, REG-CFG 64×64) and compared with a float64 softmax (8 rows × 64,
    `results/<run>/summary.csv`, kind `stock`): U[0, 1) rel error 1.6·10⁻² (the stock app's own inputs; its CI check passes
    only because the host reference in `tests/regression/softmax/main.cpp` uses the same polynomial and the same
    max = 0); all-negative U[−30, −1) rel 5.6·10¹¹ (p = 0.073 where 1.3·10⁻¹³ is correct). Two causes: the running max
    starts at 0 (`kernel.cpp:16`), and the "exp" is a 4-term polynomial (`common.h`) that is accurate only near 0 and is
    evaluated in double (`0.25`, `0.333`, `0.5` are double literals, soft-float on rv32imaf: 297 cycles/element). With an
    accurate exp, max = 0 alone is harmless for moderate negative logits (shift invariance) but gives rel error 1.05 at
    U[−120, −100) (every exp subnormal) and 512/512 NaN at U[−200, −150) (every exp 0, 0/0). `softmax_fixed` (running max
    from −∞) is correct on all of them. Status: measured.
27. **Device exp costs: musl `expf` is soft-float double; MUFU `vx_expf_fast` error grows with |x|.** The kernel
    libc/libm is musl (`~/tools/libc32/bin/musl-clang`, linked by `tests/regression/common.mk:95`; found the same,
    methodology §14.0); its `expf` computes in
    double, and rv32imaf has no D extension, so a call runs `__muldf3`/`__adddf3`: ≈ 752 instructions, ≈ 510 cycles
    per element at REG-CFG (exp probe), vs 36 / 25 for the float-only `ps_expf` (max 0.864 ulp on the 65,541-point probe grid; 0.93 ulp on an 8 M-point random sample) and 10 / 7.8 for
    MUFU EX2. `vx_mufu.h` only says MUFU results are "approximate" (`sw/kernel/include/vx_mufu.h:15`); measured,
    `vx_expf_fast(x) = ex2(x·fl(log₂e))` (`vx_mufu.h:52-53`) has
    relative error up to 62 ulp (3.8·10⁻⁶) at x = −81.8: besides ex2's 2 ulp, the prescale's rounding and the constant
    `1.44269504088896341f` = 0x3fb8aa3b, which is 0.224 u below log₂e, shift the exponent, giving ≤ (1.224|x| + 4)u
    relative (asserted at 65,541 points over [−104, 89]; subnormal results are flushed to 0). Consequence: the MUFU softmax
    tolerance is its derived bound (≤ 1.44·10⁻⁵ at n = 4096), not 1e-5 (methodology §14.1). Status: measured.
28. **WGMMA K limit (resolves item 18): the stock `sgemm_tcu_wg_dxa` passes its own check far beyond K = 64; any K that
    is a multiple of tileK works.** At TIER-CFG (NT = 32, ISSUE_WIDTH = 2, WGMMA_NRC = 32, fp16 → fp32) the stock
    app's `max(8, K/5)` ULP gate (`tests/regression/sgemm_tcu_wg_dxa/main.cpp:29-31`) passed at K = 128 (drift 3 ULP,
    bound 25), 288 (≤ 5 / 57), 512 (≤ 8 / 102), 768 (≤ 8 / 153) and 1792 (≤ 15 / 358) — 30 stock runs, all PASSED
    (`results/<run>/summary.csv`, `path = tcu_stock`). The `ci/testcases/dxa.yaml:504-506` comment ("K = 128 drifts past the
    10-ULP gate") describes the old fixed gate and is stale; no host-side K split is needed. The real constraint is
    K % tileK = 0 (tileK = 16 here, `sw/common/tensor_cfg.h:358`), not K % 64. Also measured: the copied device code in
    `kernels/linear` (`ps_linear -v tcu`) executes exactly the stock instruction count at every compared shape and
    0.995–1.001× its cycles (`results/<run>/tcu_equiv.csv`; residual = different module image / buffer addresses, the
    data are not the same either). Status: measured.
29. **GEMV `rows` layout at power-of-two row strides: L1 set conflicts, not capacity (cause corrected
    after a discriminating run).** `ps_gemv_f32 -v rows` (one thread per output row, as the stock `sgemv`,
    `kernels/gemv_f32/kernel.cpp:35`) at TIER-CFG (NT = 32) on llama8b_s8 w1 (1792 × 512 fp32, row stride 2,048 B) takes
    907,642 cycles and reads 4.79 MB from DRAM (1.30× the 3.68 MB compulsory; `--perf=7`), vs 261,884 cycles / 3.70 MB for
    the warp-per-row `lanes` variant. Cause: the L1 is 16 KiB, 4-way, 64 B lines = 64 sets (`VX_config.toml:220-221`);
    the 32 lanes of a warp load 32 rows at a 2,048 B stride, which map onto only **2 sets** (8 ways for 32 live lines), so
    lines are evicted before their next 3 float4 reads. Stride 3,072 B (K = 768) or 7,168 B (K = 1792) maps onto 4 sets;
    stride 1,152 B (K = 288, stories15M w1) onto 32 sets — and there `rows` is as fast as `lanes` (0.97×) although its
    live-line footprint equals llama w1's, so capacity is not the cause (an earlier version of this item said it was). Discriminating
    runs (`results/<run>/diag.csv`; first run): the same `rows` kernel with K = 528
    (stride 2,112 B, 3 % more bytes) takes **247,430 cycles (3.67× faster than K = 512)** with DRAM reads at 1.005× the
    compulsory bytes (3.80 MB), and is faster than `lanes` at K = 512 (261,884) or K = 528 (284,121). Rows/lanes cycle ratios over
    the benchmark shapes follow the set count: 10.1× (128 × 512), 5.5× (512 × 512), 3.5× (1792 × 512), 3.8× (K = 1792),
    6.4× (stories w2, K = 768), 0.97× (K = 288). Remedy: make the row stride an odd number of 64 B lines (e.g. +16 floats: 2,112 B = 33 lines), since
    any stride whose line count shares a large power-of-two factor with the 64 sets conflicts (K = 768 = 48 lines → 4
    sets, although 768 is not a power of two), or use `lanes`; the choice depends on the row stride, not on NT alone (at NT = 4, `rows` was faster
    on the 288/768/512-wide shapes). Consequence for the full-model driver and the machine characterization: store weights with a padded leading dimension or pick the
    variant per (shape, NT), and never read a rows-vs-lanes difference at power-of-two strides as latency evidence.
   measured.
30. **`__builtin_fmaf` compiles to a libcall in the kernel build (toolchain, not Vortex).** The regression kernel flags
    (`tests/regression/common.mk:86-90`) leave clang's default `-fmath-errno` on for riscv32-unknown-elf, so
    `__builtin_fmaf(a, b, c)` becomes `jal fmaf` (musl: `fmadd.s; ret`) with argument moves around it, while
    `__builtin_elementwise_fma` (and `a * b + c` under `#pragma clang fp contract(fast)`, tested) emit `fmadd.s` inline; both are the same fused operation
    (one rounding), so results are bit-identical. Measured (smoke-dxa, same tiles): prefill attention (16, 2, 128),
    S = 64: 31.05 M → 16.06 M instructions, 90.1 M → 49.1 M cycles, identical error statistics. Affected builds: `jal <fmaf>`
    count 251 in `ps_decode_attn_gqa`, 30 in `ps_softmax_fixed` (its `ps_expf`), 0 in `ps_linear`. maps the
    builtin to the inline form inside its own kernel; fixing the others (e.g. `-fno-math-errno` in the shared kernel flags)
    is an installer/common-flags change and changes their instruction and cycle counts. Status: measured
    (`build/tmp/p27_kernel_{fmaf_libcall,inline_fma}.dis`, methodology §14.4).
    **Update (2026-11-06): diagnosed and fixed for every app.** *Cause:* the clang driver passes `-fmath-errno`
    by default for `riscv32-unknown-elf` (`clang -###`); under it the front end emits `__builtin_fmaf` and `fmaf` as a
    call to the library `fmaf` (LLVM IR `call @fmaf`, an errno-writing call) instead of `@llvm.fma.f32`. Not
    `-ffp-contract` (it only governs `a * b + c`, which contracts to inline `fmadd.s` by default; that is why `ps_linear`
    and `ps_gemv_f32` never had calls) and not a RISC-V lowering problem (`llvm.fma.f32` → `fmadd.s`). Compile
    experiments: `build/tmp/p211/cc/` (`t.cpp`, `diag.sh`, `diag.out`). *Fix:* one shared line,
    `VX_CFLAGS += -fno-math-errno`, in every Makefile generated by `scripts/install_kernel_apps.sh` (no per-kernel macro;
    the earlier macro is kept but redundant). The flag removes only the errno contract of libm calls and enables no
    value-changing optimisation; musl's `fmaf` is `fmadd.s; ret` and its `sqrtf` is `fsqrt.s; ret`, so the inline and
    called forms compute the same IEEE results. Assembly of every `ps_*` kernel with vs without the flag (default
    config): `decode_attn_gqa` 251 → 0 calls (251 inline `fmadd.s`), `softmax_fixed` 30 → 0 (31 fused ops), `rmsnorm` loses the
    never-taken `call sqrtf` errno fallback after its inline `fsqrt.s` (−44 instructions per 4-row launch, 1.9–4.8 %
    fewer cycles), `rope` one commuted `fmul.s` (same instruction and cycle counts), every other kernel identical.
    *Evidence:* 83 runs on the old and the fixed build with the same host code (`results/<run>/ab.csv`): every output
    digest (FNV-1a 64 over the output bit patterns, new `PSDIGEST` line in `ps_common.h`) and every error statistic is
    identical; all 164 fixed-build batch runs with an validated twin match the error statistics of the validated
    logs exactly, as do the 81 old-build runs with PSCHECK lines, whose instructions and cycles also equal the validated rows
    (`results/<run>/vs_old.csv`); the TEST modules pass on the fixed build with the same statistics. *Effect:* software-exp softmax 18.7 → 11.1 cycles/element (1.68×), its exp
    25.3 → 16.4 cycles; gqa decode attention 2.33–2.40× fewer cycles at (16, 2, 128), 1.96–2.00× at (8, 2, 64),
    naive 1.39×; all other kernels (except rmsnorm) unchanged to the cycle. Numbers: methodology §14.1/§14.2
    addenda, `results/<run>/`. Status: measured.
31. **Stock `sgemm_tcu_wg_dxa` computes wrong output tiles at 16 cores for M = N ≥ 768, depending on DRAM timing
   .** At PS-BASE-S1-16c (TIER-CFG core + MUFU, 16 cores, L2 4 MiB, 8 HBM2 channels, +11 fixed DRAM latency) the
    stock app's own check fails: `-m 1024 -n 1024 -k 64` 128 wrong outputs (every printed error index lies in one 64 × 64
    output tile, rows 768–831, cols 256–319, reading index = row · N + col), `-m 768 -n 768` 256, `-m 2048 -n 1024` 256, `-m 1024 -n 1024 -k 128` 384; each wrong value is off by up
    to ~10⁶ ULP (not rounding), and a re-run (rep 1) gives the identical failure (deterministic for a fixed configuration).
    `-m 512 -n 512 -k 64` passes at 16 cores, and every shape run at 1, 4 and 8 cores passed (512² and 1024² × 64 at 1, 4 and 8 cores,
    512² × 128 at 4, 2048 × 1024 × 64 at 8; 768² was not run there) (`results/<run>/rows.jsonl`, kinds
    `tcu_peak`, `diag16`). Discrimination (`experiments/<run>.yaml` + two CLI runs, same rows file): at 16 cores
    TIER-CFG (default DRAM: 2 channels, 1 MiB L2, latency 0) passes with and without MUFU; PS-BASE-S1-16c fails with
    MUFU removed, with latency 0 (8 channels, 4 MiB), and with 2 channels + 1 MiB L2 at latency 11. So the failure is
    not caused by MUFU or by one memory knob: any change of the memory timing at 16 cores exposes it, which points to a
    timing-dependent ordering bug (a race between the DXA tile copy, its barrier and the WGMMA consumer, or in SimX's
    model of it) that the default timing happens to avoid. Not diagnosed further (stock code, third_party read-only).
    Consequences: PS-BASE-S1-16c's TCU peak is measured on the passing 512 × 512 × 64 shape only (labelled in
    `machine_params.csv`); any TCU/WGMMA result at 16 cores must check correctness per run (the stock app and
    `ps_linear -v tcu` do); runs at 4 cores, must re-check before using 16-core TCU numbers, and because the failure is
    timing-dependent, **every WGMMA/DXA run whose memory timing differs from the configurations tested here (P5 memory
    mechanisms included, at any core count) must check correctness per run**. Status: measured,
    cause unverified.
32. **At PS-BASE-S1-4c (NT = 32) the attention kernels read 1.8–5.6× their compulsory bytes from DRAM within one launch,
    and the 8-way L2 is the visible lever.** Measured per cold launch on (8, 2, 64) (`results/<run>/table_v0.csv`,
    column `dram_rd_over_ws` = DRAM reads × 64 B / compulsory bytes of q, K, V, out): decode `gqa_shared_kv` 1.77–3.42×
    (at REG-CFG, NT = 4: 1.01–1.06× E3 at T ≥ 256), prefill `lmem_tiled` 2.85–5.63×, `ps_mixed_attn` 2.09–2.55×; the
    linear layers stay at 1.00–1.01× except the M = 256 TCU GEMMs (3.5–4.2×, per-CTA strip re-reads). With
    16 L2 ways at the same 1 MiB the attention DRAM bytes (reads + writes) drop 15–42 % (decode), 13–26 % (prefill) and
    30–33 % (mixed), the DRAM reads alone 18–38 % / 18–31 % / 30–31 % (reads / compulsory: decode 1.10–2.81×, prefill
    2.07–4.41×), while a ¼-size L2 raises the bytes by only +0.5–4.5 % and the reads by +0.7–5.7 %
    (`results/<run>/traffic_l2knobs.csv`), so the extra traffic is **conflict misses**, not capacity (inference); a plausible mapping, not traced: the K/V rows of a
    (b, KV head) are 256 B runs at a 512 B stride (`kv_dim · 4`) in arrays aligned to large powers of two, which crowd a
    few L2 sets per bank (the L1 analogue is errata 29). The cycle effect is small (16 ways: decode attention −1.7 %,
    prefill −1.6 % time-weighted), consistent with these kernels not being DRAM-bound (§11.5 / README). The
    musl-libm element-wise kernels read far more than their data (rope 13–372×, silu_gate 35–39×; 16 ways halves rope's and
    cuts silu_gate's 3–5×); the likely source is instruction fetch of the soft-float library code through the L2 (I-cache
    misses), which was **not isolated** (no instruction-side DRAM counter is printed). Consequences: mechanism D
    (insertion) and A2 (way masks) experiments on attention must report the 8- vs 16-way baseline, because part of the
    "reuse" a policy recovers here is conflict misses that associativity alone removes; the earlier memhier calibration must not
    fit DRAM traffic of these kernels with a compulsory-traffic model. Status: measured (L2 8 ways / 1 MiB = the default `VX_config.toml:250-251`); cause unverified
    (inferred from the ways/size contrast, not traced).
- 32 addendum (2026-11-07): element-wise traffic under 16 L2 ways — rope is unchanged
  at B = 1 and roughly halved at B ≥ 4; silu_gate traffic is cut 2.7–5.5× (not "3–5×").

33. **Request flags die at the first cache miss; CSR writes repeat per SFU packet; DXA reads bypass L1.**
    (a) Stock SimX copies `MemFlags` from the LSU request into the L1 request (`sim/simx/mem/lsu_mem_adapter.cpp:45,140`,
    `mem_coalescer.cpp:274`), but every request a cache sends *downstream* is built from scratch with empty flags: fills
    (`sim/simx/mem/cache.cpp:1414-1419,1630-1636`), write-through stores (`:1206,1246,1556`) and write-backs
    (`:1056,1325,1679`); only the non-LLC AMO pass-through forwards the flag word. Any per-request attribute (a class, a
    priority) is therefore invisible at L2 and DRAM unless a patch copies it; patch mem-class copies the class only
    (methodology §12.4). (b) *(corrected)* A CSR instruction is executed once **per active
    SFU packet**: when `VX_CFG_NUM_SFU_LANES < VX_CFG_NUM_THREADS` the SFU `Dispatcher` already splits each warp
    instruction into one trace per packet that has an active lane, each carrying only that packet's thread mask
    (`sim/simx/dispatcher.cpp:31` `num_packets_ = NUM_THREADS / num_lanes`, `:77-131`; the SFU dispatcher is created with
    `VX_CFG_NUM_SFU_LANES`, `sim/simx/core.cpp:286`), and `SfuUnit` calls `CsrUnit::process` once per split trace
    (`sim/simx/sfu_unit.cpp:431`); `process` then loops over packets (`sim/simx/csr_unit.cpp:420-458`) but sees only one
    with active lanes. Idempotent CSR writes are unaffected, but a counting CSR such as `ps_mark` counts once per active
    packet trace, i.e. up to ⌈NUM_THREADS / NUM_SFU_LANES⌉ times per warp instruction. The per-`process()` guard that
    patch mem-class added (`ps_mark_written` in `csr_unit.cpp`, and its comment "one unit per warp instruction") is
    therefore **ineffective**; the patch is not re-cut for a comment, so this erratum is the correction of record. Kernels
    must write `ps_mark` from a single lane (`if (tid == 0)`, as `ps_decode_attn_gqa` and `ps_mixed_attn` do; tested by
    `tests/test_mem_class.py::test_marks_single_lane`). No measured count is affected: every configuration has SFU
    lanes = NUM_THREADS (`VX_config.toml:59,103`), so `num_packets_ = 1`, and both kernels mark from one lane. Carry-over
    to: count only the first packet trace of an instruction (`trace->sop`, `dispatcher.cpp:130`) or assert on it.
    (c) DXA GMEM reads
    enter the socket's memory arbiter as its lowest-priority input, next to the L1 caches, and go straight to the L2 (no
    L1 lookup) (`sim/simx/socket.cpp:206-233`). (d) The bit-field layout of `MemFlags` depended on
    `VX_CFG_EXT_DXA_ENABLE` (`sim/simx/types.h:439-475`): fields after the DXA sideband would move between builds; the
    patch fixes the class at bits 21–23 with a pad and asserts it. Status: checked.
    ** addendum (33b fixed in patch ps-counters).** Patch `0003-ps-counters` performs the `ps_mark` write only on the
    first SFU-packet trace of the instruction (`trace->sop`, `sim/simx/dispatcher.cpp:130`: `sop = trace->sop && (0 ==
    block_pid)`, so only the first emitted trace — the packet holding the first active lane — has it), so a warp
    CSR-write instruction now counts exactly one unit for any `NUM_SFU_LANES` and any active-lane pattern. Measured with
    the test app `kernels/ps_marklanes` (unguarded / last-lane-only / upper-half marks) at NT = 4 / SFU lanes = 2 and
    NT = 8 / SFU lanes = 2 (2 and 4 packets) and the control SFU = NT (`results/<run>/`, `tests/test_pstrace.py::
    test_marks_one_per_warp_instruction`). The three wrong comments of patch 0002 (`csr_unit.cpp` set_csr and process,
    `core.h` ps_mark) are rewritten in patch 0003. `kernels/common/ps_hint.h`'s header still says "per active SFU-packet
    trace" (left unedited on purpose: an edit there changes every `ps_*` app-source hash, methodology §12.11); the
    one-lane contract it states remains valid and is what makes one mark mean one (CTA, tile). Status: checked.
    ** note (MPM class 8).** Stock `VX_types.toml` reserves MPM class 8 (`VX_DCR_MPM_CLASS_RESERVED2`); no source in
    the pinned tree uses it (`grep -rn RESERVED2` finds only the toml line), so it is renamed `VX_DCR_MPM_CLASS_PHASESHIFT`.
    The MPM window holds 29 user counters per class (CSR ids 3..31); the 256 PhaseShift counters are paged through the
    upper bits of the 10-bit class field of the DCR tag (`sw/runtime/common/perf.cpp` `vx_device_mpm_query`: tag =
    class << 22 | csr << 16 | core; `sim/simx/socket.cpp` strips the core, `core.cpp` takes class = tag >> 6), methodology
    §12.9. Status: checked.

34. **SimX caches hold data, so keeping L2 lines across launches needs host-write invalidation; the L2 init walk
    costs `sets_per_bank` cycles per launch.** `CacheBank` sectors carry the line bytes (`sector_t::data`,
    `sim/simx/mem/cache.cpp:140-155`, ps tree), so a line kept across the per-launch reset (persist_l2, methodology
    §5.2) would return stale data after a host write to device memory. Measured negative control (`results/<run>`,
    `PS_PERSIST_NO_INVAL=1`): ps_llama teacher-forced stories260K aborts in its first launches with
    `misaligned memory access: addr=0x1` (stale kernel-argument / buffer lines), while with the invalidation hook in
    the runtime's `dram_write` path (`sw/runtime/simx/vortex.cpp`, ps patch 0004) the same run PASSES with logits
    bit-identical to the cold run (446 L2 lines dropped over 432 launches). Every host write to device memory in the
    SimX runtime goes through that one hook (command-processor DMA writes, uploads, completion words), which is what
    makes the invalidation complete. Also: `Cache::Impl::reset()` sets `init_cycles_ = sets_per_bank` at every launch,
    i.e. each launch starts with an L2 init walk of `sets_per_bank` cycles during which the L2 accepts nothing (stock
    behaviour on both trees; persist mode skips it). Status: checked (read + measured).

35. **Fill semantics for mechanism D: SimX fills insert at MRU, RTL fills do not touch PLRU, and in both the post-fill
    replay through the pipeline IS a lookup hit (it promotes), while fill-forwarded reads are not.** RTL
    (`vortex-trees/ps/hw/rtl/cache`, = pristine): the PLRU store is written only on `lookup_valid && lookup_hit`
    (`VX_cache_repl.sv:111,132`, errata 15), with `lookup_valid = do_lookup_st1 && ~pipe_stall`
    (`VX_cache_bank.sv:563`) and `do_lookup_st1` = a read or write with `is_creq` (`:238-240`), where
    `is_creq = creq_enable || replay_enable` (`:466`): an MSHR replay that goes through the pipeline is a lookup and
    hits, so it advances PLRU to that way. The fill itself is `is_fill`, not a lookup (no PLRU write). The leading
    plain reads of a filled chain are completed by the fill-forward drain (`fwd_head`, `:1084`) at the response port
    without entering the pipeline, so they do not update PLRU. SimX mirrors the replay/forward split
    (`sim/simx/mem/cache.cpp` `processForward`, comment "Forwarded reads skip the replacement-state update"; the
    Replay case calls `update_lru(hit)`), but `processFill` sets `lru_ctr = 0` (MRU) — and, a trap for any
    re-implementation, `victim_line.reset()` also zeroes `lru_ctr` before the install, so "leave the state alone"
    must save the victim way's age first (patch 0005 does; the first draft did not and measured rtl_like_fill
    identical to stock). Measured (dev debug counters, removed before the patch; `results/<run>`): at the L2 of
    `ps_wkv_micro -w 512 -c 1024 -r 4` PS-BASE-S1-4c, per bank ≈ 12,290 forwarded reads vs 16 pipeline replays for
    ≈ 12,296 fills, i.e. in SimX virtually every L2 read fill drains by forwarding, so the fill's own state write is
    the only recency update a missing line gets. Consequence: SimX's `rtl_like_fill` baseline (patch 0005, insert bit
    4: the filled way keeps its pre-fill age, the oldest of the set) behaves as LRU insertion for every class
    (Qureshi et al.'s LIP [C33]), and the RTL baseline is expected to be close to it; D's gain is reported against
    both (methodology §14.D). The RTL PLRU is a tree PLRU, SimX's "PLRU" is an exact age counter (per-line
    `lru_ctr`), so `rtl_like_fill` reproduces the RTL's *policy* (fills do not promote), not its exact victim order.
   read + measured.

36. **Header edits do not rebuild SimX's per-run objects, and the stock DRAM controller cannot take a class in `source_id`
   .**
    - *(a) Stale objects.* `make runtime-simx` (what vxrun/blackbox run) builds `libsimx.so` with
      `DESTDIR=build/vortex-*/sw/runtime`, so the objects land in `sw/runtime/obj/`. The `-include $(DEPS)` of
      `sim/simx/Makefile` (`sim/simx/Makefile:158`, `:272`), however, lists the `.d` files derived from the standalone
      `OBJ_DIR` (`sim/simx/obj/`). Make
      therefore has no header dependencies for the objects it actually links: editing only a header (here `dram_sim.h`,
      which changes `DramSim`'s size) relinks objects compiled against the old layout. Observed as
      `malloc(): unaligned tcache chunk detected` on the first run after the edit (`memory.o` stale). Workaround used in
     : delete `sw/runtime/obj/**/*.o` after every header edit (`results/<run>/rebuild.sh`; ccache keeps the full
      rebuild cheap). Steps that edit only `.cpp` files are not affected.
    - *(b) Out-of-range statistics index.* Ramulator 2's `GenericDRAMController` sizes its per-source read row-hit
      statistics by `frontend->get_num_cores()` (`generic_dram_controller.cpp:75`), which is 1 for the GEM5 frontend
      DramSim uses, and indexes them with `source_id` whenever `source_id != -1` (`generic_dram_controller.cpp:262`). DramSim passes 0, so any other value (for example, a request class) would
      write out of bounds. Patch 0007's `PsTwoClass` sizes them for the 8 classes and bounds-checks the index.
   read + measured.

37. **SimX can corrupt a private stack round-trip in a few lanes of one warp, depending on memory timing.**
    - *Symptom.* `ps_rope -b 64 -d 512 -m 128 -H 64 -S 64 -N 1 -l 0` (seed 1) at PS-BASE-S1-4c fails
      deterministically: 14 q elements, worst 2.4e-2 of the pair norm. The q digest `3c77b532847ed215` is identical on
      the ps and base trees.
    - *What goes wrong.* Lanes 1–7 of one warp (row 26, head 2) compute musl `powf(10000, hd/64)` with the
      `log2_inline` term `r = z*invc - 1` equal to 0. This reproduces their outputs bit for bit
      (`results/<run>/powf_r0_model.py`). Every other lane is bit-exact.
      - r lives in s1:s2, which `__adddf3`/`__muldf3` save and restore through the stack.
      - The corrupted lanes are the active lanes of one 32-byte dcache word of the interleaved stack slot. Lane 0 of
        that word is masked, having taken powf's y = 0 exit.
    - *Not libm or the kernel.* The same arguments are correctly rounded on a single-thread probe, and row 26 alone
      (B = 1) passes.
    - *Timing dependence.* Only the DRAM latency changes (11 → 0, PS-BASE-S1-4c-lat0), and the launch becomes
      bit-exact. E1 and REG-CFG are also bit-exact, with one common digest.
    - *Where to look (candidates, LSU or cache).*
      - LSU/core side: `vortex-trees/ps/sim/simx/lsu_unit.cpp:149` (request build: tid-stable lane slots, size-0 holes
        for inactive threads); the response lane mapping under a partial thread mask; store→load ordering of a
        thread's stack round-trip.
      - The stack interleave map, `vortex-trees/ps/sim/simx/stack_interleave.h:21`: lane L of a warp holds word L of
        a frame slot.
      - Cache side: `vortex-trees/ps/sim/simx/mem/cache.cpp:632` (`has_ordered_reqs`) and `cache.cpp:2168`
        (write-through write miss). The base tree has the same code at
        `vortex-trees/base/sim/simx/mem/cache.cpp:1592`.
      - D2/D3 (`results/<run>/batch_d2.out`, `batch_d3.out`) ran L1 + L2 shadow memories on a failing launch
        and found **0 violations** in every bank. That was read as pointing to the LSU/core side first.
      - **Superseded (D6/D7, 2026-11-10).** The D2/D3 L1 shadow skipped exactly the bytes a younger store had
        overwritten, which is the WAR case. With a previous-version check, D6 (LSU) and D7 (L1) catch the event: an L1
        `replay-rsp` returns a younger store's bytes to a load the bank admitted first. The cause is the CacheBank
        ordering hazard in item 38, not the LSU.
    - *Status.* An upstream SimX memory-path defect. For the earlier `-b 8 -S 152` launch it is located in the cache bank
      (item 38). For this item's seed-1 launch it is **inferred**: D10 passes with the item-38 candidate fix, but
      the failing load itself was not traced.
    - *Workaround.* None in code. Every number must come from a run whose output check PASSED, as the harness already
      enforces. That is necessary but not sufficient: corruption of this class can stay within tolerance and PASS.
      A2-¼ rows `70733256` (rope `-b 66`), `76fcd952` (silu `-b 40`) and `0f718e27` (silu `-b 50`) passed yet
      differ from their unmasked twins.
      - A failing element-wise check at a configuration whose tolerance derivation holds is a simulator suspect.
        Example: seed 1 here, with error 2.4e-2 against a derived bound of 2.7e-5.
      - Not every failure is one. The large-S rope overshoots (S 4096–8192, about 1.4e-4 against tol 1e-4) are a
        documented numerics bound (`results/<run>/README.md` item 6), not this defect.
    - *Related A2 finding, for one launch only* (`ps_rope -b 8 -S 152`). the earlier ¼-mask failure reproduces exactly
      with no mask on a stock 256 KiB 2-way L2 (`results/<run>/rows.jsonl` R4/R5; run 09). Same symptom class,
      likely the same defect.
      - The other six ¼-mask rope failures were not tested unmasked at equivalent capacity/associativity:
        non-finite q at B 24/40/64, and O(1) q and k errors at B 68/256.
      - The silu ½-mask failure was later reproduced unmasked (R7, stock 4-way 512 KiB, identical digest). The
        other six rope launches remain untested, so A2 as a whole is not shown to be mask-independent.
    - *Resolved (item 38's applied fix).* The seed-1 launch PASSES on the fixed ps and base trees: q digest
      6a18d826, 24,970,912 instrs. The 3 A2-¼ rows that passed but differed now equal their unmasked twins.
   measured (`results/<run>/README.md`); (`results/<run>/README.md`).
38. **SimX CacheBank loses ordering for a replay while it is in the bank pipe: a younger store can overtake an older
    load (WAR), and the mirror RAW case (open) was confirmed and fixed.** Stock code, identical in base and ps; pointers are to the
    pristine clone `sim/simx/mem/cache.cpp`.
    - *Mechanism (WAR, confirmed).*
      - `MSHR::dequeue()` (`sim/simx/mem/cache.cpp:596`) frees an entry when `processInputs` pushes its replay into
        the bank pipe (`cache.cpp:783`), not when the replay leaves it. From then on `lookup()`, `has_ordered_reqs()`
        and `has_pending_fill()` cannot see that access.
      - Write-through stores skip admission-time coalescing (`is_wt_store`, `cache.cpp:916`). A store admitted while
        the chain's fill rides the pipe is therefore ahead of the chain's replays.
      - At pipe exit the hit path defers a write only if `mshr_.lookup(...)` finds the chain (`cache.cpp:1501`). If the
        chain's only entry was already dequeued, the store merges as a plain hit. The older load's replay then reads
        the merged sector (`cache.cpp:1467`) and returns the younger value.
      - The fill-forward drain (`processForward`, `cache.cpp:1098`) answers the load first, but only when `crsp_sent_`
        is clear and `core_rsp_out` is not full. So the outcome depends on response-port timing.
      - Second route, every level including the write-back L2: a store admitted while a pure-load chain is draining is
        not coalesced either (`must_order` false, `cache.cpp:924`).
    - *Observed.* D7 and verification run CHK1_nofix_2way (`results/<run>/logs_dbg/`): 3 L1 WAR events in one
      `ps_rope -b 8 -S 152` launch on a 256 KiB 2-way L2 (= the A2 ¼ geometry). One of them, a callee-saved restore
      `lw s0, 0x2c(sp)` in the `__muldf3` epilogue, corrupts sin in warp 2, lanes 8–9. It causes the A2 failures of
      rope ¼ `-b 8 -S 152` and silu ½ `-b 64 -n 1792` (both reproduced bit-identically without any mask, R4/R5/R7). It
      is the inferred cause of item 37's seed-1 failure (D10 passes with the fix). The other 6 A2 rope ¼ failures
      are not yet tied to it.
    - *Candidate fix (superseded).* `results/<run>/instr/add_war_fix.py` (env `PS_FIX_WAR`) deferred a
      store hit on a sector with an in-pipe replay. It covered WAR/WAW only. It also deferred stores behind YOUNGER
      in-pipe replays, which can itself create a RAW.
    - *RAW twin: confirmed.* A younger load hit can leave the pipe ahead of an older write's in-pipe replay (a
      deferred merge). The directed L2 sweep (`results/<run>/README.md` TEST 2) makes stock SimX return the
      pre-store sector:
      - 287 wrong loads at L2 write-back latency 4 and 8, from 47 ld-over-st (RAW) and 156 st-over-ld (WAR) checker
        violations;
      - the trigger is a store hit while a chain of about 6 or more same-sector misses drains.
      The random stress test also shows it at an L1 with a 4-deep pipe. In the same runs, a second WAR route appears at
      the write-back L2 (156 st-over-ld): a store admitted while a pure-load chain drains.
    - *Applied fix (2026-11-10/11: ps patch `0014-cache-ordering` + the identical base patch
      `patches/vortex/base/0014-cache-ordering.patch`; methodology §18.6).*
      - Rule: same-sector accesses with at least one store or atomic perform in admission order. Store/store pairs
        are byte-precise; pairs with a load are sector-precise, because loads carry byteen = 0.
      - *Replay-entry gate:* the oldest ready replay does not enter the pipe while a conflicting Core/AmoProbe request
        is in the pipe. The stock hit-path rule then sees it and defers that request in admission order.
      - *Forward gate:* no fill-forward while an older conflicting request is in the pipe.
      - No deadlock: only the input stalls, for at most the pipe latency plus output stalls.
      - Read-only order checker `PS_ORDER_CHECK=1` (`PSORDER` lines).
      - Evidence:
        - Same-binary pair (dev copy): fix off = stock R4 bit for bit (FAIL, 3 L1 violations); fix on = PASS, digest
          6cd6d257, 0 violations.
        - Unit stress test and L2 RAW sweep: 0 wrong loads, 0 violations, 0 deadlocks on both trees.
        - seed 1 PASSES on ps and base: 24,970,912 instrs = the correct lat0 path, which confirms item 37's
          inference.
        - R1/R4/R5/R7 PASS.
        - `noop_regression.sh --all` 34/34; compat set 117 pass + 39 rejected-by-design.
        - All 8 A2 failures PASS, and all 11 A2 rows equal their unmasked twins bit for bit.
      - Cost: 278 of the 293 re-validated gate rows are cycle-identical; max |Δ| 0.47 % (rope/silu); no digest change;
        no gate number moves.
      - Not covered: the AMO-passthrough "vanished sector" refill path. It is stock behaviour, unused by current
        kernels, and the checker would flag it.
      - Deadlock exception on the same path: a vanished-sector refill at the pipe head waits on `mshr_.full()`. Only a
        replay dequeue, which is input-side, frees a slot, so gate G can hold that dequeue behind a conflicting Core
        request and deadlock. It is unreachable while `VX_CFG_EXT_A_ENABLED` = 0 (every current configuration) and
        must be fixed before any A-extension configuration is used.
    - *Consequence (as applied).*
      - Every run key changed: ps tree fingerprint `eadd393c…`, base `6a85c358…`.
      - "base" now means upstream + infra 0000–0002 + this correctness fix.
      - Rows measured stay valid as rows of the unfixed trees and are never reused across the fingerprint
        change.
   measured + code read (`results/<run>/README.md`); fixed + measured
    (`results/<run>/README.md`).

39. **Stock `sgemm_tcu_wg_dxa_mcast` fails to launch on the base tree at PS-BASE-S1-4c (2026-11-14).**
    - *Symptom.* `vxrun` status `failed`, exit 2, run key `e18c53db…` (`results/<run>/rows.jsonl`, base tree, no
      arguments, PS-BASE-S1-4c); the ps tree fails the same way (`results/<run>/rows.jsonl`, key `fdf8aa5b…`). The
      host reports launch error 3; the cause is unverified.
    - *Scope.* Pre-existing and independent of every ps patch, because the base tree (upstream + infra patches) fails too.
      The cause was not diagnosed; multicast cost was measured with `dxa_copy_mcast` and
      `sgemm2_dxa_mcast` instead. The edge-decode study uses 1 and 4 cores and no multicast.

40. **musl `sinf`/`cosf` are not correctly rounded: results within 2^-10 ulp of a rounding tie can be the neighbouring float
    (toolchain, not Vortex; 2026-11-14).**
    - *Evidence.* The device RoPE table of each tested window has exactly one entry that differs from the correctly rounded
      float, each at 0.5002-0.5007 ulp: stories15M index 3894, stories260K 2398, llama8b_s8 1043 and 6678
      (`results/<run>/README.md:40`, item 2; an independent numpy long-double recomputation with those four entries patched
      reproduces all four device FNV-1a digests).
    - *Consequence.* Host checks of device sin/cos tables must use a tolerance of 0.5 + 2^-9 ulp, not bit equality with a
      correctly rounded host value. Paper numerics note (not a defect of the RoPE-table kernel).
