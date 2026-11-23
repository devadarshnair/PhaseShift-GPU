"""PhaseShift analytical memory-hierarchy model (specification: `docs/methodology.md` §9).

Turns one serving iteration (an opgen op trace) into cycles on a machine (roofline.Machine) with an
L1 / LMEM / L2 / DRAM hierarchy (`Hier`) under a PhaseShift knob vector (`Knobs`):

  A1 `lmem_use`         LMEM bytes a CTA uses for tiles -> tile size (re-read factor) and CTAs per core (occupancy);
  A2 `a2_weight_ways`   L2 way mask: weights fill only these ways, every other class the remaining ways;
  B  `b_two_class`      DRAM arbitration gives decode-al requests non-preemptive priority over bulk ones;
  C  `dxa_depth`        DXA (async-copy) prefetch depth in tiles for tile streams (0 = plain LSU loads);
  D  `kv_insert`        L2 insertion of streamed KV: "mru" (normal), "lru" (insert at LRU) or "bypass" (no allocate).

Model in one paragraph (each item is a numbered modeling choice M1–M14 of methodology §9, which also lists the
expected failure mode of every approximation): every kernel launch (one op, or one fused attention block) reads its
compulsory bytes (opgen) per class plus tiling re-reads; each read stream has a reuse window (distinct bytes, by
class, touched between two references to the same byte). A reference hits in L2 iff its window fits the capacity
the knobs leave to its class (steady-state LRU footprint-vs-capacity cliff). Cross-launch reuse exists only when
`persist_across_launch` is True (False on Vortex: SimX resets every cache at each launch, methodology §2
consequence 4, errata 12). DRAM time is bytes/(η·BW) bounded below by Little's law on the outstanding requests of
each issue path (LSU demand: L1 MSHRs and warps; DXA: depth × tile bytes), with a non-preemptive-priority M/D/1
queueing wait per class (B). Launch time = max(compute, memory) (+ the non-overlapped part of single-buffered tiles)
+ one unhidden miss latency + launch overhead. All parameters are fields of `Hier` (calibrated).

Units: bytes, FLOP, cycles of the machine's core clock (seconds only in `__call__`, for the scheduler).
Pure Python + standard library (NumPy only in the queue simulation used by the tests).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from model import opgen, roofline

PS_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("weight", "act", "kv", "kv_stream")
# weight    = model weights (linear layers, norm vectors, embedding rows);
# act       = activations (short-lived tensors between ops);
# kv        = KV-cache bytes that are re-used inside a launch or written this step: KV read by prefill-chunk
#             attention (re-read by every query tile) and every KV write;
# kv_stream = KV-cache bytes read by decode attention that were written in an earlier step (touched exactly once per
#             step; the target of mechanism D). The decode token's own K/V, written this step, is class kv.

# ================================================================================================= parameters


@dataclass(frozen=True)
class Hier:
    """Memory-hierarchy parameters of one machine (all per-core values are per core / SM). Every value carries a
    source or "assumption" in `sources`; calibrates eta_dram, eta_comp, latencies and launch_overhead."""

    name: str
    cores: int
    warps: int                    # resident warps per core
    line: int                     # bytes per memory request = cache sector = MEM_BLOCK
    l1_bytes: float               # per-core data cache (GPU-like: combined L1/shared array, see l1_lmem_shared)
    l1_mshr: int                  # per-core outstanding L1 misses
    l1_lat: float                 # L1 hit latency (cycles)
    lmem_bytes: float             # per-core scratchpad capacity usable by A1
    l1_lmem_shared: bool          # True: LMEM is carved out of the l1_bytes array (NVIDIA-style); False: separate.
    #                               Recorded only: L1 is not modelled as a general cache (M11), so the carveout's
    #                               L1 side has no effect on the cost
    l2_bytes: float               # whole-chip L2 (math.inf allowed in tests)
    l2_ways: int
    l2_banks: int
    l2_mshr: int                  # per bank
    l2_lat: float                 # L2 hit latency beyond L1 (cycles)
    dram_lat: float               # unloaded DRAM latency beyond L2 (cycles)
    dram_channels: int
    l2_bytes_per_cycle: float     # L2 bandwidth (all banks)
    mlp_per_warp: float = 2.0     # outstanding lines per warp (LSU path)
    warps_per_cta: int = 4        # warps of one CTA (A1 occupancy unit)
    threads_per_warp: int = 32    # SIMT width (Vortex NT, GPU warp)
    acc_per_thread: int = 128     # fp32 output accumulators one thread can hold in registers (caps GEMM/attention tiles)
    gemm_tk: int = 32             # K depth of a GEMM tile (tile footprint = Tk·T·(q_a + q_w))
    l1_tile_frac: float = 0.5     # share of the L1 that all co-resident CTAs' implicit tiles (no LMEM) together use
    act_tail_frac: float = 0.5    # share of the producer launch's traffic that follows a given output byte
    eta_dram: float = 1.0         # achievable / peak DRAM bandwidth (1 measure it)
    eta_comp: float = 1.0         # achievable / peak compute
    launch_overhead: float = 0.0  # cycles per kernel launch (fits it; 0 = not modelled yet)
    overlap: float = 1.0          # ω ∈ [0, 1], share of the shorter of compute / memory time hidden behind the longer
    #                               (M3). 1 = the model exactly (full overlap; single-buffered LMEM tiles
    #                               hide 1 − 1/n_cta of it); 0 = compute and memory fully serialized
    flush_untimed: bool = False   # True (SimX): dirty lines left at the end of a launch are written back untimed
    persist_across_launch: bool = False
    hit_model: str = "cliff"      # "cliff" (LRU) | "proportional" (hit = min(1, capacity/window))
    cliff_ways: float = 1.0       # cliff = ramp this many L2 ways wide, centred on the capacity (0 = ideal step)
    sources: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if self.hit_model not in ("cliff", "proportional"):
            raise ValueError(f"unknown hit_model {self.hit_model!r}")
        for k in ("cores", "warps", "line", "l2_ways", "l2_banks", "l2_mshr", "l1_mshr", "dram_channels"):
            if getattr(self, k) < 1:
                raise ValueError(f"{k} must be >= 1")
        if self.l1_bytes < 0 or self.l2_bytes < 0 or self.lmem_bytes < 0:
            raise ValueError("capacities must be >= 0")
        if not 0.0 <= self.overlap <= 1.0:
            raise ValueError("overlap must be in [0, 1]")

    @property
    def hit_lat(self) -> float:
        """Latency of a request that hits in L2 (L1 lookup + L2)."""
        return self.l1_lat + self.l2_lat

    @property
    def miss_lat(self) -> float:
        """L0: unloaded latency of a request that goes to DRAM (L1 + L2 + DRAM)."""
        return self.l1_lat + self.l2_lat + self.dram_lat


@dataclass(frozen=True)
class Knobs:
    """One PhaseShift configuration c (methodology §1). Defaults = the stock configuration (no LMEM tiles, no DXA,
    normal insertion, shared L2, FR-FCFS-like single-class DRAM queue)."""

    lmem_use: int = 0                  # A1: LMEM bytes per CTA (0 = no LMEM tiling)
    a2_weight_ways: int | None = None  # A2: L2 ways weight fills may use (None = no way mask)
    b_two_class: bool = False          # B: decode-critical class has DRAM priority
    dxa_depth: int = 0                 # C: DXA tiles in flight per CTA (0 = LSU loads, single buffer)
    kv_insert: str = "mru"             # D: "mru" | "lru" | "bypass" for the demoted KV classes
    kv_insert_scope: str = "decode"    # D applies to decode-read KV only ("decode", phase-aware) or to all KV ("all")

    def validate(self, h: Hier) -> "Knobs":
        if self.kv_insert not in ("mru", "lru", "bypass"):
            raise ValueError(f"kv_insert must be mru/lru/bypass, not {self.kv_insert!r}")
        if self.kv_insert_scope not in ("decode", "all"):
            raise ValueError(f"kv_insert_scope must be decode/all, not {self.kv_insert_scope!r}")
        if self.lmem_use < 0 or self.lmem_use > h.lmem_bytes:
            raise ValueError(f"lmem_use {self.lmem_use} outside [0, {h.lmem_bytes}]")
        if self.dxa_depth < 0 or int(self.dxa_depth) != self.dxa_depth:
            raise ValueError("dxa_depth must be a non-negative integer")
        if self.dxa_depth > 0 and self.lmem_use <= 0:
            raise ValueError("DXA prefetch (dxa_depth > 0) lands in LMEM: it needs lmem_use > 0")
        if self.a2_weight_ways is not None and not 1 <= self.a2_weight_ways <= h.l2_ways - 1:
            raise ValueError(f"a2_weight_ways must be in [1, {h.l2_ways - 1}] (or None)")
        return self

    @property
    def label(self) -> str:
        a2 = "-" if self.a2_weight_ways is None else str(self.a2_weight_ways)
        return (f"A1={self.lmem_use} A2={a2} B={int(self.b_two_class)} C={self.dxa_depth} "
                f"D={self.kv_insert}{'' if self.kv_insert_scope == 'decode' else '/all'}")


# ------------------------------------------------------------------------------------------------- machines

VX_CONFIG = PS_ROOT / "third_party" / "vortex" / "VX_config.toml"
PROFILES = PS_ROOT / "third_party" / "vortex" / "profiles"
DRAM_ARRAY_NS = 50.0
"""Unloaded DRAM array latency in ns: the midpoint of the default SimX DRAM's idle open-row read (14 cycles) and
row-conflict read (26 cycles) at the 400 MHz default clock (docs/simulation.md:203-205): 20 cycles / 400 MHz = 50 ns.
SimX holds array timings in ns (same lines), so other clocks scale the cycle count."""


def _clog2(x: float) -> int:
    return max(0, math.ceil(math.log2(x))) if x > 1 else 0


def vortex_hier(cores: int, threads: int = 32, warps: int = 8) -> Hier:
    """Vortex placeholder hierarchy at the tier configuration (NT = 32, NW = 8, --l2cache), from the VX_config.toml
    defaults (line numbers in `sources`; tests/test_memhier.py re-reads the literals). UNMEASURED: replace
    latencies and efficiencies with measured/fitted values."""
    line = 64
    l1 = 16384
    lmem = 1 << _clog2(max(16384, threads * 2048))
    l2_banks = 1 << _clog2(min(2 * cores, 16))
    dram_lat = DRAM_ARRAY_NS * 400e6 * 1e-9 + 0
    return Hier(
        name=f"vortex-S1-{cores}c", cores=cores, warps=warps, line=line, l1_bytes=l1, l1_mshr=16,
        l1_lat=2 + max(0, _clog2(l1) - 14), lmem_bytes=lmem, l1_lmem_shared=False, l2_bytes=1048576, l2_ways=8,
        l2_banks=l2_banks, l2_mshr=16, l2_lat=4, dram_lat=dram_lat, dram_channels=2,
        l2_bytes_per_cycle=l2_banks * line, threads_per_warp=threads, acc_per_thread=16,
        flush_untimed=True, persist_across_launch=False,
        sources={
            "line": "VX_config.toml:66 MEM_BLOCK_SIZE = 64 (L1 line = MEM_BLOCK :73; L2 sector = MEM_BLOCK :75)",
            "l1_bytes,l1_mshr": "VX_config.toml:220 DCACHE_SIZE = 16384, :229 DCACHE_MSHR_SIZE = 16",
            "l1_lat": "VX_config.toml:237 DCACHE_LATENCY = 2 + max(0, clog2(size) - clog2(16384))",
            "lmem_bytes": f"VX_config.toml:280-288 LMEM = 2^clog2(max(16384, NUM_LSU_LANES x 2048)), LSU lanes = NT = "
                          f"{threads} (:107)",
            "l2_bytes,l2_ways,l2_mshr,l2_lat": "VX_config.toml:250,251,255,256 (1 MiB, 8 ways, 16 MSHR/bank, 4 cycles)",
            "l2_banks": "derived: VX_config.toml:261 pow2(clog2(min(L2_NUM_REQS, 16))), L2_NUM_REQS = NUM_SOCKETS x "
                        "L1_MEM_PORTS (sim/simx/constants.h:66,80), socket size 1 (:6), L1_MEM_PORTS = min(DCACHE banks,"
                        " 2 memory banks) = 2 (:247; NT = 32 gives 4 dcache banks, :225,245). Formula-verified "
                        "for VX_CFG_VM_ENABLE = false (:22) and SOCKET_SIZE 1; with VM on the page-walker slot "
                        "adds one L2 request (constants.h:74), e.g. 16 banks at 4 cores. Not measured on a build",
            "acc_per_thread": "assumption: 16 of the 32 RV32F floating-point registers per thread hold accumulators "
                              "(the rest hold operands and addresses); the GEMM kernel fixes the real value",
            "dram_lat": "assumption: 50 ns array midpoint (DRAM_ARRAY_NS, docs/simulation.md:203-205) x 400 MHz "
                        "(:85) + PLATFORM_MEMORY_LATENCY 0 (:95)",
            "dram_channels": "VX_config.toml:80 PLATFORM_MEMORY_NUM_BANKS = 2 (one channel per bank)",
            "l2_bytes_per_cycle": "assumption: one 64 B sector per L2 bank per cycle",
            "warps": f"tier NW = {warps}", "mlp_per_warp": "VX_config.toml:116 '~2 in-flight loads per warp'",
            "flush_untimed": "errata 12: the SimX flush is untimed and outside kernel cycles",
            "persist_across_launch": "False: SimX resets every cache at each launch (methodology §2 cons. 4)",
            "others": "assumptions explained in methodology §9 (warps_per_cta, gemm_tk, l1_tile_frac, act_tail_frac, "
                      "eta_dram = eta_comp = 1, launch_overhead = 0)"})


def psbase_hier(cores: int, path=None) -> Hier:
    """PS-BASE-S1-<cores>c hierarchy (methodology §11): the vortex_hier structure with
    PS-BASE's sizes (L2 = 256 KiB per core, 0.5 DRAM channel per core) and the MEASURED unloaded latencies of
    results/<run>/machine_params.csv, split so that the model's hit and miss latencies equal the measured ones
    (modeling choice): l1_lat = measured L1 load latency, l2_lat = measured L2 hit - L1, dram_lat = measured random-order
    DRAM-miss latency L0 - L2 hit (so L_hit = l1 + l2 = the measured L2 hit and L0 = l1 + l2 + dram = the measured L0).
    The measured load latencies include the LSU/coalescer path that the placeholder's 2/4-cycle cache latencies omit.
    The remaining structural values (L1, MSHRs, L2 bank count, LMEM) are inherited from vortex_hier. Its L2 bank count
    pow2(min(2 * cores, 16)) equals SimX's at PS-BASE (8 / 16 / 16 at 4 / 8 / 16 cores: L2_NUM_REQS = sockets x
    min(4 dcache banks, DRAM channels), sim/simx/constants.h:59-66); the -DVX_CFG_L2_NUM_BANKS=1 the app build prints
    comes from gen_config.py and does not reach SimX. Everything else
    (efficiencies, MLP, overlap) stays uncalibrated."""
    import csv
    import dataclasses
    from pathlib import Path
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "results" / "P3.1" / "machine_params.csv"
    with open(path) as f:
        row = next(r for r in csv.DictReader(f) if r["config"] == f"PS-BASE-S1-{cores}c")
    l1, l2, l0 = float(row["l1_load_lat"]), float(row["l2_load_lat_est"]), float(row["dram_load_lat_rand"])
    base = vortex_hier(cores)
    src = dict(base.sources)
    src.update({"l2_bytes,l2_ways,l2_mshr,l2_lat": f"PS-BASE: L2 = {row['l2_bytes']} B (256 KiB per core), 8 ways, 16 "
                                                    f"MSHR/bank; l2_lat measured (below)",
                "dram_channels": f"PS-BASE: {row['dram_channels']} HBM2 channels (0.5 per core)",
                "l1_lat,l2_lat,dram_lat": f"measured (results/P3.1/machine_params.csv, PS-BASE-S1-{cores}c): L1 load "
                                          f"{l1}, L2 hit {l2} (estimate), L0 {l0} cycles; split as l1 = L1, l2 = L2 - "
                                          f"L1, dram = L0 - L2"})
    return dataclasses.replace(base, name=f"psbase-S1-{cores}c", l2_bytes=int(row["l2_bytes"]),
                               dram_channels=int(row["dram_channels"]), l1_lat=l1, l2_lat=l2 - l1, dram_lat=l0 - l2,
                               sources=src)


GPU_L2_BW_ANCHOR = 4
"""GPU-like effective L2 bandwidth multiplier, ANCHORED TO THE ROOFLINE: an assumption, not a measurement.
The unanchored value (one 64 B sector per bank per cycle, 32 banks copied from the profiles' power-of-two DRAM
interleave) left Llama-3.1-8B prefill at S = 512 L2-bound at ≥ 4.35× its roofline bound for every knob vector, while
vendor GEMMs of that size run near compute-bound. 4 is the smallest integer for which the best knob_grid vector is
within 1.5× of max(FLOP bound, DRAM bound) on both h100-like (1.24×) and a100-like (1.10×); 3 gives 1.56× on h100.
It is an *effective* bandwidth that also stands for tile-reuse mechanisms the model lacks (larger rectangular
warpgroup tiles, cluster multicast, L1 hits on tile loads). Methodology §9 M12; test_gpu_prefill_roofline_anchor."""


def gpu_hier(profile: str, machine: roofline.Machine | None = None, l2_bw_anchor: float = GPU_L2_BW_ANCHOR) -> Hier:
    """GPU-like hierarchy from a Vortex profile (third_party/vortex/profiles/<profile>.json): sizes from the profile,
    latencies from the same VX_config.toml formulas the profile would be simulated with, MSHRs by the bandwidth-delay
    rule (assumption, methodology §9 M12), effective L2 bandwidth anchored to the roofline (GPU_L2_BW_ANCHOR;
    pass l2_bw_anchor=1 for the unanchored value). persist_across_launch = True (real GPUs keep L2 across launches)."""
    p = json.loads((PROFILES / f"{profile}.json").read_text())
    core, chip, plat = p["core"], p["chip"], p["platform"]
    m = machine or roofline.machines()[f"{profile}-like"]
    clk_mhz = plat["VX_CFG_PLATFORM_CLOCK_RATE"]
    line = 64
    l1 = core["VX_CFG_DCACHE_SIZE"]
    l2 = chip["VX_CFG_L2_SIZE"]
    banks = chip["VX_CFG_L2_NUM_BANKS"]
    l1_lat = 2 + max(0, _clog2(l1) - 14)
    l2_lat = 4 + max(0, _clog2(l2) - 20)
    fixed = plat.get("VX_CFG_PLATFORM_MEMORY_LATENCY", 190)
    dram_lat = fixed + DRAM_ARRAY_NS * clk_mhz * 1e-3
    l0 = l1_lat + l2_lat + dram_lat
    need = math.ceil(2 * m.dram_bytes_per_cycle * l0 / line)     # outstanding lines covering 2x the BW-delay product
    src = f"third_party/vortex/profiles/{profile}.json"
    return Hier(
        name=f"{profile}-like", cores=chip["VX_CFG_NUM_CORES"], warps=core["VX_CFG_NUM_WARPS"], line=line,
        l1_bytes=l1, l1_mshr=math.ceil(need / chip["VX_CFG_NUM_CORES"]), l1_lat=l1_lat, lmem_bytes=l1,
        l1_lmem_shared=True, l2_bytes=l2, l2_ways=chip["VX_CFG_L2_NUM_WAYS"], l2_banks=banks,
        l2_mshr=math.ceil(need / banks), l2_lat=l2_lat, dram_lat=dram_lat,
        dram_channels=plat["VX_CFG_PLATFORM_MEMORY_NUM_BANKS"], l2_bytes_per_cycle=banks * line * l2_bw_anchor,
        flush_untimed=False, persist_across_launch=True,
        sources={
            "cores,warps,l1_bytes,l2_bytes,l2_ways,l2_banks,dram_channels": f"{src} (which cites the vendor whitepaper)",
            "lmem_bytes,l1_lmem_shared": "assumption: the whole combined L1/shared array may be carved out as LMEM "
                                         "(real maximum shared-memory carveouts are smaller)",
            "l1_lat,l2_lat": "VX_config.toml:237,256 latency formulas applied to the profile sizes",
            "dram_lat": (f"{src} PLATFORM_MEMORY_LATENCY {fixed} cycles + 50 ns array (DRAM_ARRAY_NS) x {clk_mhz} MHz"
                         if "VX_CFG_PLATFORM_MEMORY_LATENCY" in plat else
                         f"assumption: a100 profile's 190 cycles (this profile says 'latency not calibrated') + 50 ns "
                         f"array x {clk_mhz} MHz"),
            "l1_mshr,l2_mshr": f"assumption: outstanding lines = 2 x BW x L0 / line = {need} (bandwidth-delay rule), "
                               f"split over cores / banks",
            "l2_bytes_per_cycle": f"anchored to the roofline: an assumption, not a measurement. "
                                  f"{l2_bw_anchor:g} x (one 64 B sector per bank per cycle); GPU_L2_BW_ANCHOR, "
                                  f"methodology §9 M12",
            "acc_per_thread": "assumption: 128 fp32 accumulators per thread (a 128x128 output tile on a 4-warp CTA), "
                              "within the 255 registers a thread may use",
            "persist_across_launch": "True: GPU caches are not reset per launch"})


def hier_for(machine_name: str) -> Hier:
    """Hierarchy paired with a roofline.machines() entry."""
    mm = re.fullmatch(r"vortex-S1-(\d+)c", machine_name)
    if mm:
        return vortex_hier(int(mm.group(1)))
    mm = re.fullmatch(r"psbase-S1-(\d+)c", machine_name)
    if mm:
        return psbase_hier(int(mm.group(1)))
    mm = re.fullmatch(r"(\w+)-like", machine_name)
    if mm:
        return gpu_hier(mm.group(1))
    raise KeyError(machine_name)


# ================================================================================================= cache model


def _floor_pow2(x: float) -> int:
    return 1 << int(math.floor(math.log2(x))) if x >= 1 else 1


def demoted_classes(k: Knobs) -> frozenset:
    """Classes whose fills D inserts at LRU / does not allocate (M8). Phase-aware scope: only decode-read KV."""
    if k.kv_insert == "mru":
        return frozenset()
    return frozenset({"kv_stream"} if k.kv_insert_scope == "decode" else {"kv_stream", "kv"})


def capacity_hit(F: float, cap: float, model: str = "cliff", width: float = 0.0) -> float:
    """Steady-state hit probability of a reference whose reuse window holds F distinct bytes, in a cache region of
    `cap` bytes (M2). "cliff": the LRU footprint-vs-capacity cliff as a ramp of `width` bytes centred on cap,
    clamp((cap + width/2 − F)/width, 0, 1) (unbiased: the mean capacity is cap; width = 0: the ideal step
    1[F ≤ cap]); "proportional": min(1, cap/F)."""
    if F <= 0 or math.isinf(cap):
        return 1.0
    if cap <= 0:
        return 0.0
    if model == "cliff":
        if width <= 0:
            return 1.0 if F <= cap else 0.0
        return min(1.0, max(0.0, (cap + width / 2 - F) / width))
    return min(1.0, cap / F)


def hit_rate(cls: str, window: dict, h: Hier, k: Knobs) -> float:
    """L2 hit probability of a class-`cls` reference whose reuse window holds window[c] distinct bytes of class c
    (the reused stream itself included). Knob effects (M7–M9):
      * A2 way mask: weights live in a·C/ways, every other class in the rest; partitions do not displace each other;
      * D lru: demoted lines sit in the LRU position, so they survive only while the window fits one way (C/ways),
        and they hold at most one way per set against the other classes (capacity − C/ways, not their bytes);
      * D bypass: demoted lines are never allocated (hit 0) and displace nothing."""
    C = h.l2_bytes
    if k.a2_weight_ways is not None:
        wfrac = k.a2_weight_ways / h.l2_ways
        part = {"weight"} if cls == "weight" else set(CLASSES) - {"weight"}
        cap_p = C * (wfrac if cls == "weight" else 1 - wfrac)
    else:
        part, cap_p = set(CLASSES), C
    dem = demoted_classes(k) & part
    one_way = C / h.l2_ways                      # inf for an infinite cache (then the −one way below is skipped)
    ramp = h.cliff_ways * one_way if math.isfinite(C) else 0.0      # M2: width of the cliff (one way by default)
    if cls in dem:
        if k.kv_insert == "bypass":
            return 0.0
        F = sum(window.get(c, 0.0) for c in part)
        return capacity_hit(F, min(cap_p, one_way), h.hit_model, ramp)
    F = sum(window.get(c, 0.0) for c in part if c not in dem)
    cap = cap_p
    if k.kv_insert == "lru" and any(window.get(c, 0.0) > 0 for c in dem) and math.isfinite(C):
        cap = cap_p - one_way
    return capacity_hit(F, cap, h.hit_model, ramp)


# ================================================================================================= DRAM queueing


def md1_wait_fifo(rho: float, s: float) -> float:
    """M/D/1 mean queueing wait (Pollaczek–Khinchine with deterministic service s): W = ρ·s / (2(1−ρ)) (M5)."""
    return math.inf if rho >= 1 else rho * s / (2 * (1 - rho))


def md1_wait_priority(rho_hi: float, rho_lo: float, s: float) -> tuple[float, float]:
    """Non-preemptive two-class priority M/D/1 mean waits (M5): residual work W0 = Σ_i λ_i·E[S²]/2 = (ρ_hi+ρ_lo)·s/2;
    W_hi = W0/(1−ρ_hi); W_lo = W0/((1−ρ_hi)(1−ρ_hi−ρ_lo))."""
    w0 = (rho_hi + rho_lo) * s / 2
    w_hi = math.inf if rho_hi >= 1 else w0 / (1 - rho_hi)
    w_lo = math.inf if rho_hi + rho_lo >= 1 else w0 / ((1 - rho_hi) * (1 - rho_hi - rho_lo))
    return w_hi, w_lo


def simulate_priority_queue(lam_hi: float, lam_lo: float, s: float, n: int = 100_000, seed: int = 0,
                            priority: bool = True, warmup: float = 0.02) -> tuple[float, float]:
    """Discrete-event check of md1_*: one server, Poisson arrivals of two classes, deterministic service s,
    non-preemptive priority (or FIFO). Returns the mean waits (high, low) over requests after a warm-up share."""
    import numpy as np
    from collections import deque
    rng = np.random.default_rng(seed)
    lam = lam_hi + lam_lo
    arr = np.cumsum(rng.exponential(1 / lam, n))
    hi = rng.random(n) < lam_hi / lam
    qh, ql = deque(), deque()
    t_free, i, served = 0.0, 0, 0
    sums, cnts = [0.0, 0.0], [0, 0]
    skip = int(warmup * n)
    while served < n:
        while i < n and arr[i] <= t_free:
            (qh if hi[i] else ql).append(i)
            i += 1
        if not qh and not ql:
            t_free = arr[i]
            continue
        if priority:
            j = qh.popleft() if qh else ql.popleft()
        else:
            j = qh.popleft() if (qh and (not ql or qh[0] < ql[0])) else ql.popleft()
        start = max(t_free, arr[j])
        t_free = start + s
        served += 1
        if j >= skip:
            c = 0 if hi[j] else 1
            sums[c] += start - arr[j]
            cnts[c] += 1
    return sums[0] / max(cnts[0], 1), sums[1] / max(cnts[1], 1)


# ================================================================================================= launches


@dataclass
class Stream:
    """One read stream of a launch (M1). first = compulsory bytes; cross = reuse window of the previous reference in
    an earlier launch or step (None: no earlier reference); reread = tiling re-read bytes with window `within`."""

    cls: str
    path: str                  # "lsu" | "dxa"
    prio: str                  # "crit" | "bulk"
    first: float
    cross: dict | None = None
    reread: float = 0.0
    within: dict | None = None


@dataclass
class LaunchResult:
    names: str
    layer: int
    like: int
    cycles: float
    t_comp: float
    t_mem: float
    t_bw: float
    t_l2: float
    t_lsu: float
    t_dxa: float
    w_crit: float
    w_bulk: float
    n_cta: int
    bound: str
    flops: float
    dram_rd: dict
    dram_wr: float
    l2_rd: dict               # read bytes requested at L2 (compulsory + re-reads), by class
    hit_cross: dict           # bytes that hit in L2 thanks to cross-launch reuse, by class
    n_slots: dict = field(default_factory=dict)   # outstanding request slots used per active issue path (M11)


@dataclass
class IterResult:
    cycles: float
    seconds: float
    flops: float
    dram_rd: dict
    dram_wr: float
    l2_rd: dict
    hit_cross: dict
    compulsory_rd: float
    launches: list

    @property
    def dram_bytes(self) -> float:
        return sum(self.dram_rd.values()) + self.dram_wr


BOUND_TOL = 0.05   # LaunchResult.bound: within 5 % of the bandwidth bound counts as bandwidth-bound (label only)
_ATTN_CORE = re.compile(r"^attn_(qk|softmax|av)$")      # opgen _attn_core op names (not attn_norm/residual)
_GROUP_RE = re.compile(r"^(prefill|decode) n=(\d+) ctx=(\d+) x(\d+)$")


def parse_group(group: str) -> tuple[str, int, int, int]:
    """(kind, n_new, ctx, count) of an opgen attention group label `"<kind> n=<n> ctx=<ctx> x<count>"`."""
    mm = _GROUP_RE.match(group)
    if not mm:
        raise ValueError(f"unparseable attention group {group!r}")
    return mm.group(1), int(mm.group(2)), int(mm.group(3)), int(mm.group(4))


def split_launches(ops: Sequence[opgen.Op], fused_attention: bool = True, fused_mixed: bool = True) -> list[list]:
    """Kernel launches of an op trace (M1): every op is a launch, except that a fused-attention block (consecutive
    `attn_qk` / `attn_softmax` / `attn_av` ops of one layer, every sequence group) is one launch when `fused_mixed` (the
    fused mixed-batch kernel), or one launch per sequence group otherwise."""
    core = lambda x: _ATTN_CORE.match(x.name) is not None
    out, i = [], 0
    while i < len(ops):
        o = ops[i]
        if fused_attention and core(o):
            j = i
            while j < len(ops) and ops[j].layer == o.layer and core(ops[j]):
                j += 1
            if fused_mixed:
                out.append(list(ops[i:j]))
            else:
                groups: dict[str, list] = {}
                for x in ops[i:j]:
                    groups.setdefault(x.group, []).append(x)
                out += list(groups.values())
            i = j
        else:
            out.append([o])
            i += 1
    return out


def _add(a: dict, b: dict, f: float = 1.0) -> dict:
    out = dict(a)
    for c, v in b.items():
        out[c] = out.get(c, 0.0) + f * v
    return out


class MemHier:
    """Iteration cost model: `cost_seqs(seqs, knobs)` -> IterResult (cycles); calling the object with a scheduler
    `Iteration` and a `Knobs` returns seconds (serving/scheduler.py CostFn, time base seconds).

    persist_across_launch: None = the machine's default (False for Vortex, True for GPU-like; sensitivity runs
    pass True for Vortex). use_layer_classes: cost one representative layer per opgen layer class and multiply
    (exact: layers of a class have identical ops, M13). routing / zipf_s / mla_prefill_form go to opgen.gen_ops."""

    def __init__(self, m: opgen.ModelShape, machine: roofline.Machine | str, hier: Hier | None = None,
                 q_w: float = 2, q_a: float = 2, q: float = 2, causal: bool = True, fused_attention: bool = True,
                 fused_mixed: bool = True, persist_across_launch: bool | None = None, use_layer_classes: bool = True,
                 routing: str = "uniform", zipf_s: float = 1.0, mla_prefill_form: str = "absorbed", memo: int = 65536):
        self.m = m
        self.machine = roofline.machines()[machine] if isinstance(machine, str) else machine
        h = hier or hier_for(self.machine.name)
        if persist_across_launch is not None:
            h = dataclasses.replace(h, persist_across_launch=bool(persist_across_launch))
        self.h = h
        self.q_w, self.q_a, self.q = q_w, q_a, q
        self.causal, self.fused, self.fused_mixed = causal, fused_attention, fused_mixed
        self.use_layer_classes = use_layer_classes
        # opgen options passed through unchanged (MoE routing, MLA prefill form; methodology §2 conventions 9-10)
        self.gen_kw = dict(routing=routing, zipf_s=zipf_s, mla_prefill_form=mla_prefill_form)
        self._cached = lru_cache(maxsize=memo)(self._cost_uncached)

    # ------------------------------------------------------------------------------------------- public API

    def cost_seqs(self, seqs: Sequence[opgen.Seq], knobs: Knobs = Knobs(), detail: bool = False) -> IterResult:
        knobs.validate(self.h)
        if detail:
            return self._cost_uncached(tuple(seqs), knobs, True)
        return self._cached(tuple(seqs), knobs, False)

    def cycles(self, seqs: Sequence[opgen.Seq], knobs: Knobs = Knobs()) -> float:
        return self.cost_seqs(seqs, knobs).cycles

    def cost_launch(self, ops: Sequence[opgen.Op], knobs: Knobs = Knobs()) -> LaunchResult:
        """Cost of ONE cold kernel launch made of `ops` (calibration against single-launch SimX rows): the
        same `_streams` / `_launch_time` path as `cost_seqs`, with no earlier launch (empty cross-launch windows),
        which is exact when persist_across_launch is False (cold caches at every launch, §9.2). Refused with
        persistence on, where a launch's cost depends on what earlier launches inserted."""
        if self.h.persist_across_launch:
            raise ValueError("cost_launch is defined for persist_across_launch=False only (cold single launches)")
        knobs.validate(self.h)
        L = list(ops)
        step = dict.fromkeys(CLASSES, 0.0)
        streams, writes, flops_t, flops = self._streams(L, knobs, self._attn_info(L), step, {}, {})
        return self._launch_time(L, knobs, streams, writes, flops_t, flops)

    def __call__(self, iteration, knobs: Knobs | None = None) -> float:
        """scheduler CostFn: seconds for one serving.scheduler.Iteration under `knobs`."""
        return self.cost_seqs(iteration_seqs(iteration, self.m), knobs or Knobs()).seconds

    # ------------------------------------------------------------------------------------------- internals

    def _tile_budget(self, k: Knobs) -> float:
        """Bytes of one CTA's tile buffer (M10): LMEM use split over dxa_depth + 1 buffers; without LMEM the
        co-resident CTAs' implicit tiles together use l1_tile_frac of the L1, so one CTA gets l1_tile_frac·L1/n_cta
        (n_cta·β ≤ the L1 share, tested)."""
        if k.lmem_use > 0:
            return k.lmem_use / (k.dxa_depth + 1)
        return max(self.h.line, self.h.l1_tile_frac * self.h.l1_bytes / self._n_cta(k))

    def _n_cta(self, k: Knobs) -> int:
        """CTAs per core: LMEM-limited when tiles use LMEM (A1 occupancy), else warp-limited."""
        by_warps = max(1, self.h.warps // self.h.warps_per_cta)
        if k.lmem_use <= 0:
            return by_warps
        return max(1, min(by_warps, int(self.h.lmem_bytes // k.lmem_use)))

    def _cost_uncached(self, seqs: tuple, k: Knobs, detail: bool) -> IterResult:
        m, h = self.m, self.h
        like = opgen.layer_classes(m) if self.use_layer_classes else {l: 1 for l in range(m.L)}
        ops = opgen.gen_ops(m, list(seqs), q_w=self.q_w, q_a=self.q_a, q=self.q, causal=self.causal,
                            fused_attention=self.fused, layers=sorted(like), **self.gen_kw)
        mult = lambda o: like.get(o.layer, 1)
        attn_info = self._attn_info(ops)
        launches = split_launches(ops, self.fused, self.fused_mixed)
        # steady-state step window (M4): distinct bytes one iteration inserts, by class
        step = dict.fromkeys(CLASSES, 0.0)
        act_layer: dict[int, float] = {}
        for o in ops:
            f = mult(o)
            step["weight"] += o.w_rd * f
            step["kv"] += o.kv_wr * f
            if o.kind == "attn" and o.kv_rd:
                kind, n, _, _ = parse_group(o.group)
                old = o.kv_rd * (1 - min(1.0, n / max(o.N if o.name.endswith("qk") else o.K, 1)))
                step["kv_stream" if kind == "decode" else "kv"] += old * f
            elif o.kv_rd:
                step["kv"] += o.kv_rd * f
            act_layer[o.layer] = act_layer.get(o.layer, 0.0) + o.a_wr
        pre_post = sum(v for l, v in act_layer.items() if l < 0 or l >= m.L)
        step["act"] = pre_post + max([v for l, v in act_layer.items() if 0 <= l < m.L] or [0.0])

        tot_rd, tot_l2, tot_hx = dict.fromkeys(CLASSES, 0.0), dict.fromkeys(CLASSES, 0.0), dict.fromkeys(CLASSES, 0.0)
        tot = dict(cycles=0.0, flops=0.0, wr=0.0)
        recs = []
        prev_bytes: dict = {}
        since_kv_write: dict = {}
        for L in launches:
            streams, writes, flops_t, flops = self._streams(L, k, attn_info, step, prev_bytes, since_kv_write)
            r = self._launch_time(L, k, streams, writes, flops_t, flops)
            f = mult(L[0])
            tot["cycles"] += r.cycles * f
            tot["flops"] += flops * f
            tot["wr"] += r.dram_wr * f
            for c in CLASSES:
                tot_rd[c] += r.dram_rd[c] * f
                tot_l2[c] += r.l2_rd[c] * f
                tot_hx[c] += r.hit_cross[c] * f
            r.like = f
            if detail:
                recs.append(r)
            # windows for the next launches: bytes this launch inserted, by class (compulsory reads + writes)
            mine = dict.fromkeys(CLASSES, 0.0)
            for s in streams:
                mine[s.cls] += s.first
            for c, v in writes.items():
                mine[c] += v
            prev_bytes = mine
            if any(o.kv_wr for o in L):
                since_kv_write = dict(mine)
            elif since_kv_write:
                since_kv_write = _add(since_kv_write, mine)
        comp = sum((o.w_rd + o.a_rd + o.kv_rd) * mult(o) for o in ops)
        return IterResult(cycles=tot["cycles"], seconds=tot["cycles"] / self.machine.clock_hz, flops=tot["flops"],
                          dram_rd=tot_rd, dram_wr=tot["wr"], l2_rd=tot_l2, hit_cross=tot_hx, compulsory_rd=comp,
                          launches=recs)

    def _attn_info(self, ops) -> dict:
        """Per (layer, group): QKᵀ head length dk, AV head length dv and KV bytes per key per sequence (K + V parts),
        used for the prefill attention tile (M10)."""
        info: dict = {}
        for o in ops:
            if o.kind != "attn":
                continue
            d = info.setdefault((o.layer, o.group), dict(dk=0, dv=0, kv=0.0))
            _, _, _, cnt = parse_group(o.group)
            kt = o.N if o.name.endswith("qk") else o.K
            if o.name.endswith("qk"):
                d["dk"] = o.K
            else:
                d["dv"] = o.N
            d["kv"] += o.kv_rd / (cnt * max(kt, 1))
            d["kt"] = kt
        return info

    def _gemm_tile(self, k: Knobs) -> int:
        """Square output tile T of a GEMM whose tile buffer holds Tk·T·(q_a + q_w) bytes, capped so that the T×T fp32
        accumulators fit the CTA's registers: T² ≤ acc_per_thread · warps_per_cta · threads_per_warp (M10)."""
        h = self.h
        by_buf = _floor_pow2(self._tile_budget(k) / (h.gemm_tk * (self.q_a + self.q_w)))
        return min(by_buf, self._acc_cap_gemm())

    def _acc_cap_gemm(self) -> int:
        h = self.h
        return _floor_pow2(math.sqrt(h.acc_per_thread * h.warps_per_cta * h.threads_per_warp))

    def _attn_reread(self, n: int, ctx: int, W: int, causal: bool, info: dict, k: Knobs) -> tuple[float, int]:
        """KV re-read factor of a prefill chunk (M10): query tiles of Tq rows each stream the keys they attend;
        r = Σ_tiles keys_touched(tile) / keys_touched(chunk). Tq fits Q (g query heads) + K + V of one KV head per
        row in the tile buffer. Decode (n = 1) reads its KV once: r = 1."""
        if n <= 1:
            return 1.0, 1
        m = self.m
        kv_heads = 1 if m.is_mla else m.H_kv
        g = m.H_q / kv_heads
        row = g * info["dk"] * self.q_a + info["kv"] / kv_heads
        h = self.h
        # accumulator cap: Tq rows × g heads × d_v fp32 outputs must fit the CTA's registers (M10)
        tq_acc = _floor_pow2(h.acc_per_thread * h.warps_per_cta * h.threads_per_warp / max(g * max(info["dv"], 1), 1))
        tq = min(_floor_pow2(self._tile_budget(k) / max(row, 1e-9)), tq_acc)
        kt = opgen._keys_touched(n, ctx, W)
        if not causal:
            return float(math.ceil(n / tq)), tq
        tot = 0
        for a in range(0, n, tq):
            tot += opgen._keys_touched(min(tq, n - a), ctx + a, W)
        return tot / kt, tq

    def _streams(self, L, k: Knobs, attn_info, step, prev_bytes, since_kv_write):
        h, m = self.h, self.m
        persist = h.persist_across_launch
        uses_lmem = k.lmem_use > 0
        tile_path = "dxa" if k.dxa_depth > 0 else "lsu"
        T = self._gemm_tile(k)
        n_cta = self._n_cta(k)
        streams: list[Stream] = []
        writes = dict.fromkeys(CLASSES, 0.0)
        flops_t: dict[str, float] = {}
        flops = 0.0
        has_decode = any(o.kind == "attn" and parse_group(o.group)[0] == "decode" for o in L)
        dec_kv = sum(o.kv_rd for o in L if o.kind == "attn" and parse_group(o.group)[0] == "decode")
        act_window = _add({}, prev_bytes, h.act_tail_frac)
        step_w = step if persist else None
        for o in L:
            unit = roofline.op_unit(o, self.machine)
            flops_t[unit] = flops_t.get(unit, 0.0) + o.flops
            flops += o.flops
            writes["act"] += o.a_wr
            writes["kv"] += o.kv_wr
            if o.kind == "linear":
                # M10: GEMM tiles of T x T outputs; weights re-read once per row block, activations per column block
                # distinct weight matrices: routed experts touched (MoE) or per-head matrices (MLA, batch = H)
                matrices = max(o.n_active, 1.0) if o.n_active else max(o.batch, 1)
                w_mat = o.w_rd / matrices
                rows = o.M / max(o.n_active, 1.0) if o.n_active else o.M
                r_w = math.ceil(max(rows, 1) / T)
                r_a = math.ceil(o.N / T)
                # concurrency (M10): P = cores·n_cta output tiles are in flight at once, issued row-block-major.
                # A re-read whose earlier reference is by a tile still in flight (activation panel: next tile;
                # weight panel: the tile cb later) happens while the in-flight tiles step through K together, so
                # its window is their current K-chunks; otherwise it is the panels streamed in between.
                Mt, Nt = min(T, max(rows, 1)), min(T, o.N)
                a_panel, w_panel = Mt * o.K * self.q_a, o.K * Nt * self.q_w
                cb = math.ceil(o.N / T)
                P = h.cores * n_cta
                n_fly = min(P, r_w * cb)
                c_rows, c_cols = min(math.ceil(P / cb), r_w), min(P, cb)
                kstep = {"act": n_fly * h.gemm_tk * Mt * self.q_a, "weight": n_fly * h.gemm_tk * Nt * self.q_w}
                path_t = tile_path if uses_lmem else "lsu"
                # tiles are issued in waves of P: a share of the lockstep re-reads has its partner tile in the next
                # wave (1/P of activation re-reads, cb/P of weight re-reads); those use the serial window
                w_serial = {"weight": w_mat, "act": c_rows * (a_panel + Mt * o.N * self.q_a)}
                a_serial = {"act": c_rows * a_panel + min(P, cb * r_w) * Mt * Nt * self.q_a,
                            "weight": c_cols * w_panel}
                if o.w_rd:
                    rr = (r_w - 1) * o.w_rd
                    sh = cb / P if P > cb else 1.0
                    streams.append(Stream("weight", path_t, "crit", o.w_rd, step_w, rr * (1 - sh), kstep))
                    if sh > 0 and rr > 0:
                        streams.append(Stream("weight", path_t, "crit", 0.0, None, rr * sh, w_serial))
                if o.a_rd:
                    rr = (r_a - 1) * o.a_rd
                    sh = 1 / P if P > 1 else 1.0
                    streams.append(Stream("act", path_t, "crit", o.a_rd,
                                          _add(act_window, {"act": o.a_rd}) if persist else None, rr * (1 - sh), kstep))
                    if sh > 0 and rr > 0:
                        streams.append(Stream("act", path_t, "crit", 0.0, None, rr * sh,
                                              a_serial if P > 1 else {"act": a_panel + Mt * Nt * self.q_a,
                                                                      "weight": w_panel}))
                if o.kv_rd:      # MLA kv_b_decompress reads the cached latent (prefill only)
                    streams.append(Stream("kv", "lsu", "crit", o.kv_rd, step_w))
            elif o.kind == "attn":
                kind, n, ctx, cnt = parse_group(o.group)
                info = attn_info[(o.layer, o.group)]
                prio = "bulk" if (kind == "prefill" and has_decode) else "crit"
                if o.kv_rd:
                    cls = "kv_stream" if kind == "decode" else "kv"
                    kt = info["kt"]
                    fresh = min(1.0, n / max(kt, 1))
                    r_kv, tq = self._attn_reread(n, ctx, m.windows[o.layer], self.causal, info, k)
                    path = tile_path if uses_lmem else "lsu"
                    within = None
                    if r_kv > 1:
                        # work items = (sequence, KV head) pairs; at most cores·n_cta run at once (M10)
                        kvh = 1 if m.is_mla else m.H_kv
                        conc = min(cnt * kvh, h.cores * n_cta)
                        within = {cls: info["kv"] * kt / kvh * conc,
                                  "act": tq * info["dk"] * self.q_a * (m.H_q / kvh) * conc,
                                  "kv_stream": dec_kv / r_kv}
                    old, new = o.kv_rd * (1 - fresh), o.kv_rd * fresh
                    if old:
                        streams.append(Stream(cls, path, prio, old, step_w, (r_kv - 1) * old, within))
                    if new:  # keys written this step by the layer's KV-write launch (M4); the write filled them as
                        # class kv (normal insertion), so D's policy for decode-read KV does not apply to them
                        streams.append(Stream("kv", path, prio, new,
                                              _add(since_kv_write, {"kv": new}) if persist else None,
                                              (r_kv - 1) * new, within))
                if o.a_rd:
                    streams.append(Stream("act", "lsu", prio, o.a_rd,
                                          _add(act_window, {"act": o.a_rd}) if persist else None))
            else:   # elementwise / embed / fused softmax
                if o.a_rd:
                    pr = "bulk" if (o.group and parse_group(o.group)[0] == "prefill" and has_decode) else "crit"
                    streams.append(Stream("act", "lsu", pr, o.a_rd,
                                          _add(act_window, {"act": o.a_rd}) if persist else None))
                if o.w_rd:   # norm weight vectors are re-read every step; embedding rows of new tokens are not (M4)
                    streams.append(Stream("weight", "lsu", "crit", o.w_rd, None if o.kind == "embed" else step_w))
                if o.kv_rd:
                    streams.append(Stream("kv", "lsu", "crit", o.kv_rd, step_w))
        return streams, writes, flops_t, flops

    def _launch_time(self, L, k: Knobs, streams, writes, flops_t, flops) -> LaunchResult:
        h, mc = self.h, self.machine
        line = h.line
        rd = dict.fromkeys(CLASSES, 0.0)
        l2 = dict.fromkeys(CLASSES, 0.0)
        hx = dict.fromkeys(CLASSES, 0.0)
        # per issue path: hit lines, miss lines by priority class
        P = {p: dict(hit=0.0, crit=0.0, bulk=0.0) for p in ("lsu", "dxa")}
        D = dict(crit=0.0, bulk=0.0)
        for s in streams:
            h_x = hit_rate(s.cls, s.cross, h, k) if (h.persist_across_launch and s.cross is not None) else 0.0
            h_in = hit_rate(s.cls, s.within, h, k) if (s.reread > 0 and s.within is not None) else 0.0
            miss = s.first * (1 - h_x) + s.reread * (1 - h_in)
            req = s.first + s.reread
            rd[s.cls] += miss
            l2[s.cls] += req
            hx[s.cls] += s.first * h_x
            P[s.path]["hit"] += (req - miss) / line
            P[s.path][s.prio] += miss / line
            D[s.prio] += miss
        # timed write-backs (M6): with an untimed end-of-launch flush (SimX), dirty lines still in L2 cost nothing; a
        # line written uniformly during the launch is evicted iff more than C bytes follow it: fraction (1 − C/F)⁺
        wr = sum(writes.values())
        if wr and h.flush_untimed:
            F = sum(s.first + s.reread for s in streams) + wr
            wr *= max(0.0, 1.0 - h.l2_bytes / F) if math.isfinite(h.l2_bytes) else 0.0
        bw = h.eta_dram * mc.dram_bytes_per_cycle
        t_bw = (D["crit"] + D["bulk"] + wr) / bw
        t_l2 = (sum(l2.values()) + sum(writes.values())) / h.l2_bytes_per_cycle
        # outstanding request slots per issue path (M11)
        n_cta = self._n_cta(k)
        uses_lmem = k.lmem_use > 0 and any(o.kind in ("linear", "attn") for o in L)
        aw = min(h.warps, n_cta * h.warps_per_cta) if uses_lmem else h.warps
        l2_slots = h.l2_banks * h.l2_mshr
        N = dict(lsu=h.cores * min(h.l1_mshr, aw * h.mlp_per_warp),
                 dxa=h.cores * n_cta * k.dxa_depth * math.ceil(self._tile_budget(k) / line))
        active = [p for p in N if P[p]["crit"] + P[p]["bulk"] + P[p]["hit"] > 0]
        # both paths draw on the same L2 MSHRs: scale the active paths down together if they exceed them (M11)
        n_raw = sum(N[p] for p in active)
        if n_raw > l2_slots:
            for p in active:
                N[p] *= l2_slots / n_raw
        s_ch = line * h.dram_channels / bw            # service time of one request on its channel (cycles)
        n_tot = sum(N[p] for p in active)
        n_crit = sum(N[p] * P[p]["crit"] / max(P[p]["crit"] + P[p]["bulk"], 1e-300) for p in N)
        cap_fifo = n_tot / h.dram_channels * s_ch
        cap_crit = (n_crit / h.dram_channels + 1) * s_ch

        def waits(t):
            rho_c, rho_b = D["crit"] / (bw * t), (D["bulk"] + wr) / (bw * t)
            if k.b_two_class:
                wc, wb = md1_wait_priority(rho_c, rho_b, s_ch)
                return min(wc, cap_crit, cap_fifo), min(wb, cap_fifo)
            w = min(md1_wait_fifo(rho_c + rho_b, s_ch), cap_fifo)
            return w, w

        def t_path(p, t):
            if N[p] <= 0 or P[p]["crit"] + P[p]["bulk"] + P[p]["hit"] == 0:
                return 0.0
            wc, wb = waits(t)
            busy = P[p]["hit"] * h.hit_lat + P[p]["crit"] * (h.miss_lat + wc) + P[p]["bulk"] * (h.miss_lat + wb)
            return busy / N[p]

        t_lo = max(t_bw, t_l2)
        g = lambda t: t - max(t_path("lsu", t), t_path("dxa", t))
        if not active or (t_lo > 0 and g(t_lo) >= 0):
            t_mem = t_lo
        else:
            lo = t_lo
            hi = max(t_lo, 1e-12)
            while g(hi) < 0:
                hi *= 2
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid > lo and g(mid) < 0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= 1e-9 * hi:
                    break
            t_mem = hi
        t_lsu, t_dxa = t_path("lsu", t_mem), t_path("dxa", t_mem)
        wc, wb = waits(t_mem) if t_mem > 0 else (0.0, 0.0)
        t_comp = sum(f / (mc.flops_per_cycle[u] * h.eta_comp) for u, f in flops_t.items())
        # M3: overlap. Single-buffered LMEM tiles (no DXA) serialize load and compute within a CTA; n_cta co-resident
        # CTAs hide 1 - 1/n_cta of the shorter phase. The overlap factor ω (Hier.overlap) scales the hidden
        # share: exposed share of the shorter phase = 1 − ω (full overlap) or 1 − ω·(1 − 1/n_cta) (single-buffered);
        # ω = 1 reproduces the formulas exactly.
        # (ω = 1 takes the original expressions, so outputs stay bit-identical.)
        w = h.overlap
        if uses_lmem and k.dxa_depth == 0:
            body = (max(t_comp, t_mem) + min(t_comp, t_mem) / n_cta if w == 1.0
                    else max(t_comp, t_mem) + min(t_comp, t_mem) * (1.0 - w * (1.0 - 1.0 / n_cta)))
        else:
            body = max(t_comp, t_mem) if w == 1.0 else max(t_comp, t_mem) + min(t_comp, t_mem) * (1.0 - w)
        n_rd = sum(rd.values())
        exposed = h.miss_lat if n_rd > 0 else (h.hit_lat if sum(l2.values()) > 0 else 0.0)
        cycles = body + exposed + h.launch_overhead
        # reporting label: the fixed point always lands at or above max(t_bw, t_l2); within BOUND_TOL of it the launch
        # is labelled by that bandwidth bound, otherwise "latency" (Little's law limits it)
        if t_comp >= t_mem:
            bound = "compute"
        elif t_mem <= (1 + BOUND_TOL) * max(t_bw, t_l2):
            bound = "bandwidth" if t_bw >= t_l2 else "l2"
        else:
            bound = "latency"
        return LaunchResult(names="+".join(dict.fromkeys(o.name for o in L)), layer=L[0].layer, like=1,
                            cycles=cycles, t_comp=t_comp, t_mem=t_mem, t_bw=t_bw, t_l2=t_l2, t_lsu=t_lsu, t_dxa=t_dxa,
                            w_crit=wc, w_bulk=wb, n_cta=n_cta if uses_lmem else 0, bound=bound, flops=flops,
                            dram_rd=rd, dram_wr=wr, l2_rd=l2, hit_cross=hx, n_slots={p: N[p] for p in active})


# ================================================================================================= scheduler glue


def iteration_seqs(it, m: opgen.ModelShape | None = None, collapse_decodes: bool = True) -> tuple:
    """opgen sequence groups of a serving.scheduler.Iteration (M14). A prefill chunk [s0, s0+n) is
    Seq("prefill", n, ctx=s0, emits=last); a decode at context T is Seq("decode", 1, ctx=T−1).
    collapse_decodes: B decodes with Σ T are represented by at most two groups at ⌊ΣT/B⌋ and ⌊ΣT/B⌋+1 with the same
    B and Σ T — exact for every decode quantity linear in T (KV bytes, attention FLOPs), which is all of them
    unless a layer has a sliding window; with windows (m given and any window > 0) decodes are grouped exactly."""
    seqs = [opgen.Seq("prefill", p.n_tokens, p.start, 1, bool(p.last)) for p in it.prefill]
    ctxs = [d.context for d in it.decode]
    if ctxs:
        windowed = m is not None and any(m.windows)
        if collapse_decodes and not windowed:
            B, S = len(ctxs), sum(ctxs)
            lo, n_hi = S // B, S % B
            if B - n_hi:
                seqs.append(opgen.Seq("decode", 1, lo - 1, B - n_hi))
            if n_hi:
                seqs.append(opgen.Seq("decode", 1, lo, n_hi))
        else:
            for T, c in sorted(Counter(ctxs).items()):
                seqs.append(opgen.Seq("decode", 1, T - 1, c))
    return tuple(seqs)


def knob_grid(h: Hier, lmem_fracs=(0, 0.25, 0.5, 1.0), depths=(0, 1, 2, 4), inserts=("mru", "lru", "bypass"),
              two_class=(False, True), a2=(None,)) -> list[Knobs]:
    """All valid knob vectors of a small grid (DXA depth > 0 only with LMEM tiles). picks its own subset."""
    out = []
    for fr in lmem_fracs:
        lm = int(h.lmem_bytes * fr)
        for dd in depths:
            if dd > 0 and lm == 0:
                continue
            for ins in inserts:
                for b in two_class:
                    for w in a2:
                        out.append(Knobs(lmem_use=lm, dxa_depth=dd, kv_insert=ins, b_two_class=b, a2_weight_ways=w))
    return out


# ================================================================================================= CLI


def _scenarios(m: opgen.ModelShape) -> dict:
    cap = m.max_seq or 1 << 30
    S, T1, T2, C = min(512, cap), min(512, cap), min(4096, cap), min(256, cap)
    return {f"prefill_B1_S{S}": opgen.prefill(1, S), f"decode_B1_T{T1}": opgen.decode(1, T1),
            f"decode_B16_T{T2}": opgen.decode(16, T2),
            f"mixed_chunk{C}+decode_B15_T{T2}": opgen.mixed([(C, 0)], [(15, T2)])}


def _example_knobs(h: Hier) -> dict:
    """A handful of knob vectors for results/<run>/example_costs.csv (not a sweep; sweeps)."""
    half = int(h.lmem_bytes // 2)
    return {"stock": Knobs(), "A1 half-LMEM": Knobs(lmem_use=half),
            "A1 half + C depth2": Knobs(lmem_use=half, dxa_depth=2), "D lru": Knobs(kv_insert="lru"),
            "D bypass": Knobs(kv_insert="bypass"), "B two-class": Knobs(b_two_class=True),
            "A2 weight 3/4 ways": Knobs(a2_weight_ways=h.l2_ways * 3 // 4),
            "A1 half + C2 + B + D lru": Knobs(lmem_use=half, dxa_depth=2, b_two_class=True, kv_insert="lru")}


def main(argv=None):
    ap = argparse.ArgumentParser(description="example memhier predictions (uncalibrated)")
    ap.add_argument("--out", default="results/P1.4")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = [(opgen.from_llama2c_bin(PS_ROOT / "data" / "models" / "stories15M.bin"), 4),
              (opgen.from_hf_config(PS_ROOT / "configs" / "unsloth__Meta-Llama-3.1-8B.json"), 2)]
    rows = []
    for m, qb in models:
        for mname, persist in (("vortex-S1-8c", False), ("vortex-S1-8c", True), ("h100-like", True)):
            mh = MemHier(m, mname, q_w=qb, q_a=qb, q=qb, persist_across_launch=persist)
            for sc, seqs in _scenarios(m).items():
                base = None
                for kname, kn in _example_knobs(mh.h).items():
                    r = mh.cost_seqs(seqs, kn)
                    base = base or r.cycles
                    rows.append(dict(model=m.name, machine=mname, persist=persist, scenario=sc, knobs=kname,
                                     knob_label=kn.label, cycles=f"{r.cycles:.6g}", speedup_vs_stock=f"{base / r.cycles:.4f}",
                                     dram_rd=f"{sum(r.dram_rd.values()):.6g}", dram_wr=f"{r.dram_wr:.6g}",
                                     compulsory_rd=f"{r.compulsory_rd:.6g}",
                                     dram_over_compulsory=f"{sum(r.dram_rd.values()) / r.compulsory_rd:.4f}",
                                     weight_rd=f"{r.dram_rd['weight']:.6g}", kv_rd=f"{r.dram_rd['kv'] + r.dram_rd['kv_stream']:.6g}",
                                     act_rd=f"{r.dram_rd['act']:.6g}", flops=f"{r.flops:.6g}"))
    with open(out / "example_costs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hs = [vortex_hier(4), vortex_hier(8), vortex_hier(16), gpu_hier("a100"), gpu_hier("h100")]
    with open(out / "hier_params.csv", "w", newline="") as f:
        flds = [x.name for x in dataclasses.fields(Hier) if x.name != "sources"]
        w = csv.writer(f)
        w.writerow(flds + ["miss_lat", "sources"])
        for hh in hs:
            w.writerow([getattr(hh, x) for x in flds] + [hh.miss_lat,
                                                          " | ".join(f"{k}: {v}" for k, v in hh.sources.items())])
    # sensitivity of the largest knob effect in the examples (phase-aware D in a fused mixed launch on Vortex) to the
    # two least certain cache-model choices: the hit model (M2) and the L2 capacity
    m15 = models[0][0]
    sens = []
    for hm in ("cliff", "proportional"):
        for l2 in (2**19, 2**20, 2**21, 2**22):
            hh = dataclasses.replace(vortex_hier(8), hit_model=hm, l2_bytes=l2)
            mh = MemHier(m15, "vortex-S1-8c", hier=hh, q_w=4, q_a=4, q=4)
            for C, B, T in ((64, 15, 256), (256, 15, 256), (256, 4, 128)):
                seqs = opgen.mixed([(C, 0)], [(B, T)])
                a0, d0 = mh.cost_seqs(seqs), mh.cost_seqs(seqs, Knobs(kv_insert="lru"))
                b0 = mh.cost_seqs(seqs, Knobs(b_two_class=True))
                sens.append(dict(hit_model=hm, l2_bytes=l2, chunk=C, decode_B=B, decode_T=T,
                                 cycles_stock=f"{a0.cycles:.6g}", speedup_D_lru=f"{a0.cycles / d0.cycles:.4f}",
                                 speedup_B=f"{a0.cycles / b0.cycles:.4f}"))
    with open(out / "sensitivity_mixed_D.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0]))
        w.writeheader()
        w.writerows(sens)
    prm = sensitivity_params(models)
    with open(out / "sensitivity_params.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prm[0]))
        w.writeheader()
        w.writerows(prm)
    print(f"wrote {len(rows)} rows to {out / 'example_costs.csv'}, {out / 'hier_params.csv'}, "
          f"{out / 'sensitivity_mixed_D.csv'} and {len(prm)} rows to {out / 'sensitivity_params.csv'}")


def sensitivity_params(models) -> list[dict]:
    """Methodology §9.6: speedup of the key knob vectors over the stock knobs under each parameter variation that
    must report (hit model, cliff width, L2 ±1 line / ×½ / ×2, stock tile, accumulator cap, warps_per_cta,
    DRAM latency, L2 bandwidth)."""
    (m15, _), (m8, _) = models
    cases = [  # (model, q, machine, scenario label, seqs, knob label, knob builder)
        (m15, 4, "vortex-S1-8c", "mixed 256+15x256", opgen.mixed([(256, 0)], [(15, 256)]), "D lru", lambda h: Knobs(kv_insert="lru")),
        (m15, 4, "vortex-S1-8c", "mixed 256+31x256", opgen.mixed([(256, 0)], [(31, 256)]), "D lru", lambda h: Knobs(kv_insert="lru")),
        (m15, 4, "vortex-S1-8c", "mixed 256+15x256", opgen.mixed([(256, 0)], [(15, 256)]), "B two-class",
         lambda h: Knobs(b_two_class=True)),
        (m8, 2, "vortex-S1-8c", "prefill 512", opgen.prefill(1, 512), "A1 half", lambda h: Knobs(lmem_use=int(h.lmem_bytes // 2))),
        (m8, 2, "vortex-S1-8c", "mixed 256+15x4096", opgen.mixed([(256, 0)], [(15, 4096)]), "A1 half + C2",
         lambda h: Knobs(lmem_use=int(h.lmem_bytes // 2), dxa_depth=2)),
        (m8, 2, "h100-like", "prefill 512", opgen.prefill(1, 512), "A1 half", lambda h: Knobs(lmem_use=int(h.lmem_bytes // 2))),
        (m8, 2, "h100-like", "prefill 512", opgen.prefill(1, 512), "A1 half + C2",
         lambda h: Knobs(lmem_use=int(h.lmem_bytes // 2), dxa_depth=2)),
        (m8, 2, "h100-like", "decode 16x4096", opgen.decode(16, 4096), "A1 half + C2",
         lambda h: Knobs(lmem_use=int(h.lmem_bytes // 2), dxa_depth=2)),
    ]
    variations = [("default", {}), ("hit_model=proportional", dict(hit_model="proportional")),
                  ("cliff_ways=0 (step)", dict(cliff_ways=0.0)), ("cliff_ways=2", dict(cliff_ways=2.0)),
                  ("L2 -1 line", "l2-64"), ("L2 +1 line", "l2+64"), ("L2 x0.5", "l2*0.5"), ("L2 x2", "l2*2"),
                  ("l1_tile_frac=0.25", dict(l1_tile_frac=0.25)), ("l1_tile_frac=1", dict(l1_tile_frac=1.0)),
                  ("acc_per_thread x0.5", "acc*0.5"), ("acc_per_thread x2", "acc*2"),
                  ("warps_per_cta=2", dict(warps_per_cta=2)), ("warps_per_cta=8", dict(warps_per_cta=8)),
                  ("dram_lat x0.5", "lat*0.5"), ("dram_lat x4", "lat*4"),
                  ("l2_bytes_per_cycle x0.5", "l2bw*0.5"), ("l2_bytes_per_cycle x2", "l2bw*2"),
                  ("GPU L2 BW unanchored (/GPU_L2_BW_ANCHOR)", "unanchored")]
    out = []
    for m, q, mach, sc, seqs, kl, kb in cases:
        base = hier_for(mach)
        for vname, v in variations:
            if v == "unanchored":
                if mach.startswith("vortex"):
                    continue
                hh = dataclasses.replace(base, l2_bytes_per_cycle=base.l2_bytes_per_cycle / GPU_L2_BW_ANCHOR)
            elif isinstance(v, dict):
                hh = dataclasses.replace(base, **v)
            else:
                key, op, val = re.fullmatch(r"(\w+)([+\-*])([\d.]+)", v).groups()
                fld = dict(l2="l2_bytes", acc="acc_per_thread", lat="dram_lat", l2bw="l2_bytes_per_cycle")[key]
                x = getattr(base, fld)
                x = x + float(val) if op == "+" else x - float(val) if op == "-" else x * float(val)
                hh = dataclasses.replace(base, **{fld: int(x) if fld == "acc_per_thread" else x})
            c = MemHier(m, mach, hier=hh, q_w=q, q_a=q, q=q)
            k = kb(hh)
            a0, b0 = c.cost_seqs(seqs), c.cost_seqs(seqs, k)
            out.append(dict(model=m.name, machine=mach, scenario=sc, knobs=kl, variation=vname,
                            stock_T=c._gemm_tile(Knobs()), knob_T=c._gemm_tile(k), cycles_stock=f"{a0.cycles:.6g}",
                            cycles_knob=f"{b0.cycles:.6g}", speedup=f"{a0.cycles / b0.cycles:.4f}"))
    return out


if __name__ == "__main__":
    main()
