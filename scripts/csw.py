#!/usr/bin/env python3
"""— switching cost C_sw, intra-kernel C_warmup and break-even N_min (E9). Methodology §13.

Pure functions (unit-tested in tests/test_csw.py) plus a CLI that writes results/<run>/{csw.csv, warmup.csv, nmin.csv,
medians.csv} PS_TRACE files. Every flush number is ANALYTICAL (RTL formula,
third_party/vortex/docs/designs/cache_subsystem.md:671-692, §6.5 walk + §6.7 cost table) and is labelled so in every
output row; SimX's flush walk is untimed and outside kernel cycles, so it cannot be measured here.

Run: nice -n 19 conda run -n phaseshift --no-capture-output python -B scripts/csw.py   (small job, < 200 MB)
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS))
sys.path.insert(0, str(PS / "scripts"))

import pstrace  # noqa: E402

ANALYTICAL = "analytical (RTL formula)"
SECTOR_BYTES = 64          # DRAM transaction = L2 sector = MEM_BLOCK (VX_config.toml:66,75); one write-back each


# ------------------------------------------------------------------------------------------------ flush (analytical)
@dataclass(frozen=True)
class CacheGeom:
    name: str
    size: int        # bytes
    ways: int
    banks: int
    line: int        # bytes (L2: 2 x MEM_BLOCK = 128 B, sectored; VX_config.toml:74)
    writeback: bool

    @property
    def lines_per_bank(self) -> int:
        """LINES_PER_BANK = sets per bank = size / (banks x ways x line) (RTL CS_LINE_SEL_BITS range)."""
        n, r = divmod(self.size, self.banks * self.ways * self.line)
        if r or n <= 0:
            raise ValueError(f"{self.name}: size not a multiple of banks*ways*line")
        return n


def walk_cycles(g: CacheGeom) -> int:
    """RTL flush walk per bank (cache_subsystem.md §6.7 table): write-through LINES_PER_BANK (all ways cleared in
    parallel), write-back LINES_PER_BANK x NUM_WAYS (each (way, line) visited, a dirty one emits a write-back).
    All banks walk concurrently, so this is also the level's walk time."""
    return g.lines_per_bank * (g.ways if g.writeback else 1)


def flush_cycles(levels: list[CacheGeom], dirty_sectors: int, t_wr: float) -> float:
    """Analytical hierarchy flush: the levels walk one after another (bank 0 of a level forwards the flush downstream
    only after its own walk, §6.5), plus the dirty write-backs serialized through the memory-request queue at
    t_wr cycles per 64-B sector (§6.7 "plus writeback cycles for the dirty fraction serialized through mreq_queue").
    Adding the write-back time to the walk (no overlap) is an upper bound; the FSM's 1-cycle states are ignored."""
    if dirty_sectors < 0 or t_wr < 0:
        raise ValueError("dirty_sectors and t_wr must be >= 0")
    return sum(walk_cycles(g) for g in levels) + dirty_sectors * t_wr


# ------------------------------------------------------------------------------------------------ C_warmup (measured)
def cum_cycles(work_rows: list[dict]) -> dict[int, int]:
    """Launch-local cycle at which the work counter first reached k, for every k that closes a row (n = 1 traces:
    every k = 1..W). The LAST row closes at the end of the launch (epilogue included, methodology §12.10), so the
    value for k = W is the launch end, not the W-th unit."""
    out = {}
    for r in work_rows:
        out.setdefault(r["work_end"], r["cycle"])
    return out


@dataclass(frozen=True)
class Warmup:
    units: int          # W
    per_round: int      # P
    k: int
    a: int
    b: int
    steady: float       # s, cycles per unit
    first_k: int        # cycles of the first k units
    c_warmup: float     # first_k - k*s
    kernel_cycles: int  # Σ work-row cycles


def c_warmup(work_rows: list[dict], per_round: int, k: int | None = None) -> Warmup:
    """C_warmup = (cycles of the first k units) − k × (steady-state cycles per unit of the SAME launch).

    Steady state s = (cum(b) − cum(a)) / (b − a) with a = P·⌊R/2⌋, b = P·(R − 1), R = ⌊W/P⌋ full rounds of P units
    (P = units completed concurrently, one per resident CTA): the second half of the launch, in whole rounds, without
    the last round (its last row also holds the epilogue/drain). Whole rounds because units finish in bursts of P
    (one per core), so per-unit deltas are bimodal and only round-aligned differences are stable. k defaults to P
    (the first round: every CTA's first tile). Needs R >= 3."""
    if not work_rows:
        raise ValueError("no work rows")
    if per_round < 1:
        raise ValueError("per_round must be >= 1")
    cum = cum_cycles(work_rows)
    W = max(cum)
    R = W // per_round
    if R < 3:
        raise ValueError(f"need >= 3 full rounds, have {R} (W={W}, P={per_round})")
    k = per_round if k is None else k
    a, b = per_round * (R // 2), per_round * (R - 1)
    if not (0 < k < W and a < b and a in cum and b in cum and k in cum):
        raise ValueError("trace does not resolve the needed work counts (use PS_TRACE_EVERY dividing P)")
    s = (cum[b] - cum[a]) / (b - a)
    return Warmup(W, per_round, k, a, b, s, cum[k], cum[k] - k * s, sum(r["cycles"] for r in work_rows))


def round_cycles(work_rows: list[dict], per_round: int) -> list[int]:
    """Cycles of each full round (diagnostic: shows drift and per-wave prologues)."""
    cum = cum_cycles(work_rows)
    W = max(cum)
    pts = [0] + [cum[j] for j in range(per_round, W + 1, per_round) if j in cum]
    return [y - x for x, y in zip(pts, pts[1:])]


# ------------------------------------------------------------------------------------------------ N_min (E9)
def n_min(c_enter: float, c_exit: float, c_warm: float, saved_per_iter: float) -> int:
    """E9: N_min = ⌈(C_enter + C_exit + C_warmup) / ΔC_saved⌉ (whole iterations). 0 means a mode pays from its
    first iteration (no break-even constraint). ΔC_saved must be > 0 (a mode that saves nothing never pays)."""
    if saved_per_iter <= 0:
        raise ValueError("saved_per_iter must be > 0")
    tot = c_enter + c_exit + c_warm
    if tot < 0:
        raise ValueError("costs must be >= 0")
    return math.ceil(tot / saved_per_iter - 1e-12)


def breakeven_fraction(total_cost: float, median_len: float, step_cycles: float) -> float:
    """Smallest per-iteration saving (as a fraction of a decode step) for which N_min <= median_len:
    N_min <= m  ⇔  C/(f·T) <= m  ⇔  f >= C/(m·T)."""
    return total_cost / (median_len * step_cycles)


# ------------------------------------------------------------------------------------------------ CLI
GEOM = {   # resolved with results/<run>/l2_geometry.cpp (l2_geometry.out) from the ps tree's VX_config.h
    "PS-BASE-S1-4c": [CacheGeom("L1D", 16384, 4, 4, 64, False), CacheGeom("L2", 1048576, 8, 8, 128, True)],
    "REG-CFG": [CacheGeom("L1D", 16384, 4, 2, 64, False), CacheGeom("L2", 1048576, 8, 8, 128, True)],
}
T_WR = {   # cycles per 64-B write-back
    # measured proxy: stream bandwidth (results/<run>/machine_params.csv stream_bytes_per_cycle 40.707; a READ
    # stream — no write-stream microbenchmark exists) ; bound: peak 64 B/cycle (2 channels x 32 B/cycle)
    "PS-BASE-S1-4c": {"meas_stream": SECTOR_BYTES / 40.707, "peak": SECTOR_BYTES / 64.0},
    "REG-CFG": {"peak": SECTOR_BYTES / 64.0},
}
STEP_CYCLES = 44301707 / 6   # stories15M tf, 6 decode positions at PS-BASE-S1-4c (results/<run>/rows.jsonl, case tf6)
TRACES = [  # (label, config, file, P, kernel) — all PS_HINTS=1 marked binaries, unit = marks, n = 1
    ("decode_T128", "PS-BASE-S1-4c", "results/P4.4/traces/decode_T128_n1.csv", 4, "ps_decode_attn_gqa -B 2 -H 8 -K 2 -D 64 -T 128"),
    ("decode_T512", "PS-BASE-S1-4c", "results/P4.4/traces/decode_T512_n1.csv", 4, "ps_decode_attn_gqa -B 2 -H 8 -K 2 -D 64 -T 512"),
    ("decode_T1024", "PS-BASE-S1-4c", "results/P4.4/traces/decode_T1024_n1.csv", 4, "ps_decode_attn_gqa -B 2 -H 8 -K 2 -D 64 -T 1024"),
    ("mixed_c64_T256", "PS-BASE-S1-4c", "results/P4.4/traces/mixed_c64_T256_n1.csv", 4, "ps_mixed_attn -H 8 -K 2 -D 64 -c 64 -o 0 -T 256 -n 4"),
    ("decode_B2K4D32_T128_2waves", "PS-BASE-S1-4c", "results/P4.3/traces/gqa_base4c_n1.csv", 4, "ps_decode_attn_gqa -B 2 -H 8 -K 4 -D 32 -T 128"),
    ("decode_B4K2_T1024_2waves", "PS-BASE-S1-4c", "results/P4.3/traces/overhead_base4c_n1.csv", 4, "ps_decode_attn_gqa -B 4 -H 8 -K 2 -D 64 -T 1024"),
    ("decode_B2K4D32_T128_2waves", "REG-CFG", "results/P4.3/traces/gqa_reg_n1.csv", 4, "ps_decode_attn_gqa -B 2 -H 8 -K 4 -D 32 -T 128"),
]
SIGMAS = {"unscaled": 1.0, "PS-BASE-S1-4c (1 MiB L2)": (131072 / 2**25) / (13824 / 2**20),
          "PS-BASE-S1-8c (2 MiB L2)": (131072 / 2**25) / (13824 / 2**21)}
FRACTIONS = [0.001, 0.005, 0.01, 0.05, 0.10]


def flush_dirty_sectors(rows: list[dict]) -> int:
    """Write-backs issued during the flush(es) of the trace (kind=flush rows), all classes: dram_wr_bytes / 64.
    Issued, never completed counts (methodology §12.9: completions undercount)."""
    return sum(v for r in rows if r["kind"] == "flush" for n, v in r.items() if n.startswith("dram_wr_bytes_")) // SECTOR_BYTES


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def medians() -> list[dict]:
    import numpy as np
    from serving import traces as tr
    reqs = tr.load_trace("azure_conv")
    out = []
    for name, sg in SIGMAS.items():
        for mc in (None, 256):
            if sg == 1.0 and mc is not None:
                continue
            sc = reqs if sg == 1.0 else tr.scale_lengths(reqs, sg, max_context=mc)
            raw = reqs if sg == 1.0 else tr.scale_lengths(reqs, sg)
            clipped = float(np.mean([(a.input_len, a.output_len) != (b.input_len, b.output_len) for a, b in zip(sc, raw)]))
            o = np.array([r.output_len for r in sc])
            out.append({"trace": "azure_conv", "sigma_case": name, "sigma": sg, "max_context": mc or "",
                        "n_requests": len(sc), "median_decode_len": float(np.median(o)),
                        "clipped_frac": clipped})
    return out


def main() -> int:
    res = PS / "results" / "P4.4"
    # --- C_warmup
    warm = []
    for label, cfg, f, P, kern in TRACES:
        _, rows = pstrace.read_trace(PS / f)
        wr = pstrace.work_rows(rows, 0)
        w = c_warmup(wr, P)
        w2 = c_warmup(wr, P, 2 * P)
        rc = round_cycles(wr, P)
        half = len(rc) // 2
        warm.append({"label": label, "config": cfg, "kernel": kern, "trace": f, "units_W": w.units, "P": P,
                     "steady_a": w.a, "steady_b": w.b, "steady_cyc_per_unit": w.steady,
                     "first_P_cycles": w.first_k, "c_warmup_kP": w.c_warmup, "c_warmup_k2P": w2.c_warmup,
                     "kernel_cycles": w.kernel_cycles, "c_warmup_kP_frac": w.c_warmup / w.kernel_cycles,
                     "round1": rc[0], "round2": rc[1], "round_half": rc[half] if half < len(rc) else "",
                     "round_last_full": rc[-1],
                     # 2-wave launches only: wave-1 first round minus wave-2 first round (same per-CTA prologue,
                     # warm icache/L2 metadata) = the cold-launch part of C_warmup, prologue removed (§13.3)
                     "cold_minus_wave2_round": (rc[0] - rc[half]) if label.endswith("2waves") else "",
                     "flush_wb_sectors": flush_dirty_sectors(rows), "label_kind": "measured (SimX trace)"})
    write_csv(res / "warmup.csv", warm)
    # --- C_sw per change type
    csw = []
    for cfg, levels in GEOM.items():
        walks = {g.name: walk_cycles(g) for g in levels}
        common = {"config": cfg, "L1D_lines_per_bank": levels[0].lines_per_bank, "L1D_walk": walks["L1D"],
                  "L2_lines_per_bank": levels[1].lines_per_bank, "L2_walk": walks["L2"]}
        for ct, dcr, note in [("policy_D_insert", 1, "VX_DCR_PHASE_INSERT_POLICY"),
                              ("policy_B_dram_arb", 2, "VX_DCR_PHASE_DRAM_SCHED + _DRAM_AGING"),
                              ("policy_C_dxa_depth", 2, "VX_DCR_PHASE_DXA_DEPTH0 + _DEPTH1"),
                              ("LMEM_A1", 0, "kernel-boundary (build/launch) change; every launch already starts cold")]:
            csw.append({"change_type": ct, **common, "dirty_case": "n/a", "dirty_sectors": 0, "t_wr": "",
                        "c_sw_cycles": 0.0, "host_dcr_writes": dcr, "label": "by construction (host DCR write / launch boundary; 0 device cycles)",
                        "note": note})
        dirty_cases = [("none", 0), ("all_L2_dirty", levels[1].size // SECTOR_BYTES)]
        dirty_cases += [(f"measured_end_of_{w['label']}", w["flush_wb_sectors"]) for w in warm if w["config"] == cfg]
        for tname, t in T_WR[cfg].items():
            for dc, ds in dirty_cases:
                csw.append({"change_type": "waymask_A2_flush", **common, "dirty_case": dc, "dirty_sectors": ds,
                            "t_wr": t, "c_sw_cycles": flush_cycles(levels, ds, t), "host_dcr_writes": 1,
                            "label": ANALYTICAL, "note": f"walk L1D+L2 + dirty x t_wr ({tname})"})
    write_csv(res / "csw.csv", csw)
    # --- medians and N_min
    med = medians()
    write_csv(res / "medians.csv", med)
    m_by = {(m["sigma_case"], m["max_context"]): m["median_decode_len"] for m in med}
    ps_warm = [w for w in warm if w["config"] == "PS-BASE-S1-4c"]
    cw_max = max(w["c_warmup_kP"] for w in ps_warm if w["label"].startswith("decode"))
    cw_mixed = max(w["c_warmup_kP"] for w in ps_warm if w["label"].startswith("mixed"))
    a2 = [r for r in csw if r["config"] == "PS-BASE-S1-4c" and r["change_type"] == "waymask_A2_flush"
          and r["note"].endswith("(meas_stream)")]
    a2_worst = max(r["c_sw_cycles"] for r in a2)
    a2_meas = max(r["c_sw_cycles"] for r in a2 if r["dirty_case"].startswith("measured"))
    cases = [("policy_only (D/B/C)", 0.0, 0.0, "0 (no state lost)"),
             ("LMEM_A1 (kernel boundary)", 0.0, 0.0, "0 (launch is cold anyway)"),
             ("A2 flush, measured dirty, + C_warmup", a2_meas, cw_max, "max decode C_warmup(k=P), PS-BASE-S1-4c"),
             ("A2 flush, all L2 dirty, + C_warmup", a2_worst, cw_max, "max decode C_warmup(k=P), PS-BASE-S1-4c"),
             ("A2 flush, all L2 dirty, + mixed estimator (upper bound)", a2_worst, cw_mixed,
              "mixed_attn literal estimator (heterogeneous; not a pure warm-up)")]
    nm = []
    for name, csw_c, cw, cwnote in cases:
        for f in FRACTIONS:
            saved = f * STEP_CYCLES
            row = {"change_type": name, "C_enter": csw_c, "C_exit": csw_c, "C_warmup": cw, "C_warmup_src": cwnote,
                   "step_cycles": STEP_CYCLES, "saving_frac": f, "dC_saved": saved,
                   "N_min": n_min(csw_c, csw_c, cw, saved),
                   "label": (ANALYTICAL + " C_sw + measured C_warmup") if csw_c else "by construction"}
            for (sc, mc), mv in m_by.items():
                row[f"median[{sc}{' ctx' + str(mc) if mc else ''}]"] = mv
            nm.append(row)
    write_csv(res / "nmin.csv", nm)
    for (sc, mc), mv in m_by.items():
        print(f"median O' azure_conv {sc} ctx={mc or '-'}: {mv}")
    for name, csw_c, cw, _ in cases:
        tot = 2 * csw_c + cw
        for (sc, mc), mv in m_by.items():
            if tot:
                print(f"{name}: C_total={tot:.0f} cyc; f* for N_min<=median({sc},{mc or '-'})={mv}: "
                      f"{breakeven_fraction(tot, mv, STEP_CYCLES):.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
