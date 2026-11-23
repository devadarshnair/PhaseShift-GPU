"""TEST: analytical flush formula, N_min (E9) and the intra-kernel C_warmup estimator.

Cheap tests (no SimX). Methodology §13; code scripts/csw.py.
"""
import math
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS / "scripts"))

import csw  # noqa: E402


# ---------------------------------------------------------------- flush formula: cache_subsystem.md:684-692 example
def test_worked_example_l2_1mb_8way_4bank_128B():
    l2 = csw.CacheGeom("L2", 1 << 20, 8, 4, 128, True)
    assert l2.lines_per_bank == 256                       # 1 MiB / (4 x 8 x 128 B)
    assert csw.walk_cycles(l2) == 256 * 8 == 2048         # write-back: LINES_PER_BANK x NUM_WAYS
    wrong = csw.CacheGeom("L2", 1 << 20, 8, 4, 64, True)  # a 64-byte line would wrongly give 512 lines
    assert wrong.lines_per_bank == 512 and csw.walk_cycles(wrong) == 4096


def test_write_through_walks_lines_once():
    l1 = csw.CacheGeom("L1D", 16384, 4, 4, 64, False)
    assert l1.lines_per_bank == 16 and csw.walk_cycles(l1) == 16   # all ways cleared in parallel
    wb = csw.CacheGeom("L1D", 16384, 4, 4, 64, True)
    assert csw.walk_cycles(wb) == 64


def test_flush_cycles_sum_of_walks_plus_writebacks():
    lv = csw.GEOM["PS-BASE-S1-4c"]
    assert csw.flush_cycles(lv, 0, 1.5) == 16 + 1024
    assert csw.flush_cycles(lv, 100, 1.5) == 16 + 1024 + 150
    with pytest.raises(ValueError):
        csw.flush_cycles(lv, -1, 1.0)
    with pytest.raises(ValueError):
        csw.CacheGeom("x", 1000, 8, 4, 128, True).lines_per_bank


def test_psbase_geometry_matches_resolved_header():
    """GEOM must equal what results/<run>/l2_geometry.cpp resolved from the ps tree's VX_config.h."""
    txt = (PS / "results" / "P4.4" / "l2_geometry.out").read_text()
    sec = txt.split("== PSBASE")[1].split("== REG")[0]
    assert "dcache size=16384 ways=4 line=64 banks=4 writeback=0" in sec
    assert "l2 size=1048576 ways=8 line=128 banks=8 writeback=1" in sec
    l1, l2 = csw.GEOM["PS-BASE-S1-4c"]
    assert (l1.size, l1.ways, l1.line, l1.banks, l1.writeback) == (16384, 4, 64, 4, False)
    assert (l2.size, l2.ways, l2.line, l2.banks, l2.writeback) == (1048576, 8, 128, 8, True)
    reg = txt.split("== REG")[1]
    assert "dcache size=16384 ways=4 line=64 banks=2 writeback=0" in reg
    assert "l2 size=1048576 ways=8 line=128 banks=8 writeback=1" in reg


# ---------------------------------------------------------------- N_min: hand calculations
def test_n_min_hand_calculation():
    # (1040 + 1040 + 11196) / 7383.6 = 1.798 -> 2
    assert csw.n_min(1040, 1040, 11196, 7383.6) == 2
    assert csw.n_min(100, 100, 0, 50) == 4            # exactly 4.0 -> 4 (no spurious round-up)
    assert csw.n_min(100, 100, 1, 50) == 5            # 4.02 -> 5
    assert csw.n_min(0, 0, 0, 10) == 0                # free switch: pays from the first iteration
    with pytest.raises(ValueError):
        csw.n_min(1, 1, 1, 0)
    assert csw.breakeven_fraction(1000.0, 10, 1e4) == pytest.approx(0.01)
    # consistency: at f = f* the N_min equals the median
    f = csw.breakeven_fraction(64794.0, 38, 7.38e6)
    assert csw.n_min(64794.0 / 2, 64794.0 / 2, 0, f * 7.38e6) == 38


# ---------------------------------------------------------------- C_warmup estimator on a synthetic trace
def _trace(per_unit: list[int], tail: int = 0) -> list[dict]:
    rows, cyc = [], 0
    for k, c in enumerate(per_unit, 1):
        cyc += c + (tail if k == len(per_unit) else 0)
        rows.append({"kind": "work", "launch": 0, "work_end": k, "work_at": k, "cycle": cyc,
                     "cycles": c + (tail if k == len(per_unit) else 0)})
    return rows


def test_c_warmup_synthetic_bursts():
    # 4 cores in lockstep: each round = one long unit + three short ones (bimodal, like the real traces)
    steady_round = [9000, 300, 400, 300]                    # 10,000 cycles per round of P = 4 -> s = 2,500
    cold_round = [14000, 300, 400, 300]                     # first round 5,000 cycles slower
    rows = _trace(cold_round + steady_round * 7, tail=777)  # W = 32, R = 8, last row holds an epilogue
    w = csw.c_warmup(rows, 4)
    assert (w.units, w.a, w.b) == (32, 16, 28)
    assert w.steady == pytest.approx(2500.0)
    assert w.first_k == 15000 and w.c_warmup == pytest.approx(5000.0)
    assert csw.c_warmup(rows, 4, 8).c_warmup == pytest.approx(5000.0)   # warm-up confined to round 1
    assert w.kernel_cycles == 15000 + 7 * 10000 + 777
    assert csw.round_cycles(rows, 4) == [15000] + [10000] * 6 + [10777]


def test_c_warmup_zero_when_no_cold_start_and_needs_rounds():
    rows = _trace([100] * 12)
    assert csw.c_warmup(rows, 4).c_warmup == pytest.approx(0.0)
    with pytest.raises(ValueError):
        csw.c_warmup(_trace([100] * 8), 4)                 # R = 2 < 3


def test_c_warmup_on_stored_trace_matches_hand_value():
    """decode T128 (trace): s = (cum(12) - cum(8)) / 4 and C_warmup = cum(4) - 4 s, from the raw rows."""
    import pstrace
    f = PS / "results" / "P4.4" / "traces" / "decode_T128_n1.csv"
    if not f.exists():
        pytest.skip("trace not generated")
    _, rows = pstrace.read_trace(f)
    wr = pstrace.work_rows(rows, 0)
    cum = {r["work_end"]: r["cycle"] for r in wr}
    s = (cum[12] - cum[8]) / 4
    w = csw.c_warmup(wr, 4)
    assert w.c_warmup == pytest.approx(cum[4] - 4 * s)
    assert w.kernel_cycles == sum(r["cycles"] for r in wr)


def test_outputs_label_every_analytical_number():
    import csv
    p = PS / "results" / "P4.4" / "csw.csv"
    if not p.exists():
        pytest.skip("csw.csv not generated")
    for r in csv.DictReader(open(p)):
        if r["change_type"].startswith("waymask"):
            assert r["label"] == csw.ANALYTICAL
        else:
            assert float(r["c_sw_cycles"]) == 0.0 and r["label"].startswith("by construction")
    for r in csv.DictReader(open(PS / "results" / "P4.4" / "nmin.csv")):
        assert ("analytical (RTL formula)" in r["label"]) == (float(r["C_enter"]) > 0)
        n = csw.n_min(float(r["C_enter"]), float(r["C_exit"]), float(r["C_warmup"]), float(r["dC_saved"]))
        assert n == int(r["N_min"]) and n >= 0 and not math.isnan(n)
