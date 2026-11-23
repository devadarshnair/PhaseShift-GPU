"""oracle study (results/<run>/oracle_study.py): byte-identical recomputation of every analysis output, the methodology §5 ordering per workload,
DP = zero-cost when C_sw = 0, an independent recomputation of the G2/G2b region flags from oracle.csv (rules),
the oracle-schedule JSON, the PS_SCHEDULE files, and (when present) the replay table."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PS = Path(__file__).resolve().parents[1]
P62 = PS / "results" / "P6.2"
sys.path.insert(0, str(P62))
sys.path.insert(0, str(PS / "scripts"))
if not (PS / "results" / "P6.1" / "costs.csv").exists() or not (P62 / "oracle.csv").exists():
    pytest.skip("cost table / outputs absent", allow_module_level=True)

import oracle_study as OS  # noqa: E402
import mkschedule as MS  # noqa: E402

REL = 1e-9


def _oracle():
    return list(csv.DictReader(open(P62 / "oracle.csv")))


def _geo(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def test_recomputation_is_deterministic():
    """Every file written by `analyze` equals a fresh in-memory recomputation byte for byte (two runs, too)."""
    a = OS.outputs()
    b = OS.outputs()
    assert a == b
    for name, text in a.items():
        assert (P62 / name).read_text() == text, name


def test_ordering_and_dp_equals_zero_cost():
    for r in _oracle():
        tz, tdp, tk, ts, tg = (float(r[k]) for k in ("T_zero", "T_dp", "T_kernel", "T_static", "T_global_static"))
        assert tz <= tdp * (1 + REL) and tdp <= tk * (1 + REL) and tk <= ts * (1 + REL) and ts <= tg * (1 + REL)
        # every configuration differs only in policy-only knobs: C_sw = 0, so DP = zero-cost oracle
        assert tdp == pytest.approx(tz, rel=REL)
        assert float(r["speedup_dp_vs_static"]) == pytest.approx(ts / tdp, rel=1e-8)
        assert float(r["dram_ratio_dp_vs_static"]) == pytest.approx(float(r["dram_dp"]) / float(r["dram_static"]),
                                                                   rel=1e-8)


def test_gates_recomputed_independently():
    """G2 passes iff >= 2 regions have geomean DP/static speedup >= 1.10 and CI lower bound - 1 > bound;
    G2b stops iff the DP/per-kernel geomean speedup is < 1.05 in every region."""
    rows = _oracle()
    summ = {r["region"]: r for r in csv.DictReader(open(P62 / "gap_by_region.csv"))}
    assert set(summ) == set(OS.REGIONS)
    n_pass, below5 = 0, []
    for reg in OS.REGIONS:
        rs = [r for r in rows if r["region"] == reg]
        assert len(rs) == int(summ[reg]["n"]) >= 10
        g = _geo([float(r["speedup_dp_vs_static"]) for r in rs])
        gk = _geo([float(r["speedup_dp_vs_kernel"]) for r in rs])
        assert g == pytest.approx(float(summ[reg]["speedup_dp_vs_static"]), rel=1e-8)
        assert gk == pytest.approx(float(summ[reg]["speedup_dp_vs_kernel"]), rel=1e-8)
        ok = g - 1 >= 0.10 and float(summ[reg]["speedup_dp_vs_static_lo"]) - 1 > OS.BOUND
        assert int(summ[reg]["G2_region_pass"]) == int(ok)
        n_pass += ok
        below5.append(gk - 1 < 0.05)
        assert int(summ[reg]["G2b_region_below5"]) == int(below5[-1])
    meta = json.loads((P62 / "schedules.json").read_text())["meta"]
    assert meta["G2"] == ("PASS" if n_pass >= 2 else "FAIL")
    assert meta["G2b"] == ("FAIL (STOP)" if all(below5) else "PASS")
    assert meta["bound"] == OS.BOUND >= 0.0104


def test_des_oracle_schedule():
    from serving import des
    pol = des.load_oracle_schedule(P62 / "oracle_schedule.json")
    assert set(pol.by_region) == set(des.REGIONS)
    rows = _oracle()
    for reg, key in OS.DES_REGION.items():
        r = _geo([float(o["T_dp"]) / float(o["T_default"]) for o in rows if o["region"] == reg])
        assert pol.by_region[key]["ratio"] == pytest.approx(r, abs=2e-9)
        assert 0 < pol.by_region[key]["ratio"] <= 1.0


def test_schedule_files_valid():
    files = sorted((P62 / "sched").glob("*.sched"))
    assert files
    for f in files:
        lines = MS.parse(f.read_text())
        assert lines and lines[0].work == 0
        assert all(a.work <= b.work for a, b in zip(lines, lines[1:]))
        assert {k for ln in lines for k, _ in ln.sets} <= set(OS.KNOB_ORDER)


def test_replay_table():
    p = P62 / "replay.csv"
    if not p.exists():
        pytest.skip("replay not evaluated yet")
    rp = list(csv.DictReader(open(p)))
    orc = {r["workload"]: r for r in _oracle()}
    for reg in OS.REGIONS:
        assert sum(r["region"] == reg for r in rp) >= 10, reg
    for r in rp:
        comp, rep = float(r["composed_iter"]), float(r["replayed_iter"])
        assert comp == pytest.approx(float(orc[r["workload"]]["T_dp"]), rel=REL)
        # CSV values carry 9 significant digits, so a recomputed error can differ by ~1e-9 absolute
        assert float(r["err_iter"]) == pytest.approx(abs(comp - rep) / rep, abs=1e-8)
    summ = {r["region"]: r for r in csv.DictReader(open(P62 / "replay_summary.csv"))}
    e = np.array([float(r["err_iter"]) for r in rp])
    assert float(summ["all"]["p95_err_iter"]) == pytest.approx(float(np.percentile(e, 95)), abs=1e-8)
