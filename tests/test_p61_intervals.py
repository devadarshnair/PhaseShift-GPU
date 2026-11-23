"""interval extraction (results/<run>/costlib.py): regrouping of an n = 1 trace, flush folding, and (when the
 re-trace runs exist) equality with a trace taken directly at the final n (methodology §6.6.C)."""
import csv
import json
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS / "results" / "P6.1"))
import costlib as CL  # noqa: E402

COLS = ["launch", "row", "kind", "interval", "unit", "every", "work_end", "work_at", "cycle", "cycles", "instrs",
        "marks_prefill", "marks_decode", "marks_other"]


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, restval="0")
        w.writeheader()
        w.writerows(rows)


def _trace(n, W, cyc):
    rows, k = [], 0
    for i in range(W):
        rows.append(dict(launch=0, row=i, kind="work", interval=i + 1, unit="marks", every=n, work_end=i + 1,
                         work_at=i + 1, cycle=0, cycles=cyc[i], instrs=1, marks_decode=1))
    rows.append(dict(launch=0, row=W, kind="flush", interval=0, unit="marks", every=n, work_end=W, work_at=W,
                     cycle=0, cycles=300, instrs=0))
    return rows


def test_regroup_n1(tmp_path):
    p = tmp_path / "t.csv"
    cyc = [10, 1, 20, 2, 30, 3, 40]
    _write(p, _trace(1, 7, cyc))
    iv = CL.intervals(p, 3)
    assert [x["cycles"] for x in iv] == [31, 35, 40 + 300]
    assert [x["work_end"] for x in iv] == [3, 6, 7]
    assert iv[-1]["flush_cycles"] == 300
    assert sum(x["cycles"] for x in iv) == sum(cyc) + 300
    assert len(iv) == CL.expected_rows(7, 3)


def test_unmarked_one_interval(tmp_path):
    p = tmp_path / "u.csv"
    _write(p, [dict(launch=0, row=0, kind="work", interval=1, unit="marks", every=1, work_end=0, work_at=0, cycle=5,
                    cycles=5000, instrs=9),
               dict(launch=0, row=1, kind="flush", interval=0, unit="marks", every=1, work_end=0, work_at=0, cycle=0,
                    cycles=7, instrs=0)])
    iv = CL.intervals(p, 4)
    assert len(iv) == 1 and iv[0]["cycles"] == 5007


def test_bad_multiple(tmp_path):
    p = tmp_path / "b.csv"
    t = _trace(2, 4, [1, 2, 3, 4])
    _write(p, t)
    with pytest.raises(ValueError):
        CL.intervals(p, 3)


def _retrace_pairs():
    rows = PS / "results" / "P6.1" / "rows.jsonl"
    if not rows.exists():
        return []
    out = []
    for line in rows.read_text().splitlines():
        r = json.loads(line)
        t = r.get("tags") or {}
        if r.get("status") == "ok" and t.get("extra") == "retrace":
            base = r["env"]["PS_TRACE"].replace("_base_retrace.csv", "_base.csv")
            if Path(base).exists() and Path(r["env"]["PS_TRACE"]).exists():
                out.append((base, r["env"]["PS_TRACE"], int(r["env"]["PS_TRACE_EVERY"])))
    return out


@pytest.mark.parametrize("pair", _retrace_pairs() or [None])
def test_regroup_equals_retrace(pair):
    if pair is None:
        pytest.skip("no re-trace runs yet")
    base, direct, n = pair
    a = CL.intervals(base, n)
    b = CL.intervals(direct, n)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x == y
