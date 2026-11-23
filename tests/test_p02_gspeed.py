"""Simulation-speed tier (G-SPEED): the decision record's numbers are the ones results/<run>/summarize.py derives from rows.jsonl.

Light (reads results/<run>/rows.jsonl only). Checks: every rate uses the rows' /usr/bin/time wall; R and the tier
follow the table; contains the generated block verbatim.
"""
import importlib.util
import json
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
SUM = PS / "results" / "P0.2" / "summarize.py"

pytestmark = pytest.mark.skipif(not (PS / "results/P0.2/rows.jsonl").exists(), reason="rows not present")


def _mod():
    spec = importlib.util.spec_from_file_location("p02_summarize", SUM)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_rates_use_usr_bin_time_walls_and_tier():
    m = _mod()
    rows = m.load_rows()
    out, ok = m.summarize(rows)
    assert ok
    by_key = {r["run_key"]: r for r in rows}
    for x in out:
        assert x["cycles_per_s"] == round(x["cycles"] / x["sim_wall_s"])
        assert x["instrs_per_s"] == round(x["instrs"] / x["sim_wall_s"])
    t = m.pick_tier(out)
    assert t["case"] == "tier-256" and t["args"] == "-m 256 -n 256 -k 64"
    rep1 = [r for r in by_key.values() if r["tags"].get("case") == "tier-256" and r["rep"] == 1 and not r["direct"]
            and r["status"] == "ok"][-1]
    assert t["sim_wall_s"] == rep1["wall_s"]                      # the /usr/bin/time wall, not the ledger wall
    assert t["cycles_per_s"] == round(58679 / 1.40) == 41914
    assert m.tier_of(t["cycles_per_s"]) == "C"
    assert [m.tier_of(v) for v in (29_999, 30_000, 199_999, 200_000, 1_000_000)] == ["STOP", "C", "C", "B", "A"]


def test_decision_record_block_in_sync():
    m = _mod()
    out, _ = m.summarize(m.load_rows())
    t = m.pick_tier(out)
    block = m.render(out, t, m.budgets(out, t))
    doc = (PS / "docs/decisions/G-SPEED.md").read_text()
    assert m.docs_block(doc) == block
    assert "58,679 cycles / 1.4 s" in block and "41,914 cycles/s ⇒ tier C" in block
    for b in m.budgets(out, t):   # cap = R_c * ((W - n_b*90)/N - o)
        per_run = (b["wall_budget_h"] * 3600 - b["builds_est"] * m.BUILD_S) / b["runs_est"] - b["overhead_s"]
        assert abs(b["cap_cycles_per_run"] - b["R_cycles_per_s"] * per_run) <= b["R_cycles_per_s"] * 0.06 + 1
    json.dumps(out)   # summary is plain data
