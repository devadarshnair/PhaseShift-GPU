"""Tests: edge static characterization, SW*, the (methodology §18.10).

Non-vortex (always run; no SimX):
  test_batch_files_up_to_date      experiments/<run>_*.yaml == results/<run>/make_batch.py (the `res` group needs sw_star.json)
  test_near_tie_rule               sw_select.sw_star on synthetic candidates: min wins; within 0.5 % of the min the simplest wins (fewest DXA stages, then the default)
  test_gate_formula                compose.bound: U = sum f_k h_k, S* = 1/(1-U) - 1, on a hand-computed example; aggregation = geomean(1+S) - 1 (defined at S = 0)
  test_canonical_args              argument identity ignores flag order (`-L 1 -G 32` == `-G 32 -L 1`)
  test_p95_nearest_rank            the eps_EP percentile definition
  test_forced128_golden            derived golden: 127 forced lines, tokens unchanged vs the llama2.c golden, make_golden.py --check
  test_rows_ok_and_passed          every row is status ok + passed (failed rows are recorded in the transcripts, not stored as ok)
  test_stored_results              (when results exist) launch classes complete; every launch group of every workload/track has a cycle count; launch counts of a token
                                   (14 L + 2, +4 L + 1 with the Q8_0 quantizer); every resident pair meets the launch-2 L2-miss criterion (<= 5 %); h_mem* in [0, 1);
                                   headroom_star.csv recomputes (§2.7 aggregation) to the number in summary.json; analyze.py --check byte-identical
  test_methodology_18_10           documentation present
Vortex-marked (PS_RUN_VORTEX=1; outer-flock pattern of results/<run>/run_conditions.sh; NOT under lowload.sh):
  test_three_rows_reproduce        3 stored rows (2 pseudo-random cold / resident rows < 3 M cycles + the anomalous cls B = 1 control launch) re-run in a private step directory reproduce their cycles exactly
"""
import csv
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
EP4 = PS / "results" / "EP.4"
sys.path.insert(0, str(EP4))
sys.path.insert(0, str(PS / "scripts"))


def _mods():
    import compose as C
    import ep4lib as L
    import make_batch as MB
    import sw_select as SW
    return L, C, MB, SW


def _has_results():
    return (EP4 / "summary.json").is_file()


def test_batch_files_up_to_date():
    _, _, MB, _ = _mods()
    stale = []
    for g in MB.GROUPS:
        p = PS / "experiments" / f"EP.4_{g}.yaml"
        if p.is_file() and p.read_text() != MB.dump(g):
            stale.append(p.name)
    assert not stale


def _cand(track, v, cyc, B=1):
    return dict(track=track, model="stories15M", matrix="qkvo", M=288, K=288, GS=32, B=B, variant=v, app="a", config="c", args=f"-m 288 -L 1 -v {v}", cycles=cyc, instrs=1, dram_rd=1, src="t")


def test_near_tie_rule(monkeypatch):
    L, _, _, SW = _mods()
    monkeypatch.setattr(SW, "default_variant", lambda track, B, K: "rows")
    # clear winner
    assert SW.sw_star([_cand("P8", "rows", 1000), _cand("P8", "rows_inl", 900)])[0]["variant"] == "rows_inl"
    # rows_inl is 0.4 % faster: tied -> the default `rows`
    assert SW.sw_star([_cand("P8", "rows", 1000), _cand("P8", "rows_inl", 996)])[0]["variant"] == "rows"
    # rows_inl 0.6 % faster: not tied
    assert SW.sw_star([_cand("P8", "rows", 1000), _cand("P8", "rows_inl", 994)])[0]["variant"] == "rows_inl"
    # a DXA kernel within 0.5 % of a non-DXA one loses; among DXA depths the smaller wins
    assert SW.sw_star([_cand("P32", "dxa_d2", 1000), _cand("P32", "rows_ilp4", 1004)])[0]["variant"] == "rows_ilp4"
    assert SW.sw_star([_cand("P32", "dxa_d4", 1000), _cand("P32", "dxa_d2", 1003)])[0]["variant"] == "dxa_d2"
    # own tracks never enter another track's minimum
    c = SW.sw_star([_cand("P8", "rows", 1000), _cand("P8T", "tcu_i8", 100)])
    assert len(c) == 1 and c[0]["variant"] == "rows"


def test_gate_formula():
    L, C, _, _ = _mods()
    la = [dict(kind="gemm", n=2, cycles=300, h=0.10), dict(kind="gemm", n=1, cycles=200, h=0.30), dict(kind="attn", n=1, cycles=200, h=None)]
    Ct, fW, fa, U, S = C.bound(la)
    assert Ct == 1000 and fW == pytest.approx(0.8) and fa == pytest.approx(0.2)
    assert U == pytest.approx(0.6 * 0.10 + 0.2 * 0.30)   # 0.12
    assert S == pytest.approx(1 / (1 - 0.12) - 1)
    la[0]["h"] = None
    assert C.bound(la)[4] is None
    assert L.geo([0.0, 0.0]) == pytest.approx(0.0)
    assert L.geo([0.21, 0.0]) == pytest.approx(math.sqrt(1.21) - 1)


def test_canonical_args():
    L, _, _, _ = _mods()
    assert L.canon("-m 288 -n 288 -w q80 -d 2 -B 1 -L 1 -G 32") == L.canon("-m 288 -n 288 -w q80 -d 2 -B 1 -G 32 -L 1")
    assert L.canon("-m 1 -L 1") != L.canon("-m 1 -L 2")


def test_p95_nearest_rank():
    import importlib.util   # by path: other steps' tests also put an `analyze.py` on sys.path
    spec = importlib.util.spec_from_file_location("ep4_analyze", EP4 / "analyze.py")
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    assert analyze.p95([0.01 * i for i in range(1, 21)]) == pytest.approx(0.19)    # n = 20: rank ceil(19) = 19
    assert analyze.p95([5.0]) == 5.0


def test_forced128_golden():
    g = EP4 / "golden_forced128_15M.txt"
    s8 = PS / "results" / "S8" / "golden_tokens_15M.txt"
    rows = [ln.split() for ln in g.read_text().splitlines() if ln and not ln.startswith("#")]
    base = [ln.split() for ln in s8.read_text().splitlines() if ln and not ln.startswith("#")]
    assert len(rows) == len(base) == 200
    assert sum(int(r[3]) for r in rows[:127]) == 127 and all(int(r[3]) == int(b[3]) for r, b in zip(rows[127:], base[127:]))
    assert [r[:3] for r in rows] == [b[:3] for b in base]
    assert subprocess.run([sys.executable, "-B", str(EP4 / "make_golden.py"), "--check"]).returncode == 0


def _rows():
    p = EP4 / "rows.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def test_rows_ok_and_passed():
    last = {}
    for r in _rows():
        last[r["run_key"]] = r
    bad = [r["args"] for r in last.values() if r["status"] != "ok" or not r.get("passed")]
    assert not bad, bad[:5]


@pytest.mark.skipif(not _has_results(), reason="results not written yet")
def test_stored_results():
    L, C, _, SW = _mods()
    s = json.loads((EP4 / "summary.json").read_text())
    t = s["tests"]
    assert t["all_workload_tracks_complete"] and not t["missing_expected_cells"]
    assert t["all_resident_pairs_launch2_miss_le_5pct"]
    lo, hi = t["h_mem_star_range"]
    assert -0.10 <= lo and hi < 1   # one documented class is negative (P8 llama8b_s8 cls B = 1, rows_inl: -9.0 %, README)
    # token launch counts
    cls = {(c["track"], c["model"], c["matrix"], c["B"]): c for c in json.loads((EP4 / "sw_star.json").read_text())["classes"]}
    for model, Lm in (("stories15M", 6), ("llama8b_s8", 32)):
        for track, extra in (("P32", 0), ("P8", 4 * Lm + 1), ("P8K8", 4 * Lm + 1)):
            la = C.token(model, 1, 128, track, cls)
            n = sum(x["n"] for x in la)
            assert n == 14 * Lm + 2 + extra, (model, track, n)
            assert all(x["cycles"] > 0 for x in la)
    # recomputed from headroom_star.csv
    rows = list(csv.DictReader((EP4 / "headroom_star.csv").open()))
    U = {}
    for r in rows:
        U[(r["workload"], r["track"])] = U.get((r["workload"], r["track"]), 0.0) + float(r["f_h"])
    for track in ("P8", "P32", "P8K8"):
        v = [1 / (1 - u) - 1 for (w, tr), u in U.items() if tr == track]
        assert len(v) == 18
        assert L.geo(v) == pytest.approx(s["gate"][track]["G"], abs=2e-6)
    assert subprocess.run([sys.executable, "-B", str(EP4 / "analyze.py"), "--check"]).returncode == 0


def test_repeat_substitution_includes_classifier(monkeypatch):
    """the "B repeated B = 1 launches" sensitivity applies to the per-layer matrices AND the classifier;
    strict = any gain, tie = the repeat must be > 0.5 % faster (2.4)."""
    _, C, _, _ = _mods()
    monkeypatch.setattr(C, "hmem", lambda c: dict(h=0.1, C_res=c["cycles"] * 0.9))
    def cls(cyc, v):
        return dict(cycles=cyc, instrs=10, dram_rd=5, variant=v, app="a", args="x")
    classes = {("P8", "stories15M", "cls", 1): cls(100, "rows"), ("P8", "stories15M", "cls", 2): cls(200, "rows_inl"),
               ("P8", "stories15M", "cls", 4): cls(400, "rows_inl")}
    # B = 2: 2 * 100 = 200 is not < 200 (strict) -> batched kept; B = 4 with c = 401: 400 < 401 strict yes, tie: 400 < 401 * 0.995 = 399.0 no
    cyc, *_ , c = C._substitute(classes[("P8", "stories15M", "cls", 2)], classes, "P8", "stories15M", "cls", 2, "strict", False)
    assert cyc == 200 and c["variant"] == "rows_inl"
    classes[("P8", "stories15M", "cls", 4)] = cls(401, "rows_inl")
    cyc, ins, dr, h, c = C._substitute(classes[("P8", "stories15M", "cls", 4)], classes, "P8", "stories15M", "cls", 4, "strict", False)
    assert (cyc, ins, dr, c["variant"]) == (400, 40, 20, "rowsx4") and h["h"] == 0.1 and h["C_res"] == pytest.approx(360)
    assert C._substitute(classes[("P8", "stories15M", "cls", 4)], classes, "P8", "stories15M", "cls", 4, "tie", False)[0] == 401
    classes[("P8", "stories15M", "cls", 4)] = cls(500, "rows_inl")
    assert C._substitute(classes[("P8", "stories15M", "cls", 4)], classes, "P8", "stories15M", "cls", 4, "tie", False)[0] == 400
    # B = 1 and repeat=False never substitute
    assert C._substitute(classes[("P8", "stories15M", "cls", 4)], classes, "P8", "stories15M", "cls", 4, False, False)[0] == 500


@pytest.mark.skipif(not _has_results(), reason="results not written yet")
def test_repeat_sensitivity_pinned():
    """values of the corrected sensitivity (classifier included), independently recomputed: P8 9.81 % strict / 10.08 % tie,
    P8K8 9.73 / 9.99, P32 15.45 (repeat never wins there)"""
    g = json.loads((EP4 / "summary.json").read_text())["gate"]
    assert g["P8"]["G_repeat"] == pytest.approx(0.0981, abs=5e-5) and g["P8"]["G_repeat_tie"] == pytest.approx(0.1008, abs=5e-5)
    assert g["P8K8"]["G_repeat"] == pytest.approx(0.0973, abs=5e-5) and g["P8K8"]["G_repeat_tie"] == pytest.approx(0.0999, abs=5e-5)
    assert g["P32"]["G_repeat"] == pytest.approx(0.1545, abs=5e-5) == pytest.approx(g["P32"]["G_repeat_tie"], abs=5e-5)


@pytest.mark.skipif(not _has_results(), reason="results not written yet")
def test_capacity_residency_split():
    """information table: cap + res = h* per class, so U_cap + U_res = U of the gate (same shares)"""
    s = json.loads((EP4 / "summary.json").read_text())["capacity_residency_split"]
    p8 = s["P8|all"]
    assert p8["S_capacity"] == pytest.approx(0.1142, abs=5e-5) and p8["S_residency"] == pytest.approx(0.0393, abs=5e-5)
    assert s["P32|all"]["S_capacity"] == pytest.approx(0.0026, abs=5e-5) and s["P32|all"]["S_residency"] == pytest.approx(0.1510, abs=5e-5)


def test_methodology_18_10():
    txt = (PS / "docs" / "methodology.md").read_text()
    assert "### 18.10" in txt and "SW*" in txt.split("### 18.10", 1)[1]


@pytest.mark.vortex
def test_three_rows_reproduce(tmp_path):
    rows = [r for r in _rows() if r["status"] == "ok" and r["perf"] in (7, 5)]
    rnd = random.Random(4)
    pick = rnd.sample([r for r in rows if r["cycles"] < 3_000_000], 2)
    # plus the one anomalous resident pair of the gate (P8 llama8b_s8 cls B = 1 rows_inl: h_mem* = -9 %): its control launch 1 must reproduce exactly
    pick += [r for r in rows if r["app"] == "ps_gemv_q80" and "-m 16032 -n 512" in r["args"] and "-v rows_inl" in r["args"] and "-B 1" in r["args"] and "-L 1 -P" in r["args"]][:1]
    py = Path.home() / "miniconda3/envs/phaseshift/bin/python"
    for r in pick:
        step = "EP.4rerun"
        cmd = [str(py), "-B", str(PS / "scripts" / "vxrun.py"), "run", "--step", step, "--tree", "ps", "--app", r["app"], "--args", r["args"], "--config", r["config"],
               "--perf", str(r["perf"]), "--force"]
        env = dict(os.environ, PS_LOCK=os.environ.get("PS_LOCK", "/tmp/ep4-inner.lock"))
        subprocess.run(cmd, check=True, env=env, cwd=PS)
    new = [json.loads(x) for x in (PS / "results" / step / "rows.jsonl").read_text().splitlines()]
    by = {(n["app"], n["args"], n["config"], n["perf"]): n for n in new}
    for r in pick:
        assert by[(r["app"], r["args"], r["config"], r["perf"])]["cycles"] == r["cycles"], r["args"]
