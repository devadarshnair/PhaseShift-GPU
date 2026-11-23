"""proxy-mode tests (no simulator): regeneration, hand example, cross-check against tokens.csv, pinned values, no energy units; trigger pins."""
import csv, json, math, re, subprocess, sys
from pathlib import Path
PS = Path(__file__).resolve().parents[1]
R7, R6 = PS / "results" / "EP.7", PS / "results" / "EP.6"
S4B = "EP.4b SW*'"

def rows(p): return list(csv.DictReader(open(p)))

def test_regeneration_identical():
    assert subprocess.run([sys.executable, "-B", str(R7 / "proxy.py"), "--check"], capture_output=True, text=True).returncode == 0

def test_hand_example_rmsnorm():
    # stories15M B=1: 2L+1 = 13 rmsnorm launches; one launch: reads=79, writes=34 requests of 64 B (stored perf-7 row of ps_rmsnorm -b 1 -n 288)
    k = [r for r in rows(R7 / "proxy_by_kind.csv") if r["source"] == S4B and r["workload"] == "st_B1_T8" and r["track"] == "P8" and r["kind"] == "rmsnorm"][0]
    assert int(k["dram_rd_per_step"]) == 13 * 64 * 79 == 65728
    assert int(k["dram_wr_per_step"]) == 13 * 64 * 34 == 28288

def test_matches_ep4b_tokens():
    tok = {(r["workload"], r["track"]): int(float(r["dram_rd_bytes"])) for r in rows(PS / "results/EP.4b/tokens.csv")}
    n = 0
    for r in rows(R7 / "proxy_token.csv"):
        if r["source"] == S4B:
            d = tok[(r["workload"], r["track"])] - int(r["dram_rd_per_step"])
            # tokens.csv keeps the negative (unclamped) rope setup subtraction; clamps at 0, so only flagged rows may differ, and only upward
            assert d == 0 or (r["rope_clamped"] == "1" and d < 0); n += 1
    assert n == 54

def test_pinned_values():
    d = {(r["source"], r["workload"], r["track"]): r for r in rows(R7 / "proxy_token.csv")}
    a = d[(S4B, "st_B1_T8", "P8")]
    assert (int(a["dram_rd_per_step"]), int(a["dram_wr_per_step"]), int(a["lmem_per_step"]), int(a["l2_acc_per_step"])) == (22930048, 5218048, 41760, 1599692)
    assert int(d[(S4B, "ll_B1_T128", "P32")]["dram_rd_per_step"]) == 492788288
    assert int(d[(S4B, "ll_B1_T128", "P8")]["dram_rd_per_step"]) == 152241088
    assert int(d["EP.4 SW*", "ll_B1_T128", "P8"]["dram_wr_per_step"]) == 202147008
    assert d[(S4B, "ll_B1_T128", "P32")]["l2_acc_per_step"] == ""          # not measured, not estimated

def test_no_energy_units():
    pat = re.compile(r"\d\s*(pJ|nJ|uJ|mJ|J|mW|W|kW)\b")
    for f in list(R7.glob("*.csv")) + [R7 / "README.md"]:
        assert not pat.search(f.read_text()), f

def test_ep6_trigger_pins():
    j = json.loads((R6 / "trigger.json").read_text())
    assert abs(j["G6_static"] - 0.46771) < 1e-4          # = summary.json ep6_trigger_F1_mix.G6
    assert abs(j["G6_dyn"] - 0.44781) < 1e-4 and abs(j["G6_dyn_weights_only"] - 0.13301) < 1e-4
    assert j["class_share_dyn_min"] >= 0.20 and j["n_workloads"] == 18
    ep4b = json.loads((PS / "results/EP.4b/summary.json").read_text())["ep6_trigger_F1_mix"]["G6"]
    assert math.isclose(j["G6_static"], ep4b, rel_tol=1e-9)

def test_rope_counters_clamped_and_flagged():
    """no per-kind counter is negative; the clamped (table - setup < 0) rows are exactly the flagged rope rows"""
    bk = rows(R7 / "proxy_by_kind.csv")
    assert any(r["clamped"] == "1" for r in bk)
    for r in bk:
        for k, v in r.items():
            if k.endswith("_per_step") and v: assert float(v) >= 0, (r["workload"], r["kind"], k)
        if r["clamped"] == "1": assert r["kind"] == "rope"
    tok = {(r["source"], r["workload"], r["track"]): r["rope_clamped"] for r in rows(R7 / "proxy_token.csv")}
    for r in bk:
        if r["clamped"] == "1": assert tok[(r["source"], r["workload"], r["track"])] == "1"
    import importlib; sys.path.insert(0, str(R7)); L = importlib.import_module("proxy_lib")
    a = {k: 5 for k in L.KEYS}; b = {k: 9 for k in L.KEYS}
    o = L._sub(a, b); assert o["clamped"] and all(o[k] == 0 for k in L.KEYS)
    o = L._sub(b, a); assert not o["clamped"] and all(o[k] == 4 for k in L.KEYS)
