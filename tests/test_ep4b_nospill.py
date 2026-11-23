"""Tests: spill-free Q8_0 batched GEMV (register fence), SW*', ', capacity/residency split, G-EPB evaluation 2
(docs/ addendum §2.6; methodology §18.11).

Non-vortex (always run; no SimX):
  test_batch_files_up_to_date        experiments/<run>_*.yaml == results/<run>/make_batch.py
  test_tie_order_new_kernels_last    patched sw_star: on a tie the old kernels win over rows_nsp / rows_nsp_ilp2; a clear minimum wins; the patch restores ep4lib / sw_select
  test_static_spill_parser           static_spill.analyse on a hand-written disassembly: backward-branch loop ranges, stack stores / reloads, row-loop vs group-loop split
  test_static_spill_s1               S1 (results/<run>/static_spill.csv, linked kernel.elf of E1): every fixed-GS path of rows_nsp (B 1-4, GS 32 / 64) and rows_nsp_ilp2 (B 1, 2) has 0 stack stores in a group loop and <= 8 reloads
  test_rows_bitexact                 every stored new-variant row: status ok, passed, PSBITEXACT mismatches=0 in its log
  test_stored_results                analyze.py --check byte-identical, results/<run>/analyze.py --check byte-identical; 27 classes complete; ' recomputed by hand from headroom_star.csv; attr == cold cycles; persist pairs <= 5 %
  test_ident_and_hcap                the 3 identity rows reproduce cycles / instructions / digest exactly; the earlier walk identity on the hcap classes within 1 cycle
  test_pinned_results                the numbers of the README (D2, G_res, G-EPB 2a, outcome, contamination record)
  test_consistency_check_flags_wrong_machine_rows   the shared-build-dir contamination check on synthetic rows
  test_methodology_18_11             documentation present
Vortex-marked (PS_RUN_VORTEX=1; outer qflock hold of /tmp/ml-hardware-bench.lock by the caller, private results dir build/tmp/ep4b_test; lowload exit 75 -> skip):
  test_nsp_odd_shapes[...]           rows_nsp / rows_nsp_nu (B 1..4) and rows_nsp_ilp2 (B 1, 2) on odd shapes (M in {50, 70, 100}, B = 3, GS in {16, 128}: the generic-GS path) at E1 and E4:
                                     PASSED, PSCHECK errors=0, PSBITEXACT mismatches=0, digest == the NumPy port's
  test_ilp2_b4_fails_on_purpose      rows_nsp_ilp2 at B = 4 is not dispatched: the run FAILS (FAILED! in the log, status failed)
  test_pscheck_grid_rows[...]        two grid classes (stories15M qkvo B = 4, llama8b_s8 w2 B = 2) at E1: PSCHECK vs_f64 errors = 0 and bit-exact
"""
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
EP4 = PS / "results" / "EP.4"
EP4B = PS / "results" / "EP.4b"
sys.path.insert(0, str(EP4))
sys.path.insert(1, str(PS / "scripts"))
sys.path.insert(2, str(Path(__file__).resolve().parent))
STEM_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) tol=(\S+) errors=(\d+)", re.M)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def AB():
    return _load("ep4b_analyze", EP4B / "analyze.py")


@pytest.fixture(scope="module")
def MBB():
    return _load("ep4b_make_batch", EP4B / "make_batch.py")


@pytest.fixture(scope="module")
def SS():
    return _load("ep4b_static_spill", EP4B / "static_spill.py")


def _has_results():
    return (EP4B / "summary.json").is_file()


def _rows():
    p = EP4B / "rows.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def test_batch_files_up_to_date(MBB):
    stale = []
    for g in MBB.GROUPS:
        p = PS / "experiments" / f"EP.4b_{g}.yaml"
        if p.is_file() and p.read_text() != MBB.dump(g):
            stale.append(p.name)
    assert not stale


def test_tie_order_new_kernels_last(AB):
    import ep4lib as L
    import sw_select as SW
    orig = (L.load_all, L.cand_spec, L.VARIANTS, SW.EXPECT, SW.sw_star)

    def cand(v, cyc, B=2):
        return dict(track="P8", model="stories15M", matrix="qkvo", M=288, K=288, GS=32, B=B, variant=v, app="a", config="c", args=f"-m 288 -L 1 -v {v}", cycles=cyc,
                    instrs=1, dram_rd=1, src="t")
    with AB.patched(AB.NEW_F):
        assert SW.sw_star is AB._sw_star_b
        # tie within 0.5 %: the old kernel wins although the new one is 0.4 % faster
        c = AB._sw_star_b([cand("rows", 1004), cand("rows_inl", 1003), cand("rows_nsp", 1000)])
        assert [x["variant"] for x in c] == ["rows"] and set(c[0]["tied"]) == {"rows", "rows_inl", "rows_nsp"}
        # a non-default old kernel beats the new one on a tie, too
        c = AB._sw_star_b([cand("rows", 2000), cand("rows_inl", 1004), cand("rows_nsp", 1000)])
        assert c[0]["variant"] == "rows_inl"
        # a clear minimum (> 0.5 %) wins
        c = AB._sw_star_b([cand("rows", 2000), cand("rows_inl", 1500), cand("rows_nsp", 1000), cand("rows_nsp_ilp2", 1010)])
        assert c[0]["variant"] == "rows_nsp"
        # between the two new kernels on a tie: rows_nsp first
        c = AB._sw_star_b([cand("rows", 2000), cand("rows_nsp", 1003), cand("rows_nsp_ilp2", 1000)])
        assert c[0]["variant"] == "rows_nsp"
    assert (L.load_all, L.cand_spec, L.VARIANTS, SW.EXPECT, SW.sw_star) == orig       # restored


DISASM = """80000000 <f>:
80000000:  addi sp, sp, -0x40
80000004:  sw ra, 0x3c(sp)
80000008:  sw a0, 0x8(sp)
8000000c:  lw a1, 0x0(a0)
80000010:  lw a2, 0x8(sp)
80000014:  add a1, a1, a2
80000018:  sw a3, 0x8(sp)
8000001c:  bne a1, a4, 0x80000010 <f+0x10>
80000020:  lw a5, 0xc(sp)
80000024:  fsw fa0, 0x0(a6)
80000028:  bnez a5, 0x80000008 <f+0x8>
8000002c:  ret
"""


def test_static_spill_parser(SS, tmp_path):
    ins = []
    for line in DISASM.splitlines()[1:]:
        a, rest = line.split(":", 1)
        parts = rest.split(None, 1)
        ins.append((int(a, 16), parts[0], parts[1] if len(parts) > 1 else ""))
    r = SS.analyse(ins)
    # loops: inner [0x10, 0x1c] (group-level: 1 stack store at 0x18, 1 reload at 0x10) and outer [0x08, 0x28] (contains the output fsw: row-level)
    assert r["n_loops"] == 2 and r["frame_bytes"] == 0x40
    assert r["stores_in_loops"] == 1 and r["reloads_max_inner_loop"] == 1
    assert r["row_stores"] == 2 and r["row_reloads_max"] == 2


@pytest.mark.skipif(not (EP4B / "static_spill.csv").is_file(), reason="static_spill.csv not written yet")
def test_static_spill_s1():
    rows = [r for r in csv.DictReader((EP4B / "static_spill.csv").open()) if r["variant"] in ("rows_nsp", "rows_nsp_ilp2") and r["GS"] in ("32", "64")]
    cfgs = {r["config"] for r in rows}
    assert "PS-BASE-S0" in cfgs
    for cfg in cfgs:
        mine = {r["function"]: r for r in rows if r["config"] == cfg}
        want = {f"nsp_rows_B{b}_GS{g}" for b in (1, 2, 3, 4) for g in (32, 64)} | {f"nsp_rows2_B{b}_GS{g}" for b in (1, 2) for g in (32, 64)}
        assert want <= set(mine), cfg
        for fn in want:
            r = mine[fn]
            assert int(r["stores_in_loops"]) == 0 and int(r["reloads_max_inner_loop"]) <= 8 and r["s1_pass"] == "1", (cfg, fn)


NEW = ("rows_nsp", "rows_nsp_ilp2", "rows_nsp_nu")


@pytest.mark.skipif(not _rows(), reason="rows not written yet")
def test_rows_bitexact():
    n = 0
    for r in _rows():
        if r["status"] != "ok" or r["app"] != "ps_gemv_q80" or not any(f"-v {v} " in r["args"] + " " for v in NEW):
            continue
        assert r.get("passed")
        log = (PS / r["log"]).read_text(errors="replace")
        assert re.search(r"^PSBITEXACT: launch\d:runq_exact n=\d+ mismatches=0 variant=rows_nsp", log, re.M), r["args"]
        n += 1
    assert n >= 45


@pytest.mark.skipif(not _has_results(), reason="results not written yet")
def test_stored_results(AB):
    s = json.loads((EP4B / "summary.json").read_text())
    assert s["tests"]["all_classes_complete"] and not s["tests"]["missing_expected_cells"] and s["tests"]["P32_gate_equals_ep4"]
    assert s["tests"]["attr_equals_cold_all"] is True
    # analyze.py --check, byte-identical
    assert AB.main(["analyze.py", "--check"]) == 0
    assert subprocess.run([sys.executable, str(EP4 / "analyze.py"), "--check"], capture_output=True, text=True).returncode == 0
    # ' recomputed by hand from headroom_star.csv: S*_w = 1 / (1 - sum_k f_k h_k) - 1, aggregate geomean(1 + S) - 1 over the 18 P8 workloads
    by = {}
    for r in csv.DictReader((EP4B / "headroom_star.csv").open()):
        if r["track"] == "P8":
            by[r["workload"]] = by.get(r["workload"], 0.0) + float(r["f_k"]) * float(r["h_mem_star"])
    assert len(by) == 18
    g = math.exp(sum(math.log(1 / (1 - u)) for u in by.values()) / 18) - 1
    assert g == pytest.approx(s["gate_G_EP1_prime"]["P8"]["G"], abs=2e-6)
    # persist pairs of every SW*' class: launch 2 L2 read misses <= 5 % of launch 1 (headroom_star.csv `miss_ok`)
    assert all(r["miss_ok"] == "1" for r in csv.DictReader((EP4B / "headroom_star.csv").open()) if r["track"] == "P8")
    # cap + res + delta/C_cold = h* per workload on average: U_cap + U_res + (sum f_k delta / C_cold) = U_h (checked on the aggregate relation S_h >= S_res, S_h >= S_cap when cap >= 0)
    for r in csv.DictReader((EP4B / "capres.csv").open()):
        assert float(r["U_h"]) >= float(r["U_res"]) - 1e-12 or float(r["U_cap"]) < 0


@pytest.mark.skipif(not _has_results(), reason="results not written yet")
def test_ident_and_hcap():
    s = json.loads((EP4B / "summary.json").read_text())
    ident = list(csv.DictReader((EP4B / "ident.csv").open()))
    # appending kernel_nsp.cpp moved no instruction and no result bit; cycles are identical on 2 of 3 rows and +90 cycles (+0.027 %, inside the 0.1 % stock-identity rule) on the third
    assert len(ident) == 3 and all(r["instrs_equal"] == r["digest_equal"] == "1" for r in ident)
    assert all(abs(int(r["cycles_ep4b"]) - int(r["cycles_ep4"])) <= 0.001 * int(r["cycles_ep4"]) for r in ident) and sum(r["cycles_equal"] == "1" for r in ident) >= 2
    assert all(r["equal"] == 1 for r in s["ident_tile"]) and len(s["ident_tile"]) == 3     # the second appended TU (kernel_tile.cpp) moved no F1 / F2 cycle
    hc = list(csv.DictReader((EP4B / "hcap.csv").open()))
    assert len(hc) == 3 and all(abs(int(r["diff"])) <= 1 for r in hc), hc


@pytest.mark.skipif(not _has_results(), reason="results not written yet")
def test_pinned_results():
    """numbers the README reports (any change of a row or of the rules shows up here)"""
    s = json.loads((EP4B / "summary.json").read_text())
    g = s["gate_G_EP1_prime"]
    assert g["P8"]["G"] == pytest.approx(0.0617, abs=5e-5) and g["P8"]["verdict"] == "FAIL" and g["P32"]["G"] == pytest.approx(0.1545, abs=5e-5)
    assert s["G_EP1_evaluation1_P8"] == pytest.approx(0.1621, abs=5e-5)
    assert s["D2_S_cap_P8"] == pytest.approx(0.0065, abs=5e-5) and s["D2_pass"] is True
    assert s["G_res_for_EP5"]["P8"] == pytest.approx(0.0540, abs=5e-5)
    assert s["G_EPB_eval2"]["eval2a_ep4_sw_star"]["P8"]["all"]["value"] == pytest.approx(0.1139, abs=5e-5)
    assert s["outcome"] == "NOT FIXED" and s["n_S1_pass"] == 26 and s["n_D1_pass"] == 9 and s["n_D1_wave_pass_posthoc"] == 27
    assert s["n_classes_sw_star_new_variant"] == 26
    assert not s["contamination"]["open_inconsistent"] and len(s["contamination"]["superseded_wrong_machine_rows"]) == 2


def test_consistency_check_flags_wrong_machine_rows(AB, monkeypatch):
    """the shared-build-dir contamination check: rows of one (app, args) must agree on instructions per launch; the minority row is flagged"""
    def row(a, perf, instrs, cfg="PS-BASE-S0"):
        return dict(status="ok", app="ps_gemv_q80", args=a, config=cfg, perf=perf, instrs=instrs, passed=True, tags={}, log="x")
    rows = [row("-m 8 -n 64 -L 1", 7, 1000), row("-m 8 -n 64 -L 1 -A", 8, 250), row("-m 8 -n 64 -L 2 -P", 5, 2000, "EP-CTRL-S0-L2big-P"), row("-m 8 -n 64 -L 1 -P", 5, 1000, "EP-CTRL-S0-L2big-P")]
    monkeypatch.setattr(AB, "_rows_b", lambda: rows)
    bad, last = AB.consistency()
    assert [b["perf"] for b in bad] == [8] and bad[0]["majority"] == 1000.0 and len(last) == 4
    monkeypatch.setattr(AB, "_rows_b", lambda: rows[:2])           # a tie (2 rows) flags both: no strict majority
    assert len(AB.consistency()[0]) == 2


def test_methodology_18_11():
    txt = (PS / "docs" / "methodology.md").read_text()
    assert "### 18.11" in txt
    sec = txt.split("### 18.11", 1)[1]
    assert "fence" in sec and "S_stack" in sec and "G-EPB" in sec and "libsimx" in sec


# --------------------------------------------------------------------------------------------- vortex (SimX)
@pytest.fixture(scope="module")
def vx():
    import vxrun
    priv = PS / "build" / "tmp" / "ep4b_test"
    shutil.rmtree(priv, ignore_errors=True)
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), str(PS / "build" / "vortex-ps")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun("EP.4b-test", results=priv)


def _run(vx, args, config="PS-BASE-S0", perf=1):
    row = vx.run(dict({"tree": "ps", "perf": perf, "app": "ps_gemv_q80", "args": args, "config": config}), timeout_s=3600, rss_gb=0.5)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


ODD = [(50, 256, 16, 3), (70, 512, 128, 3), (100, 256, 16, 1), (50, 256, 128, 2), (70, 256, 64, 4)]
ODD_CASES = [(v, c, cfg) for cfg in ("PS-BASE-S0", "PS-BASE-S1-4c") for c in ODD for v in ("rows_nsp", "rows_nsp_nu", "rows_nsp_ilp2") if not (v == "rows_nsp_ilp2" and c[3] > 2)]


@pytest.mark.vortex
@pytest.mark.parametrize("v,case,cfg", ODD_CASES, ids=[f"{c[2]}_{c[0]}_M{c[1][0]}_N{c[1][1]}_GS{c[1][2]}_B{c[1][3]}" for c in ODD_CASES])
def test_nsp_odd_shapes(vx, v, case, cfg):
    from test_ep_q80 import py_digests
    M, N, GS, B = case
    row, log = _run(vx, f"-m {M} -n {N} -G {GS} -B {B} -v {v}", cfg)
    assert row["status"] == "ok", row["reason"]
    c = {m.group(1): (float(m.group(4)), int(m.group(7))) for m in STEM_RE.finditer(log)}
    assert c["launch0:vs_f64"][1] == 0
    assert re.search(rf"^PSBITEXACT: launch0:runq_exact n=\d+ mismatches=0 variant={v}$", log, re.M)
    d = py_digests(M, N, GS, B, 1)
    assert re.search(rf"^PSDIGEST: launch0:vs_f64 n={B * M} fnv1a64={d['y']}$", log, re.M)


@pytest.mark.vortex
@pytest.mark.parametrize("B", [3, 4])
def test_ilp2_b4_fails_on_purpose(vx, B):
    row, log = _run(vx, f"-m 64 -n 256 -G 32 -B {B} -v rows_nsp_ilp2")
    assert row["status"] == "failed" and "FAILED!" in log and "PASSED!" not in log


@pytest.mark.vortex
@pytest.mark.parametrize("m,n,G,B", [(288, 288, 32, 4), (512, 1792, 64, 2)])
def test_pscheck_grid_rows(vx, m, n, G, B):
    for v in ("rows_nsp",) + (("rows_nsp_ilp2",) if B <= 2 else ()):
        row, log = _run(vx, f"-m {m} -n {n} -G {G} -B {B} -v {v}")
        assert row["status"] == "ok", row["reason"]
        assert all(int(x.group(7)) == 0 for x in STEM_RE.finditer(log)) and STEM_RE.search(log)
        assert re.search(rf"^PSBITEXACT: launch0:runq_exact n=\d+ mismatches=0 variant={v}$", log, re.M)
