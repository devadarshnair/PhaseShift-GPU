"""Tests: composition and ablation harness (experiments/knobs.yaml, scripts/knobs.py, ps patch 0013-compose,
scripts/noop_regression.sh --all, results/<run>/compat.csv). Semantics: docs/methodology.md §14.E; evidence results/<run>/.

All cheap (no SimX run here; the SimX evidence is produced by results/<run>/run_*.sh and checked from its stored rows):
  test_patch_0013_files        0013-compose touches only sim/simx/processor.cpp, nothing in third_party/; the relaxed
                               check keeps every non-D insert policy exclusive with an L2 mask
  test_knobs_validate          knobs.validate(): every K value legal, A2 x D cross product legal, every ablation cell
                               legal, every probe illegal with its message (the Python mirror of the SimX checks)
  test_reduced_set_shape       the 9 vectors; static = no args, no schedule; a single-mechanism vector differs from
                               static only in its own mechanism's knobs; all-ideal == full-practical on this matrix
  test_probe_messages_in_simx  every probe's abort text is a SimX message of the ps tree
  test_schedules_parse         every generated results/<run>/sched/*.sched passes scripts/mkschedule.py
  test_host_W_option           the nine apps that got -W parse it and write [dcr_phase] only when it is given
  test_kernel_binaries_same    device kernel.vxbin hashes host edits equal the ones before (per tree)
  test_noop_all                results/<run>/noop_regression.csv: REG + 13 P2 kernels + 13 all-modes-0 runs, all identical
  test_compat_complete         results/<run>/compat.csv: (9 + 3) x 13 cells, ablation cells pass, probes rejected-by-design
"""
import csv
import re
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
RES = PS / "results" / "P5.5"
TREE = PS / "vortex-trees" / "ps"
sys.path.insert(0, str(PS / "scripts"))
import knobs  # noqa: E402
import mkschedule as mk  # noqa: E402

W_APPS = {"gemv_f32": "gemv_host.h", "linear": "main.cpp", "prefill_attn": "main.cpp", "softmax_fixed": "main.cpp",
          "rmsnorm": "main.cpp", "residual": "main.cpp", "silu_gate": "main.cpp", "rope": "main.cpp", "llama": "main.cpp"}
VECTORS = ["static", "A", "B", "C", "D", "A+B", "all-ideal", "full-practical", "oracle"]


@pytest.fixture(scope="module")
def doc():
    return knobs.load()


def test_patch_0013_files():
    p = PS / "patches/vortex/ps/0013-compose.patch"
    txt = p.read_text()
    assert re.findall(r"^diff --git a/(\S+) b/", txt, re.M) == ["sim/simx/processor.cpp"]
    assert "(cfg.insert_policy & ~PhaseConfig::INS_D_MASK) != 0" in txt
    src = (TREE / "sim/simx/processor.cpp").read_text()
    assert "cfg.l2_waymask != 0 && (cfg.insert_policy & ~PhaseConfig::INS_D_MASK) != 0" in src
    assert "cfg.l2_waymask != 0 && cfg.insert_policy != 0" not in src
    # UCP-lite keeps its own exclusivities (patch a-baselines), untouched
    assert 'if (cfg.insert_policy != 0) fail("flags(l2_ucp)"' in src


def test_knobs_validate(doc):
    assert knobs.validate(doc) == []
    # the mirror agrees with the SimX rule on the cases that matter
    assert knobs.legality({"l2": 0xF, "insert": 1}) == ""
    assert knobs.legality({"l2": 0xF, "insert": 2}) == ""
    for ins in (0x10, 0x20, 0x40, 0x100, 0x11):
        assert "other than D modes" in knobs.legality({"l2": 0xF, "insert": ins}), hex(ins)
    assert knobs.legality({"l1": 0x3, "insert": 0x20}) == ""   # an L1-only mask never met the L2 insert rule
    assert "l2_waymask" in knobs.legality({"l2": 0xF, "l2_ucp": 8})


def test_reduced_set_shape(doc):
    assert list(doc["ablation"]) == VECTORS
    for k in knobs.kernels(doc):
        st = knobs.resolve(doc, "static", k)
        assert st.args == k["args"] and st.segments == []
        for v, m in (("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")):
            c = knobs.resolve(doc, v, k)
            for other in set("ABCD") - {m}:
                assert c.applied[other] == "off", (v, k["id"], c.applied)
        ai, fp = knobs.resolve(doc, "all-ideal", k), knobs.resolve(doc, "full-practical", k)
        assert (ai.args, ai.segments) == (fp.args, fp.segments)
        o = knobs.resolve(doc, "oracle", k, half=1000)
        assert len(o.segments) == 2 and o.segments[0][1] != o.segments[1][1]
        assert (o.args, o.segments) != (ai.args, ai.segments)


def test_probe_messages_in_simx(doc):
    src = "".join((TREE / f).read_text() for f in ("sim/simx/processor.cpp", "sim/common/phase_config.h"))
    for name, pr in doc["probes"].items():
        assert pr["abort"] in src, name


def test_schedules_parse():
    files = sorted((RES / "sched").glob("*.sched"))
    if not files:
        pytest.skip("results/P5.5/sched not generated yet")
    for f in files:
        mk.parse(f.read_text())


@pytest.mark.parametrize("app,src", sorted(W_APPS.items()))
def test_host_W_option(app, src):
    txt = (PS / "kernels" / app / src).read_text()
    assert re.search(r'getopt\(argc, argv, "[^"]*W:[^"]*"\)', txt) or app in ("rmsnorm", "residual", "silu_gate", "rope")
    if app in ("rmsnorm", "residual", "silu_gate", "rope"):
        assert re.search(r'getopt\(argc, argv, "[^"]*W:[^"]*h"\)', txt)
        assert "ps_elt::waymask_opt(optarg)" in txt
        h = (PS / "kernels/elt_common/elt_host.h").read_text()
        assert "if (waymask_opt_state().have) {" in h
    else:
        assert "parse_waymask_arg(optarg" in txt
        assert re.search(r"if \((ps_)?have_wm\) \{|else if \(have_wm\) \{", txt)


def test_kernel_binaries_same():
    before, after = RES / "kernel_vxbin_before.txt", RES / "kernel_vxbin_after.txt"
    if not after.exists():
        pytest.skip("results/P5.5/kernel_vxbin_after.txt not produced yet")
    b = {tuple(l.split()[:2]): l.split()[2] for l in before.read_text().splitlines() if l.strip()}
    a = {tuple(l.split()[:2]): l.split()[2] for l in after.read_text().splitlines() if l.strip()}
    common = set(a) & set(b)
    assert len(common) >= 20
    diff = {k: (b[k], a[k]) for k in common if a[k] != b[k]}
    # ps_gemv_f32 / ps_twolaunch share one kernel built with different -L defaults; their files flip with the last
    # build of either app, so they are compared across the pair (README).
    ok_pair = {("ps", "ps_gemv_f32"), ("ps", "ps_twolaunch"), ("base", "ps_gemv_f32"), ("base", "ps_twolaunch")}
    assert set(diff) <= ok_pair, diff
    for t, k in diff:
        assert a[(t, k)] in {b.get((t, "ps_gemv_f32")), b.get((t, "ps_twolaunch"))}, (t, k)


def test_noop_all():
    p = RES / "noop_regression.csv"
    if not p.exists():
        pytest.skip("results/P5.5/noop_regression.csv not produced yet")
    rows = list(csv.DictReader(p.open()))
    by = {}
    for r in rows:
        by.setdefault(r["reg"], []).append(r)
    assert len(by.get("p2", [])) == 13 and len(by.get("zero", [])) == 13, {k: len(v) for k, v in by.items()}
    assert len(rows) == 8 + 13 + 13
    assert all(r["identical"] == "True" for r in rows), [r for r in rows if r["identical"] != "True"]


def test_compat_complete(doc):
    p = RES / "compat.csv"
    if not p.exists():
        pytest.skip("results/P5.5/compat.csv not produced yet")
    rows = list(csv.DictReader(p.open()))
    ks = [k["id"] for k in knobs.kernels(doc)]
    want = {(v, k) for v in list(doc["ablation"]) + list(doc["probes"]) for k in ks}
    got = {(r["vector"], r["kernel"]) for r in rows}
    assert got == want and len(rows) == len(want)
    for r in rows:
        assert r["status"] == ("rejected-by-design" if r["kind"] == "probe" else "pass"), r
        if r["kind"] == "probe":
            assert r["reason"].startswith("Error:"), r
        else:
            assert r["cycles"] and r["instrs"]
            assert r["wm_violations"] in ("", "0"), r
