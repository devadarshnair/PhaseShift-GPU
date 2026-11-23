"""Tests (sampling errors computed and reported; the bound written in the results README as a single
number). Unit tests of the interpolation and early-stop primitives of results/<run>/analyze.py (methodology §6.3/§6.4)
on synthetic data, plus consistency checks of the stored results. Model-only, no SimX."""
import csv
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

PS = Path(__file__).resolve().parents[1]
RES = PS / "results" / "P3.3"


def _load():
    spec = importlib.util.spec_from_file_location("p33_analyze", RES / "analyze.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _load()


def test_interp1_exact_for_affine_and_no_extrapolation():
    xs, ys = [64, 256, 1024], [100 + 3 * x for x in (64, 256, 1024)]
    assert A.interp1(xs, ys, 128) == pytest.approx(100 + 3 * 128)
    assert A.interp1(xs, ys, 32) is None and A.interp1(xs, ys, 2048) is None


def test_loglog_exact_for_power_law():
    xs = [16, 32, 64]
    ys = [7 * x ** 2 for x in xs]
    assert A.interp1(xs, ys, 24, "log") == pytest.approx(7 * 24 ** 2)


def test_quad_needs_four_points_and_is_exact_for_quadratics():
    f = lambda s: 5000 + 30 * s + 2 * s * s  # noqa: E731
    xs = [16, 32, 48, 64]
    assert A.along(xs, [f(x) for x in xs], 24, "quad", "lin") == pytest.approx(f(24))
    # with 3 points the rule falls back to piecewise-linear (§6 needs >= 4 points)
    xs3 = [16, 32, 64]
    lin = A.interp1(xs3, [f(x) for x in xs3], 48)
    assert A.along(xs3, [f(x) for x in xs3], 48, "quad", "lin") == pytest.approx(lin)


def test_bilinear_exact_for_bilinear_form():
    f = lambda B, T: 1000 + 50 * B + 2 * T + 3 * B * T  # noqa: E731
    train = [({"B": B, "T": T}, f(B, T)) for B in (1, 2, 4) for T in (64, 256, 1024)]
    assert A.predict("decode_attn", train, {"B": 3, "T": 128}, "lin", "both") == pytest.approx(f(3, 128))
    assert A.predict("decode_attn", train, {"B": 3, "T": 256}, "lin", "secondary") == pytest.approx(f(3, 256))


def test_early_stop_rule_on_synthetic_prefixes():
    # constant marginal rate -> stops at j = 2 (third prefix) and projects exactly
    pre = [(64, 1000 + 10 * 12, 12), (128, 1000 + 10 * 36, 36), (256, 1000 + 10 * 120, 120)]
    rows, samples = [], []
    A.early_stop("syn", "cfg", pre, 1000 + 10 * 432, 432, rows, samples, "prefill_attn")
    assert samples[0]["estimate"] == pytest.approx(1000 + 10 * 432)
    assert "stopped" in samples[0]["note"] and rows[-1]["used"] == 1
    # rate changes by 20 % -> no stop, forced projection from the largest prefix
    pre2 = [(1, 100, 10), (2, 200, 20), (3, 320, 30)]
    rows, samples = [], []
    A.early_stop("syn2", "cfg", pre2, 1000, 80, rows, samples, "x")
    assert "NO STOP" in samples[0]["note"]
    assert samples[0]["estimate"] == pytest.approx(320 + 12 * 50)


def test_corners_are_never_held_out():
    pts = [(f"B{B}_T{T}", {"B": B, "T": T}, 1, "new") for B in (1, 2, 3, 4) for T in (64, 128, 256, 512, 1024)]
    assert A.corners("decode_attn", pts) == {(64, 1), (64, 4), (1024, 1), (1024, 4)}


# ------------------------------------------------------------------ stored results
needs_results = pytest.mark.skipif(not (RES / "bound.json").exists(), reason="results/P3.3 not generated")


@needs_results
def test_bound_is_max_of_component_p95_and_in_readme():
    b = json.loads((RES / "bound.json").read_text())
    assert b["bound"] == pytest.approx(max(v["p95"] for v in b["components"].values()))
    assert b["pooled"]["p95"] <= b["bound"] + 1e-12
    txt = (RES / "README.md").read_text()
    m = re.search(r"\*\*P3\.3 bound = ([0-9.]+) %\*\*", txt)
    assert m, "README must state the bound as a single number"
    assert float(m.group(1)) == pytest.approx(100 * b["bound"], abs=0.005)


@needs_results
def test_sampling_error_csv_recomputes():
    rows = list(csv.DictReader(open(RES / "sampling_error.csv")))
    assert {r["subtest"] for r in rows} >= {"a1", "a2", "b1", "b2", "c"}
    b = json.loads((RES / "bound.json").read_text())
    for st, comp in b["components"].items():
        e = [abs(float(r["rel_err"])) for r in rows if r["subtest"] == st and r["in_bound"] == "1"]
        assert len(e) == comp["n"]
        assert float(np.percentile(e, 95)) == pytest.approx(comp["p95"], rel=1e-4)
    for r in rows:
        m, e = float(r["measured"]), float(r["estimate"])
        assert float(r["rel_err"]) == pytest.approx((e - m) / m, rel=1e-4, abs=1e-9)


@needs_results
def test_holdout_is_seeded_and_twenty_percent():
    h = list(csv.DictReader(open(RES / "holdout.csv")))
    assert {r["seed"] for r in h} == {"3303"}
    by = {}
    for r in h:
        by.setdefault(r["family"], []).append(r)
    for fam, rs in by.items():
        assert len(rs) == int(round(0.2 * int(rs[0]["n_points"])))


@needs_results
def test_repro_rows_equal_stored():
    rep = list(csv.DictReader(open(RES / "repro.csv")))
    assert len(rep) == 6 and all(r["equal"] == "1" for r in rep)


def test_methodology_section_6_filled():
    txt = (PS / "docs" / "methodology.md").read_text()
    for h in ("### 6.1 ", "### 6.2 (a) Layer multiplication", "### 6.3 (b)", "### 6.4 (c)", "### 6.5"):
        assert h in txt, h


@needs_results
def test_second_pass_variants_consistent():
    """The (c) convention variants and the restated post-hoc scenario follow the max-of-p95
    rule, and the primary bound is unchanged by them."""
    b = json.loads((RES / "bound.json").read_text())
    v = b["variants_attempt2"]
    others = max(c["p95"] for k, c in b["components"].items() if k != "c")
    assert v["bound_other_components_max_p95"] == pytest.approx(others)
    for name, c in v["c_variants"].items():
        assert c["bound"] == pytest.approx(max(others, c["p95"])), name
    prim = [c for n, c in v["c_variants"].items() if n.startswith("8 kernels") and "forced" in n][0]
    assert prim["p95"] == pytest.approx(b["components"]["c"]["p95"]) and prim["bound"] == pytest.approx(b["bound"])
    s = v["posthoc_scenario_v2"]
    assert s["bound"] == pytest.approx(max(c["p95"] for c in s["components"].values()))
    loo = list(csv.DictReader(open(RES / "loo.csv")))
    mc = [abs(float(r["rel_err"])) for r in loo if r["family"] == "mixed_attn" and r["axis"] == "c"
          and r["interpolant"] == "primary"]
    assert len(mc) == 9 and float(np.percentile(mc, 95)) == pytest.approx(v["loo"]["mixed_attn:c:primary"]["p95"])
