"""Tests (the fit recovers known parameters on synthetic data), plus the pieces the
calibration relies on: the fixed, recorded, cell-grouped train/test split; `MemHier.cost_launch` equal to the
iteration path; the new `Hier.overlap` (ω = 1 is the model); the tension metric reproducing the earlier numbers on
measured cycles; and results/<run> reproducing from the stored fit. Model-only, no SimX."""
import csv
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from model import calibrate as C
from model import opgen
from model.memhier import Hier, Knobs, MemHier, hier_for, split_launches

PS = Path(__file__).resolve().parents[1]
RES = PS / "results" / "P3.4"


@pytest.fixture(scope="module")
def cases():
    cs, split, exc = C.build_cases()
    return cs, split, exc


# ------------------------------------------------------------------------------------------------- split


def test_rows_used_and_excluded(cases):
    cs, _, exc = cases
    rows = C.load_table()
    assert len(cs) == sum(1 for r in rows if r["how"] == "run") == 306
    assert sum(exc.values()) == len(rows) - len(cs)
    assert {c.family for c in cs} == set(C.FAMILIES)
    assert all(c.measured > 0 for c in cs)


def test_split_fixed_by_seed_grouped_and_stratified(cases):
    cs, split, _ = cases
    cs2, split2, _ = C.build_cases(seed=C.SPLIT_SEED)
    assert split == split2                                   # same seed -> same split
    _, split3, _ = C.build_cases(seed=C.SPLIT_SEED + 1)
    assert split3 != split                                   # the seed actually drives it
    by_cell = {}
    for c in cs:
        by_cell.setdefault((c.family, c.cell), set()).add(c.split)
    assert all(len(v) == 1 for v in by_cell.values())        # every setting of a cell on one side
    for fam in C.FAMILIES:
        sides = [s for (f, _), s in split.items() if f == fam]
        assert "train" in sides and "test" in sides
        assert abs(sides.count("train") / len(sides) - C.TRAIN_FRAC) <= 0.5 / len(sides) + 1e-9
    assert C.MIN_LAUNCH == min(c.measured for c in cs)        # launch_overhead bound = shortest launch
    if (RES / "split.csv").exists():                         # the recorded split is this one
        with open(RES / "split.csv") as f:
            rec = {(r["family"], r["cell"]): r["split"] for r in csv.DictReader(f)}
            assert rec == split
        with open(RES / "split.csv") as f:
            assert {int(r["seed"]) for r in csv.DictReader(f)} == {C.SPLIT_SEED}


# ------------------------------------------------------------------------------------------------- memhier pieces


def test_cost_launch_matches_iteration_path():
    """Σ cost_launch over an iteration's launches = cost_seqs with persistence off (cold launches are independent)."""
    m = C.SHAPE
    for seqs in (opgen.decode(4, 256), opgen.prefill(2, 64), opgen.mixed([(32, 32)], [(8, 512)])):
        mh = MemHier(m, "psbase-S1-4c", memo=0, q_w=4, q_a=4, q=4, persist_across_launch=False)
        for k in (Knobs(), Knobs(lmem_use=32768, dxa_depth=2)):
            it = mh.cost_seqs(seqs, k)
            ops = opgen.gen_ops(m, seqs, q_w=4, q_a=4, q=4, layers=[0])
            tot = sum(mh.cost_launch(L, k).cycles for L in split_launches(ops))
            assert math.isclose(tot, it.cycles, rel_tol=1e-12)


def test_cost_launch_refuses_persistence():
    mh = MemHier(C.SHAPE, "psbase-S1-4c", memo=0, persist_across_launch=True)
    with pytest.raises(ValueError):
        mh.cost_launch(opgen.gen_ops(C.SHAPE, opgen.decode(1, 64), layers=[0])[:1])


def test_overlap_limits_and_default():
    """ω = 1 (default) is the overlap rule; ω = 0 serializes compute and memory; cycles fall monotonically in ω."""
    assert Hier.__dataclass_fields__["overlap"].default == 1.0
    assert hier_for("vortex-S1-8c").overlap == 1.0 and hier_for("h100-like").overlap == 1.0
    with pytest.raises(ValueError):
        dataclasses.replace(hier_for("psbase-S1-4c"), overlap=1.5)
    ops = C._attn_core(C._layer_ops(opgen.decode(4, 256), 4, 4, 4))
    for k in (Knobs(), Knobs(lmem_use=2624), Knobs(lmem_use=35392, dxa_depth=2)):
        prev = None
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            h = dataclasses.replace(hier_for("psbase-S1-4c"), overlap=w, warps_per_cta=8)
            r = MemHier(C.SHAPE, "psbase-S1-4c", hier=h, memo=0, q_w=4, q_a=4, q=4).cost_launch(ops, k)
            body = r.cycles - h.miss_lat - h.launch_overhead
            if w == 0.0:
                assert math.isclose(body, r.t_comp + r.t_mem, rel_tol=1e-9)
            if w == 1.0:
                n_cta = max(1, min(h.warps // h.warps_per_cta, int(h.lmem_bytes // k.lmem_use))) if k.lmem_use else 1
                single = k.lmem_use > 0 and k.dxa_depth == 0
                exp = max(r.t_comp, r.t_mem) + (min(r.t_comp, r.t_mem) / n_cta if single else 0.0)
                assert math.isclose(body, exp, rel_tol=1e-9)
            if prev is not None:
                assert r.cycles <= prev + 1e-9
            prev = r.cycles


def test_calibrated_set_does_not_touch_defaults():
    """The calibration builds a separate Hier/Machine; memhier's psbase hierarchy and roofline machine are unchanged."""
    h, m = C.base_hier(), C.base_machine()
    th = dict(C.defaults(), eta_fpu=0.1, eta_dram=0.5, overlap=0.3, launch_overhead=999.0)
    h2, m2 = C.hier_for_theta(th), C.machine_for_theta(th)
    assert h2.eta_dram == 0.5 and h.eta_dram == 1.0 and h.overlap == 1.0 and h.launch_overhead == 0.0
    assert m2.flops_per_cycle["fpu"] == pytest.approx(0.1 * m.flops_per_cycle["fpu"])
    assert hier_for("psbase-S1-4c") == h


# ------------------------------------------------------------------------------------------------- metrics, tension


def test_metrics():
    m = C.metrics(np.array([110.0, 90.0, 200.0]), np.array([100.0, 100.0, 100.0]))
    assert m["mape"] == pytest.approx((10 + 10 + 100) / 3)
    assert m["median_ape"] == pytest.approx(10.0) and m["max_ape"] == pytest.approx(100.0)
    assert m["log_rmse"] == pytest.approx(math.sqrt((math.log(1.1) ** 2 + math.log(0.9) ** 2 + math.log(2) ** 2) / 3))


def test_tension_metric_reproduces_p32_on_measured_cycles():
    """phase_R on measured cycles equals the earlier time-weighted R (results/<run>/tension.csv), and the per-type bests are
    the earlier (16 ways, PLRU, DXA d2, LMEM 32 KiB; L2 size ¼ / 1× / ¼)."""
    rows = C.load_table()
    cyc = {(r["op"], r["cell"], r["setting"]): float(r["cycles"]) for r in rows if r["how"] != "variant"}
    R = C.phase_R(cyc, rows)
    with open(PS / "results" / "P3.2" / "tension.csv") as f:
        ref = {(r["phase"], r["setting"]): float(r["R_time_weighted"]) for r in csv.DictReader(f)}
    for k, v in ref.items():
        assert R[k] == pytest.approx(v, abs=6e-6), k
    t = {x["knob"]: x for x in C.tension(R)}
    assert (t["L2 size"]["best_prefill"], t["L2 size"]["best_decode"], t["L2 size"]["best_mixed"]) == \
        ("l2_quarter", "base", "l2_quarter")
    assert t["L2 size"]["max_cross_loss"] == pytest.approx(0.00803, abs=1e-4) and t["L2 size"]["material"] == 0
    assert t["L2 ways"]["best_decode"] == "l2_ways16" and t["DXA depth"]["best_prefill"] == "dxa2"
    assert t["LMEM budget"]["best_prefill"] == "lmem32k" and t["LMEM budget"]["best_decode"] == "n/a"
    assert not any(x["material"] for x in t.values())


# ------------------------------------------------------------------------------------------------- synthetic recovery

TRUE = dict(eta_fpu=0.05, eta_tcu=1.2, eta_dram=0.5, overlap=0.4, lat_miss_scale=1.6, lat_hit_scale=1.3,
            mlp_per_warp=0.6, launch_overhead=400.0)
TRUE_HIT = ("cliff", 1.0)


def _synthetic(cs, theta, hit, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for c in cs:
        y = C.predict_case(theta, c, hit) * (math.exp(rng.normal(0, noise)) if noise else 1.0)
        out.append(dataclasses.replace(c, measured=y))
    return out


def test_fit_recovers_known_parameters_on_synthetic_data(cases):
    """Noise-free synthetic 'measurements' generated by the model at known θ* on the real train rows: the fit,
    started from the uncalibrated defaults plus seeded random starts, recovers every parameter (relative error
    ≤ 2 %; overlap and launch_overhead absolute-relative) and predicts the held-out rows exactly."""
    cs, _, _ = cases
    syn = _synthetic(cs, TRUE, TRUE_HIT)
    train = [c for c in syn if c.split == "train"]
    test = [c for c in syn if c.split == "test"]
    fr = C.fit(train, C.PARAMS, TRUE_HIT, n_starts=4, seed=0)
    assert fr.loss < 1e-10
    for k, v in TRUE.items():
        assert fr.theta[k] == pytest.approx(v, rel=0.02), (k, fr.theta[k], v)
    pt = C.predict(fr.theta, test, TRUE_HIT)
    assert C.metrics(pt, [c.measured for c in test])["max_ape"] < 1e-3


def test_fit_with_noise_and_hit_choice(cases):
    """With 5 % log-normal noise the discrete hit-model loop picks the generating hit model by train loss, and every
    parameter the local identifiability report calls well determined lands near θ*. (Latency scale and MLP per warp
    trade off along the Little's-law ratio L0 / slots, so with noise they need not be individually recovered.)"""
    cs, _, _ = cases
    hit = ("proportional", 1.0)
    syn = _synthetic(cs, TRUE, hit, noise=0.05, seed=7)
    train = [c for c in syn if c.split == "train"]
    best, allh = C.fit_all_hits(train, C.PARAMS, None, n_starts=3, seed=0,
                                hits=(("cliff", 1.0), ("proportional", 1.0)))
    assert best.hit == hit
    ident = {r["param"]: r for r in C.identifiability(best, C.PARAMS)}
    assert set(ident) == set(C.defaults())
    # parameters the identifiability report calls well determined (relative standard error < 10 %) are recovered
    # within max(15 %, 3 standard errors); the compute efficiencies must be among them
    good = [k for k, r in ident.items() if r["identified"] and r["rel_se"] < 0.10]
    assert {"eta_fpu", "eta_tcu"} <= set(good), ident
    for k in good:
        tol = max(0.15, 3 * ident[k]["rel_se"])
        assert best.theta[k] == pytest.approx(TRUE[k], rel=tol), (k, best.theta[k], TRUE[k])


# ------------------------------------------------------------------------------------------------- stored results


@pytest.mark.skipif(not (RES / "fit.csv").exists(), reason="results/P3.4 not generated yet")
def test_results_reproduce_from_stored_fit(cases):
    """predictions.csv (primary) is reproduced by the code from fit.csv, and the README verdict matches mape.csv."""
    cs, _, _ = cases
    th, hit = C.load_fit(RES / "fit.csv", "primary")
    pred = {c.key: C.predict_case(th, c, hit) for c in cs}
    with open(RES / "predictions.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] == "primary"]
    assert len(rows) == len(cs)
    for r in rows:
        k = f"{r['op']}|{r['cell']}|{r['setting']}"
        assert float(r["predicted"]) == pytest.approx(pred[k], rel=1e-5)
    with open(RES / "mape.csv") as f:
        m = next(r for r in csv.DictReader(f) if (r["variant"], r["split"], r["group_kind"]) == ("primary", "test", "all"))
    qualitative = float(m["mape"]) > C.MAPE_QUALITATIVE
    readme = (RES / "README.md").read_text()
    verdict = readme.split("\n\n")[1] if readme.startswith("# ") else readme   # the paragraph after the title
    mape_txt = f"{float(m['mape']):.1f} %"
    assert mape_txt in verdict, (mape_txt, verdict[:200])
    if qualitative:            # > 25 %: the verdict paragraph says "qualitative only", never "quantitative"
        assert "**qualitative only**" in verdict and "quantitative" not in verdict.lower().replace("qualitative", "")
    else:                      # <= 25 %: it must say "quantitative" and must not say "qualitative only"
        assert "quantitative" in verdict.lower() and "qualitative only" not in verdict.lower()
    with open(RES / "summary.json") as f:
        import json
        assert json.load(f)["verdict"] == ("qualitative only" if qualitative else "quantitative (MAPE <= 25 %)")
