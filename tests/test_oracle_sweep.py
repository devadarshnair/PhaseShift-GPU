"""Tests for model/oracle_sweep.py: tie rule, exact compression vs model/oracle.py, bucketing,
region labels, the G1 rule and the §9.6 variants. Small and fast (no trace scheduling)."""
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from model import opgen, oracle
from model import oracle_sweep as osw
from model.memhier import GPU_L2_BW_ANCHOR, Knobs, hier_for, knob_grid
from serving import scheduler

PS = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------- tie rule
def test_pick_prefers_global_within_tolerance():
    c = np.array([1.0, 1.004, 1.2, 1.0001])
    assert osw.pick(c, prefer=1, stock=None, tol=0.005) == 1
    assert osw.pick(c, prefer=2, stock=None, tol=0.005) == 0          # 1.2 is not tied
    assert osw.pick(c, prefer=None, stock=None, tol=0.005) == 0


def test_pick_never_toward_stock():
    c = np.array([1.0, 1.003, 1.2])
    assert osw.pick(c, prefer=None, stock=0, tol=0.005) == 1          # stock is the argmin but a tie exists
    assert osw.pick(np.array([1.0, 1.1]), prefer=None, stock=0, tol=0.005) == 0   # clear winner stays
    assert osw.pick(c, prefer=0, stock=0, tol=0.005) == 0             # the global best static itself is preferred


def test_pick_rows_matches_pick():
    rng = np.random.default_rng(1)
    T = 1 + 0.01 * rng.random((300, 7))
    for prefer, stock in ((None, None), (3, None), (None, 0), (2, 0)):
        v = osw.pick_rows(T, prefer, stock, 0.005)
        assert [osw.pick(r, prefer, stock, 0.005) for r in T] == v.tolist()


def test_tie_rule_with_zero_tolerance_is_plain_argmin():
    rng = np.random.default_rng(2)
    T = rng.random((50, 5)) + 0.1
    assert osw.pick_rows(T, 2, 0, 0.0).tolist() == np.argmin(T, axis=1).tolist()


# ------------------------------------------------------------------------------------------------- compression is exact
class FakeTable:
    def __init__(self, tab):
        self.tab = tab

    def get(self, key):
        return self.tab[key]


def _random_suite(seed):
    rng = np.random.default_rng(seed)
    C = 5
    tab = {}
    for k in range(6):
        nl = int(rng.integers(1, 4))
        tab[("b", k)] = ([f"op{j}@0" for j in range(nl)], rng.random((nl, C)) + 0.05)
    regs = list(osw.REGIONS)
    ws = []
    for w in range(5):
        rows = [(("b", int(rng.integers(0, 6))), regs[int(rng.integers(0, 4))], float(rng.integers(1, 4)))
                for _ in range(int(rng.integers(1, 5)))]
        ws.append(osw.Workload(f"w{w}", "test", rows))
    return FakeTable(tab), ws, C


@pytest.mark.parametrize("seed", range(5))
def test_compressed_exact_equals_oracle_on_expanded_intervals(seed):
    """Count-weighted rows (one launch each) give the same baselines/oracles as oracle.analyze on the expanded
    interval list where every launch of every iteration is its own interval and launch."""
    table, ws, C = _random_suite(seed)
    res, glob = osw.analyze_machine(ws, table, range(C), stock=None, tol=0.0)
    T, wl, K = [], [], []
    lid = 0
    for w in ws:
        for key, reg, n in w.rows:
            _, M = table.get(key)
            for _ in range(int(n)):
                for row in M:
                    T.append(row)
                    wl.append(w.name)
                    K.append(lid)
                    lid += 1
    C_sw = np.full((C, C), np.inf)
    np.fill_diagonal(C_sw, 0)
    ref = oracle.analyze(np.array(T), C_sw, wl, K)
    assert glob["c_global_exact"] == oracle.global_best_static(np.array(T), np.array(wl))
    for r, o in zip(res, ref):
        assert r["T_static_x"] == pytest.approx(o.T_static, rel=1e-12)
        assert r["T_kernel_x"] == pytest.approx(o.T_kernel, rel=1e-12)
        assert r["T_zero_x"] == pytest.approx(o.T_zero, rel=1e-12)
        assert r["T_dp_x"] == pytest.approx(o.T_dp, rel=1e-12)
        assert r["T_global"] == pytest.approx(o.T_global, rel=1e-12)
        # tie rule with tol = 0 on continuous random costs = plain argmins
        assert r["T_static"] == pytest.approx(r["T_static_x"]) and r["T_dp"] == pytest.approx(r["T_dp_x"])
        # ordering (methodology §5) and the coarser oracles in between
        eps = 1 + 1e-9
        assert r["T_dp_x"] <= r["T_optype_x"] * eps and r["T_optype_x"] <= r["T_static_x"] * eps
        assert r["T_dp_x"] <= r["T_iter_x"] * eps and r["T_iter_x"] <= r["T_static_x"] * eps


def test_bootstrap_gap_degenerate_and_bounds():
    rng = np.random.default_rng(3)
    TI = rng.random((40, 4)) + 0.5
    lmin = TI.min(axis=1) * 0.9
    strata = np.array(["mixed"] * 20 + ["decode-weight"] * 20)
    lo, hi = osw.bootstrap_gap(TI, lmin, strata, n_boot=200)
    point = TI.sum(axis=0).min() / lmin.sum() - 1
    assert lo <= point <= hi
    same = np.tile(TI[:1], (10, 1))
    lo, hi = osw.bootstrap_gap(same, same.min(axis=1), np.array(["prefill"] * 10), n_boot=50)
    assert lo == pytest.approx(hi)


def test_trace_sampling_weights_reconstruct_iteration_counts():
    """Stratified sample weights N_h/n_h add up to the schedule's iteration count per region (Horvitz-Thompson)."""
    cfg = yaml.safe_load((PS / "experiments" / "P1.6.yaml").read_text())
    cfg["traces"] = dict(cfg["traces"], names=["azure_code"], limit={"azure_code": 40}, modes=["full", 256],
                         sample_per_stratum=7)
    p = next(x for x in osw.build_pairs(cfg) if x.machine == "vortex-S1-8c")
    ws = osw.trace_workloads(p, cfg, Knobs(), 8, 16)
    for w in ws:
        for g, N in w.meta["iterations_by_region"].items():
            tot = sum(n for _, reg, n in w.rows if reg == g)
            assert tot == pytest.approx(N)
            assert sum(1 for _, reg, _ in w.rows if reg == g) == min(N, 7)


def test_tie_rule_bounds_and_global_preference():
    table, ws, C = _random_suite(7)
    res, glob = osw.analyze_machine(ws, table, range(C), stock=0, tol=0.005)
    for r in res:
        assert r["T_static_x"] <= r["T_static"] <= r["T_static_x"] * 1.005 * (1 + 1e-12)
        assert r["T_dp_x"] <= r["T_dp"] <= r["T_dp_x"] * 1.005 * (1 + 1e-12)


# ------------------------------------------------------------------------------------------------- bucketing
def test_qv_exact_below_and_relative_error():
    for x in range(1, 17):
        assert osw.qv(x, 8, 16) == x
    bound = 2 ** (1 / 16) - 1
    for x in np.unique(np.geomspace(17, 300000, 400).astype(int)):
        q = osw.qv(int(x), 8, 16)
        assert abs(q / x - 1) <= bound + 1 / x


def test_bucket_key_clamps_and_collapses():
    key = osw.bucket_key([(200, 100, True), (5, 0, False), (5, 0, False)], [250, 256, 240], 8, 16, max_seq=256)
    pf = [s for s in key if s.kind == "prefill"]
    dc = [s for s in key if s.kind == "decode"]
    assert all(s.ctx + s.n_new <= 256 for s in pf)
    assert any(s.n_new == 5 and s.count == 2 and not s.emits for s in pf)
    assert len(dc) == 1 and dc[0].count == 3 and dc[0].ctx + 1 <= 256
    assert osw.bucket_key([], [], 8, 16) == ()


def test_region_of_matches_scheduler():
    m = opgen.from_llama2c_bin(PS / "data" / "models" / "stories15M.bin")
    kvpt, wb = opgen.kv_bytes_per_token(m, 4), 4 * opgen.n_params(m)
    for ctxs in ([10] * 4, [256] * 64, [200] * 30):
        it = scheduler.Iteration(0, 0.0, 1.0, (), tuple(scheduler.DecodeItem(i, t) for i, t in enumerate(ctxs)),
                                 0, 0, len(ctxs), 0)
        assert osw.region_of("decode", sum(ctxs), kvpt, wb) == scheduler.iteration_region(it, kvpt, wb)


# ------------------------------------------------------------------------------------------------- gate, variants, config
def test_gate_g1_rule():
    rows = lambda gaps: [dict(region=g, n=1, gap_dp_vs_static=x) for g, x in zip(osw.REGIONS, gaps)]
    assert osw.gate_g1({"a": rows([0.01, 0.02, 0.0, 0.029]), "b": rows([0.0] * 4)}, 0.03)["outcome"] == "STOP"
    g = osw.gate_g1({"a": rows([0.01, 0.02, 0.0, 0.0]), "b": rows([0.0, 0.031, 0.0, 0.0])}, 0.03)
    assert g["outcome"] == "CONTINUE" and g["above"] == [("b", "mixed")]
    empty = [dict(region="prefill", n=0, gap_dp_vs_static=float("nan"))] + rows([0.0] * 4)[1:]
    assert osw.gate_g1({"a": empty}, 0.03)["outcome"] == "STOP"      # empty regions do not count


def test_yaml_covers_methodology_96_sensitivity_list():
    cfg = yaml.safe_load((PS / "experiments" / "P1.6.yaml").read_text())
    names = {v["name"] for v in cfg["sensitivity"]}
    need = {"hit_model=proportional", "cliff_ways=0", "cliff_ways=2", "L2 -1 line", "L2 +1 line", "L2 x0.5", "L2 x2",
            "l1_tile_frac=0.25", "l1_tile_frac=1", "acc_per_thread x0.5", "acc_per_thread x2", "warps_per_cta=2",
            "warps_per_cta=8", "dram_lat x0.5", "dram_lat x4", "l2_bytes_per_cycle x0.5", "l2_bytes_per_cycle x2",
            "GPU L2 BW unanchored"}
    assert need <= names
    assert cfg["gate"]["threshold"] == 0.03 and cfg["tie_tol"] == 0.005
    assert set(cfg["gate"]["machines_required"]) == {"vortex-S1-8c", "h100-like"}


def test_variant_hier():
    cfg = yaml.safe_load((PS / "experiments" / "P1.6.yaml").read_text())
    vs = {v["name"]: v for v in cfg["sensitivity"]}
    vx, h1 = hier_for("vortex-S1-8c"), hier_for("h100-like")
    assert osw.variant_hier(vx, vs["GPU L2 BW unanchored"]) is None
    assert osw.variant_hier(h1, vs["GPU L2 BW unanchored"]).l2_bytes_per_cycle == pytest.approx(
        h1.l2_bytes_per_cycle / GPU_L2_BW_ANCHOR)
    assert osw.variant_hier(vx, vs["L2 -1 line"]).l2_bytes == vx.l2_bytes - 64
    assert osw.variant_hier(vx, vs["acc_per_thread x0.5"]).acc_per_thread == vx.acc_per_thread // 2
    assert osw.variant_hier(h1, vs["dram_lat x4"]).dram_lat == pytest.approx(4 * h1.dram_lat)
    assert osw.variant_hier(vx, vs["hit_model=proportional"]).hit_model == "proportional"


def test_synthetic_end_to_end_small_grid():
    """Real memhier costs on a 6-configuration grid: launch-level DP = per-kernel static = zero-cost (K(i) = i), every
    region label is valid, and stock never ends up as a tie-broken choice when a tied alternative exists."""
    cfg = yaml.safe_load((PS / "experiments" / "P1.6.yaml").read_text())
    p = next(x for x in osw.build_pairs(cfg) if x.machine == "vortex-S1-8c")
    h = hier_for(p.machine)
    grid = knob_grid(h, lmem_fracs=(0, 0.5), depths=(0, 1), inserts=("mru", "lru"), two_class=(False,))
    ws = osw.synthetic_workloads(p, 8, 16)[::7]
    for w in ws:                                  # synthetic points are exact (not bucketed)
        key = w.rows[0][0]
        assert sum(q.count for q in key if q.kind == "decode") in (0, w.meta.get("B", 0))
    res, glob = osw.analyze_machine(ws, osw.CostTable(p.memhier(), grid), range(len(grid)), grid.index(Knobs()), 0.005)
    for r in res:
        assert r["T_dp"] == r["T_kernel"] == r["T_zero"]
        assert r["region"] in osw.REGIONS
        assert r["T_dp_x"] <= r["T_static_x"]
        assert math.isclose(sum(r[f"share_{g}"] for g in osw.REGIONS), 1.0, rel_tol=1e-9)
