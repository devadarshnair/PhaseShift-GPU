"""Tests (revision 2.3): DP = brute force over all schedules (n ≤ 8, |C| ≤ 4) with the
interval-dependent C_sw,i of methodology §5; input validation (C_sw(c,c) ≠ 0, missing / non-monotone launch id);
§5 ordering zero-cost ≤ DP ≤ T_kernel ≤ global best static; C_sw = ∞ ⇒ DP = T_kernel; C_sw = ∞ and one launch ⇒
DP = best static; C_sw = 0 ⇒ DP = zero-cost; DP invariant to C_sw when every interval is its own launch; capture in
[0, 1] between static and oracle; plus regret, bootstrap, tuned statics and the O(n·|C|²) implementation."""
import inspect
import itertools
import math

import numpy as np
import pytest

from model import oracle
from model.oracle import analyze, dp_oracle, per_kernel_static, schedule_cost, validate, zero_cost_oracle

INF = math.inf


def rand_instance(rng, n=None, C=None, p_new=0.3, inf_frac=0.0):
    n = n or int(rng.integers(1, 9))
    C = C or int(rng.integers(1, 5))
    T = rng.uniform(1, 10, size=(n, C))
    K = np.cumsum(np.r_[0, rng.random(n - 1) < p_new]).astype(int)
    csw = rng.uniform(0, 15, size=(C, C))
    csw[rng.random((C, C)) < inf_frac] = INF
    np.fill_diagonal(csw, 0)
    return T, csw, K


def brute(T, csw, K):
    """Exhaustive minimum over all |C|^n schedules (vectorized; independent of the DP code): Σ_i T(i, c_i) plus
    C_sw(c_{i-1}, c_i) wherever K(i) = K(i-1)."""
    n, C = T.shape
    S = np.array(list(itertools.product(range(C), repeat=n)))           # (|C|^n, n)
    cost = T[np.arange(n), S].sum(axis=1)
    for i in range(1, n):
        if K[i] == K[i - 1]:
            cost = cost + csw[S[:, i - 1], S[:, i]]
    return cost.min()


def test_dp_equals_brute_force():
    rng = np.random.default_rng(11)
    for trial in range(400):
        T, csw, K = rand_instance(rng, inf_frac=0.2 if trial % 2 else 0.0)
        best, sched = dp_oracle(T, csw, K)
        assert best == pytest.approx(brute(T, csw, K), rel=1e-12, abs=1e-12)
        assert schedule_cost(T, csw, K, sched) == pytest.approx(best, rel=1e-12)   # back-pointers recover it


def test_dp_matches_methodology_reference_recurrence():
    """Same recurrence as the reference implementation in tests/test_methodology.py::_dp (pure Python)."""
    from tests.test_methodology import _dp
    rng = np.random.default_rng(12)
    for _ in range(200):
        T, csw, K = rand_instance(rng, n=int(rng.integers(1, 30)))
        assert dp_oracle(T, csw, K)[0] == pytest.approx(_dp(T.tolist(), K.tolist(), csw.tolist()))


def test_switch_paid_only_inside_a_launch():
    T = np.array([[1.0, 10.0], [10.0, 1.0]])
    csw = np.array([[0.0, 5.0], [5.0, 0.0]])
    assert dp_oracle(T, csw, [0, 0])[0] == 7.0           # switch inside a launch costs 5: 1 + 1 + 5
    assert dp_oracle(T, csw, [0, 1])[0] == 2.0           # free at the launch boundary
    assert list(dp_oracle(T, csw, [0, 1])[1]) == [0, 1]
    T2 = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert dp_oracle(T2, csw, [0, 0])[0] == 3.0 and list(dp_oracle(T2, csw, [0, 0])[1]) == [0, 0]


# ------------------------------------------------------------------------------------------- validation
def test_validation_rejects_bad_inputs():
    T = np.ones((3, 2))
    ok = np.array([[0, 1], [1, 0]])
    validate(T, ok, [0, 0, 1])
    with pytest.raises(ValueError, match=r"C_sw\(c, c\)"):
        validate(T, np.array([[0, 1], [1, 0.5]]), [0, 0, 1])
    with pytest.raises(ValueError, match="launch id"):
        validate(T, ok, None)
    with pytest.raises(ValueError, match="missing"):
        validate(T, ok, [0, 1])
    with pytest.raises(ValueError, match="missing"):
        validate(T, ok, [0, None, 1])
    with pytest.raises(ValueError, match="missing"):
        validate(T, ok, [0, float("nan"), 1])
    with pytest.raises(ValueError, match="integers"):
        validate(T, ok, [0, 0.5, 1])
    with pytest.raises(ValueError, match="decrease"):
        validate(T, ok, [0, 2, 1])
    with pytest.raises(ValueError, match="≥ 0"):
        validate(T, np.array([[0, -1], [1, 0]]), [0, 0, 1])
    with pytest.raises(ValueError, match="finite"):
        validate(np.array([[1, INF], [1, 1], [1, 1]]), ok, [0, 0, 1])
    with pytest.raises(ValueError, match="C_sw must be"):
        validate(T, np.zeros((3, 3)), [0, 0, 1])
    with pytest.raises(ValueError, match="contiguous"):
        validate(np.ones((3, 2)), ok, [0, 0, 0], workload=["a", "b", "a"])
    with pytest.raises(ValueError, match="decrease"):
        validate(np.ones((4, 2)), ok, [3, 4, 1, 0], workload=["a", "a", "b", "b"])
    validate(np.ones((4, 2)), ok, [3, 4, 0, 1], workload=["a", "a", "b", "b"])   # ids restart per workload: fine
    with pytest.raises(ValueError, match="C_sw"):
        dp_oracle(T, np.array([[1, 1], [1, 0]]), [0, 0, 1])                       # the DP itself validates


# ------------------------------------------------------------------------------------------- §5 properties
def test_ordering_on_random_suites():
    rng = np.random.default_rng(13)
    for _ in range(200):
        nw = int(rng.integers(1, 5))
        Ts, Ks, W = [], [], []
        C = int(rng.integers(1, 5))
        for w in range(nw):
            T, _, K = rand_instance(rng, n=int(rng.integers(1, 12)), C=C)
            Ts.append(T)
            Ks.append(K)
            W += [w] * len(T)
        _, csw, _ = rand_instance(rng, n=1, C=C, inf_frac=0.2)
        T, K = np.vstack(Ts), np.concatenate(Ks)
        res = analyze(T, csw, W, K)
        for r in res:
            assert r.T_zero <= r.T_dp + 1e-9 <= r.T_kernel + 2e-9 <= r.T_static + 3e-9 <= r.T_global + 4e-9


def test_infinite_switch_cost_gives_per_kernel_static():
    rng = np.random.default_rng(14)
    for _ in range(200):
        T, _, K = rand_instance(rng, n=int(rng.integers(1, 15)))
        C = T.shape[1]
        csw = np.full((C, C), INF)
        np.fill_diagonal(csw, 0)
        assert dp_oracle(T, csw, K)[0] == pytest.approx(per_kernel_static(T, K)[0])
        one = np.zeros(len(T), dtype=int)                                   # single launch ⇒ best static
        assert dp_oracle(T, csw, one)[0] == pytest.approx(oracle.best_static(T)[0])


def test_zero_switch_cost_gives_zero_cost_oracle():
    rng = np.random.default_rng(15)
    for _ in range(200):
        T, _, K = rand_instance(rng, n=int(rng.integers(1, 15)))
        C = T.shape[1]
        assert dp_oracle(T, np.zeros((C, C)), K)[0] == pytest.approx(zero_cost_oracle(T)[0])


def test_dp_invariant_to_csw_when_every_interval_is_a_launch():
    rng = np.random.default_rng(16)
    for _ in range(200):
        T, csw, _ = rand_instance(rng, n=int(rng.integers(1, 15)), inf_frac=0.3)
        K = np.arange(len(T))
        C = T.shape[1]
        d = dp_oracle(T, csw, K)[0]
        assert d == dp_oracle(T, np.zeros((C, C)), K)[0] == pytest.approx(zero_cost_oracle(T)[0])
        assert d == pytest.approx(per_kernel_static(T, K)[0])


def test_capture_in_unit_interval_between_static_and_oracle():
    rng = np.random.default_rng(17)
    for _ in range(500):
        T, csw, K = rand_instance(rng, n=int(rng.integers(2, 12)), C=int(rng.integers(2, 5)))
        ts = oracle.best_static(T)[0]
        td = dp_oracle(T, csw, K)[0]
        if ts - td < 1e-9:
            assert math.isnan(oracle.capture(td, ts, td)) or oracle.capture(td, ts, td) == pytest.approx(1)
            continue
        tp = td + rng.random() * (ts - td)                    # a practical policy between oracle and static
        for f in (oracle.capture, oracle.capture_time):
            c = f(tp, ts, td)
            assert -1e-12 <= c <= 1 + 1e-12
        assert oracle.capture(ts, ts, td) == pytest.approx(0) and oracle.capture(td, ts, td) == pytest.approx(1)
    assert oracle.capture_time(15, 20, 10) == 0.5 and oracle.capture(15, 20, 10) == pytest.approx((1/15 - 1/20) / (1/10 - 1/20))
    assert math.isnan(oracle.capture(5, 5, 5))


def test_regret():
    T = np.array([[1.0, 3.0], [3.0, 1.0], [1.0, 3.0]])
    csw = np.array([[0, 0.5], [0.5, 0]])
    K = [0, 0, 0]
    _, s_or = dp_oracle(T, csw, K)
    assert list(s_or) == [0, 1, 0]
    c_or = oracle.interval_costs(T, csw, K, s_or)
    assert list(c_or) == [1.0, 1.5, 1.5]
    c_det = oracle.interval_costs(T, csw, K, [0, 0, 0])
    r = oracle.regret_per_interval(c_det, c_or)
    assert list(r) == pytest.approx([0, 1.0, -1 / 3])
    tw = oracle.regret_time_weighted(c_det, c_or)
    assert tw == pytest.approx((5 - 4) / 4) == pytest.approx(np.sum(r * c_or / c_or.sum()))


def test_tuned_statics_and_global():
    # workload a: prefill likes c0, decode likes c1; workload b likewise
    T = np.array([[1, 5, 3], [5, 1, 3], [1, 5, 3], [5, 1, 3]], dtype=float)
    W, K = ["a", "a", "b", "b"], [0, 1, 0, 1]
    phase = np.array(["prefill", "decode", "prefill", "decode"])
    res = analyze(T, np.zeros((3, 3)) + np.where(np.eye(3), 0, 100), W, K, phase=phase)
    for r in res:
        assert r.T_prefill_tuned == 6 and r.T_decode_tuned == 6
        assert r.T_static == 6 and r.T_kernel == 2 and r.T_dp == 2 and r.T_zero == 2
        assert r.speedup_dp_vs_static == 3 and r.speedup_dp_vs_kernel == 1 and r.gap_abs == 4
    # the global best static normalizes per workload: c1 is 1.0x for w2 and 1.25x for w1 -> beats c0 (1.0x, 2x)
    T2 = np.array([[4, 5], [10, 5]], dtype=float)
    assert oracle.global_best_static(T2, ["w1", "w2"]) == 1
    E = T * 2
    r = analyze(T, np.zeros((3, 3)), W, K, E=E)[0]
    assert r.E_static == 12 and r.E_dp == 4


def test_bootstrap_and_summary():
    v = np.array([1.1, 1.2, 1.05, 1.3, 1.15, 1.25])
    s, lo, hi = oracle.bootstrap_ci(v, n_boot=2000, seed=3)
    assert s == pytest.approx(oracle.geomean(v)) and lo <= s <= hi and lo >= v.min() and hi <= v.max()
    assert oracle.bootstrap_ci(v, n_boot=2000, seed=3) == (s, lo, hi)            # deterministic
    s2, lo2, hi2 = oracle.bootstrap_ci(v, n_boot=2000, seed=3, groups=[0, 0, 1, 1, 2, 2])
    assert s2 == s and lo2 <= s2 <= hi2
    const = oracle.bootstrap_ci(np.full(5, 1.2), n_boot=500)
    assert const == pytest.approx((1.2, 1.2, 1.2))
    T = np.array([[2, 1], [2, 1], [1, 2], [1, 2]], dtype=float)
    res = analyze(T, np.zeros((2, 2)), ["a", "a", "b", "b"], [0, 0, 0, 0])
    summ = oracle.region_summary(res, {"a": "decode", "b": "prefill"})
    assert set(summ) == {"decode", "prefill"} and summ["decode"]["speedup_dp_vs_static"] == 1.0
    with pytest.raises(ValueError):
        oracle.geomean([1, 0])


def test_dp_is_quadratic_not_exponential():
    """O(n·|C|²); a 20,000-interval × 16-config instance runs in well under a second."""
    import time
    rng = np.random.default_rng(18)
    T, csw, K = rand_instance(rng, n=20000, C=16, p_new=0.01)
    t0 = time.perf_counter()
    dp_oracle(T, csw, K)
    assert time.perf_counter() - t0 < 5
    src = inspect.getsource(oracle)
    assert "itertools" not in src and "product(" not in src


def test_zero_cost_workloads_rejected():
    ok = np.zeros((2, 2))
    with pytest.raises(ValueError, match="zero oracle cost"):
        validate(np.zeros((3, 2)), ok, [0, 0, 1])
    with pytest.raises(ValueError, match="zero oracle cost"):          # every interval has a free configuration
        analyze(np.array([[0, 1], [1, 0.0]]), ok, ["a", "a"], [0, 0])
    with pytest.raises(ValueError, match="'b'"):
        validate(np.array([[1, 1], [0, 0.0]]), ok, [0, 0], workload=["a", "b"])
    res = analyze(np.array([[0, 1], [1, 1.0]]), ok, ["a", "a"], [0, 0])   # one free interval is fine
    assert res[0].T_dp == 1 and res[0].speedup_dp_vs_static == 1
    big = 1e30                                                           # infeasible config as a large finite cost
    r = analyze(np.array([[1, big], [big, 2.0]]), ok, ["a", "a"], [0, 1])[0]
    assert r.T_dp == 3 and r.T_static < big * 2


# ---- oracle.csv recomputation from results/<run>/costs.csv is deterministic --------------
def _p62():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "results" / "P6.2" / "oracle_study.py"
    if not p.exists() or not (p.parent / "oracle.csv").exists():
        pytest.skip("outputs not present")
    spec = importlib.util.spec_from_file_location("p62_oracle_study", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, p.parent


def test_p62_recomputation_deterministic():
    m, d = _p62()
    a, b = m.outputs(), m.outputs()
    assert a == b                                            # two in-process recomputations agree
    for name, text in a.items():                            # and equal the stored files byte for byte
        assert (d / name).read_text() == text, name


def test_p62_orderings_and_dp_equals_zero_cost():
    m, _ = _p62()
    out, summ, _, meta = m.analyze_all()
    for r in out:   # methodology §5 ordering; C_sw = 0 for policy-only knobs ⇒ DP = zero-cost oracle
        assert r["T_zero"] <= r["T_dp"] + 1e-6 <= r["T_kernel"] + 2e-6 <= r["T_static"] + 3e-6
        assert r["T_static"] <= r["T_global_static"] + 1e-6 and r["T_static"] <= r["T_default"] + 1e-6
        assert abs(r["T_dp"] - r["T_zero"]) < 1e-6
    assert {s["region"]: s["n"] for s in summ} == {"decode-weight": 16, "decode-KV": 10, "prefill": 16, "mixed": 16}
    for s in summ:
        assert s["speedup_dp_vs_static_lo"] <= s["speedup_dp_vs_static"] <= s["speedup_dp_vs_static_hi"]
    assert meta["G2"] in ("PASS", "FAIL")
