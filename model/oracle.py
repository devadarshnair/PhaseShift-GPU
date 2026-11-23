"""PhaseShift oracle analysis: static baselines, zero-cost and switch-cost-aware (DP) oracles,
capture (E10), regret (E8), geomean and bootstrap CIs. Definitions: `docs/methodology.md` §3 (baselines) and §5
(oracle, ordering T_zero ≤ T_DP ≤ T_kernel ≤ T_static); implementation choices: methodology §5 "Implementation notes
".

Inputs (one row per interval i, intervals in execution order; numpy arrays):
  T[i, c]      cost of interval i under configuration c (cycles or seconds; finite, ≥ 0);
  C_sw[c′, c]  cost of switching from c′ to c inside a kernel launch (≥ 0, may be +inf; C_sw[c, c] must be 0);
  workload[i]  workload label (a workload's intervals form one contiguous block);
  launch[i]    kernel-launch id K(i): integers, non-decreasing within each workload (launch order);
  (a kernel-type label is not needed: every baseline groups by launch, methodology §3; callers may keep one
   alongside for diagnostics);
  E[i, c]      optional energy matrix, evaluated on the chosen schedules.
Intervals are work units (methodology §5), so T[i, ·] is comparable across configurations.
An infeasible configuration for some interval (e.g. a tile that does not fit) is passed as a large finite cost
(e.g. 1e30), never inf or NaN: T must be finite so that sums, ratios and the DP stay defined; the DP then avoids it
whenever any feasible schedule exists. A forbidden switch, in contrast, is C_sw = +inf (allowed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


# ------------------------------------------------------------------------------------------------- validation


def validate(T, C_sw, launch, workload=None):
    """Check shapes and the §5 preconditions; return (T, C_sw, launch, workload) as numpy arrays.

    Rejects: non-2-D or empty T; non-finite or negative T; a workload whose zero-cost oracle T_zero is 0 (then every
    ratio is 0/0 or x/0); C_sw not |C|×|C|; negative or NaN C_sw; C_sw(c, c) ≠ 0;
    a missing launch id (None, NaN, wrong length, non-integer); launch ids that decrease within a workload; a
    workload whose intervals are not contiguous."""
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] == 0 or T.shape[1] == 0:
        raise ValueError("T must be a non-empty n×|C| matrix")
    if not np.all(np.isfinite(T)) or np.any(T < 0):
        raise ValueError("T must be finite and ≥ 0")
    n, C = T.shape
    C_sw = np.asarray(C_sw, dtype=float)
    if C_sw.shape != (C, C):
        raise ValueError(f"C_sw must be {C}×{C}")
    if np.any(np.isnan(C_sw)) or np.any(C_sw < 0):
        raise ValueError("C_sw must be ≥ 0 (inf allowed)")
    if np.any(np.diag(C_sw) != 0):
        raise ValueError("C_sw(c, c) must be 0 (staying in a configuration costs nothing)")
    if launch is None:
        raise ValueError("a kernel-launch id K(i) is required for every interval")
    launch_l = list(launch)
    if len(launch_l) != n or any(k is None or (isinstance(k, float) and math.isnan(k)) for k in launch_l):
        raise ValueError("launch ids missing")
    if any(not float(k).is_integer() for k in launch_l):
        raise ValueError("launch ids must be integers")
    launch = np.asarray(launch_l, dtype=np.int64)
    workload = np.zeros(n, dtype=np.int64) if workload is None else np.asarray(workload)
    if len(workload) != n:
        raise ValueError("workload labels missing")
    seen, prev = set(), object()
    for w in workload.tolist():
        if w != prev:
            if w in seen:
                raise ValueError(f"workload {w!r} is not contiguous")
            seen.add(w)
            prev = w
    for w, idx in _blocks(workload):
        if np.any(np.diff(launch[idx]) < 0):
            raise ValueError(f"launch ids decrease within workload {w!r} (must be in launch order)")
        if T[idx].min(axis=1).sum() == 0:
            raise ValueError(f"workload {w!r} has zero oracle cost (every interval has a zero-cost configuration): "
                             f"speedups, gap ratios and capture would be undefined")
    return T, C_sw, launch, workload


def _blocks(workload) -> list[tuple[object, np.ndarray]]:
    """(label, interval indices) per workload, in order of first appearance."""
    out, start = [], 0
    wl = list(workload)
    for i in range(1, len(wl) + 1):
        if i == len(wl) or wl[i] != wl[start]:
            out.append((wl[start], np.arange(start, i)))
            start = i
    return out


def switch_cost_at(C_sw: np.ndarray, launch: np.ndarray, i: int) -> np.ndarray:
    """C_sw,i (methodology §5): all zeros if interval i starts a new launch (K(i) ≠ K(i−1)), else C_sw."""
    return np.zeros_like(C_sw) if launch[i] != launch[i - 1] else C_sw


# ------------------------------------------------------------------------------------------------- per-workload costs


def schedule_cost(T, C_sw, launch, schedule) -> float:
    """Cost of one workload's schedule c_1…c_n: Σ_i T(i, c_i) + Σ_{i≥2} C_sw,i(c_{i−1}, c_i) (initial config free)."""
    T, C_sw, launch, _ = validate(T, C_sw, launch)
    return float(interval_costs(T, C_sw, launch, schedule).sum())


def interval_costs(T, C_sw, launch, schedule) -> np.ndarray:
    """Per-interval cost of a schedule: T(i, c_i) plus the switch cost paid on entering interval i (charged to the
    interval the new configuration serves). Used for per-interval regret (E8)."""
    T = np.asarray(T, dtype=float)
    C_sw = np.asarray(C_sw, dtype=float)
    s = np.asarray(schedule, dtype=np.int64)
    launch = np.asarray(launch)
    cost = T[np.arange(len(s)), s].copy()
    for i in range(1, len(s)):
        cost[i] += switch_cost_at(C_sw, launch, i)[s[i - 1], s[i]]
    return cost


def dp_oracle(T, C_sw, launch) -> tuple[float, np.ndarray]:
    """Switch-cost-aware oracle (methodology §5) for ONE workload:
    J(1, c) = T(1, c);  J(i, c) = T(i, c) + min_c′ [J(i−1, c′) + C_sw,i(c′, c)];  T_DP = min_c J(n, c).
    O(n·|C|²) time, O(n·|C|) back-pointers; returns (T_DP, schedule). Ties go to the lowest configuration index
    (np.argmin), so the result is deterministic."""
    T, C_sw, launch, _ = validate(T, C_sw, launch)
    n, C = T.shape
    J = T[0].copy()
    back = np.zeros((n, C), dtype=np.int64)
    back[0] = np.arange(C)
    for i in range(1, n):
        cand = J[:, None] + switch_cost_at(C_sw, launch, i)    # cand[c′, c] = J(i−1, c′) + C_sw,i(c′, c)
        back[i] = np.argmin(cand, axis=0)
        J = T[i] + cand[back[i], np.arange(C)]
    c = int(np.argmin(J))
    best = float(J[c])
    sched = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        sched[i] = c
        c = int(back[i, c])
    return best, sched


def zero_cost_oracle(T) -> tuple[float, np.ndarray]:
    """T_zero = Σ_i min_c T(i, c): best configuration per interval, switching free (methodology §5)."""
    T = np.asarray(T, dtype=float)
    s = np.argmin(T, axis=1)
    return float(T[np.arange(len(T)), s].sum()), s


def best_static(T) -> tuple[float, int]:
    """Per-workload best static (methodology §3): T_static = min_c Σ_i T(i, c); returns (cost, c)."""
    tot = np.asarray(T, dtype=float).sum(axis=0)
    c = int(np.argmin(tot))
    return float(tot[c]), c


def per_kernel_static(T, launch) -> tuple[float, np.ndarray]:
    """Per-kernel best static (methodology §3): T_kernel = Σ_k min_c Σ_{i: K(i)=k} T(i, c), one configuration per
    launch, switching free at launch boundaries. Returns (cost, schedule)."""
    T = np.asarray(T, dtype=float)
    launch = np.asarray(launch)
    sched = np.empty(len(T), dtype=np.int64)
    total = 0.0
    for k in np.unique(launch):
        idx = np.flatnonzero(launch == k)
        tot = T[idx].sum(axis=0)
        c = int(np.argmin(tot))
        sched[idx] = c
        total += float(tot[c])
    return total, sched


# ------------------------------------------------------------------------------------------------- suite level


def geomean(x) -> float:
    """Geometric mean (used for speedups and normalized costs: they are ratios, methodology §3/§5)."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("geomean needs positive values")
    return float(np.exp(np.mean(np.log(x))))


def global_best_static(T, workload, mask=None) -> int:
    """c_global = argmin_c geomean_w [Σ_{i∈w} T(i, c) / T_static(w)] (methodology §3): each workload normalized by
    its own best static cost. With `mask` (boolean per interval) only the masked intervals count, and workloads
    without masked intervals are skipped — this gives the prefill-/decode-tuned static configurations."""
    T = np.asarray(T, dtype=float)
    mask = np.ones(len(T), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    ratios = []
    for _, idx in _blocks(workload):
        idx = idx[mask[idx]]
        if len(idx) == 0:
            continue
        tot = T[idx].sum(axis=0)
        ratios.append(tot / tot.min() if tot.min() > 0 else np.where(tot > 0, np.inf, 1.0))
    if not ratios:
        raise ValueError("no intervals selected")
    R = np.vstack(ratios)
    with np.errstate(divide="ignore"):
        score = np.exp(np.mean(np.log(R), axis=0))
    return int(np.argmin(score))


@dataclass
class WorkloadResult:
    """Per-workload costs (lower is better) and schedules of every baseline and oracle."""

    workload: object
    n: int
    T_static: float
    c_static: int
    T_global: float
    T_kernel: float
    T_zero: float
    T_dp: float
    T_prefill_tuned: float | None
    T_decode_tuned: float | None
    sched_dp: np.ndarray
    sched_kernel: np.ndarray
    E_static: float | None = None
    E_dp: float | None = None

    @property
    def speedup_dp_vs_static(self) -> float:
        """G2 metric: T_static / T_DP."""
        return self.T_static / self.T_dp

    @property
    def speedup_dp_vs_kernel(self) -> float:
        """G2b metric: T_kernel / T_DP."""
        return self.T_kernel / self.T_dp

    @property
    def gap_abs(self) -> float:
        """Absolute oracle gap T_static − T_DP (always reported next to ratios, methodology §5)."""
        return self.T_static - self.T_dp


def analyze(T, C_sw, workload, launch, phase=None, E=None) -> list[WorkloadResult]:
    """Run every baseline and oracle of methodology §3/§5 on a suite.

    phase: optional per-interval label in {"prefill", "decode", …}; the prefill-/decode-tuned static configurations
    are global_best_static restricted to that phase's intervals (methodology §5 implementation notes), then applied
    to the whole workload. E: optional energy matrix, reported for the per-workload static and DP schedules."""
    T, C_sw, launch, workload = validate(T, C_sw, launch, workload)
    c_glob = global_best_static(T, workload)
    c_pf = c_dec = None
    if phase is not None:
        phase = np.asarray(phase)
        if np.any(phase == "prefill"):
            c_pf = global_best_static(T, workload, phase == "prefill")
        if np.any(phase == "decode"):
            c_dec = global_best_static(T, workload, phase == "decode")
    out = []
    for w, idx in _blocks(workload):
        Tw, Kw = T[idx], launch[idx]
        ts, cs = best_static(Tw)
        tk, sk = per_kernel_static(Tw, Kw)
        tz, _ = zero_cost_oracle(Tw)
        td, sd = dp_oracle(Tw, C_sw, Kw)
        r = WorkloadResult(w, len(idx), ts, cs, float(Tw[:, c_glob].sum()), tk, tz, td,
                           None if c_pf is None else float(Tw[:, c_pf].sum()),
                           None if c_dec is None else float(Tw[:, c_dec].sum()), sd, sk)
        if E is not None:
            Ew = np.asarray(E, dtype=float)[idx]
            r.E_static, r.E_dp = float(Ew[:, cs].sum()), float(Ew[np.arange(len(idx)), sd].sum())
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------- capture, regret


def capture(T_policy: float, T_static: float, T_oracle: float) -> float:
    """E10 with performance P = 1/T (work per time; each workload's work is fixed, so this is throughput):
    capture = (1/T_policy − 1/T_static) / (1/T_oracle − 1/T_static). 0 = static, 1 = oracle. Undefined (nan) when
    the oracle gap is zero; callers report it only when the gap exceeds the sampling-error bound (§5)."""
    den = 1 / T_oracle - 1 / T_static
    return float("nan") if den == 0 else (1 / T_policy - 1 / T_static) / den


def capture_time(T_policy: float, T_static: float, T_oracle: float) -> float:
    """Reduction-ratio form for lower-is-better metrics (methodology E10 note): (T_static − T_policy) /
    (T_static − T_oracle). Also in [0, 1] when the policy lies between static and oracle."""
    den = T_static - T_oracle
    return float("nan") if den == 0 else (T_static - T_policy) / den


def regret_per_interval(cost_policy, cost_oracle) -> np.ndarray:
    """E8 per interval: R_i = (T_det,i − T_oracle,i) / T_oracle,i, with per-interval costs from `interval_costs`
    (switch costs charged to the interval entered). Individual R_i can be negative (the oracle may sacrifice an
    interval to save later)."""
    a, b = np.asarray(cost_policy, dtype=float), np.asarray(cost_oracle, dtype=float)
    return (a - b) / b


def regret_time_weighted(cost_policy, cost_oracle) -> float:
    """E8 time-weighted: Σ_i w_i·R_i with w_i = T_oracle,i / Σ T_oracle = (Σ T_det − Σ T_oracle) / Σ T_oracle."""
    a, b = np.asarray(cost_policy, dtype=float), np.asarray(cost_oracle, dtype=float)
    return float((a.sum() - b.sum()) / b.sum())


# ------------------------------------------------------------------------------------------------- statistics


def bootstrap_ci(values, stat: Callable = geomean, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
                 groups: Sequence | None = None) -> tuple[float, float, float]:
    """Percentile bootstrap: resample the units (workloads or seeds) with replacement n_boot times, recompute
    `stat`, return (stat(values), lower, upper) for a (1 − alpha) interval. With `groups`, whole groups are
    resampled (e.g. all workloads of one arrival seed together). Deterministic for a given seed."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        units = [np.array([i]) for i in range(len(v))]
    else:
        g = np.asarray(groups)
        units = [np.flatnonzero(g == u) for u in dict.fromkeys(g.tolist())]
    k = len(units)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        boots[b] = stat(v[np.concatenate([units[j] for j in pick])])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi)


def region_summary(results: Sequence[WorkloadResult], region: dict | None = None) -> dict:
    """Geomean (over each region's workloads) of DP speedup over per-workload and per-kernel best static, plus
    zero-cost speedup and total absolute gap (methodology §5). region: {workload: region name} (default: one
    region "all")."""
    region = region or {r.workload: "all" for r in results}
    out = {}
    for reg in dict.fromkeys(region[r.workload] for r in results):
        rs = [r for r in results if region[r.workload] == reg]
        out[reg] = dict(n=len(rs), speedup_dp_vs_static=geomean([r.speedup_dp_vs_static for r in rs]),
                        speedup_dp_vs_kernel=geomean([r.speedup_dp_vs_kernel for r in rs]),
                        speedup_zero_vs_static=geomean([r.T_static / r.T_zero for r in rs]),
                        gap_abs=sum(r.gap_abs for r in rs))
    return out
