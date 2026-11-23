"""Discrete-event serving simulator (every definition and modeling choice: methodology §15).

Level 2 of the two-level methodology (§8): request arrivals are replayed against an iteration-level continuous-batching
engine whose iteration durations come cycle-cost table (`results/<run>/iteration_costs.csv`, columns
`c:<config>`), under a hardware *policy* that picks the configuration of every iteration:

* `StaticPolicy(c)` — every iteration runs configuration c (c = "base" is the baseline).
* `HintPolicy(region→c, csw_cycles)` — hint-driven PhaseShift: the serving engine knows each iteration's composition
  (prefill chunks / decodes, §12 hints), classifies it into a §4 region and runs that region's configuration; a
  switch between different configurations costs `csw_cycles` (C_sw, §13) added to the iteration that switches.
* `OraclePolicy(...)` — a DP-oracle schedule supplied from outside (produces them; this module only defines the
  interface, `load_oracle_schedule`): per iteration index or per region, either a configuration name or a cost ratio
  r = T_oracle / T_base applied to the interpolated base cost.

Engine semantics mirror serving/scheduler.py (methodology §10) with unlimited KV capacity (no preemption; see
§15 for why) and additionally record every output token's emission time, which TBT needs. `tests/test_des.py`
cross-checks first-token and finish times against `run_schedule` on the same inputs.

The event loop is a heapq of (time, kind, seq) events: ARRIVAL (0) < ITER_END (1) < SCHEDULE (2) at equal times, so
every request that arrives at time t is waiting before the iteration that starts at t is formed (arrival ≤ t).
"""
from __future__ import annotations

import csv
import dataclasses
import heapq
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from serving.traces import Request

PS_ROOT = Path(__file__).resolve().parents[1]
P61_DIR = PS_ROOT / "results" / "P6.1"
CLOCK_HZ = 400e6           # Vortex default core clock (methodology §11); cycles → seconds
KV_BOUND_SUM_T = 14_314    # grid (llama8b_s8, fp32): decode iteration is KV-bound iff Σ T ≥ 14,314 (ρ_KV ≥ 1)
REGIONS = ("prefill", "mixed", "decode-weight", "decode-kv")
# Default hint mapping = topk.json best-by-region per mechanism among the configurations that exist as table
# columns (B_s256 alone is in the table, but the region winners are combined only where a cross column exists).
DEFAULT_HINT_MAP = {"prefill": "B_s1024", "mixed": "D_byp", "decode-weight": "D_lru", "decode-kv": "D_lru"}

_ARRIVAL, _ITER_END, _SCHEDULE = 0, 1, 2


# ------------------------------------------------------------------------------------------------ iteration model
@dataclasses.dataclass(frozen=True)
class IterSummary:
    """What the cost model needs from one iteration (aggregates over its prefill chunks and decodes)."""
    idx: int
    prefill: tuple[tuple[int, int, int], ...]     # (rid, start offset s0, n tokens)
    decode_ctx: tuple[int, ...]                   # per decode: attended context T = stored + 1

    @property
    def kind(self) -> str:
        if self.prefill and self.decode_ctx:
            return "mixed"
        return "prefill" if self.prefill else "decode"

    @property
    def region(self) -> str:
        k = self.kind
        if k != "decode":
            return k
        return "decode-kv" if sum(self.decode_ctx) >= KV_BOUND_SUM_T else "decode-weight"

    def features(self) -> np.ndarray:
        """x = [1, N_pf, P_pf, N_dec, ΣT, ΣT·1[N_dec = 1]] (§15.2): prefill tokens, prefill attention pairs
        Σ n·(s0 + (n+1)/2), decodes, decode context sum, and the context sum again when there is a single decode
        (measures ≈ 2× the per-token attention cost at B = 1; empirical, see §15.2)."""
        npf = sum(n for _, _, n in self.prefill)
        pairs = sum(n * (s0 + (n + 1) / 2) for _, s0, n in self.prefill)
        st = float(sum(self.decode_ctx))
        return np.array([1.0, npf, pairs, len(self.decode_ctx), st, st if len(self.decode_ctx) == 1 else 0.0])


_KIND_COLS = {"decode": [0, 3, 4, 5], "prefill": [0, 1, 2], "mixed": [0, 1, 2, 3, 4]}


class CostTable:
    """Per (iteration kind, configuration) least-squares linear model iteration cost in cycles.

    Fit points: the workloads of that kind (grid + trace iterations). Non-gate use (allows
    interpolation here). `fit_report()` gives the relative residuals at the fit points.
    """

    def __init__(self, coefs: Mapping[tuple[str, str], np.ndarray], configs: Sequence[str],
                 residuals: Mapping[tuple[str, str], np.ndarray] | None = None):
        self.coefs = dict(coefs)
        self.configs = tuple(configs)
        self.residuals = dict(residuals or {})

    @classmethod
    def from_p61(cls, root: Path = P61_DIR) -> "CostTable":
        wl = {r["workload"]: r for r in csv.DictReader(open(root / "workloads.csv"))}
        rows = list(csv.DictReader(open(root / "iteration_costs.csv")))
        configs = [k[2:] for k in rows[0] if k.startswith("c:")]
        pts: dict[str, list[tuple[np.ndarray, dict]]] = {"decode": [], "prefill": [], "mixed": []}
        for r in rows:
            w = wl[r["workload"]]
            pts[w["kind"]].append((_workload_features(w), r))
        coefs, res = {}, {}
        for kind, lst in pts.items():
            cols = _KIND_COLS[kind]
            X = np.array([f[cols] for f, _ in lst])
            for c in configs:
                y = np.array([float(r[f"c:{c}"]) for _, r in lst])
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                full = np.zeros(6)
                full[cols] = beta
                coefs[(kind, c)] = full
                res[(kind, c)] = (X @ beta - y) / y
        return cls(coefs, configs, res)

    def cycles(self, it: IterSummary, config: str) -> float:
        if config not in self.configs:
            raise KeyError(f"configuration {config!r} not in the cost table")
        v = float(self.coefs[(it.kind, config)] @ it.features())
        if not v > 0:
            raise ValueError(f"non-positive interpolated cost {v} for iteration {it.idx} ({it.kind})")
        return v

    def fit_report(self) -> dict:
        return {f"{k}|{c}": {"n": len(r), "max_abs_rel": float(np.max(np.abs(r))),
                             "median_abs_rel": float(np.median(np.abs(r)))} for (k, c), r in self.residuals.items()}

    def ratio_error_report(self) -> dict:
        """Max |predicted T(c)/T(base) − measured T(c)/T(base)| relative, over the fit points, per configuration:
        the resolution limit of policy comparisons made with this table (§15.2)."""
        out = {}
        for c in self.configs:
            if c == "base" or not self.residuals:
                continue
            out[c] = max(float(np.max(np.abs((1 + self.residuals[(k, c)]) / (1 + self.residuals[(k, "base")]) - 1)))
                         for k in _KIND_COLS)
        return out


def _workload_features(w: dict) -> np.ndarray:
    kind = w["kind"]
    if kind == "decode":
        B, T = int(w["B"]), int(w["T"])
        sum_t = int(w["sumT_exact"]) if w.get("sumT_exact") else int(w["sumT"])
        return IterSummary(0, (), tuple([T] * (B - 1) + [sum_t - T * (B - 1)])).features()
    if kind == "prefill":
        B, S, s0 = int(w["B"]), int(w["S"]), int(w["s0"] or 0)
        return IterSummary(0, tuple((i, s0, S) for i in range(B)), ()).features()
    ctxs = tuple(int(x) for x in w["ctxs"].split(";"))
    return IterSummary(0, ((0, int(w["s0"] or 0), int(w["c"])),), ctxs).features()


class LinearCost:
    """Toy/explicit cost model: cycles = coef[kind, config] · x (used by the hand-computed tests)."""

    def __init__(self, coefs: Mapping[tuple[str, str], Sequence[float]], configs: Sequence[str]):
        self.coefs = {k: np.asarray(v, float) for k, v in coefs.items()}
        self.configs = tuple(configs)

    cycles = CostTable.cycles


# ------------------------------------------------------------------------------------------------ policies
class StaticPolicy:
    def __init__(self, config: str = "base"):
        self.config = config
        self.name = f"static:{config}"

    def cost(self, it: IterSummary, table, prev: str | None) -> tuple[float, str]:
        return table.cycles(it, self.config), self.config


class HintPolicy:
    def __init__(self, mapping: Mapping[str, str] = DEFAULT_HINT_MAP, csw_cycles: float = 0.0, name="phaseshift-hint"):
        missing = set(REGIONS) - set(mapping)
        if missing:
            raise ValueError(f"hint mapping lacks regions {sorted(missing)}")
        self.mapping, self.csw, self.name = dict(mapping), float(csw_cycles), name

    def cost(self, it, table, prev):
        c = self.mapping[it.region]
        sw = self.csw if (prev is not None and prev != c) else 0.0
        return table.cycles(it, c) + sw, c


class OraclePolicy:
    """External DP-oracle schedule (stub interface outputs). Entry lookup: `by_index[it.idx]` if given and
    in range, else `by_region[it.region]`, else error. An entry is a configuration name (cost = that column) or
    {"ratio": r} (cost = r · base cost; the oracle's per-iteration cost incl. its own C_sw relative to base)."""

    def __init__(self, by_index: Sequence | None = None, by_region: Mapping | None = None, name="dp-oracle"):
        if by_index is None and by_region is None:
            raise ValueError("oracle schedule is empty")
        self.by_index, self.by_region, self.name = list(by_index or []), dict(by_region or {}), name

    def cost(self, it, table, prev):
        if it.idx < len(self.by_index):
            e = self.by_index[it.idx]
        elif it.region in self.by_region:
            e = self.by_region[it.region]
        else:
            raise KeyError(f"oracle schedule has no entry for iteration {it.idx} (region {it.region})")
        if isinstance(e, str):
            return table.cycles(it, e), e
        r = float(e["ratio"])
        if not r > 0:
            raise ValueError(f"oracle ratio must be > 0, got {r}")
        return r * table.cycles(it, "base"), f"oracle:{r:g}"


def load_oracle_schedule(path: str | Path) -> OraclePolicy:
    """JSON `{"by_index": [...], "by_region": {...}}` (either key optional) → OraclePolicy."""
    d = json.loads(Path(path).read_text())
    return OraclePolicy(d.get("by_index"), d.get("by_region"), name=d.get("name", "dp-oracle"))


# ------------------------------------------------------------------------------------------------ engine
@dataclasses.dataclass(frozen=True)
class EngineConfig:
    max_batch: int = 16                 # B_max (decode grid reaches B = 16)
    chunk_size: int | None = 64         # chunked prefill budget c (mixed grid c ≤ 64); None = full prefill
    token_budget: int = 8192            # full mode only
    clock_hz: float = CLOCK_HZ


@dataclasses.dataclass
class ReqResult:
    rid: int
    arrival: float
    input_len: int
    output_len: int
    token_times: list = dataclasses.field(default_factory=list)   # emission time of output tokens 1..O

    @property
    def ttft(self) -> float:
        return self.token_times[0] - self.arrival

    @property
    def tbts(self) -> np.ndarray:
        return np.diff(np.asarray(self.token_times, float))

    @property
    def tpot(self) -> float:
        """(t_O − t_1)/(O − 1); NaN for O = 1."""
        o = len(self.token_times)
        return (self.token_times[-1] - self.token_times[0]) / (o - 1) if o > 1 else float("nan")

    @property
    def finish(self) -> float:
        return self.token_times[-1]


@dataclasses.dataclass
class SimResult:
    requests: dict[int, ReqResult]
    iterations: list[tuple[float, float, str, str]]      # (t_start, t_end, region, config)
    n_prefill_tokens: int
    n_decode_tokens: int


class _S:
    __slots__ = ("req", "prefilled", "generated")

    def __init__(self, req):
        self.req, self.prefilled, self.generated = req, 0, 0


def simulate(requests: Sequence[Request], policy, table, cfg: EngineConfig = EngineConfig()) -> SimResult:
    reqs = sorted(requests, key=lambda r: (r.arrival, r.rid))
    if len({r.rid for r in reqs}) != len(reqs):
        raise ValueError("duplicate request ids")
    for r in reqs:
        if r.input_len < 1 or r.output_len < 1:
            raise ValueError(f"request {r.rid}: lengths must be >= 1")
    out = {r.rid: ReqResult(r.rid, r.arrival, r.input_len, r.output_len) for r in reqs}
    heap: list = []
    seq = 0

    def push(t, kind, payload=None):
        nonlocal seq
        heapq.heappush(heap, (t, kind, seq, payload))
        seq += 1

    for r in reqs:
        push(r.arrival, _ARRIVAL, r)
    waiting: list[_S] = []        # FCFS
    running: list[_S] = []        # admission order
    busy = False
    in_flight = None
    prev_cfg = None
    iters = []
    n_pf_tok = n_dec_tok = 0
    n_it = 0

    def form(t):
        """Form the next iteration at time t (semantics, unlimited KV); returns None if there is no work."""
        prefill, decode = [], []
        if cfg.chunk_size is None:
            budget = cfg.token_budget
            while waiting and len(running) < cfg.max_batch:
                s = waiting[0]
                n = s.req.input_len
                if prefill and n > budget:
                    break
                waiting.pop(0)
                running.append(s)
                prefill.append((s, 0, n))
                budget -= n
            if not prefill:
                decode = [s for s in running]
        else:
            budget = cfg.chunk_size
            for s in running:
                if budget <= 0:
                    break
                if s.prefilled < s.req.input_len:
                    n = min(s.req.input_len - s.prefilled, budget)
                    prefill.append((s, s.prefilled, n))
                else:
                    n = 1
                    decode.append(s)
                budget -= n
            while budget > 0 and waiting and len(running) < cfg.max_batch:
                s = waiting.pop(0)
                running.append(s)
                n = min(s.req.input_len, budget)
                prefill.append((s, 0, n))
                budget -= n
        if not prefill and not decode:
            return None
        return prefill, decode

    while heap:
        t, kind, _, payload = heapq.heappop(heap)
        if kind == _ARRIVAL:
            waiting.append(_S(payload))
            if not busy:
                push(t, _SCHEDULE)
        elif kind == _ITER_END:
            prefill, decode = payload
            for s, st, n in prefill:
                s.prefilled += n
                n_pf_tok += n
                if s.prefilled == s.req.input_len:       # prompt complete: emits the first output token
                    s.generated += 1
                    out[s.req.rid].token_times.append(t)
            for s in decode:
                s.generated += 1
                n_dec_tok += 1
                out[s.req.rid].token_times.append(t)
            running[:] = [s for s in running if s.generated < s.req.output_len]
            busy = False
            push(t, _SCHEDULE)
        else:  # _SCHEDULE
            if busy:
                continue
            f = form(t)
            if f is None:
                continue
            prefill, decode = f
            it = IterSummary(n_it, tuple((s.req.rid, st, n) for s, st, n in prefill),
                             tuple(s.prefilled + s.generated for s in decode))
            # decode context T = stored + 1 = (S + generated − 1) + 1 = S + generated
            cyc, used = policy.cost(it, table, prev_cfg)
            if not (cyc >= 0 and math.isfinite(cyc)):
                raise ValueError(f"bad cost {cyc} at iteration {n_it}")
            prev_cfg = used
            dur = cyc / cfg.clock_hz
            iters.append((t, t + dur, it.region, used))
            n_it += 1
            busy = True
            push(t + dur, _ITER_END, (prefill, decode))
    for r in out.values():
        assert len(r.token_times) == r.output_len, (r.rid, len(r.token_times), r.output_len)
    return SimResult(out, iters, n_pf_tok, n_dec_tok)


# ------------------------------------------------------------------------------------------------ workload
def poisson_workload(pool: Sequence[Request], rate: float, n: int, seed: int) -> list[Request]:
    """n requests: Poisson arrivals at `rate` req/s (exponential gaps, first arrival at the first gap) and lengths
    drawn i.i.d. with replacement from `pool`; one numpy Generator per seed (§15.4)."""
    if not rate > 0:
        raise ValueError("rate must be > 0")
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate, size=n)
    idx = rng.integers(0, len(pool), size=n)
    t = np.cumsum(gaps)
    return [Request(i, float(t[i]), pool[j].input_len, pool[j].output_len) for i, j in enumerate(idx)]


# ------------------------------------------------------------------------------------------------ metrics
PCTS = (50, 95, 99)


def run_metrics(res: SimResult, slos: Mapping[str, tuple[float, float]]) -> dict:
    """Per-run metrics (§15.3). slos: level → (TTFT SLO s, TPOT SLO s)."""
    rr = list(res.requests.values())
    ttft = np.array([r.ttft for r in rr])
    tpot = np.array([r.tpot for r in rr])
    tbt = np.concatenate([r.tbts for r in rr]) if rr else np.array([])
    m = {"n_requests": len(rr)}
    for name, arr in (("ttft", ttft), ("tpot", tpot[~np.isnan(tpot)]), ("tbt", tbt)):
        for p in PCTS:
            m[f"{name}_p{p}"] = float(np.percentile(arr, p, method="linear")) if len(arr) else float("nan")
    for lvl, (s_ttft, s_tpot) in slos.items():
        ok = (ttft <= s_ttft) & (np.isnan(tpot) | (tpot <= s_tpot))
        m[f"attain_{lvl}"] = float(ok.mean())
    return m


def goodput_rate(rates: Sequence[float], attain: Sequence[float], target: float = 0.9) -> float:
    """DistServe goodput [cand:A4]: the highest request rate whose SLO attainment is ≥ target. On a swept grid:
    the first rate r_k where attainment drops below target ends the feasible range; goodput is the linear
    interpolation of the target crossing between r_{k−1} and r_k; 0 if r_0 already fails; the largest rate if
    none fails (a lower bound, flagged by the caller)."""
    rates, attain = np.asarray(rates, float), np.asarray(attain, float)
    order = np.argsort(rates)
    rates, attain = rates[order], attain[order]
    for k in range(len(rates)):
        if attain[k] < target:
            if k == 0:
                return 0.0
            a0, a1 = attain[k - 1], attain[k]
            return float(rates[k - 1] + (a0 - target) / (a0 - a1) * (rates[k] - rates[k - 1]))
    return float(rates[-1])


def bootstrap_ci(per_seed: np.ndarray, stat=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Seed-level percentile bootstrap (§15.5): resample the per-seed values (axis 0) with replacement; returns
    (point estimate on all seeds, lo, hi). `stat` maps an array of shape (k, ...) to a scalar."""
    x = np.asarray(per_seed, float)
    k = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boots = np.array([stat(x[i]) for i in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(x)), float(lo), float(hi)


def sweep(pool, policies, table, rates, seeds, n_requests, slos, cfg=EngineConfig(), target=0.9):
    """Full grid policies × rates × seeds; returns (per-run rows, summary rows with bootstrap CIs, goodput rows)."""
    if len(seeds) < 5:
        raise ValueError("requires >= 5 arrival seeds")
    rows = []
    for pol in policies:
        for rate in rates:
            for sd in seeds:
                res = simulate(poisson_workload(pool, rate, n_requests, sd), pol, table, cfg)
                rows.append({"policy": pol.name, "rate": rate, "seed": sd, **run_metrics(res, slos)})
    keys = [k for k in rows[0] if k not in ("policy", "rate", "seed", "n_requests")]
    summ, good = [], []
    for pol in policies:
        att = {lvl: np.zeros((len(seeds), len(rates))) for lvl in slos}
        for j, rate in enumerate(rates):
            sub = [r for r in rows if r["policy"] == pol.name and r["rate"] == rate]
            for k in keys:
                est, lo, hi = bootstrap_ci(np.array([r[k] for r in sub]))
                summ.append({"policy": pol.name, "rate": rate, "metric": k, "mean": est, "ci_lo": lo, "ci_hi": hi,
                             "n_seeds": len(sub)})
            for lvl in slos:
                att[lvl][:, j] = [r[f"attain_{lvl}"] for r in sub]
        for lvl in slos:
            est, lo, hi = bootstrap_ci(att[lvl], stat=lambda a: goodput_rate(rates, a.mean(axis=0), target))
            good.append({"policy": pol.name, "slo": lvl, "goodput_rps": est, "ci_lo": lo, "ci_hi": hi,
                         "censored": bool(att[lvl].mean(axis=0).min() >= target)})
    return rows, summ, good
