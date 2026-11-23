"""PhaseShift analytical first-look oracle sweep.

Costs serving workloads with the uncalibrated analytical model (`model/memhier.py`, methodology §9) over the
configuration grid (A1 LMEM usage, C DXA depth, D decode-KV insertion, B two-class on/off), then computes, per
workload and per region, the gap of the zero-cost and DP oracles over the per-workload and the per-kernel best static
(methodology §3, §5) on vortex-S1-8c and h100-like, at the default model parameters and at every §9.6 sensitivity
variation. Choices this step makes are explained in methodology §9.7 and in `results/<run>/README.md`; each is also
commented where it is implemented below.

**Intervals = kernel launches** (methodology §5 asks for work-unit intervals inside launches). The analytical model
has no structure below a launch: a launch's memory time is one aggregate quantity (M3), so every sub-launch work unit
of a launch prefers the same configuration and an intra-launch switch can never pay. The finest interval the model can
express is therefore one launch, every interval is its own launch (K(i) = i), C_sw,i = 0 everywhere, and
`T_zero = T_DP = T_kernel` holds exactly, whatever C_sw is (property "the DP is invariant to C_sw values when every
interval is its own launch"). The DP-oracle gap over the per-kernel best static is 0 by construction in this model; it
cannot inform G2b. The gap over the per-workload best static is the G1 quantity.

**Compression (exact).** Identical launches (same iteration, same layer class, same op) have identical cost
rows, and with K(i) = i every baseline and oracle is a sum over intervals of either T(i, c) for one c or min_c T(i, c),
so a row holding count × the launch's cost is exact for all of them. Layer multiplicity (memhier M13) is folded in the
same way.

**Stratified sampling of exact iterations.** Trace schedules have 10³–10⁵ iterations and each costs 1–3 ms per
configuration (78 configurations), so each trace workload is a stratified random sample (strata = the four regions)
of at most `sample_per_stratum` exact iterations (memhier.iteration_seqs), each weighted N_h/n_h (Horvitz-Thompson:
every sum over iterations is estimated without bias). A percentile bootstrap within strata gives each trace
workload's sampling CI of its gap. Bucketing iterations (rounding lengths and offsets to a geometric grid) was tried
first and rejected: near the L2 cliff (h100-like mixed iterations with long prefixes) ±4 % offsets changed costs by up
to 79 % and the gap of a validation sample from 1.5 % to 2.6 %. Buckets are still used for the scheduler's
*reference* cost (iteration timing only), never for an evaluated cost.

**Mandatory baseline rules (methodology §9.6).** (1) Stock (`Knobs()`) is never a baseline: gaps are only against the
per-workload and per-kernel best static. (2) Near-tie breaking: in every argmin over configurations (global best static,
per-workload best static, per-launch choice, phase-tuned statics, oracle per-interval choice) configurations within
`tie_tol` = 0.5 % of the minimum are tied, and ties go to the global best static, never toward stock (`pick`). The gate is
evaluated with the rule ("tie-rule" columns) and, for transparency, with plain argmins ("exact" columns).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from model import opgen, oracle
from model.memhier import GPU_L2_BW_ANCHOR, Hier, Knobs, MemHier, hier_for, iteration_seqs, knob_grid
from serving import scheduler, traces

PS_ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("prefill", "mixed", "decode-weight", "decode-kv")
PHASE_OF_REGION = {"prefill": "prefill", "mixed": "mixed", "decode-weight": "decode", "decode-kv": "decode"}


# ================================================================================================= tie rule


def pick(cost: np.ndarray, prefer: int | None, stock: int | None, tol: float) -> int:
    """argmin over configurations with the methodology §9.6 near-tie rule: configurations whose cost is within
    `tol` (relative) of the minimum are tied; a tie goes to `prefer` (the global best static) if it is tied, else the
    lowest-cost tied configuration that is not `stock` (never toward stock), else the plain argmin."""
    cost = np.asarray(cost, dtype=float)
    c0 = int(np.argmin(cost))
    lim = cost[c0] * (1 + tol)
    if prefer is not None and cost[prefer] <= lim:
        return int(prefer)
    if stock is not None and c0 == stock:
        others = [(cost[c], c) for c in np.flatnonzero(cost <= lim) if c != stock]
        if others:
            return int(min(others)[1])
    return c0


def pick_rows(T: np.ndarray, prefer: int | None, stock: int | None, tol: float) -> np.ndarray:
    """Vectorised `pick` for every row of T (n × |C|)."""
    T = np.asarray(T, dtype=float)
    if T.size == 0:
        return np.zeros(0, dtype=np.int64)
    c0 = np.argmin(T, axis=1)
    mn = T[np.arange(len(T)), c0]
    lim = mn * (1 + tol)
    out = c0.copy()
    if stock is not None:
        is_stock = c0 == stock
        if np.any(is_stock):
            alt = np.where(T <= lim[:, None], T, np.inf)
            alt[:, stock] = np.inf
            a = np.argmin(alt, axis=1)
            ok = is_stock & np.isfinite(alt[np.arange(len(T)), a])
            out[ok] = a[ok]
    if prefer is not None:
        out[T[:, prefer] <= lim] = prefer
    return out


def global_static(tots: Sequence[np.ndarray], stock: int | None, tol: float) -> tuple[int, np.ndarray]:
    """c_global = argmin_c geomean_w [tot_w(c) / min tot_w] (methodology §3), with the tie rule toward non-stock
    configurations (there is no global best static yet to prefer). Returns (c, score vector)."""
    R = np.vstack([t / t.min() for t in tots])
    score = np.exp(np.mean(np.log(R), axis=0))
    return pick(score, None, stock, tol), score


# ================================================================================================= bucketing


def qv(x: float, steps: int, exact_below: int) -> int:
    """Bucket value: x itself (rounded) if ≤ exact_below, else round(2^(round(steps·log2 x)/steps))."""
    if x <= exact_below:
        return max(1, int(round(x)))
    return int(round(2.0 ** (round(steps * math.log2(x)) / steps)))


def bucket_key(prefill: Sequence[tuple[int, int, bool]], contexts: Sequence[int], steps: int, exact_below: int,
               max_seq: int = 0) -> tuple:
    """opgen sequence groups of a bucketed iteration. prefill: (n_tokens, start, last) per chunk; contexts: decode
    contexts T_i. Prefill chunks: n and the offset s0 are rounded (s0 = 0 stays 0), identical chunks are grouped.
    Decodes (methodology §9 M14 collapse): B decodes with Σ T become one group of B_q sequences at the rounded mean
    context. Rounded values are clamped so that offset + chunk and the decode context stay ≤ max_seq."""
    cap = max_seq or 1 << 30
    groups: Counter = Counter()
    for n, s0, last in prefill:
        nq = min(qv(n, steps, exact_below), cap)
        sq = 0 if s0 == 0 else qv(s0, steps, exact_below)
        sq = max(0, min(sq, cap - nq))
        groups[(nq, sq, bool(last))] += 1
    seqs = [opgen.Seq("prefill", n, s, c, e) for (n, s, e), c in sorted(groups.items())]
    if len(contexts):
        B = len(contexts)
        Tq = min(qv(sum(contexts) / B, steps, exact_below), cap)
        seqs.append(opgen.Seq("decode", 1, Tq - 1, qv(B, steps, exact_below)))
    return tuple(seqs)


def region_of(kind: str, sum_context: float, kv_per_token: float, weight_bytes: float) -> str:
    """Methodology §4 region of one (exact, un-bucketed) iteration; same rule as scheduler.iteration_region."""
    if kind != "decode":
        return kind
    return "decode-kv" if kv_per_token * sum_context / weight_bytes >= 1 else "decode-weight"


# ================================================================================================= workloads


@dataclass
class Workload:
    """A workload run on one machine: rows = [(iteration key, region, weight)], one per (sampled) iteration. The key
    is a tuple of opgen Seq groups; the weight is the number of iterations the row stands for (1 for a synthetic
    point; N_h/n_h for a stratified sample of a trace schedule)."""

    name: str
    family: str                       # synthetic-prefill | synthetic-decode | synthetic-mixed | trace
    rows: list                        # [(key, region, weight)]
    meta: dict = field(default_factory=dict)


@dataclass
class Pair:
    """One (machine, model) setup of the sweep."""

    machine: str
    m: opgen.ModelShape
    q: float
    sigma: float
    max_context: int
    b_max: int
    token_budget: int
    kv_blocks: int
    syn: dict
    gen_kw: dict

    @property
    def kv_per_token(self) -> float:
        return opgen.kv_bytes_per_token(self.m, self.q)

    @property
    def weight_bytes(self) -> float:
        """q_w·N_params: the weight bytes a dense model streams per step (methodology §4; no MoE model here)."""
        return self.q * opgen.n_params(self.m)

    def memhier(self, hier: Hier | None = None) -> MemHier:
        """MemHier for this pair. memo=0: memhier's LRU memo keeps whole IterResults (with per-launch records);
        65,536 of them can reach GBs, so this step keeps its own compact caches instead (WSL OOM, 2026-10-31)."""
        return MemHier(self.m, self.machine, hier=hier, q_w=self.q, q_a=self.q, q=self.q, memo=0, **self.gen_kw)


def load_model(spec: dict) -> opgen.ModelShape:
    if "llama2c" in spec:
        return opgen.from_llama2c_bin(PS_ROOT / spec["llama2c"])
    return opgen.from_hf_config(PS_ROOT / spec["hf"])


def build_pairs(cfg: dict) -> list[Pair]:
    """Pairs from the yaml; σ = auto uses traces.sigma_for_kv_l2 against the reference pair (methodology §10.4)."""
    raw = {p["machine"]: p for p in cfg["pairs"]}
    models = {k: load_model(p["model"]) for k, p in raw.items()}
    ref = cfg["sigma_reference"]
    l2 = {k: hier_for(k).l2_bytes for k in raw}
    kvpt = {k: opgen.kv_bytes_per_token(models[k], raw[k]["q"]) for k in raw}
    out = []
    for k, p in raw.items():
        m = models[k]
        sigma = (traces.sigma_for_kv_l2(kvpt[ref], l2[ref], kvpt[k], l2[k]) if p["sigma"] == "auto"
                 else float(p["sigma"]))
        maxc = m.max_seq if p["max_context"] == "model" else int(p["max_context"])
        b = 16
        if p.get("kv_capacity") == "never_binding":
            kv_blocks = math.ceil(p["b_max"] * maxc / b)          # every running request at full context fits
        else:
            kv_blocks = scheduler.kv_blocks_for_bytes(float(p["kv_capacity_bytes"]), kvpt[k], b)
        out.append(Pair(machine=k, m=m, q=p["q"], sigma=sigma, max_context=maxc, b_max=p["b_max"],
                        token_budget=max(1, round(sigma * p["token_budget"])), kv_blocks=kv_blocks,
                        syn=p["synthetic"],
                        gen_kw=dict(routing=p.get("routing", "uniform"),
                                    mla_prefill_form=p.get("mla_prefill_form", "absorbed"))))
    return out


def scaled_chunk(pair: Pair, c: int) -> int:
    """Chunk (token budget) scaled like the lengths: max(1, round(σ·c)) ("chunk c (scaled)")."""
    return max(1, round(pair.sigma * c))


def _split_prompt_tokens(n: int, cap: int) -> list[int]:
    """n prefill tokens as whole prompts of at most cap tokens (a chunk budget larger than the context holds several
    prompts)."""
    out = [cap] * (n // cap)
    if n % cap:
        out.append(n % cap)
    return out


def synthetic_workloads(pair: Pair, steps: int, exact_below: int) -> list[Workload]:
    """Single-iteration workloads, exact (no bucketing; an iteration repeated is the same workload up to a factor).
    Decode points whose KV footprint exceeds the KV capacity are dropped."""
    out = []
    kvpt, wb = pair.kv_per_token, pair.weight_bytes
    cap = pair.max_context
    s = pair.syn
    for B in s["prefill"]["B"]:
        for S in s["prefill"]["S"]:
            if S > cap:
                continue
            key = tuple(opgen.prefill(B, S))
            out.append(Workload(f"prefill B={B} S={S}", "synthetic-prefill", [(key, "prefill", 1.0)],
                                dict(B=B, S=S)))
    for B in s["decode"]["B"]:
        for T in s["decode"]["T"]:
            if T > cap or math.ceil(T / 16) * B > pair.kv_blocks:
                continue
            key = tuple(opgen.decode(B, T))
            reg = region_of("decode", B * T, kvpt, wb)
            out.append(Workload(f"decode B={B} T={T}", "synthetic-decode", [(key, reg, 1.0)],
                                dict(B=B, T=T, rho_kv=kvpt * B * T / wb)))
    for c in s["mixed"]["chunk"]:
        cs = scaled_chunk(pair, c)
        for B in s["mixed"]["B"]:
            if B >= cs:
                continue
            for T in s["mixed"]["T"]:
                if T > cap:
                    continue
                pieces = Counter(_split_prompt_tokens(cs - B, cap))
                key = tuple([opgen.Seq("prefill", n, 0, c, True) for n, c in sorted(pieces.items())]
                            + opgen.decode(B, T))
                out.append(Workload(f"mixed c={c}(→{cs}) B={B} T={T}", "synthetic-mixed",
                                    [(key, "mixed", 1.0)], dict(chunk=c, chunk_scaled=cs, B=B, T=T)))
    return out


def _iteration_lists(cols: dict, i: int):
    a, b = cols["pf_off"][i], cols["pf_off"][i + 1]
    pf = list(zip(cols["pf_n_tokens"][a:b].tolist(), cols["pf_start"][a:b].tolist(), cols["pf_last"][a:b].tolist()))
    a, b = cols["dc_off"][i], cols["dc_off"][i + 1]
    return pf, cols["dc_context"][a:b].tolist()


def trace_workloads(pair: Pair, cfg: dict, ref_knobs: Knobs, steps: int, exact_below: int) -> list[Workload]:
    """Schedule each trace window once with the reference configuration (fixed-schedule assumption, methodology §10.3)
    and turn it into a stratified sample of exact iterations. Only the scheduler's reference cost (which sets
    iteration timing, hence batching) uses bucketed iterations; every evaluated cost is exact."""
    mh = pair.memhier()
    cap = pair.max_context
    memo: dict[tuple, float] = {}           # bucket key -> seconds under ref_knobs (compact cache)

    def cost(it, _hw=None):
        pf = [(p.n_tokens, p.start, p.last) for p in it.prefill]
        key = bucket_key(pf, [d.context for d in it.decode], steps, exact_below, cap)
        if key not in memo:
            memo[key] = mh.cost_seqs(key, ref_knobs).seconds
        return memo[key]

    out = []
    tcfg = cfg["traces"]
    n_per, seed = int(tcfg["sample_per_stratum"]), int(tcfg.get("seed", 0))
    for ti, name in enumerate(tcfg["names"]):
        lim = tcfg["limit"][name] if isinstance(tcfg["limit"], dict) else tcfg["limit"]
        raw = traces.load_trace(name, PS_ROOT, limit=lim)
        reqs = traces.scale_lengths(raw, pair.sigma, max_context=cap, policy="clip")
        n_clip = sum(1 for r0, r1 in zip(raw, reqs)
                     if (r1.input_len, r1.output_len) != (traces._scale_len(r0.input_len, pair.sigma),
                                                          traces._scale_len(r0.output_len, pair.sigma)))
        ref_ratio = traces.median_kv_l2_ratio(raw, pair.kv_per_token, hier_for(pair.machine).l2_bytes)
        ach_ratio = traces.median_kv_l2_ratio(reqs, pair.kv_per_token, hier_for(pair.machine).l2_bytes)
        for mi, mode in enumerate(tcfg["modes"]):
            if mode == "full":
                sc = scheduler.SchedulerConfig(max_batch=pair.b_max, token_budget=pair.token_budget, chunk_size=None,
                                               block_size=16, kv_capacity_blocks=pair.kv_blocks)
                cs = None
            else:
                cs = scaled_chunk(pair, int(mode))
                # B_max ≤ c in chunked mode (methodology §10.2): otherwise decodes would be skipped
                sc = scheduler.SchedulerConfig(max_batch=min(pair.b_max, cs), token_budget=pair.token_budget,
                                               chunk_size=cs, block_size=16, kv_capacity_blocks=pair.kv_blocks)
            t0 = time.time()
            res = scheduler.run_schedule(reqs, sc, cost)
            cols = res.cols
            per = res.per_iteration()
            kind = per["kind"]                      # 0 decode, 1 prefill, 2 mixed (methodology §4)
            kvb = pair.kv_per_token * per["sum_context"] / pair.weight_bytes
            reg = np.where(kind == 1, "prefill", np.where(kind == 2, "mixed",
                                                          np.where(kvb >= 1, "decode-kv", "decode-weight")))
            # Stratified sample (strata = regions) of EXACT iterations, weight N_h/n_h (Horvitz-Thompson):
            # bucketing distorted costs near the L2 cliff (h100-like mixed iterations, results/<run>/README.md)
            rng = np.random.default_rng([seed, ti, mi])
            rows = []
            its = res.iterations
            n_by, s_by = {}, {}
            for g in REGIONS:
                idx = np.flatnonzero(reg == g)
                n_by[g] = int(len(idx))
                if not len(idx):
                    continue
                take = idx if len(idx) <= n_per else np.sort(rng.choice(idx, n_per, replace=False))
                s_by[g] = int(len(take))
                wgt = len(idx) / len(take)
                for i in take:
                    rows.append((iteration_seqs(its[int(i)], pair.m), g, wgt))
            preempt = sum(r.preemptions for r in res.requests.values())
            w = Workload(f"{name} {'full' if mode == 'full' else f'c={mode}(→{cs})'}", "trace", rows,
                         dict(trace=name, mode=mode, chunk_scaled=cs, n_requests=len(reqs), sigma=pair.sigma,
                              clipped=n_clip, clipped_frac=n_clip / len(reqs), median_kv_l2_unscaled=ref_ratio,
                              median_kv_l2_scaled=ach_ratio, b_max=sc.max_batch, n_iterations=res.n_iterations,
                              iterations_by_region=n_by, sampled_by_region=s_by,
                              preemptions=preempt, makespan_s=res.makespan, sched_cpu_s=time.time() - t0,
                              kv_tokens_median=float(np.median(cols["kv_tokens"]))))
            out.append(w)
    return out


# ================================================================================================= costing


RSS_LIMIT_MB = 1300   # abort well below the 1.5 GB per-job ceiling (WSL OOM rule, 2026-10-31)


def rss_mb() -> float:
    """Current resident set size in MB (Linux /proc)."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**20


def rss_guard():
    r = rss_mb()
    if r > RSS_LIMIT_MB:
        raise MemoryError(f"RSS {r:.0f} MB above the {RSS_LIMIT_MB} MB guard")


class CostTable:
    """Per-launch costs of every bucket under every configuration: bucket key -> (launch labels, matrix
    [n_launch × |C|] of cycles × layer multiplicity). Computed with cost_seqs(..., detail=True) (memhier: per-launch
    costs bypass the memo)."""

    def __init__(self, mh: MemHier, grid: Sequence[Knobs]):
        self.mh, self.grid = mh, list(grid)
        self.tab: dict[tuple, tuple[list[str], np.ndarray]] = {}

    def get(self, key: tuple) -> tuple[list[str], np.ndarray]:
        if key not in self.tab:
            names, cols = None, []
            for k in self.grid:
                r = self.mh.cost_seqs(key, k, detail=True)
                nm = [f"{L.names}@{L.layer}" for L in r.launches]
                if names is None:
                    names = nm
                elif nm != names:
                    raise RuntimeError("launch structure depends on the knobs")
                v = np.array([L.cycles * L.like for L in r.launches], dtype=float)
                if not math.isclose(v.sum(), r.cycles, rel_tol=1e-9):
                    raise RuntimeError("per-launch costs do not add up to the iteration cost")
                cols.append(v)
            self.tab[key] = (names, np.column_stack(cols))
            if len(self.tab) % 250 == 0:
                rss_guard()
        return self.tab[key]


def workload_matrices(w: Workload, table: CostTable):
    """(T_launch [launch rows × C], launch op labels, launch-row regions, T_iter [iteration rows × C], iteration-row
    regions, iteration index of every launch row): weight-scaled rows; exact compression (module docstring)."""
    TL, ops, regs, TI, iregs, it_of = [], [], [], [], [], []
    for j, (key, reg, n) in enumerate(w.rows):
        names, M = table.get(key)
        TL.append(n * M)
        ops += [nm.split("@")[0] for nm in names]
        regs += [reg] * len(names)
        it_of += [j] * len(names)
        TI.append(n * M.sum(axis=0, keepdims=True))
        iregs.append(reg)
    return np.vstack(TL), np.array(ops), np.array(regs), np.vstack(TI), np.array(iregs), np.array(it_of)


def bootstrap_gap(TI: np.ndarray, lmin_it: np.ndarray, strata: np.ndarray, n_boot: int = 300, seed: int = 0):
    """Sampling-error CI of a sampled workload's DP gap over its per-workload static (plain argmins): resample the
    sampled iterations with replacement within each stratum (weights unchanged), recompute
    min_c Σ T_iter / Σ (launch-level row minima) − 1; percentile 95 % interval."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == g) for g in dict.fromkeys(strata.tolist())]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        out[b] = TI[idx].sum(axis=0).min() / lmin_it[idx].sum() - 1
    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(lo), float(hi)


# ================================================================================================= analysis


def _phase_mask(regs: np.ndarray, phase: str) -> np.ndarray:
    return np.array([PHASE_OF_REGION[r] == phase for r in regs], dtype=bool)


def analyze_machine(workloads: Sequence[Workload], table: CostTable, cfg_idx: Sequence[int], stock: int | None,
                    tol: float) -> tuple[list[dict], dict]:
    """Every baseline and oracle of methodology §3/§5 for the workloads of one machine, restricted to the
    configurations cfg_idx; both with the §9.6 tie rule and with plain argmins. Returns (per-workload rows, globals).

    Memory-light (WSL OOM rule, 2026-10-31): two passes, each building one workload's matrices at a time. Pass 1 keeps
    only |C|-vectors (totals, per-phase totals) to choose the global and phase-tuned statics; pass 2 rebuilds each
    workload's matrices for the per-row choices, which need the global best static (tie rule)."""
    cfg_idx = np.asarray(list(cfg_idx))
    stock_l = None if stock is None or stock not in cfg_idx else int(np.flatnonzero(cfg_idx == stock)[0])
    tots, ph = [], {"prefill": [], "decode": []}
    for w in workloads:                                          # pass 1
        TL, ops, regs, TI, iregs, _ = workload_matrices(w, table)
        TL = TL[:, cfg_idx]
        tots.append(TL.sum(axis=0))
        for phase in ph:
            msk = _phase_mask(regs, phase)
            if msk.any():
                ph[phase].append(TL[msk].sum(axis=0))
        del TL, TI
    c_glob, score = global_static(tots, stock_l, tol)
    c_glob_exact = int(np.argmin(score))

    def tuned(rs):
        if not rs:
            return None, None
        R = np.vstack([t / t.min() for t in rs])
        sc = np.exp(np.mean(np.log(R), axis=0))
        return pick(sc, c_glob, stock_l, tol), int(np.argmin(sc))

    (c_pf, c_pf_x), (c_dec, c_dec_x) = tuned(ph["prefill"]), tuned(ph["decode"])
    out = []
    for w, tot in zip(workloads, tots):                          # pass 2
        TL, ops, regs, TI, iregs, it_of = workload_matrices(w, table)
        TL, TI = TL[:, cfg_idx], TI[:, cfg_idx]
        r = dict(workload=w.name, family=w.family, n_rows=len(TL))
        # exact (plain argmin; methodology §3/§5 definitions)
        r["T_static_x"] = float(tot.min())
        r["c_static_x"] = int(np.argmin(tot))
        r["T_dp_x"] = r["T_zero_x"] = r["T_kernel_x"] = float(TL.min(axis=1).sum())   # K(i) = i (module docstring)
        r["T_iter_x"] = float(TI.min(axis=1).sum())
        byop = {o: TL[ops == o].sum(axis=0) for o in dict.fromkeys(ops.tolist())}
        r["T_optype_x"] = float(sum(v.min() for v in byop.values()))
        if w.family == "trace":
            lmin_it = np.bincount(it_of, weights=TL.min(axis=1), minlength=len(TI))
            r["gap_ci_lo_x"], r["gap_ci_hi_x"] = bootstrap_gap(TI, lmin_it, iregs)
        # tie rule (methodology §9.6 rule 2), every argmin
        cs = pick(tot, c_glob, stock_l, tol)
        r["c_static"] = int(cs)
        r["T_static"] = float(tot[cs])
        pl = pick_rows(TL, c_glob, stock_l, tol)
        r["T_dp"] = r["T_zero"] = r["T_kernel"] = float(TL[np.arange(len(TL)), pl].sum())
        pi = pick_rows(TI, c_glob, stock_l, tol)
        r["T_iter"] = float(TI[np.arange(len(TI)), pi].sum())
        r["T_optype"] = float(sum(v[pick(v, c_glob, stock_l, tol)] for v in byop.values()))
        r["T_global"] = float(tot[c_glob])
        r["T_prefill_tuned"] = None if c_pf is None else float(tot[c_pf])
        r["T_decode_tuned"] = None if c_dec is None else float(tot[c_dec])
        r["T_stock"] = None if stock_l is None else float(tot[stock_l])      # context only, never a baseline (§9.6)
        # region membership: majority of the time under the global best static (methodology §4; choice in §9.7)
        share = {g: float(TL[regs == g, c_glob].sum() / tot[c_glob]) for g in REGIONS}
        r.update({f"share_{g}": share[g] for g in REGIONS})
        r["region"] = max(REGIONS, key=lambda g: share[g])
        # region-restricted view (diagnostic): per region present, the sub-workload of that region's iterations
        for g in REGIONS:
            msk = regs == g
            if msk.any():
                tg = TL[msk].sum(axis=0)
                r[f"rr_{g}_static"] = float(tg[pick(tg, c_glob, stock_l, tol)])
                r[f"rr_{g}_dp"] = float(TL[msk][np.arange(int(msk.sum())), pl[msk]].sum())
        out.append(r)
        del TL, TI
    glob = dict(c_global=int(c_glob), c_global_exact=c_glob_exact, c_prefill_tuned=c_pf, c_decode_tuned=c_dec,
                c_prefill_tuned_exact=c_pf_x, c_decode_tuned_exact=c_dec_x, cfg_idx=cfg_idx.tolist())
    return out, glob


def _gm(x) -> float:
    return oracle.geomean(x) if len(x) else float("nan")


def region_rows(results: Sequence[dict], share_min: float = 0.01) -> list[dict]:
    """Per region (workload membership, methodology §4): geomean over its workloads of every speedup, as gaps
    (speedup − 1). Also the region-restricted diagnostic (geomean over workloads holding ≥ share_min of time there)."""
    out = []
    for g in REGIONS:
        rs = [r for r in results if r["region"] == g]
        d = dict(region=g, n=len(rs))
        for sfx in ("", "_x"):
            d[f"gap_dp_vs_static{sfx}"] = _gm([r[f"T_static{sfx}"] / r[f"T_dp{sfx}"] for r in rs]) - 1
            d[f"gap_zero_vs_static{sfx}"] = _gm([r[f"T_static{sfx}"] / r[f"T_zero{sfx}"] for r in rs]) - 1
            d[f"gap_dp_vs_kernel{sfx}"] = _gm([r[f"T_kernel{sfx}"] / r[f"T_dp{sfx}"] for r in rs]) - 1
            d[f"gap_zero_vs_kernel{sfx}"] = _gm([r[f"T_kernel{sfx}"] / r[f"T_zero{sfx}"] for r in rs]) - 1
            d[f"gap_iter_vs_static{sfx}"] = _gm([r[f"T_static{sfx}"] / r[f"T_iter{sfx}"] for r in rs]) - 1
            d[f"gap_optype_vs_static{sfx}"] = _gm([r[f"T_static{sfx}"] / r[f"T_optype{sfx}"] for r in rs]) - 1
        d["max_workload_gap_dp_vs_static"] = max((r["T_static"] / r["T_dp"] - 1 for r in rs), default=float("nan"))
        d["global_regret"] = _gm([r["T_global"] / r["T_static"] for r in rs]) - 1
        d["prefill_tuned_regret"] = _gm([r["T_prefill_tuned"] / r["T_static"] for r in rs
                                         if r["T_prefill_tuned"] is not None]) - 1
        d["decode_tuned_regret"] = _gm([r["T_decode_tuned"] / r["T_static"] for r in rs
                                        if r["T_decode_tuned"] is not None]) - 1
        rr = [r for r in results if f"rr_{g}_static" in r and r[f"share_{g}"] >= share_min]
        d["rr_n"] = len(rr)
        d["rr_gap_dp_vs_static"] = _gm([r[f"rr_{g}_static"] / r[f"rr_{g}_dp"] for r in rr]) - 1
        out.append(d)
    return out


def gate_g1(gaps: dict[str, list[dict]], threshold: float, key: str = "gap_dp_vs_static") -> dict:
    """STOP iff the DP-oracle gap is < threshold in EVERY (non-empty) region on EVERY machine."""
    below = {mach: {d["region"]: float(d[key]) < threshold for d in rows if int(d["n"]) > 0}
             for mach, rows in gaps.items()}
    stop = all(all(v.values()) for v in below.values())
    return dict(outcome="STOP" if stop else "CONTINUE", below=below,
                above=[(m, g) for m, v in below.items() for g, b in v.items() if not b])


# ================================================================================================= variants


def variant_hier(base: Hier, v: dict) -> Hier | None:
    """Hier for one §9.6 sensitivity entry (None if the entry does not apply to this machine)."""
    if "only" in v and base.name not in v["only"]:
        return None
    if v.get("unanchored"):
        return dataclasses.replace(base, l2_bytes_per_cycle=base.l2_bytes_per_cycle / GPU_L2_BW_ANCHOR)
    upd = {}
    for f, x in v.get("set", {}).items():
        upd[f] = x
    for f, x in v.get("add", {}).items():
        upd[f] = getattr(base, f) + x
    for f, x in v.get("scale", {}).items():
        y = getattr(base, f) * x
        upd[f] = int(y) if isinstance(getattr(base, f), int) else y
    return dataclasses.replace(base, **upd)


# ================================================================================================= driver


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.6g}"
    return x


def _load_cfg(cfg_path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(cfg_path).read_text())


def _cache_path(out: Path, machine: str) -> Path:
    return out / "logs" / "cache" / f"workloads_{machine}.json"


def _save_workloads(path: Path, ws: Sequence[Workload], meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = [dict(name=w.name, family=w.family, meta=w.meta,
                rows=[[[[q.kind, q.n_new, q.ctx, q.count, q.emits] for q in key], reg, n]
                      for key, reg, n in w.rows]) for w in ws]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(meta=meta, workloads=enc), default=str))
    tmp.replace(path)


def _load_workloads(path: Path) -> tuple[list[Workload], dict]:
    d = json.loads(path.read_text())
    ws = []
    for e in d["workloads"]:
        rows = [(tuple(opgen.Seq(*q) for q in key), reg, n) for key, reg, n in e["rows"]]
        ws.append(Workload(e["name"], e["family"], rows, e["meta"]))
    return ws, d["meta"]


def _append_csv(path: Path, rows: list[dict]):
    """Append rows (fixed column set per file; header written once)."""
    if not rows:
        return
    new = not path.exists()
    keys = list(rows[0]) if new else next(csv.reader(open(path)))
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows([{k: _fmt(r.get(k, "")) for k in keys} for r in rows])
        f.flush()


def build(cfg: dict, out: Path, machine: str, log=print) -> None:
    """Phase 1 (one machine): synthetic workloads, schedule reference config, trace schedules → bucket rows
    (cached on disk, idempotent), bucket validation, workloads.csv rows."""
    path = _cache_path(out, machine)
    if path.exists():
        log(f"[{machine}] workloads cached at {path}; skipping build")
        return
    steps, eb = cfg["bucket"]["steps_per_octave"], cfg["bucket"]["exact_below"]
    tol = float(cfg["tie_tol"])
    p = next(x for x in build_pairs(cfg) if x.machine == machine)
    t0 = time.time()
    h = hier_for(p.machine)
    grid = knob_grid(h, **{k: tuple(v) for k, v in cfg["grid"].items()})
    stock = grid.index(Knobs())
    syn = synthetic_workloads(p, steps, eb)
    res_syn, g_syn = analyze_machine(syn, CostTable(p.memhier(), grid), range(len(grid)), stock, tol)
    ref = grid[g_syn["c_global"]]      # schedule reference: global best static of the synthetic grid (§9.7)
    log(f"[{machine}] σ={p.sigma:.4f}; {len(syn)} synthetic workloads; schedule reference {ref.label}; "
        f"{time.time() - t0:.0f} s, RSS {rss_mb():.0f} MB")
    tr = []
    ws = trace_workloads(p, cfg, ref, steps, eb)
    if True:
        for w in ws:
            log(f"  {w.name}: {w.meta['n_iterations']} iterations, sampled {w.meta['sampled_by_region']}, "
                f"{w.meta['preemptions']} preemptions, clipped {w.meta['clipped_frac']:.3f}, "
                f"sched {w.meta['sched_cpu_s']:.0f} s, RSS {rss_mb():.0f} MB")
        tr += ws
        rss_guard()
    ws = syn + tr
    meta = dict(machine=machine, model=p.m.name, q=p.q, sigma=p.sigma, max_context=p.max_context, b_max=p.b_max,
                token_budget=p.token_budget, kv_blocks=p.kv_blocks, sched_ref=ref.label, n_configs=len(grid),
                stock_index=stock, n_rows=sum(len(w.rows) for w in ws), build_s=time.time() - t0)
    rows = [dict(machine=machine, workload=w.name, family=w.family,
                 **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in w.meta.items()}) for w in ws]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    _append_csv(out / "workloads.csv", [{k: r.get(k, "") for k in keys} for r in rows])
    _save_workloads(path, ws, meta)
    log(f"[{machine}] build done: {len(ws)} workloads, {meta['n_rows']} iteration rows, {time.time() - t0:.0f} s, "
        f"RSS {rss_mb():.0f} MB")


def sweep(cfg: dict, out: Path, machine: str, variants: Sequence[str] | None = None, log=print) -> None:
    """Phase 2 (one machine): for every variant not yet done, cost all buckets, analyze (all configs and B off),
    append rows.jsonl / gap_by_region.csv, write a done marker (idempotent, resumable)."""
    tol = float(cfg["tie_tol"])
    ws, meta = _load_workloads(_cache_path(out, machine))
    p = next(x for x in build_pairs(cfg) if x.machine == machine)
    vlist = [{"name": "default"}] + list(cfg["sensitivity"])
    if variants:
        vlist = [v for v in vlist if v["name"] in variants]
    done_dir = out / "logs" / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    base = hier_for(machine)
    for v in vlist:
        marker = done_dir / f"{v['name'].replace(' ', '_').replace('/', '_')}__{machine}"
        if marker.exists():
            continue
        hh = base if v["name"] == "default" else variant_hier(base, v)
        if hh is None:
            continue
        t0 = time.time()
        grid = knob_grid(hh, **{k: tuple(x) for k, x in cfg["grid"].items()})
        stock = grid.index(Knobs())
        tab = CostTable(p.memhier(hh), grid)
        jrows, grows = [], []
        for space, idx in (("all", list(range(len(grid)))),
                           ("B-off", [i for i, k in enumerate(grid) if not k.b_two_class])):
            res, glob = analyze_machine(ws, tab, idx, stock, tol)
            lab = lambda c: "" if c is None else grid[glob["cfg_idx"][c]].label
            for r in res:
                jrows.append(dict(variant=v["name"], machine=machine, space=space, **r,
                                  c_static_label=lab(r["c_static"]), c_global_label=lab(glob["c_global"])))
            for d in region_rows(res):
                grows.append(dict(variant=v["name"], machine=machine, space=space, **d,
                                  c_global=lab(glob["c_global"]), c_global_exact=lab(glob["c_global_exact"]),
                                  c_prefill_tuned=lab(glob["c_prefill_tuned"]),
                                  c_decode_tuned=lab(glob["c_decode_tuned"])))
            if space == "all":
                log(f"[{v['name']}] {machine}: " + "; ".join(
                    f"{d['region']} n={d['n']} DP gap {100 * d['gap_dp_vs_static']:.2f}%" for d in grows))
        with open(out / "rows.jsonl", "a") as f:
            for r in jrows:
                f.write(json.dumps(r) + "\n")
            f.flush()
        _append_csv(out / "gap_by_region.csv", grows)
        marker.write_text(f"{time.time() - t0:.1f} s, {len(tab.tab)} buckets, RSS {rss_mb():.0f} MB\n")
        log(f"[{v['name']}] {machine}: {len(tab.tab)} buckets in {time.time() - t0:.0f} s, RSS {rss_mb():.0f} MB")
        del tab


def summarize(cfg: dict, out: Path, log=print) -> dict:
    """Phase 3: per variant from gap_by_region.csv (tie rule, exact argmins, B off) → gate_g1.json."""
    thr = float(cfg["gate"]["threshold"])
    rows = list(csv.DictReader(open(out / "gap_by_region.csv")))
    gates = {}
    machines = cfg["gate"]["machines_required"]
    for v in dict.fromkeys(r["variant"] for r in rows):
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["variant"] == v:
                by[r["space"]][r["machine"]].append(r)
        # A variation that does not apply to a machine (e.g. the GPU-only unanchored L2 bandwidth) leaves that
        # machine at its default parameters: G1 needs both machines, so the default rows stand in for it.
        fallback = [m for m in machines if m not in by["all"]]
        for m in fallback:
            for r in rows:
                if r["variant"] == "default" and r["machine"] == m:
                    by[r["space"]][m].append(r)
        gates[v] = dict(machines=sorted(by["all"]), default_rows_for=fallback, tie_rule=gate_g1(by["all"], thr),
                        exact=gate_g1(by["all"], thr, "gap_dp_vs_static_x"), b_off=gate_g1(by["B-off"], thr))
        log(f"[{v}] G1 → {gates[v]['tie_rule']['outcome']} (exact argmins: {gates[v]['exact']['outcome']}, "
            f"B off: {gates[v]['b_off']['outcome']}); machines {gates[v]['machines']}")
    pairs = {}
    for m in cfg["gate"]["machines_required"]:
        pth = _cache_path(out, m)
        if pth.exists():
            pairs[m] = json.loads(pth.read_text())["meta"]
    s = dict(threshold=thr, tie_tol=float(cfg["tie_tol"]), pairs=pairs, gates=gates)
    (out / "gate_g1.json").write_text(json.dumps(s, indent=1, default=str))
    return s


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="analytical oracle sweep (uncalibrated model)")
    ap.add_argument("phase", choices=("build", "sweep", "summarize"))
    ap.add_argument("--config", default=str(PS_ROOT / "experiments" / "P1.6.yaml"))
    ap.add_argument("--machine", default=None)
    ap.add_argument("--variants", nargs="*", default=None, help="only these variant names (default: all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = _load_cfg(Path(a.config))
    out = Path(a.out or PS_ROOT / cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    if a.phase == "build":
        build(cfg, out, a.machine, log)
    elif a.phase == "sweep":
        sweep(cfg, out, a.machine, a.variants, log)
    else:
        summarize(cfg, out, log)
    log(f"peak-ish RSS now {rss_mb():.0f} MB")


if __name__ == "__main__":
    sys.exit(main())
