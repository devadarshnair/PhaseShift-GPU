"""— calibrate the analytical memory-hierarchy model (`model/memhier.py`) against SimX (specification and every modeling choice: `docs/methodology.md` §9.8).

Data: `results/<run>/table_v0.csv` (PS-BASE-S1-4c, one cold-cache SimX launch per row, so the model runs with
`persist_across_launch=False` and every row is one memhier launch, `MemHier.cost_launch`). Rows used: every `how = run`
row (base + the stock-knob settings that change the binary or its arguments) of the five op families; excluded and why:
`same_as_base` rows (their value IS the base row's: counting them would weight a cell by how many knobs leave it
untouched) and the `naive` decode variant (a kernel that re-reads K/V once per query head; opgen's GQA op has no such
re-read structure, so the model cannot represent it).

Fit: global parameters θ (PARAMS: compute efficiencies per unit, DRAM efficiency, overlap, latency scales, effective
MLP per warp, launch overhead) plus a discrete hit-model choice, by least squares on r_i = ln(pred_i / meas_i) over a
70 % TRAIN split of grid CELLS (all settings of a cell stay on one side; stratified per op family; numpy seed
SPLIT_SEED), multi-start `scipy.optimize.least_squares` (trf, bounds, deterministic starts). MAPE is reported on the
30 % TEST cells. If MAPE > 25 % the model is "qualitative only" (methodology §9.8).

Memory: memhier is built with memo=0 (its default 65,536-entry memo is a known hazard) and an RSS guard aborts above
RSS_LIMIT_MB. Pure Python + numpy/scipy (project env); no SimX runs.

CLI:  python -B model/calibrate.py [--out results/<run>] [--starts N]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import opgen, roofline
from model.memhier import Hier, Knobs, MemHier, psbase_hier

PS_ROOT = Path(__file__).resolve().parents[1]
TABLE = PS_ROOT / "results" / "P3.2" / "table_v0.csv"
SPLIT_SEED = 3404          # numpy default_rng seed of the train/test cell split (recorded in results/<run>/split.csv)
TRAIN_FRAC = 0.70
MAPE_QUALITATIVE = 25.0    # MAPE > 25 % -> "qualitative only"
RSS_LIMIT_MB = 1000
CORES = 4                  # PS-BASE-S1-4c (the only machine of table_v0)

# The layer shape of every cell (methodology §11.5): scaled Llama-3.1-8B `llama8b_s8`. vocab is unused (no LM head
# rows); L = 1 (one layer per launch).
SHAPE = opgen.ModelShape(name="llama8b_s8", d=512, L=1, H_q=8, H_kv=2, D_h=64, d_ff=1792, vocab=32000,
                         tie_embeddings=False, source="results/P3.2 (methodology §11.5)")

# ------------------------------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    log: bool               # fitted in log space (positive scale parameters)
    default: float          # uncalibrated value (psbase_hier / memhier defaults)
    meaning: str


FPU_STRUCT = 256.0 / 56.104     # structural / measured FPU peak at 4 cores (methodology §11.3)
TCU_STRUCT = 4096.0 / 155.854   # structural / measured TCU peak
DRAM_PEAK = 64.0 / 40.707       # channel peak / measured LSU stream bandwidth
LAT_MAX = 4.0                   # effective / unloaded latency: the earlier loaded stream latency is <= 2.87x L0 (§11.4)
MIN_LAUNCH = 836.0              # shortest launch of table_v0 (elementwise residual_B1 at L2 ½ and ¼): a per-launch
#                                 overhead cannot exceed a whole launch (tests/test_calibrate.py checks the value)

PARAMS: tuple[Param, ...] = (
    Param("eta_fpu", 1e-3, FPU_STRUCT, True, 1.0,
          "achieved / measured-peak FP32 FLOP rate of the FPU kernels (attention, GEMV, element-wise); the measured "
          "peak is the 8-chain FMA loop (56.1 FLOP/cycle), the upper bound the structural 256"),
    Param("eta_tcu", 1e-3, TCU_STRUCT, True, 1.0,
          "achieved / measured-peak TCU FLOP rate of the WGMMA GEMM (measured peak 155.9 FLOP/cycle at 1024^2 x 64; "
          "upper bound the structural 4,096)"),
    Param("eta_dram", 0.02, DRAM_PEAK, True, 1.0,
          "achievable / measured-stream DRAM bandwidth (40.7 B/cycle); upper bound the channel peak 64 B/cycle"),
    Param("overlap", 0.0, 1.0, False, 1.0,
          "omega: share of the shorter of compute and memory time hidden behind the longer (M3; Hier.overlap)"),
    Param("lat_miss_scale", 1.0, LAT_MAX, True, 1.0,
          "multiplies the DRAM part of the unloaded miss latency (L0 - L2 hit = 51.7 cycles measured); >= 1 because "
          "the measured unloaded latency is the floor of any effective latency"),
    Param("lat_hit_scale", 1.0, LAT_MAX, True, 1.0,
          "multiplies the L1 and L2 hit latencies (13.5 + 14.7 cycles measured); >= 1 as above"),
    Param("mlp_per_warp", 0.05, 2.0, True, 2.0,
          "effective outstanding lines per warp on the LSU path (M11); above 2 the 16 L1 MSHRs per core bind at 8 warps"),
    Param("launch_overhead", 0.0, MIN_LAUNCH, False, 0.0,
          "cycles added to every launch (kernel start, CTA dispatch, final barrier; M3); at most the shortest measured "
          "launch"),
)

# Hit-model choices fitted as a discrete outer loop (methodology §9.5): (hit_model, cliff_ways).
# The default (cliff, one way) comes first: another choice replaces it only if its train loss is lower by more than
# HIT_TIE (relative), so a choice the data cannot tell apart does not move the default.
HIT_CHOICES = (("cliff", 1.0), ("cliff", 0.0), ("cliff", 2.0), ("proportional", 1.0))
HIT_TIE = 1e-3

FAMILIES = ("decode_attn", "prefill_attn", "mixed_attn", "gemv", "tcu_gemm", "elementwise")


def defaults() -> dict:
    return {p.name: p.default for p in PARAMS}


# ------------------------------------------------------------------------------------------------- cases


@dataclass
class Case:
    """One table_v0 row as one memhier launch."""

    key: str                    # op|cell|setting
    op: str                     # table op column
    family: str                 # FAMILIES
    phase: str
    cell: str
    setting: str
    measured: float             # SimX cycles
    ops: list                   # opgen ops of the launch
    q_w: float
    q_a: float
    q: float
    unit: str                   # "fpu" | "tcu": the unit the SimX kernel computes on
    knobs: Knobs
    l2_scale: float = 1.0
    l2_ways: int = 8
    warps_per_cta: int = 8
    gemm_tk: int = 32
    split: str = ""             # "train" | "test"
    meta: dict = field(default_factory=dict)


def _arg(args: str, flag: str, cast=int):
    m = re.search(rf"(?:^|\s){re.escape(flag)}\s+(\S+)", args)
    return cast(m.group(1)) if m else None


def _layer_ops(seqs, q_w, q_a, q):
    return opgen.gen_ops(SHAPE, seqs, q_w=q_w, q_a=q_a, q=q, causal=True, fused_attention=True, layers=[0])


def _attn_core(ops):
    return [o for o in ops if o.layer == 0 and o.name in ("attn_qk", "attn_softmax", "attn_av")]


def _dxa_depth(setting: str) -> int:
    m = re.match(r"dxa(\d)", setting)
    return int(m.group(1)) if m else 0


def make_case(r: dict) -> Case:
    """Map one table_v0 row to its memhier launch. Modeling choices (methodology §9.8 "Row mapping"):
    * dtypes as the kernels: attention, GEMV and element-wise are fp32 (q = q_a = q_w = 4 B); the TCU GEMM reads fp16
      (q_w = q_a = 2 B);
    * compute unit as the kernel (the rule `op_unit` would put attention on the TCU): FPU kernels are costed on a
      machine without a TCU entry;
    * LMEM use = the launch's measured `lmem_bytes` (attention); DXA depth from the setting; the TCU GEMM is a
      single-buffered LMEM tile of (32 x 16 + 16 x 64) x 2 B = 3,072 B (stock app, DXA_DOUBLE_BUFFER off) with
      gemm_tk = 16 and 2 warps per CTA (its log: "CTA tile: M=32 (warps=2)", "WGMMA tile: M=16, N=64, K=16");
    * every FPU kernel launches CTAs of all NT x NW threads of a core: warps_per_cta = 8 (methodology §14.6 for the
      attention kernels; assumption for GEMV / element-wise, where it only sets the implicit tile budget);
    * the M = 16 TCU cell ran as an M = 32 GEMM (README), so it is costed as M = 32;
    * L2 knobs scale `l2_bytes` / set `l2_ways`; FIFO has no model counterpart (memhier has no replacement policy), so
      it is predicted equal to base."""
    op, cell, setting, args = r["op"], r["cell"], r["setting"], r["args"]
    meas = float(r["cycles"])
    l2_scale = {"l2_half": 0.5, "l2_quarter": 0.25}.get(setting, 1.0)
    l2_ways = 16 if setting == "l2_ways16" else 8
    base = dict(key=f"{op}|{cell}|{setting}", op=op, phase=r["phase"], cell=cell, setting=setting, measured=meas,
                l2_scale=l2_scale, l2_ways=l2_ways)
    lmem = int(float(r["lmem_bytes"])) if r.get("lmem_bytes") not in (None, "") else 0
    dxa = _dxa_depth(setting)
    if op in ("decode_attn", "prefill_attn", "mixed_attn"):
        if op == "decode_attn":
            seqs = opgen.decode(int(r["B"]), int(r["T"]))
        elif op == "prefill_attn":
            seqs = opgen.prefill(int(r["B"]), int(r["S"]))
        else:
            c, s0, n, T = _arg(args, "-c"), _arg(args, "-o"), _arg(args, "-n"), _arg(args, "-T")
            seqs = opgen.mixed([(c, s0)], [(n, T)])
        ops = _attn_core(_layer_ops(seqs, 4, 4, 4))
        return Case(family=op, ops=ops, q_w=4, q_a=4, q=4, unit="fpu",
                    knobs=Knobs(lmem_use=lmem, dxa_depth=dxa), warps_per_cta=8, **base)
    if op == "linear":
        if r["app"] == "ps_gemv_f32":
            N, K = _arg(args, "-m"), _arg(args, "-n")
            o = opgen.linear(0, cell, 1, K, N, 4, 4)
            return Case(family="gemv", ops=[o], q_w=4, q_a=4, q=4, unit="fpu", knobs=Knobs(), warps_per_cta=8, **base)
        Mrun, N, K = _arg(args, "-m"), _arg(args, "-n"), _arg(args, "-k")
        o = opgen.linear(0, cell, Mrun, K, N, 2, 2)
        return Case(family="tcu_gemm", ops=[o], q_w=2, q_a=2, q=2, unit="tcu", knobs=Knobs(lmem_use=3072),
                    warps_per_cta=2, gemm_tk=16, meta=dict(M_run=Mrun), **base)
    if op == "elementwise":
        kern, B = cell.rsplit("_B", 1)
        B = int(B)
        if kern == "rope":
            # ps_rope -d 512 -m 128 -H 64: rotate q (512) and k (128), write k, v into the cache = opgen rope_kv_write
            ops = [o for o in _layer_ops(opgen.decode(B, 256), 4, 4, 4) if o.name == "rope_kv_write"]
        else:
            name = {"rmsnorm": "attn_norm", "residual": "attn_residual", "silu_gate": "silu_mul"}[kern]
            ops = [o for o in _layer_ops(opgen.decode(B, 256), 4, 4, 4) if o.name == name and o.layer == 0]
        assert len(ops) == 1, (cell, ops)
        return Case(family="elementwise", ops=ops, q_w=4, q_a=4, q=4, unit="fpu", knobs=Knobs(), warps_per_cta=8,
                    meta=dict(kernel=kern), **base)
    raise ValueError(f"unknown op {op!r}")


def load_table(path=TABLE) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def select(rows: Sequence[dict]) -> tuple[list[dict], dict]:
    """(included rows, {reason: count}) — see the module docstring for the rule."""
    inc, exc = [], {}
    for r in rows:
        if r["how"] == "run":
            inc.append(r)
        else:
            why = {"same_as_base": "same_as_base (value = the base row's)",
                   "variant": "naive decode variant (no model counterpart)"}.get(r["how"], r["how"])
            exc[why] = exc.get(why, 0) + 1
    return inc, exc


def split_cells(cases: Sequence[Case], seed: int = SPLIT_SEED, frac: float = TRAIN_FRAC) -> dict:
    """Assign every (op family, cell) to train or test: per family, the sorted cells are permuted with
    numpy.default_rng(seed) and the first round(frac * n) go to train (at least one cell on each side). All settings
    of a cell share its side, so the TEST measures generalisation to unseen shapes, not to unseen knob values of a
    shape the fit has seen. Returns {(family, cell): split} and sets Case.split."""
    rng = np.random.default_rng(seed)
    out = {}
    for fam in FAMILIES:
        cells = sorted({c.cell for c in cases if c.family == fam})
        if not cells:
            continue
        perm = rng.permutation(len(cells))
        n_tr = min(max(1, round(frac * len(cells))), len(cells) - 1) if len(cells) > 1 else 1
        for rank, i in enumerate(perm):
            out[(fam, cells[i])] = "train" if rank < n_tr else "test"
    for c in cases:
        c.split = out[(c.family, c.cell)]
    return out


def build_cases(path=TABLE, seed: int = SPLIT_SEED) -> tuple[list[Case], dict, dict]:
    inc, exc = select(load_table(path))
    cases = [make_case(r) for r in inc]
    split = split_cells(cases, seed)
    return cases, split, exc


# ------------------------------------------------------------------------------------------------- prediction

_BASE_HIER = None
_BASE_MACHINE = None


def base_hier() -> Hier:
    global _BASE_HIER
    if _BASE_HIER is None:
        _BASE_HIER = psbase_hier(CORES)
    return _BASE_HIER


def base_machine() -> roofline.Machine:
    global _BASE_MACHINE
    if _BASE_MACHINE is None:
        _BASE_MACHINE = roofline.vortex_psbase(CORES)
    return _BASE_MACHINE


def hier_for_theta(theta: dict, hit=("cliff", 1.0), case: Case | None = None) -> Hier:
    """PS-BASE-S1-4c hierarchy with the calibrated values θ (and a case's structural overrides)."""
    h0 = base_hier()
    kw = dict(eta_dram=theta["eta_dram"], eta_comp=1.0, overlap=theta["overlap"],
              l1_lat=h0.l1_lat * theta["lat_hit_scale"], l2_lat=h0.l2_lat * theta["lat_hit_scale"],
              dram_lat=h0.dram_lat * theta["lat_miss_scale"], mlp_per_warp=theta["mlp_per_warp"],
              launch_overhead=theta["launch_overhead"], hit_model=hit[0], cliff_ways=hit[1],
              persist_across_launch=False)
    if case is not None:
        kw.update(l2_bytes=h0.l2_bytes * case.l2_scale, l2_ways=case.l2_ways, warps_per_cta=case.warps_per_cta,
                  gemm_tk=case.gemm_tk)
    return dataclasses.replace(h0, **kw)


def machine_for_theta(theta: dict, unit: str = "tcu", family: str | None = None) -> roofline.Machine:
    """PS-BASE-S1-4c roofline machine with calibrated compute rates. Per-unit efficiencies multiply the measured peaks
    (memhier's own eta_comp stays 1); `unit="fpu"` drops the TCU entry so every op of the launch runs on the FPU, as
    the SimX kernel does. A per-family efficiency `eta_comp:<family>` (diagnostic variant) overrides the unit's."""
    m0 = base_machine()
    e_f = theta.get(f"eta_comp:{family}", theta["eta_fpu"]) if unit == "fpu" else theta["eta_fpu"]
    e_t = theta.get(f"eta_comp:{family}", theta["eta_tcu"]) if unit == "tcu" else theta["eta_tcu"]
    fl = {"fpu": m0.flops_per_cycle["fpu"] * e_f}
    if unit == "tcu":
        fl["tcu"] = m0.flops_per_cycle["tcu"] * e_t
    return dataclasses.replace(m0, name=f"{m0.name}-cal", flops_per_cycle=fl)


def predict_case(theta: dict, case: Case, hit=("cliff", 1.0), detail: bool = False):
    h = hier_for_theta(theta, hit, case)
    mc = machine_for_theta(theta, case.unit, case.family)
    mh = MemHier(SHAPE, mc, hier=h, q_w=case.q_w, q_a=case.q_a, q=case.q, memo=0, persist_across_launch=False)
    r = mh.cost_launch(case.ops, case.knobs)
    return r if detail else r.cycles


def predict(theta: dict, cases: Sequence[Case], hit=("cliff", 1.0)) -> np.ndarray:
    return np.array([predict_case(theta, c, hit) for c in cases])


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**20


def rss_guard():
    r = rss_mb()
    if r > RSS_LIMIT_MB:
        raise MemoryError(f"RSS {r:.0f} MB above the {RSS_LIMIT_MB} MB guard")


# ------------------------------------------------------------------------------------------------- fitting


def _to_u(p: Param, v: float) -> float:
    return math.log(v) if p.log else v


def _from_u(p: Param, u: float) -> float:
    return math.exp(u) if p.log else u


def _bounds_u(p: Param) -> tuple[float, float]:
    return (_to_u(p, p.lo), _to_u(p, p.hi))


@dataclass
class FitResult:
    theta: dict
    hit: tuple
    loss: float               # mean squared log ratio on the fitted rows
    n_rows: int
    nfev: int
    starts: list              # per start: (start theta, end loss)
    jac: np.ndarray | None = None
    free: tuple = ()
    params: tuple = ()        # the Param objects of `free` (bounds used by identifiability)


def residuals(theta, cases, hit) -> np.ndarray:
    pred = predict(theta, cases, hit)
    return np.log(pred / np.array([c.measured for c in cases]))


def fit(cases: Sequence[Case], params: Sequence[Param] = PARAMS, hit=("cliff", 1.0), fixed: dict | None = None,
        n_starts: int = 8, seed: int = 0, max_nfev: int = 400, x0s: Sequence[dict] | None = None) -> FitResult:
    """Least squares on ln(pred/meas) over `cases` for the free `params` (others from `fixed` or defaults()).
    Starts: the uncalibrated defaults (clipped into the bounds) + (n_starts − 1) points drawn uniformly in the
    transformed box with numpy.default_rng(seed), or the explicit `x0s`. Deterministic for a given seed."""
    from scipy.optimize import least_squares
    base = defaults()
    base.update(fixed or {})
    params = list(params)
    lo = np.array([_bounds_u(p)[0] for p in params])
    hi = np.array([_bounds_u(p)[1] for p in params])

    def theta_of(u):
        t = dict(base)
        for p, x in zip(params, u):
            t[p.name] = _from_u(p, float(x))
        return t

    def f(u):
        rss_guard()
        return residuals(theta_of(u), cases, hit)

    if x0s is None:
        rng = np.random.default_rng(seed)
        starts = [np.clip([_to_u(p, base.get(p.name, p.default)) for p in params], lo + 1e-9 * (hi - lo),
                          hi - 1e-9 * (hi - lo))]
        for _ in range(n_starts - 1):
            starts.append(lo + rng.random(len(params)) * (hi - lo))
    else:
        starts = [np.clip([_to_u(p, x[p.name]) for p in params], lo, hi) for x in x0s]
    best, log, nfev = None, [], 0
    for s in starts:
        r = least_squares(f, s, bounds=(lo, hi), method="trf", diff_step=1e-4, x_scale="jac", max_nfev=max_nfev)
        nfev += r.nfev
        loss = float(np.mean(r.fun ** 2))
        log.append((theta_of(s), loss))
        if best is None or loss < best[1]:
            best = (r, loss)
    r, loss = best
    return FitResult(theta=theta_of(r.x), hit=tuple(hit), loss=loss, n_rows=len(cases), nfev=nfev, starts=log,
                     jac=r.jac, free=tuple(p.name for p in params), params=tuple(params))


def fit_all_hits(cases, params=PARAMS, fixed=None, n_starts=8, seed=0, hits=HIT_CHOICES) -> tuple[FitResult, list]:
    """Fit every discrete hit-model choice; return the chosen one (the first, i.e. the default, unless another
    has a train loss lower by more than HIT_TIE) and all of them."""
    res = [fit(cases, params, hit, fixed, n_starts, seed) for hit in hits]
    best = res[0]
    for r in res[1:]:
        if r.loss < best.loss * (1 - HIT_TIE):
            best = r
    return best, res


IDENT_REL_SE = 0.25   # "identified": interior, moves predictions, relative standard error < 25 %


def identifiability(fr: FitResult, params: Sequence[Param]) -> list[dict]:
    """Local identifiability at the optimum from the residual Jacobian J (columns = transformed parameters: ln θ for
    log parameters, so their standard error is a relative one). Parameters on a bound (active constraints) or whose
    column is ~0 (they move no prediction) are reported as not identified and left out of the covariance; for the
    interior ones, se = sqrt(s² · diag((J_IᵀJ_I)⁻¹)) with s² = Σr²/(n − p_I) (Gauss-Newton approximation), the
    correlation of the estimates, and the condition number of the column-normalised J_I. "identified" = interior,
    non-zero column and relative se < IDENT_REL_SE."""
    J = fr.jac
    n, p = J.shape
    norms = np.linalg.norm(J, axis=0)
    pmap = {q.name: q for q in params}

    def at_bound(name):
        q, v = pmap[name], fr.theta[name]
        lo, hi = _bounds_u(q)
        u = _to_u(q, v)
        return abs(u - lo) <= 1e-6 * (1 + abs(lo)) or abs(u - hi) <= 1e-6 * (1 + abs(hi))

    inter = [j for j, nm in enumerate(fr.free) if not at_bound(nm) and norms[j] > 1e-8]
    cov, cond = None, float("nan")
    if inter:
        JI = J[:, inter]
        s2 = fr.loss * n / max(n - len(inter), 1)
        cov = s2 * np.linalg.pinv(JI.T @ JI)
        ni = norms[inter]
        sv = np.linalg.svd(JI / ni, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    out = []
    for j, name in enumerate(fr.free):
        q, v = pmap[name], fr.theta[name]
        row = dict(param=name, value=v, at_bound=int(at_bound(name)), jac_col_norm=float(norms[j]),
                   se_transformed=float("nan"), rel_se=float("nan"), max_abs_corr_with="", max_abs_corr=float("nan"),
                   identified=0, cond_normalised_J_interior=cond)
        if j in inter:
            a = inter.index(j)
            se = float(math.sqrt(max(cov[a, a], 0.0)))
            rel = se if q.log else se / max(abs(v), 1e-12)
            corr = {fr.free[inter[b]]: cov[a, b] / math.sqrt(cov[a, a] * cov[b, b])
                    for b in range(len(inter)) if b != a and cov[a, a] > 0 and cov[b, b] > 0}
            worst = max(corr.items(), key=lambda kv: abs(kv[1])) if corr else ("", float("nan"))
            row.update(se_transformed=se, rel_se=rel, max_abs_corr_with=worst[0], max_abs_corr=float(worst[1]),
                       identified=int(rel < IDENT_REL_SE))
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------------- metrics


def ape(pred, meas) -> np.ndarray:
    return np.abs(np.asarray(pred) / np.asarray(meas) - 1.0) * 100.0


def metrics(pred, meas) -> dict:
    a = ape(pred, meas)
    lr = np.log(np.asarray(pred) / np.asarray(meas))
    return dict(n=int(len(a)), mape=float(np.mean(a)) if len(a) else float("nan"),
                median_ape=float(np.median(a)) if len(a) else float("nan"),
                p90_ape=float(np.percentile(a, 90)) if len(a) else float("nan"),
                max_ape=float(np.max(a)) if len(a) else float("nan"),
                log_rmse=float(np.sqrt(np.mean(lr ** 2))) if len(a) else float("nan"),
                mean_log_ratio=float(np.mean(lr)) if len(a) else float("nan"))


# ------------------------------------------------------------------------------------------------- knob ranking
# Reproduces the earlier policy-tension metric (methodology §11.5) on predicted cycles: R_P(s) = Σ_c cycles(c, s) /
# Σ_c cycles(c, base) over the phase's grid cells (same_as_base cells contribute their base value), best value per knob
# type per phase, cross loss L(P <- Q) = R_P(best_Q)/R_P(best_P) − 1, material iff the bests differ and loss ≥ 1 %.

KNOB_TYPES = {"L2 size": ("base", "l2_half", "l2_quarter"), "L2 ways": ("base", "l2_ways16"),
              "replacement": ("base", "fifo"), "DXA depth": ("base", "dxa1", "dxa2", "dxa4"),
              "LMEM budget": ("base", "lmem32k", "lmem16k")}
PHASE_OPS = {"prefill": ("prefill_attn", "linear"), "decode": ("decode_attn", "linear", "elementwise"),
             "mixed": ("mixed_attn",)}
MATERIAL = 0.01


def phase_R(cyc: dict, table_rows: Sequence[dict]) -> dict:
    """cyc: {(op, cell, setting): cycles} for every grid row (run + same_as_base). Returns {(phase, setting): R}."""
    cells = {}
    for r in table_rows:
        if r["how"] == "variant":
            continue
        cells.setdefault((r["phase"], r["op"], r["cell"]), set()).add(r["setting"])
    out = {}
    settings = sorted({s for v in cells.values() for s in v})
    for ph in PHASE_OPS:
        keys = [k for k in cells if k[0] == ph]
        if not keys:
            continue
        den = sum(cyc[(op, cell, "base")] for _, op, cell in keys)
        for s in settings:
            if not all(s in cells[k] for k in keys):
                continue
            out[(ph, s)] = sum(cyc[(op, cell, s)] for _, op, cell in keys) / den
    return out


def tension(R: dict) -> list[dict]:
    rows = []
    for kt, vals in KNOB_TYPES.items():
        best = {}
        for ph in PHASE_OPS:
            # a knob type applies to a phase only if some value changes some cell (else "n/a")
            rv = {v: R[(ph, v)] for v in vals if (ph, v) in R}
            if len(rv) < 2 or all(abs(x - 1.0) < 1e-12 for x in rv.values()):
                continue
            best[ph] = min(rv, key=lambda v: (round(rv[v], 12), vals.index(v)))
        cross = []
        for p in best:
            for q in best:
                if p != q:
                    cross.append(R[(p, best[q])] / R[(p, best[p])] - 1.0)
        differs = len(set(best.values())) > 1
        mx = max(cross) if cross else 0.0
        rows.append(dict(knob=kt, **{f"best_{ph}": best.get(ph, "n/a") for ph in PHASE_OPS},
                         **{f"R_best_{ph}": R.get((ph, best[ph])) if ph in best else None for ph in PHASE_OPS},
                         best_differs=int(differs), max_cross_loss=mx,
                         material=int(differs and mx >= MATERIAL)))
    return rows


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def knob_ranking(theta, hit, cases: Sequence[Case], table_rows: Sequence[dict]) -> dict:
    """Measured vs predicted phase-level knob effects and tension verdicts, plus per-cell knob-effect agreement."""
    pred = {(c.op, c.cell, c.setting): predict_case(theta, c, hit) for c in cases}
    meas, prd = {}, {}
    for r in table_rows:
        if r["how"] == "variant":
            continue
        k = (r["op"], r["cell"], r["setting"])
        meas[k] = float(r["cycles"])
        prd[k] = pred[k] if r["how"] == "run" else pred[(r["op"], r["cell"], "base")]
    Rm, Rp = phase_R(meas, table_rows), phase_R(prd, table_rows)
    Tm, Tp = tension(Rm), tension(Rp)
    # per-cell knob effects: ratio to the cell's base, measured vs predicted, over run rows with setting != base
    eff = [(meas[k] / meas[(k[0], k[1], "base")], prd[k] / prd[(k[0], k[1], "base")], k) for k in meas
           if k[2] != "base" and (k[0], k[1], "base") in meas and k in pred]
    sign_agree = [np.sign(round(m - 1, 4)) == np.sign(round(p - 1, 4)) for m, p, _ in eff]
    return dict(R_meas=Rm, R_pred=Rp, tension_meas=Tm, tension_pred=Tp, effects=eff,
                spearman_phase_R=spearman([Rm[k] for k in Rm], [Rp[k] for k in Rm]),
                spearman_cell_effects=spearman([e[0] for e in eff], [e[1] for e in eff]),
                sign_agreement=float(np.mean(sign_agree)) if sign_agree else float("nan"))


# ------------------------------------------------------------------------------------------------- calibrated set


def load_fit(path=PS_ROOT / "results" / "P3.4" / "fit.csv", variant: str = "primary") -> tuple[dict, tuple]:
    """Calibrated θ and hit-model choice of a fit variant from results/<run>/fit.csv."""
    theta, hit = {}, None
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["variant"] != variant:
                continue
            if r["param"] == "hit_model":
                hm, cw = r["value"].split(":")
                hit = (hm, float(cw))
            else:
                theta[r["param"]] = float(r["value"])
    if not theta or hit is None:
        raise KeyError(f"variant {variant!r} not in {path}")
    return theta, hit


def calibrated_psbase(path=PS_ROOT / "results" / "P3.4" / "fit.csv", unit: str = "tcu",
                      variant: str = "primary") -> tuple[roofline.Machine, Hier]:
    """(Machine, Hier) of PS-BASE-S1-4c with the calibrated parameters — a SEPARATE parameter set: memhier and
    roofline defaults (used by the validated results) are unchanged. Valid only for cold single launches
    on PS-BASE-S1-4c (the calibration data) and, per results/<run>/README.md, only qualitatively if MAPE > 25 %. The values
    are split-specific (other split seeds move the optimum, results/<run>/seed_robustness.csv)."""
    theta, hit = load_fit(path, variant)
    return machine_for_theta(theta, unit), hier_for_theta(theta, hit)


# ------------------------------------------------------------------------------------------------- CLI


def _w(path: Path, rows: list[dict], fields: list[str] | None = None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def _mape_rows(variant, theta, hit, cases, preds) -> list[dict]:
    out = []
    meas = np.array([c.measured for c in cases])
    for split in ("train", "test", "all"):
        sel = [i for i, c in enumerate(cases) if split == "all" or c.split == split]
        groups = [("all", "all", sel), ("all", "all_but_elementwise", [i for i in sel if cases[i].family != "elementwise"])]
        groups += [("family", f, [i for i in sel if cases[i].family == f]) for f in FAMILIES]
        groups += [("phase", ph, [i for i in sel if cases[i].phase == ph]) for ph in ("prefill", "decode", "mixed")]
        groups += [("setting", s, [i for i in sel if cases[i].setting == s])
                   for s in sorted({c.setting for c in cases})]
        for kind, g, idx in groups:
            if not idx:
                continue
            m = metrics(preds[idx], meas[idx])
            out.append(dict(variant=variant, split=split, group_kind=kind, group=g, **m))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(PS_ROOT / "results" / "P3.4"))
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="seed of the random optimizer starts (not of the split)")
    ap.add_argument("--quick", action="store_true", help="primary fit only, 2 starts (development)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    table = load_table()
    cases, split, excluded = build_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    print(f"rows used {len(cases)} (train {len(train)}, test {len(test)}); excluded {excluded}", flush=True)
    _w(out / "split.csv", [dict(family=f, cell=c, split=s, seed=SPLIT_SEED, train_frac=TRAIN_FRAC,
                                n_rows=sum(1 for x in cases if x.family == f and x.cell == c))
                           for (f, c), s in sorted(split.items())])
    n_starts = 2 if a.quick else a.starts
    fit_rows, mape_rows, pred_rows, ident_rows, rank_rows, hit_rows = [], [], [], [], [], []
    variants = {}

    # V0: uncalibrated (psbase_hier + memhier defaults), for reference only
    th0 = defaults()
    variants["uncalibrated"] = (th0, ("cliff", 1.0), None)

    # primary: global θ, every included row, best hit choice by TRAIN loss
    best, allh = fit_all_hits(train, PARAMS, None, n_starts, a.seed, HIT_CHOICES[:1] if a.quick else HIT_CHOICES)
    for fr in allh:
        hit_rows.append(dict(variant="primary", hit_model=f"{fr.hit[0]}:{fr.hit[1]:g}", train_loss=fr.loss,
                             nfev=fr.nfev, chosen=int(fr is best)))
    variants["primary"] = (best.theta, best.hit, best)
    print(f"primary fit: loss {best.loss:.4f} hit {best.hit} {time.time() - t0:.0f} s RSS {rss_mb():.0f} MB", flush=True)

    if not a.quick:
        # V1 (diagnostic): one compute efficiency per op family instead of per unit; other parameters global
        fam_params = [p for p in PARAMS if p.name not in ("eta_fpu", "eta_tcu")]
        fam_params += [Param(f"eta_comp:{f}", 1e-3, TCU_STRUCT if f == "tcu_gemm" else FPU_STRUCT, True, 1.0,
                             f"compute efficiency of the {f} kernels only (diagnostic)") for f in FAMILIES]
        fixed = {"eta_fpu": 1.0, "eta_tcu": 1.0}
        b1 = min((fit(train, fam_params, hit, fixed, n_starts, a.seed) for hit in (best.hit,)), key=lambda x: x.loss)
        variants["per_family_eta"] = (b1.theta, b1.hit, b1)
        print(f"per-family fit: loss {b1.loss:.4f} {time.time() - t0:.0f} s", flush=True)
        # V2 (diagnostic): global θ fitted and evaluated without the soft-float element-wise rows
        tr2 = [c for c in train if c.family != "elementwise"]
        b2 = fit(tr2, PARAMS, best.hit, None, n_starts, a.seed)
        variants["no_elementwise"] = (b2.theta, b2.hit, b2)
        print(f"no-elementwise fit: loss {b2.loss:.4f} {time.time() - t0:.0f} s", flush=True)

    pmap = {p.name: p for p in PARAMS}
    for vname, (th, hit, fr) in variants.items():
        vc = cases if vname != "no_elementwise" else [c for c in cases if c.family != "elementwise"]
        preds = predict(th, vc, hit)
        mape_rows += _mape_rows(vname, th, hit, vc, preds)
        free = fr.free if fr is not None else ()
        for name, v in th.items():
            p = pmap.get(name)
            if p is None and name.startswith("eta_comp:"):
                p = Param(name, 1e-3, TCU_STRUCT if name.endswith("tcu_gemm") else FPU_STRUCT, True, 1.0,
                          f"compute efficiency of the {name.split(':')[1]} kernels only (diagnostic)")
            fit_rows.append(dict(variant=vname, param=name, value=v, fitted=int(name in free),
                                 lo=p.lo if p else "", hi=p.hi if p else "", default=p.default if p else "",
                                 meaning=p.meaning if p else ""))
        fit_rows.append(dict(variant=vname, param="hit_model", value=f"{hit[0]}:{hit[1]:g}",
                             fitted=int(fr is not None and vname == "primary"), lo="", hi="", default="cliff:1",
                             meaning="L2 hit model and cliff width in ways (discrete, chosen by train loss)"))
        if fr is not None:
            fit_rows.append(dict(variant=vname, param="train_loss_msle", value=fr.loss, fitted=0, lo="", hi="",
                                 default="", meaning=f"mean squared ln(pred/meas) on {fr.n_rows} train rows"))
            for r in identifiability(fr, fr.params):
                ident_rows.append(dict(variant=vname, **r))
        for c, pr in zip(vc, preds):
            lr = predict_case(th, c, hit, detail=True) if vname == "primary" else None
            pred_rows.append(dict(variant=vname, op=c.op, family=c.family, phase=c.phase, cell=c.cell,
                                  setting=c.setting, split=c.split, measured=c.measured, predicted=float(pr),
                                  ratio=float(pr / c.measured), ape=float(abs(pr / c.measured - 1) * 100),
                                  **({} if lr is None else dict(t_comp=lr.t_comp, t_mem=lr.t_mem, t_bw=lr.t_bw,
                                                                 t_lsu=lr.t_lsu, bound=lr.bound))))
        if vname != "no_elementwise":
            kr = knob_ranking(th, hit, cases, table)
            for (ph, s), rm in sorted(kr["R_meas"].items()):
                rank_rows.append(dict(variant=vname, kind="R", phase=ph, setting=s, R_meas=rm,
                                      R_pred=kr["R_pred"][(ph, s)]))
            for tm, tp in zip(kr["tension_meas"], kr["tension_pred"]):
                rank_rows.append(dict(variant=vname, kind="tension", phase="", setting=tm["knob"],
                                      best_meas="/".join(tm[f"best_{p}"] for p in PHASE_OPS),
                                      best_pred="/".join(tp[f"best_{p}"] for p in PHASE_OPS),
                                      max_cross_loss_meas=tm["max_cross_loss"], max_cross_loss_pred=tp["max_cross_loss"],
                                      material_meas=tm["material"], material_pred=tp["material"]))
            rank_rows.append(dict(variant=vname, kind="summary", phase="", setting="",
                                  spearman_phase_R=kr["spearman_phase_R"],
                                  spearman_cell_effects=kr["spearman_cell_effects"],
                                  sign_agreement_cell_effects=kr["sign_agreement"],
                                  n_cell_effects=len(kr["effects"])))
        rss_guard()

    _w(out / "fit.csv", fit_rows, ["variant", "param", "value", "fitted", "lo", "hi", "default", "meaning"])
    _w(out / "mape.csv", mape_rows)
    _w(out / "predictions.csv", pred_rows, ["variant", "op", "family", "phase", "cell", "setting", "split", "measured",
                                            "predicted", "ratio", "ape", "t_comp", "t_mem", "t_bw", "t_lsu", "bound"])
    if ident_rows:
        _w(out / "identifiability.csv", ident_rows)
    _w(out / "hit_choice.csv", hit_rows)
    _w(out / "knob_ranking.csv", rank_rows,
       ["variant", "kind", "phase", "setting", "R_meas", "R_pred", "best_meas", "best_pred", "max_cross_loss_meas",
        "max_cross_loss_pred", "material_meas", "material_pred", "spearman_phase_R", "spearman_cell_effects",
        "sign_agreement_cell_effects", "n_cell_effects"])
    tm = {(r["variant"], r["split"], r["group_kind"], r["group"]): r for r in mape_rows}
    prim = tm[("primary", "test", "all", "all")]
    verdict = "qualitative only" if prim["mape"] > MAPE_QUALITATIVE else "quantitative (MAPE <= 25 %)"
    summ = dict(rows_used=len(cases), train_rows=len(train), test_rows=len(test), excluded=excluded,
                split_seed=SPLIT_SEED, train_frac=TRAIN_FRAC, optimizer_seed=a.seed, starts=n_starts,
                primary_test_mape=prim["mape"], verdict=verdict, primary_hit=list(best.hit),
                wall_s=round(time.time() - t0, 1), peak_rss_mb=round(rss_mb(), 1))
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
    print(json.dumps(summ, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
