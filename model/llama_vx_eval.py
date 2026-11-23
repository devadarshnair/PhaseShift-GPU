"""Parse and evaluate ps_llama logs (host app runtime/llama_vx/main.cpp).

Pure functions, no SimX: used by results/<run>/summarize.py (CSV evidence) and tests/test_llama_vx.py. The evaluation
re-applies the TEST rules to the logged values independently of the app's own PASSED/FAILED:

* token rule (+ the carry-over): at every compared (non-forced) position the device
  argmax equals the golden next token, or the disagreement is a near-tie in BOTH senses -- |l1 - l2| / |l1| <= 1e-4
  (top-1 vs top-2, the literal rule) and |l1 - lg| / |l1| <= 1e-4 (lg = the golden token's own logit);
* free-running: the generated tokens equal the golden ones up to the first disagreement, which must be such a
  near-tie (comparison stops there);
* logits: rel = max_i |l_i - r_i| / max_j |r_j| against the NumPy float32 reference (model/llama_vx_ref.py),
  tolerance 1e-3 (batched prefill; methodology §14.5).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

TIE_REL = 1e-4
LOGITS_REL = 1e-3

_KV = re.compile(r"(\w+)=(\S+)")


def _num(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _fields(line: str) -> dict:
    return {k: _num(v) for k, v in _KV.findall(line)}


def parse_log(text: str) -> dict:
    """Split a ps_llama log into its machine-readable records (see the main.cpp header 'Output')."""
    out = dict(header={}, device={}, gemv=[], launches=[], tokens=[], checks=[], summary={}, gen=None,
               perf_cycles=None, perf_instrs=None, passed=False, prefill={})
    for ln in text.splitlines():
        if ln.startswith("ps_llama: "):
            out["header"] = _fields(ln)
        elif ln.startswith("device: "):
            out["device"] = _fields(ln)
        elif ln.startswith("prefill: "):
            out["prefill"] = _fields(ln)
        elif ln.startswith("PSGEMV: "):
            out["gemv"].append(_fields(ln))
        elif ln.startswith("PSL: "):
            out["launches"].append(_fields(ln))
        elif ln.startswith("PSTOK: "):
            out["tokens"].append(_fields(ln))
        elif ln.startswith("PSCHECK: "):
            f = _fields(ln)
            f["label"] = ln.split()[1]
            out["checks"].append(f)
        elif ln.startswith("PSSUM: "):
            out["summary"] = _fields(ln)
        elif ln.startswith("PSGEN: tokens="):
            out["gen"] = [int(t) for t in ln.split("=", 1)[1].strip().split(",") if t]
        else:
            m = re.match(r"^PERF: instrs=(\d+), cycles=(\d+), IPC=", ln)
            if m:
                out["perf_instrs"], out["perf_cycles"] = int(m.group(1)), int(m.group(2))
        if ln.strip() == "PASSED!":
            out["passed"] = True
    return out


def rel_margin(a: float, b: float) -> float:
    if a == 0.0:
        return 0.0 if b == 0.0 else float("inf")
    return abs(a - b) / abs(a)


def is_near_tie(top1: float, top2: float, l_golden: float, tie: float = TIE_REL) -> bool:
    """Both near-tie criteria (module docstring)."""
    return rel_margin(top1, top2) <= tie and rel_margin(top1, l_golden) <= tie


def evaluate_tokens(tokens: list[dict], mode: str) -> dict:
    """Re-apply the token rule to PSTOK records. tf/prefill: every non-forced position; free: until the first
    disagreement (inclusive)."""
    compared = agree = ties = bad = 0
    first_tie = None
    for t in tokens:
        if t["forced"]:
            continue
        ok = t["argmax"] == t["golden_next"]
        tie = (not ok) and is_near_tie(float(t["top1"]), float(t["top2"]), float(t["l_golden"]))
        compared += 1
        if ok:
            agree += 1
        elif tie:
            ties += 1
            first_tie = t["pos"] if first_tie is None else first_tie
        else:
            bad += 1
        if mode == "free" and not ok:
            break
    return dict(compared=compared, agree=agree, near_ties=ties, failures=bad, first_near_tie=first_tie,
                passed=compared > 0 and bad == 0,
                min_rel_top2=min((float(t["rel_top2"]) for t in tokens if not t["forced"]), default=None))


def logits_rel(dev: np.ndarray, ref: np.ndarray) -> float:
    """Norm-wise relative error max|dev - ref| / max|ref| of one logits vector (methodology §14.5)."""
    dev = np.asarray(dev, np.float64)
    ref = np.asarray(ref, np.float64)
    return float(np.max(np.abs(dev - ref)) / max(float(np.max(np.abs(ref))), np.finfo(np.float32).tiny))


def read_logits(path, vocab: int) -> np.ndarray:
    a = np.fromfile(Path(path), dtype="<f4")
    if a.size % vocab:
        raise ValueError(f"{path}: {a.size} floats is not a multiple of vocab {vocab}")
    return a.reshape(-1, vocab)


def op_table(launches: list[dict], positions: list[int] | None = None) -> dict:
    """Sum the PSL counters per op over the given positions (default: all). Returns {op: {cycles, instrs, mem_r,
    mem_w, mem_lt, idle, scrb, launches}}; op names as logged (rmsnorm_att, wq, ..., wcls)."""
    sel = set(positions) if positions is not None else None
    tab: dict[str, dict] = {}
    for l in launches:
        if sel is not None and l["pos"] not in sel:
            continue
        d = tab.setdefault(l["op"], dict(cycles=0, instrs=0, mem_r=0, mem_w=0, mem_lt=0, idle=0, scrb=0, launches=0))
        for k in ("cycles", "instrs", "mem_r", "mem_w", "mem_lt", "idle", "scrb"):
            d[k] += int(l[k])
        d["launches"] += 1
    return tab


def rows_set_pressure(stride_bytes: int, nt: int, l1_bytes: int = 16384, ways: int = 4, line: int = 64) -> int:
    """Python port of ps_llama's GEMV-variant rule (runtime/llama_vx/main.cpp `rows_set_pressure`, methodology §14.5):
    the largest number of the NT rows one warp of `gemv_rows` loads at once (rows r0 .. r0 + NT - 1, same column
    offset) that fall into one L1 set, over every warp-aligned r0 and every 16-byte offset within a line. `rows` is
    chosen iff the result is <= `ways` (errata 29: more rows than ways in a set evict each other)."""
    sets = l1_bytes // (line * ways)
    period = line * sets
    worst = 0
    for r0 in range(0, period, nt):
        for a in range(0, line, 16):
            cnt: dict[int, int] = {}
            for j in range(nt):
                s = ((a + (r0 + j) * stride_bytes) // line) % sets
                cnt[s] = cnt.get(s, 0) + 1
            worst = max(worst, max(cnt.values()))
    return worst


def gemv_variant(K: int, nt: int, l1_bytes: int = 16384, ways: int = 4, line: int = 64) -> str:
    return "rows" if rows_set_pressure(4 * K, nt, l1_bytes, ways, line) <= ways else "lanes"
