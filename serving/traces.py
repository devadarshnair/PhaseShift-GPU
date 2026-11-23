"""Request-trace loaders, statistics and length scaling σ (methodology §10).

Traces (fetched read-only, `data/MANIFEST.tsv`):

* Azure LLM inference trace 2023 [cand:D19] — CSV `TIMESTAMP,ContextTokens,GeneratedTokens`; TIMESTAMP is a wall-clock
  time with 7 fractional digits (e.g. `2023-11-16 18:17:03.9799600`). Arrival = seconds since the first row.
* Mooncake FAST'25 traces [cand:D20] — JSONL `{"timestamp", "input_length", "output_length", "hash_ids"}`. Per the
  release README (FAST25-release/README.md at the pinned commit, read 2026-10-31): `timestamp` is the relative arrival
  time in **milliseconds**; `hash_ids` are remapped prefix-block hashes with a block size of **512 tokens** (also
  checked on the data: len(hash_ids) = ⌈input_length/512⌉ on every row of all three files).

Each loader returns `Request`s sorted by arrival (stable, so equal timestamps keep file order) with request ids
0..n−1 in that order. Lengths must be ≥ 1 (true for all five files). Nothing under `data/` is ever written.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PS_ROOT = Path(__file__).resolve().parents[1]

TRACES = {
    "azure_code": ("data/traces/AzureLLMInferenceTrace_code.csv", "azure"),
    "azure_conv": ("data/traces/AzureLLMInferenceTrace_conv.csv", "azure"),
    "mooncake_conversation": ("data/traces/mooncake_conversation_trace.jsonl", "mooncake"),
    "mooncake_toolagent": ("data/traces/mooncake_toolagent_trace.jsonl", "mooncake"),
    "mooncake_synthetic": ("data/traces/mooncake_synthetic_trace.jsonl", "mooncake"),
}
MOONCAKE_BLOCK_TOKENS = 512


@dataclasses.dataclass(frozen=True)
class Request:
    rid: int
    arrival: float           # seconds since the trace's first arrival
    input_len: int           # prompt tokens S
    output_len: int          # generated tokens (Azure GeneratedTokens, Mooncake output_length)
    hash_ids: tuple[int, ...] | None = None   # Mooncake prefix-block hashes (carried, not used by the scheduler)


def _finish(rows: list[tuple[int, int, int, tuple | None]], ticks_per_s: int, src: str) -> list[Request]:
    """Sort by integer time (stable), subtract the first arrival in integer units, then convert to seconds (so the
    relative arrival is exact to float rounding, not limited by the precision of an absolute epoch time)."""
    rows = sorted(rows, key=lambda r: r[0])
    if not rows:
        return []
    t0 = rows[0][0]
    out = []
    for i, (t, s, o, h) in enumerate(rows):
        if s < 1 or o < 1:
            raise ValueError(f"{src}: row {i} has input {s}, output {o} (< 1)")
        out.append(Request(i, (t - t0) / ticks_per_s, s, o, h))
    return out


AZURE_TICKS_PER_S = 10_000_000      # TIMESTAMP has 7 fractional digits = 100 ns


def _azure_ticks(ts: str) -> int:
    """'2023-11-16 18:17:03.9799600' -> integer 100 ns ticks since the epoch, read as UTC (no DST/local-time
    effects); keeps all 7 digits."""
    whole, _, frac = ts.strip().partition(".")
    base = int(datetime.strptime(whole, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
    frac = (frac + "0000000")[:7] if frac else "0"
    return base * AZURE_TICKS_PER_S + int(frac)


def load_azure(path) -> list[Request]:
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        if header != ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]:
            raise ValueError(f"{path}: unexpected header {header}")
        rows = [(_azure_ticks(r[0]), int(r[1]), int(r[2]), None) for r in rd if r]
    return _finish(rows, AZURE_TICKS_PER_S, str(path))


def load_mooncake(path) -> list[Request]:
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["timestamp"] != int(d["timestamp"]):
                raise ValueError(f"{path}: non-integer millisecond timestamp {d['timestamp']}")
            rows.append((int(d["timestamp"]), int(d["input_length"]), int(d["output_length"]),
                         tuple(d["hash_ids"])))
    return _finish(rows, 1000, str(path))           # timestamps are integer milliseconds


def load_trace(name: str, root: Path = PS_ROOT, limit: int | None = None) -> list[Request]:
    """Load a pinned trace by name (see TRACES); `limit` keeps the first `limit` arrivals."""
    rel, kind = TRACES[name]
    reqs = (load_azure if kind == "azure" else load_mooncake)(Path(root) / rel)
    return reqs[:limit] if limit is not None else reqs


# ------------------------------------------------------------------------------------------------ statistics
STAT_PCTS = (50, 90, 99)


def trace_stats(requests: Sequence[Request]) -> dict:
    """Request count, span, mean arrival rate and length distribution. Percentiles use NumPy's default linear
    interpolation (the same definition as pandas `Series.quantile`)."""
    n = len(requests)
    arr = np.array([r.arrival for r in requests], dtype=float)
    out = {"n_requests": n, "duration_s": float(arr.max() - arr.min()) if n else 0.0}
    out["mean_rate_rps"] = (n - 1) / out["duration_s"] if n > 1 and out["duration_s"] > 0 else float("nan")
    for key, vals in (("input", [r.input_len for r in requests]), ("output", [r.output_len for r in requests])):
        v = np.array(vals, dtype=float)
        out[f"{key}_min"] = int(v.min())
        for p in STAT_PCTS:
            out[f"{key}_p{p}"] = float(np.percentile(v, p))
        out[f"{key}_mean"] = float(v.mean())
        out[f"{key}_max"] = int(v.max())
        out[f"{key}_total"] = int(v.sum())
    return out


# ------------------------------------------------------------------------------------------------ length scaling σ
def _scale_len(n: int, sigma: float) -> int:
    """round-half-up(σ·n), at least 1 token (a request always has a prompt token and an output token)."""
    return max(1, int(math.floor(sigma * n + 0.5)))


def scale_lengths(requests: Iterable[Request], sigma: float, max_context: int | None = None,
                  policy: str = "clip") -> list[Request]:
    """Multiply input and output lengths by σ (methodology §10). Arrivals and hash_ids are unchanged (the hash ids
    then describe σ·512-token blocks; the scheduler does not model prefix caching).

    `max_context` (optional, e.g. a stories model's seq_len): a request whose scaled footprint S' + O' exceeds it is
    either clipped (policy "clip": O' = min(O', max_context − 1), then the prompt is left-truncated to
    S' = max_context − O', the usual server behaviour for over-long prompts) or dropped (policy "drop"). Request ids
    are kept, so a dropped request leaves a gap.
    """
    if not sigma > 0:
        raise ValueError("sigma must be > 0")
    if policy not in ("clip", "drop"):
        raise ValueError(f"unknown policy {policy!r}")
    if max_context is not None and max_context < 2:
        raise ValueError("max_context must be >= 2 (one prompt and one output token)")
    out = []
    for r in requests:
        s, o = _scale_len(r.input_len, sigma), _scale_len(r.output_len, sigma)
        if max_context is not None and s + o > max_context:
            if policy == "drop":
                continue
            o = min(o, max_context - 1)
            s = max_context - o
        out.append(dataclasses.replace(r, input_len=s, output_len=o))
    return out


def kv_l2_ratios(requests: Sequence[Request], kv_bytes_per_token: float, l2_bytes: float) -> np.ndarray:
    """Per-request KV working set at completion over L2 size: M_KV/token · (S + O) / L2 (E3 with B = 1, T = S + O)."""
    return np.array([kv_bytes_per_token * (r.input_len + r.output_len) for r in requests], float) / l2_bytes


def median_kv_l2_ratio(requests: Sequence[Request], kv_bytes_per_token: float, l2_bytes: float) -> float:
    return float(np.median(kv_l2_ratios(requests, kv_bytes_per_token, l2_bytes)))


def sigma_for_kv_l2(kv_ref: float, l2_ref: float, kv_tgt: float, l2_tgt: float) -> float:
    """σ that makes the median per-request KV-working-set/L2 ratio on the target equal that on the reference:

        median_i(kv_tgt·σ(S_i+O_i))/l2_tgt = median_i(kv_ref·(S_i+O_i))/l2_ref
        ⇒ σ = (kv_ref / l2_ref) / (kv_tgt / l2_tgt),

    because the median commutes with multiplication by a positive constant. The trace cancels: σ depends only on
    the (model, machine) pairs. Rounding to whole tokens moves the achieved median by ≤ 1 token's worth, and clipping
    at max_context leaves the median unchanged as long as the scaled median footprint is ≤ max_context
    (check with `median_kv_l2_ratio` after `scale_lengths`; methodology §10)."""
    for v in (kv_ref, l2_ref, kv_tgt, l2_tgt):
        if not v > 0:
            raise ValueError("all arguments must be > 0")
    return (kv_ref / l2_ref) / (kv_tgt / l2_tgt)


# ------------------------------------------------------------------------------------------------ CLI
def write_stats_csv(path: Path, names: Sequence[str] = tuple(TRACES), root: Path = PS_ROOT) -> list[dict]:
    rows = []
    for name in names:
        st = trace_stats(load_trace(name, root))
        rows.append({"trace": name, "file": TRACES[name][0], **st})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="write trace statistics")
    ap.add_argument("--out", default=str(PS_ROOT / "results" / "P1.3" / "trace_stats.csv"))
    a = ap.parse_args()
    for r in write_stats_csv(Path(a.out)):
        print(r)
