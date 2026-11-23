#!/usr/bin/env python3
"""pstrace.py -- parser for the PhaseShift work-unit interval trace and the MPM class-8 report.

Inputs (written by the ps-tree SimX, patch ps-counters; format and rules: docs/methodology.md §12.9-§12.10):
  * the trace file of PS_TRACE=<file> (CSV): fixed columns
        launch, row, kind (work|flush), interval, unit (marks|instrs), every (n), work_end, work_at,
        cycle (launch-local cycle at the row's close), cycles (delta), instrs (delta)
    then every class-8 counter by name (VX_PS_CNT_* order). Each row holds the counter DELTAS since the previous row;
    the dxa_max_inflight_* columns are the maximum WITHIN the row instead.
  * a run log with `--perf=8` (VORTEX_PROFILING=8): lines `PERF: ps: name=value, ...` (device-wide totals since
    device open) and the global `PERF: instrs=..., cycles=..., IPC=...` line.

Library use (tests/test_pstrace.py, later steps):
  read_trace(path)               -> (counter names, rows as dicts)
  parse_perf_ps(text)            -> {name: total} from the PERF: ps: lines
  parse_perf_global(text)        -> (instrs, cycles) or None
  totals(rows, names)            -> sum over rows (max for maxima)
  check_totals(rows, names, perf)-> [(name, trace_total, perf_total)] mismatches (empty = exact)
  work_rows(rows, launch=None)   -> the kind == work rows (of one launch)
  expected_work_rows(units, n)   -> ceil(units / n), and 1 for a launch without units (one whole-launch row)
  hist_quantile(counts, q)       -> (lo, hi) cycle bounds of the log2 bucket holding the q-quantile
  interval_summary(rows, names)  -> per work row: cycles, instrs, units, per-class DRAM bytes, mean/P99 read wait

CLI:
  pstrace.py check <trace.csv> <run.log>     totals and per-launch row counts; exit 0 iff exact
  pstrace.py summary <trace.csv> [--out f.csv]
Pure Python (stdlib only); small memory footprint (rows are small).
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

FIXED = ["launch", "row", "kind", "interval", "unit", "every", "work_end", "work_at", "cycle", "cycles", "instrs"]
STR_COLS = {"kind", "unit"}
CLASSES = ["ord", "wgt", "kvp", "kvs", "dxa", "c5", "c6", "c7"]          # = VX_PS_CLASS_* order
MARK_TYPES = ["other", "prefill", "decode", "m3", "m4", "m5", "m6", "m7"]  # = VX_PS_MARK_* order
HIST_BUCKETS = 16                                                         # = VX_PS_HIST_BUCKETS
PERF_PS_RE = re.compile(r"^PERF: ps: (.*)$", re.M)
GLOBAL_RE = re.compile(r"^PERF: instrs=(\d+), cycles=(\d+), IPC=", re.M)


def is_max(name: str) -> bool:
    """Counters that are maxima (dxa_max_inflight_s*): interval value = max within the row, total = max over rows."""
    return name.startswith("dxa_max_inflight_")


def read_trace(path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        if header[:len(FIXED)] != FIXED:
            raise ValueError(f"{path}: not a PS_TRACE file (header {header[:len(FIXED)]})")
        names = header[len(FIXED):]
        rows = []
        for rec in rd:
            if not rec:
                continue
            if len(rec) != len(header):
                raise ValueError(f"{path}: row {len(rows)} has {len(rec)} fields, header {len(header)}")
            rows.append({k: (v if k in STR_COLS else int(v)) for k, v in zip(header, rec)})
    return names, rows


def parse_perf_ps(text: str) -> dict:
    out = {}
    for m in PERF_PS_RE.finditer(text):
        for item in m.group(1).split(", "):
            k, _, v = item.partition("=")
            out[k.strip()] = int(v)
    return out


def parse_perf_global(text: str):
    m = GLOBAL_RE.findall(text)
    return (int(m[-1][0]), int(m[-1][1])) if m else None


def totals(rows: list[dict], names: list[str]) -> dict:
    out = {}
    for n in names + ["cycles", "instrs"]:
        vals = [r[n] for r in rows]
        out[n] = (max(vals) if vals else 0) if is_max(n) else sum(vals)
    return out


def check_totals(rows: list[dict], names: list[str], perf: dict) -> list[tuple]:
    """Every counter the perf report has must equal the trace total exactly (sum, or max for maxima); a counter
    missing on either side is a mismatch too."""
    tot = totals(rows, names)
    bad = [(n, tot.get(n), perf.get(n)) for n in sorted(set(names) | set(perf)) if tot.get(n) != perf.get(n)]
    return bad


def work_rows(rows: list[dict], launch: int | None = None) -> list[dict]:
    return [r for r in rows if r["kind"] == "work" and (launch is None or r["launch"] == launch)]


def expected_work_rows(units: int, n: int) -> int:
    """Rows of one launch: ceil(units / n); a launch with no unit at all still gets one (whole-launch) row."""
    return max(1, math.ceil(units / n))


def launch_units(rows: list[dict], launch: int) -> int:
    w = work_rows(rows, launch)
    return w[-1]["work_at"] if w else 0


def hist_quantile(counts: list[int], q: float) -> tuple[float, float] | None:
    """Bounds [lo, hi) in cycles of the log2 bucket that holds the q-quantile (bucket 0 = [0, 2), k = [2^k, 2^(k+1)),
    the last bucket is open: hi = inf). None without samples. The quantile is known only to this resolution."""
    total = sum(counts)
    if total == 0:
        return None
    need = q * total
    acc = 0
    for k, c in enumerate(counts):
        acc += c
        if acc >= need and c > 0:
            lo = 0.0 if k == 0 else float(2 ** k)
            hi = math.inf if k == len(counts) - 1 else float(2 ** (k + 1))
            return lo, hi
    return None


def hist_of(row: dict, cls: str) -> list[int]:
    return [row[f"dram_rd_hist_{cls}_b{b}"] for b in range(HIST_BUCKETS)]


def interval_summary(rows: list[dict], names: list[str]) -> list[dict]:
    out = []
    for r in work_rows(rows):
        s = {k: r[k] for k in ("launch", "interval", "work_end", "cycle", "cycles", "instrs")}
        for c in CLASSES[:5]:
            s[f"dram_bytes_{c}"] = r[f"dram_rd_bytes_{c}"] + r[f"dram_wr_bytes_{c}"]
            done = r[f"dram_rd_done_{c}"]
            s[f"dram_rd_wait_mean_{c}"] = (r[f"dram_rd_wait_{c}"] / done) if done else ""
            p99 = hist_quantile(hist_of(r, c), 0.99)
            s[f"dram_rd_wait_p99_hi_{c}"] = "" if p99 is None else p99[1]
        out.append(s)
    return out


def check(trace: Path, log: Path, out=sys.stdout) -> int:
    names, rows = read_trace(trace)
    text = log.read_text(errors="replace")
    perf = parse_perf_ps(text)
    rc = 0
    if not perf:
        print(f"pstrace: {log}: no `PERF: ps:` lines (run with --perf=8)", file=out)
        return 1
    bad = check_totals(rows, names, perf)
    for n, t, p in bad[:20]:
        print(f"pstrace: MISMATCH {n}: trace {t} perf {p}", file=out)
    rc |= bool(bad)
    g = parse_perf_global(text)
    tot = totals(rows, names)
    if g is not None:
        print(f"pstrace: instrs trace {tot['instrs']} perf {g[0]}; cycles trace {tot['cycles']} perf {g[1]}", file=out)
    for launch in sorted({r["launch"] for r in rows}):
        w = work_rows(rows, launch)
        units = launch_units(rows, launch)
        n = w[0]["every"] if w else 1
        exp = expected_work_rows(units, n)
        ok = len(w) == exp
        rc |= not ok
        print(f"pstrace: launch {launch}: {len(w)} work rows, units {units}, n {n}, expected {exp} "
              f"{'OK' if ok else 'MISMATCH'}", file=out)
    print(f"pstrace: {'PASS' if rc == 0 else 'FAIL'} ({len(names)} counters, {len(rows)} rows)", file=out)
    return int(rc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("trace", type=Path)
    c.add_argument("log", type=Path)
    s = sub.add_parser("summary")
    s.add_argument("trace", type=Path)
    s.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return check(a.trace, a.log)
    names, rows = read_trace(a.trace)
    summ = interval_summary(rows, names)
    f = open(a.out, "w", newline="") if a.out else sys.stdout
    if summ:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    if a.out:
        f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
