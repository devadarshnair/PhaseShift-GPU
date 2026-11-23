#!/usr/bin/env python3
"""psclass.py -- parse the PS_DEBUG_CLASS report of the ps tree's SimX (ps patch mem-class).

  psclass.py <app log> [...]        print one JSON summary per log (the last report in each log)

With PS_DEBUG_CLASS=1, SimX (vortex-trees/ps/sim/simx/processor.cpp ProcessorImpl::ps_class_report) prints after every
launch (`when=end`) and after every cache flush (`when=flush`):

  PSCLASS launch=k when=W block=64 dram_total_bytes=N dram_rd_bytes=c0,..,c7 dram_wr_bytes=c0,..,c7
  PSCLASS-REGION launch=k when=W region=kv0|kv1|wgt|other bytes=c0,..,c7
  PSCLASS-CACHE launch=k when=W level=l1i|l1d|l2|l3 reqs=c0,..,c7 mixed_merges=M prio_upgrades=U
  PSCLASS-DXA launch=k when=W gmem_reqs=c0,..,c7
  PSMARK launch=k when=W core=i counts=t0,..,t7
  PSMARK-TOTAL launch=k when=W counts=t0,..,t7

Class index c = VX_PS_CLASS_* (0 ORDINARY, 1 WEIGHT, 2 KV_PERSIST, 3 KV_STREAM, 4 BULK_DXA, 5-7 reserved); mark type
t = VX_PS_MARK_* (0 OTHER, 1 PREFILL_TILE, 2 DECODE_TILE). Byte and request counters are cumulative since device open;
`dram_total_bytes` is an independent count (the processor's MPM mem_reads + mem_writes, counted where requests enter the
memory model) times the block size. Mark counts are per launch. A report is the group of lines with the same
(launch, when); `parse` returns the LAST complete group in the log (normally the flush after the last launch, which
includes the write-backs of the flush).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CLASSES = ["ORDINARY", "WEIGHT", "KV_PERSIST", "KV_STREAM", "BULK_DXA", "RSVD5", "RSVD6", "RSVD7"]
KV_CLASSES = (2, 3)   # KV_PERSIST, KV_STREAM
MARKS = ["OTHER", "PREFILL_TILE", "DECODE_TILE", "RSVD3", "RSVD4", "RSVD5", "RSVD6", "RSVD7"]

_KV = re.compile(r"(\w+)=(\S+)")


def _ints(v: str) -> list[int]:
    return [int(x) for x in v.split(",")]


def _fields(line: str) -> dict:
    return dict(_KV.findall(line))


def parse_all(text: str) -> list[dict]:
    """Every report in the log, in order: [{launch, when, block, dram_total, dram_rd, dram_wr, regions, caches, dxa,
    marks, marks_total}]."""
    reports: list[dict] = []
    # a PSCLASS line opens a new report; the other lines attach to the newest report with the same (launch, when)
    current: dict[tuple, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(("PSCLASS", "PSMARK")):
            continue
        tag = line.split(" ", 1)[0]
        f = _fields(line)
        if "launch" not in f or "when" not in f:
            continue
        lw = (int(f["launch"]), f["when"])
        if tag == "PSCLASS":
            r = {"launch": lw[0], "when": lw[1], "block": int(f["block"]), "dram_total": int(f["dram_total_bytes"]),
                 "dram_rd": _ints(f["dram_rd_bytes"]), "dram_wr": _ints(f["dram_wr_bytes"]), "regions": {},
                 "caches": {}, "dxa": None, "marks": {}, "marks_total": None}
            reports.append(r)
            current[lw] = r
            continue
        r = current.get(lw)
        if r is None:
            continue
        if tag == "PSCLASS-REGION":
            r["regions"][f["region"]] = _ints(f["bytes"])
        elif tag == "PSCLASS-CACHE":
            r["caches"][f["level"]] = {"reqs": _ints(f["reqs"]), "mixed_merges": int(f["mixed_merges"]),
                                       "prio_upgrades": int(f["prio_upgrades"])}
        elif tag == "PSCLASS-DXA":
            r["dxa"] = _ints(f["gmem_reqs"])
        elif tag == "PSMARK":
            r["marks"][int(f["core"])] = _ints(f["counts"])
        elif tag == "PSMARK-TOTAL":
            r["marks_total"] = _ints(f["counts"])
    return reports


def parse(text: str) -> dict | None:
    """The last complete report (has its PSMARK-TOTAL line) or None."""
    reps = [r for r in parse_all(text) if r["marks_total"] is not None]
    return reps[-1] if reps else None


def derived(r: dict) -> dict:
    """Sums and fractions the Tests use."""
    cls_bytes = [a + b for a, b in zip(r["dram_rd"], r["dram_wr"])]
    kv = [0] * 8
    for reg in ("kv0", "kv1"):
        for c, v in enumerate(r["regions"].get(reg, [0] * 8)):
            kv[c] += v
    kv_total = sum(kv)
    region_total = sum(sum(v) for v in r["regions"].values())
    return {
        "class_bytes": cls_bytes,
        "class_sum": sum(cls_bytes),
        "dram_total": r["dram_total"],
        "region_sum": region_total,
        "kv_region_bytes": kv,
        "kv_region_total": kv_total,
        "kv_labelled_frac": (sum(kv[c] for c in KV_CLASSES) / kv_total) if kv_total else None,
        "kv_bulk_dxa_frac": (kv[4] / kv_total) if kv_total else None,
    }


def parse_pshint(text: str) -> dict | None:
    """The host's `PSHINT: ps_hints=h expect_marks=a,b,c` line (host apps)."""
    m = re.search(r"^PSHINT: ps_hints=(\d+) expect_marks=([\d,]+)", text, re.M)
    if not m:
        return None
    return {"ps_hints": int(m.group(1)), "expect_marks": _ints(m.group(2))}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    rc = 0
    for p in argv:
        text = Path(p).read_text(errors="replace")
        r = parse(text)
        if r is None:
            print(json.dumps({"log": p, "error": "no PS_DEBUG_CLASS report"}))
            rc = 1
            continue
        out = {"log": p, "launch": r["launch"], "when": r["when"], **derived(r), "dxa": r["dxa"],
               "marks_total": r["marks_total"], "pshint": parse_pshint(text),
               "caches": r["caches"]}
        print(json.dumps(out))
    return rc


if __name__ == "__main__":
    sys.exit(main())
