#!/usr/bin/env python3
"""mkschedule.py -- write / validate PS_SCHEDULE files (format: docs/methodology.md §5.1).

The C++ parser is vortex-trees/ps/sim/simx/ps_schedule.h (ps patch 0004-sched-replay); this module applies the same
rules so a bad schedule (e.g. a DP schedule that asks for a way-mask change) is refused before a SimX run.

  mkschedule.py write OUT  [--launch K] W:knob=v[,knob=v...] [W:...]   one switch per argument, in order
  mkschedule.py check FILE                                             parse; exit 0 iff valid, prints the lines
  mkschedule.py from-csv OUT CSV                                       CSV columns: work_count,knob,value[,launch]
                                                                       (one knob per row; rows with the same
                                                                       (launch, work_count) become one line)
Python API: Line, parse(text), format_lines(lines), make_lines(segments), ScheduleError.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# knob -> max value (inclusive); the policy-only [dcr_phase] fields + classification ranges (ps_schedule.h)
KNOBS: dict[str, int] = {"mode": 0xFFFFFFFF, "insert": 0xFFFFFFFF, "dram_sched": 0xFFFFFFFF,
                         "dram_aging": 0xFFFFFFFF, "range_acct_only": 1}
KNOBS.update({f"dxa_depth{c}": 0xFF for c in range(8)})
KNOBS["dxa_fdp"] = 0xFFFF   # ps patch fdp-lite: FDP-lite DXA policy epoch E (0 = off), methodology §14.C.b
KNOBS["l2_ucp"] = 0xFFF8   # ps patch a-baselines: UCP-lite flags bits 15:3 (0 = off; bit 3 required), §14.A.b
KNOBS.update({k: (1 << 64) - 1 for k in ("kv0_base", "kv0_limit", "kv1_base", "kv1_limit", "wgt_base", "wgt_limit")})
# rejected knobs and why (same messages' intent as the C++ parser)
REJECTED: dict[str, str] = {
    "l1_waymask": "a way-mask change (A2) needs a flush, not a schedule switch",
    "l2_waymask": "a way-mask change (A2) needs a flush, not a schedule switch",
    "lmem": "an LMEM change (A1) needs a kernel boundary",
    "kmu_lmem": "an LMEM change (A1) needs a kernel boundary",
    "persist_l2": "persist_l2 is a launch-boundary sensitivity knob, never scheduled",
    "flags": "persist_l2 is a launch-boundary sensitivity knob, never scheduled",
}


class ScheduleError(ValueError):
    pass


@dataclass
class Line:
    work: int
    sets: list[tuple[str, int]] = field(default_factory=list)
    launch: int | None = None   # None = every launch

    def text(self) -> str:
        head = (f"launch={self.launch} " if self.launch is not None else "") + str(self.work)
        return head + "".join(f" {k}={_fmt(k, v)}" for k, v in self.sets)


def _fmt(knob: str, v: int) -> str:
    return hex(v) if knob.endswith(("_base", "_limit")) else str(v)


def _u64(tok: str) -> int:
    if not tok or tok[0] in "+-" or not re.fullmatch(r"(0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)", tok):
        raise ValueError(tok)
    v = int(tok, 0) if not re.fullmatch(r"0[0-7]+", tok) else int(tok, 8)   # strtoull base 0: leading 0 = octal
    if v >= 1 << 64:
        raise ValueError(tok)
    return v


def check_set(knob: str, value: int, where: str = "") -> None:
    if knob in REJECTED:
        raise ScheduleError(f"{where}'{knob}' rejected: {REJECTED[knob]}")
    if knob not in KNOBS:
        raise ScheduleError(f"{where}unknown knob '{knob}'")
    if not 0 <= value <= KNOBS[knob]:
        raise ScheduleError(f"{where}value out of range for '{knob}': {value}")
    if knob == "l2_ucp" and value != 0 and (value & 0x7 or not value & 0x8):   # same rule as ps_schedule.h
        raise ScheduleError(f"{where}l2_ucp needs bit 3 set and only bits 15:3: {value:#x}")


def parse(text: str) -> list[Line]:
    out: list[Line] = []
    last = 0
    for n, raw in enumerate(text.splitlines(), 1):
        tok = raw.split("#", 1)[0].split()
        if not tok:
            continue
        where = f"line {n}: "
        launch = None
        i = 0
        if tok[0].startswith("launch="):
            try:
                launch = _u64(tok[0][7:])
            except ValueError:
                raise ScheduleError(f"{where}bad launch selector '{tok[0]}'") from None
            i = 1
        try:
            work = _u64(tok[i]) if i < len(tok) else None
        except ValueError:
            work = None
        if work is None:
            raise ScheduleError(f"{where}expected a work_count (non-negative integer)")
        if work < last:
            raise ScheduleError(f"{where}work_count decreases (lines must be in non-decreasing work_count order)")
        last = work
        if i + 1 >= len(tok):
            raise ScheduleError(f"{where}a line needs at least one knob=value")
        line = Line(work=work, launch=launch)
        for t in tok[i + 1:]:
            k, eq, v = t.partition("=")
            if not eq or not k:
                raise ScheduleError(f"{where}expected knob=value, got '{t}'")
            try:
                val = _u64(v)
            except ValueError:
                raise ScheduleError(f"{where}bad value in '{t}'") from None
            check_set(k, val, where)
            line.sets.append((k, val))
        out.append(line)
    if not out:
        raise ScheduleError("no schedule lines")
    return out


def format_lines(lines: list[Line], header: str = "") -> str:
    body = "".join(f"# {h}\n" for h in header.splitlines() if h) + "".join(l.text() + "\n" for l in lines)
    parse(body)   # never write something SimX would refuse
    return body


def make_lines(segments: list[tuple[int, dict[str, int]]], launch: int | None = None) -> list[Line]:
    return [Line(work=w, sets=list(d.items()), launch=launch) for w, d in segments]


def _seg_arg(a: str) -> tuple[int, dict[str, int]]:
    w, _, rest = a.partition(":")
    d = {}
    for kv in rest.split(","):
        k, _, v = kv.partition("=")
        d[k] = _u64(v)
    return _u64(w), d


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write"); w.add_argument("out"); w.add_argument("--launch", type=int); w.add_argument("seg", nargs="+")
    c = sub.add_parser("check"); c.add_argument("file")
    f = sub.add_parser("from-csv"); f.add_argument("out"); f.add_argument("csv")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "write":
            text = format_lines(make_lines([_seg_arg(s) for s in a.seg], a.launch), "written by scripts/mkschedule.py")
            Path(a.out).write_text(text)
        elif a.cmd == "check":
            for l in parse(Path(a.file).read_text()):
                print(l.text())
        else:
            groups: dict[tuple, Line] = {}
            for r in csv.DictReader(open(a.csv)):
                la = int(r["launch"]) if r.get("launch") not in (None, "") else None
                key = (la, int(r["work_count"]))
                groups.setdefault(key, Line(work=key[1], launch=la)).sets.append((r["knob"], _u64(r["value"])))
            lines = sorted(groups.values(), key=lambda l: l.work)
            Path(a.out).write_text(format_lines(lines, f"written by scripts/mkschedule.py from {a.csv}"))
    except (ScheduleError, ValueError) as e:
        print(f"mkschedule: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
