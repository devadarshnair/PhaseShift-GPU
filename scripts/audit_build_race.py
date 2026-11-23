#!/usr/bin/env python3
"""Flag rows of a vxrun rows.jsonl whose instrs look like a wrong-machine run (shared build-tree race, BUG-QUEUE
2026-11-14). Read-only. A row is flagged when its global PERF instrs deviates > THRESH (default 20 %) from
  (a) the median of the other rows with the same (app, args, canonical CONFIGS)  [reps / duplicates], or
  (b) the median of the same (app, args) under a different config, ONLY if the row's config differs AND the row is an
      outlier relative to that group's median by > THRESH while at least one other row sits within THRESH of the
      group median (so a legitimately config-dependent app is only reported as 'cross-config', weaker evidence).
Also flags status-ok rows with config_stamp_ok == False (vxrun's own post-run check, newer rows only).
(c) cycle twins (check, cond. 4; blind spot of the instruction test: a 4-core library runs a grid = 1 launch on core 0, so instrs
    do not move): per (app, config, base args), the newest passed ok row of each kind is compared across kinds,
      E1 cold (perf 7, `-L 1`)  ==  `-L 1 -A` twin (perf 8)             cycles equal,
      control `-L 1 -P` (perf 5) == `-L 1 -P -A` twin (perf 8) - 65,464  (65,280-cycle init walk + 184).
    Rows already superseded by a later same-key row are not used (last wins), as in results/<run>/analyze.py. Files without such twins
    (e.g. other apps) report 0 pairs checked.
Usage: audit_build_race.py ROWS.jsonl [--thresh 0.2] [--ok-only]"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--thresh", type=float, default=0.2)
    ap.add_argument("--ok-only", action="store_true", help="only consider status == ok rows")
    a = ap.parse_args(argv)
    rows = []
    for ln in open(a.rows):
        ln = ln.strip()
        if ln:
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    use = [(i, r) for i, r in enumerate(rows) if r.get("instrs") and (not a.ok_only or r.get("status") == "ok")]
    same_cfg, same_app = defaultdict(list), defaultdict(list)
    cfgkey = lambda r: tuple(r.get("configs_canonical") or [r.get("configs_effective")])
    for i, r in use:
        same_cfg[(r["app"], r["args"], cfgkey(r))].append((i, r))
        same_app[(r["app"], r["args"])].append((i, r))
    flags = []
    for i, r in use:
        v = r["instrs"]
        g = [x["instrs"] for j, x in same_cfg[(r["app"], r["args"], cfgkey(r))] if j != i]
        if g:
            m = statistics.median(g)
            if abs(v - m) / m > a.thresh:
                flags.append((i, r, "same-config", m))
                continue
        o = [x["instrs"] for j, x in same_app[(r["app"], r["args"])] if j != i and cfgkey(x) != cfgkey(r)]
        if o:
            m = statistics.median(o)
            if abs(v - m) / m > a.thresh:
                flags.append((i, r, "cross-config", m))
    for i, r in enumerate(rows):
        if r.get("config_stamp_ok") is False and r.get("status") == "ok":
            flags.append((i, r, "stamp-mismatch", None))
    INIT_WALK = 65464
    last = {}
    for i, r in enumerate(rows):
        if r.get("status") == "ok" and r.get("passed") and r.get("cycles"):
            ar = r["args"]
            base = re.sub(r"\s*-(L \d+|P|A)\b", "", ar).strip()
            m = re.search(r"-L (\d+)", ar)
            last[(r["app"], r.get("config"), r.get("perf"), base, int(m.group(1)) if m else 1,
                  bool(re.search(r"-P\b", ar)), bool(re.search(r"-A\b", ar)))] = (i, r)
    pairs = 0
    for (app, cfg, perf, base, nl, hasP, hasA), (i, r) in sorted(last.items(), key=lambda kv: kv[1][0]):
        if hasA or nl != 1:
            continue
        if perf == 7 and not hasP:
            key, off = (app, cfg, 8, base, 1, False, True), 0
        elif perf == 5 and hasP:
            key, off = (app, cfg, 8, base, 1, True, True), INIT_WALK
        else:
            continue
        if key in last:
            pairs += 1
            j, t = last[key]
            if r["cycles"] != t["cycles"] - off:
                flags.append((i, r, "cycle-twin", None))
                flags.append((j, t, "cycle-twin", None))
    print(f"cycle-twin pairs checked: {pairs}")
    print(f"{a.rows}: {len(rows)} rows, {len(use)} with instrs, {len(flags)} flagged (thresh {a.thresh:.0%})")
    for i, r, why, m in sorted(flags, key=lambda f: f[0]):
        ref = f"ref median {m:.0f} ({(r.get('instrs') or 0) / m - 1:+.0%})" if m else ""
        print(f"  line {i + 1:4d} {why:14s} {r.get('status')} {r.get('app')} {r.get('args')!r} "
              f"cfg={r.get('config') or r.get('configs_env')} rep{r.get('rep')} instrs={r.get('instrs')} {ref} "
              f"started={r.get('started')} step={r.get('step')}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
