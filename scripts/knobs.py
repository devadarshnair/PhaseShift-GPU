#!/usr/bin/env python3
"""knobs.py -- the PhaseShift knob space K and the reduced ablation set (docs/methodology.md §14.E).

  knobs.py validate [FILE]                  every value of K, every ablation vector x P2 kernel, every probe: legality
                                            as implemented in SimX (ps patches 0001-0013); exit 0 iff consistent
  knobs.py resolve VECTOR KERNEL_ID [--half N]   print the app args / env / PS_SCHEDULE lines of one cell
Python API (used by results/<run>/make_batch.py, results/<run>/compat.py, scripts/noop_regression.py --all, P6):
  load(path) -> doc;  kernels(doc) -> list;  resolve(doc, vector, kernel, half=None) -> Cell;  legality(cfg) -> str
  schedule_text(cell) -> str (mkschedule.py format; '' when the cell needs no schedule)

Modeling choices (why the cells look the way they do) are in methodology §14.E; the rules mirrored here are cited next to
each check. A knob vector is a full assignment of the mechanisms' knobs; an ablation vector is a *policy*: per launch,
the vector's mechanisms take their per-phase setting from `phase_maps` for the launch's phase label (ideal / practical
detector) or from the oracle hook (oracle). `resolve` turns (vector, kernel) into the concrete knob vector of that cell.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

PS = Path(os.environ.get("PS", Path(__file__).resolve().parents[1]))
DEFAULT = PS / "experiments" / "knobs.yaml"
MECHS = ("A", "B", "C", "D")
POLICY_FIELDS = ("insert", "dram_sched", "dram_aging", "dxa_depth4", "l2_ucp", "dxa_fdp")

# SimX encodings (sim/common/phase_config.h, ps_dram_b.h) -- mirrored, not imported
INS_D_MASK, INS_RTL, INS_WINDOW, INS_REPL = 0x3, 0x10, 0x100, 0x60
INS_VALID = 0x1fff113 | INS_REPL


def load(path: Path | str = DEFAULT) -> dict:
    import yaml
    doc = yaml.safe_load(Path(path).read_text())
    for k in doc["kernels"]:
        k["args"] = k["args"].replace("$PS", str(PS))
    return doc


def kernels(doc: dict) -> list[dict]:
    return doc["kernels"]


def kernel(doc: dict, kid: str) -> dict:
    for k in doc["kernels"]:
        if k["id"] == kid:
            return k
    raise KeyError(kid)


@dataclass
class Cell:
    vector: str
    kernel: str
    app: str
    args: str                                   # the kernel's args + the vector's host args
    segments: list = field(default_factory=list)   # [(work, {knob: value})] for PS_SCHEDULE (empty = no schedule)
    launch: dict = field(default_factory=dict)      # resolved launch-boundary knobs {A1, A2, ranges}
    applied: dict = field(default_factory=dict)     # per mechanism: what acts in this cell ("off", value, "n/a ...")
    expect_abort: str = ""                          # probes: the SimX message substring (rejected by design)
    probe: bool = False

    def policy_cfg(self, i: int = -1) -> dict:
        """Effective policy fields after segment i (default: the last), starting from reset (0)."""
        cfg = dict.fromkeys(POLICY_FIELDS, 0)
        for _, s in self.segments[: (len(self.segments) if i < 0 else i + 1)]:
            cfg.update(s)
        return cfg


def _mask_words(doc: dict, a2: str) -> tuple[int, int]:
    w = doc["knobs"]["A2"]["values"][a2]
    l1, l2 = w.split(":")
    return int(l1, 0), int(l2, 0)


def legality(cfg: dict) -> str:
    """'' if SimX accepts the config, else the abort reason (substring of the SimX message). cfg keys: l1, l2, insert,
    dram_sched, dram_aging, dxa_depth4, l2_ucp, dxa_fdp (missing = 0). Order = SimX's: ps_waymask_check at the launch /
    schedule start (processor.cpp), then the push into the L2 (cluster.cpp insert_policy_error), then DramSim."""
    g = lambda k: int(cfg.get(k, 0) or 0)  # noqa: E731
    l1, l2, ins, ucp, ds = g("l1"), g("l2"), g("insert"), g("l2_ucp"), g("dram_sched")
    if ucp:   # patch a-baselines
        if l2:
            return "exclusive with a non-zero l2_waymask"
        if ins:
            return "exclusive with a non-zero L2 insert policy"
    for word, ways, lvl in ((l1, 4, "l1"), (l2, 8, "l2")):   # PS-BASE-S1-4c associativities (§14.A)
        if (word & 0xffff) >> ways or (word >> 16) >> ways:
            return f"{lvl} mask bit >= associativity"
    if l2 and (ins & ~INS_D_MASK):   # patch compose: only D's own modes compose with an L2 mask
        return "exclusive with an L2 insert policy other than D modes 1/2"
    if ins:   # patch kv-insert / d-baselines: PhaseConfig::insert_policy_error
        if ins & ~INS_VALID:
            return "reserved bits set"
        if ins & INS_D_MASK == 3:
            return "D mode 3 is reserved"
        if ins & INS_WINDOW and ins & INS_D_MASK:
            return "window baseline and D are exclusive"
        if not ins & INS_WINDOW and ins & 0x1fff000:
            return "window fields set without the window bit"
        if ins & INS_REPL == INS_REPL:
            return "replacement baseline 3 is reserved"
        if ins & INS_REPL and ins & (INS_D_MASK | INS_RTL | INS_WINDOW):
            return "SRRIP / SHiP-Mem-lite are exclusive with D, rtl_like_fill and the window"
    if ds:   # patch dram-2class / b-baselines (§14.B, §14.B.b)
        mode, pol = ds & 0x3, (ds >> 4) & 0xF
        if ds >> 16 or ds & 0xC:
            return "dram_sched reserved bits"
        if mode == 3:
            return "dram_sched mode 3"
        if pol and mode:
            return "dram_sched baseline policy with mode bits"
        if pol > 2:
            return "baseline policy reserved"
        if not mode and not pol and ds & 0xFF00:
            return "dram_sched cap with mode 0"
    if g("dxa_depth4") > 0xFF:
        return "dxa_depth out of range"
    return ""


def _phase_setting(doc: dict, mech: str, phase: str):
    return (doc["phase_maps"].get(mech) or {}).get(phase)


def resolve(doc: dict, vector: str, k: dict, half: int | None = None) -> Cell:
    """Concrete knob vector of (ablation vector or probe, kernel). half = the kernel's static instruction count // 2
    (oracle hook_check only; None -> the segment switch is omitted and noted)."""
    kn = doc["knobs"]
    app, phase = k["app"], k["phase"]
    cell = Cell(vector, k["id"], app, k["args"])
    a1, a2, ranges = "large", "none", False
    pol: list[tuple[int, dict]] = []
    if vector in doc.get("probes", {}):
        pr = doc["probes"][vector]
        a2 = (pr.get("launch") or {}).get("A2", "none")
        pol = [(0, dict(pr["set"]))]
        cell.expect_abort, cell.probe = pr["abort"], True
        cell.applied = {"A": f"A2={a2}", "B": "-", "C": "-", "D": "-", "probe": str(pr["set"])}
    else:
        v = doc["ablation"][vector]
        mechs = v["mechanisms"]
        if v["detector"] == "oracle":
            hc = v["hook_check"]
            a1, a2, ranges = hc["launch"]["A1"], hc["launch"]["A2"], bool(hc["launch"].get("ranges"))
            for seg in hc["segments"]:
                at = seg["at"]
                if at == "half_static_instrs":
                    at = hc["llama_at"] if app == "ps_llama" else half
                if at is None:
                    continue
                s = {}
                for m in ("B", "C", "D"):
                    s.update(kn[m]["values"][seg[m]])
                pol.append((int(at), s))
            cell.applied = {m: "hook_check" for m in MECHS}
        else:
            s0: dict = {}
            for m in MECHS:
                st = _phase_setting(doc, m, phase) if m in mechs else None
                if m == "A":
                    st = st or {}
                    a1, a2 = st.get("A1", "large"), st.get("A2", "none")
                    cell.applied["A"] = f"A1={a1},A2={a2}" if st else "off"
                elif st is None:
                    cell.applied[m] = "off"
                else:
                    s0.update(kn[m]["values"][st])
                    cell.applied[m] = st
                    if m == "D":
                        ranges = True
            if s0:
                pol = [(0, s0)]
    # launch-boundary knobs -> host args
    extra = []
    if a1 != "large":
        if app in kn["A1"]["arg"]:
            extra.append(f"{kn['A1']['arg'][app]} {kn['A1']['values'][a1]}")
        else:
            cell.applied["A1_note"] = "n/a: app has no tile variant"
    if a2 != "none":
        extra.append(f"{kn['A2']['arg']} {kn['A2']['values'][a2]}")
    if ranges:
        if app in kn["D"]["range_arg"]:
            extra.append(kn["D"]["range_arg"][app])
        else:
            cell.applied["D_note"] = "inert: app has no KV range option (no KV_STREAM class)"
    cell.args = " ".join([k["args"]] + extra)
    cell.segments = pol
    cell.launch = {"A1": a1, "A2": a2, "ranges": ranges}
    if vector == "oracle" and half is None and app != "ps_llama":
        cell.applied["oracle_note"] = "segment switch omitted (static instrs unknown)"
    return cell


def cell_errors(doc: dict, cell: Cell) -> list[str]:
    """Abort reasons SimX would give for this cell (one per segment that is illegal)."""
    l1, l2 = _mask_words(doc, cell.launch["A2"])
    errs = []
    for i in range(max(1, len(cell.segments))):
        cfg = dict(cell.policy_cfg(i) if cell.segments else {}, l1=l1, l2=l2)
        e = legality(cfg)
        if e:
            errs.append(e)
    return errs


def schedule_text(doc: dict, cell: Cell) -> str:
    if not cell.segments:
        return ""
    out = [f"# GENERATED by scripts/knobs.py for vector {cell.vector} x kernel {cell.kernel}; "
           f"unit = PS_SCHEDULE_UNIT={doc['schedule_unit']}"]
    for w, s in cell.segments:
        out.append(f"{w} " + " ".join(f"{kk}={vv}" for kk, vv in s.items()))
    return "\n".join(out) + "\n"


def validate(doc: dict) -> list[str]:
    errs = []
    kn = doc["knobs"]
    for m in ("B", "C", "D"):
        for name, s in kn[m]["values"].items():
            e = legality(s)
            if e:
                errs.append(f"K {m}.{name}: {e}")
    for name in kn["A2"]["values"]:
        l1, l2 = _mask_words(doc, name)
        e = legality({"l1": l1, "l2": l2})
        if e:
            errs.append(f"K A2.{name}: {e}")
    # every K member of A2 x D composes (patch 0013), the cross product is legal
    for a2 in kn["A2"]["values"]:
        for d in kn["D"]["values"].values():
            l1, l2 = _mask_words(doc, a2)
            e = legality(dict(d, l1=l1, l2=l2))
            if e:
                errs.append(f"K A2.{a2} x D {d}: {e}")
    want = {"static", "A", "B", "C", "D", "A+B", "all-ideal", "full-practical", "oracle"}
    if set(doc["ablation"]) != want:
        errs.append(f"ablation set {sorted(doc['ablation'])} != {sorted(want)}")
    for v in doc["ablation"]:
        for k in doc["kernels"]:
            c = resolve(doc, v, k, half=1000)
            for e in cell_errors(doc, c):
                errs.append(f"{v} x {k['id']}: illegal ({e})")
    for p, pr in doc.get("probes", {}).items():
        for k in doc["kernels"]:
            c = resolve(doc, p, k)
            got = cell_errors(doc, c)
            if not got or pr["abort"] not in got[0]:
                errs.append(f"probe {p} x {k['id']}: expected abort '{pr['abort']}', model gives {got}")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("file", nargs="?", default=str(DEFAULT))
    r = sub.add_parser("resolve")
    r.add_argument("vector")
    r.add_argument("kernel")
    r.add_argument("--half", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        doc = load(a.file)
        errs = validate(doc)
        for e in errs:
            print("knobs: ERROR", e)
        n = len(doc["ablation"]) * len(doc["kernels"])
        print(f"knobs: {'OK' if not errs else 'FAIL'} ({n} ablation cells, "
              f"{len(doc.get('probes', {})) * len(doc['kernels'])} probe cells)")
        return 0 if not errs else 1
    doc = load()
    c = resolve(doc, a.vector, kernel(doc, a.kernel), a.half)
    print("args:", c.args)
    print("applied:", c.applied)
    print("errors:", cell_errors(doc, c))
    print(schedule_text(doc, c), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
