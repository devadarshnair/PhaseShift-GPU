"""Tests: ps patch a-baselines -- UCP-lite [C40], a phase-ignorant utility-based L2 way-partitioning baseline for
mechanism A2 (semantics: docs/methodology.md §14.A.b; evidence: results/<run>/).

Non-vortex (cheap):
  test_patch_files              patches/vortex/ps/0012-a-baselines.patch touches exactly the 5 SimX files.
  test_schedule_knob            mkschedule.py accepts l2_ucp (bit 3 + bits 15:3) and rejects malformed words; the stored
                                reject_bad_word schedule fails to parse; flags stays rejected.
  test_lookahead_reference      the independent Python Lookahead (used to replay the trace) on hand cases.
  test_no_phase_input          : the UCP-lite code reads no mode / phase / schedule / mark / range input;
                                of PhaseConfig it uses only l2_ucp(), ucp_epoch(), ucp_class().
  test_static_masks_rerun_identical  the earlier whole RESULTS sweep re-run on the binary equals the earlier rows.
  test_noop_regression_csv      results/<run>/noop_regression.csv: every REG entry identical (base, ps 0002-0011).
Vortex-marked (PS_RUN_VORTEX=1; rows of experiments/<run>.yaml in results/<run>; same specs -> same run keys -> reused):
  test_epoch_trace              every PSUCP trace: the partition sums to the way count, gives each application >= 1 way,
                                changes only at epoch boundaries start + k*N (consecutive epochs, old == previous new),
                                and the new partition equals an INDEPENDENT Python replay of the Lookahead on the printed
                                pre-halving counters; counters are consistent with halving; violations = 0; PASSED.
  test_two_segment_schedule     UCP-lite switched on / off by a PS_SCHEDULE line inside a launch (no flush): no trace
                                before the on-switch, no epoch after the off-switch, PASSED.
  test_rejections               UCP-lite + l2_waymask, + insert policy, malformed word, class >= 8: the run fails.
"""
import csv
import json
import re
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
TREE = PS / "vortex-trees" / "ps"
RES = PS / "results" / "P5.4b"
PATCH = PS / "patches" / "vortex" / "ps" / "0012-a-baselines.patch"
sys.path.insert(0, str(PS / "scripts"))
sys.path.insert(0, str(PS / "tests"))
import mkschedule as mk  # noqa: E402
import test_dram_2class as t52  # noqa: E402  (helpers: _run)

FILES = ["sim/common/phase_config.h", "sim/simx/mem/cache.cpp", "sim/simx/mem/cache.h", "sim/simx/processor.cpp",
         "sim/simx/ps_schedule.h"]
WAYS = 8
_run = t52._run


# ------------------------------------------------------------------------------------------------ reference model
def lookahead(hA, hB, ways=WAYS):
    """Independent re-implementation of the rule in methodology §14.A.b (paper Algorithm 2 + one way per app)."""
    h = (list(hA), list(hB))
    out = [1, 1]
    bal = ways - 2

    def gain(a, m, n):
        return sum(h[a][m:n])

    while bal > 0:
        best = []
        for a in (0, 1):
            bg, bk = 0, 0
            for k in range(1, bal + 1):
                g = gain(a, out[a], out[a] + k)
                if g > 0 and (bk == 0 or g * bk > bg * k):
                    bg, bk = g, k
            best.append((bg, bk))
        (g0, k0), (g1, k1) = best
        if k0 == 0 and k1 == 0:
            out[0] += bal - bal // 2
            out[1] += bal // 2
            break
        if k1 == 0:
            win = 0
        elif k0 == 0:
            win = 1
        else:
            win = 1 if g1 * k0 > g0 * k1 else 0
        out[win] += best[win][1]
        bal -= best[win][1]
    return out


def ucp_trace(txt):
    """List of launches; each = {'start': dict, 'epochs': [dict], 'end': dict|None} from PSUCP lines, in order."""
    segs, cur = [], None
    for line in re.findall(r"^PSUCP (.*)$", txt, re.M):
        kind = line.split()[0]
        d = {}
        for k, v in re.findall(r"(\w+)=(\S+)", line):
            if k in ("hitsA", "hitsB"):
                d[k] = [int(x) for x in v.split(",")]
            elif k in ("alloc", "old", "new", "acc"):
                d[k] = [int(x) for x in v.split(":")]
            else:
                d[k] = int(v)
        if kind == "start":
            cur = {"start": d, "epochs": [], "end": None}
            segs.append(cur)
        elif kind.startswith("epoch"):
            assert cur is not None and cur["end"] is None, line
            cur["epochs"].append(d)
        elif kind == "end":
            assert cur is not None and cur["end"] is None, line
            cur["end"] = d
    return segs


def check_segment(seg):
    st = seg["start"]
    assert sum(st["alloc"]) == st["ways"] == WAYS
    prev = st["alloc"]
    prev_h = None
    for k, e in enumerate(seg["epochs"]):
        assert e["epoch"] == k
        assert e["start"] == st["start"] and e["len"] == st["len"]
        assert e["cycle"] == st["start"] + (k + 1) * st["len"], (k, e["cycle"])   # only at epoch boundaries
        assert e["old"] == prev, (k, e["old"], prev)                             # no change between boundaries
        assert sum(e["new"]) == WAYS and min(e["new"]) >= 1, e["new"]
        assert e["new"] == lookahead(e["hitsA"], e["hitsB"]), (k, e)              # independent replay
        if prev_h is not None:   # counters were halved at the previous boundary, then only incremented
            for a in ("hitsA", "hitsB"):
                assert all(x >= y // 2 for x, y in zip(e[a], prev_h[a])), (k, a)
        prev, prev_h = e["new"], e
    if seg["end"] is not None:
        en = seg["end"]
        assert en["violations"] == 0
        assert en["epochs"] == len(seg["epochs"])
        assert en["alloc"] == prev
        assert en["cycle"] < st["start"] + (len(seg["epochs"]) + 1) * st["len"]
    return len(seg["epochs"])


# ------------------------------------------------------------------------------------------------ cheap
def test_patch_files():
    assert PATCH.exists()
    got = sorted(set(re.findall(r"^diff --git a/(\S+) b/", PATCH.read_text(), re.M)))
    assert got == sorted(FILES), got


def test_schedule_knob():
    assert mk.parse("0 l2_ucp=0x108\n")[0].sets == [("l2_ucp", 0x108)]
    mk.parse("0 l2_ucp=0x3048\n100 l2_ucp=0\n")
    for bad in ("0x10", "0x109", "0x10008"):
        with pytest.raises(mk.ScheduleError):
            mk.parse(f"0 l2_ucp={bad}\n")
    with pytest.raises(mk.ScheduleError):
        mk.parse((RES / "sched" / "reject_bad_word.sched").read_text())
    with pytest.raises(mk.ScheduleError):
        mk.parse("0 flags=8\n")
    src = (TREE / "sim/simx/ps_schedule.h").read_text()
    assert 'k == "l2_ucp"' in src and "l2_ucp needs bit 3 set" in src


def test_lookahead_reference():
    z = [0] * WAYS
    assert lookahead(z, z) == [4, 4]                                   # no utility: even split
    assert lookahead([5, 0, 0, 0, 0, 0, 0, 0], z) == [4, 4]           # only MRU hits: nothing beyond way 1
    assert lookahead([1, 0, 0, 9, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0, 0]) == [6, 2]   # lookahead sees the 4th way
    assert lookahead(z, [0, 1, 1, 1, 1, 1, 1, 1]) == [1, 7]
    assert lookahead([0, 4, 4, 0, 0, 0, 0, 0], [0, 3, 3, 3, 0, 0, 0, 0]) == [4, 4]   # 3:4 by utility, the zero-utility leftover to A
    for a, b in (([0, 9, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 9]), ([7] * 8, [7] * 8)):
        r = lookahead(a, b)
        assert sum(r) == WAYS and min(r) >= 1


def _ucp_source():
    """The UCP-lite code: struct PsUcp, the bank's ps_ucp_* members, the victim branch and the processor check."""
    src = (TREE / "sim/simx/mem/cache.cpp").read_text()
    body = src[src.index("struct PsUcp {"):src.index("class CacheBank : public SimObject<CacheBank> {")]
    i = src.index("void ps_set_ucp(PsUcp *u)")
    body += src[i:src.index("const Cache::PerfStats &perf_stats() const", i)]
    i = src.index("if (wm_word == 0 && this->ps_ucp_on())")
    body += src[i:src.index("} else if (wm_word == 0) {", i)]
    return body


def test_no_phase_input():
    body = re.sub(r"//[^\n]*", "", _ucp_source())
    for bad in ("mode", "phase_id", "ps_mark", "sched", "kv[", "wgt", "range", "ins_", "dxa", "launch", "PS_PHASE"):
        assert bad not in body, bad
    used = set(re.findall(r"(?:cfg|pc)\.(\w+)\(", body))
    assert used <= {"l2_ucp", "ucp_epoch", "ucp_class"}, used
    # the only external inputs: the request class, the L2's own accesses / lines, the cycle count
    assert "SimPlatform::instance().cycles()" in body


def _rows(step):
    p = PS / "results" / step / "rows.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def test_static_masks_rerun_identical():
    mine = [r for r in _rows("P5.4b") if r.get("status") == "ok" and (r.get("tags") or {}).get("rerun") == "P5.4"]
    if not mine:
        pytest.skip("RESULTS re-run not yet produced")
    ref = {}
    for r in _rows("P5.4"):
        if r.get("status") == "ok":
            ref.setdefault((r["app"], r["args"], r["config"], json.dumps(r.get("env") or {}, sort_keys=True)),
                           set()).add((r["instrs"], r["cycles"]))
    n = 0
    for r in mine:
        k = (r["app"], r["args"], r["config"], json.dumps(r.get("env") or {}, sort_keys=True))
        assert k in ref, k
        assert ref[k] == {(r["instrs"], r["cycles"])}, (k, ref[k], r["instrs"], r["cycles"])
        n += 1
    assert n == 83, n   # every spec of experiments/<run>.yaml (45 decode + 20 mixed + 12 wkv + 6 wkv_l1)


def test_noop_regression_csv():
    p = RES / "noop_regression.csv"
    if not p.exists():
        pytest.skip("results/P5.4b/noop_regression.csv not yet produced")
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 8 and all(r["identical"] == "True" for r in rows)


# ------------------------------------------------------------------------------------------------ vortex
def _batch_specs(name="P5.4b.yaml"):
    import yaml
    import vxrun
    doc = yaml.safe_load((PS / "experiments" / name).read_text())
    return [{k: v for k, v in sp.items() if not k.startswith("_")}
            for sp in vxrun.expand_batch(doc, vxrun.load_named_configs()) if sp.get("rep", 0) == 0]


@pytest.fixture(scope="module")
def vx():
    import os
    import subprocess
    import vxrun
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), str(PS / "build" / "vortex-ps")],
                       capture_output=True, text=True, env=dict(os.environ, PS=str(PS)))
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun("P5.4b")


def _specs(**tags):
    return [s for s in _batch_specs() if all(str((s.get("tags") or {}).get(k)) == str(v) for k, v in tags.items())]


@pytest.mark.vortex
def test_epoch_trace(vx):
    specs = _specs(group="U") + _specs(group="F")
    assert len(specs) == 10
    total = 0
    for sp in specs:
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt, sp["args"]
        segs = ucp_trace(txt)
        assert len(segs) >= 1 and all(s["end"] is not None for s in segs), sp["args"]
        for s in segs:
            total += check_segment(s)
            assert s["end"]["fills"] > 0
    assert total >= 100, total   # non-vacuous: many repartitions replayed


@pytest.mark.vortex
def test_two_segment_schedule(vx):
    for case in ("off_on", "on_off"):
        (sp,) = _specs(group="S", case=case)
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt
        sw = [int(c) for c in re.findall(r"^PSSCHED launch=\d+ line=\d+ work_count=\d+ work=\d+ cycle=(\d+)", txt, re.M)]
        segs = ucp_trace(txt)
        assert len(segs) == 1, case
        for s in segs:
            check_segment(s)
        s = segs[0]
        if case == "off_on":
            assert len(sw) == 1 and s["start"]["cycle"] >= sw[0] and s["epochs"], (sw, s["start"])
        else:
            assert len(sw) == 2 and s["start"]["cycle"] >= sw[0]
            assert all(e["cycle"] <= sw[1] for e in s["epochs"]) and s["epochs"]
            assert sw[1] <= s["end"]["cycle"] < sw[1] + s["start"]["len"], (sw, s["end"])


@pytest.mark.vortex
def test_rejections(vx):
    want = {"ucp_plus_mask": "exclusive with a non-zero l2_waymask",
            "ucp_plus_insert": "exclusive with a non-zero L2 insert policy",
            "bad_word": "l2_ucp needs bit 3 set",
            "bad_class": "class A >= the number of classes"}
    specs = _specs(group="X")
    assert {s["tags"]["case"] for s in specs} == set(want)
    for sp in specs:
        row, txt = _run(vx, sp, expect_ok=False)
        assert row["status"] != "ok", sp["tags"]
        assert want[sp["tags"]["case"]] in txt, (sp["tags"], row["log"])
        assert "PASSED!" not in txt
