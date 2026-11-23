"""Tests: ps patch dxa-class (mechanism C: DXA per-stream class from reserved META bits, per-class in-flight cap at
work-list dispatch, phase-dependent depth via [dcr_phase] / PS_SCHEDULE, L2-prefetch-only mode, class-tagged GMEM reads).
Semantics: docs/methodology.md §14.C; evidence: results/<run>/.

Non-vortex (cheap):
  test_patch_files            patches/vortex/ps/0009-dxa-class.patch touches exactly the 8 files, nothing in third_party/.
  test_meta_layout_static     the stock META fields end at bit 12 (RTL VX_dxa_pkg + dxa_meta.h widths sum to 13); the
                              PhaseShift fields are bits 15:13 (class) and 16 (L2-only); the RTL reads only DIM, ELEMSZ
                              and LAYOUT of the descriptor META, so bits 13..31 are free in RTL and stock SimX.
  test_stored_schedule_files  every results/<run>/sched/*.sched parses with scripts/mkschedule.py.
Vortex-marked (PS_RUN_VORTEX=1; rows of experiments/<run>.yaml in results/<run>; same specs -> same run keys -> reused):
  test_cap_every_test_run     every TEST run with a constant schedule: PSDXAC max in flight of class c <= dxa_depth[c]
                              whenever that cap is set, and the class-8 counter dxa_max_inflight_s* <= the cap of the
                              stream's class; capped runs record cap stalls; uncapped runs reach the stock 8.
  test_cap_other_class_noop   a cap on a class the stream does not use changes nothing (cycles identical).
  test_zero_schedule_noop     an explicit all-zero depth schedule equals no schedule (cycles identical).
  test_class_tag              -C 3: every DXA GMEM read is KV_STREAM (PSDXAC gmem_reqs, L2 and DRAM per-class counters),
                              none is BULK_DXA; without -C every DXA read stays BULK_DXA.
  test_l2_only                -P descriptors: 0 LMEM writes, every read's response dropped (l2only_dropped =
                              l2only_reads > 0), > 0 L2 fills of the stream's class (L2 misses, allocate-on-miss, no
                              bypass), the GMEM re-read check passes; the LMEM twin writes LMEM.
  test_functional_all_pass    every P2 kernel + the class-tagged DXA variants + the multicast apps PASS under caps.
  test_two_segment            PS_SCHEDULE on the depth knob (BULK cap 1 -> stock at 200,000 instrs, and stock -> cap 1):
                              interval-trace window max <= 1 in every row while the cap is in force, > 1 after release;
                              stock-first run identical to the static run up to the switch row.
  test_results_cap_bound      every ok RESULTS row (results/<run>/rows.jsonl, constant schedules) obeys the cap bound.
  test_noop_regression_csv    results/<run>/noop_regression.csv: every REG entry identical (base, ps 0002-0008).
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
TREE = PS / "vortex-trees" / "ps"
RES = PS / "results" / "P5.3"
PATCH = PS / "patches" / "vortex" / "ps" / "0009-dxa-class.patch"
sys.path.insert(0, str(PS / "scripts"))
sys.path.insert(0, str(PS / "tests"))
import mkschedule as mk  # noqa: E402
import pstrace  # noqa: E402
import test_dram_2class as t52  # noqa: E402  (helpers: _run)

FILES = ["sim/simx/dxa/dxa_core.cpp", "sim/simx/dxa/dxa_core.h", "sim/simx/processor.cpp", "sim/simx/processor_impl.h",
         "sim/simx/socket.cpp", "sim/simx/socket.h", "sw/common/dxa_meta.h", "sw/runtime/include/dxa.h"]
KVS, BULK = 3, 4
_run = t52._run


# ------------------------------------------------------------------------------------------------ non-vortex
def test_patch_files():
    t = PATCH.read_text()
    files = sorted(set(re.findall(r"^diff --git a/(\S+) b/", t, re.M)))
    assert files == FILES, files
    assert not re.search(r"^diff --git a/(.*/)?\.ps_", t, re.M)


def test_meta_layout_static():
    meta = TREE / "sw/common/dxa_meta.h"
    if not meta.exists():
        pytest.skip("no ps tree")
    m = meta.read_text()
    widths = [int(x) for x in re.findall(r"#define DXA_DESC_META_(?:DIM|ELEMSZ|LAYOUT|SWIZZLE|INTERLEAVE|L2PROMO)_BITS\s+(\d+)", m)]
    assert len(widths) == 6 and sum(widths) == 13
    pkg = (TREE / "hw/rtl/dxa/VX_dxa_pkg.sv").read_text()
    rw = [int(x) for x in re.findall(r"localparam DXA_DESC_META_\w+_BITS\s+= (\d+);", pkg)]
    assert rw == widths
    assert re.search(r"#define DXA_DESC_META_PS_CLASS_LSB\s+13\b", m)
    assert re.search(r"#define DXA_DESC_META_PS_CLASS_BITS\s+3\b", m)
    assert re.search(r"#define DXA_DESC_META_PS_L2ONLY_LSB\s+16\b", m)
    setup = (TREE / "hw/rtl/dxa/VX_dxa_setup.sv").read_text()
    used = set(re.findall(r"desc_data\.meta\[(DXA_DESC_META_\w+?)_LSB", setup))
    assert used == {"DXA_DESC_META_ELEMSZ", "DXA_DESC_META_DIM", "DXA_DESC_META_LAYOUT"}, used
    core = (TREE / "sim/simx/dxa/dxa_core.cpp").read_text()
    assert "mreq.flags.mem_class = w.ps_cls;" in core
    assert "if (simobject_->ps_cls_capped(w.ps_cls)) {" in core


def test_stored_schedule_files():
    files = sorted((RES / "sched").glob("*.sched"))
    assert len(files) >= 36
    for f in files:
        mk.parse(f.read_text())


# ------------------------------------------------------------------------------------------------ vortex helpers
def _batch_specs(name="P5.3.yaml"):
    import yaml
    import vxrun
    doc = yaml.safe_load((PS / "experiments" / name).read_text())
    return [{k: v for k, v in sp.items() if not k.startswith("_")}
            for sp in vxrun.expand_batch(doc, vxrun.load_named_configs()) if sp.get("rep", 0) == 0]


@pytest.fixture(scope="module")
def vx():
    import subprocess
    import vxrun
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), str(PS / "build" / "vortex-ps")],
                       capture_output=True, text=True, env=dict(os.environ, PS=str(PS)))
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun("P5.3")


def _specs(**tags):
    return [s for s in _batch_specs() if all(str((s.get("tags") or {}).get(k)) == str(v) for k, v in tags.items())]


def _one(**tags):
    got = _specs(**tags)
    assert len(got) == 1, (tags, len(got))
    return got[0]


def psdxac(txt):
    l = re.findall(r"^PSDXAC (.*)$", txt, re.M)
    assert l, "no PSDXAC line"
    d = {}
    for k, v in re.findall(r"(\w+)=(\S+)", l[-1]):
        d[k] = [int(x) for x in v.split(",")] if "," in v else int(v)
    return d


def perf(txt, name):
    m = re.findall(rf"\b{name}=(\d+)", txt)
    assert m, name
    return int(m[-1])


def depth_of(txt):
    """dxa_depth knobs of a constant schedule (the single PSSCHED line); all 0 without a schedule; None if >1 line."""
    s = re.findall(r"^PSSCHED .* dxa_depth=(\S+)", txt, re.M)
    if not s:
        return [0] * 8
    if len(s) > 1:
        return None
    return [int(x) for x in s[0].split(",")]


def stream_class(spec):
    a = spec["args"] if isinstance(spec["args"], str) else " ".join(spec["args"])
    m = re.search(r"-C (\d+)", a)
    return int(m.group(1)) if m and spec["app"] != "dxa_copy_mcast" else BULK


def check_cap(txt, cls):
    d = depth_of(txt)
    if d is None:
        return None
    x = psdxac(txt)
    for c in range(8):
        if d[c]:
            assert x["max_inflight"][c] <= d[c], (c, d, x)
    cap = d[cls]
    s8 = [perf(txt, f"dxa_max_inflight_s{s}") for s in range(4)]
    if cap and sum(x["gmem_reqs"]) == x["gmem_reqs"][cls]:   # the class-8 counter (per stream) under a one-class run
        assert max(s8) <= cap, (s8, cap)
    return d, x, s8


# ------------------------------------------------------------------------------------------------ vortex
@pytest.mark.vortex
def test_cap_every_test_run(vx):
    n = capped = 0
    for sp in _batch_specs():
        if sp["tags"].get("group") == "S" and "twoseg" in sp["tags"].get("case", ""):
            continue
        row, txt = _run(vx, sp)
        cls = stream_class(sp)
        got = check_cap(txt, cls)
        if got is None:
            continue
        d, x, s8 = got
        n += 1
        if d[cls] and x["gmem_reqs"][cls] > 0:
            capped += 1
            if d[cls] < 8:
                assert x["cap_stalls"][cls] > 0, (sp["app"], sp["args"], d, x)
                assert x["max_inflight"][cls] == d[cls], (sp["app"], sp["args"], d, x)
        elif sum(x["gmem_reqs"]) > 1000 and not any(d):
            assert x["max_inflight"][cls] == 8 and sum(x["cap_stalls"]) == 0, (sp["app"], sp["args"], x)
    assert n >= 50 and capped >= 20, (n, capped)


@pytest.mark.vortex
def test_cap_other_class_noop(vx):
    for sp in _specs(group="C", capcls=KVS) + _specs(group="C", capcls=BULK):
        cls = int(sp["tags"]["cls"])
        ref = [s for s in _specs(group="C", cls=cls, cap=0) if s["app"] == sp["app"]]
        assert len(ref) == 1
        r1, t1 = _run(vx, sp)
        r0, t0 = _run(vx, ref[0])
        assert (r1["instrs"], r1["cycles"]) == (r0["instrs"], r0["cycles"]), sp["args"]
        assert sum(psdxac(t1)["cap_stalls"]) == 0


@pytest.mark.vortex
def test_zero_schedule_noop(vx):
    for sp in _specs(group="Z"):
        ref = [s for s in _specs(group="C", cls=BULK, cap=0) if s["app"] == sp["app"]]
        r1, _ = _run(vx, sp)
        r0, _ = _run(vx, ref[0])
        assert (r1["instrs"], r1["cycles"]) == (r0["instrs"], r0["cycles"]), sp["app"]


@pytest.mark.vortex
def test_class_tag(vx):
    names = {KVS: "kvs", BULK: "dxa"}
    for cls in (KVS, BULK):
        for sp in _specs(group="C", cls=cls, cap=0):
            _, txt = _run(vx, sp)
            x = psdxac(txt)
            other = KVS + BULK - cls
            assert x["gmem_reqs"][cls] > 0 and x["gmem_reqs"][other] == 0, x
            assert perf(txt, f"l2_acc_{names[cls]}") >= x["gmem_reqs"][cls]
            assert perf(txt, f"dram_rd_bytes_{names[cls]}") > 0
            assert perf(txt, f"l2_acc_{names[other]}") == 0   # no KV ranges programmed: only the DXA tag gives kvs
            if cls == KVS:
                assert perf(txt, "l2_acc_dxa") == 0 and perf(txt, "dram_rd_bytes_dxa") == 0


@pytest.mark.vortex
def test_l2_only(vx):
    n = 0
    for sp in _specs(group="L"):
        row, txt = _run(vx, sp)
        assert re.search(r"^PASSED!?\s*$", txt, re.M), row["log"]
        assert "bulk_mismatches=0" in txt
        x = psdxac(txt)
        a = sp["args"]
        if "-P" in a:
            cls = KVS if "-C 3" in a else BULK
            nm = "kvs" if cls == KVS else "dxa"
            assert x["lmem_writes"] == 0, x
            assert x["l2only_reads"] == x["l2only_dropped"] == x["gmem_reqs"][cls] > 0, x
            assert perf(txt, f"l2_miss_{nm}") > 0 and perf(txt, f"l2_fill_bypassed_{nm}") == 0
            n += 1
        else:
            assert x["lmem_writes"] > 0 and x["l2only_reads"] == 0, x
    assert n >= 7


@pytest.mark.vortex
def test_functional_all_pass(vx):
    specs = _specs(group="F")
    assert len(specs) >= 20
    for sp in specs:
        row, txt = _run(vx, sp)
        assert re.search(r"^PASSED!?\s*$", txt, re.M), row["log"]


def _trace_max(rows):
    return [max(r[f"dxa_max_inflight_s{s}"] for s in range(4)) for r in rows]


@pytest.mark.vortex
def test_two_segment(vx):
    W = 200000
    st, sc = _one(group="S", case="static"), _one(group="S", case="static_cap1")
    t1, t2 = _one(group="S", case="twoseg_cap1"), _one(group="S", case="twoseg_cap0to1")
    out = {}
    for sp in (st, sc, t1, t2):
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt
        _, rows = pstrace.read_trace(sp["env"]["PS_TRACE"])
        out[sp["tags"]["case"]] = (row, pstrace.work_rows(rows, 0))
    ms, mc = _trace_max(out["static"][1]), _trace_max(out["static_cap1"][1])
    assert max(mc) <= 1 and max(ms) > 1
    w1 = out["twoseg_cap1"][1]
    k = next(i for i, r in enumerate(w1) if r["work_end"] >= W)
    m1 = _trace_max(w1)
    assert max(m1[:k + 1]) <= 1, m1[:k + 1]          # cap 1 in force up to (and in) the switch row
    assert max(m1[k + 1:]) > 1, m1[k + 1:]           # released after
    w2 = out["twoseg_cap0to1"][1]
    k2 = next(i for i, r in enumerate(w2) if r["work_end"] >= W)
    for i in range(k2 + 1):
        assert w2[i] == out["static"][1][i], i      # stock prefix identical to the static run
    m2 = _trace_max(w2)
    assert max(m2[k2 + 2:]) <= 1 and max(m2[:k2 + 1]) > 1, m2
    cyc = {c: out[c][0]["cycles"] for c in out}
    assert len(set(cyc.values())) == 4, cyc


@pytest.mark.vortex
def test_results_cap_bound():
    p = RES / "rows.jsonl"
    n = 0
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("status") != "ok" or (r.get("tags") or {}).get("group") not in ("R", "B", "M"):
            continue
        txt = (PS / r["log"]).read_text(errors="replace")
        assert "PASSED" in txt, r["log"]
        cls = int(r["tags"].get("cls", BULK))
        if check_cap(txt, cls) is not None:
            n += 1
    assert n >= 100, n


@pytest.mark.vortex
def test_noop_regression_csv():
    rows = list(csv.DictReader((RES / "noop_regression.csv").open()))
    assert len(rows) >= 8 and all(r["identical"] == "True" for r in rows), rows
