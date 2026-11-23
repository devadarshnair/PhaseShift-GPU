"""Tests: software weight-streaming GEMV variants (methodology §18.8).

Non-vortex (always run; no SimX):
  test_stock_sources_untouched         the stock device sources (gemv_f32/kernel.cpp, gemv_q80/kernel.cpp, linear/kernel.cpp) are byte-identical
                                       to their content (sha256), the new entries live in translation units of their own, and
                                       ps_llama's module (runtime/llama_vx) neither lists nor reaches any of them
  test_new_entry_layout                app.conf of the three apps lists the extra TU after the stock one; every new `__kernel` is in
                                       the new files only; ps_twolaunch reaches the same files by symlink
  test_batch_files_up_to_date          experiments/<run>_*.yaml == results/<run>/make_batch.py output
  test_analysis_up_to_date             results/<run>/*.csv == results/<run>/analyze.py output
  test_dxa_beat_model                  the host's predicted LMEM write beats / GMEM reads of a DXA transfer (the enumeration rule of dxa_core.cpp)
                                       agree with a direct re-implementation here on the shapes of the tests
  test_ep3_results                     results/<run>/rows.jsonl: every used row ok and passed, every PSCHECK clean, Q8_0 kernels bit-exact vs runq order
                                       (PSBITEXACT), equal output digests across all Q8_0 kernels and the DXA kernel for the same inputs
  test_ep3_stock_identity              stock entries re-run here == rows (cycles and instructions; B = 1)
  test_dxa_counters_match_prediction   DXA transfers / LMEM write beats == the host prediction; predicted distinct reads == gmem_reads - gmem_dedup
  test_dxa_cycles_vs_non_dxa           DXA cycles below the best non-DXA kernel at the largest K, or the reason recorded in the README
  test_persist_pairs_in_stored_results launch-2 L2 read misses <= 5 % of launch 1 in every stored pair (res.csv), every app present
  test_overfetch_diagnosis_files       diag2.csv: touch kernels read ~1.0x Wq from DRAM; stack share attribution exists
  test_methodology_18_8                modeling choices documented

Vortex-marked (PS_RUN_VORTEX=1; each SimX run via scripts/vxrun.py -> lowload.sh, serial, private results dir build/tmp/ep3_test, deleted at
session start; lowload exit 75 -> skip "DEFERRED"):
  test_shfl_semantics                  segmented butterfly and shfl_idx on SimX == the model of alu_unit.cpp
  test_f32_variant / test_q80_variant / test_dxa_variant / test_linear_tracks   live correctness runs at E1 and E4 (NT = 32; the DXA Q8_0 kernel
                                       needs GS / 4 <= NT, so REG-CFG's NT = 4 is not used); odd shapes (M not a multiple of 8, B = 3, GS 16 / 128)
  test_persist_pairs                   launch-2 L2 read misses <= 5 % of launch 1 and every launch produces its digest, for the new apps
"""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS))
from model import ref_q80 as q  # noqa: E402

RES = PS / "results" / "EP.3"
EP0 = PS / "results" / "EP.0"
EP2 = PS / "results" / "EP.2"
F32 = np.float32
STEM_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) tol=(\S+) errors=(\d+)", re.M)
SHAPES = [  # (model, matrix, M, K) = results/<run>/make_batch.py SHAPES
    ("stories15M", "qkvo", 288, 288), ("stories15M", "w13", 768, 288), ("stories15M", "w2", 288, 768), ("stories15M", "cls", 32000, 288),
    ("llama8b_s8", "qo", 512, 512), ("llama8b_s8", "kv", 128, 512), ("llama8b_s8", "w13", 1792, 512),
    ("llama8b_s8", "w2", 512, 1792), ("llama8b_s8", "cls", 16032, 512)]
# sha256 of the stock device sources as they were when started (= the earlier final content; the entries are NOT in these files)
STOCK_SHA = {
    "kernels/gemv_f32/kernel.cpp": None,
    "kernels/gemv_q80/kernel.cpp": None,
    "kernels/linear/kernel.cpp": None,
}


def sha(p):
    return hashlib.sha256((PS / p).read_bytes()).hexdigest()


def gs_of(model):
    return 32 if model == "stories15M" else 64


# --------------------------------------------------------------------------------------------- source layout
def test_stock_sources_untouched():
    rec = json.loads((RES / "stock_sha256.json").read_text())
    for p in STOCK_SHA:
        assert sha(p) == rec[p], p
    # no new entry in a stock file
    for p, names in (("kernels/gemv_f32/kernel.cpp", ("_ilp",)), ("kernels/gemv_q80/kernel.cpp", ("_ilp", "_inl", "touch")),
                     ("kernels/linear/kernel.cpp", ("i8", "ilp"))):
        txt = (PS / p).read_text()
        assert not any(re.search(rf"__kernel void \w*{n}", txt) for n in names), p
    # ps_llama's module: its translation units are the nine/ten; none of files is in its app.conf, and its reused links
    # still point at the stock kernel.cpp files
    conf = (PS / "runtime/llama_vx/app.conf").read_text()
    for f in ("kernel_ilp", "kernel_sw", "gemv_dxa"):
        assert f not in conf
    for k in ("gemv_f32", "gemv_q80", "linear"):
        link = PS / f"runtime/llama_vx/reused/{k}/kernel.cpp"
        assert link.is_symlink() and os.readlink(link).endswith(f"kernels/{k}/kernel.cpp")


def test_new_entry_layout():
    for d, extra in (("gemv_f32", "kernel_ilp.cpp"), ("gemv_q80", "kernel_sw.cpp")):
        conf = (PS / f"kernels/{d}/app.conf").read_text()
        # layout stays pinned as the FIRST two sources; later additions (kernel_nsp.cpp, kernel_tile.cpp) may append
        assert re.search(rf"^VX_SRCS=kernel\.cpp {re.escape(extra)}(\s+\S+\.cpp)*\s*$", conf, re.M), (d, conf)
        assert (PS / f"kernels/{d}/{extra}").is_file()
    for f in ("app.conf", "kernel_ilp.cpp", "kernel.cpp", "gemv_host.h", "common.h"):
        assert (PS / "kernels/ps_twolaunch" / f).is_symlink(), f
    assert (PS / "kernels/gemv_dxa/kernel.cpp").is_file() and (PS / "kernels/gemv_dxa/main.cpp").is_file()
    ilp = (PS / "kernels/gemv_f32/kernel_ilp.cpp").read_text()
    assert sorted(re.findall(r"__kernel void (\w+)", ilp)) == ["gemv_lanes_ilp2", "gemv_rows_ilp2", "gemv_rows_ilp4"]
    sw = (PS / "kernels/gemv_q80/kernel_sw.cpp").read_text()
    assert sorted(re.findall(r"__kernel void (\w+)", sw)) == sorted(
        ["gemv_q80_rows_inl", "gemv_q80_rows_ilp2", "gemv_q80_rows_ilp4", "q80_touch_rows", "q80_touch_rows4", "q80_touch_lanes", "q80_shfl_probe"])
    assert re.findall(r"__kernel void (\w+)", (PS / "kernels/gemv_dxa/kernel.cpp").read_text()) == ["gemv_dxa"]


def test_batch_files_up_to_date():
    r = subprocess.run([sys.executable, "-B", str(RES / "make_batch.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_analysis_up_to_date():
    if not (RES / "rows.jsonl").is_file():
        pytest.skip("results/EP.3/rows.jsonl not produced yet")
    r = subprocess.run([sys.executable, "-B", str(RES / "analyze.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# --------------------------------------------------------------------------------------------- stored results

def dxa_beats(gbase, stride, tile0, tile1, eb):
    """Independent re-implementation of sim/simx/dxa/dxa_core.cpp enumerate_work_list for a row-major 2-D tile (64 B GMEM lines, 64 B LMEM words)."""
    beats = reads = 0
    prev = None
    for row in range(tile1):
        g0 = gbase + row * stride
        s0 = row * tile0 * eb
        e0 = 0
        while e0 < tile0:
            g, s = g0 + e0 * eb, s0 + e0 * eb
            span = min(64 - (g & 63), 64 - (s & 63), (tile0 - e0) * eb, 64)
            span -= span % eb
            beats += 1
            if (g & ~63) != prev:
                reads += 1
                prev = g & ~63
            e0 += span // eb
    return beats, reads


def test_dxa_beat_model():
    # a 64-byte-aligned f32 tile of 8 rows x 288 floats (1152 B rows): 18 lines per row, each one beat
    assert dxa_beats(0x1000, 1152, 288, 8, 4) == (8 * 18, 8 * 18)
    # a q80 tile of 8 rows x 288 bytes: rows are 288 B apart (4.5 lines): consecutive rows share the boundary line, so reads < beats
    b, r = dxa_beats(0x1000, 288, 288, 8, 1)
    assert r == 36 and b >= r          # the 8 x 288 B tile is one contiguous 2,304 B region = 36 lines, each read once
    # a one-line tile
    assert dxa_beats(0x40, 64, 16, 1, 4) == (1, 1)


# --------------------------------------------------------------------------------------------- stored results (results/<run>/rows.jsonl)
def _load_rows():
    p = RES / "rows.jsonl"
    if not p.is_file():
        pytest.skip("results/EP.3/rows.jsonl not produced")
    sys.path.insert(0, str(RES))
    import analyze  # noqa: E402
    return analyze, analyze.load()


def _log(r):
    return (PS / r["log"]).read_text(errors="replace")


def test_ep3_results():
    """Every row the analysis uses: ok and passed; per group the output checks of TEST on the stored logs."""
    an, rows = _load_rows()
    groups = {}
    for r in rows:
        groups.setdefault(r["tags"].get("group"), []).append(r)
    # coverage by arguments (a run two batches share is stored once, under the tags of the batch that ran it first)
    assert len(an.f32_rows(rows)) == 45
    assert len(an.q80_rows(rows)) >= 52
    assert len(an.sel(rows, "ps_gemv_dxa", an.E1, 7, P=None, L="1")) >= 62
    assert len(list(__import__("csv").DictReader((RES / "lin.csv").open()))) == 72
    assert len(an.sel(rows, "ps_gemv_q80", an.E1, 8, A=None)) >= 24 and len(an.sel(rows, "ps_gemv_q80", None, 8, A=True)) >= 24
    e4 = [r for r in rows if r["config"] == "PS-BASE-S1-4c"]       # E4 correctness: every new variant on every non-classifier shape passed
    assert len({(r["app"], r["args"]) for r in e4}) >= 77
    for g, rs in groups.items():
        for r in rs:
            log = _log(r)
            assert "PASSED!" in log, (g, r["args"])
            # every output check in the log is clean (PSCHECK errors=0), every launch digest is present
            for m in STEM_RE.finditer(log):
                assert int(m.group(7)) == 0, (g, r["args"], m.group(0))
            if ((r["app"] == "ps_gemv_q80" and not re.search(r"-v (touch|lanes|shfl)", r["args"])) or (r["app"] == "ps_gemv_dxa" and "-w q80" in r["args"])):
                mism = [int(m.group(1)) for m in re.finditer(r"PSBITEXACT: \S+ n=\d+ mismatches=(\d+)", log)]
                assert mism and not any(mism), (g, r["args"], mism)     # Q8_0 variants are bit-exact to runq.c order
                assert "PSDIGEST:" in log
    # the Q8_0 variants and DXA flavour of one shape produce one and the same output digest (same arithmetic, same inputs)
    by = {}
    for r in rows:
        t = r["tags"]
        if t.get("group") in ("q80", "dxa") and r["app"] in ("ps_gemv_q80", "ps_gemv_dxa") and (t.get("track") == "P8"):
            d = re.search(r"PSDIGEST: launch0:vs_f64 n=\d+ fnv1a64=([0-9a-f]+)", _log(r))
            if d and (r["app"] == "ps_gemv_dxa" or t["variant"] != "lanes"):
                by.setdefault((t["M"], t["K"], t["B"]), set()).add(d.group(1))
    assert by and all(len(v) == 1 for v in by.values()), {k: v for k, v in by.items() if len(v) != 1}


def test_ep3_stock_identity():
    """The new translation units do not move the stock entries: the stock `rows` / `lanes` kernels re-run here have the instruction count
    (and cycles) (fp32) (Q8_0) rows of the same shape."""
    an, rows = _load_rows()
    ep0 = an.ep0_stock()
    n = 0
    for r in an.latest(rows, group="f32"):
        t = r["tags"]
        sv, srow = ep0[(t["M"], t["K"])]
        if t["variant"] == sv:
            assert (r["cycles"], r["instrs"]) == (srow["cycles"], srow["instrs"]), (t, r["cycles"], srow["cycles"])
            n += 1
    assert n == 9
    ep2 = an.ep2_stock()
    m = 0
    for r in an.latest(rows, group="q80"):
        t = r["tags"]
        e2 = ep2.get((t["M"], t["K"], t["B"], "rows")) or ep2.get((t["M"], t["K"], t["B"], "auto"))
        if t["variant"] == "rows" and e2:
            assert r["instrs"] == e2["instrs"], (t, r["instrs"], e2["instrs"])        # the same code
            if t["B"] == 1:
                assert r["cycles"] == e2["cycles"], (t, r["cycles"], e2["cycles"])    # B = 1: identical to the cycle
            else:
                assert abs(r["cycles"] / e2["cycles"] - 1) < 0.02, (t, r["cycles"], e2["cycles"])   # B >= 2: the earlier rows predate
            m += 1
    assert m >= 9


def test_dxa_counters_match_prediction():
    """DXA engine counters of one launch == the host's PSDXAEXP prediction (LMEM write beats and transfers), for every dxac row."""
    an, rows = _load_rows()
    n = 0
    for r in an.latest(rows, group="dxac"):
        if r["tags"].get("track") == "probe":
            continue
        txt = _log(r)
        exp = re.search(r"PSDXAEXP: transfers=(\d+) lmem_writes=(\d+) gmem_reads=(\d+) staged_bytes=(\d+)", txt)
        got = [DXA for DXA in (an.DXA_RE.search(x) for x in r["perf_raw"]) if DXA]
        assert exp and got, r["args"]
        xf, lw, gr, sb = (int(g) for g in exp.groups())
        t, gm, gd, lmw = (int(g) for g in got[-1].groups())
        assert (t, lmw) == (xf, lw), (r["args"], (t, lmw), (xf, lw))
        assert gm - gd == gr, (r["args"], gm, gd, gr)   # the engine counts the dedup'd (same-line) reads too; the prediction counts distinct lines
        n += 1
    assert n >= 20


def test_dxa_cycles_vs_non_dxa():
    """DXA cycles below the non-DXA variant at the largest K, or the reason recorded (results/<run>/README.md)."""
    an, rows = _load_rows()
    lines = list(__import__("csv").DictReader((RES / "dxa.csv").open()))
    assert lines
    readme = (RES / "README.md").read_text()
    for fl in ("f32", "q80"):
        big = [r for r in lines if r["flavour"] == fl and r["B"] == "1" and r["depth"] == "2" and r["model"] == "llama8b_s8" and r["matrix"] == "w2"]
        assert big, fl
        faster = float(big[0]["speedup_vs_best_sw"]) > 1.0
        assert faster or f"DXA {fl} is slower" in readme, (fl, big[0]["speedup_vs_best_sw"])


def test_persist_pairs_in_stored_results():
    """persist mode in every SW* app: launch-2 L2 read misses <= 5 % of launch 1 and every launch passes (res.csv)."""
    an, rows = _load_rows()
    lines = list(__import__("csv").DictReader((RES / "res.csv").open()))
    assert len(lines) >= 34, len(lines)
    assert all(r["within_5pct"] == "yes" for r in lines), [r for r in lines if r["within_5pct"] != "yes"]
    apps = {r["app"] for r in lines}
    assert {"ps_twolaunch", "ps_gemv_f32", "ps_gemv_q80", "ps_gemv_dxa", "ps_linear"} <= apps
    for r in an.latest(rows, group="res"):
        if "-L 2" in r["args"]:
            log = _log(r)
            assert len(re.findall(r"PSDIGEST: launch\d+", log)) >= 2 and "PASSED!" in log


def test_overfetch_diagnosis_files():
    """the over-fetch attribution (diag2.csv) exists for the three shapes and the weights-only kernels read ~1.0x of Wq from DRAM"""
    import csv
    lines = list(csv.DictReader((RES / "diag2.csv").open()))
    assert len(lines) >= 24
    for r in lines:
        if r["kernel"].startswith("touch") and r["machine"] == "E1":
            assert 0.95 <= float(r["dram_wgt_over_Wq"]) <= 1.10, r


def test_methodology_18_8():
    txt = (PS / "docs" / "methodology.md").read_text()
    assert "### 18.8" in txt
    sec = txt.split("### 18.8", 1)[1]
    for needle in ("segmented butterfly", "shfl_idx", "over-fetch", "P8T", "h_mem\\*", "bit-exact", "ILP", "DXA"):
        assert needle in sec, needle


# --------------------------------------------------------------------------------------------- vortex (SimX), PS_RUN_VORTEX=1
@pytest.fixture(scope="module")
def vx():
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun  # noqa: E402
    priv = PS / "build" / "tmp" / "ep3_test"
    shutil.rmtree(priv, ignore_errors=True)
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun("EP.3-test", results=priv)


def _run(vx, app, args, config="PS-BASE-S0", tree="ps", perf=1):
    row = vx.run(dict({"tree": tree, "perf": perf, "app": app, "args": args, "config": config}), timeout_s=3600)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


@pytest.mark.vortex
def test_shfl_semantics(vx):
    """segmented butterfly (S = 4, 8, 16, 32) and shfl_idx (whole warp, in 8, in 4) on SimX == the model of alu_unit.cpp"""
    row, log = _run(vx, "ps_gemv_q80", "-m 32 -n 32 -v shfl_probe")
    assert row["status"] == "ok", row["reason"]
    assert re.search(r"^PSSHFL: threads=256 slots=12 mismatches=0 ", log, re.M)


@pytest.mark.vortex
@pytest.mark.parametrize("config", ["PS-BASE-S0", "PS-BASE-S1-4c"])
@pytest.mark.parametrize("v", ["rows_ilp2", "rows_ilp4", "lanes_ilp2"])
def test_f32_variant(vx, config, v):
    row, log = _run(vx, "ps_twolaunch", f"-m 70 -n 288 -v {v} -L 1", config=config)
    assert row["status"] == "ok", row["reason"]
    assert all(int(m.group(7)) == 0 for m in STEM_RE.finditer(log)) and "PASSED!" in log


@pytest.mark.vortex
@pytest.mark.parametrize("config", ["PS-BASE-S0", "PS-BASE-S1-4c"])
@pytest.mark.parametrize("v,B", [("rows_inl", 1), ("rows_ilp2", 3), ("rows_ilp4", 4)])
def test_q80_variant(vx, config, v, B):
    row, log = _run(vx, "ps_gemv_q80", f"-m 50 -n 256 -G 64 -B {B} -v {v} -L 1", config=config)
    assert row["status"] == "ok", row["reason"]
    assert re.search(r"^PSBITEXACT: launch0:runq_exact n=\d+ mismatches=0 variant=" + v + "$", log, re.M)


@pytest.mark.vortex
@pytest.mark.parametrize("config", ["PS-BASE-S0", "PS-BASE-S1-4c"])
@pytest.mark.parametrize("fl,d,B,GS", [("f32", 1, 1, 0), ("f32", 4, 3, 0), ("q80", 2, 1, 32), ("q80", 4, 4, 64), ("q80", 1, 2, 16), ("q80", 2, 3, 128)])
def test_dxa_variant(vx, config, fl, d, B, GS):
    N = 512 if GS in (64, 128, 0) else 288
    a = f"-m 100 -n {N} -w {fl} -d {d} -B {B} -L 1" + (f" -G {GS}" if GS else "")
    row, log = _run(vx, "ps_gemv_dxa", a, config=config)
    assert row["status"] == "ok", row["reason"]
    assert all(int(m.group(7)) == 0 for m in STEM_RE.finditer(log))
    if fl == "q80":
        assert re.search(r"^PSBITEXACT: launch0:runq_exact n=\d+ mismatches=0 variant=dxa$", log, re.M)


@pytest.mark.vortex
@pytest.mark.parametrize("track,cfg,v", [("fp16", "PS-BASE-S0", "tcu"), ("int8", "EP-SMOKE-S0-tcuint8", "tcu")])
@pytest.mark.parametrize("B", [1, 2, 4])
def test_linear_tracks(vx, track, cfg, v, B):
    row, log = _run(vx, "ps_linear", f"-m {B} -n 70 -k 96 -v {v} -L 1", config=cfg)
    assert row["status"] == "ok", row["reason"]
    assert "PASSED!" in log
    if track == "int8":
        assert re.search(r"^PSCHECK: (?:launch0:)?vs_int_exact n=\d+ errors=0", log, re.M)


@pytest.mark.vortex
@pytest.mark.parametrize("app,a", [("ps_twolaunch", "-m 96 -n 288 -v rows_ilp2"), ("ps_gemv_q80", "-m 96 -n 288 -G 32 -v rows_ilp2"),
                                     ("ps_gemv_dxa", "-m 96 -n 288 -w q80 -G 32 -d 2"), ("ps_linear", "-m 4 -n 96 -k 288 -v tcu")])
def test_persist_pairs(vx, app, a):
    cfg = "EP-CTRL-S0-L2big-P"
    r1, l1 = _run(vx, app, a + " -L 1 -P", config=cfg, perf=5)
    r2, l2 = _run(vx, app, a + " -L 2 -P", config=cfg, perf=5)
    assert r1["status"] == r2["status"] == "ok"
    m = lambda l: int(re.search(r"l2cache: reqs=\d+, miss_r=(\d+)", l).group(1))   # noqa: E731
    assert m(l2) - m(l1) <= 0.05 * m(l1)
    assert len(re.findall(r"PSDIGEST: launch\d+", l2)) >= 2
