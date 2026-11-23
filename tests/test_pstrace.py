"""Tests: ps patch ps-counters (MPM class 8 PHASESHIFT + work-unit interval trace) and scripts/pstrace.py.

Non-vortex (always run, cheap):
  test_layout_constants             VX_types.toml [ps_counters]: groups of 8 (classes / mark types), DXA streams, the
                                    histogram last, VX_PS_CNT_NUM, class 8 = PHASESHIFT, pages fit the 10-bit class tag.
  test_counter_names_agree          sim/simx/ps_counters.h and sw/runtime/common/perf.cpp spell the same class / mark
                                    names and (group base, name) table; pstrace.py's CLASSES / MARK_TYPES match.
  test_sop_counting_in_source       csr_unit.cpp counts ps_mark only on trace->sop; the three wrong patch-0002 comments
                                    are gone (carry-overs 1 and 2).
  test_read_trace_and_totals        pstrace on a synthetic two-launch trace with flush rows and a MAX column.
  test_expected_work_rows / test_hist_quantile / test_parse_perf / test_check_cli
Vortex-marked (PS_RUN_VORTEX=1): the rows of experiments/<run>.yaml in results/$PS_PSTRACE_STEP (default; same
specs -> same run keys -> reused; another id re-runs everything through vxrun -> lowload.sh):
  test_marks_one_per_warp_instruction  ps_marklanes, NUM_SFU_LANES < NUM_THREADS and unguarded / divergent marks:
                                    the class-8 mark count == the host's (#warps x reps) (carry-over 3).
  test_trace_sums_equal_totals      every traced run: sum of all rows == the --perf=8 end-of-run totals EXACTLY
                                    (maxima: max of rows), sum of `cycles` / `instrs` == the global PERF line.
  test_trace_no_timing_effect       traced vs untraced run of the same binary: identical instrs, cycles and class-8
                                    totals.
  test_trace_alignment              marked kernel under REG-CFG and PS-BASE-S1-4c: work rows per launch ==
                                    ceil(marks / n), row k closes at work k*n (last row: at the end of the launch).
  test_existing_classes_identical   ps-tree rows of group E have PERF lines identical to the base-tree reference rows.
  test_class8_consistency           class-8 sums agree with the stock classes (L2 reqs/misses, DRAM reads/writes) of
                                    the same binary, every DRAM read completed, and sum occ == sum wait (Little) for
                                    every class without writes still in the DRAM model at the end.
lowload exit 75 -> pytest.skip("DEFERRED").
"""
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
TREE = PS / "vortex-trees" / "ps"
TOML = TREE / "VX_types.toml"
PSC_H = TREE / "sim" / "simx" / "ps_counters.h"
PERF_CPP = TREE / "sw" / "runtime" / "common" / "perf.cpp"
CSR_CPP = TREE / "sim" / "simx" / "csr_unit.cpp"
CORE_H = TREE / "sim" / "simx" / "core.h"
sys.path.insert(0, str(PS / "scripts"))
import pstrace  # noqa: E402

STEP = os.environ.get("PS_PSTRACE_STEP", "P4.3")
needs_tree = pytest.mark.skipif(not TOML.exists(), reason="vortex-trees/ps not created")


def _toml_section(name: str) -> dict:
    out, cur = {}, None
    for line in TOML.read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if s.startswith("[") and s.endswith("]"):
            cur = s.strip("[]")
            continue
        if cur == name and "=" in s:
            k, v = (x.strip() for x in s.split("=", 1))
            out[k] = int(v, 0)
    return out


# --------------------------------------------------------------------------------------------- cheap tests
@needs_tree
def test_layout_constants():
    c = _toml_section("ps_counters")
    ncls = _toml_section("ps_mem_class")["VX_PS_NUM_CLASSES"]
    nmark = _toml_section("ps_mark_type")["VX_PS_NUM_MARK_TYPES"]
    groups = ["L1L2_RD_BYTES", "L1L2_WR_BYTES", "L2_ACCESSES", "L2_MISSES", "L2_FILLS_DEMOTED", "L2_FILLS_BYPASSED",
              "DRAM_RD_BYTES", "DRAM_WR_BYTES", "DRAM_OCC", "DRAM_BP", "DRAM_RD_WAIT", "DRAM_RD_DONE", "DRAM_WR_WAIT",
              "DRAM_WR_DONE"]
    base = 0
    for g in groups:                       # contiguous per-class groups starting at 0
        assert c[f"VX_PS_CNT_{g}"] == base, g
        base += ncls
    assert c["VX_PS_CNT_MARKS"] == base
    assert c["VX_PS_CNT_DXA_BYTES"] == base + nmark
    assert c["VX_PS_CNT_DXA_MAX_INFLIGHT"] == c["VX_PS_CNT_DXA_BYTES"] + c["VX_PS_DXA_STREAMS"]
    assert c["VX_PS_CNT_DRAM_RD_HIST"] == c["VX_PS_CNT_DXA_MAX_INFLIGHT"] + c["VX_PS_DXA_STREAMS"]
    assert c["VX_PS_CNT_NUM"] == c["VX_PS_CNT_DRAM_RD_HIST"] + ncls * c["VX_PS_HIST_BUCKETS"]
    assert c["VX_PS_HIST_BUCKETS"] == pstrace.HIST_BUCKETS
    mpm = _toml_section("dcr_mpm_class")
    assert mpm["VX_DCR_MPM_CLASS_PHASESHIFT"] == 8 and "VX_DCR_MPM_CLASS_RESERVED2" not in mpm
    assert max(mpm.values()) < (1 << c["VX_PS_MPM_PAGE_SHIFT"])        # pages never collide with another class
    pages = math.ceil(c["VX_PS_CNT_NUM"] / c["VX_PS_MPM_PAGE_SIZE"])
    assert pages <= (1 << (10 - c["VX_PS_MPM_PAGE_SHIFT"]))             # the DCR tag carries 10 class bits
    # page size = the user MPM window VX_CSR_MPM_USER .. VX_CSR_MPM_BASE + 31
    base_csr = {k: v for k, v in (re.findall(r"^(VX_CSR_MPM_(?:BASE|USER)) = (0x[0-9A-Fa-f]+)", TOML.read_text(), re.M))}
    assert c["VX_PS_MPM_PAGE_SIZE"] == 32 - (int(base_csr["VX_CSR_MPM_USER"], 16) - int(base_csr["VX_CSR_MPM_BASE"], 16))


def _str_array(text: str, var: str) -> list:
    m = re.search(var + r"\[[A-Z_]+\] = \{([^}]*)\}", text)
    assert m, var
    return re.findall(r'"([^"]*)"', m.group(1))


def _group_table(text: str) -> list:
    return re.findall(r"\{(VX_PS_CNT_[A-Z0-9_]+), \"([a-z0-9_]+)\"\}", text)


@needs_tree
def test_counter_names_agree():
    h, p = PSC_H.read_text(), PERF_CPP.read_text()
    # ps_counters.h: two `n[...]` arrays (classes, then mark types); perf.cpp: `cls[...]`, `mark[...]`
    arrays_h = re.findall(r"static const char\* const n\[[A-Z_]+\] = \{([^}]*)\}", h)
    assert len(arrays_h) == 2
    cls_h, mark_h = (re.findall(r'"([^"]*)"', a) for a in arrays_h)
    assert cls_h == _str_array(p, r"static const char\* const cls") == pstrace.CLASSES
    assert mark_h == _str_array(p, r"static const char\* const mark") == pstrace.MARK_TYPES
    assert _group_table(h) == _group_table(p) and len(_group_table(h)) == 14
    for tok in ('"dxa_bytes_s"', '"dxa_max_inflight_s"', '"dram_rd_hist_"', '"_b"', '"marks_"'):
        assert tok in h and tok in p, tok


@needs_tree
def test_sop_counting_in_source():
    csr = CSR_CPP.read_text()
    assert re.search(r"if \(writes && trace->sop && !ps_mark_written\)", csr)    # carry-over 1
    # carry-over 2: the three wrong patch-0002 comments are gone
    assert "process() calls this once per CSR instruction (not once per SFU packet)" not in csr
    assert "so only the first packet\n  // with an active lane performs the write" not in csr
    assert "with an active lane performs the write (every packet still returns the read value)" not in csr
    core = CORE_H.read_text()
    assert "(one count per CSR\n  // write instruction of a warp; type" not in core
    assert "trace->sop" in core and "trace->sop" in csr


def _write_trace(path: Path, names: list, rows: list):
    with open(path, "w") as f:
        f.write(",".join(pstrace.FIXED + names) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in pstrace.FIXED + names) + "\n")


NAMES = ["dram_rd_bytes_ord", "marks_decode", "dxa_max_inflight_s0"]


def _row(launch, row, kind, interval, work_end, work_at, cycles, instrs, v):
    return dict(zip(pstrace.FIXED, [launch, row, kind, interval, "marks", 2, work_end, work_at, 0, cycles, instrs]),
                **dict(zip(NAMES, v)))


SYN = [_row(0, 0, "work", 1, 2, 2, 100, 10, [64, 2, 3]), _row(0, 1, "work", 2, 3, 3, 50, 5, [128, 1, 1]),
       _row(0, 2, "flush", 0, 3, 3, 7, 0, [0, 0, 0]),
       _row(1, 3, "work", 1, 2, 2, 90, 9, [0, 2, 5]), _row(1, 4, "work", 2, 4, 4, 30, 3, [64, 2, 0]),
       _row(1, 5, "flush", 0, 4, 4, 0, 0, [0, 0, 0])]


def test_read_trace_and_totals(tmp_path):
    p = tmp_path / "t.csv"
    _write_trace(p, NAMES, SYN)
    names, rows = pstrace.read_trace(p)
    assert names == NAMES and len(rows) == 6 and rows[2]["kind"] == "flush"
    tot = pstrace.totals(rows, names)
    assert tot == {"dram_rd_bytes_ord": 256, "marks_decode": 7, "dxa_max_inflight_s0": 5, "cycles": 277, "instrs": 27}
    assert pstrace.check_totals(rows, names, {"dram_rd_bytes_ord": 256, "marks_decode": 7, "dxa_max_inflight_s0": 5}) == []
    bad = pstrace.check_totals(rows, names, {"dram_rd_bytes_ord": 255, "marks_decode": 7, "dxa_max_inflight_s0": 5})
    assert bad == [("dram_rd_bytes_ord", 256, 255)]
    assert pstrace.check_totals(rows, names, {"dram_rd_bytes_ord": 256, "marks_decode": 7}) == [("dxa_max_inflight_s0", 5, None)]
    assert [len(pstrace.work_rows(rows, L)) for L in (0, 1)] == [2, 2]
    assert pstrace.launch_units(rows, 0) == 3 and pstrace.launch_units(rows, 1) == 4
    with pytest.raises(ValueError):
        bad_p = tmp_path / "bad.csv"
        bad_p.write_text("a,b\n1,2\n")
        pstrace.read_trace(bad_p)


def test_expected_work_rows():
    assert [pstrace.expected_work_rows(u, 3) for u in (0, 1, 3, 4, 32)] == [1, 1, 1, 2, 11]
    assert pstrace.expected_work_rows(32, 8) == 4 and pstrace.expected_work_rows(32, 1) == 32


def test_hist_quantile():
    h = [0] * 16
    assert pstrace.hist_quantile(h, 0.99) is None
    h[3], h[5] = 98, 2                       # 98 in [8,16), 2 in [32,64)
    assert pstrace.hist_quantile(h, 0.5) == (8.0, 16.0)
    assert pstrace.hist_quantile(h, 0.98) == (8.0, 16.0)
    assert pstrace.hist_quantile(h, 0.99) == (32.0, 64.0)
    h2 = [1] + [0] * 14 + [1]
    assert pstrace.hist_quantile(h2, 0.4) == (0.0, 2.0)
    assert pstrace.hist_quantile(h2, 1.0) == (32768.0, math.inf)


LOG = """PERF: ps: dram_rd_bytes_ord=256, marks_decode=7
PERF: ps: dxa_max_inflight_s0=5
PERF: ps-derived: dram_rd_wait_mean_ord=3.00
PERF: instrs=27, cycles=277, IPC=0.097
PASSED!
"""


def test_parse_perf():
    assert pstrace.parse_perf_ps(LOG) == {"dram_rd_bytes_ord": 256, "marks_decode": 7, "dxa_max_inflight_s0": 5}
    assert pstrace.parse_perf_global(LOG) == (27, 277)
    assert pstrace.parse_perf_global("nothing") is None


def test_check_cli(tmp_path):
    t, log = tmp_path / "t.csv", tmp_path / "r.log"
    _write_trace(t, NAMES, SYN)
    log.write_text(LOG)
    r = subprocess.run([sys.executable, "-B", str(PS / "scripts" / "pstrace.py"), "check", str(t), str(log)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "launch 0: 2 work rows, units 3, n 2, expected 2 OK" in r.stdout
    log.write_text(LOG.replace("=256", "=255"))
    r = subprocess.run([sys.executable, "-B", str(PS / "scripts" / "pstrace.py"), "check", str(t), str(log)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "MISMATCH dram_rd_bytes_ord" in r.stdout


# --------------------------------------------------------------------------------------------- vortex tests
def _batch_specs():
    """The expanded run specs of experiments/<run>.yaml (list-valued args / perf expanded as vxrun does), rep 0 only."""
    import yaml
    import vxrun
    doc = yaml.safe_load((PS / "experiments" / "P4.3.yaml").read_text())
    out = []
    for sp in vxrun.expand_batch(doc, vxrun.load_named_configs()):
        sp = {k: v for k, v in sp.items() if not k.startswith("_")}
        if sp.get("rep", 0) == 0:
            out.append(sp)
    return sorted(out, key=lambda x: json.dumps(x, sort_keys=True))


@pytest.fixture(scope="module")
def vx():
    import vxrun
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), str(PS / "build" / "vortex-ps")],
                       capture_output=True, text=True, env=dict(os.environ, PS=str(PS)))
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun(STEP)


def _run(vx, spec):
    s = {k: v for k, v in spec.items() if k not in ("timeout", "mem", "reps")}
    s.setdefault("env", {})
    row = vx.run(s, timeout_s=int(spec.get("timeout", 3600)))
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    assert row["status"] == "ok", (row["status"], row.get("reason"), row["log"])
    return row, (PS / row["log"]).read_text(errors="replace")


def _group(g):
    return [s for s in _batch_specs() if (s.get("tags") or {}).get("group") == g]


def _sid(s):
    env = s.get("env") or {}
    return f"{s['app']}|{s.get('config')}|{s.get('configs', '')}|{s['args']}|p{s['perf']}|" + \
           (Path(env["PS_TRACE"]).stem if "PS_TRACE" in env else "notrace")


def _traced():
    return [s for s in _batch_specs() if "PS_TRACE" in (s.get("env") or {}) and (s.get("tags") or {}).get("group") != "O"]


@pytest.mark.vortex
@pytest.mark.parametrize("spec", [s for s in _group("M") if "PS_TRACE" not in (s.get("env") or {})], ids=_sid)
def test_marks_one_per_warp_instruction(vx, spec):
    row, log = _run(vx, spec)
    m = re.search(r"^PSMARKLANES expect=(\d+) type=(\d+) mode=(\d+) grid=(\d+) warps=(\d+) reps=(\d+) nt=(\d+)", log, re.M)
    assert m, "no PSMARKLANES line"
    expect, typ, nt = int(m.group(1)), int(m.group(2)), int(m.group(7))
    assert nt == int(spec["tags"]["nt"])
    perf = pstrace.parse_perf_ps(log)
    got = {t: perf[f"marks_{t}"] for t in pstrace.MARK_TYPES}
    want = {t: (expect if i == typ else 0) for i, t in enumerate(pstrace.MARK_TYPES)}
    assert got == want, (got, want)
    assert expect == 4 * 2 * 3


@pytest.mark.vortex
@pytest.mark.parametrize("spec", _traced(), ids=_sid)
def test_trace_sums_equal_totals(vx, spec):
    row, log = _run(vx, spec)
    names, rows = pstrace.read_trace(spec["env"]["PS_TRACE"])
    assert len(names) == 256 and rows
    perf = pstrace.parse_perf_ps(log)
    assert len(perf) == 256
    assert pstrace.check_totals(rows, names, perf) == []
    tot = pstrace.totals(rows, names)
    instrs, cycles = pstrace.parse_perf_global(log)
    assert (tot["instrs"], tot["cycles"]) == (instrs, cycles)
    assert {r["unit"] for r in rows} == {spec["env"].get("PS_TRACE_UNIT", "marks")}
    assert {r["every"] for r in rows} == {int(spec["env"].get("PS_TRACE_EVERY", "1"))}


def _twin(spec):
    """The untraced run of the same binary / config / args (env {})."""
    same = ("app", "config", "configs", "args", "perf", "cores", "threads", "warps", "l2")
    for s in _batch_specs():
        if all(s.get(k) == spec.get(k) for k in same) and not s.get("env"):
            return s
    raise AssertionError(f"no untraced twin for {_sid(spec)}")


@pytest.mark.vortex
@pytest.mark.parametrize("spec", _traced(), ids=_sid)
def test_trace_no_timing_effect(vx, spec):
    row, log = _run(vx, spec)
    trow, tlog = _run(vx, _twin(spec))
    assert (row["instrs"], row["cycles"]) == (trow["instrs"], trow["cycles"])
    assert pstrace.parse_perf_ps(log) == pstrace.parse_perf_ps(tlog)


@pytest.mark.vortex
@pytest.mark.parametrize("spec", [s for s in _traced() if s["env"].get("PS_TRACE_UNIT", "marks") == "marks"], ids=_sid)
def test_trace_alignment(vx, spec):
    row, log = _run(vx, spec)
    names, rows = pstrace.read_trace(spec["env"]["PS_TRACE"])
    n = int(spec["env"].get("PS_TRACE_EVERY", "1"))
    marks = sum(v for k, v in pstrace.parse_perf_ps(log).items() if k.startswith("marks_"))
    launches = sorted({r["launch"] for r in rows})
    assert marks > 0 and len(launches) == 1
    w = pstrace.work_rows(rows, launches[0])
    assert len(w) == math.ceil(marks / n), (len(w), marks, n)
    assert [r["interval"] for r in w] == list(range(1, len(w) + 1))
    assert [r["work_end"] for r in w[:-1]] == [n * k for k in range(1, len(w))]
    assert w[-1]["work_end"] == w[-1]["work_at"] == marks
    assert sum(r[k] for r in w for k in names if k.startswith("marks_")) == marks


@pytest.mark.vortex
def test_trace_alignment_two_configs(vx):
    """The same marked kernel under REG-CFG and PS-BASE-S1-4c: row k covers the same work units under both."""
    by_cfg = {}
    for s in _traced():
        if s["app"] == "ps_decode_attn_gqa" and s["env"].get("PS_TRACE_UNIT", "marks") == "marks":
            _run(vx, s)
            _, rows = pstrace.read_trace(s["env"]["PS_TRACE"])
            by_cfg.setdefault(int(s["env"].get("PS_TRACE_EVERY", "1")), {})[s["config"]] = \
                [r["work_end"] for r in pstrace.work_rows(rows)]
    for n in (1, 3, 8):
        assert set(by_cfg[n]) == {"REG-CFG", "PS-BASE-S1-4c"}, by_cfg[n].keys()
        a, b = by_cfg[n]["REG-CFG"], by_cfg[n]["PS-BASE-S1-4c"]
        assert a == b, (n, a, b)


def _ref_row(spec):
    """Base-tree row of results/<ref step> with the same app, args, config, perf class and env."""
    ref = spec["tags"]["ref"]
    best = None
    for line in (PS / "results" / ref / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        if (r.get("status"), r.get("tree"), r["app"], r["args"], r.get("config"), int(r["perf"]), r.get("env") or {}) == \
           ("ok", "base", spec["app"], spec["args"], spec.get("config"), int(spec["perf"]), {}):
            best = r
    assert best, f"no base reference row in results/{ref}"
    return best, (PS / best["log"]).read_text(errors="replace")


def _perf_lines(text):
    return [l for l in text.splitlines() if l.startswith("PERF: ")]


E_SPECS = _group("E")


@pytest.mark.vortex
@pytest.mark.parametrize("spec", E_SPECS, ids=_sid)
def test_existing_classes_identical(vx, spec):
    row, log = _run(vx, spec)
    bref, blog = _ref_row(spec)
    assert (row["instrs"], row["cycles"]) == (bref["instrs"], bref["cycles"])
    assert _perf_lines(log) == _perf_lines(blog)
    assert len(_perf_lines(log)) >= 2


@pytest.mark.vortex
def test_class8_consistency(vx):
    base = {"app": "ps_decode_attn_gqa", "config": "REG-CFG", "configs": "-DPS_HINTS=1",
            "args": "-B 2 -H 8 -K 4 -D 32 -T 128 -v gqa", "tree": "ps", "timeout": 3600}
    r8, l8 = _run(vx, dict(base, perf=8, env={}))
    r5, l5 = _run(vx, dict(base, perf=5, env={}))
    r7, l7 = _run(vx, dict(base, perf=7, env={}))
    assert r8["cycles"] == r5["cycles"] == r7["cycles"]
    p = pstrace.parse_perf_ps(l8)
    C = pstrace.CLASSES
    l2 = [tuple(map(int, m)) for m in re.findall(r"l2cache: reqs=(\d+), miss_r=(\d+) \(hit=\d+%\), miss_w=(\d+)", l5)]
    assert l2
    assert sum(p[f"l2_acc_{c}"] for c in C) == sum(x[0] for x in l2)
    assert sum(p[f"l2_miss_{c}"] for c in C) == sum(x[1] + x[2] for x in l2)
    mem = re.search(r"memory: reqs=(\d+) \(r=(\d+), w=(\d+)\)", l7)
    assert mem
    assert sum(p[f"dram_rd_bytes_{c}"] for c in C) == 64 * int(mem.group(2))
    assert sum(p[f"dram_wr_bytes_{c}"] for c in C) == 64 * int(mem.group(3))
    for c in C:
        assert p[f"dram_rd_done_{c}"] * 64 == p[f"dram_rd_bytes_{c}"]        # every read completed
        # writes of the LAST flush can still sit in the DRAM model when the host reads the counters (the flush loop
        # waits for SimPlatform idle, which does not cover DramSim's write queue): done <= sent, and Little's law
        # sum occ = sum waits holds exactly iff nothing is in flight, otherwise occ exceeds the completed waits
        wr_all = p[f"dram_wr_done_{c}"] * 64 == p[f"dram_wr_bytes_{c}"]
        assert p[f"dram_wr_done_{c}"] * 64 <= p[f"dram_wr_bytes_{c}"]
        waits = p[f"dram_rd_wait_{c}"] + p[f"dram_wr_wait_{c}"]
        assert (p[f"dram_occ_{c}"] == waits) if wr_all else (p[f"dram_occ_{c}"] > waits), c
        assert sum(p[f"dram_rd_hist_{c}_b{b}"] for b in range(16)) == p[f"dram_rd_done_{c}"]
        assert p[f"l2_miss_{c}"] <= p[f"l2_acc_{c}"]
        assert p[f"l2_fill_demoted_{c}"] == p[f"l2_fill_bypassed_{c}"] == 0
    assert p["marks_decode"] > 0 and p["l2_acc_kvs"] > 0
