"""Tests: ps patch sched-replay (PS_SCHEDULE work-unit schedule replay + persist_l2 sensitivity knob).

Non-vortex (always run, cheap):
  test_mkschedule_roundtrip / test_mkschedule_rejects   scripts/mkschedule.py writes what it parses; A1/A2/persist
                                    knobs, unknown knobs, decreasing work counts, bad values are refused.
  test_mkschedule_cli               write / check / from-csv.
  test_cpp_parser_rules_match       ps_schedule.h accepts exactly mkschedule.KNOBS and rejects mkschedule.REJECTED.
  test_persist_guard_in_host_helper kernels/common/ps_phase.h: PS_ALLOW_PERSIST_L2 defaults to 0, both write paths
                                    call the refusal first, the default-config static_assert is still there.
  test_persist_guard_compiles       g++ -fsyntax-only on a tiny host TU: persist refused by default, allowed with the
                                    macro, any other macro value is a compile error.
  test_stored_schedule_files        every results/<run>/sched/*.sched parses (except the deliberate reject case).
Vortex-marked (PS_RUN_VORTEX=1): rows of experiments/<run>.yaml in results/$PS_SCHEDULE_STEP (default; same specs
-> same run keys -> reused; another id re-runs through vxrun -> lowload.sh):
  test_constant_schedule_equals_static   const_kv == static -R kv, const_policy == no schedule: instrs, cycles, every
                                    class-8 total identical.
  test_two_segment_switch           instrs unit, n = 10000: switch at W = 150000 (multiple of n) -> the switch is in
                                    the cycle that closes row 15 (work_end 150000), rows 1..15 carry no KV_STREAM
                                    class traffic, row 16 does; W = 145000 -> the switch lies inside row 15 (the first
                                    row with work_end >= W), rows 1..14 carry none, row 15 does; cycles identical to
                                    the static runs; the static -R kv trace has KV traffic in those early rows.
  test_marks_unit_switch            PS_HINTS=1, marks unit, switch at mark 16: PSSCHED work=16 at the cycle of the trace
                                    row with work_end 16; cycles identical to the unscheduled run.
  test_rejected_knob_aborts         a schedule with l2_waymask: SimX refuses to run.
  test_persist_l2_launch2_misses    ps_twolaunch 256 x 256: launch-2 L2 read misses <= 5 % of launch 1 with
                                    persist_l2 (perf 5 differences and per-launch class-8 trace rows), cold mode ~100 %.
  test_persist_refused_without_macro   -P without -DPS_ALLOW_PERSIST_L2=1 -> FAILED, nothing written.
  test_persist_llama_no_stale_data  teacher-forced stories260K on the ps tree with persist_l2: PASSED, logits
                                    bit-identical to the cold run; the no-invalidation negative control differs.
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
SCHED_H = TREE / "sim" / "simx" / "ps_schedule.h"
PHASE_H = PS / "kernels" / "common" / "ps_phase.h"
RES = PS / "results" / "P4.5"
sys.path.insert(0, str(PS / "scripts"))
import mkschedule as mk  # noqa: E402
import pstrace  # noqa: E402

STEP = os.environ.get("PS_SCHEDULE_STEP", "P4.5")
needs_tree = pytest.mark.skipif(not SCHED_H.exists(), reason="vortex-trees/ps without patch sched-replay")


# ------------------------------------------------------------------------------------------------ non-vortex
def test_mkschedule_roundtrip():
    lines = mk.make_lines([(0, {"kv0_base": 0x11000, "kv0_limit": 0x52C00}), (150000, {"mode": 1, "dxa_depth3": 7})])
    text = mk.format_lines(lines, "hdr")
    back = mk.parse(text)
    assert [(l.work, l.sets, l.launch) for l in back] == [(l.work, l.sets, l.launch) for l in lines]
    assert mk.parse("launch=2 5 insert=1  # c\n\n")[0].launch == 2
    assert mk.parse("0x10 mode=0x3")[0].work == 16


@pytest.mark.parametrize("text", [
    "10 l2_waymask=3", "10 l1_waymask=1", "0 lmem=4096", "0 kmu_lmem=1", "0 persist_l2=1", "0 flags=1",
    "0 nosuch=1", "10 mode=1\n5 mode=0", "-1 mode=1", "10", "10 mode", "10 mode=x", "10 dxa_depth0=256",
    "10 range_acct_only=2", "10 dxa_depth8=1", "launch=x 0 mode=1", "", "# only a comment\n", "08 mode=1",
])
def test_mkschedule_rejects(text):
    with pytest.raises(mk.ScheduleError):
        mk.parse(text)


def test_mkschedule_cli(tmp_path):
    out = tmp_path / "a.sched"
    assert mk.main(["write", str(out), "0:kv0_base=0x11000,kv0_limit=0x52c00", "150000:kv0_limit=0"]) == 0
    assert [l.work for l in mk.parse(out.read_text())] == [0, 150000]
    assert mk.main(["check", str(out)]) == 0
    assert mk.main(["write", str(tmp_path / "b.sched"), "10:l2_waymask=3"]) == 1
    assert not (tmp_path / "b.sched").exists()
    c = tmp_path / "s.csv"
    c.write_text("work_count,knob,value,launch\n100,mode,1,\n0,kv0_base,0x100,\n0,kv0_limit,0x200,\n7,insert,2,1\n")
    assert mk.main(["from-csv", str(tmp_path / "c.sched"), str(c)]) == 0
    got = mk.parse((tmp_path / "c.sched").read_text())
    assert [(l.work, l.launch, l.sets) for l in got] == [(0, None, [("kv0_base", 0x100), ("kv0_limit", 0x200)]),
                                                         (7, 1, [("insert", 2)]), (100, None, [("mode", 1)])]


def _strip_cpp_comments(t: str) -> str:
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", t, flags=re.S))


@needs_tree
def test_cpp_parser_rules_match():
    src = _strip_cpp_comments(SCHED_H.read_text())
    accepted = set(re.findall(r'k == "([a-z0-9_]+)"', src))
    rejected = {k for k in accepted if k in mk.REJECTED}
    assert rejected == set(mk.REJECTED), rejected
    plain = accepted - rejected
    assert plain | {f"dxa_depth{c}" for c in range(8)} == set(mk.KNOBS), plain ^ set(mk.KNOBS)
    assert 'rfind("dxa_depth", 0) == 0' in src and "k[9] <= '7'" in src
    # the rejection branches come before any accepting branch (an A2 knob can never fall through)
    first_accept = src.index('k == "mode"')
    for k in mk.REJECTED:
        assert src.index(f'k == "{k}"') < first_accept, k
    assert "work_count decreases" in src


def test_persist_guard_in_host_helper():
    t = _strip_cpp_comments(PHASE_H.read_text())
    assert re.search(r"#ifndef PS_ALLOW_PERSIST_L2\s*#define PS_ALLOW_PERSIST_L2 0\s*#endif", t)
    assert "static_assert(PS_ALLOW_PERSIST_L2 == 0 || PS_ALLOW_PERSIST_L2 == 1" in t
    assert "static_assert((PhaseCfg{}.flags & phase_dcr::FLAG_PERSIST_L2) == 0" in t
    for fn in ("phase_enqueue", "phase_write"):
        body = t[t.index(f"inline int {fn}("):]
        body = body[:body.index("\n}\n")]
        first = [l.strip() for l in body.splitlines()[1:] if l.strip()][0]
        assert first == "if (persist_refused(c)) return kPersistRefused;", (fn, first)


def _vortex_include_dirs():
    # the include path of the generated app Makefiles (build/vortex-ps/tests/regression/<app>, host compile line)
    return [PS / "kernels" / "common", TREE / "sw" / "runtime" / "include", TREE / "sw" / "kernel" / "include",
            PS / "build" / "vortex-ps" / "sw", PS / "build" / "vortex-ps" / "hw", TREE / "sw" / "common"]


@pytest.mark.skipif(not (PS / "build" / "vortex-ps" / "sw" / "VX_types.h").exists(), reason="ps build not configured")
def test_persist_guard_compiles(tmp_path):
    tu = tmp_path / "g.cpp"
    tu.write_text('#include "ps_phase.h"\n'
                  "static_assert(ps::kPersistL2Allowed == (EXPECT == 1), \"guard\");\n"
                  "int main() { ps::PhaseCfg c; c.flags = ps::phase_dcr::FLAG_PERSIST_L2;\n"
                  "  return ps::persist_refused(c) == (EXPECT == 0) ? 0 : 1; }\n")
    inc = sum((["-I", str(d)] for d in _vortex_include_dirs()), [])

    def cc(*defs, run=False):
        exe = tmp_path / "g"
        cmd = ["nice", "-n", "19", "g++", "-std=c++17", *inc, *defs, str(tu)]
        cmd += ["-o", str(exe)] if run else ["-fsyntax-only"]
        return subprocess.run(cmd, capture_output=True, text=True)
    r = cc("-DEXPECT=0")
    assert r.returncode == 0, r.stderr[-2000:]
    assert cc("-DEXPECT=1", "-DPS_ALLOW_PERSIST_L2=1").returncode == 0
    bad = cc("-DEXPECT=1", "-DPS_ALLOW_PERSIST_L2=2")
    assert bad.returncode != 0 and "PS_ALLOW_PERSIST_L2 must be 0 or 1" in bad.stderr


def test_stored_schedule_files():
    files = sorted((RES / "sched").glob("*.sched"))
    assert len(files) >= 5
    for f in files:
        if f.name.startswith("reject_"):
            with pytest.raises(mk.ScheduleError):
                mk.parse(f.read_text())
        else:
            mk.parse(f.read_text())


# ------------------------------------------------------------------------------------------------ vortex
def _batch_specs():
    import yaml
    import vxrun
    doc = yaml.safe_load((PS / "experiments" / "P4.5.yaml").read_text())
    out = []
    for sp in vxrun.expand_batch(doc, vxrun.load_named_configs()):
        sp = {k: v for k, v in sp.items() if not k.startswith("_")}
        if sp.get("rep", 0) == 0:
            out.append(sp)
    return out


@pytest.fixture(scope="module")
def vx():
    import vxrun
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), str(PS / "build" / "vortex-ps")],
                       capture_output=True, text=True, env=dict(os.environ, PS=str(PS)))
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun(STEP)


def _spec(group, case, args_contains=None):
    got = [s for s in _batch_specs() if (s.get("tags") or {}).get("group") == group
           and s["tags"].get("case") == case and (args_contains is None or args_contains in s["args"])]
    assert len(got) == 1, (group, case, args_contains, len(got))
    return got[0]


def _run(vx, spec, expect_ok=True):
    s = {k: v for k, v in spec.items() if k not in ("timeout", "mem", "reps")}
    s.setdefault("env", {})
    row = vx.run(s, timeout_s=int(spec.get("timeout", 3600)))
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    if expect_ok:
        assert row["status"] == "ok", (row["status"], row.get("reason"), row["log"])
    return row, (PS / row["log"]).read_text(errors="replace")


def _trace(spec):
    return pstrace.read_trace(spec["env"]["PS_TRACE"])


SCHED_RE = re.compile(r"^PSSCHED launch=(\d+) line=(\d+) work_count=(\d+) work=(\d+) cycle=(\d+) (.*)$", re.M)


@pytest.mark.vortex
@pytest.mark.parametrize("case,twin", [("const_kv", "static_kv"), ("const_policy", "static_none")])
def test_constant_schedule_equals_static(vx, case, twin):
    a, ta = _run(vx, _spec("S", case))
    b, tb = _run(vx, _spec("S", twin))
    assert (a["instrs"], a["cycles"]) == (b["instrs"], b["cycles"])
    assert pstrace.parse_perf_ps(ta) == pstrace.parse_perf_ps(tb)
    sw = SCHED_RE.findall(ta)
    assert len(sw) == 1 and sw[0][2] == "0" and sw[0][4] == "0"   # one line, applied before the first cycle
    if case == "const_kv":
        assert pstrace.parse_perf_ps(ta)["l2_acc_kvs"] > 0          # the range is in force (not a vacuous equality)


@pytest.mark.vortex
@pytest.mark.parametrize("case", ["twoseg150k", "twoseg145k"])
def test_two_segment_switch(vx, case):
    sp = _spec("S", case)
    W = int(sp["tags"]["switch"])
    n = int(sp["env"]["PS_TRACE_EVERY"])
    row, txt = _run(vx, sp)
    for twin in ("static_none", "static_kv"):
        t, _ = _run(vx, _spec("S", twin))
        assert (t["instrs"], t["cycles"]) == (row["instrs"], row["cycles"])
    names, rows = _trace(sp)
    w = pstrace.work_rows(rows, 0)
    kv_cols = [c for c in names if c.endswith("_kvs") and not c.startswith(("dram_bp", "dxa"))]
    kv = [sum(r[c] for c in kv_cols) for r in w]
    k_sw = next(i for i, r in enumerate(w) if r["work_end"] >= W)      # 0-based index of the row where work reached W
    assert w[k_sw]["work_end"] == math.ceil(W / n) * n
    sw = [m for m in SCHED_RE.findall(txt) if int(m[2]) == W]
    assert len(sw) == 1 and int(sw[0][3]) >= W
    c_sw = int(sw[0][4])
    if W % n == 0:
        # the switch happens in the cycle that closes row k_sw: that row and every earlier one ran the old config,
        # row k_sw + 1 is the first under the new one
        assert c_sw == w[k_sw]["cycle"]
        first_new = k_sw + 1
    else:
        # the switch happens inside row k_sw (= the first row with work_end >= W): first row with new-class traffic
        assert w[k_sw - 1]["cycle"] < c_sw < w[k_sw]["cycle"]
        first_new = k_sw
    assert all(v == 0 for v in kv[:first_new]), kv[:first_new]
    assert kv[first_new] > 0
    assert sum(r["l1l2_rd_bytes_kvs"] for r in w) > 0
    # static -R kv trace: KV traffic in every row that has K/V loads at all (the switch, not the kernel, gates it)
    _run(vx, _spec("S", "trace_kv"))
    _, rk = _trace(_spec("S", "trace_kv"))
    assert sum(sum(r[c] for c in kv_cols) for r in pstrace.work_rows(rk, 0)[:first_new]) > 0


@pytest.mark.vortex
def test_marks_unit_switch(vx):
    sp = _spec("S", "marks16")
    row, txt = _run(vx, sp)
    st, _ = _run(vx, _spec("S", "marks_static"))
    assert (st["instrs"], st["cycles"]) == (row["instrs"], row["cycles"])
    _, rows = _trace(sp)
    w = pstrace.work_rows(rows, 0)
    r16 = next(r for r in w if r["work_end"] == 16)
    sw = SCHED_RE.findall(txt)
    assert len(sw) == 1 and sw[0][2] == "16" and sw[0][3] == "16" and int(sw[0][4]) == r16["cycle"]
    assert "mode=0x1 " in sw[0][5]


@pytest.mark.vortex
def test_rejected_knob_aborts(vx):
    row, txt = _run(vx, _spec("S", "reject_waymask"), expect_ok=False)
    assert row["status"] != "ok"
    assert "l2_waymask' rejected" in txt


def _l2(txt):
    m = re.findall(r"l2cache: reqs=(\d+), miss_r=(\d+)", txt)
    assert m, "no l2cache perf line"
    return int(m[-1][0]), int(m[-1][1])


@pytest.mark.vortex
def test_persist_l2_launch2_misses(vx):
    c1, t1 = _run(vx, _spec("P", "cold", "-L 1"))
    c2, t2 = _run(vx, _spec("P", "cold", "-L 2"))
    p1, u1 = _run(vx, _spec("P", "persist", "-L 1 "))
    p2, u2 = _run(vx, _spec("P", "persist", "-L 2 "))
    assert "PASSED!" in u2 and "PSPERSIST launches=2" in u2 and "PSPERSIST-HOST: persist_l2=1" in u2
    l1_cold, l2_cold = _l2(t1)[1], _l2(t2)[1] - _l2(t1)[1]
    l1_per, l2_per = _l2(u1)[1], _l2(u2)[1] - _l2(u1)[1]
    assert l1_per == l1_cold                                  # launch 1 is cold in both modes
    assert l2_cold >= 0.95 * l1_cold                          # cold mode: launch 2 misses again
    assert l2_per <= 0.05 * l1_per, (l2_per, l1_per)          #
    # per-launch rows of the class-8 trace (one work row per launch): same conclusion, L2 read misses all classes
    for case, lo, hi in (("cold_trace", 0.95, 1.05), ("persist_trace", 0.0, 0.05)):
        sp = _spec("P", case)
        _run(vx, sp)
        names, rows = _trace(sp)
        miss = [sum(r[c] for c in names if c.startswith("l2_miss_")) for r in pstrace.work_rows(rows)]
        assert len(miss) == 2
        assert lo * miss[0] <= miss[1] <= hi * miss[0], (case, miss)


@pytest.mark.vortex
def test_persist_refused_without_macro(vx):
    row, txt = _run(vx, _spec("P", "refused"), expect_ok=False)
    assert row["status"] != "ok"
    assert "persist_l2 refused" in txt and "PSPERSIST" not in txt.replace("persist_l2 refused", "")


@pytest.mark.vortex
def test_persist_llama_no_stale_data(vx):
    import numpy as np
    sys.path.insert(0, str(PS))
    from model import llama_vx_eval as ev
    out = {}
    for case in ("cold", "persist", "persist_noinval"):
        sp = _spec("L", case)
        row, txt = _run(vx, sp, expect_ok=(case != "persist_noinval"))
        out[case] = (row, txt, ev.parse_log(txt))
    for case in ("cold", "persist"):
        p = out[case][2]
        assert p["passed"] and p["summary"]["compared"] == 2 and p["summary"]["failures"] == 0, case
        assert len(p["checks"]) == 6 and all(c["errors"] == 0 for c in p["checks"]), case
    assert "PSPERSIST launches=" in out["persist"][1] and "host_inval_lines=0" not in out["persist"][1]
    lc = np.fromfile(RES / "llama_cold.f32", dtype=np.float32)
    lp = np.fromfile(RES / "llama_persist.f32", dtype=np.float32)
    assert lc.size > 0 and np.array_equal(lc, lp)              # persist changes timing only, never values
    assert out["persist"][0]["cycles"] < out["cold"][0]["cycles"]
    # negative control: without the host-write invalidation stale lines are read -> the run is wrong
    nrow, ntxt, npar = out["persist_noinval"]
    ln = RES / "llama_noinval.f32"
    wrong = (nrow["status"] != "ok" or not npar.get("passed")
             or not ln.exists() or ln.stat().st_size != lc.nbytes
             or not np.array_equal(np.fromfile(ln, dtype=np.float32), lc))
    assert wrong, "negative control did not change the result: the stale-data test would not be live"
