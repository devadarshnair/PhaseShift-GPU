"""Tests: ps patch dram-2class (mechanism B: two-class DEMAND / BULK DRAM arbitration with a max-wait aging window
W, inside Ramulator's controller and at DramSim's front FIFO). Semantics: docs/methodology.md §14.B; evidence:
results/<run>/.

Non-vortex (cheap):
  test_patch_files               patches/vortex/ps/0007-dram-2class.patch exists, touches exactly the 9 files, no .ps_*
                                 marker, and nothing under third_party/ramulator (the pinned Ramulator is not patched).
  test_controller_is_generic     static review aid: every code line of Ramulator's GenericDRAMController appears, in
                                 order, in PsTwoClass (sim/common/ps_dram_ctrl.cpp), except the listed replaced lines;
                                 with dram_sched = 0 the only decision path is the configured scheduler (FRFCFS).
  test_encoding                  dram_sched encoding / rejection messages in dram_sim.cpp.
  test_stored_schedule_files     every results/<run>/sched/*.sched parses with scripts/mkschedule.py.
Vortex-marked (PS_RUN_VORTEX=1; rows of experiments/<run>.yaml in results/$PS_B_STEP, default; same specs -> same
run keys -> reused; lowload exit 75 -> skip "DEFERRED"):
  test_queue_location            the measurement: stock FR-FCFS on ps_mix_micro at PS-BASE: the front FIFO's
                                 mean depth (over busy DRAM clocks) is < 1 on every channel, while Ramulator's read
                                 buffer holds more on average (requests queue inside Ramulator) -> Ramulator path.
  test_mode0_schedule_identical  an explicit dram_sched=0 schedule reproduces the no-schedule run exactly.
  test_strict_invariants         strict mode (dram_sched=1), W in {0, 64, 256, 1024}, both machines, both bulk
                                 intensities: BULK grants while a DEMAND request waited in the same buffer and the
                                 BULK class wait was <= W: 0 on every channel; for W > 0 the maximum BULK class wait at
                                 a BULK grant <= W + the longest observed service time; the workload is not trivially
                                 safe: under FR-FCFS the same counter is > 0.
  test_functional_all_pass       every P2 kernel PASSES under strict and work-conserving B (W = 64).
  test_two_segment               PS_SCHEDULE switches strict B on at 200,000 instructions: trace rows before the switch
                                 are identical to the static run (every counter), the controller spent clocks in both
                                 modes, and the kernel cycles change.
  test_rejections                dram_sched = 3, a reserved bit, a BULK slot cap with mode 0: SimX aborts.
  test_noop_regression_csv       results/<run>/noop_regression.csv: every REG entry identical.
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
TREE = PS / "vortex-trees" / "ps"
RES = PS / "results" / "P5.2"
PATCH = PS / "patches" / "vortex" / "ps" / "0007-dram-2class.patch"
sys.path.insert(0, str(PS / "scripts"))
import mkschedule as mk  # noqa: E402
import pstrace  # noqa: E402

STEP = os.environ.get("PS_B_STEP", "P5.2")
FILES = ["sim/common/dram_sim.cpp", "sim/common/dram_sim.h", "sim/common/ps_dram_b.h", "sim/common/ps_dram_ctrl.cpp",
         "sim/simx/mem/memory.cpp", "sim/simx/mem/memory.h", "sim/simx/processor.cpp", "sim/simx/processor_impl.h",
         "third_party/Makefile"]


# ------------------------------------------------------------------------------------------------ non-vortex
def test_patch_files():
    t = PATCH.read_text()
    files = sorted(set(re.findall(r"^diff --git a/(\S+) b/", t, re.M)))
    assert files == FILES, files
    assert not re.search(r"^diff --git a/(.*/)?\.ps_", t, re.M)
    assert "third_party/ramulator/" not in "".join(re.findall(r"^diff --git a/(\S+)", t, re.M))


def _code_lines(text):
    out = []
    for l in text.splitlines():
        c = l.split("//")[0].strip()
        if c and not c.startswith(("*", "/*", "#include")):
            out.append(re.sub(r"\s+", " ", c))
    return out


# Generic lines PsTwoClass replaces (each documented in ps_dram_ctrl.cpp's header comment).
REPLACED = {
    'class GenericDRAMController final : public IDRAMController, public Implementation {',
    'RAMULATOR_REGISTER_IMPLEMENTATION(IDRAMController, GenericDRAMController, "Generic", "A generic DRAM controller.");',
    'm_num_cores = frontend->get_num_cores();',
    'if (req->source_id != -1)',
    'if (req_it = m_scheduler->get_best_request(buffer); req_it != buffer.end()) {',
}


def test_controller_is_generic():
    src = TREE / "third_party/ramulator/src/dram_controller/impl/generic_dram_controller.cpp"
    if not src.exists():
        pytest.skip("no ps tree")
    g = _code_lines(src.read_text())
    ours = _code_lines((TREE / "sim/common/ps_dram_ctrl.cpp").read_text())
    j, missing = 0, []
    for line in g:
        if line in REPLACED:
            continue
        k = j
        while k < len(ours) and ours[k] != line:
            k += 1
        if k == len(ours):
            missing.append(line)
        else:
            j = k + 1
    assert not missing, missing[:5]
    t = (TREE / "sim/common/ps_dram_ctrl.cpp").read_text()
    assert "(m_ps->mode == 0) ? m_scheduler->get_best_request(buffer) : select_two_class(buffer)" in t


def test_encoding():
    t = (TREE / "sim/common/dram_sim.cpp").read_text() if TREE.exists() else pytest.skip("no ps tree")
    for s in ("v & ~0xff03u", '"mode 3 is reserved"', '"a BULK slot cap needs mode 1 or 2"', '"reserved bits set"',
              'controller["impl"] = "PsTwoClass";', "static_cast<int>(tx.cls)"):
        assert s in t, s


def test_stored_schedule_files():
    files = sorted((RES / "sched").glob("*.sched"))
    assert len(files) >= 16
    for f in files:
        mk.parse(f.read_text())


# ------------------------------------------------------------------------------------------------ vortex
def _batch_specs():
    import yaml
    import vxrun
    doc = yaml.safe_load((PS / "experiments" / "P5.2.yaml").read_text())
    return [{k: v for k, v in sp.items() if not k.startswith("_")}
            for sp in vxrun.expand_batch(doc, vxrun.load_named_configs()) if sp.get("rep", 0) == 0]


@pytest.fixture(scope="module")
def vx():
    import vxrun
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), str(PS / "build" / "vortex-ps")],
                       capture_output=True, text=True, env=dict(os.environ, PS=str(PS)))
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun(STEP)


def _specs(**tags):
    return [s for s in _batch_specs() if all(str((s.get("tags") or {}).get(k)) == str(v) for k, v in tags.items())]


def _one(**tags):
    got = _specs(**tags)
    assert len(got) == 1, (tags, len(got))
    return got[0]


def _run(vx, spec, expect_ok=True):
    s = {k: v for k, v in spec.items() if k not in ("timeout", "mem", "reps", "rss")}
    s.setdefault("env", {})
    row = vx.run(s, timeout_s=int(spec.get("timeout", 7200)), rss_gb=spec.get("rss"))
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    if expect_ok:
        assert row["status"] == "ok", (row["status"], row.get("reason"), row["log"])
    return row, (PS / row["log"]).read_text(errors="replace")


def lines(txt, tag):
    """Last launch's lines of a PS_DRAM_REPORT tag, as dicts (numbers converted), one per channel / class."""
    rows = [dict((k, float(v) if "." in v else int(v)) if re.fullmatch(r"[0-9.]+", v) else (k, v)
                 for k, v in re.findall(r"(\w+)=(\S+)", l)) for l in re.findall(rf"^{tag} (.*)$", txt, re.M)]
    if not rows:
        return []
    last = max(r["launch"] for r in rows)
    return [r for r in rows if r["launch"] == last]


@pytest.mark.vortex
def test_queue_location(vx):
    row, txt = _run(vx, _one(group="Q", case="mix", machine="base"))
    q, b = lines(txt, "PSDRAMQ"), lines(txt, "PSDRAMB")
    assert len(q) == len(b) == 2 and "PASSED!" in txt
    for qc, bc in zip(q, b):
        assert qc["front_depth_mean_busy"] < 1.0, qc
        assert bc["rdq_mean_busy"] > qc["front_depth_mean_busy"], (qc, bc)
        assert bc["ticks_m0"] == bc["ctrl_ticks"]
    for case in ("gemv", "bulk", "mix_heavy"):
        _, t = _run(vx, _one(group="Q", case=case, machine="base"))
        assert "PASSED!" in t and len(lines(t, "PSDRAMQ")) == 2


@pytest.mark.vortex
def test_mode0_schedule_identical(vx):
    r0, t0 = _run(vx, _one(group="Q", case="mix", machine="base"))
    rz, tz = _run(vx, _one(group="Z"))
    assert (r0["instrs"], r0["cycles"]) == (rz["instrs"], rz["cycles"])
    for tag in ("PSDRAMQ", "PSDRAMB", "PSDRAMLAT"):
        assert lines(t0, tag) == lines(tz, tag), tag


@pytest.mark.vortex
def test_strict_invariants(vx):
    checked = 0
    for sp in _specs(group="I"):
        m = re.fullmatch(r"m(\d)_w(\d+)", sp["tags"]["mode"])
        mode, W = int(m.group(1)), int(m.group(2))
        _, txt = _run(vx, sp)
        assert "PASSED!" in txt
        for b in lines(txt, "PSDRAMB"):
            assert b[f"ticks_m{mode}"] == b["ctrl_ticks"] and b["aging_max"] == W, b
            if mode != 1:
                continue
            assert b["viol"] == 0, (sp["tags"], b)
            if W > 0:
                assert b["bwait_max"] <= W + b["svc_max"], (sp["tags"], b)
            checked += 1
    assert checked == 2 * 2 * 4 * 2   # machines x cases x W x channels
    # not trivially safe: FR-FCFS grants BULK while DEMAND waits on both machines
    for machine in ("base", "lowbw"):
        _, txt = _run(vx, _one(group="Q", case="mix", machine=machine))
        assert all(b["viol"] > 0 for b in lines(txt, "PSDRAMB")), machine
    # aging is exercised: at the quarter data rate strict W = 64 promotes BULK heads
    _, txt = _run(vx, _one(group="I", case="mix", machine="lowbw", mode="m1_w64"))
    assert sum(b["aged_grants"] for b in lines(txt, "PSDRAMB")) > 0


@pytest.mark.vortex
def test_functional_all_pass(vx):
    specs = _specs(group="F")
    assert len(specs) >= 30
    for sp in specs:
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt, row["log"]
        b = lines(txt, "PSDRAMB")
        mode = int(sp["tags"]["mode"][1])
        assert b and all(x[f"ticks_m{mode}"] > 0 for x in b), row["log"]


@pytest.mark.vortex
@pytest.mark.parametrize("machine", ["base", "lowbw"])
def test_two_segment(vx, machine):
    sp = _one(group="T", case="twoseg", machine=machine)
    st = _one(group="T", case="static", machine=machine)
    W, n = int(sp["tags"]["switch"]), int(sp["env"]["PS_TRACE_EVERY"])
    rs, _ = _run(vx, st)
    row, txt = _run(vx, sp)
    _, rows = pstrace.read_trace(sp["env"]["PS_TRACE"])
    _, rows_s = pstrace.read_trace(st["env"]["PS_TRACE"])
    w, ws = pstrace.work_rows(rows, 0), pstrace.work_rows(rows_s, 0)
    k_sw = next(i for i, r in enumerate(w) if r["work_end"] >= W)
    assert W % n == 0 and w[k_sw]["work_end"] == W
    sw = [m for m in re.findall(r"^PSSCHED launch=0 line=\d+ work_count=(\d+) work=(\d+) cycle=(\d+)", txt, re.M)
          if int(m[0]) == W]
    assert len(sw) == 1 and int(sw[0][2]) == w[k_sw]["cycle"]
    first_new = k_sw + 1
    for i in range(first_new):
        assert w[i] == ws[i], i
    assert any(w[i] != ws[i] for i in range(first_new, min(len(w), len(ws))))
    b = lines(txt, "PSDRAMB")
    # (viol is cumulative and also counts FR-FCFS's BULK-over-DEMAND grants of the mode-0 prefix, so it is not checked
    # here; the strict-mode invariant is test_strict_invariants')
    assert all(x["ticks_m0"] > 0 and x["ticks_m1"] > 0 for x in b), b
    assert row["cycles"] != rs["cycles"]


@pytest.mark.vortex
@pytest.mark.parametrize("case,msg", [("x_mode3", "mode 3 is reserved"), ("x_reserved_bit", "reserved bits set"),
                                      ("x_cap_mode0", "a BULK slot cap needs mode 1 or 2")])
def test_rejections(vx, case, msg):
    row, txt = _run(vx, _one(group="X", case=case), expect_ok=False)
    assert row["status"] != "ok" and "PASSED!" not in txt
    assert msg in txt, row["log"]


@pytest.mark.vortex
def test_noop_regression_csv():
    rows = list(csv.DictReader((RES / "noop_regression.csv").open()))
    assert len(rows) >= 8 and all(r["identical"] == "True" for r in rows), rows
