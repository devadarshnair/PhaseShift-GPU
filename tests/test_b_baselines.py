"""Tests: ps patch b-baselines (the B baselines: strict demand-first, BLISS, PADC-lite, as further modes of the
 PsTwoClass controller). Semantics: docs/methodology.md §14.B.b; evidence: results/<run>/.

Non-vortex (cheap):
  test_patch_files               patches/vortex/ps/0008-b-baselines.patch touches exactly the 8 files, no .ps_* marker,
                                 nothing under third_party/ (the pinned Ramulator is not patched).
  test_bliss_port_static         the pinned BLISS plugin / scheduler rules appear in PsTwoClass (streak counter, reset on
                                 a source change, `>= threshold`, clearing on `clk % interval == 0`, safe -> ready ->
                                 FCFS fold) and the defaults equal the pinned plugin's (4, 10000).
  test_encoding                  dram_sched baseline field (bits 7:4) and its rejection messages in dram_sim.cpp.
  test_stored_schedule_files     every results/<run>/sched/*.sched parses with scripts/mkschedule.py.
  test_bliss_replay_model        the trace checker below on a synthetic trace (accepts a correct one, rejects a wrong one).
Vortex-marked (PS_RUN_VORTEX=1; rows of experiments/<run>.yaml in results/<run>; same specs -> same run keys -> reused):
  test_bliss_invariant           counter trace (PS_BLISS_TRACE), 3 runs x 2 channels: replaying the serviced-request
                                 sequence, a source is blacklisted exactly when its streak exceeds T, and every bit is
                                 cleared at every multiple of the clearing interval (and only there); no blacklisted
                                 request was granted while a non-blacklisted one waited in the same buffer.
  test_padc_epochs               PADC-lite epoch trace (PS_PADC_TRACE=1): epochs end every E core cycles; the state is
                                 low exactly when 100 u < theta of a judged epoch; with theta 255 % BULK is deprioritised
                                 (low clocks > 0) and no BULK grant passes a waiting DEMAND request while low; with the
                                 default theta every DXA byte is consumed (u ~ 1), PADC-lite never deprioritises and
                                 reproduces FR-FCFS exactly (cycles and the PSDRAMB lines).
  test_policies_selectable       each policy runs the controller in its own mode for every controller clock (m3 / m4).
  test_functional_all_pass       every P2 kernel PASSES under BLISS and PADC-lite.
  test_two_segment               PS_SCHEDULE switches to BLISS / PADC-lite at 200,000 instructions: trace rows before the
                                 switch are identical to the static FR-FCFS run, both modes were clocked, cycles change.
  test_rejections                reserved policy 3, policy with mode bits, policy with a bit above 15: SimX aborts.
  test_noop_regression_csv       results/<run>/noop_regression.csv: every REG entry identical (base, ps 0002-0007).
"""
import csv
import os
import re
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
TREE = PS / "vortex-trees" / "ps"
RES = PS / "results" / "P5.2b"
PATCH = PS / "patches" / "vortex" / "ps" / "0008-b-baselines.patch"
sys.path.insert(0, str(PS / "scripts"))
sys.path.insert(0, str(PS / "tests"))
import mkschedule as mk  # noqa: E402
import pstrace  # noqa: E402

FILES = ["sim/common/dram_sim.cpp", "sim/common/dram_sim.h", "sim/common/ps_dram_b.h", "sim/common/ps_dram_ctrl.cpp",
         "sim/simx/mem/memory.cpp", "sim/simx/mem/memory.h", "sim/simx/processor.cpp", "sim/simx/processor_impl.h"]


# ------------------------------------------------------------------------------------------------ non-vortex
def test_patch_files():
    t = PATCH.read_text()
    files = sorted(set(re.findall(r"^diff --git a/(\S+) b/", t, re.M)))
    assert files == FILES, files
    assert not re.search(r"^diff --git a/(.*/)?\.ps_", t, re.M)
    assert not any(f.startswith("third_party/") for f in files)


def test_bliss_port_static():
    plug = TREE / "third_party/ramulator/src/dram_controller/impl/plugin/bliss/bliss.cpp"
    if not plug.exists():
        pytest.skip("no ps tree")
    p = plug.read_text()
    assert 'param<int>("blacklist_thresh").default_val(4)' in p
    assert 'param<int>("unblacklist_cycles").default_val(10000)' in p
    assert "if (m_consequtive_src_id >= m_blacklist_thresh) {" in p
    assert "if (m_clk % m_unblacklist_cycles == 0) {" in p
    c = (TREE / "sim/common/ps_dram_ctrl.cpp").read_text()
    for s in ("m_bl_consec++;", "if (src != m_bl_prev) {", "m_bl_consec = 0;", "if (m_bl_consec >= T) {",
              "static_cast<uint64_t>(m_clk) % I == 0", "m_ps->bliss_thresh ? m_ps->bliss_thresh : 4",
              "m_ps->bliss_clear ? m_ps->bliss_clear : 10000",
              "if (s1 != s2) { if (!s1) cand = next; continue; }", "if (r1 != r2) { if (!r1) cand = next; continue; }",
              "if (!(cand->arrive <= next->arrive)) cand = next;",
              "return m_ps->padc_low ? select_two_class(b, true) : m_scheduler->get_best_request(b);"):
        assert s in c, s
    # the decision path is still there for modes 0-2
    assert "(m_ps->mode == 0) ? m_scheduler->get_best_request(buffer) : select_two_class(buffer)" in c


def test_encoding():
    t = (TREE / "sim/common/dram_sim.cpp").read_text() if TREE.exists() else pytest.skip("no ps tree")
    for s in ("pol = (v >> 4) & 0xf", "v & ~0xfff0u", '"baseline policy reserved (bits 7:4: 1 BLISS, 2 PADC-lite)"',
              '"a baseline policy (bits 7:4) needs bits 1:0 = 0 and no bit above 15"', "v & ~0xff03u",
              '"mode 3 is reserved"', "impl_->set_b(3, 0, 8, cap ? cap : 4, cfg.dram_aging ? cfg.dram_aging : 10000);",
              "impl_->set_b(4, 0, 8);", "padc_theta_ = cap ? cap : 50;"):
        assert s in t, s


def test_stored_schedule_files():
    files = sorted((RES / "sched").glob("*.sched"))
    assert len(files) >= 14
    for f in files:
        mk.parse(f.read_text())


def check_bliss_trace(path, T, I):
    """Replay one channel's PS_BLISS_TRACE file. Returns (#serviced rows, #clears, #blacklist sets). Raises
    AssertionError on any deviation from the rule: streak of the serviced source (+1 on the same source, else 1);
    bit set iff the streak > T (bits stay set until a clear); clears exactly at every multiple of I."""
    bl, prev, streak, n, clears, sets = {}, None, 0, 0, [], 0
    last_clk = None
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split(",")
        clk = int(f[1])
        if f[0] == "C":
            assert clk % I == 0, line
            assert not clears or clk == clears[-1] + I, (line, clears[-1:])
            clears.append(clk)
            bl = {}
            continue
        assert f[0] == "S", line
        src, st, after = int(f[2]), int(f[3]), int(f[4])
        streak = streak + 1 if src == prev else 1
        prev = src
        assert st == streak, (line, streak)
        if streak > T:
            if not bl.get(src):
                sets += 1
            bl[src] = True
        assert after == (1 if bl.get(src) else 0), (line, bl)
        if last_clk is not None:
            assert clk >= last_clk, line
        last_clk = clk
        n += 1
    # no clear was skipped: every multiple of I up to the last serviced clock has a C row
    if last_clk is not None and last_clk >= I:
        assert clears and clears[0] == I and clears[-1] >= (last_clk // I) * I, (clears[:2], clears[-1:], last_clk)
    return n, len(clears), sets


def test_bliss_replay_model(tmp_path):
    ok = tmp_path / "ok.0"
    ok.write_text("# h\nS,1,0,1,0\nS,2,0,2,0\nS,3,0,3,1\nS,4,4,1,0\nC,5,,,\nS,6,4,2,0\nS,7,4,3,1\nC,10,,,\nS,11,0,1,0\n")
    assert check_bliss_trace(ok, 2, 5) == (7, 2, 2)
    bad = tmp_path / "bad.0"
    bad.write_text("# h\nS,1,0,1,0\nS,2,0,2,1\n")   # blacklisted one request too early
    with pytest.raises(AssertionError):
        check_bliss_trace(bad, 2, 5)
    skip = tmp_path / "skip.0"
    skip.write_text("# h\nS,1,0,1,0\nC,10,,,\nS,16,0,1,0\n")   # the clear at 5 is missing
    with pytest.raises(AssertionError):
        check_bliss_trace(skip, 2, 5)


# ------------------------------------------------------------------------------------------------ vortex
import test_dram_2class as t52  # noqa: E402  (helpers: _run, lines)


def _batch_specs():
    import yaml
    import vxrun
    doc = yaml.safe_load((PS / "experiments" / "P5.2b.yaml").read_text())
    return [{k: v for k, v in sp.items() if not k.startswith("_")}
            for sp in vxrun.expand_batch(doc, vxrun.load_named_configs()) if sp.get("rep", 0) == 0]


@pytest.fixture(scope="module")
def vx():
    import subprocess
    import vxrun
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), str(PS / "build" / "vortex-ps")],
                       capture_output=True, text=True, env=dict(os.environ, PS=str(PS)))
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun("P5.2b")


def _specs(**tags):
    return [s for s in _batch_specs() if all(str((s.get("tags") or {}).get(k)) == str(v) for k, v in tags.items())]


def _one(**tags):
    got = _specs(**tags)
    assert len(got) == 1, (tags, len(got))
    return got[0]


_run, lines = t52._run, t52.lines


@pytest.mark.vortex
def test_bliss_invariant(vx):
    params = {"bliss": (4, 10000), "bliss_t2_i1000": (2, 1000)}
    checked = 0
    for sp in _specs(group="K"):
        mode = sp["tags"]["mode"]
        if mode == "m0":
            continue
        T, I = params[mode]
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt
        xs = lines(txt, "PSDRAMX")
        assert len(xs) == 2
        for ch, x in enumerate(xs):
            # (PSDRAMX shows the knobs in force after the launch, i.e. the host config the schedule restored; the
            # parameters the controller used are in the trace header, written when BLISS first clocked)
            hdr = Path(sp["env"]["PS_BLISS_TRACE"] + f".{ch}").read_text().splitlines()[0]
            assert f"thresh={T} clear={I}" in hdr, hdr
            assert x["bliss_unsafe_grants"] == 0, x
            n, nc, sets = check_bliss_trace(sp["env"]["PS_BLISS_TRACE"] + f".{ch}", T, I)
            assert n == x["bliss_serviced"] > 1000 and nc == x["bliss_clears"] > 0 and sets > 0, (n, nc, sets, x)
            assert x["bliss_bl_grants"] > 0   # blacklisting acted on the schedule
            checked += 1
    assert checked == 6


@pytest.mark.vortex
def test_padc_epochs(vx):
    for sp in _specs(group="P"):
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt
        theta = 255 if sp["tags"]["mode"] == "padc_force" else 50
        eps = [dict((k, float(v)) for k, v in re.findall(r"(\w+)=(-?[0-9.]+)", l))
               for l in re.findall(r"^PSPADCEP (.*)$", txt, re.M)]
        assert len(eps) >= 5
        cyc = [int(e["cycle"]) for e in eps]
        assert all(b - a == 10000 for a, b in zip(cyc, cyc[1:])), cyc[:5]
        low = 0
        for e in eps:
            if e["dxa_bytes"] > 0:
                low = int(100.0 * e["u"] < theta)
            assert int(e["low"]) == low, e
        xs = lines(txt, "PSDRAMX")
        assert all(x["padc_low_viol"] == 0 for x in xs), xs
        if theta == 255:
            assert all(x["padc_low_ticks"] > 0 for x in xs), xs
        else:
            assert all(x["padc_low_ticks"] == 0 for x in xs), xs
    # default theta = FR-FCFS exactly (same binary): the FR-FCFS twin has the RESULTS m0 row's run key (reused)
    for machine in ("base", "lowbw"):
        sp = _one(group="P", mode="padc", machine=machine)
        rp, tp = _run(vx, sp)
        m0 = dict(sp, env={k: v for k, v in sp["env"].items() if k not in ("PS_SCHEDULE", "PS_PADC_TRACE")},
                  tags=dict(sp["tags"], mode="m0"))
        r0, t0 = _run(vx, m0)
        assert (rp["instrs"], rp["cycles"]) == (r0["instrs"], r0["cycles"]), machine
        strip = lambda rows: [{k: v for k, v in r.items() if not k.startswith("ticks_m")} for r in rows]  # noqa: E731
        assert strip(lines(tp, "PSDRAMB")) == strip(lines(t0, "PSDRAMB")), machine


@pytest.mark.vortex
def test_policies_selectable(vx):
    for mode, m in (("bliss", 3), ("padc", 4)):
        for sp in _specs(group="F", mode=mode)[:4]:
            _, txt = _run(vx, sp)
            for b, x in zip(lines(txt, "PSDRAMB"), lines(txt, "PSDRAMX")):
                assert x[f"ticks_m{m}"] == b["ctrl_ticks"] > 0, (mode, b, x)


@pytest.mark.vortex
def test_functional_all_pass(vx):
    specs = _specs(group="F")
    assert len(specs) >= 30
    for sp in specs:
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt, row["log"]


@pytest.mark.vortex
@pytest.mark.parametrize("machine", ["base", "lowbw"])
@pytest.mark.parametrize("case,m", [("twoseg_bliss", 3), ("twoseg_padc", 4)])
def test_two_segment(vx, machine, case, m):
    sp = _one(group="T", case=case, machine=machine)
    st = _one(group="T", case="static", machine=machine)
    W, n = int(sp["tags"]["switch"]), int(sp["env"]["PS_TRACE_EVERY"])
    rs, _ = _run(vx, st)
    row, txt = _run(vx, sp)
    _, rows = pstrace.read_trace(sp["env"]["PS_TRACE"])
    _, rows_s = pstrace.read_trace(st["env"]["PS_TRACE"])
    w, ws = pstrace.work_rows(rows, 0), pstrace.work_rows(rows_s, 0)
    k_sw = next(i for i, r in enumerate(w) if r["work_end"] >= W)
    assert W % n == 0 and w[k_sw]["work_end"] == W
    for i in range(k_sw + 1):
        assert w[i] == ws[i], i
    assert any(w[i] != ws[i] for i in range(k_sw + 1, min(len(w), len(ws))))
    for b, x in zip(lines(txt, "PSDRAMB"), lines(txt, "PSDRAMX")):
        assert b["ticks_m0"] > 0 and x[f"ticks_m{m}"] > 0, (b, x)
    assert row["cycles"] != rs["cycles"]


@pytest.mark.vortex
@pytest.mark.parametrize("case,msg", [("x_policy3", "baseline policy reserved"),
                                      ("x_policy_mode", "a baseline policy (bits 7:4) needs bits 1:0 = 0"),
                                      ("x_policy_high", "a baseline policy (bits 7:4) needs bits 1:0 = 0")])
def test_rejections(vx, case, msg):
    row, txt = _run(vx, _one(group="X", case=case), expect_ok=False)
    assert row["status"] != "ok" and "PASSED!" not in txt
    assert msg in txt, row["log"]


@pytest.mark.vortex
def test_noop_regression_csv():
    rows = list(csv.DictReader((RES / "noop_regression.csv").open()))
    assert len(rows) >= 8 and all(r["identical"] == "True" for r in rows), rows
