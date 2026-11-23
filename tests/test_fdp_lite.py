"""Tests: ps patch fdp-lite (C baseline: FDP-lite, a phase-ignorant feedback throttle of the DXA in-flight depth by
measured accuracy and lateness, evaluated every epoch of E DxaCore cycles; after C27). Semantics: docs/methodology.md
§14.C.b; evidence: results/<run>/.

Non-vortex (cheap):
  test_patch_files            patches/vortex/ps/0010-fdp-lite.patch touches exactly the 9 files, nothing in third_party/.
  test_policy_no_phase_input  the FDP-lite policy code (DxaCore::ps_fdp_tick/_sync/_epoch_end and the
                              socket's input lambda) reads no phase id (mode), no class label, no per-class knob, no
                              schedule position, no KV/weight range: of [dcr_phase] it reads only dxa_fdp()/dxa_fdp_epoch().
  test_replay_rule_unit       the Python replay of the decision rule on hand-made epochs.
  test_stored_schedule_files  every results/<run>/sched/*.sched parses with scripts/mkschedule.py.
Vortex-marked (PS_RUN_VORTEX=1; rows of experiments/<run>.yaml in results/<run>; same specs -> same run keys -> reused):
  test_epoch_trace_replay     every PSFDPEP trace (groups T, K, S): per DxaCore, epoch k ends at t = k*E; level in
                              [0, 4] and depth = FDP_DEPTH[level] in [1, 8]; the level of epoch k+1 = the "next" of
                              epoch k (no change between boundaries); the half-decayed counters and the next level are
                              recomputed independently from the raw epoch deltas and must match exactly; the per-epoch
                              max in flight <= max(depth in force, in flight at the epoch start); the PSFDP summary
                              (epochs, ups, downs, depth_cycles) equals the replay, which pins the depth to the epoch
                              boundaries cycle by cycle (sum over epochs of E*depth + the tail at the last level).
                              Not vacuous: >= 1 up and >= 1 down over the traces.
  test_static_cap_bound       FDP-lite under a static BULK cap 2: max in flight <= 2 (PSDXAC) in every epoch.
  test_off_noop               dxa_fdp=0 equals no schedule (cycles identical), and prints no PSFDP line.
  test_functional_all_pass    every P2 kernel + the multicast apps PASS under FDP-lite with E = 2048.
  test_two_segment            PS_SCHEDULE on the policy knob: FDP-lite -> off at 200,000 instrs and off -> FDP-lite;
                              epochs only while on; window max <= 4 (FDP depth on mix never rises above the initial 4)
                              while on, > 4 once stock; stock-first prefix rows identical to the static stock run.
  test_results_rerun_identical the cap modes and stock re-run on this binary equal the earlier stored rows exactly.
  test_noop_regression_csv    results/<run>/noop_regression.csv: every REG entry identical (base, ps 0002-0009).
"""
import csv
import json
import re
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
TREE = PS / "vortex-trees" / "ps"
RES = PS / "results" / "P5.3b"
PATCH = PS / "patches" / "vortex" / "ps" / "0010-fdp-lite.patch"
sys.path.insert(0, str(PS / "scripts"))
sys.path.insert(0, str(PS / "tests"))
import mkschedule as mk  # noqa: E402
import pstrace  # noqa: E402
import test_dram_2class as t52  # noqa: E402  (helpers: _run)

FILES = ["sim/common/phase_config.h", "sim/simx/barrier_unit.cpp", "sim/simx/barrier_unit.h", "sim/simx/dxa/dxa_core.cpp",
         "sim/simx/dxa/dxa_core.h", "sim/simx/processor.cpp", "sim/simx/ps_schedule.h", "sim/simx/socket.cpp",
         "sim/simx/socket.h"]
# Constants of the policy (dxa_core.h); the replay below is written from methodology §14.C.b, not from the C++.
DEPTH = [1, 2, 4, 6, 8]
INIT, ACC_LOW, LATE_THR = 2, 0.40, 0.01
_run = t52._run


# ------------------------------------------------------------------------------------------------ replay
def replay_epoch(level, s, d):
    """One epoch end. s = (sl, sd, sr, slt) half-decayed counters before, d = raw (dl, dd, dr, dlt) of the epoch.
    Returns (next level, new s, judged)."""
    s = tuple(0.5 * a + 0.5 * float(b) for a, b in zip(s, d))
    judged = d[1] > 0
    if judged:
        acc = min(1.0, s[0] / s[1])
        late = s[3] / s[2] if s[2] > 0.0 else 0.0
        if acc < ACC_LOW:
            level = max(0, level - 1)
        elif late > LATE_THR:
            level = min(len(DEPTH) - 1, level + 1)
    return level, s, judged


def test_replay_rule_unit():
    s0 = (0.0, 0.0, 0.0, 0.0)
    assert replay_epoch(2, s0, (0, 0, 0, 0))[0] == 2                     # nothing to judge: hold
    assert replay_epoch(2, s0, (0, 100, 0, 0))[0] == 1                   # accuracy 0: down
    assert replay_epoch(0, s0, (0, 100, 0, 0))[0] == 0                   # floor
    assert replay_epoch(2, s0, (100, 100, 10, 1))[0] == 3                # accurate, 10 % late: up
    assert replay_epoch(4, s0, (100, 100, 10, 10))[0] == 4               # ceiling
    assert replay_epoch(2, s0, (100, 100, 10, 0))[0] == 2                # accurate, never late: hold
    lv, s, _ = replay_epoch(2, s0, (0, 100, 0, 0))                       # decay: a late read of the earlier fill counts
    assert replay_epoch(lv, s, (100, 0, 1, 1))[2] is False


# ------------------------------------------------------------------------------------------------ non-vortex
def test_patch_files():
    t = PATCH.read_text()
    files = sorted(set(re.findall(r"^diff --git a/(\S+) b/", t, re.M)))
    assert files == FILES, files
    assert not re.search(r"^diff --git a/(.*/)?\.ps_", t, re.M)
    assert "third_party/" not in "".join(files)


def _body(src, sig):
    i = src.index(sig)
    j = src.index("{", i)
    depth, k = 0, j
    while True:
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[j:k + 1]
        k += 1


def test_policy_no_phase_input():
    core = TREE / "sim/simx/dxa/dxa_core.cpp"
    if not core.exists():
        pytest.skip("no ps tree")
    c = core.read_text()
    bodies = [_body(c, f"void DxaCore::{f}()") for f in ("ps_fdp_tick", "ps_fdp_sync", "ps_fdp_epoch_end")]
    sock = (TREE / "sim/simx/socket.cpp").read_text()
    bodies.append(_body(sock, "dxa_core_->ps_fdp_set_source("))
    code = "\n".join(re.sub(r"//[^\n]*", "", b) for b in bodies)
    for bad in (r"\bmode\b", r"dxa_depth", r"ps_cls\b", r"\bps_cls\.", r"mem_class", r"\bps_sched", r"\bkv\[", r"\bwgt\b",
                r"insert_policy", r"dram_sched", r"VX_PS_CLASS", r"phase_id", r"ps_mark", r"PREFILL|DECODE|prefill|decode"):
        assert not re.search(bad, code), bad
    uses = set(re.findall(r"phase_cfg_\.(\w+)", code))
    assert uses == {"dxa_fdp", "dxa_fdp_epoch"}, uses
    hdr = (TREE / "sim/simx/dxa/dxa_core.h").read_text()
    cap = _body(hdr, "uint32_t ps_cls_cap(uint32_t cls) const")
    assert "ps_fdp_on_" in cap and "mode" not in cap


def test_stored_schedule_files():
    files = sorted((RES / "sched").glob("*.sched"))
    if not files:
        pytest.skip("no schedules generated yet")
    assert len(files) >= 7
    for f in files:
        mk.parse(f.read_text())


# ------------------------------------------------------------------------------------------------ vortex helpers
def _batch_specs(name="P5.3b.yaml"):
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
    return vxrun.VxRun("P5.3b")


def _specs(name="P5.3b.yaml", **tags):
    return [s for s in _batch_specs(name) if all(str((s.get("tags") or {}).get(k)) == str(v) for k, v in tags.items())]


def _one(**tags):
    got = _specs(**tags)
    assert len(got) == 1, (tags, len(got))
    return got[0]


def _kv(line):
    d = {}
    for k, v in re.findall(r"(\w+)=(\S+)", line):
        if "," in v:
            d[k] = [int(x) for x in v.split(",")]
        else:
            try:
                d[k] = int(v)
            except ValueError:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
    return d


def epochs(txt):
    out = {}
    for l in re.findall(r"^PSFDPEP (.*)$", txt, re.M):
        e = _kv(l)
        out.setdefault(e["dxa"], []).append(e)
    return out


def psfdp(txt):
    l = re.findall(r"^PSFDP (.*)$", txt, re.M)
    return _kv(l[-1]) if l else None


def psdxac(txt):
    l = re.findall(r"^PSDXAC (.*)$", txt, re.M)
    assert l, "no PSDXAC line"
    return _kv(l[-1])


def check_trace(txt, cap=None):
    """Replay every DxaCore's epoch trace of one constant-schedule launch. Returns (n epochs, ups, downs)."""
    ep = epochs(txt)
    assert ep, "no PSFDPEP lines"
    tot = {"epochs": 0, "ups": 0, "downs": 0, "judged": 0, "depth_cycles": 0}
    for name, es in ep.items():
        level, s = INIT, (0.0, 0.0, 0.0, 0.0)
        for k, e in enumerate(es, 1):
            E = e["E"]
            assert e["epoch"] == k and e["t"] == k * E, (name, k, e)                # boundary = every E cycles
            assert e["level"] == level, (name, k, e)                                 # no change between boundaries
            assert 0 <= e["level"] < len(DEPTH) and 0 <= e["next"] < len(DEPTH)
            assert e["depth"] == DEPTH[e["level"]] and e["next_depth"] == DEPTH[e["next"]]
            assert 1 <= e["depth"] <= 8
            nxt, s, judged = replay_epoch(level, s, (e["d_lmem"], e["d_dxa"], e["rel"], e["late"]))
            assert (e["s_lmem"], e["s_dxa"], e["s_rel"], e["s_late"]) == s, (name, k, e, s)
            assert e["judged"] == int(judged) and e["next"] == nxt, (name, k, e, nxt)
            assert abs(e["next"] - e["level"]) <= 1
            bound = DEPTH[level] if cap is None else min(DEPTH[level], cap)
            assert e["emax"] <= max(bound, e["inf0"]), (name, k, e)
            tot["ups"] += nxt > level
            tot["downs"] += nxt < level
            tot["judged"] += judged
            tot["depth_cycles"] += E * DEPTH[level]
            level = nxt
        tot["epochs"] += len(es)
        tot.setdefault("tail", []).append((name, level, es[-1]["t"], es[-1]["E"]))
    f = psfdp(txt)
    assert f is not None
    assert (f["epochs"], f["ups"], f["downs"], f["judged"]) == (tot["epochs"], tot["ups"], tot["downs"], tot["judged"]), (f, tot)
    # every DxaCore ran for on_cycles / n_dxa cycles; the tail after the last boundary is at the last level
    n = len(ep)
    assert f["on_cycles"] % n == 0, f
    per = f["on_cycles"] // n
    tail = sum((per - t) * DEPTH[lv] for _, lv, t, _ in tot["tail"])
    assert all(0 <= per - t < E for _, _, t, E in tot["tail"]), (per, tot["tail"])
    assert f["depth_cycles"] == tot["depth_cycles"] + tail, (f, tot["depth_cycles"], tail)
    return tot["epochs"], tot["ups"], tot["downs"]


# ------------------------------------------------------------------------------------------------ vortex
@pytest.mark.vortex
def test_epoch_trace_replay(vx):
    n = ups = downs = runs = 0
    for sp in _specs(group="T") + _specs(group="K"):
        row, txt = _run(vx, sp)
        assert re.search(r"^PASSED!?\s*$", txt, re.M), row["log"]
        e, u, d = check_trace(txt, cap=2 if sp["tags"]["group"] == "K" else None)
        n, ups, downs, runs = n + e, ups + u, downs + d, runs + 1
    assert runs >= 14 and n >= 500 and ups >= 1 and downs >= 1, (runs, n, ups, downs)


@pytest.mark.vortex
def test_static_cap_bound(vx):
    row, txt = _run(vx, _one(group="K"))
    x = psdxac(txt)
    assert x["max_inflight"][4] <= 2 and x["gmem_reqs"][4] > 0, x
    assert max(e["emax"] for es in epochs(txt).values() for e in es) <= 2


@pytest.mark.vortex
def test_off_noop(vx):
    for app in ("ps_mix_micro", "ps_decode_attn_gqa"):
        off = [s for s in _specs(group="Z", mode="fdp_off") if s["app"] == app]
        ref = [s for s in _specs(group="Z", mode="m0") if s["app"] == app]
        assert len(off) == len(ref) == 1
        r1, t1 = _run(vx, off[0])
        r0, t0 = _run(vx, ref[0])
        assert (r1["instrs"], r1["cycles"]) == (r0["instrs"], r0["cycles"]), app
        assert psfdp(t1) is None and psfdp(t0) is None and not epochs(t1)


@pytest.mark.vortex
def test_functional_all_pass(vx):
    specs = _specs(group="F")
    assert len(specs) >= 18
    for sp in specs:
        row, txt = _run(vx, sp)
        assert re.search(r"^PASSED!?\s*$", txt, re.M), row["log"]


def _trace_max(rows):
    return [max(r[f"dxa_max_inflight_s{s}"] for s in range(4)) for r in rows]


@pytest.mark.vortex
def test_two_segment(vx):
    W = 200000
    out = {}
    for case in ("static", "static_fdp", "twoseg_fdp_off", "twoseg_off_fdp"):
        sp = _one(group="S", case=case)
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt
        _, rows = pstrace.read_trace(sp["env"]["PS_TRACE"])
        out[case] = (row, pstrace.work_rows(rows, 0), txt)
    # on mix the bulk stream is accurate and never late, so FDP-lite never rises above its initial depth 4
    for case in ("static_fdp", "twoseg_fdp_off", "twoseg_off_fdp"):
        for es in epochs(out[case][2]).values():
            assert all(e["depth"] <= 4 and e["next_depth"] <= 4 for e in es), case
    ms, mf = _trace_max(out["static"][1]), _trace_max(out["static_fdp"][1])
    assert max(mf) <= 4 and max(ms) > 4, (ms, mf)
    w1 = out["twoseg_fdp_off"][1]
    k = next(i for i, r in enumerate(w1) if r["work_end"] >= W)
    m1 = _trace_max(w1)
    assert max(m1[:k + 1]) <= 4, m1[:k + 1]
    assert max(m1[k + 1:]) > 4, m1[k + 1:]
    w2 = out["twoseg_off_fdp"][1]
    k2 = next(i for i, r in enumerate(w2) if r["work_end"] >= W)
    for i in range(k2 + 1):
        assert w2[i] == out["static"][1][i], i      # stock prefix identical to the static run
    m2 = _trace_max(w2)
    assert max(m2[k2 + 2:]) <= 4 and max(m2[:k2 + 1]) > 4, m2
    # epochs: none after the switch-off, none before the switch-on (t restarts at the enable)
    e1 = sum(len(v) for v in epochs(out["twoseg_fdp_off"][2]).values())
    e0 = sum(len(v) for v in epochs(out["static_fdp"][2]).values())
    e2 = sum(len(v) for v in epochs(out["twoseg_off_fdp"][2]).values())
    assert 0 < e1 < e0 and 0 < e2 < e0, (e1, e2, e0)
    for case in ("twoseg_fdp_off", "twoseg_off_fdp"):
        for es in epochs(out[case][2]).values():
            assert [e["t"] for e in es] == [k * es[0]["E"] for k in range(1, len(es) + 1)]


def _rows(step):
    p = PS / "results" / step / "rows.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


@pytest.mark.vortex
def test_results_rerun_identical():
    """Stock cap modes on this binary == the earlier stored rows (same app, args, configs, env)."""
    import vxrun
    fp = vxrun.tree_fingerprint("ps")
    def key(r):
        return (r["app"], r["args"] if isinstance(r["args"], str) else " ".join(r["args"]),
                tuple(r["configs_canonical"]), json.dumps(r.get("env") or {}, sort_keys=True))
    old = {key(r): r for r in _rows("P5.3") if r.get("status") == "ok" and (r.get("tags") or {}).get("group") == "R"}
    n = 0
    for r in _rows("P5.3b"):
        t = r.get("tags") or {}
        if r.get("status") != "ok" or r.get("tree_fp") != fp or t.get("group") != "R" or t.get("policy") not in ("stock", "cap"):
            continue
        o = old.get(key(r))
        assert o is not None, key(r)
        assert (r["instrs"], r["cycles"]) == (o["instrs"], o["cycles"]), key(r)
        n += 1
    assert n == 56, n   # 14 cells (decode 3, prefill 2, mix 2; x 2 machines) x (stock + 3 caps)


@pytest.mark.vortex
def test_noop_regression_csv():
    rows = list(csv.DictReader((RES / "noop_regression.csv").open()))
    assert len(rows) >= 8 and all(r["identical"] == "True" for r in rows), rows
