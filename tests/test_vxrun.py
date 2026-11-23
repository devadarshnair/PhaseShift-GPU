"""Tests for scripts/vxrun.py.

Fixtures in tests/fixtures/vxrun/ (copied from results/S7/logs/, i.e. real smoke runs, with their /usr/bin/time files):
  c1-base.log           smoke case 1 first run (SimX + app build, then sgemm -n32)
  c1-base-rerun.log     smoke case 1 rerun (run only)
  c5-base-rerun.log     smoke case 5 dxa_copy -d1 (PASS string "PASSED" without "!", dxa PERF lines)
  c6-base-rerun.log     smoke case 6 sgemm_tcu_wg_dxa (WGMMA+DXA, 32 threads, 8 warps)
  c7-base-rerun.log     smoke case 7 sgemm -n64, 4 cores, L2, PLRU (l2cache PERF line)
  tier256-rerun.log     TIER case rep 1 (sgemm_tcu_wg_dxa -m 256 -n 256 -k 64, prints "Elapsed time: N ms")
  synthetic-failed.log  c1-base-rerun.log with "PASSED!" replaced by sgemm's failure output (main.cpp:159-161:
                        "Found N errors!" + "FAILED!") and make's error line; its .time file says exit status 2.
  synthetic-truncated.log  c1-base-rerun.log cut in the middle of a PERF line (before the global PERF line and the
                        PASS string), i.e. what a killed run leaves behind.
Non-vortex tests never start a simulator: the runner is injected. The vortex-marked test (PS_RUN_VORTEX=1) re-runs smoke case 1
case 1 through vxrun (blackbox and --direct) with a PRIVATE results dir/ledger under build/tmp/vxrun_live (its jobs
are not in results/lowload_ledger.tsv, but they still take the machine-wide lowload flock, so they never overlap).
"""
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS / "scripts"))
import vxrun  # noqa: E402

FIX = PS / "tests" / "fixtures" / "vxrun"
NAMED = vxrun.load_named_configs(PS / "experiments" / "configs.yaml")


def _s7(case):
    with open(PS / "results/S7/summary.csv", newline="") as f:
        return next(r for r in csv.DictReader(f) if r["case"] == case)


def _log(name):
    return vxrun.parse_log((FIX / name).read_text())


# ------------------------------------------------------------------------------------------------- parsers
@pytest.mark.parametrize("fname,case", [("c1-base-rerun.log", "1"), ("c1-base.log", "1"), ("c5-base-rerun.log", "5"),
                                        ("c6-base-rerun.log", "6"), ("c7-base-rerun.log", "7")])
def test_parse_real_logs_match_s7_summary(fname, case):
    p = _log(fname)
    ref = _s7(case)
    assert p["passed"] and not p["failed_str"]
    assert (p["instrs"], p["cycles"]) == (int(ref["instrs"]), int(ref["cycles"]))
    assert vxrun.classify(0, p) == ("ok", "")


def test_parse_perf_key_values():
    p = _log("c1-base-rerun.log")
    assert p["ipc"] == 0.42 and p["app_elapsed_ms"] == 234
    assert len(p["perf_raw"]) == 6                       # 5 sub-lines + the global line, raw lines kept
    kv = p["perf"]
    assert kv["scheduler.idle_pct"] == 58
    assert kv["scheduler.occupancy"] == 4.0 and kv["scheduler.occupancy.pct"] == 100
    assert kv["memory.loads"] == 69632 and kv["memory.load_lat"] == 11.48
    assert kv["branches.divergent"] == 0 and kv["branches.divergent.rate_pct"] == 0
    assert p["blackbox_configs"] is None or "PERF_ENABLE" in p["blackbox_configs"]

    kv7 = _log("c7-base-rerun.log")["perf"]
    assert kv7["core0.l2cache.reqs"] == 12040
    assert kv7["core0.l2cache.miss_r"] == 2080 and kv7["core0.l2cache.miss_r.hit_pct"] == 79
    assert kv7["core0.l2cache.miss_w"] == 2048 and kv7["core0.l2cache.miss_w.hit_pct"] == 0
    assert kv7["core0.l2cache.evicts"] == 256

    kv5 = _log("c5-base-rerun.log")["perf"]
    assert kv5["core0.dxa.transfers"] == 8 and kv5["dxa.avg_gmem_lat"] == 30.0
    assert kv5["core0.dxa.gmem_dedup.rate_pct"] == 0

    assert _log("c6-base-rerun.log")["perf"]["inst_mix.tcu_pct"] == 59


def test_parse_failed_log():
    p = _log("synthetic-failed.log")
    assert p["failed_str"] and not p["passed"]
    assert p["instrs"] == 48400                           # perf is dumped before the check (main.cpp:156-161)
    t = vxrun.parse_time((FIX / "synthetic-failed.log.time").read_text())
    st, why = vxrun.classify(t["time_exit"], p)
    assert st == "failed" and "FAILED" in why
    assert vxrun.classify(0, p)[0] == "failed"          # a FAILED string fails even with exit 0


def test_parse_truncated_log():
    p = _log("synthetic-truncated.log")
    assert not p["passed"] and p["instrs"] is None and p["cycles"] is None
    assert "inst_mix.alu" not in p["perf"]               # the cut line has no value: ignored, no crash
    st, why = vxrun.classify(0, p)
    assert st == "failed" and "truncated" in why
    assert vxrun.classify(143, p)[0] == "failed"


def test_classify_deferred_and_timeout():
    p = _log("synthetic-truncated.log")
    assert vxrun.classify(75, p)[0] == "deferred"
    assert vxrun.classify(124, p)[0] == "timeout"
    # 137 = SIGKILL: a timeout only when the run lasted its whole --timeout (timeout's -k escalation) ...
    assert vxrun.classify(137, p, wall_s=630.0, timeout_s=600)[0] == "timeout"
    # ... otherwise someone else killed it (OOM killer) -> oom, not timeout
    assert vxrun.classify(137, p, wall_s=12.0, timeout_s=600)[0] == "oom"
    assert vxrun.classify(137, p)[0] == "oom"


def test_classify_memory_cap_abort_is_oom():
    text = (FIX / "c1-base-rerun.log").read_text().replace(
        "PASSED!\n", "terminate called after throwing an instance of 'std::bad_alloc'\n  what():  std::bad_alloc\n")
    p = vxrun.parse_log(text)
    assert p["mem_error"] and vxrun.classify(134, p)[0] == "oom"
    assert not _log("synthetic-failed.log")["mem_error"]


def test_parse_elapsed_time_format():
    """sgemv and sgemm_tcu_wg_dxa print 'Elapsed time: %lg ms' (real TIER rerun log)."""
    p = _log("tier256-rerun.log")
    assert p["app_elapsed_ms"] == 743 and (p["instrs"], p["cycles"]) == (27552, 58679) and p["passed"]
    assert vxrun.parse_log("Elapsed time: 12.5 ms\n")["app_elapsed_ms"] == 12.5


def test_parse_time_file_matches_ledger():
    t = vxrun.parse_time((FIX / "c1-base-rerun.log.time").read_text())
    ledger = (PS / "results/lowload_ledger.tsv").read_text().splitlines()
    end = [ln.split("\t") for ln in ledger if ln.startswith("end\t") and "\tS7-c1-base-rerun\t" in ln][-1]
    assert t["maxrss_kb"] == int(end[4])
    assert t["time_exit"] == 0 and t["wall_s"] is not None and 0 < t["wall_s"] <= float(end[5])
    assert vxrun.parse_time("") == {"wall_s": None, "maxrss_kb": None, "user_s": None, "sys_s": None,
                                    "time_exit": None}
    assert vxrun.parse_time("\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.5\n")["wall_s"] == 3723.5


# ---------------------------------------------------------------------------------------------- keys / configs
def _spec(**kw):
    d = {"tree": "base", "app": "sgemm", "args": "-n32", "perf": 1}
    d.update(kw)
    return d


def _key(spec, fp="F", ah=""):
    s = vxrun.resolve(spec, NAMED)
    return vxrun.run_key(s, fp, ah)


def test_canonical_configs_order_and_last_wins():
    assert vxrun.canonical_configs(["-DB=1", "-DA", "-DB=2"]) == ["-DA", "-DB=2"]
    assert vxrun.canonical_configs(["-DB=2", "-DA"]) == vxrun.canonical_configs(["-DA", "-DB=1", "-DB=2"])


def test_named_config_equals_explicit_spelling():
    named = vxrun.resolve(_spec(config="REG-CFG"), NAMED)
    explicit = vxrun.resolve(_spec(configs="-DVX_CFG_L2_REPL_POLICY=2  -DVX_CFG_DCACHE_REPL_POLICY=2", cores=4,
                                   l2=True), NAMED)
    assert named["configs_canonical"] == explicit["configs_canonical"]
    assert _key(_spec(config="REG-CFG")) == _key(_spec(configs="-DVX_CFG_L2_REPL_POLICY=2 "
                                                               "-DVX_CFG_DCACHE_REPL_POLICY=2", cores=4, l2=True))
    assert _key(_spec(config="REG-CFG")) == _key(_spec(config="smoke-l2-plru-4c"))


def test_effective_configs_literal_matches_blackbox_echo():
    """The CONFIGS literal vxrun predicts (used by --direct builds) equals what blackbox echoed in smoke case 6 and 7."""
    for fname, cfg in (("c7-base-rerun.log", "smoke-l2-plru-4c"), ("c6-base-rerun.log", "smoke-wgmma-dxa")):
        echoed = _log(fname)["blackbox_configs"]
        assert vxrun.resolve(_spec(config=cfg), NAMED)["configs_effective"] == echoed


def test_structural_flag_overrides_configs_macro():
    s = vxrun.resolve(_spec(configs="-DVX_CFG_NUM_THREADS=8 -DFOO", threads=32), NAMED)
    assert "NUM_THREADS" not in s["configs_env"] and "-DFOO" in s["configs_env"]
    assert "-DVX_CFG_NUM_THREADS=32" in s["configs_canonical"]
    assert s["blackbox_flags"] == ["--threads=32", "--perf=1"]


def test_key_is_stable_and_sensitive():
    k = _key(_spec())
    assert k == _key(_spec()) and k == _key(_spec(args="  -n32 "))          # whitespace-normalised args
    assert len(k) == 64
    changed = [_key(_spec(args="-n64")), _key(_spec(perf=5)), _key(_spec(tree="ps")), _key(_spec(app="sgemv")),
               _key(_spec(env={"A": "1"})), _key(_spec(rep=1)), _key(_spec(direct=True)),
               _key(_spec(configs="-DX")), _key(_spec(), fp="G"), _key(_spec(), ah="h")]
    assert len(set(changed + [k])) == len(changed) + 1
    # tags do not change the key
    assert _key(_spec(tags={"case": "a"})) == k


def test_key_recipe_pinned():
    """The key material/format must not drift silently (rows.jsonl skip logic depends on it)."""
    s = vxrun.resolve({"tree": "base", "app": "sgemm", "args": "-n32", "perf": 1}, {})
    material = {"tree": "base", "tree_fp": "F", "configs": ["-DPERF_ENABLE"], "app": "sgemm", "app_src": "",
                "args": "-n32", "perf": 1, "env": {}, "rep": 0, "direct": False}
    import hashlib
    expect = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert vxrun.run_key(s, "F", "") == expect


def test_unknown_config_and_bad_tree_rejected():
    with pytest.raises(SystemExit):
        vxrun.resolve(_spec(config="nope"), NAMED)
    with pytest.raises(SystemExit):
        vxrun.resolve(_spec(tree="upstream"), NAMED)


@pytest.mark.skipif(not (PS / "vortex-trees/base/.ps_patchset").exists(), reason="vortex-trees/base not built")
def test_tree_fingerprint_deterministic():
    vxrun._FP_CACHE.clear()
    a = vxrun.tree_fingerprint("base")
    vxrun._FP_CACHE.clear()
    assert vxrun.tree_fingerprint("base") == a and len(a) == 64
    heads = vxrun.tree_heads("base")
    assert len(heads["tree"]) == 40 and len(heads.get("ramulator", "0" * 40)) == 40


def test_app_source_hash(tmp_path):
    assert vxrun.app_source_hash("sgemm", tmp_path, tmp_path) == ""
    (tmp_path / "kernels/gemv_f32").mkdir(parents=True)
    (tmp_path / "kernels/common").mkdir(parents=True)
    (tmp_path / "kernels/gemv_f32/main.cpp").write_text("int main(){}")
    h1 = vxrun.app_source_hash("ps_gemv_f32", tmp_path / "build", tmp_path)
    (tmp_path / "kernels/common/ps_common.h").write_text("//")
    h2 = vxrun.app_source_hash("ps_gemv_f32", tmp_path / "build", tmp_path)
    assert h1 and h2 and h1 != h2


# ------------------------------------------------------------------------------------- runner / skip / store
class FakeRunner:
    """Writes a fixture log + time file where lowload would, returns a chosen exit code; records the commands."""

    def __init__(self, fixture="c1-base-rerun.log", code=0, app_time=True):
        self.fixture, self.code, self.calls, self.app_time = fixture, code, [], app_time

    def __call__(self, cmd, cwd, env, log_path, time_path):
        self.calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        if self.fixture:
            shutil.copy(FIX / self.fixture, log_path)
            tf = FIX / (self.fixture + ".time")
            if tf.exists():
                shutil.copy(tf, time_path)
        if "vxrun-direct" in cmd and self.app_time:          # --direct: the app's own time file
            shutil.copy(FIX / "c1-base-rerun.log.time", cmd[cmd.index("vxrun-direct") + 5])
        return self.code


def _vx(tmp_path, runner):
    v = vxrun.VxRun("PT", ps=tmp_path, runner=runner, named=NAMED, results=tmp_path / "results")
    v.fingerprint = lambda tree: "FP-" + tree
    v.app_hash = lambda app, bdir: ""
    v.heads = lambda tree: {"tree": "0" * 40}
    return v


def test_run_appends_row_and_skips_done_key(tmp_path):
    fr = FakeRunner()
    v = _vx(tmp_path, fr)
    row = v.run(_spec(tags={"case": "1"}), timeout_s=600)
    assert row["status"] == "ok" and row["instrs"] == 48400 and row["cycles"] == 115353
    assert row["maxrss_kb"] == 19176 and row["tags"] == {"case": "1"} and row["rss_req_gb"] == 2.0
    cmd = fr.calls[0]["cmd"]
    assert cmd[0].endswith("scripts/lowload.sh") and cmd[cmd.index("--rss") + 1] == "2"
    assert cmd[cmd.index("--mem") + 1] == "4" and cmd[cmd.index("--label") + 1] == row["label"]
    assert cmd[cmd.index("--") + 1:cmd.index("--") + 5] == ["timeout", "-k", "30", "600"]
    assert "./ci/blackbox.sh" in cmd and "--driver=simx" in cmd and "--app=sgemm" in cmd
    assert "--args=-n32" in cmd and cmd[-1] == "--perf=1" and "CONFIGS=" in cmd
    assert fr.calls[0]["cwd"] == tmp_path / "build" / "vortex-base"
    env = fr.calls[0]["env"]
    assert env["PS_STEP"] == "PT" and env["TMPDIR"] == str(tmp_path / "build" / "tmp")
    assert env["PS_RESULTS_DIR"] == str(tmp_path / "results")
    assert (tmp_path / "results/PT/.heartbeat").exists()
    # row has everything needed to reproduce the run
    for k in ("cmd", "cwd", "tree", "tree_fp", "configs_env", "configs_effective", "configs_canonical", "app",
              "args", "perf", "env", "build_dir", "log"):
        assert k in row
    # same spec again -> skipped, runner not called, no new row
    again = v.run(_spec())
    assert again.get("skipped") and len(fr.calls) == 1
    assert len(v.store.load()) == 1
    # a different key runs; --force re-runs a done key
    v.run(_spec(args="-n64"))
    v.run(_spec(), force=True)
    assert len(fr.calls) == 3 and len(v.store.load()) == 3
    # rss estimate now comes from earlier rows: max(1.2 x 19176 KB, 0.6 GB floor) = 0.6
    v.run(_spec(rep=1))
    assert v.store.load()[-1]["rss_req_gb"] == 0.6


def test_failed_deferred_timeout_rows_are_not_skipped(tmp_path):
    for fixture, code, status in (("synthetic-failed.log", 2, "failed"), (None, 75, "deferred"),
                                  ("synthetic-truncated.log", 124, "timeout"),
                                  ("synthetic-truncated.log", 0, "failed")):
        fr = FakeRunner(fixture, code)
        v = _vx(tmp_path / status / str(code), fr)
        assert v.run(_spec())["status"] == status
        assert v.run(_spec())["status"] == status          # not ok -> runs again
        assert len(fr.calls) == 2


def test_direct_mode_one_job(tmp_path):
    """--direct = ONE lowload job: make (CONFIGS in the environment) then the app under its own /usr/bin/time."""
    fr = FakeRunner()
    v = _vx(tmp_path, fr)
    row = v.run(_spec(config="REG-CFG", direct=True, args="-n128"), timeout_s=60)
    assert row["status"] == "ok" and len(fr.calls) == 1
    assert row["simx_maxrss_kb"] == row["maxrss_kb"] == 19176     # from the app's own time file
    cmd = fr.calls[0]["cmd"]
    inner = cmd[cmd.index("--") + 5:]
    s = vxrun.resolve(_spec(config="REG-CFG"), NAMED)
    i = inner.index("vxrun-direct")
    assert inner[:3] == ["env", "bash", "-c"] and inner[3] == vxrun.DIRECT_SCRIPT
    app_dir = tmp_path / "build/vortex-base/tests/regression/sgemm"
    assert inner[i + 1:i + 4] == [s["configs_effective"], str(app_dir), "sgemm"]
    assert inner[i + 5].endswith(".log.app.time")
    run = inner[i + 6:]
    assert run[0] == "env" and run[1].startswith("LD_LIBRARY_PATH=" + str(tmp_path / "build/vortex-base/sw/runtime"))
    assert "VORTEX_DRIVER=simx" in run and "VORTEX_PROFILING=1" in run and run[-2:] == ["./sgemm", "-n128"]
    # the script really builds with CONFIGS in the environment and times only the app
    assert 'env CONFIGS="$cfg" make -C "$dir" "$app" kernel.vxbin runtime-simx' in vxrun.DIRECT_SCRIPT
    assert 'exec /usr/bin/time -v -o "$atime" "$@"' in vxrun.DIRECT_SCRIPT
    # failing direct build (no app time file written) -> failed row
    fb = FakeRunner(None, 2, app_time=False)
    r2 = _vx(tmp_path / "b", fb).run(_spec(direct=True))
    assert r2["status"] == "failed" and "direct-mode build failed" in r2["reason"] and len(fb.calls) == 1


def test_direct_script_really_runs(tmp_path):
    """Execute DIRECT_SCRIPT with a fake make (on PATH) and a fake app: build log written, app timed separately."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "make").write_text('#!/bin/sh\necho "make CONFIGS=$CONFIGS $*"\n')
    (bindir / "make").chmod(0o755)
    app = tmp_path / "app"
    app.mkdir()
    (app / "a").write_text('#!/bin/sh\necho "PASSED! $*"\n')
    (app / "a").chmod(0o755)
    blog, atime = tmp_path / "b.log", tmp_path / "a.time"
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    r = subprocess.run(["bash", "-c", vxrun.DIRECT_SCRIPT, "x", "-DA -DB=1", str(app), "a", str(blog), str(atime),
                        "./a", "-n", "8"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "PASSED! -n 8" in r.stdout
    assert "make CONFIGS=-DA -DB=1 -C " in blog.read_text() and "kernel.vxbin runtime-simx" in blog.read_text()
    assert vxrun.parse_time(atime.read_text())["maxrss_kb"] > 0
    (bindir / "make").write_text("#!/bin/sh\nexit 2\n")
    atime.unlink()
    r = subprocess.run(["bash", "-c", vxrun.DIRECT_SCRIPT, "x", "", str(app), "a", str(blog), str(atime), "./a"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "vxrun-direct: build failed" in r.stdout and not atime.exists()


_WRITER = textwrap.dedent("""
    import sys, os, time
    sys.path.insert(0, {scripts!r})
    import vxrun
    st = vxrun.RowStore({path!r})
    i = 0
    while True:
        st.append({{"run_key": "k%d" % i, "status": "ok", "pad": "x" * 2000}})
        i += 1
        if i == 50:
            open({flag!r}, "w").close()
""")


def test_jsonl_survives_killed_writer(tmp_path):
    path = tmp_path / "rows.jsonl"
    flag = tmp_path / "started"
    p = subprocess.Popen([sys.executable, "-c", _WRITER.format(scripts=str(PS / "scripts"), path=str(path),
                                                               flag=str(flag))])
    try:
        for _ in range(600):
            if flag.exists():
                break
            import time
            time.sleep(0.05)
        assert flag.exists(), "writer did not start"
    finally:
        p.send_signal(signal.SIGKILL)
        p.wait()
    rows = vxrun.RowStore(path).load()
    assert len(rows) >= 50
    assert [r["run_key"] for r in rows] == [f"k{i}" for i in range(len(rows))]   # no row lost or garbled
    # a writer killed in the middle of a line (simulated: partial JSON, no newline) does not poison the store
    with open(path, "ab") as f:
        f.write(b'{"run_key": "partial", "sta')
    st = vxrun.RowStore(path)
    assert len(st.load()) == len(rows)
    st.append({"run_key": "after", "status": "ok"})
    after = st.load()
    assert len(after) == len(rows) + 1 and after[-1]["run_key"] == "after"
    assert path.read_bytes().endswith(b"\n")
    assert "after" in st.ok_keys() and "partial" not in st.ok_keys()


# ------------------------------------------------------------------------------------------------------ batch
def test_expand_batch_grid_sort_and_reps():
    doc = {"step": "PT", "defaults": {"tree": "base", "perf": 1, "timeout": 99},
           "runs": [{"app": "sgemm", "args": ["-n64", "-n128"], "config": ["REG-CFG", "smoke-default"]},
                    {"app": "sgemv", "args": "-m512 -n512", "config": "smoke-default", "reps": 2},
                    {"app": "sgemm", "args": "-n128", "cores": [8, 16], "l2": True,
                     "configs": "-DVX_CFG_DCACHE_REPL_POLICY=2 -DVX_CFG_L2_REPL_POLICY=2"}]}
    specs = vxrun.expand_batch(doc, NAMED)
    assert len(specs) == 4 + 2 + 2
    keys = [" ".join(vxrun.resolve({k: v for k, v in s.items() if not k.startswith("_")}, NAMED)
                     ["configs_canonical"]) for s in specs]
    # grouped: each distinct CONFIGS appears in one contiguous block
    blocks = [k for i, k in enumerate(keys) if i == 0 or keys[i - 1] != k]
    assert len(blocks) == len(set(keys)) == 4
    sgemv = [s for s in specs if s["app"] == "sgemv"]
    assert [s["rep"] for s in sgemv] == [0, 1]
    assert all(s["timeout"] == 99 for s in specs)


def test_run_batch_stop_on_fail_and_continue(tmp_path):
    batch = tmp_path / "b.yaml"
    batch.write_text("step: PT\ndefaults: {tree: base}\nruns:\n"
                     "  - {app: sgemm, args: ['-n32', '-n64', '-n128'], config: smoke-default}\n")
    out = open(os.devnull, "w")
    fr = FakeRunner("synthetic-failed.log", 2)
    rc = vxrun.run_batch(batch, vx_factory=lambda step: _vx(tmp_path / "a", fr), stop_on_fail=True, out=out)
    assert rc == 1 and len(fr.calls) == 1
    fr2 = FakeRunner("synthetic-failed.log", 2)
    rc = vxrun.run_batch(batch, vx_factory=lambda step: _vx(tmp_path / "b", fr2), out=out)
    assert rc == 1 and len(fr2.calls) == 3                 # sweeps record failures and continue
    fr3 = FakeRunner()
    rc = vxrun.run_batch(batch, vx_factory=lambda step: _vx(tmp_path / "c", fr3), out=out)
    assert rc == 0 and len(fr3.calls) == 3
    rc = vxrun.run_batch(batch, vx_factory=lambda step: _vx(tmp_path / "c", fr3), out=out)
    assert rc == 0 and len(fr3.calls) == 3                 # restart: all keys done -> nothing re-run
    fr4 = FakeRunner(None, 75)
    rc = vxrun.run_batch(batch, vx_factory=lambda step: _vx(tmp_path / "d", fr4), out=out)
    assert rc == 75 and len(fr4.calls) == 1                # deferred stops the batch


def test_estimate_rss():
    s = vxrun.resolve(_spec(), NAMED)
    assert vxrun.estimate_rss_gb([], s) == 2.0
    rows = [{"tree": "base", "app": "sgemm", "configs_canonical": s["configs_canonical"], "maxrss_kb": 2 * 1048576},
            {"tree": "ps", "app": "sgemm", "configs_canonical": s["configs_canonical"], "maxrss_kb": 9 * 1048576}]
    assert vxrun.estimate_rss_gb(rows, s) == 2.4


def test_cli_help_runs():
    r = subprocess.run([sys.executable, "-B", str(PS / "scripts/vxrun.py"), "run", "--help"], capture_output=True,
                       text=True)
    assert r.returncode == 0 and "--direct" in r.stdout


# ------------------------------------------------------------------------------------------------------ live
@pytest.mark.vortex
def test_live_s7_case1_same_cycles():
    """Re-run smoke case 1 (sgemm -n32, base, CONFIGS="", perf 1) through vxrun -> same instrs/cycles as the stored smoke run; then the
    --direct variant (app binary only) -> same instrs/cycles. Private results dir + ledger; DEFERRED -> skip."""
    priv = PS / "build" / "tmp" / "vxrun_live"
    if priv.exists():
        shutil.rmtree(priv)
    v = vxrun.VxRun("P0.1-live", results=priv)
    ref = _s7("1")
    row = v.run({"tree": "base", "app": "sgemm", "args": "-n32", "config": "smoke-default", "perf": 1},
                timeout_s=3600)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    assert row["status"] == "ok", row["reason"]
    assert (row["instrs"], row["cycles"]) == (int(ref["instrs"]), int(ref["cycles"]))
    assert row["maxrss_kb"] > 0 and row["wall_s"] > 0
    d = v.run({"tree": "base", "app": "sgemm", "args": "-n32", "config": "smoke-default", "perf": 1, "direct": True},
              timeout_s=3600)
    if d["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    assert d["status"] == "ok", d["reason"]
    assert (d["instrs"], d["cycles"]) == (int(ref["instrs"]), int(ref["cycles"]))
    assert d["simx_maxrss_kb"] > 0 and not any(d["rebuilt"].values())
    assert d["job_wall_s"] >= d["wall_s"] and len(d["tree_heads"]["tree"]) == 40
    # skip logic on the real store
    assert v.run({"tree": "base", "app": "sgemm", "args": "-n32", "config": "smoke-default"}).get("skipped")


# ------------------------------------------------------------------------- file content in the run key
def _fkey(tmp_path, spec, fp="F"):
    """Key through resolve + file_hashes (the path plan() takes), files resolved against tmp_path."""
    s = vxrun.resolve(spec, NAMED)
    s["file_hashes"] = vxrun.file_hashes(s["args"], s["env"], [tmp_path])
    return vxrun.run_key(s, fp, "")


def test_key_covers_argument_file_content(tmp_path):
    f = tmp_path / "sw_star.json"
    f.write_text('{"a": 1}')
    spec = _spec(app="ps_llama", args=f"-V {f}")
    k1 = _fkey(tmp_path, spec)
    assert k1 == _fkey(tmp_path, spec)                              # deterministic
    f.write_text('{"a": 2}')                                        # same path, edited content
    k2 = _fkey(tmp_path, spec)
    assert k2 != k1
    f.write_text('{"a": 1}')                                        # content restored -> key restored
    assert _fkey(tmp_path, spec) == k1
    g = tmp_path / "renamed.json"                                   # rename to identical content: content part same
    g.write_text('{"a": 1}')
    assert vxrun.file_hashes(f"-V {g}", {}, [tmp_path]) == vxrun.file_hashes(f"-V {f}", {}, [tmp_path])
    assert list(vxrun.file_hashes("-V sw_star.json", {}, [tmp_path]).values()) == \
        list(vxrun.file_hashes(f"-V {g}", {}, [tmp_path]).values())  # relative path resolves too
    assert "=" not in k1 and len(vxrun.file_hashes(f"--knobs={f}", {}, [tmp_path])) == 1   # --opt=file form


def test_key_covers_env_file_content_and_skips_outputs(tmp_path):
    sch = tmp_path / "x.sched"
    sch.write_text("0 a=1\n")
    spec = _spec(app="ps_gemv_f32", env={"PS_SCHEDULE": str(sch), "PS_SCHEDULE_UNIT": "instrs"})
    k1 = _fkey(tmp_path, spec)
    sch.write_text("0 a=2\n")                                       # PS_SCHEDULE content change changes the key
    assert _fkey(tmp_path, spec) != k1
    # output files must not enter the key: PS_TRACE (env) and the value of -o exist after a run but are not inputs
    out = tmp_path / "trace.csv"
    spec_t = _spec(env={"PS_TRACE": str(out)}, args=f"-n 2 -o {out}")
    before = _fkey(tmp_path, spec_t)
    out.write_text("written by the run\n")
    assert _fkey(tmp_path, spec_t) == before
    assert vxrun.file_hashes(f"-o {out}", {"PS_TRACE": str(out)}, [tmp_path]) == {}


def test_key_unchanged_when_no_file_named(tmp_path):
    for spec in (_spec(), _spec(args="-n64 -k 3", env={"A": "1"}), _spec(app="ps_rope", args="-b 64 -d 512 -S 64 -l 0")):
        s = vxrun.resolve(spec, NAMED)
        assert vxrun.file_hashes(s["args"], s["env"], [tmp_path / "nonexistent-dir"]) == {}
        s["file_hashes"] = {}
        assert vxrun.run_key(s, "F", "") == _key(spec)                         # identical to the pre-file-hash recipe
    test_key_recipe_pinned()                                                   # and the pinned recipe still holds


def test_old_rows_resolve_via_legacy_key(tmp_path):
    f = tmp_path / "sch.sched"
    f.write_text("0\n")
    fr = FakeRunner()
    v = _vx(tmp_path, fr)
    spec = _spec(env={"PS_SCHEDULE": str(f)})
    s = v.plan(spec)
    assert s["file_hashes"] and s["legacy_key"] and s["legacy_key"] != s["run_key"]
    old = {"run_key": s["legacy_key"], "status": "ok", "vxrun_version": 2, "started": "2999-01-01T00:00:00+00:00"}
    v.store.append(old)
    assert v.run(spec).get("legacy_key_match") and not fr.calls                # file older than the row: old row stands
    f.write_text("1\n")
    old2 = dict(old, started="2000-01-01T00:00:00+00:00")                       # file edited after that row ran
    v2 = _vx(tmp_path / "b", FakeRunner())
    v2.store.append(old2)
    assert v2.legacy_hit(v2.plan(spec), v2.store.load()) is None
    # a legacy row of the current version (no such thing for the new recipe) is never accepted
    v3 = _vx(tmp_path / "c", FakeRunner())
    v3.store.append(dict(old, vxrun_version=3))
    assert v3.legacy_hit(v3.plan(spec), v3.store.load()) is None


def test_row_records_file_hashes_and_provenance(tmp_path):
    f = tmp_path / "k.json"
    f.write_text("{}")
    v = _vx(tmp_path, FakeRunner())
    adir = tmp_path / "build/vortex-base/tests/regression/sgemm"
    adir.mkdir(parents=True)
    (adir / "kernel.vxbin").write_bytes(b"ELF")
    row = v.run(_spec(args=f"-K {f}"))
    assert row["file_hashes"] and row["provenance"]["kernel_vxbin"] and row["vxrun_version"] == 4
    # provenance is not in the key: changing the binary does not change the run key
    (adir / "kernel.vxbin").write_bytes(b"ELF2")
    assert v.plan(_spec(args=f"-K {f}"))["run_key"] == row["run_key"]


def test_stale_runtime_objects_guard(tmp_path):
    src, bd = tmp_path / "src", tmp_path / "build/vortex-base"
    (src / "sim/simx").mkdir(parents=True)
    (bd / "sw/runtime/obj").mkdir(parents=True)
    hdr, lib, obj = src / "sim/simx/dram_sim.h", bd / "sw/runtime/libsimx.so", bd / "sw/runtime/obj/memory.o"
    for p in (hdr, lib, obj):
        p.write_text("x")
    os.utime(hdr, (1000, 1000)); os.utime(lib, (2000, 2000))
    assert vxrun.stale_runtime_objects(src, bd) == []                          # library newer than every header
    os.utime(hdr, (3000, 3000))                                                # header edited after the library
    assert vxrun.stale_runtime_objects(src, bd) == [obj]
    v = _vx(tmp_path, FakeRunner())
    (tmp_path / "vortex-trees").mkdir()
    (tmp_path / "vortex-trees/base").symlink_to(src)
    row = v.run(_spec())
    assert row["stale_objs_dropped"] == 1 and not obj.exists()


# ------------------------------------------------------------- regression fixes (2026-11-14)
def _mk_ps_app(root):
    (root / "kernels/gemv_f32").mkdir(parents=True)
    (root / "kernels/common").mkdir(parents=True)
    (root / "kernels/gemv_f32/main.cpp").write_text('#include "ps_common.h"\nint main(){ return 0; }\n')
    (root / "kernels/gemv_f32/kernel.cpp").write_text("void k(){}\n")
    (root / "kernels/common/ps_common.h").write_text("// header\nint f(){ return 1; }\n")
    (root / "kernels/common/unused.h").write_text("int unused;\n")


def test_app_dep_hash_ignores_comments_and_unrelated_headers(tmp_path):
    """D1: comment edits and edits of an uncompiled kernels/common header do not change the key; code edits do."""
    _mk_ps_app(tmp_path)
    h = lambda: vxrun.app_dep_hash("ps_gemv_f32", tmp_path / "build", tmp_path)
    h0 = h()
    assert h0 and vxrun.app_dep_hash("sgemm", tmp_path, tmp_path) == ""
    (tmp_path / "kernels/common/ps_common.h").write_text(
        "// a rewritten, longer\n// comment block\nint f(){   return 1; } /* trailing */\n")
    assert h() == h0                                        # comment + whitespace only
    (tmp_path / "kernels/common/unused.h").write_text("int unused2;\n")
    assert h() == h0                                        # never #included by the app
    (tmp_path / "kernels/common/ps_common.h").write_text("int f(){ return 2; }\n")
    assert h() != h0                                        # real code edit of an included header
    h1 = h()
    (tmp_path / "kernels/gemv_f32/kernel.cpp").write_text("void k(){ int x = 1; }\n")
    assert h() != h1                                        # device source edit
    # the legacy hash still sees every byte of kernels/common (so it differs from the new one and is comment-sensitive)
    l0 = vxrun.app_source_hash("ps_gemv_f32", tmp_path / "build", tmp_path)
    (tmp_path / "kernels/common/unused.h").write_text("// c\nint unused2;\n")
    assert vxrun.app_source_hash("ps_gemv_f32", tmp_path / "build", tmp_path) != l0


def test_strip_comments_respects_literals():
    t = 'const char* u = "http://x"; int a = 1\'000; // gone\n/* multi\nline */ int b;\n'
    out = vxrun.strip_comments(t)
    assert "http://x" in out and "gone" not in out and "multi" not in out and "1'000" in out


def test_legacy_app_hash_rows_stay_valid(tmp_path):
    """D1: a v<=3 ok row keyed with the whole-directory hash still resolves; a v4 row under that key does not."""
    _mk_ps_app(tmp_path)
    fr = FakeRunner()
    v = _vx(tmp_path, fr)
    v.app_hash = lambda app, bdir: vxrun.app_dep_hash(app, bdir, tmp_path)
    v.legacy_app_hash = lambda app, bdir: vxrun.app_source_hash(app, bdir, tmp_path)
    spec = _spec(app="ps_gemv_f32")
    s = v.plan(spec)
    assert s["legacy_app_key"] and s["legacy_app_key"] != s["run_key"]
    v.store.append({"run_key": s["legacy_app_key"], "status": "ok", "vxrun_version": 3})
    assert v.run(spec).get("skipped") and not fr.calls            # old row resolved, nothing re-run
    v2 = _vx(tmp_path / "b", FakeRunner())
    v2.app_hash, v2.legacy_app_hash = v.app_hash, v.legacy_app_hash
    v2.store.append({"run_key": s["legacy_app_key"], "status": "ok", "vxrun_version": 4})
    assert v2.legacy_hit(v2.plan(spec), v2.store.load()) is None
    # edit the legacy-only-visible text (a comment): the new key is unchanged, the old row's key no longer matches
    (tmp_path / "kernels/common/unused.h").write_text("// c\nint unused;\n")
    s2 = v.plan(spec)
    assert s2["run_key"] == s["run_key"] and s2["legacy_app_key"] != s["legacy_app_key"]


def test_expect_fail_is_cached(tmp_path):
    """D2: a failed row with the same key is a hit only for a spec with `expect: fail`."""
    fr = FakeRunner(code=1)
    v = _vx(tmp_path, fr)
    spec = _spec(expect="fail")
    row = v.run(spec)
    assert row["status"] == "failed" and row["expected_failure"] and len(fr.calls) == 1
    again = v.run(spec)
    assert again.get("skipped") and again["expected_failure"] and len(fr.calls) == 1      # cached
    v.run(_spec())                                                                         # no expect: runs again
    assert len(fr.calls) == 2
    with pytest.raises(SystemExit):
        vxrun.resolve(_spec(expect="maybe"), NAMED)


def test_attached_output_options_not_hashed(tmp_path):
    """D10: --out=FILE, --output=FILE and -oFILE name outputs, so an existing file must not enter the key."""
    f = tmp_path / "o.bin"
    f.write_text("x")
    for args in (f"--out={f}", f"--output={f}", f"-o{f}", f"-o {f}", f"--out {f}"):
        assert vxrun.file_hashes(args, {}, [tmp_path]) == {}, args
    assert vxrun.file_hashes(f"--in={f}", {}, [tmp_path])            # inputs are still hashed
    assert vxrun.file_hashes(f"-K {f}", {}, [tmp_path])


# ------------------------------------------------------------- per-build-tree lock + config-stamp sanity (BUG-QUEUE)
def test_tree_lock_held_across_run_and_serialises(tmp_path):
    """The flock on <build_dir>/.vxrun.lock is held while the runner (build+run) executes, and released afterwards."""
    import fcntl
    bdir = tmp_path / "build" / "vortex-base"
    bdir.mkdir(parents=True)
    seen = {}

    def runner(cmd, cwd, env, log_path, time_path):
        fd = os.open(bdir / ".vxrun.lock", os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen["held"] = False
            except BlockingIOError:
                seen["held"] = True
        finally:
            os.close(fd)
        return FakeRunner()(cmd, cwd, env, log_path, time_path)

    v = _vx(tmp_path, runner)
    assert v.run(_spec())["status"] == "ok"
    assert seen["held"] is True
    fd = os.open(bdir / ".vxrun.lock", os.O_RDWR)           # released after the run
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(fd)


def test_tree_lock_blocks_second_process(tmp_path):
    import time
    lp = tmp_path / ".vxrun.lock"
    code = ("import sys,time; sys.path.insert(0, %r); import vxrun, pathlib; t=time.time(); "
            "fd=vxrun.lock_tree(pathlib.Path(%r)); print(round(time.time()-t,2))" % (str(PS / "scripts"), str(lp)))
    fd = vxrun.lock_tree(lp)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    time.sleep(1.0)
    assert p.poll() is None                                  # still waiting on our lock
    os.close(fd)
    out, _ = p.communicate(timeout=20)
    assert float(out) >= 0.8


def test_tree_lock_path_fallback(tmp_path):
    (tmp_path / "b").mkdir()
    assert vxrun.tree_lock_path(tmp_path / "b", tmp_path / "r") == tmp_path / "b" / ".vxrun.lock"
    assert vxrun.tree_lock_path(tmp_path / "nob", tmp_path / "r") == tmp_path / "r" / ".vxrun.nob.lock"


def test_check_config_stamp():
    can = ["-DPERF_ENABLE", "-DVX_CFG_NUM_CORES=4", "-DVX_CFG_NUM_THREADS=8"]
    good = "-O3 -DVX_CFG_NUM_CORES=1 -DVX_CFG_NUM_CORES=4 -DPERF_ENABLE\n-DVX_CFG_NUM_THREADS=8\n-lfoo\n"
    assert vxrun.check_config_stamp(good, can) == (True, "ok")
    ok, note = vxrun.check_config_stamp("-DVX_CFG_NUM_CORES=1 -DPERF_ENABLE -DVX_CFG_NUM_THREADS=8\n", can)
    assert ok is False and "VX_CFG_NUM_CORES=4 (stamp: 1)" in note
    ok, note = vxrun.check_config_stamp("-DPERF_ENABLE\n", can)
    assert ok is False and "absent" in note
    assert vxrun.check_config_stamp(None, can)[0] is None


def test_config_mismatch_marks_row_not_ok(tmp_path):
    """A PASSED run whose config.stamp lacks the requested CONFIGS (another job rebuilt the tree) is not ok."""
    app_dir = tmp_path / "build/vortex-base/tests/regression/sgemm"
    app_dir.mkdir(parents=True)
    fr = FakeRunner()
    v = _vx(tmp_path, fr)
    s = vxrun.resolve(_spec(), NAMED)
    stamp = app_dir / "config.stamp"
    stamp.write_text("-DVX_CFG_NUM_CORES=99\n")      # a foreign build: no -DPERF_ENABLE
    bad = v.run(_spec())
    assert bad["status"] == "failed" and "config mismatch" in bad["reason"] and bad["config_stamp_ok"] is False
    stamp.write_text(s["configs_effective"] + "\n")
    good = v.run(_spec(), force=True)
    assert good["status"] == "ok" and good["config_stamp_ok"] is True
