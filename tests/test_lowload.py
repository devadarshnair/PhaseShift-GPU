"""S2 tests for scripts/lowload.sh.

Every test uses a private lock (PS_LOCK) and a private results dir (PS_RESULTS_DIR) under pytest's tmp_path, so it
never touches the user's /tmp/ml-hardware-bench.lock or the real results/lowload_ledger.tsv.
"""
import os
import resource
import subprocess
import sys
import time

import pytest

PY = sys.executable


# Regression tests (2026-11-14): these tests exercise lowload.sh's real admission
# rule (MemAvailable >= RSS + 0.75 GB, retry every 60 s) and its default 4 GiB VA cap, so their outcome depended on the
# host. They now skip, with the reason in the report, when the host cannot satisfy that rule. Assertions are unchanged.
def _mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def _require_mem(gb):
    avail = _mem_available_gb()
    if avail < gb:
        pytest.skip(f"MemAvailable {avail:.2f} GB < {gb} GB: lowload admission would wait (60 s steps) and the "
                    "test would time out instead of testing")


def _require_default_va_cap():
    """lowload's default --mem is 4 GiB; its inner prlimit cannot raise above the driver's own hard RLIMIT_AS."""
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    if hard != resource.RLIM_INFINITY and hard < 4 << 30:
        pytest.skip(f"pytest driver runs under a {hard >> 20} MiB address-space cap (< lowload's 4 GiB default); "
                    "run the suite with lowload --mem >= 4")


@pytest.fixture
def ll(ps_root, tmp_path):
    env = dict(os.environ)
    env.update(PS_LOCK=str(tmp_path / "bench.lock"), PS_RESULTS_DIR=str(tmp_path / "results"))
    env.pop("PS_STEP", None)
    env.pop("PS_LOWLOAD_HELD", None)
    env["PS_NO_MEMWATCH"] = "1"     # D9: tests must not autostart the real memwatch.sh
    script = str(ps_root / "scripts" / "lowload.sh")
    _require_default_va_cap()
    _require_mem(1.0)               # smallest --rss used below is 0.1 GB (+ 0.75 GB margin)

    def run(args, extra_env=None, **kw):
        e = dict(env)
        e.update(extra_env or {})
        return subprocess.run([script, *args], env=e, capture_output=True, text=True, timeout=120, **kw)

    run.env = env
    run.script = script
    run.ledger = tmp_path / "results" / "lowload_ledger.tsv"
    run.tmp = tmp_path
    return run


def ledger_rows(path):
    return [line.split("\t") for line in path.read_text().splitlines()]


def test_niceness_and_makeflags(ll):
    code = ("s=open('/proc/self/stat').read(); f=s[s.rindex(')')+2:].split();"
            "import os; print(f[16], os.environ['MAKEFLAGS'], os.environ['THREADS'])")
    r = ll(["--rss", "0.1", "--", PY, "-c", code], extra_env={"MAKEFLAGS": "-j8"})
    assert r.returncode == 0, r.stderr
    nice, makeflags, threads = r.stdout.split()
    assert nice == "19"            # field 19 of /proc/self/stat (index 16 after the comm field)
    assert makeflags == "-j1"      # forced even though the caller exported -j8
    assert threads == "1"


def test_va_cap_blocks_2gib_allocation(ll):
    r = ll(["--rss", "0.1", "--mem", "1", "--", PY, "-c", "b = bytearray(2 << 30)"])
    assert r.returncode != 0
    assert "MemoryError" in r.stderr


def test_lock_held_while_job_runs(ll):
    p = subprocess.Popen([ll.script, "--rss", "0.1", "--label", "hold", "--", "sleep", "3"], env=ll.env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while not (ll.ledger.exists() and "start\t" in ll.ledger.read_text()):
            assert time.time() - t0 < 10, "job never started"
            time.sleep(0.05)
        probe = subprocess.run(["flock", "-n", ll.env["PS_LOCK"], "true"])
        assert probe.returncode != 0, "lock was free while the job ran"
    finally:
        assert p.wait(timeout=30) == 0
    assert subprocess.run(["flock", "-n", ll.env["PS_LOCK"], "true"]).returncode == 0  # released afterwards


def test_ledger_start_and_end_lines(ll):
    r = ll(["--rss", "0.1", "--label", "S2:ledger", "--", PY, "-c",
            "b = bytearray(80 << 20); b[::4096] = b'x' * len(b[::4096]); raise SystemExit(3)"])
    assert r.returncode == 3
    rows = ledger_rows(ll.ledger)
    starts = [x for x in rows if x[0] == "start"]
    ends = [x for x in rows if x[0] == "end"]
    assert len(starts) == 1 and len(ends) == 1
    assert starts[0][1] == ends[0][1]                    # same wrapper pid pairs the lines
    assert starts[0][2] == "S2:ledger" and starts[0][3] == "0.1"
    assert ends[0][2] == "S2:ledger"
    assert ends[0][3] == "3"                             # exit status
    assert int(ends[0][4]) >= 80 * 1024                  # maxrss_kb covers the 80 MiB buffer
    assert float(ends[0][5]) >= 0                        # wall_s
    tf = ll.tmp / "results" / "logs" / "lowload_time" / f"S2:ledger.{starts[0][1]}.time"
    assert tf.exists() and "Maximum resident set size" in tf.read_text()   # under git-ignored results/**/logs/


def test_nonzero_exit_through_log(ll):
    log = ll.tmp / "logs" / "job.log"
    r = ll(["--rss", "0.1", "--log", str(log), "--", "bash", "-c", "echo to-log; echo err-log >&2; exit 5"])
    assert r.returncode == 5                             # pipefail: tee's 0 does not mask the job's 5
    text = log.read_text()
    assert "to-log" in text and "err-log" in text
    assert (ll.tmp / "logs" / "job.log.time").exists()
    assert ledger_rows(ll.ledger)[-1][3] == "5"


def test_admission_deferred_exit_75(ll):
    t0 = time.time()
    r = ll(["--rss", "9999", "--", "true"], extra_env={"PS_ADMIT_WAIT_S": "5", "PS_ADMIT_SLEEP_S": "1"})
    assert r.returncode == 75
    assert time.time() - t0 < 30
    assert "DEFERRED" in r.stderr
    assert not ll.ledger.exists() or "start\t" not in ll.ledger.read_text()   # the job never ran
    assert subprocess.run(["flock", "-n", ll.env["PS_LOCK"], "true"]).returncode == 0  # lock not left held


def test_default_rss_from_ledger(ll):
    # A recorded end line with maxrss 2 GiB for prefix "S9" -> expected RSS 1.2 x 2 GiB = 2.4 GB.
    _require_mem(3.5)               # admission needs 2.4 + 0.75 = 3.15 GB (+ slack for fluctuation)
    ll.ledger.parent.mkdir(parents=True, exist_ok=True)
    ll.ledger.write_text("end\t1\tS9:old\t0\t2097152\t1.0\tts\n")
    r = ll(["--label", "S9:new", "--", "true"])
    assert r.returncode == 0, r.stderr
    start = [x for x in ledger_rows(ll.ledger) if x[0] == "start"][-1]
    assert float(start[3]) == pytest.approx(2.4, abs=1e-3)
    r = ll(["--label", "other", "--", "true"])         # no history for this prefix -> 1.5 GB default
    start = [x for x in ledger_rows(ll.ledger) if x[0] == "start"][-1]
    assert start[3] == "1.5"


def test_heartbeat_touched_during_job(ll):
    hb = ll.tmp / "results" / "SX" / ".heartbeat"
    # The job samples the heartbeat mtime over ~2.6 heartbeat intervals and prints the distinct values seen.
    code = ("import os, time\n"
            f"p = {str(hb)!r}\n"
            "seen = set()\n"
            "t0 = time.time()\n"
            "while time.time() - t0 < 2.6:\n"
            "    if os.path.exists(p): seen.add(os.stat(p).st_mtime_ns)\n"
            "    time.sleep(0.05)\n"
            "print(len(seen))\n")
    r = ll(["--rss", "0.1", "--", PY, "-c", code], extra_env={"PS_STEP": "SX", "PS_HEARTBEAT_S": "1"})
    assert r.returncode == 0, r.stderr
    assert int(r.stdout.strip()) >= 2


def test_default_lock_path_matches_heavy_sh(ps_root):
    text = (ps_root / "scripts" / "lowload.sh").read_text()
    assert 'LOCK="${PS_LOCK:-/tmp/ml-hardware-bench.lock}"' in text
    heavy = ps_root.parent / "src" / "bench" / "heavy.sh"
    if heavy.exists():
        assert "/tmp/ml-hardware-bench.lock" in heavy.read_text()
    assert "exec nice" not in text and "exec flock" not in text   # job runs as a child, no exec


# ----------------------------------------------------------------------------------- signals and parsing
def _start_killable(ll, seconds):
    """Start the wrapper on a job that records its pid and sleeps; return (wrapper Popen, job pid, heartbeat path)."""
    pidf = ll.tmp / "job.pid"
    env = dict(ll.env, PS_STEP="SIG", PS_HEARTBEAT_S="1")
    p = subprocess.Popen([ll.script, "--rss", "0.1", "--label", "sig", "--", "bash", "-c",
                          f"echo $$ > {pidf}; exec sleep {seconds}"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while not (pidf.exists() and pidf.read_text().strip()):
        assert time.time() - t0 < 10, "job never started"
        time.sleep(0.05)
    time.sleep(0.5)
    return p, int(pidf.read_text()), ll.tmp / "results" / "SIG" / ".heartbeat"


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill(0); treat it as dead
    try:
        return open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _heartbeat_stopped(hb, window=2.5):
    m1 = hb.stat().st_mtime_ns
    time.sleep(window)
    return hb.stat().st_mtime_ns == m1


def _lock_free(ll):
    return subprocess.run(["flock", "-n", ll.env["PS_LOCK"], "true"]).returncode == 0


def test_sigterm_wrapper_kills_job_writes_end_and_stops_heartbeat(ll):
    p, job, hb = _start_killable(ll, 30)
    p.terminate()                                       # SIGTERM to the wrapper only
    assert p.wait(timeout=20) == 143
    t0 = time.time()
    while _alive(job):                                  # job is not orphaned: the signal was forwarded
        assert time.time() - t0 < 5, "job survived the wrapper's SIGTERM"
        time.sleep(0.1)
    time.sleep(1.5)                                     # let the heartbeat loop notice
    assert _heartbeat_stopped(hb)
    ends = [x for x in ledger_rows(ll.ledger) if x[0] == "end"]
    assert len(ends) == 1 and ends[0][3] == "143"       # end line written with the job's status
    assert _lock_free(ll)


def test_sigkill_wrapper_keeps_lock_while_job_alive_and_stops_heartbeat(ll):
    p, job, hb = _start_killable(ll, 12)               # long enough to stay alive under machine load
    p.kill()                                            # SIGKILL: no trap can run
    p.wait(timeout=10)
    assert _alive(job)                                  # job keeps running ...
    assert not _lock_free(ll)                           # ... and keeps the machine lock (inherited fd)
    time.sleep(1.5)
    assert _heartbeat_stopped(hb)                       # loop exits once the wrapper pid is gone
    assert _alive(job) and not _lock_free(ll)
    t0 = time.time()
    while _alive(job):
        assert time.time() - t0 < 30
        time.sleep(0.2)
    time.sleep(0.2)
    assert _lock_free(ll)                               # freed when the job ends
    assert not any(x[0] == "end" for x in ledger_rows(ll.ledger))   # SIGKILL: no end line (documented)


@pytest.mark.parametrize("flag", ["--rss", "--mem", "--label", "--log"])
def test_value_flag_without_value_is_usage_error(ll, flag):
    r = ll([flag])
    assert r.returncode == 2 and "needs a value" in r.stderr


def test_label_prefix_is_exact(ll):
    _require_mem(2.7)               # default 1.5 GB expected RSS + 0.75 GB margin (+ slack)
    ll.ledger.parent.mkdir(parents=True, exist_ok=True)
    ll.ledger.write_text("end\t1\tS60:big\t0\t4194304\t1.0\tts\n")   # 4 GiB under prefix S60
    r = ll(["--label", "S6:x", "--", "true"])
    assert r.returncode == 0, r.stderr
    start = [x for x in ledger_rows(ll.ledger) if x[0] == "start"][-1]
    assert start[3] == "1.5"                            # S6 must not inherit S60's history


def test_stdin_reaches_job(ll):
    r = ll(["--rss", "0.1", "--", "cat"], input="piped\n")
    assert r.returncode == 0 and r.stdout == "piped\n"


# ----------------------------------------------------------------------------------- whole-group handling
def _group_members(pgid):
    """PIDs (non-zombie) whose process group is pgid, from /proc."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            fields = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, IndexError):
            continue
        if fields[0] != "Z" and int(fields[2]) == pgid:
            out.append(int(d))
    return out


JOBS = {
    # ignores TERM (the ignore disposition is inherited by sleep) -> needs the group KILL after the grace period
    "ignores_term": 'trap "" TERM; echo started; sleep 45',
    # 3 s graceful cleanup after TERM, while a background child also runs
    "slow_cleanup": 'trap "sleep 3; exit 9" TERM; echo started; sleep 46 & wait',
}


@pytest.mark.parametrize("use_log", [False, True], ids=["nolog", "log"])
@pytest.mark.parametrize("job", sorted(JOBS))
def test_term_waits_for_whole_group_and_lock_never_free_while_job_lives(ll, job, use_log):
    pgidf = ll.tmp / "job.pgid"
    script = f'read -r _ _ _ _ pg _ < /proc/$$/stat; echo $pg > {pgidf}; {JOBS[job]}'
    args = ["--rss", "0.1"] + (["--log", str(ll.tmp / "logs" / "j.log")] if use_log else []) + ["--", "bash", "-c", script]
    grace = 10 if job == "slow_cleanup" else 3          # slow_cleanup must finish its 3 s cleanup inside the grace
    env = dict(ll.env, PS_KILL_GRACE_S=str(grace))
    p = subprocess.Popen([ll.script, *args], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while not (pgidf.exists() and pgidf.read_text().strip()):
        assert time.time() - t0 < 10, "job never started"
        time.sleep(0.05)
    pgid = int(pgidf.read_text())
    time.sleep(0.5)
    assert len(_group_members(pgid)) >= 2
    p.terminate()
    t_sig = time.time()
    violations = 0
    while p.poll() is None:                       # probe the lock while the wrapper is still handling the signal
        members = _group_members(pgid)
        if members and _lock_free(ll):
            violations += 1
        time.sleep(0.05)
        assert time.time() - t_sig < 40, "wrapper did not exit"
    elapsed = time.time() - t_sig
    assert _group_members(pgid) == [], "job processes survived the wrapper"
    assert violations == 0, "lock was free while a job process was alive"
    assert p.returncode == 143
    if job == "slow_cleanup":
        assert 2.5 <= elapsed < grace             # waited for the 3 s cleanup; finished before the KILL deadline
    else:
        assert grace - 0.5 <= elapsed < grace + 10   # waited the grace period, then KILLed the group
    assert _lock_free(ll)
    ends = [x for x in ledger_rows(ll.ledger) if x[0] == "end"]
    assert len(ends) == 1


def test_heartbeat_interval_must_be_positive(ll):
    r = ll(["--rss", "0.1", "--", "true"], extra_env={"PS_HEARTBEAT_S": "0"})
    assert r.returncode == 2 and "PS_HEARTBEAT_S" in r.stderr


def test_wrapper_never_unlocks_shared_lock(ps_root):
    text = (ps_root / "scripts" / "lowload.sh").read_text()
    body = text.split("write_end() {", 1)[1].split("\n}\n", 1)[0]
    assert "flock -u" not in body.replace("never `flock -u 9`", "")   # write_end only closes fd 9
    assert "exec 9>&-" in body


# --- regression fixes (2026-11-14) ---------------------------------------------------------------------------------
def test_reentrant_lowload_via_ps_lowload_held(ll):
    """D3: a lowload.sh run inside a lowload job must not wait on the lock its parent holds."""
    r = ll(["--rss", "0.1", "--label", "outer", "--", ll.script, "--rss", "0.1", "--label", "inner", "--", "true"],
           extra_env={"PS_ADMIT_WAIT_S": "20"})
    assert r.returncode == 0, r.stderr
    assert "skipping queue/admission" in r.stderr
    labels = [row[2] for row in ledger_rows(ll.ledger) if row[0] == "start"]
    assert labels == ["outer", "inner"]


def test_lowload_inside_exclusive_outer_hold_does_not_deadlock(ll):
    """D3: an ancestor holding an exclusive flock on the lock (qflock.sh / flock LOCK cmd) must not make lowload exit 75."""
    lock = ll.env["PS_LOCK"]
    r = subprocess.run(["flock", lock, ll.script, "--rss", "0.1", "--label", "held", "--", "true"], env=ll.env,
                       capture_output=True, text=True, timeout=60)
    # env of this inner call has no PS_LOWLOAD_HELD; detection must come from /proc/locks
    assert r.returncode == 0, r.stderr
    assert "outer holder" in r.stderr


def test_unrelated_exclusive_hold_still_blocks(ll):
    """D3 guard: a lock held by a NON-ancestor must still be waited for (and defer with 75)."""
    lock = ll.env["PS_LOCK"]
    holder = subprocess.Popen(["flock", lock, "sleep", "8"])
    try:
        time.sleep(1)
        r = ll(["--rss", "0.1", "--", "true"], extra_env={"PS_ADMIT_WAIT_S": "2", "PS_ADMIT_SLEEP_S": "1"})
        assert r.returncode == 75, r.stderr
    finally:
        holder.kill()
        holder.wait()


def test_memwatch_autostart_block_present(ps_root):
    """D9: lowload.sh carries the heavygate.sh memwatch autostart block, before the lock fd is opened."""
    txt = (ps_root / "scripts" / "lowload.sh").read_text()
    i = txt.index("flock -n /tmp/research-memwatch.lock true")
    assert "memwatch.sh" in txt[i:i + 300]
    assert i < txt.index('exec 9>>"$LOCK"')
