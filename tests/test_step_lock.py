"""S2 tests for scripts/step_lock.sh. Uses a private PS_RESULTS_DIR under tmp_path."""
import os
import subprocess
import time

import pytest


@pytest.fixture
def sl(ps_root, tmp_path):
    env = dict(os.environ)
    env["PS_RESULTS_DIR"] = str(tmp_path / "results")
    script = str(ps_root / "scripts" / "step_lock.sh")
    started = []

    def run(*args, extra_env=None):
        e = dict(env)
        e.update(extra_env or {})
        r = subprocess.run([script, *args], env=e, capture_output=True, text=True, timeout=60)
        if args[0] == "acquire" and r.returncode == 0:
            started.append((args[1], e))
        return r

    run.dir = tmp_path / "results"
    yield run
    for sid, e in started:  # never leave holders behind
        subprocess.run([script, "release", sid], env=e, capture_output=True, timeout=60)


def status(sl, sid, **kw):
    return sl("status", sid, **kw).stdout.strip()


def test_acquire_twice_fails_and_release_frees(sl):
    assert status(sl, "S2T") == "free"
    assert sl("acquire", "S2T").returncode == 0
    assert status(sl, "S2T") == "live"
    second = sl("acquire", "S2T")
    assert second.returncode != 0
    assert status(sl, "S2T") == "live"
    assert sl("release", "S2T").returncode == 0
    assert status(sl, "S2T") == "free"


def test_acquire_after_stale_heartbeat_and_leftover_release_stays_live(sl):
    d = sl.dir / "S2S"
    d.mkdir(parents=True)
    hb = d / ".heartbeat"
    hb.touch()
    old = time.time() - 2 * 3600
    os.utime(hb, (old, old))                       # interrupted attempt: heartbeat 2 h old
    (d / ".release").touch()                       # ... and a leftover release request
    assert sl("acquire", "S2S").returncode == 0    # default PS_STALE_S (30 min)
    assert not (d / ".release").exists()
    t0 = time.time()
    while time.time() - t0 < 3.5:                  # live for >= 3 s afterwards
        assert status(sl, "S2S") == "live"
        time.sleep(0.25)


def test_status_free_after_stale_heartbeat(sl):
    env = {"PS_STALE_S": "2"}
    assert sl("acquire", "S2H", extra_env=env).returncode == 0
    assert status(sl, "S2H") == "live"
    t0 = time.time()
    while status(sl, "S2H") == "live":             # no touches: holder exits once heartbeat > 2 s old
        assert time.time() - t0 < 15, "holder did not exit after the heartbeat went stale"
        time.sleep(0.25)
    assert status(sl, "S2H") == "free"


def test_touch_keeps_lock_live(sl):
    env = {"PS_STALE_S": "2"}
    assert sl("acquire", "S2K", extra_env=env).returncode == 0
    t0 = time.time()
    while time.time() - t0 < 5:                    # > 2 x stale window, kept alive by touches
        assert sl("touch", "S2K").returncode == 0
        assert status(sl, "S2K") == "live"
        time.sleep(0.5)
    assert sl("release", "S2K", extra_env=env).returncode == 0
    assert status(sl, "S2K") == "free"


def test_status_on_unknown_step_is_free_and_creates_nothing(sl):
    assert status(sl, "NEVER") == "free"
    assert not (sl.dir / "NEVER").exists()
