"""S10b thin Vortex smoke tests (b TEST). Vortex-marked: skipped unless PS_RUN_VORTEX=1 (conftest.py).

test_s7_case1_sgemm_base
    Re-runs smoke case 1 exactly as results/S7/run_s7.sh does it (base build dir $VX_BASE, CONFIGS="" ,
    `./ci/blackbox.sh --driver=simx --app=sgemm --args=-n32 --perf=1`, through scripts/lowload.sh with
    --rss 2 --mem 4) and asserts `PASSED!` plus instrs/cycles EQUAL to results/S7/summary.csv case 1 (parsed, not
    hard-coded). The base build may have been configured last for another CONFIGS, so blackbox may rebuild SimX
    first (~2 min on this shared machine); the timeout is generous for that plus lowload's admission wait.
    lowload runs with PS_RESULTS_DIR=build/tmp/vortex_smoke, i.e. a PRIVATE ledger
    (build/tmp/vortex_smoke/lowload_ledger.tsv) so test runs do not mix into results/lowload_ledger.tsv; the run log
    is build/tmp/vortex_smoke/S10b-c1-base.log. The machine-wide lowload flock is still the shared one.
    lowload exit 75 (= DEFERRED: not enough free memory within the admission window) is NOT a failure:
    the test calls pytest.skip("DEFERRED ...").

test_mk_vortex_tree_rerun_is_noop[base|ps]
    Re-runs scripts/mk_vortex_tree.sh --patchset <p> WITHOUT --recreate. In that mode the script never deletes or
    recreates an existing tree: it exits 0 with "no-op" when the stored fingerprint matches, else exits 1 untouched.
    Asserts exit 0, the "no-op" message, and that vortex-trees/<p>/.ps_patchset is byte-identical before and after.
"""
import csv
import os
import re
import subprocess

import pytest

pytestmark = pytest.mark.vortex

DEFERRED = 75
# SimX rebuild (~2 min, more on a loaded machine) + run (~1 s) + up to lowload's admission wait if memory is short.
SMOKE_TIMEOUT_S = int(os.environ.get("PS_SMOKE_TIMEOUT_S", "3600"))
PERF_RE = re.compile(r"^PERF: instrs=(\d+), cycles=(\d+)", re.M)


def _s7_case(ps_root, case_id):
    with open(ps_root / "results/S7/summary.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["case"] == case_id:
                return row
    raise AssertionError(f"case {case_id} not in results/S7/summary.csv")


def test_s7_case1_sgemm_base(ps_root):
    """smoke case 1 (sgemm -n32, base tree) via lowload; exit 75 from lowload -> pytest.skip("DEFERRED"), not a failure."""
    ref = _s7_case(ps_root, "1")
    assert ref["tree"] == "base" and ref["args"] == "-n32" and ref["PASSED"] == "True"

    priv = ps_root / "build/tmp/vortex_smoke"
    priv.mkdir(parents=True, exist_ok=True)
    log = priv / "S10b-c1-base.log"
    # Same command as run_s7.sh run_case c1 (first run), minus PS_STEP (no heartbeat into results/).
    script = (
        f'source "{ps_root}/scripts/env.sh" && cd "$VX_BASE" && '
        f'"$LL" --rss 2 --mem 4 --label S10b-c1-base --log "{log}" -- '
        'env "CONFIGS=" ./ci/blackbox.sh --driver=simx --app=sgemm "--args=-n32" --perf=1'
    )
    env = dict(os.environ)
    env.pop("PS_STEP", None)
    env["PS_RESULTS_DIR"] = str(priv)
    env["TMPDIR"] = str(ps_root / "build/tmp")  # /tmp is a small RAM tmpfs (errata 21)
    proc = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=SMOKE_TIMEOUT_S)
    if proc.returncode == DEFERRED:
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"lowload/blackbox exit {proc.returncode}; tail:\n{out[-3000:]}"
    assert "PASSED!" in out, out[-3000:]
    m = PERF_RE.findall(out)
    assert m, "no global 'PERF: instrs=..., cycles=...' line:\n" + out[-3000:]
    instrs, cycles = (int(x) for x in m[-1])
    assert (instrs, cycles) == (int(ref["instrs"]), int(ref["cycles"]))


@pytest.mark.parametrize("patchset", ["base", "ps"])
def test_mk_vortex_tree_rerun_is_noop(ps_root, patchset):
    marker = ps_root / "vortex-trees" / patchset / ".ps_patchset"
    assert marker.is_file(), f"{marker} missing (tree not built by mk_vortex_tree.sh?)"
    before = marker.read_bytes()
    proc = subprocess.run([str(ps_root / "scripts/mk_vortex_tree.sh"), "--patchset", patchset],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no-op" in proc.stdout, proc.stdout + proc.stderr
    assert marker.read_bytes() == before
