"""Tests: ps_softmax_fixed.

All tests here are vortex-marked (PS_RUN_VORTEX=1): each SimX run goes through scripts/vxrun.py -> lowload.sh, serial.
They use a PRIVATE results dir build/tmp/softmax_test (rows + lowload ledger), deleted at session start, so they never
skip on old rows and never mix into results/<run>. lowload exit 75 -> pytest.skip("DEFERRED").

  test_softmax_cases[warp-*]    the fixed kernel (one warp per row, running max from -inf, software exp ps_expf) on
                                the cases -- all-negative rows (the stock bug case, moderate and deep), large
                                positive, length 1, length 4096 -- plus lengths below NT (3) and with a tail (37), and
                                the grid-stride path (-g 1) -> PASSED, every PSCHECK errors=0, tol == 1e-5 (the
                                tolerance; the derived bound is below it for every n <= 4096, main.cpp "Error bound").
  test_softmax_fixture[...]     attention fixtures (scores -> probs; stories260K L0/L4 at T = 18, 512; stories15M
                                L0/L5 at T = 18, 256) -> PASSED vs float64 and vs the NumPy float32 probs.
  test_exp_probe_sw             ps_expf at 65,541 points over [-104, 89] + specials: within 2 ulp (normal outputs),
                                one subnormal ulp below 2^-126.
  test_stock_behaviour          the stock kernel (verbatim) and stock-with-accurate-exp on the bug inputs, report-only
                                (-x): records which are wrong (asserted, so a change in behaviour is noticed).
  test_mufu_refused_without_ext the MUFU kernel on a device without the extension -> FAILED before launch.
  test_softmax_cases[mufu-*], test_exp_probe_mufu
                                the same with the MUFU EX2 exp, on a SimX built with -DVX_CFG_EXT_MUFU_ENABLE; tol is
                                the MUFU kernel's derived bound (1.44e-5 at n = 4096; 1.18e-5 at n = 1), which exceeds the 1e-5.
The MUFU tests come last (one SimX rebuild for the config switch).
"""
import re
import shutil

import pytest

PS = __import__("pathlib").Path(__file__).resolve().parents[1]
FIX = PS / "build" / "fixtures"
PLAN_TOL = 1e-5
PSCHECK_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) "
                        r"tol=(\S+) errors=(\d+)", re.M)
HDR_RE = re.compile(r"^ps_softmax: .* bound=(\S+) tol=(\S+)", re.M)
MUFU = {"configs": "-DVX_CFG_EXT_MUFU_ENABLE"}
SMOKE = {"config": "smoke-default"}

# cases (+ edge shapes); args are appended to "-v <variant>"
CASES = {
    "allneg": "-r 8 -n 64 -a -30 -b -1",          # the stock bug case
    "allneg_deep": "-r 8 -n 64 -a -200 -b -150",   # every exp(x - 0) underflows: stock+exact exp gives 0/0
    "largepos": "-r 8 -n 64 -a 100 -b 200",        # exp(x) alone would overflow (x > 88.7)
    "len1": "-r 8 -n 1",
    "len3_lt_nt": "-r 8 -n 3 -a -5 -b 5",          # fewer columns than lanes: idle lanes keep -inf / 0
    "len37_tail": "-r 5 -n 37 -a -5 -b 5",
    "len4096": "-r 2 -n 4096 -a -10 -b 10",
    "grid_stride": "-r 8 -n 64 -a -5 -b 5 -g 1",   # 1 block of 16 threads = 4 warps for 8 rows
}
FIXTURES = ["stories260K/L0_P17_attention", "stories260K/L4_P511_attention",
            "stories15M/L0_P17_attention", "stories15M/L5_P255_attention"]


@pytest.fixture(scope="module")
def vx():
    import subprocess
    import sys
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun  # noqa: E402
    priv = PS / "build" / "tmp" / "softmax_test"
    if priv.exists():
        shutil.rmtree(priv)
    assert subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], capture_output=True).returncode == 0
    return vxrun.VxRun("P2.4-test", results=priv)


def _run(vx, args, cfg):
    spec = {"tree": "base", "app": "ps_softmax_fixed", "args": args, "perf": 1, **cfg}
    row = vx.run(spec, timeout_s=3600)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


def _checks(log):
    return {m.group(1): {"max_rel_err": float(m.group(4)), "tol": float(m.group(6)), "errors": int(m.group(7))}
            for m in PSCHECK_RE.finditer(log)}


def _assert_pass(row, log, variant, n_expect=None):
    assert row["status"] == "ok", row["reason"]
    c = _checks(log)
    assert "launch0:vs_f64" in c, log[-2000:]
    for k, v in c.items():
        assert v["errors"] == 0 and v["max_rel_err"] <= v["tol"], (k, v)
    bound, tol = map(float, HDR_RE.search(log).groups())
    if variant == "warp":
        assert bound <= PLAN_TOL and c["launch0:vs_f64"]["tol"] == PLAN_TOL   # tolerance applies
    else:   # PSCHECK prints tol with 2 significant digits; the header line has 4
        assert tol == pytest.approx(max(PLAN_TOL, bound), rel=1e-3)
        assert c["launch0:vs_f64"]["tol"] == pytest.approx(tol, rel=0.05)
    assert "nonfinite=0" in log
    return c


@pytest.mark.vortex
@pytest.mark.parametrize("case", list(CASES))
def test_softmax_cases_warp(vx, case):
    row, log = _run(vx, f"{CASES[case]} -v warp", SMOKE)
    _assert_pass(row, log, "warp")
    if case == "grid_stride":
        assert "(grid overridden by -g)" in log


@pytest.mark.vortex
@pytest.mark.parametrize("fx", FIXTURES)
def test_softmax_fixture(vx, fx):
    d = FIX / fx
    if not (d / "meta.json").is_file():
        pytest.fail(f"fixture {d} missing: run `python -B -m model.make_fixtures model --model ...`")
    row, log = _run(vx, f"-f {d} -v warp", SMOKE)
    c = _assert_pass(row, log, "warp")
    assert set(c) == {"fixture_vs_f64", "launch0:vs_f64", "launch0:vs_fixture"}


@pytest.mark.vortex
def test_softmax_libm_variant(vx):
    """the musl-libm exp variant (cost reference) is also correct (bound uses 1 ulp for musl expf)"""
    row, log = _run(vx, f"{CASES['allneg']} -v libm", SMOKE)
    _assert_pass(row, log, "warp")
    assert "kernel=softmax_warp_libm" in log


@pytest.mark.vortex
def test_exp_probe_sw(vx):
    row, log = _run(vx, "-p sw", SMOKE)
    assert row["status"] == "ok", row["reason"]
    c = _checks(log)["exp_sw:err_over_allowed"]
    assert c["errors"] == 0 and c["max_rel_err"] <= 1.0
    max_ulp = float(re.search(r"PSEXP: impl=sw max_ulp=(\S+)", log).group(1))
    assert max_ulp <= 2.0


@pytest.mark.vortex
def test_stock_behaviour(vx):
    """Report-only runs (exit 0 / PASSED! by design, -x); the verdict line records correct vs wrong."""
    expect = {
        ("stock", "allneg"): "wrong",           # polynomial exp evaluated far from 0 (max starts at 0)
        ("stock", "posunit"): "wrong",          # even on the stock app's own inputs U[0,1): polynomial error ~1.6e-2
        ("stock_expf", "allneg"): "correct",    # with an accurate exp, max = 0 is harmless here (shift invariance)
        ("stock_expf", "allneg_deep"): "wrong",  # every exp(x - 0) underflows -> 0/0 = NaN
        ("warp", "allneg_deep"): "correct",
    }
    args = {"allneg": CASES["allneg"], "allneg_deep": CASES["allneg_deep"], "posunit": "-r 8 -n 64 -a 0 -b 1"}
    for (variant, case), verdict in expect.items():
        row, log = _run(vx, f"{args[case]} -v {variant} -x", SMOKE)
        assert row["status"] == "ok", row["reason"]
        assert f"PSVERDICT: {variant} {verdict}" in log, (variant, case, log[-1500:])
        if (variant, case) == ("stock_expf", "allneg_deep"):
            assert "nonfinite=512" in log


@pytest.mark.vortex
@pytest.mark.psbase_skip
def test_mufu_refused_without_ext(vx):
    row, log = _run(vx, "-r 2 -n 8 -v mufu", SMOKE)
    assert row["status"] == "failed" and "need a device built with -DVX_CFG_EXT_MUFU_ENABLE" in log


# ---- MUFU configuration last (one SimX rebuild)
@pytest.mark.vortex
@pytest.mark.parametrize("case", ["allneg", "allneg_deep", "largepos", "len1", "len3_lt_nt", "len4096"])
def test_softmax_cases_mufu(vx, case):
    row, log = _run(vx, f"{CASES[case]} -v mufu", MUFU)
    _assert_pass(row, log, "mufu")
    assert "mufu_hw=1" in log


@pytest.mark.vortex
def test_exp_probe_mufu(vx):
    row, log = _run(vx, "-p mufu", MUFU)
    assert row["status"] == "ok", row["reason"]
    c = _checks(log)["exp_mufu:err_over_allowed"]
    assert c["errors"] == 0 and c["max_rel_err"] <= 1.0
