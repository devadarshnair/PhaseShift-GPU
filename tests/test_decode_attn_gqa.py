"""Tests: ps_decode_attn_gqa.

Cheap (always run):
  test_shares_softmax_warp_ops   kernels/decode_attn_gqa/warp_ops.h is a symlink to kernels/softmax_fixed/warp_ops.h
                                 (shared ps_expf; the symlink keeps vxrun's app-source hash covering it).
  test_lmem_layout_matches_host  the LMEM footprint formula of common.h (parsed) shapes: the default
                                 Tt = 32 fits 16 KiB for naive/gqa; the dxa Tt the host would choose is >= 1.

Vortex-marked (PS_RUN_VORTEX=1; serial SimX runs through scripts/vxrun.py -> lowload.sh; config smoke-dxa = the
default 1-core machine with the DXA extension, so all three variants run on the same build):
  test_attn_grid[...]            grid: (H_q, H_kv, D) in {(8,4,8), (6,6,48), (8,2,64), (16,2,128)} x B in
                                 {1, 4} x T in {1, 17, 256, 1024} x variant in {naive, gqa, dxa(depth 2)} -> PASSED,
                                 every PSCHECK errors = 0, max_rel_err <= tol, tol = max(1e-4, derived bound).
                                 Cache rows T..Tmax-1 are NaN (Tmax = T + 3), so reading past T fails.
  test_attn_all_negative[...]    every logit negative (q > 0, K < 0) for each shape, T = 256, B = 4, all variants
                                 (moderate logits, -1.6 to -6.4: no underflow, so this alone cannot catch errata 7).
  test_attn_deep_negative_multitile[...]  logits -207 … -109 (mean -157) (would underflow every weight with a max started at 0),
                                 T = 100: several tiles plus a partial last tile on every variant.
  Every vortex test also asserts PSINFO within_plan_tol=1, i.e. the rel 1e-4 itself.
  test_attn_dxa_depth[...]       dxa with depth 1 and 4 for each shape (T = 256, B = 4).
  test_attn_fixture[...]         attention fixtures (stories260K L0_P17, L4_P511; stories15M L0_P17, L5_P255),
                                 all variants -> PASSED vs float64 and vs the NumPy float32 out.
  test_attn_dxa_refused_without_ext   dxa on a build without DXA -> FAILED before launch.
The vortex tests use a PRIVATE results dir build/tmp/decode_attn_test (rows + ledger), deleted at session start.
"""
import re
import shutil
import subprocess

import pytest

PS = __import__("pathlib").Path(__file__).resolve().parents[1]
FIX = PS / "build" / "fixtures"
PLAN_TOL = 1e-4
PSCHECK_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) "
                        r"tol=(\S+) errors=(\d+)", re.M)
HDR_RE = re.compile(r"^ps_decode_attn: .* bound=(\S+) tol=(\S+)", re.M)
SHAPES = {"260K": (8, 4, 8), "15M": (6, 6, 48), "8-2-64": (8, 2, 64), "16-2-128": (16, 2, 128)}
VARIANTS = {"naive": "-v naive", "gqa": "-v gqa", "dxa": "-v dxa -p 2"}
CFG = {"config": "smoke-dxa"}
FIXTURES = ["stories260K/L0_P17_attention", "stories260K/L4_P511_attention",
            "stories15M/L0_P17_attention", "stories15M/L5_P255_attention"]


# ------------------------------------------------------------------------------------------------- cheap
def test_shares_softmax_warp_ops():
    p = PS / "kernels" / "decode_attn_gqa" / "warp_ops.h"
    assert p.is_symlink() and p.resolve() == (PS / "kernels" / "softmax_fixed" / "warp_ops.h").resolve()


def _lmem_floats(G, D, Tt, depth, dxa):
    r16 = lambda n: (n + 15) & ~15  # noqa: E731  (common.h ps_attn_round16)
    total = 2 * r16(G * D) + r16(G * Tt) + r16(3 * G)
    return total + (2 * depth * r16(Tt * D) if dxa else 0)


def test_lmem_layout_matches_host():
    src = (PS / "kernels" / "decode_attn_gqa" / "common.h").read_text()
    # the formula above mirrors common.h; guard against silent edits there
    for frag in ("L.acc = L.qs + ps_attn_round16(G * D)", "L.ps = L.acc + ps_attn_round16(G * D)",
                 "L.stat = L.ps + ps_attn_round16(G * Tt)", "L.ks = L.stat + ps_attn_round16(3 * G)",
                 "L.vs = L.ks + (dxa ? depth * ps_attn_round16(Tt * D) : 0)",
                 "L.total = L.vs + (dxa ? depth * ps_attn_round16(Tt * D) : 0)"):
        assert frag in src, frag
    lmem = 16384
    for hq, hkv, d in SHAPES.values():
        g = hq // hkv
        assert _lmem_floats(g, d, 32, 0, False) * 4 <= lmem
        for depth in (1, 2, 4):
            assert _lmem_floats(g, d, 1, depth, True) * 4 <= lmem, (hq, hkv, d, depth)


# ------------------------------------------------------------------------------------------------- vortex
@pytest.fixture(scope="module")
def vx():
    import sys
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun  # noqa: E402
    priv = PS / "build" / "tmp" / "decode_attn_test"
    if priv.exists():
        shutil.rmtree(priv)
    assert subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], capture_output=True).returncode == 0
    return vxrun.VxRun("P2.5-test", results=priv)


def _run(vx, args, cfg=CFG):
    row = vx.run({"tree": "base", "app": "ps_decode_attn_gqa", "args": args, "perf": 1, **cfg}, timeout_s=7200)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


def _assert_pass(row, log, fixture=False):
    assert row["status"] == "ok", (row["reason"], log[-2500:])
    c = {m.group(1): (float(m.group(4)), float(m.group(6)), int(m.group(7))) for m in PSCHECK_RE.finditer(log)}
    want = {"launch0:vs_f64"} | ({"fixture_vs_f64", "launch0:vs_fixture"} if fixture else set())
    assert set(c) == want, c
    for k, (rel, tol, err) in c.items():
        assert err == 0 and rel <= tol, (k, rel, tol)
    bound, tol = map(float, HDR_RE.search(log).groups())
    assert tol == pytest.approx(max(PLAN_TOL, bound), rel=1e-3)
    assert "within_plan_tol=1" in log     # the criterion itself (rel 1e-4 vs float64), whatever the bound
    return c


def _shape_args(shape, B, T):
    hq, hkv, d = SHAPES[shape]
    return f"-B {B} -H {hq} -K {hkv} -D {d} -T {T}"


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("T", [1, 17, 256, 1024])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_attn_grid(vx, shape, B, T, variant):
    row, log = _run(vx, f"{_shape_args(shape, B, T)} {VARIANTS[variant]}")
    _assert_pass(row, log)


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("shape", list(SHAPES))
def test_attn_all_negative(vx, shape, variant):
    row, log = _run(vx, f"{_shape_args(shape, 4, 256)} -d neg {VARIANTS[variant]}")
    _assert_pass(row, log)


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_attn_deep_negative_multitile(vx, variant):
    """Logits -207 … -109 (mean -157) (q scaled by 100, every q.k < 0): a running max initialised to 0 would underflow every weight
    (l = 0 -> 0/0), so this catches errata 7 in online form; T = 100 with Tt = 32 (every variant at D = 8) is 4 tiles
    with a partial last tile of 4, the first such case on the plain naive/gqa path."""
    row, log = _run(vx, f"-B 2 -H 8 -K 4 -D 8 -T 100 -d neg -q 100 {VARIANTS[variant]}")
    _assert_pass(row, log)
    assert "ntiles=1 " not in log


@pytest.mark.vortex
@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_attn_dxa_depth(vx, shape, depth):
    row, log = _run(vx, f"{_shape_args(shape, 4, 256)} -v dxa -p {depth}")
    _assert_pass(row, log)
    assert f"depth={depth}" in log


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("fx", FIXTURES)
def test_attn_fixture(vx, fx, variant):
    d = FIX / fx
    if not (d / "meta.json").is_file():
        pytest.fail(f"fixture {d} missing: run `python -B -m model.make_fixtures model --model ...`")
    row, log = _run(vx, f"-f {d} {VARIANTS[variant]}")
    _assert_pass(row, log, fixture=True)


@pytest.mark.vortex
@pytest.mark.psbase_skip
def test_attn_dxa_refused_without_ext(vx):
    row, log = _run(vx, f"{_shape_args('260K', 1, 17)} -v dxa", {"config": "smoke-default"})
    assert row["status"] == "failed" and "needs a device built with -DVX_CFG_EXT_DXA_ENABLE" in log
