"""Tests: ps_prefill_attn, tiled causal prefill attention (lmem_tiled, dxa_dbuf).

Cheap (always run):
  test_shares_softmax_warp_ops     kernels/prefill_attn/warp_ops.h is a symlink to kernels/softmax_fixed/warp_ops.h
                                   (shared ps_expf; the symlink keeps vxrun's app-source hash covering it).
  test_lmem_layout_matches_host    the LMEM footprint formula of common.h (parsed); every shape fits 16 KiB at
                                   Qt = 1 for lmem and dxa depth 1/2/4; the host's default tile rule (re-implemented)
                                   gives the tiles methodology §14.4 tabulates.
  test_prefill_fixture_is_causal_attention  model/make_prefill_fixtures.py on stories260K (N = 24, private dir): every
                                   `out` row equals a float64 causal attention of the fixture's q/k/v within NumPy's
                                   float32 error, and token 17 equals the decode fixture L0_P17_attention bit for bit.

Vortex-marked (PS_RUN_VORTEX=1; serial SimX runs through scripts/vxrun.py -> lowload.sh; config smoke-dxa = the
default 1-core machine with the DXA extension, so both variants run on one build):
  test_prefill_grid[...]           grid: (H_q, H_kv, D) in {(8,4,8), (6,6,48), (8,2,64), (16,2,128)} x
                                   S in {16, 64, 256, 512} x variant in {lmem, dxa (depth 2)}, B = 1, default tiles ->
                                   PASSED, every PSCHECK errors = 0, max_rel_err <= tol = max(1e-4, derived bound),
                                   and PSINFO within_plan_tol=1 (the rel 1e-4 itself). Cache rows >= S are NaN.
  test_prefill_deep_negative_multitile[...]  every visible logit < -104 (asserted from the app's logits=[..] line), so
                                   a running max started at 0 would underflow every weight (errata 7, online form);
                                   B = 2, S = 100, Qt = 8, Tk = 12: 9 key tiles with a partial last tile, partial
                                   last query tile, diagonal tiles; both variants.
  test_prefill_chunked[...]        s0 > 0 (a chunked-prefill chunk: queries s0 .. s0+S-1 over keys 0 .. s0+S-1), both
                                   variants, two shapes.
  test_prefill_padding_rows[...]   G * Qt = 3 (padded to 4 rows), Tk = 5: padding rows never stored, both variants.
  test_prefill_dxa_depth[...]      dxa depth 1 and 4 (S = 64) for each shape.
  test_prefill_fixture[...]        prefill fixtures of stories260K (L0, L4; N = 512) and stories15M (L0, L5; N = 256),
                                   full prefill and a chunk (-o), both variants -> PASSED vs float64 and vs NumPy out.
  test_prefill_dxa_refused_without_ext   dxa on a build without DXA -> FAILED before launch.
The vortex tests use a PRIVATE results dir build/tmp/prefill_attn_test (rows + ledger), deleted at session start.
"""
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

PS = __import__("pathlib").Path(__file__).resolve().parents[1]
FIX = PS / "build" / "fixtures"
PLAN_TOL = 1e-4
PSCHECK_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) "
                        r"tol=(\S+) errors=(\d+)", re.M)
HDR_RE = re.compile(r"^ps_prefill_attn: .* Qt=(\d+) Tk=(\d+) depth=(\d+) rows=(\d+) nqt=(\d+) ntiles_max=(\d+) .*"
                    r"logits=\[(\S+),(\S+)\] bound=(\S+) tol=(\S+)", re.M)
SHAPES = {"260K": (8, 4, 8), "15M": (6, 6, 48), "8-2-64": (8, 2, 64), "16-2-128": (16, 2, 128)}
VARIANTS = {"lmem": "-v lmem", "dxa": "-v dxa -p 2"}
CFG = {"config": "smoke-dxa"}
FIXTURES = {"260K-L0": "stories260K_prefill/L0_N512_attention_prefill",
            "260K-L4": "stories260K_prefill/L4_N512_attention_prefill",
            "15M-L0": "stories15M_prefill/L0_N256_attention_prefill",
            "15M-L5": "stories15M_prefill/L5_N256_attention_prefill"}


# ------------------------------------------------------------------------------------------------- cheap
def test_shares_softmax_warp_ops():
    p = PS / "kernels" / "prefill_attn" / "warp_ops.h"
    assert p.is_symlink() and p.resolve() == (PS / "kernels" / "softmax_fixed" / "warp_ops.h").resolve()


def _r16(n):
    return (n + 15) & ~15


def lmem_floats(G, D, Qt, Tk, nbuf):
    """common.h ps_pattn_lmem (total floats)."""
    rows = (G * Qt + 3) & ~3
    return 2 * _r16(rows * D) + _r16(rows * Tk) + _r16(3 * rows) + 2 * nbuf * _r16(Tk * D)


def max_tk(G, D, Qt, nbuf, cap):
    return next((tk for tk in range(32, 0, -1) if lmem_floats(G, D, Qt, tk, nbuf) * 4 <= cap), 0)


def default_tiles(G, D, S, nbuf, cap=16384, B=1, H_kv=1, cores=4):
    """main.cpp's default tile rule (header comment "LMEM and tiles")."""
    best = best_tk = 0
    for cand in (64, 32, 16, 8, 4, 2, 1):
        if (cand > S and cand > 1) or G * cand > 128:
            continue
        tk = max_tk(G, D, cand, nbuf, cap)
        if tk >= 8:
            best, best_tk = cand, tk
            break
    if best == 0:
        best, best_tk = 1, max_tk(G, D, 1, nbuf, cap)
    while best > 1 and B * H_kv * -(-S // best) < cores:
        best //= 2
        best_tk = max_tk(G, D, best, nbuf, cap)
    return best, best_tk


def test_lmem_layout_matches_host():
    src = (PS / "kernels" / "prefill_attn" / "common.h").read_text()
    for frag in ("return (G * Qt + 3u) & ~3u;", "L.acc = L.qs + ps_pattn_round16(L.rows * D)",
                 "L.ps = L.acc + ps_pattn_round16(L.rows * D)", "L.stat = L.ps + ps_pattn_round16(L.rows * Tk)",
                 "L.ks = L.stat + ps_pattn_round16(3 * L.rows)", "L.vs = L.ks + nbuf * ps_pattn_round16(Tk * D)",
                 "L.total = L.vs + nbuf * ps_pattn_round16(Tk * D)"):
        assert frag in src, frag
    main = (PS / "kernels" / "prefill_attn" / "main.cpp").read_text()
    for frag in ("for (uint32_t cand = 64; cand >= 1; cand >>= 1)", "if (G * cand > 128) continue;",
                 "if (tk >= 8) { best = cand; best_tk = tk; break; }", "< ncores"):
        assert frag in main, frag
    for hq, hkv, d in SHAPES.values():
        g = hq // hkv
        for nbuf in (1, 2, 4):
            assert max_tk(g, d, 1, nbuf, 16384) >= 1, (hq, hkv, d, nbuf)
    # the tiles methodology §14.4 tabulates (16 KiB, S = 512, B = 1, 4 cores)
    assert default_tiles(8, 128, 512, 1, H_kv=2) == (1, 7)
    assert default_tiles(8, 128, 512, 2, H_kv=2) == (1, 3)
    assert default_tiles(8, 128, 512, 4, H_kv=2) == (1, 1)
    assert default_tiles(4, 64, 512, 1, H_kv=2) == (4, 13)
    assert default_tiles(1, 48, 512, 1, H_kv=6) == (16, 22)
    assert default_tiles(2, 8, 512, 1, H_kv=4) == (64, 11)


def test_prefill_fixture_is_causal_attention(tmp_path):
    from model.make_prefill_fixtures import make_prefill_fixtures
    from model.make_fixtures import load_fixture
    idx = make_prefill_fixtures("stories260K", [0], 24, tmp_path / "pf")
    meta, a = load_fixture(tmp_path / "pf" / idx["fixtures"][0])
    p = meta["params"]
    H, Hk, hs = p["n_heads"], p["n_kv_heads"], p["head_size"]
    N = a["q"].shape[0]
    q = a["q"].astype(np.float64).reshape(N, H, hs)
    k = a["k_cache"].astype(np.float64).reshape(N, Hk, hs)[:, np.arange(H) // (H // Hk)]
    v = a["v_cache"].astype(np.float64).reshape(N, Hk, hs)[:, np.arange(H) // (H // Hk)]
    s = np.einsum("ihd,thd->hit", q, k) / np.sqrt(hs)
    s = np.where(np.tril(np.ones((N, N), bool))[None], s, -np.inf)
    pr = np.exp(s - s.max(-1, keepdims=True))
    pr /= pr.sum(-1, keepdims=True)
    ref = np.einsum("hit,thd->ihd", pr, v)
    scale = np.einsum("hit,thd->ihd", pr, np.abs(v))
    rel = np.abs(a["out"].reshape(N, H, hs) - ref) / scale
    assert rel.max() < 1e-5, rel.max()      # NumPy float32 (T <= 24 terms): ~1e-6 expected
    dec = FIX / "stories260K" / "L0_P17_attention"
    if (dec / "meta.json").is_file():
        _, d = load_fixture(dec)
        assert np.array_equal(d["q"], a["q"][17]) and np.array_equal(d["out"], a["out"][17])
        assert np.array_equal(d["k_cache"], a["k_cache"][:18])


# ------------------------------------------------------------------------------------------------- vortex
@pytest.fixture(scope="module")
def vx():
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun  # noqa: E402
    priv = PS / "build" / "tmp" / "prefill_attn_test"
    if priv.exists():
        shutil.rmtree(priv)
    assert subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], capture_output=True).returncode == 0
    return vxrun.VxRun("P2.7-test", results=priv)


def _run(vx, args, cfg=CFG):
    row = vx.run({"tree": "base", "app": "ps_prefill_attn", "args": args, "perf": 1, **cfg}, timeout_s=14400)
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
    h = HDR_RE.search(log)
    bound, tol = float(h.group(9)), float(h.group(10))
    assert tol == pytest.approx(max(PLAN_TOL, bound), rel=1e-3)
    assert "within_plan_tol=1" in log     # the criterion itself (rel 1e-4 vs float64), whatever the bound
    return h


def _shape_args(shape, B, S):
    hq, hkv, d = SHAPES[shape]
    return f"-B {B} -H {hq} -K {hkv} -D {d} -S {S}"


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("S", [16, 64, 256, 512])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_prefill_grid(vx, shape, S, variant):
    row, log = _run(vx, f"{_shape_args(shape, 1, S)} {VARIANTS[variant]}")
    _assert_pass(row, log)


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_prefill_deep_negative_multitile(vx, variant):
    row, log = _run(vx, f"-B 2 -H 8 -K 4 -D 8 -S 100 -Q 8 -t 12 -d neg -q 150 {VARIANTS[variant]}")
    h = _assert_pass(row, log)
    assert float(h.group(8)) < -104.0, h.group(0)      # every visible logit: exp(s - 0) would underflow to 0
    assert int(h.group(6)) == 9 and int(h.group(5)) == 13   # 9 key tiles (100 = 8*12 + 4), 13 query tiles (tail 4)


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("case", ["-B 1 -H 8 -K 2 -D 64 -S 48 -o 80", "-B 2 -H 6 -K 6 -D 48 -S 37 -o 20"])
def test_prefill_chunked(vx, case, variant):
    row, log = _run(vx, f"{case} {VARIANTS[variant]}")
    _assert_pass(row, log)
    assert " s0=" in log and " s0=0 " not in log


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_prefill_padding_rows(vx, variant):
    row, log = _run(vx, f"-B 1 -H 6 -K 6 -D 48 -S 30 -Q 3 -t 5 {VARIANTS[variant]}")
    h = _assert_pass(row, log)
    assert int(h.group(4)) == 4 and int(h.group(1)) == 3


@pytest.mark.vortex
@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_prefill_dxa_depth(vx, shape, depth):
    row, log = _run(vx, f"{_shape_args(shape, 1, 64)} -v dxa -p {depth}")
    h = _assert_pass(row, log)
    assert int(h.group(3)) == depth


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("chunk", ["", "-S 200 -o 136"])
@pytest.mark.parametrize("fx", list(FIXTURES))
def test_prefill_fixture(vx, fx, chunk, variant):
    d = FIX / FIXTURES[fx]
    if not (d / "meta.json").is_file():
        pytest.fail(f"fixture {d} missing: run `python -B -m model.make_prefill_fixtures --model ...`")
    row, log = _run(vx, f"-f {d} {chunk} {VARIANTS[variant]}".replace("  ", " "))
    _assert_pass(row, log, fixture=True)


@pytest.mark.vortex
@pytest.mark.psbase_skip
def test_prefill_dxa_refused_without_ext(vx):
    row, log = _run(vx, f"{_shape_args('260K', 1, 16)} -v dxa", {"config": "smoke-default"})
    assert row["status"] == "failed" and "needs a device built with -DVX_CFG_EXT_DXA_ENABLE" in log
