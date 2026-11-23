"""Tests: ps_mixed_attn, fused mixed-batch attention: ONE launch, CTAs [0, n_p) a prefill chunk
(queries s0 .. s0+c-1 over keys 0 .. s0+c-1), CTAs [n_p, n_p+n_d) decode queries (one per sequence, ragged contexts).

Cheap (always run):
  test_shared_headers               warp_ops.h, prefill_common.h, decode_common.h are symlinks to the files
                                    (shared ps_expf and LMEM layouts; the symlinks keep vxrun's app-source hash covering them).
  test_bodies_match_approved_kernels  the floating-point phases of the two CTA bodies (S, M1, M2, V, final store) are the
                                    validated lmem_tiled / gqa_shared_kv code, text-identical (the vortex test
                                    test_mixed_equals_separate proves the outputs bit-identical).
  test_mixed_fixture_generator      model/make_mixed_fixtures.py: synthetic and real-model (stories260K prefill fixture,
                                    N = 24, private dir) fixtures; every NumPy output row equals a float64 attention
                                    within NumPy's float32 error; numpy_attention == Llama2cRef.attention.

Vortex-marked (PS_RUN_VORTEX=1; serial SimX runs through scripts/vxrun.py -> lowload.sh; config REG-CFG = 4 cores, so
prefill and decode CTAs run concurrently on different cores):
  test_mixed_numpy[...]            : (c, decode count) in {(16, 4), (64, 8), (256, 16)} x inputs {stories260K
                                    L4 (8,4,8) and stories15M L0 (6,6,48) real-model data, synthetic (8,2,64)}: NumPy
                                    fixtures with chunk offsets s0 > 0 (except 15M c = 256, s0 = 0: N = 256) and ragged
                                    decode contexts (1, partial tiles, multi-tile) -> PASSED: PSCHECK errors = 0 for
                                    device vs float64 AND device vs NumPy (both parts), NumPy vs float64, max_rel <= tol,
                                    within_plan_tol = 1 (rel 1e-4), one valid timestamp record per CTA.
  test_mixed_gqa8_d128              (16, 2, 128): G = 8 decode template and the D = 128 prefill tiles (Qt 1, Tk 7), synthetic.
  test_mixed_equals_separate[...]   the fused launch's prefill and decode outputs are BIT-identical (PSDIGEST) to separate
                                    ps_prefill_attn (lmem) and ps_decode_attn_gqa (gqa) launches on the same
                                    inputs (the app draws its inputs exactly as those apps do).
  test_mixed_deep_negative          every logit < -104 in both parts (errata 7, online form), multi-tile both parts.
  test_mixed_single_part[...]       decode only (c = 0) and prefill only (no decode sequence).
  test_mixed_order_and_ts[...]      decode-first CTA order (-r) and timestamps off (-a 0) give the same output digests.
The vortex tests use a PRIVATE results dir build/tmp/mixed_attn_test (rows + ledger), deleted at session start.
"""
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

PS = __import__("pathlib").Path(__file__).resolve().parents[1]
KDIR = PS / "kernels"
PLAN_TOL = 1e-4
PSCHECK_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) "
                        r"tol=(\S+) errors=(\d+)", re.M)
DIGEST_RE = re.compile(r"^PSDIGEST: (\S+) n=(\d+) fnv1a64=([0-9a-f]+)", re.M)
HDR_RE = re.compile(r"^ps_mixed_attn: .* c=(\d+) s0=(\d+) .* n_seq=(\d+) .* Qt=(\d+) Tk=(\d+) .* n_p=(\d+) n_d=(\d+) "
                    r"grid=(\d+) order=(\S+) ts=(\d) .*logits_p=\[(\S+),(\S+)\] logits_d=\[(\S+),(\S+)\]", re.M)
CTA_RE = re.compile(r"^PSCTA: cta=(\d+) region=([pd]) idx=(\d+) core=(\d+) start=(\d+) end=(\d+)$", re.M)
CFG = {"config": "REG-CFG"}

# points: (c, decode count) -> (s0, ragged decode contexts); contexts cover T = 1, partial tiles (Tt = 32),
# exact tile multiples and multi-tile sequences
POINTS = {
    (16, 4): (40, [1, 17, 33, 100]),
    (64, 8): (100, [5, 32, 64, 65, 96, 129, 150, 200]),
    (256, 16): (None, [1, 9, 17, 31, 32, 33, 48, 64, 80, 97, 120, 128, 160, 200, 230, 256]),
}
SOURCES = {   # name -> (make_mixed_fixture kwargs, s0 used for the c = 256 point)
    "260K-L4": ({"model": "stories260K", "layer": 4}, 200),     # (8, 4, 8), prefill fixture N = 512
    "15M-L0": ({"model": "stories15M", "layer": 0}, 0),         # (6, 6, 48), prefill fixture N = 256
    "syn-8-2-64": ({"shape": (8, 2, 64), "seed": 7}, 64),       # scaled Llama-8B-like (bench shape)
}


# ------------------------------------------------------------------------------------------------- cheap
def test_shared_headers():
    for name, target in (("warp_ops.h", "softmax_fixed/warp_ops.h"), ("prefill_common.h", "prefill_attn/common.h"),
                         ("decode_common.h", "decode_attn_gqa/common.h")):
        p = KDIR / "mixed_attn" / name
        assert p.is_symlink() and p.resolve() == (KDIR / target).resolve(), name


# (ps patch mem-class): PhaseShift hint/marker statements (kernels/common/ps_hint.h) expand to ((void)0) unless the
# kernel is built with -DPS_HINTS=1 (PS_HINTS=0 ELFs are byte-identical to the pre-hint builds, results/<run>/elf_sha_*).
# The body comparison drops ONLY whole lines that consist of one such statement (optionally guarded by `if (tid == 0)`);
# every other line, in particular every floating-point statement, must still match character for character.
_PS_HINT_LINE = re.compile(r"^[ \t]*(?:if \(tid == 0\) )?PS_(?:MEM_HINT|MARK)\(PS_(?:CLASS|MARK)_[A-Z_]+\);[ \t]*\n", re.M)


def _strip_ps(text: str) -> str:
    return _PS_HINT_LINE.sub("", text)


def _between(src: str, start: str, end: str) -> str:
    i = src.index(start)
    return _strip_ps(src[i:src.index(end, i)])


def test_ps_hint_line_filter_is_narrow():
    """The filter removes exactly the hint/mark statement lines and nothing else."""
    src = ("  PS_MEM_HINT(PS_CLASS_KV_PERSIST);\n    if (tid == 0) PS_MARK(PS_MARK_DECODE_TILE);\n"
           "  ks[i] = kbase[g]; PS_MEM_HINT(PS_CLASS_ORDINARY);\n  // PS_MEM_HINT(PS_CLASS_X);\n  a = fma(b, c, a);\n")
    assert _strip_ps(src) == "  ks[i] = kbase[g]; PS_MEM_HINT(PS_CLASS_ORDINARY);\n  // PS_MEM_HINT(PS_CLASS_X);\n  a = fma(b, c, a);\n"
    mixed = (KDIR / "mixed_attn" / "kernel.cpp").read_text()
    dec = (KDIR / "decode_attn_gqa" / "kernel.cpp").read_text()
    # every hint/mark statement in the two kernels is a whole line the filter recognises (none hides inside other code)
    for t in (mixed, dec):
        code = [ln for ln in t.splitlines() if not ln.lstrip().startswith("//")]
        uses = [ln for ln in code if "PS_MEM_HINT(" in ln or "PS_MARK(" in ln]
        assert uses and len(uses) == len(_PS_HINT_LINE.findall(t)), uses


def test_bodies_match_approved_kernels():
    mixed = (KDIR / "mixed_attn" / "kernel.cpp").read_text()
    pre = (KDIR / "prefill_attn" / "kernel.cpp").read_text()
    dec = (KDIR / "decode_attn_gqa" / "kernel.cpp").read_text()
    mp = mixed[mixed.index("void prefill_cta("):mixed.index("template <uint32_t G>")]
    md = mixed[mixed.index("void decode_cta("):mixed.index("decode_dispatch(")]
    # prefill phases S .. V (up to the end-of-tile barrier) and the final store
    a = _between(pre, "    // S: scores, item = (4-row block rb, key t)", "    if (MODE == Mode::Dxa)\n      stage_bar[s]")
    b = _between(mp, "    // S: scores, item = (4-row block rb, key t)", "    sync.arrive_and_wait();                      // Ks/Vs")
    assert a == b
    assert _between(pre, "  // out = acc / l for the valid rows", "\n}\n") == _between(mp, "  // out = acc / l for the valid rows", "\n}\n")
    assert _between(pre, "  // Q tile -> LMEM", "  const uint32_t depth") == _between(mp, "  // Q tile -> LMEM", "  for (uint32_t j = 0;")
    # the lmem copy loop
    assert _between(pre, "      // cooperative copy of rows", "      sync.arrive_and_wait();") == \
        _between(mp, "      // cooperative copy of rows", "      sync.arrive_and_wait();")
    # decode phases S .. V and the final store
    a = _between(dec, "    // S: scores of the tile, thread per position, all G heads", "    if (MODE == Mode::Dxa)\n      stage_bar[s]")
    b = _between(md, "    // S: scores of the tile, thread per position, all G heads", "    sync.arrive_and_wait();                      // ps is")
    assert a == b
    assert "for (uint32_t i = tid; i < G * D; i += nth) out[i] = acc[i] / lst[i / D];" in dec
    assert "for (uint32_t i = tid; i < G * D; i += nth) out[i] = acc[i] / lst[i / D];" in md
    # the inline-FMA macro (errata 30) precedes warp_ops.h, as in prefill_attn
    assert mixed.index("#define __builtin_fmaf(a, b, c) __builtin_elementwise_fma") < mixed.index('#include "warp_ops.h"')


def _f64_attention(H_q, H_kv, D, q, k, v):
    G = H_q // H_kv
    qh = q.astype(np.float64).reshape(H_q, D)
    K = k.astype(np.float64).reshape(-1, H_kv, D)[:, np.arange(H_q) // G]
    V = v.astype(np.float64).reshape(-1, H_kv, D)[:, np.arange(H_q) // G]
    s = np.einsum("hd,thd->ht", qh, K) / np.sqrt(D)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("ht,thd->hd", p, V).reshape(-1), np.einsum("ht,thd->hd", p, np.abs(V)).reshape(-1)


def test_mixed_fixture_generator(tmp_path):
    from model.make_fixtures import load_fixture
    from model.make_mixed_fixtures import make_mixed_fixture, numpy_attention
    from model.make_prefill_fixtures import make_prefill_fixtures
    from model.ref_llama2c import Llama2cRef, load_model
    make_prefill_fixtures("stories260K", [0], 24, tmp_path / "pf")
    cases = {"syn": dict(shape=(8, 2, 16), seed=3, chunk=6, s0=5, ctx=[1, 4, 9]),
             "m": dict(model="stories260K", layer=0, prefill_dir=tmp_path / "pf", chunk=8, s0=10, ctx=[3, 24, 1, 17])}
    for nm, kw in cases.items():
        meta = make_mixed_fixture(tmp_path / nm, **kw)
        m2, a = load_fixture(tmp_path / nm)
        assert meta == m2 and m2["op"] == "attention_mixed"
        p = m2["params"]
        H, Hk, D, c, s0, ctx = p["n_heads"], p["n_kv_heads"], p["head_size"], p["chunk"], p["s0"], p["ctx"]
        assert ctx == kw["ctx"] and c == kw["chunk"] and s0 == kw["s0"]
        assert a["q_p"].shape == (c, H * D) and a["k_p"].shape == (s0 + c, Hk * D)
        assert a["k_d"].shape == (len(ctx), max(ctx), Hk * D) and a["out_d"].shape == (len(ctx), H * D)
        worst = 0.0
        for i in range(c):
            ref, sc = _f64_attention(H, Hk, D, a["q_p"][i], a["k_p"][:s0 + i + 1], a["v_p"][:s0 + i + 1])
            worst = max(worst, (np.abs(a["out_p"][i] - ref) / sc).max())
        for b, t in enumerate(ctx):
            ref, sc = _f64_attention(H, Hk, D, a["q_d"][b], a["k_d"][b, :t], a["v_d"][b, :t])
            worst = max(worst, (np.abs(a["out_d"][b] - ref) / sc).max())
            assert not a["k_d"][b, t:].any() and not a["v_d"][b, t:].any()   # unused rows are zero (the app NaNs them)
        assert worst < 1e-5, (nm, worst)       # NumPy float32, T <= 24 terms: ~1e-6 expected
    # numpy_attention is Llama2cRef.attention (same output bits on a real layer)
    cfg, w, _ = load_model("stories260K")
    ref = Llama2cRef(cfg, w)
    rng = np.random.default_rng(0)
    q = rng.standard_normal(cfg.dim).astype(np.float32)
    kc = rng.standard_normal((7, cfg.kv_dim)).astype(np.float32)
    vc = rng.standard_normal((7, cfg.kv_dim)).astype(np.float32)
    assert np.array_equal(ref.attention(q, kc, vc)[0], numpy_attention(cfg.n_heads, cfg.n_kv_heads, cfg.head_size, q, kc, vc))


# ------------------------------------------------------------------------------------------------- vortex
PRIV = PS / "build" / "tmp" / "mixed_attn_test"


@pytest.fixture(scope="module")
def vx():
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun  # noqa: E402
    if PRIV.exists():
        shutil.rmtree(PRIV)
    assert subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], capture_output=True).returncode == 0
    return vxrun.VxRun("P2.8-test", results=PRIV)


def _run(vx, app, args, cfg=CFG):
    row = vx.run({"tree": "base", "app": app, "args": args, "perf": 1, **cfg}, timeout_s=14400)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


def _assert_pass(row, log, parts=("prefill", "decode"), fixture=False):
    assert row["status"] == "ok", (row["reason"], log[-2500:])
    c = {m.group(1): (float(m.group(4)), float(m.group(6)), int(m.group(7))) for m in PSCHECK_RE.finditer(log)}
    want = {f"{p}:vs_f64" for p in parts}
    if fixture:
        want |= {f"{p}:fixture_vs_f64" for p in parts} | {f"{p}:vs_fixture" for p in parts}
    assert set(c) == want, c
    for k, (rel, tol, err) in c.items():
        assert err == 0 and rel <= tol, (k, rel, tol)
    assert "within_plan_tol=1" in log      # the criterion itself (rel 1e-4 vs float64), whatever the bound
    h = HDR_RE.search(log)
    assert h, log[:2000]
    grid, n_p, n_d = int(h.group(8)), int(h.group(6)), int(h.group(7))
    assert grid == n_p + n_d
    if h.group(10) == "1":   # timestamps: one valid record per CTA, region ranges as laid out
        assert "ts_ok=1" in log
        ctas = [m.groups() for m in CTA_RE.finditer(log)]
        assert len(ctas) == grid and sorted(int(x[0]) for x in ctas) == list(range(grid))
        first = n_d if h.group(9) == "decode_first" else n_p
        # core ids must be < the device's core count (4 at REG-CFG; read from the header so the PS-BASE
        # re-runs at 1/8/16 cores check the same property)
        ncores = int(re.search(r" cores=(\d+) ", log).group(1))
        for cta, region, idx, core, st, en in ctas:
            cta, idx = int(cta), int(idx)
            in_first = cta < first
            assert region == ("d" if (h.group(9) == "decode_first") == in_first else "p"), (cta, region)
            assert idx == (cta if in_first else cta - first) and int(en) >= int(st) and int(core) < ncores
            assert int(en) <= row["cycles"] + 10**6      # mcycle counts from device reset; one launch per run
    return h, {m.group(1): m.group(3) for m in DIGEST_RE.finditer(log)}


def _make_fixture(src, point):
    from model.make_mixed_fixtures import make_mixed_fixture
    kw, s0_256 = SOURCES[src]
    c, nd = point
    s0, ctx = POINTS[point]
    s0 = s0_256 if s0 is None else s0
    d = PRIV / "fixtures" / f"{src}_c{c}_n{nd}"
    if not (d / "meta.json").is_file():
        try:
            make_mixed_fixture(d, chunk=c, s0=s0, ctx=ctx, name=f"{src}_c{c}_n{nd}", **kw)
        except FileNotFoundError as e:
            pytest.fail(f"{e} (prefill fixtures)")
    return d


@pytest.mark.vortex
@pytest.mark.parametrize("point", list(POINTS), ids=lambda p: f"c{p[0]}-n{p[1]}")
@pytest.mark.parametrize("src", list(SOURCES))
def test_mixed_numpy(vx, src, point):
    d = _make_fixture(src, point)
    row, log = _run(vx, "ps_mixed_attn", f"-f {d}")
    h, _ = _assert_pass(row, log, fixture=True)
    assert int(h.group(1)) == point[0] and int(h.group(3)) == point[1]


@pytest.mark.vortex
def test_mixed_gqa8_d128(vx):
    row, log = _run(vx, "ps_mixed_attn", "-H 16 -K 2 -D 128 -c 8 -o 9 -T 9,40")
    h, _ = _assert_pass(row, log)
    # default tiles at D = 128: (Qt, Tk) = (1, 7) at the 16 KiB LMEM / 4 cores of REG-CFG. The rule depends on the
    # device's LMEM size and core count (PS-BASE has NT = 32 -> 64 KiB LMEM), so the expectation is the rule
    # (tests/test_prefill_attn.py default_tiles) evaluated at the LMEM size and core count the header reports.
    import importlib.util
    spec = importlib.util.spec_from_file_location("ps_prefill_rule", PS / "tests" / "test_prefill_attn.py")
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    assert rule.default_tiles(8, 128, 8, 1, cap=16384, B=1, H_kv=2, cores=4) == (1, 7)   # the validated expectation
    m = re.search(r" cores=(\d+) lmem=\d+/(\d+) ", log)
    cores, cap = int(m.group(1)), int(m.group(2))
    assert (int(h.group(4)), int(h.group(5))) == rule.default_tiles(8, 128, 8, 1, cap=cap, B=1, H_kv=2, cores=cores)


SEP_CASES = {   # fused args -> (ps_prefill_attn args, ps_decode_attn_gqa args) on the same inputs
    "8-4-8": ("-H 8 -K 4 -D 8 -c 16 -o 8 -T 17 -n 4", "-B 1 -H 8 -K 4 -D 8 -S 16 -o 8", "-B 4 -H 8 -K 4 -D 8 -T 17 -v gqa"),
    "8-2-64": ("-H 8 -K 2 -D 64 -c 20 -o 13 -T 40 -n 3", "-B 1 -H 8 -K 2 -D 64 -S 20 -o 13",
               "-B 3 -H 8 -K 2 -D 64 -T 40 -v gqa"),
    "6-6-48-neg": ("-H 6 -K 6 -D 48 -c 12 -o 30 -T 70 -n 2 -d neg -q 20 -s 5 -u 9",
                   "-B 1 -H 6 -K 6 -D 48 -S 12 -o 30 -d neg -q 20 -s 5", "-B 2 -H 6 -K 6 -D 48 -T 70 -v gqa -d neg -q 20 -s 9"),
}


@pytest.mark.vortex
@pytest.mark.parametrize("case", list(SEP_CASES))
def test_mixed_equals_separate(vx, case):
    fa, pa, da = SEP_CASES[case]
    row, log = _run(vx, "ps_mixed_attn", fa)
    _, dg = _assert_pass(row, log)
    rp, lp = _run(vx, "ps_prefill_attn", pa)
    rd, ld = _run(vx, "ps_decode_attn_gqa", da)
    assert rp["status"] == "ok" and rd["status"] == "ok", (lp[-1500:], ld[-1500:])
    gp = {m.group(1): m.group(3) for m in DIGEST_RE.finditer(lp)}
    gd = {m.group(1): m.group(3) for m in DIGEST_RE.finditer(ld)}
    assert dg["prefill:vs_f64"] == gp["launch0:vs_f64"]
    assert dg["decode:vs_f64"] == gd["launch0:vs_f64"]


@pytest.mark.vortex
def test_mixed_deep_negative(vx):
    row, log = _run(vx, "ps_mixed_attn", "-H 8 -K 4 -D 8 -c 40 -o 30 -Q 8 -t 12 -T 100,7,65 -d neg -q 150")
    h, _ = _assert_pass(row, log)
    assert float(h.group(12)) < -104.0 and float(h.group(14)) < -104.0, h.group(0)   # max logit of each part
    assert (int(h.group(4)), int(h.group(5))) == (8, 12)


@pytest.mark.vortex
@pytest.mark.parametrize("case", ["decode-only", "prefill-only"])
def test_mixed_single_part(vx, case):
    args = {"decode-only": "-H 8 -K 2 -D 64 -c 0 -T 1,33,64", "prefill-only": "-H 8 -K 2 -D 64 -c 21 -o 4 -T 0"}[case]
    row, log = _run(vx, "ps_mixed_attn", args)
    h, _ = _assert_pass(row, log, parts=("decode",) if case == "decode-only" else ("prefill",))
    assert (int(h.group(6)) == 0) == (case == "decode-only") and (int(h.group(7)) == 0) == (case == "prefill-only")


@pytest.mark.vortex
@pytest.mark.parametrize("flag", ["-r", "-a 0"])
def test_mixed_order_and_ts(vx, flag):
    base = SEP_CASES["8-2-64"][0]
    row0, log0 = _run(vx, "ps_mixed_attn", base)
    _, d0 = _assert_pass(row0, log0)
    row, log = _run(vx, "ps_mixed_attn", f"{base} {flag}")
    h, d1 = _assert_pass(row, log)
    assert d1 == d0
    if flag == "-r":
        assert h.group(9) == "decode_first"
    else:
        assert h.group(10) == "0" and "PSCTA:" not in log
