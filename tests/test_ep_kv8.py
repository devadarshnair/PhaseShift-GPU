"""Tests: KV-precision decode attention -- int8 KV (and fp16 KV with Zfhmin) (methodology §18.9).

Non-vortex (always run; no SimX; small g++ compiles):
  test_stock_sources_untouched        kernels/{decode_attn_gqa,rope}/{kernel.cpp,common.h} and rope/kernel_tab.cpp byte-identical to their pre-int8-KV sources (sha256):
                                      the new entries live in new files
  test_entries_and_layout             app.conf of ps_decode_attn_gqa / ps_rope list the new TUs; the four attention entries and rope_kv_tab_q8 exist; ps_llama has 12 TUs
  test_lmem_layout_mirror             the LMEM footprint formula of common_kv.h (parsed); every shape fits the 16 KiB per-core LMEM for the plain kernels at Tt = 32 and
                                      the dxa kernels at depth 1..4 with Tt = 1 (as test_decode_attn_gqa)
  test_quantizer_port_matches_host    model/ref_q80.py kv8_head_quantize == the host C++ quantizer (kernels/common/ps_q80.h) bit for bit on random, constant, zero and spiky heads;
                                      the dequantization error of every element is <= scale / 2; q in [-127, 127]
  test_f16_host_conversion            kernels/decode_attn_gqa/f16_host.h (the kv16 cache the host builds) == NumPy float16 on 77 k random and edge values, both directions
  test_numpy_kv8_model                Llama2cRefKV8 stores exactly the dequantized quantized K / V; its token argmax at positions 0..5 equals the Q8_0 reference's; table RoPE
                                      equals the default RoPE within 1e-6 of the pair norm
  test_batch_files_up_to_date, test_methodology_18_9, test_epk_results   generated YAML == make_batch.py; documentation; stored results (when present)

Vortex-marked (PS_RUN_VORTEX=1; E1 = PS-BASE-S0 on the ps tree; kv16 on EP-SMOKE-S0-zfhmin = E1 + Zfhmin; each SimX run through scripts/vxrun.py -> lowload.sh;
private results dir build/tmp/epk_test, deleted at session start). Do NOT wrap this pytest in lowload.sh (nested lock, deadlock): use the outer-flock pattern of
results/<run>/run_tests.sh.
  test_attn_kv_grid[...]              (H_q, H_kv, D) in {(8,4,8), (6,6,48), (8,2,64), (16,2,128)} x B in {1, 4} x T in {1, 17, 256} x {gqa_kv8, dxa_kv8}; kv16 on two shapes:
                                      PASSED, PSCHECK errors = 0, max_rel <= tol (the derived bound of main.cpp over the dequantized operands), within_plan_tol = 1
  test_attn_kv_all_negative / deep_negative_multitile / dxa_depth     the edge cases for the new entries
  test_attn_kv_fixture                attention fixtures (stories260K L0_P17, stories15M L5_P255) through every new entry; the cache error PSKVQ is sane
  test_kv16_refused_without_zfhmin    gqa_kv16 on E1 (no Zfhmin) -> FAILED before launch
  test_rope_kv8_vs_host_and_numpy     `ps_rope -R table -K kv8`: K8 / V8 bytes and scales == the host recomputation (PSCHECK k8_exact, v8_exact, ks_exact, vs_exact: 0 errors), and the dump is
                                      reproduced bit for bit by the NumPy port; untouched cache cells (kv8_untouched)
  test_ps_llama_kv8_refused_without_q80   `-Q kv8` without `-q q80` is refused
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS))
FIX = PS / "build" / "fixtures"
RES = PS / "results" / "EP.K"
PSCHECK_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) tol=(\S+) errors=(\d+)", re.M)
PSDIGEST_RE = re.compile(r"^PSDIGEST: (\S+) n=(\d+) fnv1a64=([0-9a-f]{16})$", re.M)
HDR_RE = re.compile(r"^ps_decode_attn: .* bound=(\S+) tol=(\S+)", re.M)
KVQ_RE = re.compile(r"^PSKVQ: cache_error_vs_fp32kv max_rel=(\S+)", re.M)
E1, E1Z = "PS-BASE-S0", "EP-SMOKE-S0-zfhmin"
PLAN_TOL = 1e-4

# sha256 of the sources before the int8-KV entries were added (measured before the first edit)
STOCK_SHA = {
    "kernels/decode_attn_gqa/kernel.cpp": "4b0108c474abe78409d9834d97e8d6dc3f1c5408767b2d63baebadd9ccf1981b",
    "kernels/decode_attn_gqa/common.h": "ec108a62817dfa88a7bbfcce092f3cbda4175128e74a0d9fdcaa92e34a2b6f34",
    "kernels/rope/kernel.cpp": "cc3f2f6217dfe196620752ff56511eb1930407d17dfd43640f4804f954c94b03",
    "kernels/rope/kernel_tab.cpp": "7733c994922389e376bccbcab1aa2cae6f7b40ab3b26fbe44230f733406ee105",
    "kernels/rope/common.h": "4b8bf9d4d21bb913af740a504709cbc1f2f72cb801145a8068adae95b9232637",
}


# ------------------------------------------------------------------------------------------------- cheap
def test_stock_sources_untouched():
    for f, h in STOCK_SHA.items():
        assert hashlib.sha256((PS / f).read_bytes()).hexdigest() == h, f


def test_entries_and_layout():
    assert (PS / "kernels/decode_attn_gqa/app.conf").read_text().split() == ["VX_SRCS=kernel.cpp", "kernel_kv.cpp"]
    assert (PS / "kernels/rope/app.conf").read_text().split() == ["VX_SRCS=kernel.cpp", "kernel_tab.cpp", "kernel_kv8.cpp"]
    kv = (PS / "kernels/decode_attn_gqa/kernel_kv.cpp").read_text()
    assert sorted(re.findall(r"__kernel void (\w+)\(", kv)) == ["attn_dxa_kv16", "attn_dxa_kv8", "attn_gqa_kv16", "attn_gqa_kv8"]
    assert re.findall(r"__kernel void (\w+)\(", (PS / "kernels/rope/kernel_kv8.cpp").read_text()) == ["rope_kv_tab_q8"]
    assert "#pragma clang fp contract(off)" in (PS / "kernels/rope/kernel_kv8.cpp").read_text()
    assert "#pragma clang fp contract(off)" not in kv                       # the attention keeps the stock contraction behaviour
    srcs = re.search(r"^VX_SRCS=(.*)$", (PS / "runtime/llama_vx/app.conf").read_text(), re.M).group(1).split()
    assert len(srcs) == 12 and srcs[-2:] == ["k_kv8r.cpp", "k_kv8a.cpp"]
    for tu, inc in (("k_kv8r.cpp", "reused/rope/kernel_kv8.cpp"), ("k_kv8a.cpp", "reused/decode_attn_gqa/kernel_kv.cpp")):
        assert re.findall(r'^#include "([^"]+)"', (PS / "runtime/llama_vx" / tu).read_text(), re.M) == [inc]
    # kv16 entries only exist with Zfhmin: the guard is the compiler's feature macro
    assert kv.count("#if defined(__riscv_zfhmin)") >= 3


def _lmem_words(G, D, Tt, depth, dxa, esz):
    r16 = lambda n: (n + 15) & ~15  # noqa: E731  (common_kv.h ps_kv_round16)
    nst = depth if dxa else 1
    sc = r16(Tt) if esz == 1 else 0
    data = r16((Tt * D * esz + 3) // 4)
    return r16(G * D) * 2 + r16(G * Tt) + r16(3 * G) + 2 * nst * sc + (2 * depth * data if dxa else 0)


def test_lmem_layout_mirror():
    src = (PS / "kernels/decode_attn_gqa/common_kv.h").read_text()
    for frag in ("L.sc_stage = (esz == 1) ? ps_kv_round16(Tt) : 0u;", "L.data_stage = ps_kv_round16((Tt * D * esz + 3u) / 4u);",
                 "L.acc = L.qs + ps_kv_round16(G * D);", "L.ps = L.acc + ps_kv_round16(G * D);", "L.stat = L.ps + ps_kv_round16(G * Tt);",
                 "L.ksc = L.stat + ps_kv_round16(3 * G);", "L.vsc = L.ksc + nst * L.sc_stage;", "L.kb = L.vsc + nst * L.sc_stage;",
                 "L.vb = L.kb + (dxa ? depth * L.data_stage : 0u);", "L.total = L.vb + (dxa ? depth * L.data_stage : 0u);"):
        assert frag in src, frag
    lmem = 16384
    for hq, hkv, d in ((8, 4, 8), (6, 6, 48), (8, 2, 64), (16, 2, 128)):
        g = hq // hkv
        for esz in (1, 2):
            assert _lmem_words(g, d, 32, 1, False, esz) * 4 <= lmem
            for depth in (1, 2, 4):
                assert _lmem_words(g, d, 1, depth, True, esz) * 4 <= lmem, (hq, hkv, d, depth, esz)
        # int8 KV fits larger tiles than fp32 KV in the same LMEM (the footprint formula, esz = 4)
        big = lambda esz_: max(t for t in range(1, 33) if _lmem_words(g, d, t, 2, True, esz_) * 4 <= lmem)  # noqa: E731
        assert big(1) >= big(4) and big(2) >= big(4)


def test_quantizer_port_matches_host(tmp_path):
    from model import ref_q80 as rq
    rng = np.random.default_rng(7)
    heads = [rng.standard_normal((50, 48)).astype(np.float32), (rng.standard_normal((50, 64)) * 1e3).astype(np.float32),
             np.zeros((3, 48), np.float32), np.full((2, 64), 0.37, np.float32),
             np.concatenate([np.zeros((1, 63)), np.full((1, 1), -5.0)], axis=1).astype(np.float32),
             (rng.standard_normal((20, 8)) * 1e-30).astype(np.float32)]
    src = tmp_path / "q.cpp"
    src.write_text(textwrap.dedent(r'''
        #include <cstdio>
        #include <cstdlib>
        #include <cstring>
        #include <vector>
        #include "ps_q80.h"
        int main(int argc, char** argv) {   // stdin: hs rows, then rows * hs float32 bit patterns (hex); stdout: per row the scale bits and the int8 values
          unsigned hs, rows; if (scanf("%u %u", &hs, &rows) != 2) return 1;
          std::vector<float> x(hs); std::vector<int8_t> q(hs); float s;
          for (unsigned r = 0; r < rows; ++r) {
            for (unsigned i = 0; i < hs; ++i) { unsigned b; if (scanf("%x", &b) != 1) return 1; std::memcpy(&x[i], &b, 4); }
            ps::quantize_q80(x.data(), hs, hs, q.data(), &s);
            unsigned sb; std::memcpy(&sb, &s, 4); printf("%08x", sb);
            for (unsigned i = 0; i < hs; ++i) printf(" %d", (int)q[i]);
            printf("\n");
          }
        }'''))
    exe = tmp_path / "q"
    r = subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(PS / "kernels/common"), str(src), "-o", str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for h in heads:
        hs = h.shape[1]
        inp = f"{hs} {h.shape[0]}\n" + "\n".join(" ".join("%08x" % v for v in row.view(np.uint32)) for row in h) + "\n"
        out = subprocess.run([str(exe)], input=inp, capture_output=True, text=True, check=True).stdout.splitlines()
        for row, line in zip(h, out):
            tok = line.split()
            sc, q = np.array([int(tok[0], 16)], np.uint32).view(np.float32)[0], np.array([int(t) for t in tok[1:]], np.int8)
            qn, sn = rq.kv8_head_quantize(row, hs)
            assert qn.tolist() == q.tolist() and np.array([sn[0]]).view(np.uint32)[0] == np.array([sc]).view(np.uint32)[0]
            assert q.min() >= -127 and q.max() <= 127
            if sc > 0:
                deq = rq.kv8_dequantize(qn, sn, hs)
                assert np.all(np.abs(row.astype(np.float64) - deq.astype(np.float64)) <= 0.5 * float(sc) * (1 + 1e-5) + 1e-45)
            else:
                assert not q.any()


def test_f16_host_conversion(tmp_path):
    src = tmp_path / "f.cpp"
    src.write_text(textwrap.dedent(r'''
        #include <cstdio>
        #include "f16_host.h"
        int main() { unsigned b; while (scanf("%x", &b) == 1) { float f; std::memcpy(&f, &b, 4); unsigned short h = ps_f16::f32_to_f16_rne(f);
          float g = ps_f16::f16_to_f32(h); unsigned gb; std::memcpy(&gb, &g, 4); printf("%04x %08x\n", h, gb); } }'''))
    exe = tmp_path / "f"
    r = subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(PS / "kernels/decode_attn_gqa"), str(src), "-o", str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(-2, 2, 50000).astype(np.float32), (rng.standard_normal(20000) * 1e-3).astype(np.float32),
                        (rng.standard_normal(5000) * 1e-6).astype(np.float32),
                        np.array([0, -0.0, 65504, 65519.9, 65520, 70000, 1e-8, 6e-8, 3e-8, 5.96e-8, 2.98e-8, 2.99e-8, np.inf, -np.inf], np.float32),
                        rng.uniform(-70000, 70000, 2000).astype(np.float32)])
    out = subprocess.run([str(exe)], input="\n".join("%08x" % v for v in x.view(np.uint32)), capture_output=True, text=True, check=True).stdout.splitlines()
    with np.errstate(over="ignore"):
        h = x.astype(np.float16)
    hb, gb = h.view(np.uint16), h.astype(np.float32).view(np.uint32)
    assert len(out) == len(x)
    for i, line in enumerate(out):
        a, b = line.split()
        assert int(a, 16) == int(hb[i]) and int(b, 16) == int(gb[i]), (x[i], line)


def test_numpy_kv8_model():
    from model import ref_llama2c as ref
    from model import ref_q80 as rq
    path = rq.DEFAULT_Q80
    if not path.is_file():
        pytest.skip("build/models/stories15M_q80.bin missing (export)")
    m = rq.load_q80_checkpoint(path)
    seq, _ = ref.read_golden_tokens(PS / "results" / "EP.2" / "golden_tokens_15M_q80.txt")
    a, b = rq.Llama2cRefQ80(m), rq.Llama2cRefKV8(m)
    hs = m.cfg.head_size
    for p in range(6):
        la, lb = a.forward(seq[p], p), b.forward(seq[p], p)
        assert int(np.argmax(la)) == int(np.argmax(lb)), p
        assert float(np.max(np.abs(la - lb)) / np.max(np.abs(la))) < 0.1
    for (l, p), (kq, ks, vq, vs) in b.kv_q.items():                 # the cache holds exactly the dequantized quantized vectors
        assert np.array_equal(b.key_cache[l, p], rq.kv8_dequantize(kq, ks, hs)) and np.array_equal(b.value_cache[l, p], rq.kv8_dequantize(vq, vs, hs))
        assert kq.dtype == np.int8 and ks.dtype == np.float32 and ks.shape == (m.cfg.n_kv_heads,)
    # table RoPE vs the default RoPE (np.cos / np.sin in float32): within 1e-6 of the pair norm
    q = np.random.default_rng(3).standard_normal(m.cfg.dim).astype(np.float32)
    k = np.random.default_rng(4).standard_normal(m.cfg.kv_dim).astype(np.float32)
    q1, k1 = a.rope(q, k, 200)
    q2, k2 = b.rope(q, k, 200)
    pn = np.hypot(q[0::2], q[1::2])
    assert np.max(np.abs(q1[0::2] - q2[0::2]) / pn) < 1e-4 and np.max(np.abs(q1[1::2] - q2[1::2]) / pn) < 1e-4
    t = rq.rope_table_np(hs, 4)
    assert np.array_equal(t[0], np.stack([np.ones(hs // 2, np.float32), np.zeros(hs // 2, np.float32)], axis=1))   # pos 0: (1, 0) exactly


def test_batch_files_up_to_date():
    r = subprocess.run([sys.executable, "-B", str(RES / "make_batch.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_methodology_18_9():
    meth = (PS / "docs/methodology.md").read_text()
    assert "### 18.9 " in meth
    sec = meth[meth.index("### 18.9 "):]
    nxt = re.search(r"\n### 18\.10 |\n## 19\.", sec)
    sec = sec[:nxt.start()] if nxt else sec
    for s in ("per (position, KV head)", "127.0f", "dequantiz", "[B][H_kv][Tmax]", "scales are counted", "AI = 2g/q", "Zfhmin", "fp contract", "PSKVQ", "bit-identical",
              "tolerance", "P8K8"):
        assert s in sec, s


def test_epk_results():
    p = RES / "summary.json"
    if not p.is_file():
        pytest.skip("results/EP.K/summary.json not generated yet (analyze.py)")
    j = json.loads(p.read_text())
    assert j["rows_total"] == j["rows_ok"] and j["rows_total"] >= 100, (j["rows_total"], j["rows_ok"])
    assert j["att"]["points"] == 18 and j["att"]["variants_per_point"] == 6 and j["att"]["all_passed"] is True
    assert j["att"]["kv8_bytes_ratio_min"] > 0.25 and j["att"]["kv8_bytes_ratio_max"] < 0.30      # int8 + scale: (D + 4) / (4 D) of the fp32 bytes
    assert j["att"]["kv16_bytes_ratio"] == 0.5
    # stock entries: identical instruction counts; cycles within 0.1 % of rows (a new translation unit shifts the link layout; observed 0.004-0.008 % per unit)
    assert j["identity"]["stock_instrs_equal"] is True and j["identity"]["stock_points"] >= 8 and j["identity"]["stock_cycles_max_dev_pct"] <= 0.1
    assert j["tok"]["kv8_token_rule_passed"] is True and j["tok"]["kv8_logits_failures"] == 0


# --------------------------------------------------------------------------------------------- vortex (SimX)
@pytest.fixture(scope="module")
def vx():
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun  # noqa: E402
    priv = PS / "build" / "tmp" / "epk_test"
    if priv.exists():
        shutil.rmtree(priv)
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], env=dict(os.environ, PS=str(PS)), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun("EP.K-test", results=priv)


def _run(vx, app, args, config=E1, tree="ps", perf=1, rss=1.0):
    row = vx.run(dict(tree=tree, perf=perf, app=app, args=args, config=config), timeout_s=7200, rss_gb=rss)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


def _assert_pass(row, log, fixture=False):
    assert row["status"] == "ok", (row["reason"], log[-2500:])
    c = {m.group(1): (float(m.group(4)), float(m.group(6)), int(m.group(7))) for m in PSCHECK_RE.finditer(log)}
    assert set(c) == {"launch0:vs_f64"}, c
    for k, (rel, tol, err) in c.items():
        assert err == 0 and rel <= tol, (k, rel, tol)
    bound, tol = map(float, HDR_RE.search(log).groups())
    assert tol == pytest.approx(max(PLAN_TOL, bound), rel=1e-3)
    assert "within_plan_tol=1" in log
    return float(KVQ_RE.search(log).group(1))


SHAPES = {"260K": (8, 4, 8), "15M": (6, 6, 48), "8-2-64": (8, 2, 64), "16-2-128": (16, 2, 128)}
V8 = {"gqa_kv8": "-v gqa_kv8", "dxa_kv8": "-v dxa_kv8 -p 2"}
V16 = {"gqa_kv16": "-v gqa_kv16", "dxa_kv16": "-v dxa_kv16 -p 2"}


def _shape_args(shape, B, T):
    hq, hkv, d = SHAPES[shape]
    return f"-B {B} -H {hq} -K {hkv} -D {d} -T {T}"


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(V8))
@pytest.mark.parametrize("T", [1, 17, 256])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_attn_kv8_grid(vx, shape, B, T, variant):
    row, log = _run(vx, "ps_decode_attn_gqa", f"{_shape_args(shape, B, T)} {V8[variant]}")
    err = _assert_pass(row, log)
    if T > 1:   # cache error vs the fp32-KV attention (information; sanity bound). At T = 1 the output is V_0 itself and the metric S_i = |v_d| makes the error of a small element
        assert err < 0.1   # relative to its own size (up to 1), so there is no bound there; the kernel's PASS against the dequantized reference is the test


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(V16))
@pytest.mark.parametrize("T", [1, 17, 256])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("shape", ["15M", "8-2-64"])
def test_attn_kv16_grid(vx, shape, B, T, variant):
    row, log = _run(vx, "ps_decode_attn_gqa", f"{_shape_args(shape, B, T)} {V16[variant]}", config=E1Z)
    err = _assert_pass(row, log)
    if T > 1:
        assert err < 5e-3                              # fp16 cache error (rel. 2^-11 per element)


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(V8) + list(V16))
@pytest.mark.parametrize("shape", list(SHAPES))
def test_attn_kv_all_negative(vx, shape, variant):
    v = {**V8, **V16}[variant]
    if variant in V16 and shape not in ("15M", "8-2-64"):
        pytest.skip("kv16 shapes: 15M and 8-2-64")
    row, log = _run(vx, "ps_decode_attn_gqa", f"{_shape_args(shape, 4, 256)} -d neg {v}", config=E1Z if variant in V16 else E1)
    _assert_pass(row, log)


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(V8) + list(V16))
def test_attn_kv_deep_negative_multitile(vx, variant):
    v = {**V8, **V16}[variant]
    row, log = _run(vx, "ps_decode_attn_gqa", f"-B 2 -H 8 -K 4 -D 8 -T 100 -d neg -q 100 {v}", config=E1Z if variant in V16 else E1)
    _assert_pass(row, log)
    assert "ntiles=1 " not in log


@pytest.mark.vortex
@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_attn_kv8_dxa_depth(vx, shape, depth):
    row, log = _run(vx, "ps_decode_attn_gqa", f"{_shape_args(shape, 4, 256)} -v dxa_kv8 -p {depth}")
    _assert_pass(row, log)
    assert f"depth={depth}" in log


FIXTURES = ["stories260K/L0_P17_attention", "stories15M/L5_P255_attention"]


@pytest.mark.vortex
@pytest.mark.parametrize("variant", list(V8) + list(V16))
@pytest.mark.parametrize("fx", FIXTURES)
def test_attn_kv_fixture(vx, fx, variant):
    d = FIX / fx
    if not (d / "meta.json").is_file():
        pytest.fail(f"fixture {d} missing")
    v = {**V8, **V16}[variant]
    row, log = _run(vx, "ps_decode_attn_gqa", f"-f {d} {v}", config=E1Z if variant in V16 else E1)
    err = _assert_pass(row, log)
    assert err < (0.1 if variant in V8 else 5e-3)


@pytest.mark.vortex
def test_kv16_refused_without_zfhmin(vx):
    row, log = _run(vx, "ps_decode_attn_gqa", f"{_shape_args('260K', 1, 17)} -v gqa_kv16")
    assert row["status"] == "failed" and "needs a device built with -DVX_CFG_EXT_ZFHMIN_ENABLE" in log


ROPE_CASES = [  # (label, args): synthetic shapes (row 0 at the largest position, others random) -- hs 8, 48 (stories15M), 64 (llama8b_s8); GQA and MHA
    ("hs8_gqa_b4", "-b 4 -d 64 -m 32 -H 8 -S 512 -N 5 -l 4"),
    ("hs48_mha_b2", "-b 2 -d 288 -m 288 -H 48 -S 256 -N 6 -l 3"),
    ("hs64_gqa_b4_cap6", "-b 4 -d 512 -m 128 -H 64 -S 300 -N 3 -l 2 -c 6"),
    ("hs48_b1_seed9", "-b 1 -d 288 -m 288 -H 48 -S 256 -N 6 -l 0 -s 9"),
]


@pytest.mark.vortex
@pytest.mark.parametrize("label,args", ROPE_CASES, ids=[c[0] for c in ROPE_CASES])
def test_rope_kv8_vs_host_and_numpy(vx, label, args):
    from model import ref_q80 as rq
    dump = PS / "build" / "tmp" / "epk_test" / f"rope_{label}.bin"
    dump.parent.mkdir(parents=True, exist_ok=True)
    row, log = _run(vx, "ps_rope", f"{args} -R table -K kv8 -J {dump}")
    assert row["status"] == "ok", (row["reason"], log[-2000:])
    c = {m.group(1): int(m.group(7)) for m in PSCHECK_RE.finditer(log)}
    for k in ("q_vs_f64", "k8_exact", "v8_exact", "ks_exact", "vs_exact", "kv8_untouched", "table_rope_table_vs_host"):
        assert c[k] == 0, (k, c)
    dg = {m.group(1): m.group(3) for m in PSDIGEST_RE.finditer(log)}
    # NumPy port from the dump: tab, pos, k, v, device K8, V8, Ks, Vs
    raw = dump.read_bytes()
    hs, B, kv_dim, Hkv, npos, _ = np.frombuffer(raw, "<u4", 6)
    off = 24
    tab = np.frombuffer(raw, "<f4", int(npos) * int(hs) * 1, off).reshape(int(npos), int(hs) // 2, 2); off += tab.nbytes
    pos = np.frombuffer(raw, "<i4", int(B), off); off += 4 * int(B)
    k = np.frombuffer(raw, "<f4", int(B) * int(kv_dim), off).reshape(int(B), int(kv_dim)); off += k.nbytes
    v = np.frombuffer(raw, "<f4", int(B) * int(kv_dim), off).reshape(int(B), int(kv_dim)); off += v.nbytes
    dk = np.frombuffer(raw, np.int8, int(B) * int(kv_dim), off); off += dk.size
    dv = np.frombuffer(raw, np.int8, int(B) * int(kv_dim), off); off += dv.size
    dks = np.frombuffer(raw, "<f4", int(B) * int(Hkv), off); off += dks.nbytes
    dvs = np.frombuffer(raw, "<f4", int(B) * int(Hkv), off); off += dvs.nbytes
    assert off == len(raw)
    ek, ev, eks, evs = [], [], [], []
    for b in range(int(B)):
        kr = np.concatenate([rq.rope_rotate_f32(k[b, h * int(hs):(h + 1) * int(hs)], tab[pos[b]]) for h in range(int(Hkv))])
        q1, s1 = rq.kv8_head_quantize(kr, int(hs))
        q2, s2 = rq.kv8_head_quantize(v[b], int(hs))
        ek.append(q1); ev.append(q2); eks.append(s1); evs.append(s2)
    ek, ev, eks, evs = np.concatenate(ek), np.concatenate(ev), np.concatenate(eks), np.concatenate(evs)
    assert np.array_equal(ek, dk) and np.array_equal(ev, dv)
    assert np.array_equal(eks.view(np.uint32), dks.view(np.uint32)) and np.array_equal(evs.view(np.uint32), dvs.view(np.uint32))
    assert dg["k8_exact"] == "%016x" % rq.fnv1a64(dk.tobytes()) and dg["v8_exact"] == "%016x" % rq.fnv1a64(dv.tobytes())


@pytest.mark.vortex
def test_ps_llama_kv8_refused_without_q80(vx):
    row, log = _run(vx, "ps_llama", f"-w {PS}/data/models/stories260K.bin -T {PS}/results/S8/golden_tokens_260K.txt -n 2 -Q kv8")
    assert row["status"] == "failed" and "-Q kv8 is the P8K8 track: it needs -q q80" in log
