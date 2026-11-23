"""Tests: Q8_0 GEMV (kernels/gemv_q80), activation quantizer, ps_llama -q q80, two-launch persist mode (/
methodology §18.5; the calls it §18.3).

Non-vortex (always run; no SimX; one tiny g++ compile):
  test_round_half_away_from_zero      runq.c `round` semantics of model/ref_q80.quantize_q80 (x.5 -> away from zero, not even)
  test_quantize_matches_scalar_runq   the vectorised quantizer == a scalar float32 transcription of runq.c `quantize`
  test_matmul_q80_matches_scalar_runq the vectorised matmul == a scalar transcription of runq.c `matmul` (bit for bit)
  test_exported_weights_requantize    re-quantizing the dequantized stories15M_q80 export reproduces its int8 values and scales
                                      exactly (the file really is runq.c-format Q8_0 with GS from its header)
  test_rng_vector_equals_scalar       the GF(2)-jump xorshift == the scalar ps::Rng port
  test_host_cpp_quantizer_matches_numpy  kernels/common/ps_q80.h (host C++, compiled here) is bit-identical to the NumPy port:
                                      quantized W / x digests and every output of matmul_row_q80, several (M, N, GS, seed)
  test_q80_file_bytes                 stories15M_q80.bin: GS = 32 in the header, 1.125 B/weight, 3.56x fewer weight bytes; llama8b_s8
                                      GS = 64 -> 1.0625 B/weight, 3.76x
  test_error_model                    tol_q80 = max(1e-4, 2 gamma_{G+2})
  test_golden_tokens_q80              results/<run>/golden_tokens_15M_q80.txt: 200 steps, forced prompt prefix, token chain; the
                                      model hash in its header is the file's; the text golden matches; it differs from the fp32
                                      golden (reported)
  test_runq_tokens_patch_and_build    the patch only adds one stderr print; build/llama2c/runq_tokens exists; clone untouched
  test_batch_file_up_to_date          experiments/<run>.yaml == results/<run>/make_batch.py output
  test_ep2_results                    results/<run>/rows.jsonl (when present): every row ok, Q8_0 GEMV bit-exact vs runq order, ...
  test_methodology_18_5               modeling choices documented

Vortex-marked (PS_RUN_VORTEX=1; each SimX run via scripts/vxrun.py -> lowload.sh, serial, private results dir
build/tmp/ep2_test, deleted at session start; lowload exit 75 -> skip "DEFERRED"):
  test_gemv_q80_shape[...]            ps_gemv_q80 at REG-CFG on every stories15M and llama8b_s8 shape, B in {1, 2, 4} (the two
                                      classifiers B in {1, 4}): PASSED, vs_f64 within tol, device output BIT-identical to the host
                                      runq-order result, and its digest equal to the NumPy port's (W / x quantization digests too)
  test_gemv_q80_lanes                 the warp-per-row variant, B = 1 and 3, GS 32 / 64 / 16: within the error model
  test_quantize_kernel                quantize_q80 on the device == host quantizer bit for bit: RNG data, an all-zero group, x.5 ties,
                                      and the real stories15M activations
  test_persist_two_launch             EP-CTRL-S0-L2big-P: ps_gemv_q80 and ps_linear (small and tcu) `-L 2 -P`: every launch passes with
                                      identical digests (the output is poisoned between launches), launch-2 L2 read misses <= 5 % of
                                      launch 1; negative control: the same GEMV without persist (EP-CTRL-S0-L2big, -L 2) re-misses
"""
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS))
from model import ref_q80 as q  # noqa: E402

F32 = np.float32
Q80 = PS / "build" / "models" / "stories15M_q80.bin"
FIX = PS / "build" / "fixtures" / "ep2"
RES = PS / "results" / "EP.2"
STEM_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) tol=(\S+) errors=(\d+)", re.M)
SHAPES = [  # (model, matrix, M, K) = results/<run>/make_batch.py SHAPES
    ("stories15M", "qkvo", 288, 288), ("stories15M", "w13", 768, 288), ("stories15M", "w2", 288, 768),
    ("stories15M", "cls", 32000, 288),
    ("llama8b_s8", "qo", 512, 512), ("llama8b_s8", "kv", 128, 512), ("llama8b_s8", "w13", 1792, 512),
    ("llama8b_s8", "w2", 512, 1792), ("llama8b_s8", "cls", 16032, 512)]


def gs_of(model):
    return 32 if model == "stories15M" else 64


# --------------------------------------------------------------------------------------------- scalar runq.c transcriptions
def scalar_quantize(x, gs):
    """runq.c quantize, one scalar at a time in float32 (C `round` on the double of the float quotient)."""
    x = [F32(v) for v in x]
    qs, ss = [], []
    for g in range(len(x) // gs):
        wmax = F32(0.0)
        for i in range(gs):
            v = abs(x[g * gs + i])
            if v > wmax:
                wmax = v
        scale = F32(wmax / F32(127.0))
        ss.append(scale)
        for i in range(gs):
            if scale == 0:
                qs.append(0)
                continue
            qv = F32(x[g * gs + i] / scale)
            d = float(qv)
            r = math.floor(d + 0.5) if d >= 0 else -math.floor(-d + 0.5)      # C round(): half away from zero
            qs.append(int(r))
    return np.array(qs, np.int8), np.array(ss, F32)


def scalar_matmul_row(wq, ws, xq, xs, gs):
    val = F32(0.0)
    for g in range(len(wq) // gs):
        iv = 0
        for k in range(gs):
            iv += int(xq[g * gs + k]) * int(wq[g * gs + k])
        val = F32(val + F32(F32(F32(iv) * ws[g]) * xs[g]))
    return val


def test_round_half_away_from_zero():
    x = np.zeros(32, F32)
    x[:10] = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.5, -3.5, 126.5, -126.5]
    x[31] = 127.0                                   # wmax = 127 -> scale = 127 / 127 = 1 exactly, so x / scale = x
    qv, s = q.quantize_q80(x, 32)
    assert s[0] == 1.0
    assert list(qv[:10]) == [1, 2, 3, -1, -2, -3, 4, -4, 127, -127]     # banker's rounding would give 0, 2, 2, ...
    assert qv[31] == 127


def test_quantize_matches_scalar_runq():
    rng = np.random.default_rng(3)
    for gs, n in ((32, 288), (64, 512), (16, 96)):
        x = (rng.standard_normal(n) * rng.choice([0.01, 1.0, 40.0])).astype(F32)
        x[:gs] = 0.0                                # an all-zero group (scale 0)
        qv, s = q.quantize_q80(x, gs)
        sq, ss = scalar_quantize(x, gs)
        assert np.array_equal(qv, sq) and np.array_equal(s.view(np.uint32), ss.view(np.uint32))


def test_matmul_q80_matches_scalar_runq():
    rng = np.random.default_rng(4)
    for gs, M, N in ((32, 7, 96), (64, 5, 256)):
        W = rng.standard_normal((M, N)).astype(F32)
        x = rng.standard_normal(N).astype(F32)
        wq, ws = q.quantize_q80(W, gs)
        xq, xs = q.quantize_q80(x, gs)
        y = q.matmul_q80(wq, ws, xq, xs, gs)
        for i in range(M):
            assert y[i].view(np.uint32) == scalar_matmul_row(wq[i], ws[i], xq, xs, gs).view(np.uint32)
        yb = q.matmul_q80(wq, ws, np.stack([xq, xq]), np.stack([xs, xs]), gs)       # B = 2 path
        assert np.array_equal(yb[0].view(np.uint32), y.view(np.uint32)) and np.array_equal(yb[1], yb[0])


@pytest.mark.skipif(not Q80.is_file(), reason="build/models/stories15M_q80.bin not exported (export command)")
def test_exported_weights_requantize():
    m = q.load_q80_checkpoint(Q80)
    assert m.gs == 32 and m.cfg.dim == 288 and m.cfg.shared_classifier
    for t in (m.tok, m.wq[3], m.w2[5]):
        rows = t.q[:64]
        d = q.dequantize_q80(rows, t.s[:64], m.gs)
        qq, ss = q.quantize_q80(d, m.gs)
        assert np.array_equal(qq, rows) and np.array_equal(ss.view(np.uint32), t.s[:64].view(np.uint32))


def test_rng_vector_equals_scalar():
    for n in (5000, 100003):
        for seed in (1, 7, 0):
            assert np.array_equal(q.rng_uniform(seed, n), q.rng_uniform_scalar(seed, n))


def test_error_model():
    assert q.tol_q80(9) == 1e-4 and q.tol_q80(28) == 1e-4
    m = 10 ** 5
    assert q.tol_q80(m) > 1e-4
    assert q.default_gs(288) == 32 and q.default_gs(512) == 64 and q.default_gs(1792) == 64 and q.default_gs(768) == 64


HOST_CPP = r'''
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "ps_common.h"
#include "ps_q80.h"
int main(int argc, char** argv) {
  unsigned M = atoi(argv[1]), N = atoi(argv[2]), GS = atoi(argv[3]), seed = atoi(argv[4]);
  ps::Rng rng(seed);
  std::vector<float> W((size_t)M * N), x(N);
  for (auto& v : W) v = rng.uniform();
  for (auto& v : x) v = rng.uniform();
  std::vector<int8_t> wq(W.size()), xq(N);
  std::vector<float> ws(W.size() / GS), xs(N / GS), y(M);
  ps::quantize_q80(W.data(), W.size(), GS, wq.data(), ws.data());
  ps::quantize_q80(x.data(), N, GS, xq.data(), xs.data());
  for (unsigned i = 0; i < M; ++i)
    y[i] = ps::matmul_row_q80(wq.data() + (size_t)i * N, ws.data() + (size_t)i * (N / GS), xq.data(), xs.data(), N, GS);
  std::printf("%016llx %016llx %016llx %016llx %016llx\n", (unsigned long long)ps::fnv1a64_bytes(wq.data(), wq.size()),
              (unsigned long long)ps::fnv1a64_bytes(ws.data(), ws.size() * 4), (unsigned long long)ps::fnv1a64_bytes(xq.data(), xq.size()),
              (unsigned long long)ps::fnv1a64_bytes(xs.data(), xs.size() * 4), (unsigned long long)ps::fnv1a64_f32(y.data(), y.size()));
}
'''


def py_digests(M, N, GS, B, seed):
    v = q.rng_uniform(seed, M * N + B * N)
    wq, ws = q.quantize_q80(v[:M * N].reshape(M, N), GS)
    xq, xs = q.quantize_q80(v[M * N:].reshape(B, N), GS)
    y = q.matmul_q80(wq, ws, xq, xs, GS)
    y = np.asarray(y, F32).reshape(B, M)
    f = lambda a: "%016x" % q.fnv1a64(np.ascontiguousarray(a).tobytes())
    return dict(Wq=f(wq), Ws=f(ws), xq=f(xq), xs=f(xs), y=f(y))


def test_host_cpp_quantizer_matches_numpy(tmp_path):
    src = tmp_path / "h.cpp"
    src.write_text(HOST_CPP)
    exe = tmp_path / "h"
    r = subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-I", str(PS / "kernels/common"), str(src), "-o", str(exe)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for M, N, GS, seed in ((9, 96, 32, 1), (17, 256, 64, 2), (5, 64, 16, 3), (12, 288, 32, 4)):
        out = subprocess.run([str(exe), str(M), str(N), str(GS), str(seed)], capture_output=True, text=True, check=True).stdout.split()
        d = py_digests(M, N, GS, 1, seed)
        assert out == [d["Wq"], d["Ws"], d["xq"], d["xs"], d["y"]], (M, N, GS, seed)


def test_q80_file_bytes():
    # Q8_0 = GS int8 + one fp32 scale per group -> (GS + 4) / GS bytes per weight
    assert (32 + 4) / 32 == 1.125 and (64 + 4) / 64 == 1.0625
    assert round(4 / 1.125, 2) == 3.56 and round(4 / 1.0625, 2) == 3.76
    if not Q80.is_file():
        pytest.skip("export missing")
    raw = Q80.read_bytes()
    assert struct.unpack_from("<Ii", raw, 0) == (0x616B3432, 2)
    assert struct.unpack_from("<i", raw, 37)[0] == 32 and raw[36] == 1
    m = q.load_q80_checkpoint(Q80)
    c = m.cfg
    n_w = c.n_layers * (2 * c.dim * c.dim + 2 * c.dim * c.kv_dim + 3 * c.dim * c.hidden_dim) + c.vocab_size * c.dim
    assert abs(n_w * 1.125 - 17.1e6) < 0.1e6                      # "~17.1 MB per token"


def test_golden_tokens_q80():
    g = RES / "golden_tokens_15M_q80.txt"
    if not g.is_file():
        pytest.skip("golden not recorded (scripts/golden_runq.sh)")
    lines = g.read_text().splitlines()
    assert lines[0].startswith("# runq (llama2.c Q8_0) golden token ids")
    steps = [tuple(map(int, l.split())) for l in lines if not l.startswith("#")]
    assert len(steps) == 200 and steps[0][1] == 1
    forced = [s[3] for s in steps]
    k = forced.index(0)
    assert k >= 1 and all(forced[:k]) and not any(forced[k:])
    for i, (pos, tok, nxt, _) in enumerate(steps):
        assert pos == i
        if i + 1 < len(steps):
            assert steps[i + 1][1] == nxt
    if Q80.is_file():
        assert hashlib.sha256(Q80.read_bytes()).hexdigest() in "\n".join(lines[:4])
    assert (RES / "golden_15M_q80.txt").read_bytes().startswith(b"Once upon a time")
    fp = [tuple(map(int, l.split())) for l in (PS / "results/S8/golden_tokens_15M.txt").read_text().splitlines() if not l.startswith("#")]
    diff = next((i for i, (a, b) in enumerate(zip(steps, fp)) if a[2] != b[2]), None)
    assert diff is not None and diff > 4            # Q8_0 text diverges from fp32 after the prompt (informational: reported, not pass/fail)


def test_runq_tokens_patch_and_build():
    patch = (PS / "patches/llama2c/runq_tokens.patch").read_text().splitlines()
    added = [l for l in patch if l.startswith("+") and not l.startswith("+++")]
    assert len(added) == 3 and sum("fprintf(stderr, \"PS_TOK" in l for l in added) == 1 and all("//" in l or "fprintf" in l for l in added)
    assert not [l for l in patch if l.startswith("-") and not l.startswith("---")]
    assert (PS / "build/llama2c/runq_tokens").is_file() and (PS / "build/llama2c/BUILDINFO.runq_tokens").is_file()
    assert not (PS / "third_party/llama2.c/runq_tokens").exists()
    # the clone is untouched: no file newer than the llama2.c build stamp (the clone's .git index mtime is the only known exception:
    # a `git status` without --no-optional-locks refreshed it on 2026-11-10 17:27 started; contents unchanged)
    newer = subprocess.run(["find", str(PS / "third_party/llama2.c"), "-newer", str(PS / "results/S8/llama2c.stamp"),
                            "-not", "-path", "*/.git", "-not", "-path", "*/.git/*"], capture_output=True, text=True, check=True).stdout.split()
    assert newer == []
    st = subprocess.run(["git", "--no-optional-locks", "-C", str(PS / "third_party/llama2.c"), "status", "--porcelain"],
                        capture_output=True, text=True, check=True).stdout
    assert st == ""


def test_batch_file_up_to_date():
    r = subprocess.run([sys.executable, "-B", str(RES / "make_batch.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_methodology_18_5():
    meth = (PS / "docs/methodology.md").read_text()
    assert "### 18.5" in meth
    sec = meth[meth.index("### 18.5"):]
    for s in ("contract(off)", "2 γ_{G+2}", "roundf", "BIT-identical", "runq.c", "persist", "-L 2 -P", "weight word", "dequant"):
        assert s in sec, s


def _rows(name="rows.jsonl"):
    p = RES / name
    if not p.is_file():
        pytest.skip("results/EP.2/rows.jsonl not produced yet")
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "ok":
                out[(r["app"], r["config"], r["args"], r["perf"])] = r
    return out


def _log(r):
    return (PS / r["log"]).read_text(errors="replace")


def test_ep2_results():
    rows = _rows()
    import importlib.util
    spec = importlib.util.spec_from_file_location("ep2_make_batch", RES / "make_batch.py")
    mb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mb)
    want = mb.runs()
    for w in want:
        r = rows.get((w["app"], w["config"], w["args"], w["perf"]))
        assert r is not None and r["passed"], (w["app"], w["args"])
        log = _log(r)
        if w["app"] == "ps_gemv_q80" and "-a quant" not in w["args"]:
            for m in re.finditer(r"^PSBITEXACT: \S+ n=\d+ mismatches=(\d+) variant=(\S+)$", log, re.M):
                if m.group(2) == "rows":
                    assert m.group(1) == "0", w["args"]
            assert re.search(r"^PSCHECK: launch0:vs_f64 .* errors=0 ", log, re.M)
    toks = [r for k, r in rows.items() if k[0] == "ps_llama" and "-q q80" in k[2]]
    assert toks
    summ = re.search(r"^PSSUM: mode=tf positions=8 compared=(\d+) agree=(\d+) near_ties=(\d+) .* failures=(\d+) logits_checked=1 "
                     r"logits_failures=(\d+)", _log(toks[0]), re.M)
    assert summ and int(summ.group(1)) == 4 and int(summ.group(4)) == 0 and int(summ.group(5)) == 0


# --------------------------------------------------------------------------------------------- vortex (SimX)
@pytest.fixture(scope="module")
def vx():
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun  # noqa: E402
    priv = PS / "build" / "tmp" / "ep2_test"
    shutil.rmtree(priv, ignore_errors=True)
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun("EP.2-test", results=priv)


def _run(vx, app, args, config="REG-CFG", tree="base", perf=1):
    row = vx.run(dict({"tree": tree, "perf": perf, "app": app, "args": args, "config": config}), timeout_s=3600)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


def _kv(log, key):
    m = re.search(rf"^PSQ80: {key} n=\d+ gs=\d+ fnv1a64=([0-9a-f]{{16}})$", log, re.M)
    return m.group(1) if m else None


_PYD = {}


def _pyd(M, N, GS, B, seed=1):
    k = (M, N, GS, B, seed)
    if k not in _PYD:
        _PYD[k] = py_digests(M, N, GS, B, seed)
    return _PYD[k]


CASES = [(m, mat, M, K, B) for (m, mat, M, K) in SHAPES for B in ((1, 4) if mat == "cls" else (1, 2, 4))]


@pytest.mark.vortex
@pytest.mark.parametrize("model,mat,M,K,B", CASES, ids=[f"{c[0]}_{c[1]}_B{c[4]}" for c in CASES])
def test_gemv_q80_shape(vx, model, mat, M, K, B):
    GS = gs_of(model)
    row, log = _run(vx, "ps_gemv_q80", f"-m {M} -n {K} -G {GS} -B {B} -v rows")
    assert row["status"] == "ok", row["reason"]
    assert f"ps_gemv_q80: synthetic:s1 M={M} N={K} GS={GS} G={K // GS} B={B} variant=rows" in log
    c = {m.group(1): (float(m.group(4)), int(m.group(7)), float(m.group(6))) for m in STEM_RE.finditer(log)}
    rel, errs, tol = c["launch0:vs_f64"]
    assert errs == 0 and rel <= tol and math.isclose(tol, q.tol_q80(K // GS), rel_tol=0.05)
    assert re.search(r"^PSBITEXACT: launch0:runq_exact n=\d+ mismatches=0 variant=rows$", log, re.M)
    d = _pyd(M, K, GS, B)
    assert [_kv(log, k) for k in ("Wq", "Ws", "xq", "xs")] == [d["Wq"], d["Ws"], d["xq"], d["xs"]]
    assert re.search(rf"^PSDIGEST: launch0:vs_f64 n={B * M} fnv1a64={d['y']}$", log, re.M)      # device y == NumPy port, bit for bit


@pytest.mark.vortex
@pytest.mark.parametrize("M,N,GS,B", [(50, 256, 64, 3), (33, 96, 16, 1), (70, 288, 32, 2), (128, 512, 64, 1)])
def test_gemv_q80_lanes(vx, M, N, GS, B):
    row, log = _run(vx, "ps_gemv_q80", f"-m {M} -n {N} -G {GS} -B {B} -v lanes")
    assert row["status"] == "ok", row["reason"]
    c = {m.group(1): (float(m.group(4)), int(m.group(7)), float(m.group(6))) for m in STEM_RE.finditer(log)}
    rel, errs, tol = c["launch0:vs_f64"]
    assert errs == 0 and rel <= tol
    assert re.search(r"^PSBITEXACT: launch0:runq_exact n=\d+ mismatches=\d+ variant=lanes$", log, re.M)   # informational for lanes


@pytest.mark.vortex
def test_quantize_kernel(vx, tmp_path):
    # x file: one all-zero group, x.5 ties with wmax = 127 (scale = 1), tiny and large magnitudes
    n, B = 96, 3
    x = np.zeros((B, n), F32)
    x[0, 32:64] = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.5, -3.5, 126.5, -126.5] + [0.25] * 21 + [127.0]
    x[1, :] = np.random.default_rng(9).standard_normal(n) * 1e-3
    x[2, :] = np.random.default_rng(10).standard_normal(n) * 1e3
    f = tmp_path / "xties.f32"
    f.write_bytes(x.astype("<f4").tobytes())
    for args in (f"-m 8 -n {n} -G 32 -B {B} -a dev -x {f}", "-m 16 -n 288 -G 32 -B 1 -a dev -s 5", "-m 8 -n 512 -G 64 -B 2 -a quant -s 6",
                 f"-m 8 -n 288 -G 32 -B 4 -a dev -x {FIX / 'x288_real.f32'}", f"-m 8 -n 768 -G 32 -B 1 -a quant -x {FIX / 'x768_real.f32'}"):
        row, log = _run(vx, "ps_gemv_q80", args)
        assert row["status"] == "ok", (args, row["reason"])
        m = re.search(r"^PSQUANT: launch0 nq=\d+ ns=\d+ q_mismatch=(\d+) s_mismatch=(\d+)$", log, re.M)
        assert m and m.group(1) == "0" and m.group(2) == "0", args
    # the zero group's scale is exactly 0 and its q are 0 on the device (host-compared above); the reference agrees
    qv, s = q.quantize_q80(x[0], 32)
    assert s[0] == 0 and not qv[:32].any() and list(qv[32:42]) == [1, 2, 3, -1, -2, -3, 4, -4, 127, -127]


def _miss(row):
    """L2 read misses of a perf-5 row (perf_kv key `core0.l2cache.miss_r`; one core)."""
    for k, v in row["perf_kv"].items():
        if k == "l2cache.miss_r" or (k.endswith(".l2cache.miss_r") and k.startswith("core")):
            return int(v)
    raise AssertionError(row["perf_kv"])


@pytest.mark.vortex
def test_persist_two_launch(vx):
    P, BIG = "EP-CTRL-S0-L2big-P", "EP-CTRL-S0-L2big"
    g1, l1 = _run(vx, "ps_gemv_q80", "-m 288 -n 288 -G 32 -v rows -L 1 -P", P, "ps", 5)
    g2, l2 = _run(vx, "ps_gemv_q80", "-m 288 -n 288 -G 32 -v rows -L 2 -P", P, "ps", 5)
    for r in (g1, g2):
        assert r["status"] == "ok", r["reason"]
    m1, m2 = _miss(g1), _miss(g2)
    assert (m2 - m1) <= 0.05 * m1, (m1, m2)                              # launch 2 reads the L2-resident weights
    d = re.findall(r"^PSDIGEST: launch(\d):vs_f64 n=\d+ fnv1a64=(\S+)$", l2, re.M)
    assert len(d) == 2 and d[0][1] == d[1][1]                            # NaN-poisoned output rewritten identically
    assert len(re.findall(r"^PSBITEXACT: launch\d:runq_exact n=\d+ mismatches=0", l2, re.M)) == 2
    # negative control: the same two launches on the 64 MiB L2 WITHOUT persist: launch 2 misses again (the L2 is invalidated)
    n1, _ = _run(vx, "ps_gemv_q80", "-m 288 -n 288 -G 32 -v rows -L 1", BIG, "ps", 5)
    n2, _ = _run(vx, "ps_gemv_q80", "-m 288 -n 288 -G 32 -v rows -L 2", BIG, "ps", 5)
    a1, a2 = _miss(n1), _miss(n2)
    assert (a2 - a1) >= 0.5 * a1, (a1, a2)
    # ps_linear: the fp32 small-M GEMM and the TCU/DXA GEMM, two launches each
    for v in ("small", "tcu"):
        p1, _ = _run(vx, "ps_linear", f"-v {v} -m 4 -n 288 -k 288 -L 1 -P", P, "ps", 5)
        p2, lg = _run(vx, "ps_linear", f"-v {v} -m 4 -n 288 -k 288 -L 2 -P", P, "ps", 5)
        assert p1["status"] == "ok" and p2["status"] == "ok", (v, p2["reason"])
        k1, k2 = _miss(p1), _miss(p2)
        assert (k2 - k1) <= 0.05 * k1, (v, k1, k2)
        digs = re.findall(r"^PSDIGEST: launch(\d):vs_f64(?:_fp16in)? n=\d+ fnv1a64=(\S+)$", lg, re.M)
        assert len(digs) == 2 and digs[0][1] == digs[1][1], (v, digs)
        assert len(re.findall(r"errors=0\b", lg)) >= (2 if v == "small" else 4)       # per launch: vs_f64 (+ pad_zero for tcu)
