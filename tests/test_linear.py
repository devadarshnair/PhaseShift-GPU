"""Tests: linear layers -- ps_linear (gemm_small FPU small-M, tcu_gemm TCU GEMM) and the fixtures.

Non-vortex (always run, cheap, no compile, no SimX):
  test_llama8b_s8_dims            the "Llama-3.1-8B-shaped, scaled 1/8" dims derived from the pinned HF config.
  test_linear_fixtures            model/linear_fixtures.py into tmp: every tensor's sha256/shape, `out` == the float32
                                  product within 2 gamma_K (S_i metric), stories15M rows == the reference's
                                  per-token taps, M = 1 fixtures are 1-D (ps_gemv_f32-compatible), deterministic.
  test_tcu_kernel_is_stock_copy   kernels/linear/kernel.cpp's tcu_gemm block is the stock
                                  sgemm_tcu_wg_dxa/kernel.cpp code token for token (comments/whitespace aside), with
                                  only the two documented renames.
  test_batch_file_up_to_date      experiments/<run>.yaml == results/<run>/make_batch.py output.
  test_tolerance_formula          max(1e-4, 2 gamma_K): 1e-4 up to K = 838, the bound above it.

Vortex-marked (PS_RUN_VORTEX=1; every SimX run through scripts/vxrun.py -> lowload.sh, serial; private results dir
build/tmp/linear_test, deleted at session start; lowload exit 75 -> skip "DEFERRED"):
  test_small_fixture[...]         ps_linear -v small at REG-CFG on stories15M (M = 16, w2: K = 768) and llama8b_s8
                                  (M = 3, w2: K = 1792 > 838) -> PASSED, all PSCHECK errors = 0, tol = max(1e-4, 2 gamma_K).
  test_small_grid_stride          -g 1: each thread loops over many output features.
  test_tcu_fixture[...]           ps_linear -v tcu at TIER-CFG on real stories15M weights (M = 64, N padded 288 -> 320)
                                  and llama8b_s8 w2 at M = 3 (M padded to 32): PASSED, fp16 self-test ok, padded
                                  outputs exactly 0, vs_f64_fp16in <= 2 gamma_K, vs_fixture <= 1e-2.
  test_tcu_matches_stock_instrs   ps_linear -v tcu (synthetic) and the stock sgemm_tcu_wg_dxa at the same padded
                                  shape execute the same number of instructions (same device code).
  test_stock_tcu_k128             errata 18: the stock app passes its own K-scaled ULP check at K = 128.
"""
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

PS = __import__("pathlib").Path(__file__).resolve().parents[1]
FIX = PS / "build" / "fixtures" / "linear"
U = 2.0 ** -24
PSCHECK_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) "
                        r"tol=(\S+) errors=(\d+)", re.M)


def tol_k(K):
    return max(1e-4, 2 * K * U / (1 - K * U))


# --------------------------------------------------------------------------------------------- cheap tests
def test_llama8b_s8_dims():
    from model.linear_fixtures import llama8b_s8_dims, op_shape
    d = llama8b_s8_dims()
    assert (d["dim"], d["n_heads"], d["n_kv_heads"], d["head_size"], d["kv_dim"], d["hidden_dim"]) == \
        (512, 8, 2, 64, 128, 1792)
    assert op_shape("matmul_k", d) == (128, 512) and op_shape("matmul_w2", d) == (512, 1792)


def test_tolerance_formula():
    assert tol_k(288) == 1e-4 and tol_k(838) == 1e-4
    assert tol_k(839) > 1e-4
    assert math.isclose(tol_k(1792), 2.1365e-4, rel_tol=1e-3)


def _load(fdir):
    meta = json.loads((fdir / "meta.json").read_text())
    arr = {}
    for name, t in meta["tensors"].items():
        raw = (fdir / t["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == t["sha256"], (fdir, name)
        arr[name] = np.frombuffer(raw, "<f4").reshape(t["shape"])
    return meta, arr


@pytest.fixture(scope="module")
def gen_fixtures(tmp_path_factory):
    from model.linear_fixtures import main
    out = tmp_path_factory.mktemp("linfx") / "linear"
    assert main(["--out", str(out)]) == 0
    return out


def test_linear_fixtures(gen_fixtures):
    idx = json.loads((gen_fixtures / "index.json").read_text())["fixtures"]
    assert len(idx) == 2 * 7 * 5 + 7 * 4
    for e in idx:
        meta, a = _load(gen_fixtures / e["dir"])
        M, N, K = e["M"], e["N"], e["K"]
        assert a["w"].shape == (N, K)
        if M == 1:
            assert a["x"].shape == (K,) and a["out"].shape == (N,)
        else:
            assert a["x"].shape == (M, K) and a["out"].shape == (M, N)
        X = a["x"].reshape(M, K).astype(np.float64)
        W = a["w"].astype(np.float64)
        ref = X @ W.T
        S = np.abs(X) @ np.abs(W).T
        rel = np.abs(a["out"].reshape(M, N) - ref) / S
        assert rel.max() <= 2 * K * U / (1 - K * U), (e["dir"], rel.max())


def test_linear_fixtures_match_reference_taps(gen_fixtures):
    """stories15M rows are the reference's own per-token matmul inputs/outputs (spot check vs a fresh run)."""
    from model.ref_llama2c import MODELS, PS_ROOT, Llama2cRef, load_model, read_golden_tokens
    cfg, w, _ = load_model("stories15M")
    seq = read_golden_tokens(PS_ROOT / MODELS["stories15M"]["golden_tokens"])[0]
    got = {}

    def tap(op, layer, pos, **arrays):
        if layer == 5 and op == "matmul_w2" and pos in (0, 4):
            got[pos] = (arrays["x"].copy(), arrays["out"].copy())

    ref = Llama2cRef(cfg, w)
    for pos in range(5):
        ref.forward(seq[pos], pos, tap=tap)
    _, a = _load(gen_fixtures / "stories15M" / "L5_M5_P0_matmul_w2")
    for pos in (0, 4):
        assert np.array_equal(a["x"][pos], got[pos][0]) and np.array_equal(a["out"][pos], got[pos][1])


def test_linear_fixtures_deterministic(gen_fixtures):
    """The generator is deterministic: the tmp copy equals build/fixtures/linear (if that exists) byte for byte."""
    if not (FIX / "index.json").is_file():
        pytest.skip("build/fixtures/linear not generated yet (python -B -m model.linear_fixtures)")
    assert (FIX / "index.json").read_text() == (gen_fixtures / "index.json").read_text()
    for e in json.loads((FIX / "index.json").read_text())["fixtures"]:
        assert (FIX / e["dir"] / "meta.json").read_text() == (gen_fixtures / e["dir"] / "meta.json").read_text()


def _code_tokens(text):
    text = re.sub(r"//[^\n]*", "", text)
    return text.split()


def test_tcu_kernel_is_stock_copy():
    mine = (PS / "kernels/linear/kernel.cpp").read_text()
    block = mine.split("// BEGIN copy of", 1)[1].split("// END copy of", 1)[0]
    block = block.split("\n", 1)[1]           # drop the BEGIN line itself
    stock = (PS / "third_party/vortex/tests/regression/sgemm_tcu_wg_dxa/kernel.cpp").read_text()
    stock = stock.replace('#include "common.h"\n', "").replace("#include <vx_spawn2.h>\n", "")
    stock = stock.replace("#include <vx_intrinsics.h>\n", "")       # already included at the top of kernel.cpp
    stock = stock.replace("kernel_main(kernel_arg_t*", "tcu_gemm(ps_tcu_arg_t*")
    assert _code_tokens(block) == _code_tokens(stock)


def test_batch_file_up_to_date():
    r = subprocess.run([sys.executable, "-B", str(PS / "results/P2.6/make_batch.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# --------------------------------------------------------------------------------------------- vortex (SimX)
@pytest.fixture(scope="module")
def vx():
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun  # noqa: E402
    priv = PS / "build" / "tmp" / "linear_test"
    if priv.exists():
        shutil.rmtree(priv)
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    if not (FIX / "index.json").is_file():
        pytest.fail("build/fixtures/linear missing: run `python -B -m model.linear_fixtures`")
    return vxrun.VxRun("P2.6-test", results=priv)


def _run(vx, spec):
    row = vx.run(dict({"tree": "base", "perf": 1}, **spec), timeout_s=3600)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


def _checks(log):
    return {m.group(1): {"max_rel_err": float(m.group(4)), "errors": int(m.group(7)), "tol": float(m.group(6))}
            for m in PSCHECK_RE.finditer(log)}


@pytest.mark.vortex
@pytest.mark.parametrize("fx,K", [("stories15M/L0_M16_P0_matmul_w2", 768), ("llama8b_s8/L0_M3_P0_matmul_w2", 1792)],
                         ids=["stories15M_M16_w2", "llama8b_s8_M3_w2"])
def test_small_fixture(vx, fx, K):
    row, log = _run(vx, {"app": "ps_linear", "args": f"-f {FIX / fx}", "config": "REG-CFG"})
    assert row["status"] == "ok", row["reason"]
    c = _checks(log)
    assert set(c) == {"fixture_vs_f64", "vs_f64", "vs_fixture"}
    m = re.search(r"^ps_linear: .* tol=(\S+) flops=", log, re.M)      # full-precision tolerance (PSCHECK prints %.1e)
    assert m and math.isclose(float(m.group(1)), tol_k(K), rel_tol=1e-3), m and m.group(1)
    for k, v in c.items():
        assert v["errors"] == 0 and v["max_rel_err"] <= tol_k(K) and math.isclose(v["tol"], tol_k(K), rel_tol=0.05)


@pytest.mark.vortex
def test_small_grid_stride(vx):
    row, log = _run(vx, {"app": "ps_linear", "args": f"-f {FIX / 'stories15M/L0_M5_P0_matmul_q'} -g 1",
                         "config": "REG-CFG"})
    assert row["status"] == "ok", row["reason"]
    assert "(grid overridden by -g)" in log and "grid=1 " in log
    assert all(v["errors"] == 0 for v in _checks(log).values())


@pytest.mark.vortex
@pytest.mark.parametrize("fx,K,pad", [("stories15M/L0_M64_P0_matmul_q", 288, "M=64->64 N=288->320 K=288->288"),
                                      ("llama8b_s8/L0_M3_P0_matmul_w2", 1792, "M=3->32 N=512->512 K=1792->1792")],
                         ids=["stories15M_M64_q", "llama8b_s8_M3_w2"])
def test_tcu_fixture(vx, fx, K, pad):
    row, log = _run(vx, {"app": "ps_linear", "args": f"-v tcu -f {FIX / fx}", "config": "TIER-CFG"})
    assert row["status"] == "ok", row["reason"]
    assert "fp16_selftest: ok" in log and f"tcu_padding: {pad}" in log
    assert re.search(r"^PSCHECK: pad_zero n=\d+ errors=0$", log, re.M)
    c = _checks(log)
    assert set(c) == {"fixture_vs_f64", "vs_f64_fp16in", "vs_fixture"}
    assert c["vs_f64_fp16in"]["errors"] == 0 and c["vs_f64_fp16in"]["max_rel_err"] <= tol_k(K)
    assert c["vs_fixture"]["errors"] == 0 and c["vs_fixture"]["tol"] == 1e-2


@pytest.mark.vortex
def test_tcu_matches_stock_instrs(vx):
    mine, _ = _run(vx, {"app": "ps_linear", "args": "-v tcu -m 64 -n 288 -k 288", "config": "TIER-CFG"})
    stock, log = _run(vx, {"app": "sgemm_tcu_wg_dxa", "args": "-m 64 -n 320 -k 288", "config": "TIER-CFG"})
    assert mine["status"] == "ok" and stock["status"] == "ok"
    assert mine["instrs"] == stock["instrs"]
    assert abs(mine["cycles"] - stock["cycles"]) <= 0.02 * stock["cycles"]


@pytest.mark.vortex
def test_stock_tcu_k128(vx):
    row, log = _run(vx, {"app": "sgemm_tcu_wg_dxa", "args": "-m 128 -n 128 -k 128", "config": "TIER-CFG"})
    assert row["status"] == "ok", row["reason"]
    m = re.search(r"reference drift: max=(\d+) ULP, bound=(\d+) ULP \(K=128\)", log)
    assert m and int(m.group(1)) <= int(m.group(2))
