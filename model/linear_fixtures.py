"""Batched (M-row) linear-layer fixtures kernel app `ps_linear`.

Usage (small job, < 0.5 GB RSS, < 1 min; run niced, with -B so no bytecode is written):

    python -B -m model.linear_fixtures [--out DIR]          # default DIR = build/fixtures/linear

Writes `ps-fixture-v1` directories (format: model/make_fixtures.py docstring; reader: kernels/common/ps_fixture.h)
with tensors `w` (N, K) [weight, shared under <model>/weights/], `x` (M, K) and `out` (M, N); for M = 1 `x` is (K,) and
`out` is (N,), so the same directory also feeds `ps_gemv_f32` (decode GEMV, B = 1). Layout:

    <DIR>/<model>/L<layer>_M<M>_P<p0>_<op>/meta.json      rows = token positions p0 .. p0 + M - 1
    <DIR>/<model>/weights/L<layer>_<attr>.f32
    <DIR>/index.json                                       every fixture with model, op, layer, M, N, K, p0

Two shape families ("stories15M q/k/v/o/w1/w2/w3 shapes and Llama-8B-shaped scaled shapes"):

* **stories15M** (real weights and real activations). The NumPy llama2.c reference (model/ref_llama2c.py) runs
  the llama2.c golden token sequence (teacher-forced) through positions 0 .. 63; each fixture's `x` row r is the input of the
  op at position p0 + r and `out` row r is the reference's own fp32 result `W @ x` for that token (exactly what
  run.c's per-token matmul computes). A causal prefill of the first M tokens sees exactly these rows, so the fixtures
  are both batched-decode (M = B) and prefill-chunk test vectors.
* **llama8b_s8** -- "Llama-3.1-8B-shaped, scaled 1/8" (modeling choice). Real 8B weights are not available (and too
  big for SimX), so the shapes keep Llama-3.1-8B's aspect ratios (configs/unsloth__Meta-Llama-3.1-8B.json: d = 4096,
  32 query heads, 8 KV heads, head_dim 128, d_ff = 14336) at d' = d / 8 = 512: head_dim' = 64, H_q' = d'/64 = 8,
  H_kv' = H_q' · 8/32 = 2 (GQA group 4 kept), d_ff' = d' · 14336/4096 = 1792. These are the (H_q, H_kv, D) = (8, 2, 64)
  shapes uses for its scaled Llama case. q/o: 512 x 512, k/v: 128 x 512, w1/w3: 1792 x 512, w2: 512 x 1792
  (K = 1792 > 838, i.e. it exercises the K-dependent tolerance). Data are synthetic (deterministic seed): W ~ N(0, 1/K)
  (fan-in scaling, typical |w| ~ 0.02-0.04; a small tail is fp16-subnormal, |w| < 2^-14: up to 0.21 % of a
  llama8b_s8 weight tensor, 0.48 % for stories15M, and up to 0.78 % of an activation tensor -- covered by the
  absolute term of the TCU tolerance, docs/methodology.md §2 item 7(b)), X ~ N(0, 1) (unit RMS, like RMSNorm outputs); `out`
  = NumPy float32 `X @ W.T` (BLAS sgemm), the same role the reference plays for stories15M.

The generator is deterministic: re-running it rewrites identical bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from model.ref_llama2c import MODELS, PS_ROOT, Llama2cRef, load_model, read_golden_tokens

FORMAT = "ps-fixture-v1"
OPS = {"matmul_q": "wq", "matmul_k": "wk", "matmul_v": "wv", "matmul_o": "wo",
       "matmul_w1": "w1", "matmul_w3": "w3", "matmul_w2": "w2"}
# (M, p0) per fixture set; M = 1 at p0 = 17 (a non-trivial attention state), the rest a prefill from position 0
STORIES_SETS = [(1, 17), (3, 0), (5, 0), (16, 0), (64, 0)]
STORIES_LAYERS = (0, 5)            # first and last layer of stories15M (n_layers = 6)
LLAMA_SETS = [(1, 0), (3, 0), (16, 0), (64, 0)]
LLAMA_SCALE = 8


def _write_f32(path: Path, a: np.ndarray) -> str:
    data = np.ascontiguousarray(a, dtype="<f4").tobytes()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _entry(file: str, shape, role: str, sha: str) -> dict:
    return {"file": file, "shape": list(shape), "dtype": "<f4", "role": role, "sha256": sha}


def llama8b_s8_dims(cfg_path: Path = PS_ROOT / "configs" / "unsloth__Meta-Llama-3.1-8B.json",
                    scale: int = LLAMA_SCALE) -> dict:
    """Scaled Llama-3.1-8B dims (docstring): d/scale, head_dim/2, GQA ratio and d_ff/d kept."""
    c = json.loads(cfg_path.read_text())
    d, hq, hkv, dff = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["intermediate_size"]
    hd = c.get("head_dim") or d // hq
    d2, hd2 = d // scale, hd // 2
    hq2 = d2 // hd2
    hkv2 = hq2 * hkv // hq
    dff2 = d2 * dff // d
    assert hq2 * hd2 == d2 and hkv2 * hq == hq2 * hkv and dff2 * d == d2 * dff, "scaling not exact"
    return {"dim": d2, "n_heads": hq2, "n_kv_heads": hkv2, "head_size": hd2, "kv_dim": hkv2 * hd2,
            "hidden_dim": dff2, "source": str(cfg_path.relative_to(PS_ROOT)), "scale": scale}


def op_shape(op: str, dims: dict) -> tuple[int, int]:
    """(N, K) = (d_out, d_in) of a projection."""
    d, kv, ff = dims["dim"], dims["kv_dim"], dims["hidden_dim"]
    return {"matmul_q": (d, d), "matmul_k": (kv, d), "matmul_v": (kv, d), "matmul_o": (d, d),
            "matmul_w1": (ff, d), "matmul_w3": (ff, d), "matmul_w2": (d, ff)}[op]


class Writer:
    def __init__(self, root: Path):
        self.root = root
        self.index = []
        self.weights: dict[tuple[str, str], tuple[str, str]] = {}

    def weight(self, model: str, name: str, a: np.ndarray) -> tuple[str, str]:
        key = (model, name)
        if key not in self.weights:
            wdir = self.root / model / "weights"
            wdir.mkdir(parents=True, exist_ok=True)
            self.weights[key] = (f"../weights/{name}.f32", _write_f32(wdir / f"{name}.f32", a))
        return self.weights[key]

    def fixture(self, model: str, op: str, layer: int, p0: int, w: np.ndarray, wname: str, x: np.ndarray,
                out: np.ndarray, params: dict) -> None:
        M = x.shape[0]
        N, K = w.shape
        assert x.shape == (M, K) and out.shape == (M, N)
        name = f"L{layer}_M{M}_P{p0}_{op}"
        fdir = self.root / model / name
        fdir.mkdir(parents=True)
        rel, wsha = self.weight(model, wname, w)
        if M == 1:                       # 1-D x/out: also a ps_gemv_f32 fixture
            x, out = x[0], out[0]
        tensors = {"w": _entry(rel, w.shape, "weight", wsha),
                   "x": _entry("x.f32", x.shape, "in", _write_f32(fdir / "x.f32", x)),
                   "out": _entry("out.f32", out.shape, "out", _write_f32(fdir / "out.f32", out))}
        meta = {"format": FORMAT, "model": model, "op": op, "layer": layer, "pos": p0,
                "params": dict(params, M=M, N=N, K=K, rows="positions p0 .. p0+M-1"), "tensors": tensors}
        (fdir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        self.index.append({"dir": f"{model}/{name}", "model": model, "op": op, "layer": layer, "M": M, "N": N,
                           "K": K, "p0": p0})


def stories15m(wr: Writer) -> None:
    cfg, w, _tok = load_model("stories15M")
    n_pos = max(p0 + M for M, p0 in STORIES_SETS)
    seq = read_golden_tokens(PS_ROOT / MODELS["stories15M"]["golden_tokens"])[0]
    if len(seq) < n_pos:
        raise ValueError(f"golden sequence has {len(seq)} tokens < {n_pos}")
    xs = {(l, op): [] for l in STORIES_LAYERS for op in OPS}
    outs = {(l, op): [] for l in STORIES_LAYERS for op in OPS}

    def tap(op, layer, pos, **arrays):
        if op in OPS and layer in STORIES_LAYERS:
            xs[(layer, op)].append(np.asarray(arrays["x"], np.float32).copy())
            outs[(layer, op)].append(np.asarray(arrays["out"], np.float32).copy())

    ref = Llama2cRef(cfg, w)
    for pos in range(n_pos):
        ref.forward(seq[pos], pos, tap=tap)
    params = {"dim": cfg.dim, "hidden_dim": cfg.hidden_dim, "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
              "head_size": cfg.head_size, "kv_dim": cfg.kv_dim, "tokens": "S8 golden sequence, teacher-forced"}
    for l in STORIES_LAYERS:
        for op, attr in OPS.items():
            W = np.asarray(getattr(w, attr)[l], np.float32)
            X = np.stack(xs[(l, op)])
            Y = np.stack(outs[(l, op)])
            for M, p0 in STORIES_SETS:
                wr.fixture("stories15M", op, l, p0, W, f"L{l}_{attr}", X[p0:p0 + M], Y[p0:p0 + M], params)


def llama8b_s8(wr: Writer) -> None:
    dims = llama8b_s8_dims()
    params = dict(dims, data="synthetic: W ~ N(0, 1/K), X ~ N(0, 1), out = float32 X @ W.T")
    mmax = max(M for M, _ in LLAMA_SETS)
    for i, (op, attr) in enumerate(OPS.items()):
        N, K = op_shape(op, dims)
        rng = np.random.default_rng(1000 + i)
        W = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float32)
        X = rng.standard_normal((mmax, K)).astype(np.float32)
        for M, p0 in LLAMA_SETS:
            Xm = X[p0:p0 + M]
            Ym = (Xm @ W.T).astype(np.float32)          # float32 BLAS, as NumPy does for the model
            wr.fixture("llama8b_s8", op, 0, p0, W, f"L0_{attr}", Xm, Ym, params)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=PS_ROOT / "build" / "fixtures" / "linear")
    a = ap.parse_args(argv)
    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    wr = Writer(a.out)
    stories15m(wr)
    llama8b_s8(wr)
    (a.out / "index.json").write_text(json.dumps({"format": FORMAT, "generator": "model/linear_fixtures.py",
                                                  "fixtures": wr.index}, indent=1) + "\n")
    print(f"linear_fixtures: {len(wr.index)} fixtures in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
