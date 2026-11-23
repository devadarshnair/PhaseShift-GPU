"""Mixed-batch attention fixtures for the Vortex app ps_mixed_attn: one prefill chunk plus a batch of
decode queries, with the NumPy float32 outputs reference attention (model/ref_llama2c.py).

Usage (small job, < 0.3 GB RSS; run through scripts/lowload.sh):

    python -B -m model.make_mixed_fixtures --out DIR --chunk C --s0 S0 --ctx T1 T2 ... \
        (--model stories260K|stories15M [--layer L] | --shape H_q H_kv D [--seed N])

Format `ps-fixture-v1` (model/make_fixtures.py), op `attention_mixed`, tensors
  q_p    (c, dim)                 query of the chunk's tokens at positions s0 .. s0+c-1 (input)
  k_p    (s0+c, kv_dim)           K cache rows 0 .. s0+c-1 of the chunk's sequence (input; methodology §14.0 layout)
  v_p    (s0+c, kv_dim)           V cache rows (input)
  out_p  (c, dim)                 NumPy causal attention of every chunk token (output)
  q_d    (n_seq, dim)             one decode query per sequence (input)
  k_d    (n_seq, Tmax_d, kv_dim)  sequence b's K cache, rows 0 .. ctx[b]-1 valid, the rest 0 (input; Tmax_d = max ctx)
  v_d    (n_seq, Tmax_d, kv_dim)  V cache (input)
  out_d  (n_seq, dim)             NumPy decode attention of every sequence over its ctx[b] keys (output)
params: n_heads, n_kv_heads, head_size, dim, kv_dim, chunk (c), s0, ctx (list), source, seed / layer.

NumPy reference (why it is "the" reference): every output row is `Llama2cRef.attention`, the function the
token-by-token llama2.c reference uses for its attention, applied to one query and the keys that query may see
(prefill row i: keys 0 .. s0+i; decode sequence b: keys 0 .. ctx[b]-1). Causal prefill attention of token p is exactly
decode attention of query p over keys 0 .. p (methodology §14.4, "Prefill fixtures"), so one function serves both parts.
Sources:
  --model: inputs come prefill fixture of that model and layer (build/fixtures/<model>_prefill/,
           model/make_prefill_fixtures.py; real RoPE'd q and K/V of the llama2.c golden sequence). The chunk is rows s0 .. s0+c-1;
           decode sequence b is token ctx[b]-1 of the same sequence over its first ctx[b] cache rows (the cache rows of a
           sequence do not depend on later tokens). Every NumPy output row is then also checked to equal the prefill
           fixture's `out` row bit for bit (same function, same inputs).
  --shape: synthetic inputs, uniform in [-1, 1) float32 from numpy.random.default_rng(seed) in the order q_p, k_p, v_p,
           then per decode sequence q_d[b], k_d[b], v_d[b] (each sequence independent).
Deterministic: re-running rewrites identical bytes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from model.make_fixtures import FORMAT, _tensor_entry, _write_f32, load_fixture
from model.ref_llama2c import PS_ROOT, Llama2cRef


def numpy_attention(H_q: int, H_kv: int, D: int, q: np.ndarray, kc: np.ndarray, vc: np.ndarray) -> np.ndarray:
    """Llama2cRef.attention for one query q (dim) over kc/vc (T, kv_dim), without a model: the method only reads
    the head geometry, the GQA head map, sqrtf(head_size) and the softmax."""
    ns = SimpleNamespace(cfg=SimpleNamespace(n_heads=H_q, n_kv_heads=H_kv, head_size=D, dim=H_q * D),
                         _kv_head=np.arange(H_q) // (H_q // H_kv), _sqrt_hs=np.sqrt(np.float32(D)),
                         softmax=Llama2cRef.softmax)
    return Llama2cRef.attention(ns, q, kc, vc)[0]


def _prefill_fixture_dir(model: str, layer: int, base: Path | None = None) -> Path:
    base = Path(base) if base else PS_ROOT / "build" / "fixtures" / f"{model}_prefill"
    hits = sorted(base.glob(f"L{layer}_N*_attention_prefill"))
    if not hits:
        raise FileNotFoundError(f"no prefill fixture for {model} layer {layer} under {base} "
                                f"(run python -B -m model.make_prefill_fixtures --model {model})")
    return hits[0]


def make_mixed_fixture(out: Path, *, chunk: int, s0: int, ctx, model: str | None = None, layer: int = 0,
                       shape=None, seed: int = 0, name: str | None = None, prefill_dir: Path | None = None) -> dict:
    """Write one fixture to `out` (replaced if present); returns its meta. prefill_dir overrides the directory holding
    the model's prefill fixtures (default build/fixtures/<model>_prefill)."""
    ctx = [int(t) for t in ctx]
    c = int(chunk)
    if c < 0 or s0 < 0 or any(t < 1 for t in ctx) or (c == 0 and not ctx):
        raise ValueError("need chunk >= 0, s0 >= 0, every ctx >= 1, and a non-empty launch")
    if (model is None) == (shape is None):
        raise ValueError("give exactly one of model / shape")
    tmax = max(ctx) if ctx else 1
    n = len(ctx)
    if model is not None:
        pdir = _prefill_fixture_dir(model, layer, prefill_dir)
        meta_p, a = load_fixture(pdir)
        p = meta_p["params"]
        H_q, H_kv, D = p["n_heads"], p["n_kv_heads"], p["head_size"]
        N = a["q"].shape[0]
        if s0 + c > N or tmax > N:
            raise ValueError(f"s0 + chunk = {s0 + c} and max ctx = {tmax} must be <= {N} fixture tokens")
        q_all, k_all, v_all, o_all = a["q"], a["k_cache"], a["v_cache"], a["out"]
        q_p, k_p, v_p = q_all[s0:s0 + c], k_all[:s0 + c], v_all[:s0 + c]
        q_d = np.stack([q_all[t - 1] for t in ctx]) if n else np.zeros((0, H_q * D), np.float32)
        k_d = np.zeros((n, tmax, H_kv * D), np.float32)
        v_d = np.zeros_like(k_d)
        for b, t in enumerate(ctx):
            k_d[b, :t], v_d[b, :t] = k_all[:t], v_all[:t]
        src = {"source": model, "layer": int(meta_p["layer"]), "prefill_fixture": pdir.name}
    else:
        H_q, H_kv, D = (int(x) for x in shape)
        if H_q % H_kv:
            raise ValueError("H_kv must divide H_q")
        rng = np.random.default_rng(seed)
        u = lambda *sh: rng.uniform(-1.0, 1.0, sh).astype(np.float32)   # noqa: E731
        q_p, k_p, v_p = u(c, H_q * D), u(s0 + c, H_kv * D), u(s0 + c, H_kv * D)
        q_d = np.zeros((n, H_q * D), np.float32)
        k_d = np.zeros((n, tmax, H_kv * D), np.float32)
        v_d = np.zeros_like(k_d)
        for b, t in enumerate(ctx):
            q_d[b], k_d[b, :t], v_d[b, :t] = u(H_q * D), u(t, H_kv * D), u(t, H_kv * D)
        src = {"source": "synthetic", "seed": int(seed)}
        o_all = None
    out_p = np.zeros((c, H_q * D), np.float32)
    for i in range(c):
        T = s0 + i + 1
        out_p[i] = numpy_attention(H_q, H_kv, D, q_p[i], k_p[:T], v_p[:T])
        if o_all is not None and not np.array_equal(out_p[i], o_all[s0 + i]):
            raise AssertionError(f"NumPy attention of chunk row {i} differs from the prefill fixture's out row")
    out_d = np.zeros((n, H_q * D), np.float32)
    for b, t in enumerate(ctx):
        out_d[b] = numpy_attention(H_q, H_kv, D, q_d[b], k_d[b, :t], v_d[b, :t])
        if o_all is not None and not np.array_equal(out_d[b], o_all[t - 1]):
            raise AssertionError(f"NumPy attention of decode sequence {b} differs from the prefill fixture's out row")

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    arrays = {"q_p": (q_p, "in"), "k_p": (k_p, "in"), "v_p": (v_p, "in"), "out_p": (out_p, "out"),
              "q_d": (q_d, "in"), "k_d": (k_d, "in"), "v_d": (v_d, "in"), "out_d": (out_d, "out")}
    tensors = {k: _tensor_entry(f"{k}.f32", a_, role, _write_f32(out / f"{k}.f32", a_))
               for k, (a_, role) in arrays.items()}
    params = {"n_heads": H_q, "n_kv_heads": H_kv, "head_size": D, "dim": H_q * D, "kv_dim": H_kv * D,
              "chunk": c, "s0": int(s0), "ctx": ctx, **src}
    nm = name or f"c{c}_s{s0}_n{n}"
    meta = {"format": FORMAT, "model": model or "synthetic", "op": "attention_mixed", "layer": int(layer),
            "pos": int(s0 + c - 1), "name": nm, "params": params, "tensors": tensors}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--s0", type=int, default=0)
    ap.add_argument("--ctx", type=int, nargs="*", default=[])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--shape", type=int, nargs=3, metavar=("H_q", "H_kv", "D"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name")
    a = ap.parse_args(argv)
    m = make_mixed_fixture(a.out, chunk=a.chunk, s0=a.s0, ctx=a.ctx, model=a.model, layer=a.layer, shape=a.shape,
                           seed=a.seed, name=a.name)
    print(f"make_mixed_fixtures: {m['model']} {m['name']} H=({m['params']['n_heads']},{m['params']['n_kv_heads']},"
          f"{m['params']['head_size']}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
