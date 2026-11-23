"""Prefill (causal) attention fixtures for the Vortex app ps_prefill_attn, from the NumPy llama2.c
reference (model/ref_llama2c.py).

Usage (small job, < 0.5 GB RSS; run niced / through lowload):

    python -B -m model.make_prefill_fixtures --model stories260K [--layers 0 -1] [--tokens max] [--out DIR]

Default output: `build/fixtures/<model>_prefill/` (generated, not tracked); deterministic (re-running rewrites
identical bytes). Format `ps-fixture-v1` (model/make_fixtures.py), one directory per layer
`L<l>_N<N>_attention_prefill/` with op `attention_prefill`, `pos` = N - 1 and tensors
  q        (N, dim)     RoPE'd query of every token 0 .. N-1 (input)
  k_cache  (N, kv_dim)  K cache rows 0 .. N-1 of that layer (input; run.c's key_cache layout, methodology §14.0)
  v_cache  (N, kv_dim)  V cache rows (input)
  out      (N, dim)     the reference's attention output of every token (output)
Why this is the prefill reference: the reference processes the sequence token by token (run.c's forward), and the
attention of token p reads cache rows 0 .. p only; the K/V written at token p do not depend on later tokens. So row
p of `out` is exactly causal (prefill) attention of query p over keys 0 .. p, and the k_cache / v_cache rows are
what a prefill pass over the same N tokens writes. Any prefix of n tokens is itself a valid fixture (the app's -S),
and rows s0 .. n-1 are a chunked-prefill chunk (-o s0).
Token sequence: `make_fixtures.fixture_sequence` (the llama2.c golden sequence, continued greedily up to N tokens).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from model.make_fixtures import FORMAT, _tensor_entry, _write_f32, fixture_sequence
from model.ref_llama2c import MODELS, PS_ROOT, Llama2cRef, load_model, sha256_file


def make_prefill_fixtures(name: str, layers, n_tokens, out: Path) -> dict:
    cfg, w, _tok = load_model(name)
    L = cfg.n_layers
    layers = sorted({int(l) % L for l in layers})
    N = cfg.seq_len if n_tokens == "max" else int(n_tokens)
    if not 1 <= N <= cfg.seq_len:
        raise ValueError(f"tokens {N} outside [1, {cfg.seq_len}]")
    seq = fixture_sequence(name, N)
    q = {l: np.zeros((N, cfg.dim), np.float32) for l in layers}
    o = {l: np.zeros((N, cfg.dim), np.float32) for l in layers}

    def tap(op, layer, pos, **arrays):
        if op == "attention" and layer in q:
            q[layer][pos] = arrays["q"]
            o[layer][pos] = arrays["out"]

    ref = Llama2cRef(cfg, w)
    for pos in range(N):
        ref.forward(seq[pos], pos, tap=tap)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    written = []
    for l in layers:
        fdir = out / f"L{l}_N{N}_attention_prefill"
        fdir.mkdir()
        arrays = {"q": (q[l], "in"), "k_cache": (ref.key_cache[l, :N], "in"), "v_cache": (ref.value_cache[l, :N], "in"),
                  "out": (o[l], "out")}
        tensors = {k: _tensor_entry(f"{k}.f32", a, role, _write_f32(fdir / f"{k}.f32", a))
                   for k, (a, role) in arrays.items()}
        params = {"dim": cfg.dim, "hidden_dim": cfg.hidden_dim, "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
                  "head_size": cfg.head_size, "kv_dim": cfg.kv_dim, "seq_len": cfg.seq_len,
                  "vocab_size": cfg.vocab_size, "n_tokens": N}
        meta = {"format": FORMAT, "model": name, "op": "attention_prefill", "layer": l, "pos": N - 1,
                "params": params, "tensors": tensors}
        (fdir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        written.append(fdir.name)
    m = MODELS[name]
    index = {"format": FORMAT, "model": name, "checkpoint": m["ckpt"],
             "checkpoint_sha256": sha256_file(PS_ROOT / m["ckpt"]), "layers": layers, "n_tokens": N,
             "token_sequence": seq, "fixtures": written}
    (out / "prefill.json").write_text(json.dumps(index, indent=1) + "\n")
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--layers", nargs="+", type=int, default=[0, -1])
    ap.add_argument("--tokens", default="max", help="N tokens (default: the model's seq_len)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    out = a.out or (PS_ROOT / "build" / "fixtures" / f"{a.model}_prefill")
    idx = make_prefill_fixtures(a.model, a.layers, a.tokens, out)
    print(f"make_prefill_fixtures: {a.model} N={idx['n_tokens']} -> {out}: {', '.join(idx['fixtures'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
