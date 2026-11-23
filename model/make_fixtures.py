"""Per-op fixtures for the Vortex kernel apps, generated from the NumPy llama2.c reference.

Usage (small job, < 0.5 GB RSS; run niced):

    python -B -m model.make_fixtures model --model stories15M [--layers 0 -1] [--positions 0 17 max] [--out DIR]
    python -B -m model.make_fixtures random-gemv --rows 512 --cols 512 --seed 0 [--out DIR]
    python -B -m model.make_fixtures batch --model stories15M [--layers 0 -1] [--batches 0 17 max 0,17,100,max] [--out DIR]

Default output: `build/fixtures/<model>/` (generated, not tracked). Re-running overwrites the fixtures with
identical bytes (the generator is deterministic).

**Format `ps-fixture-v1`** (what the kernel apps read; `kernels/common/ps_fixture.h` implements the reader):

* every tensor is one raw file `<name>.f32`: little-endian IEEE-754 float32, C (row-major) order, no header;
* every fixture directory has a small JSON header `meta.json`:
  `{"format": "ps-fixture-v1", "model", "op", "layer", "pos", "params": {...},
    "tensors": {"<name>": {"file": "<relative path>", "shape": [...], "dtype": "<f4", "role": "in"|"out"|"weight",
                           "sha256": "..."}}}`.
  Paths are relative to the fixture directory; weights are shared per model under `weights/` and referenced as
  `../weights/L<l>_<name>.f32` (matrices are (out, in) row-major, i.e. `out = W @ x`, as `run.c:matmul`).
* layout: `<out>/model.json` (config, checkpoint sha256, token sequence), `<out>/weights/`,
  `<out>/L<layer>_P<pos>_<op>/` per layer op, `<out>/P<pos>_<op>/` for the per-position ops
  (`embed`, `rmsnorm_final`, `classifier`).

**Batched fixtures** (`batch` subcommand, default output `build/fixtures/<model>_batch/`): for the
element-wise decode ops every tensor gets a leading batch axis B. One batch = a list of positions; row b of the batch is
the model state of the fixture token sequence at position `positions[b]` (B independent decode rows of one layer, as a
batched decode step sees them: same layer weights, per-row position). With the default batch `0,17,100,max` a B = 4
batch has four distinct rows (no duplicated row) and covers the plan's positions {0, 17,
max} inside one batch; the single-position batches `0`, `17`, `max` are the B = 1 cases. Directory per fixture:
`L<l>_B<B>_P<p0>-<p1>-..._<op>/` (layer ops) and `B<B>_P<...>_rmsnorm_final/`; weights under `weights/` as above;
`batch.json` is the index. `meta.json` has `"pos": -1`, `"batch": B` and `params.positions` (list). Ops and tensors
(shapes (B, n) unless noted):
  rmsnorm_att / rmsnorm_ffn / rmsnorm_final: x, weight (dim,), out
  rope_kv (RoPE fused with the KV-cache write, run.c forward lines 264-279; cache pointers 254-257): q_in (B, dim), k_in, v
      (B, kv_dim) -> q_out (B, dim), k_out (B, kv_dim); params give the device KV-cache layout
      `[layer][b][pos][kv_head][head_dim]` (methodology §14 prelude) and the element offset of row b's slot
  silu_gate: hb, hb2 -> out (B, hidden_dim);  residual_att / residual_ffn: x, branch -> out (B, dim)

Token sequence: the llama2.c golden sequence (prompt + greedy run.c output) teacher-forced for the positions it covers,
continued greedily by the reference beyond it (the golden has 200 steps; `max` = seq_len − 1 needs more). The
continuation ignores run.c's stop-at-BOS rule because a fixture only needs a valid model state at that position.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from model.ref_llama2c import MODELS, PS_ROOT, Llama2cRef, load_model, read_golden_tokens, sha256_file

FORMAT = "ps-fixture-v1"
# op -> (weight tensors it uses, mapping fixture name -> Weights attribute); input/output tensors come from the tap.
LAYER_OP_WEIGHTS = {
    "rmsnorm_att": {"weight": "rms_att_weight"},
    "matmul_q": {"w": "wq"}, "matmul_k": {"w": "wk"}, "matmul_v": {"w": "wv"}, "matmul_o": {"w": "wo"},
    "rmsnorm_ffn": {"weight": "rms_ffn_weight"},
    "matmul_w1": {"w": "w1"}, "matmul_w3": {"w": "w3"}, "matmul_w2": {"w": "w2"},
}
POS_OP_WEIGHTS = {"rmsnorm_final": {"weight": "rms_final_weight"}, "classifier": {"w": "wcls"}}
OUTPUT_NAMES = {"out", "q_out", "k_out", "scores", "probs"}   # everything else from the tap is an input


def _write_f32(path: Path, a: np.ndarray) -> str:
    data = np.ascontiguousarray(a, dtype="<f4").tobytes()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _tensor_entry(file: str, a: np.ndarray, role: str, sha: str) -> dict:
    return {"file": file, "shape": list(a.shape), "dtype": "<f4", "role": role, "sha256": sha}


def load_fixture(fdir) -> tuple[dict, dict[str, np.ndarray]]:
    """Read a fixture directory back: (meta, {name: float32 array}); checks each file's sha256 and size."""
    fdir = Path(fdir)
    meta = json.loads((fdir / "meta.json").read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{fdir}: format {meta.get('format')!r} != {FORMAT}")
    arrays = {}
    for name, t in meta["tensors"].items():
        raw = (fdir / t["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != t["sha256"]:
            raise ValueError(f"{fdir}/{t['file']}: sha256 mismatch")
        arrays[name] = np.frombuffer(raw, dtype="<f4").reshape(t["shape"]).astype(np.float32)
    return meta, arrays


def fixture_sequence(name: str, n_positions: int) -> list[int]:
    """Golden token sequence (teacher-forced), extended greedily by the reference up to `n_positions` tokens fed."""
    cfg, w, _ = load_model(name)
    seq, _forced = read_golden_tokens(PS_ROOT / MODELS[name]["golden_tokens"])
    seq = list(seq)
    if n_positions <= len(seq):
        return seq[:n_positions]
    ref = Llama2cRef(cfg, w)
    for pos in range(n_positions - 1):
        logits = ref.forward(seq[pos], pos)
        if pos + 1 >= len(seq):
            seq.append(int(np.argmax(logits)))
    return seq


def make_model_fixtures(name: str, layers, positions, out: Path) -> dict:
    cfg, w, _tok = load_model(name)
    L = cfg.n_layers
    layers = sorted({l % L for l in layers})
    positions = sorted({cfg.seq_len - 1 if p == "max" else int(p) for p in positions})
    if positions[-1] >= cfg.seq_len or positions[0] < 0:
        raise ValueError(f"positions {positions} outside [0, {cfg.seq_len})")
    seq = fixture_sequence(name, positions[-1] + 1)
    if out.exists():
        shutil.rmtree(out)
    (out / "weights").mkdir(parents=True)

    weight_files: dict[tuple[int, str], tuple[str, str, np.ndarray]] = {}

    def weight_ref(layer: int, attr: str):
        key = (layer, attr)
        if key not in weight_files:
            a = getattr(w, attr)[layer] if layer >= 0 else getattr(w, attr)
            fname = f"L{layer}_{attr}.f32" if layer >= 0 else f"{attr}.f32"
            weight_files[key] = (f"../weights/{fname}", _write_f32(out / "weights" / fname, a), a)
        return weight_files[key]

    written = []

    def tap(op, layer, pos, **arrays):
        if pos not in positions or (layer >= 0 and layer not in layers):
            return
        fdir = out / (f"L{layer}_P{pos}_{op}" if layer >= 0 else f"P{pos}_{op}")
        fdir.mkdir()
        tensors, params = {}, {"dim": cfg.dim, "hidden_dim": cfg.hidden_dim, "n_heads": cfg.n_heads,
                               "n_kv_heads": cfg.n_kv_heads, "head_size": cfg.head_size, "kv_dim": cfg.kv_dim,
                               "seq_len": cfg.seq_len, "vocab_size": cfg.vocab_size, "token": seq[pos]}
        wmap = LAYER_OP_WEIGHTS.get(op, {}) if layer >= 0 else POS_OP_WEIGHTS.get(op, {})
        for fname, attr in wmap.items():
            if fname in arrays:          # the tap passes the rmsnorm weight too; store it once under weights/
                arrays.pop(fname)
            rel, sha, a = weight_ref(layer, attr)
            tensors[fname] = _tensor_entry(rel, a, "weight", sha)
        for tname, a in arrays.items():
            if tname == "token":
                continue
            a = np.asarray(a, dtype=np.float32)
            # embed's `x` is the op's output (the embedding row); for every other op `x` is an input
            role = "out" if tname in OUTPUT_NAMES or (op == "embed" and tname == "x") else "in"
            tensors[tname] = _tensor_entry(f"{tname}.f32", a, role, _write_f32(fdir / f"{tname}.f32", a))
        if op == "kv_write":
            # KV cache layout of run.c generalised to batch: [layer][b][pos][kv_head][head_dim]
            params.update(kv_cache_offset_elems=pos * cfg.kv_dim, kv_layout="[layer][b][pos][kv_head][head_dim]")
        if op in ("rmsnorm_att", "rmsnorm_ffn", "rmsnorm_final"):
            params["eps"] = 1e-5
        if op == "rope":
            params["rope_theta"] = 10000.0
        meta = {"format": FORMAT, "model": name, "op": op, "layer": layer, "pos": pos, "params": params,
                "tensors": tensors}
        (fdir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        written.append(fdir.name)

    ref = Llama2cRef(cfg, w)
    for pos in range(positions[-1] + 1):
        ref.forward(seq[pos], pos, tap=tap)

    m = MODELS[name]
    model_meta = {"format": FORMAT, "model": name, "config": dataclasses.asdict(cfg),
                  "checkpoint": m["ckpt"], "checkpoint_sha256": sha256_file(PS_ROOT / m["ckpt"]),
                  "golden_tokens": m["golden_tokens"], "layers": layers, "positions": positions,
                  "token_sequence": seq, "n_golden_tokens": len(read_golden_tokens(PS_ROOT / m["golden_tokens"])[0]),
                  "fixtures": sorted(written)}
    (out / "model.json").write_text(json.dumps(model_meta, indent=1) + "\n")
    return model_meta


# ------------------------------------------------------------------------------------------------ batched
BATCH_LAYER_OPS = ("rmsnorm_att", "rope_kv", "residual_att", "rmsnorm_ffn", "silu_gate", "residual_ffn")
BATCH_POS_OPS = ("rmsnorm_final",)
_BATCH_TAP_OPS = {"rmsnorm_att", "rope", "kv_write", "residual_att", "rmsnorm_ffn", "silu_gate", "residual_ffn",
                  "rmsnorm_final"}
_BATCH_WEIGHTS = {"rmsnorm_att": "rms_att_weight", "rmsnorm_ffn": "rms_ffn_weight", "rmsnorm_final": "rms_final_weight"}
# op -> ordered (tensor name, role); for rope_kv the inputs come from the `rope` and `kv_write` taps
_BATCH_TENSORS = {
    "rmsnorm": (("x", "in"), ("out", "out")),
    "rope_kv": (("q_in", "in"), ("k_in", "in"), ("v", "in"), ("q_out", "out"), ("k_out", "out")),
    "silu_gate": (("hb", "in"), ("hb2", "in"), ("out", "out")),
    "residual": (("x", "in"), ("branch", "in"), ("out", "out")),
}


def parse_batches(batches, seq_len: int) -> list[list[int]]:
    """["0", "17", "0,17,100,max"] -> [[0], [17], [0, 17, 100, seq_len-1]]; positions must lie in [0, seq_len)."""
    out = []
    for item in batches:
        b = [seq_len - 1 if t == "max" else int(t) for t in str(item).split(",") if t != ""]
        if not b or min(b) < 0 or max(b) >= seq_len:
            raise ValueError(f"batch {item!r}: positions must be in [0, {seq_len})")
        out.append(b)
    return out


def make_batch_fixtures(name: str, layers, batches, out: Path) -> dict:
    """Batched element-wise fixtures: see the module docstring, "Batched fixtures"."""
    cfg, w, _tok = load_model(name)
    L = cfg.n_layers
    layers = sorted({l % L for l in layers})
    batches = parse_batches(batches, cfg.seq_len)
    need = sorted({p for b in batches for p in b})
    seq = fixture_sequence(name, need[-1] + 1)
    store: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}

    def tap(op, layer, pos, **arrays):
        if op in _BATCH_TAP_OPS and pos in need and (layer < 0 or layer in layers):
            store[(op, layer, pos)] = {k: np.array(v, dtype=np.float32, copy=True) for k, v in arrays.items()
                                       if k not in ("token", "weight")}

    ref = Llama2cRef(cfg, w)
    for pos in range(need[-1] + 1):
        ref.forward(seq[pos], pos, tap=tap)

    if out.exists():
        shutil.rmtree(out)
    (out / "weights").mkdir(parents=True)
    weight_files: dict[tuple[int, str], tuple[str, str, np.ndarray]] = {}

    def weight_ref(layer: int, attr: str):
        if (layer, attr) not in weight_files:
            a = getattr(w, attr)[layer] if layer >= 0 else getattr(w, attr)
            fname = f"L{layer}_{attr}.f32" if layer >= 0 else f"{attr}.f32"
            weight_files[(layer, attr)] = (f"../weights/{fname}", _write_f32(out / "weights" / fname, a), a)
        return weight_files[(layer, attr)]

    def rows(op: str, layer: int, pos: int) -> dict[str, np.ndarray]:
        if op == "rope_kv":
            r, kv = store[("rope", layer, pos)], store[("kv_write", layer, pos)]
            if not np.array_equal(r["k_out"], kv["k"]):          # run.c writes the rotated k into the cache
                raise AssertionError(f"rope k_out != kv_write k at L{layer} P{pos}")
            return {"q_in": r["q_in"], "k_in": r["k_in"], "v": kv["v"], "q_out": r["q_out"], "k_out": r["k_out"]}
        return store[(op, layer, pos)]

    written = []
    for positions in batches:
        B = len(positions)
        ptag = "-".join(map(str, positions))
        for layer, op in [(l, o) for l in layers for o in BATCH_LAYER_OPS] + [(-1, o) for o in BATCH_POS_OPS]:
            fname = f"L{layer}_B{B}_P{ptag}_{op}" if layer >= 0 else f"B{B}_P{ptag}_{op}"
            fdir = out / fname
            fdir.mkdir()
            kind = "rmsnorm" if op.startswith("rmsnorm") else "residual" if op.startswith("residual") else op
            per_row = [rows(op, layer, p) for p in positions]
            tensors = {}
            for tname, role in _BATCH_TENSORS[kind]:
                a = np.stack([r[tname] for r in per_row]).astype(np.float32)
                tensors[tname] = _tensor_entry(f"{tname}.f32", a, role, _write_f32(fdir / f"{tname}.f32", a))
            params = {"dim": cfg.dim, "hidden_dim": cfg.hidden_dim, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                      "n_kv_heads": cfg.n_kv_heads, "head_size": cfg.head_size, "kv_dim": cfg.kv_dim,
                      "seq_len": cfg.seq_len, "vocab_size": cfg.vocab_size, "batch": B, "positions": positions,
                      "tokens": [seq[p] for p in positions]}
            if kind == "rmsnorm":
                rel, sha, a = weight_ref(layer, _BATCH_WEIGHTS[op])
                tensors["weight"] = _tensor_entry(rel, a, "weight", sha)
                params["eps"] = 1e-5
            if op == "rope_kv":
                # device KV cache [layer][b][pos][kv_head][head_dim] with batch capacity B (methodology §14 prelude):
                # row b's K (and V) slot starts at element ((layer*B + b)*seq_len + pos_b)*kv_dim
                params.update(rope_theta=10000.0, kv_layout="[layer][b][pos][kv_head][head_dim]",
                              kv_cache_offset_elems=[((layer * B + b) * cfg.seq_len + p) * cfg.kv_dim
                                                     for b, p in enumerate(positions)])
            meta = {"format": FORMAT, "model": name, "op": op, "layer": layer, "pos": -1, "batch": B,
                    "params": params, "tensors": tensors}
            (fdir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
            written.append(fname)

    m = MODELS[name]
    index = {"format": FORMAT, "model": name, "config": dataclasses.asdict(cfg), "checkpoint": m["ckpt"],
             "checkpoint_sha256": sha256_file(PS_ROOT / m["ckpt"]), "golden_tokens": m["golden_tokens"],
             "layers": layers, "batches": batches, "token_sequence": seq, "fixtures": sorted(written)}
    (out / "batch.json").write_text(json.dumps(index, indent=1) + "\n")
    return index


def make_random_gemv(rows: int, cols: int, seed: int, out: Path) -> dict:
    """Random GEMV case out = W @ x (W uniform in [-1, 1)/sqrt(cols), x uniform in [-1, 1)), float32."""
    rng = np.random.default_rng(seed)
    W = (rng.uniform(-1, 1, size=(rows, cols)) / np.sqrt(cols)).astype(np.float32)
    x = rng.uniform(-1, 1, size=cols).astype(np.float32)
    y = W @ x
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tensors = {n: _tensor_entry(f"{n}.f32", a, r, _write_f32(out / f"{n}.f32", a))
               for n, a, r in (("w", W, "weight"), ("x", x, "in"), ("out", y, "out"))}
    meta = {"format": FORMAT, "model": "random", "op": "matmul", "layer": -1, "pos": -1,
            "params": {"rows": rows, "cols": cols, "seed": seed}, "tensors": tensors}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="PhaseShift kernel fixtures (ps-fixture-v1)")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("--model", required=True, choices=list(MODELS))
    m.add_argument("--layers", nargs="+", type=int, default=[0, -1])
    m.add_argument("--positions", nargs="+", default=["0", "17", "max"])
    m.add_argument("--out")
    r = sub.add_parser("random-gemv")
    r.add_argument("--rows", type=int, required=True)
    r.add_argument("--cols", type=int, required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out")
    bt = sub.add_parser("batch", help="batched element-wise fixtures")
    bt.add_argument("--model", required=True, choices=list(MODELS))
    bt.add_argument("--layers", nargs="+", type=int, default=[0, -1])
    bt.add_argument("--batches", nargs="+", default=["0", "17", "max", "0,17,100,max"],
                    help="each item = comma-separated positions of one batch (B = number of positions)")
    bt.add_argument("--out")
    a = p.parse_args(argv)
    if a.cmd == "batch":
        out = Path(a.out) if a.out else PS_ROOT / "build" / "fixtures" / f"{a.model}_batch"
        idx = make_batch_fixtures(a.model, a.layers, a.batches, out)
        print(f"{out}: {len(idx['fixtures'])} fixtures, layers {idx['layers']}, batches {idx['batches']}")
        return 0
    if a.cmd == "model":
        out = Path(a.out) if a.out else PS_ROOT / "build" / "fixtures" / a.model
        meta = make_model_fixtures(a.model, a.layers, a.positions, out)
        print(f"{out}: {len(meta['fixtures'])} fixtures, layers {meta['layers']}, positions {meta['positions']}")
    else:
        out = Path(a.out) if a.out else PS_ROOT / "build" / "fixtures" / "random" / f"gemv_{a.rows}x{a.cols}_s{a.seed}"
        make_random_gemv(a.rows, a.cols, a.seed, out)
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
