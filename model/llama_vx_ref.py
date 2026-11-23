"""Reference logits for ps_llama's `-r` check.

Runs the NumPy float32 reference (`model/ref_llama2c.py`, mirrors llama2.c run.c [cand:D15]) teacher-forced over the
llama2.c golden token sequence and writes the logits of positions 0..n-1 as raw little-endian float32 [n][vocab] -- the format
`ps_llama -r` reads. Also writes a JSON sidecar (model, n, sha256 of checkpoint and output, per-position argmax, the
reference's own top-2 margin), so a run's evidence names exactly which reference it was checked against.

Usage (small job; run through scripts/lowload.sh like every Python job):
  python -B -m model.llama_vx_ref --model stories260K --positions 32 [--out build/fixtures/llama_vx]
Output: <out>/<model>_tf<n>.f32 and .json. Deterministic (re-running gives identical bytes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS))
from model import ref_llama2c as ref  # noqa: E402

DEFAULT_OUT = PS / "build" / "fixtures" / "llama_vx"


def make(model: str, n: int, out: Path) -> Path:
    cfg, w, _tok = ref.load_model(model)
    seq, forced = ref.read_golden_tokens(PS / ref.MODELS[model]["golden_tokens"])
    if not 1 <= n <= min(len(forced), cfg.seq_len):
        raise SystemExit(f"--positions must be in [1, {min(len(forced), cfg.seq_len)}]")
    r = ref.Llama2cRef(cfg, w, np.float32)
    logits = np.empty((n, cfg.vocab_size), np.float32)
    rows = []
    for p in range(n):
        lg = r.forward(seq[p], p).astype(np.float32)
        logits[p] = lg
        am, l1, l2 = ref.top2(lg)
        rows.append(dict(pos=p, token=seq[p], golden_next=seq[p + 1], forced=forced[p], argmax=am,
                         rel_margin=ref.rel_margin(l1, l2)))
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{model}_tf{n}.f32"
    f.write_bytes(logits.astype("<f4").tobytes())
    meta = dict(model=model, positions=n, vocab=cfg.vocab_size, dtype="float32 little endian [positions][vocab]",
                generator="model/llama_vx_ref.py (model/ref_llama2c.py Llama2cRef float32, teacher-forced "
                          "over results/S8 golden tokens)",
                checkpoint_sha256=ref.sha256_file(PS / ref.MODELS[model]["ckpt"]),
                golden_tokens=ref.MODELS[model]["golden_tokens"],
                sha256=hashlib.sha256(f.read_bytes()).hexdigest(), rows=rows)
    (out / f"{model}_tf{n}.json").write_text(json.dumps(meta, indent=1) + "\n")
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(ref.MODELS))
    ap.add_argument("--positions", type=int, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    f = make(a.model, a.positions, a.out)
    print(f"wrote {f} ({f.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
