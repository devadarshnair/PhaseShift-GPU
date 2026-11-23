"""cross-check of model/opgen.py against torch.utils.flop_counter.FlopCounterMode.

Builds `AutoModelForCausalLM.from_config` on the **meta** device (no weights, no real compute: FlopCounterMode
counts FLOPs from shapes) for unsloth/Llama-3.2-1B, runs one forward pass at (B=1, S=128), and compares the FLOPs
torch counts per nn.Linear module with opgen's linear ops (same convention: 2 FLOPs per multiply-accumulate).
Which aten ops are counted: FlopCounterMode registers formulas for mm, addmm, bmm, baddbmm, matmul-decomposed ops,
convolutions and the scaled-dot-product-attention kernels; elementwise ops (RMSNorm, RoPE, SiLU, residual, softmax)
count 0. Linear layers lower to aten.mm (no bias) / aten.addmm (bias). Attention is run with
attn_implementation="eager", whose QKᵀ and AV are aten.bmm/matmul over the full S×S rectangle (the causal mask is
added, not skipped), so it is compared with opgen at causal=False.

It also records N_params from the meta model (sum of p.numel()) for every pinned config, against opgen.n_params.
Run only through the low-load wrapper (importing torch + transformers exceeds 1 GB VA; measured ~0.4 GB RSS):

    $LL --rss 2 --label flopcheck -- python -m model.flopcount_xcheck --out results/<run>

Writes results/<run>/flopcounter_compare.csv and results/<run>/nparams_compare.csv; exit 1 if any linear-layer
FLOP total differs by more than 1 % (the TEST tolerance).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from model import opgen

TOL = 0.01


def meta_model(cfg_path: Path, attn: str = "eager"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(str(cfg_path.parent), local_files_only=True) if cfg_path.name == "config.json" \
        else AutoConfig.for_model(**json.loads(cfg_path.read_text()))
    cfg._attn_implementation = attn
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation=attn)
    return model.eval()


def flop_by_module(model, B: int, S: int):
    """Return {leaf module name: {aten op: flops}} from FlopCounterMode (leaf = deepest counted module)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    ids = torch.zeros((B, S), dtype=torch.long, device="meta")
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        # an explicit 2-D all-ones mask skips transformers' packed-sequence detection, which calls .item() (illegal on meta)
        model(input_ids=ids, attention_mask=torch.ones((B, S), dtype=torch.long, device="meta"), use_cache=False)
    counts = fc.get_flop_counts()
    return {k: {str(op): int(v) for op, v in d.items()} for k, d in counts.items()}, fc.get_total_flops()


def linear_names(model) -> dict[str, tuple[int, int]]:
    import torch.nn as nn
    return {n: (m.in_features, m.out_features) for n, m in model.named_modules() if isinstance(m, nn.Linear)}


OPGEN_NAME = {"q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj", "o_proj": "o_proj",
              "gate_proj": "gate_proj", "up_proj": "up_proj", "down_proj": "down_proj", "lm_head": "lm_head"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/P1.1")
    ap.add_argument("--config", default="configs/unsloth__Llama-3.2-1B.json")
    ap.add_argument("-B", type=int, default=1)
    ap.add_argument("-S", type=int, default=128)
    ap.add_argument("--nparams-configs", default="configs")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    import transformers
    cfg_path = Path(a.config)
    model = meta_model(cfg_path)
    counts, total = flop_by_module(model, a.B, a.S)
    lin = linear_names(model)
    # torch side: FLOPs of every nn.Linear module, aggregated by projection name (summed over layers)
    torch_by = defaultdict(int)
    ops_seen = set()
    attn_torch = 0
    leaf_keys = [k for k in counts if k != "Global"]
    for k in leaf_keys:
        # FlopCounterMode names modules with the root class prefix, e.g. "LlamaForCausalLM.model.layers.0.self_attn.q_proj"
        short = k.split(".", 1)[1] if "." in k else k
        if short in lin:
            proj = short.rsplit(".", 1)[-1]
            torch_by[proj] += sum(counts[k].values())
            ops_seen |= set(counts[k])
    # attention matmuls: FLOPs counted in self_attn modules that are not inside a Linear child
    for k in leaf_keys:
        short = k.split(".", 1)[1] if "." in k else k
        if re.fullmatch(r"model\.layers\.\d+\.self_attn", short):
            child = sum(sum(counts[c].values()) for c in leaf_keys
                        if c.startswith(k + ".") and (c.split(".", 1)[1] in lin))
            attn_torch += sum(counts[k].values()) - child
            ops_seen |= {op for op in counts[k]}
    # opgen side, same scenario: full prefill, HF computes logits for every token, eager attention = full rectangle
    m = opgen.from_hf_config(cfg_path)
    ops = opgen.gen_ops(m, opgen.prefill(a.B, a.S), q_w=2, q_a=2, q=2, causal=False, lm_head_all_tokens=True)
    og_by = defaultdict(float)
    for o in ops:
        if o.kind == "linear":
            og_by[o.name] += o.flops
    og_attn = sum(o.flops for o in ops if o.kind == "attn")
    rows, worst = [], 0.0
    for proj in sorted(set(torch_by) | set(og_by)):
        t, g = torch_by.get(proj, 0), og_by.get(proj, 0.0)
        rel = (g - t) / t if t else float("inf")
        worst = max(worst, abs(rel))
        rows.append(dict(item=proj, kind="linear", torch_flops=t, opgen_flops=int(g), rel_diff=f"{rel:.3e}"))
    t_lin, g_lin = sum(torch_by.values()), sum(og_by.values())
    rel_lin = (g_lin - t_lin) / t_lin
    rows.append(dict(item="ALL_LINEAR", kind="linear", torch_flops=t_lin, opgen_flops=int(g_lin), rel_diff=f"{rel_lin:.3e}"))
    rel_attn = (og_attn - attn_torch) / attn_torch if attn_torch else float("nan")
    rows.append(dict(item="ATTENTION_QK+AV (eager, full rectangle; informational)", kind="attn", torch_flops=attn_torch,
                     opgen_flops=int(og_attn), rel_diff=f"{rel_attn:.3e}"))
    rows.append(dict(item="TOTAL (torch Global; informational: opgen counts elementwise FLOPs torch does not)",
                     kind="total", torch_flops=total, opgen_flops=int(opgen.totals(ops)["flops"]), rel_diff=""))
    meta = dict(model=str(cfg_path), B=a.B, S=a.S, torch=torch.__version__, transformers=transformers.__version__,
                attn_implementation="eager", device="meta", aten_ops_in_linear_and_attn=" ".join(sorted(ops_seen)))
    with open(out / "flopcounter_compare.csv", "w", newline="") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        w = csv.DictWriter(f, fieldnames=["item", "kind", "torch_flops", "opgen_flops", "rel_diff"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(meta)
    # N_params check across the pinned configs (informational; failures to build are recorded, not fatal)
    prow = []
    for p in sorted(Path(a.nparams_configs).glob("*.json")):
        mm = None
        try:
            mm = meta_model(p)
            nt = sum(x.numel() for x in mm.parameters())
            err = ""
        except Exception as e:   # record and exclude ("if meta fails for some op, record and exclude it")
            nt, err = "", f"{type(e).__name__}: {str(e)[:120]}"
        ng = opgen.n_params(opgen.from_hf_config(p))
        rel = f"{(ng - nt) / nt:.3e}" if nt else ""
        prow.append(dict(config=p.name, torch_numel=nt, opgen_n_params=ng, rel_diff=rel, error=err))
        print(prow[-1])
        del mm
    with open(out / "nparams_compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)
    if abs(rel_lin) > TOL or worst > TOL:
        print(f"FAIL: linear FLOPs differ by {worst:.3%} (> {TOL:.0%})")
        return 1
    print(f"OK: linear FLOPs within {worst:.3e} relative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
