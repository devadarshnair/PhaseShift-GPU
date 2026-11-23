"""Tests: pinned HF config.json files and serving traces.

KV-cache bytes per token (methodology E3, models note Q2) at bf16 (q = 2 bytes/element):
    GQA/MHA:  2 (K and V) * L * n_kv_heads * head_dim * q,   head_dim = config `head_dim` if given else hidden/heads
    MLA:      L * (kv_lora_rank + qk_rope_head_dim) * q      (DeepSeek-V2 caches the compressed latent + RoPE key)
Expected values are the earlier per-repo table. Missing files FAIL (setup-state test, no skips).
"""
import hashlib
import json
import re

import pytest

SHA40 = re.compile(r"^[0-9a-f]{40}$")
LOCK_COLS = ["repo", "sha", "sha256", "licence", "gated", "date", "path"]
KV_BF16 = {
    "unsloth/Llama-3.2-1B": 32_768,
    "unsloth/Meta-Llama-3.1-8B": 131_072,
    "mistralai/Mistral-7B-v0.1": 131_072,
    "Qwen/Qwen2.5-7B": 57_344,
    "Qwen/Qwen2.5-7B-Instruct-1M": 57_344,
    "Qwen/Qwen2.5-32B": 262_144,
    "unsloth/Meta-Llama-3.1-70B": 327_680,
    "allenai/OLMoE-1B-7B-0924": 131_072,
    "deepseek-ai/DeepSeek-V2-Lite": 31_104,   # 27 * (512 + 64) * 2
    "HuggingFaceTB/SmolLM2-135M": 23_040,     # 30 * 2 * 3 * 64 * 2
}
OPTIONAL = ["Qwen/Qwen3-30B-A3B", "mistralai/Mixtral-8x7B-v0.1"]
TRACES = {  # file -> (bytes, rows)  [Azure row counts are recorded, not planned: see test_azure_csv]
    "data/traces/AzureLLMInferenceTrace_code.csv": (320_117, None),
    "data/traces/AzureLLMInferenceTrace_conv.csv": (719_188, None),
    "data/traces/mooncake_conversation_trace.jsonl": (3_029_533, 12_031),
    "data/traces/mooncake_toolagent_trace.jsonl": (4_415_857, 23_608),
    "data/traces/mooncake_synthetic_trace.jsonl": (1_136_313, 3_993),
}


def lock_rows(ps_root):
    rows = {}
    for l in (ps_root / "configs/configs.lock").read_text().splitlines():
        if l.startswith("#") or not l.strip():
            continue
        c = l.split("\t")
        assert len(c) == len(LOCK_COLS), l
        r = dict(zip(LOCK_COLS, c))
        assert r["repo"] not in rows, f"duplicate lock row {r['repo']}"
        rows[r["repo"]] = r
    return rows


def load_config(ps_root, repo):
    return json.loads((ps_root / "configs" / (repo.replace("/", "__") + ".json")).read_text())


def kv_bytes_per_token(cfg, q=2):
    L = cfg["num_hidden_layers"]
    if "kv_lora_rank" in cfg:  # MLA
        return L * (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * q
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads") or cfg["num_attention_heads"]
    return 2 * L * n_kv * head_dim * q


def test_lock_rows_pinned_and_hashes_match(ps_root):
    rows = lock_rows(ps_root)
    for repo in list(KV_BF16) + OPTIONAL:
        assert repo in rows, repo
        r = rows[repo]
        assert SHA40.match(r["sha"]), (repo, r["sha"])  # a commit SHA, not a branch name
        assert r["licence"] and r["licence"] != "unknown", repo
        assert r["gated"] == "false", repo  # every planned repo is an ungated source
        p = ps_root / r["path"]
        assert r["path"] == "configs/" + repo.replace("/", "__") + ".json"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"], repo
        assert (p.stat().st_mode & 0o777) == 0o444, repo
    # Llama-family mirrors carry Meta's licence terms (recorded, not dropped)
    for repo in ("unsloth/Llama-3.2-1B", "unsloth/Meta-Llama-3.1-8B", "unsloth/Meta-Llama-3.1-70B"):
        assert rows[repo]["licence"].startswith("llama3"), rows[repo]["licence"]


@pytest.mark.parametrize("repo", list(KV_BF16) + OPTIONAL)
def test_config_has_opgen_fields(ps_root, repo):
    cfg = load_config(ps_root, repo)
    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"):
        assert isinstance(cfg.get(k), int) and cfg[k] > 0, (repo, k)
    assert "num_key_value_heads" in cfg or {"kv_lora_rank", "qk_rope_head_dim"} <= cfg.keys(), repo
    moe = {"moe_intermediate_size", "num_experts", "n_routed_experts", "num_local_experts"} & cfg.keys()
    assert "intermediate_size" in cfg or moe, repo


@pytest.mark.parametrize("repo,expected", list(KV_BF16.items()))
def test_kv_bytes_per_token(ps_root, repo, expected):
    assert kv_bytes_per_token(load_config(ps_root, repo)) == expected


def test_trace_manifest_rows(ps_root):
    lines = (ps_root / "data/MANIFEST.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    rows = {r["path"]: r for r in (dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip())}
    for path, (size, _) in TRACES.items():
        r = rows[path]
        assert int(r["bytes"]) == size and (ps_root / path).stat().st_size == size, path
        assert SHA40.match(r["revision"]) and f"/{r['revision']}/" in r["url"], path
        assert r["licence"] == ("CC-BY-4.0" if "Azure" in path else "Apache-2.0"), path
        assert "git-blob-sha1" in r["verify"], path


@pytest.mark.parametrize("name", ["code", "conv"])
def test_azure_csv(ps_root, name):
    lines = (ps_root / f"data/traces/AzureLLMInferenceTrace_{name}.csv").read_text().splitlines()
    assert lines[0].strip() == "TIMESTAMP,ContextTokens,GeneratedTokens"
    assert len(lines) > 1000
    for l in lines[1:]:
        ts, ctx, gen = l.split(",")
        assert ts and int(ctx) >= 0 and int(gen) >= 0


@pytest.mark.parametrize("name,rows", [("conversation", 12_031), ("toolagent", 23_608), ("synthetic", 3_993)])
def test_mooncake_jsonl(ps_root, name, rows):
    lines = (ps_root / f"data/traces/mooncake_{name}_trace.jsonl").read_text().splitlines()
    lines = [l for l in lines if l.strip()]
    assert len(lines) == rows
    for l in lines:
        d = json.loads(l)
        assert {"timestamp", "input_length", "output_length", "hash_ids"} <= d.keys()
        assert isinstance(d["hash_ids"], list) and d["input_length"] >= 0 and d["output_length"] >= 0
