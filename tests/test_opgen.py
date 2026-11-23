"""Tests: hand calculations on the llama2.c headers, E5 = 2g/q exactly on random shapes,
E3 KV bytes/token = the KV-size table, the FlopCounterMode cross-check (linear FLOPs within 1 %), plus E1/E2/E4 identities,
window/causal closed forms vs brute force, MoE routing, MLA, mixed-iteration additivity and the CSV writer."""
import csv
import inspect
import math
import random
from pathlib import Path

import pytest

from model import opgen
from model.opgen import (Seq, attention_ai_kv, decode, from_hf_config, from_llama2c_bin, gen_ops, kv_bytes_per_token,
                         kv_bytes_total, mixed, n_params, prefill, totals)

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
MODELS = ROOT / "data" / "models"

# KV-size table (results/S9/README.md): KV bytes/token at bf16
S9_KV = {"unsloth__Llama-3.2-1B": 32768, "unsloth__Meta-Llama-3.1-8B": 131072, "mistralai__Mistral-7B-v0.1": 131072,
         "Qwen__Qwen2.5-7B": 57344, "Qwen__Qwen2.5-7B-Instruct-1M": 57344, "Qwen__Qwen2.5-32B": 262144,
         "unsloth__Meta-Llama-3.1-70B": 327680, "allenai__OLMoE-1B-7B-0924": 131072,
         "deepseek-ai__DeepSeek-V2-Lite": 31104, "HuggingFaceTB__SmolLM2-135M": 23040,
         "Qwen__Qwen3-30B-A3B": 98304, "mistralai__Mixtral-8x7B-v0.1": 131072}


def shape(name):
    return from_hf_config(CFG / f"{name}.json")


def rand_shape(rng, **kw):
    Hkv = rng.choice([1, 2, 4, 8])
    g = rng.choice([1, 2, 4, 8])
    Dh = rng.choice([8, 16, 64, 128])
    d = Hkv * g * Dh
    args = dict(name="rand", d=d, L=rng.randint(1, 4), H_q=Hkv * g, H_kv=Hkv, D_h=Dh, d_ff=rng.randint(1, 64) * 8,
                vocab=rng.randint(10, 500), tie_embeddings=rng.random() < 0.5)
    args.update(kw)
    return opgen.ModelShape(**args)


# ------------------------------------------------------------------------------------------- hand calcs (llama2.c)
def test_stories260K_header_and_kv():
    m = from_llama2c_bin(MODELS / "stories260K.bin")
    assert (m.d, m.d_ff, m.L, m.H_q, m.H_kv, m.vocab, m.max_seq) == (64, 172, 5, 8, 4, 512, 512)
    assert m.tie_embeddings and m.D_h == 8
    assert kv_bytes_per_token(m, 4) == 1280                 # 2·5·4·8·4 B fp32
    # N_params + RoPE tables (seq_len·head_size floats) + 7-int header = the .bin size (1,056,540 B, methodology §2)
    assert n_params(m) * 4 + m.max_seq * m.D_h * 4 + 28 == 1_056_540 == (MODELS / "stories260K.bin").stat().st_size


def test_stories15M_hand_values():
    m = from_llama2c_bin(MODELS / "stories15M.bin")
    assert m.d_ff == 768                                    # read from the header, not inferred (8/3·288 ≠ 768)
    assert kv_bytes_per_token(m, 4) == 13824                # 2·6·6·48·4
    ops = gen_ops(m, decode(1, 10), 4, 4, 4)
    lm = [o for o in ops if o.name == "lm_head"]
    assert len(lm) == 1 and lm[0].flops / 2 == 9_216_000    # logits GEMV 288·32000 MACs
    layer_macs = sum(o.flops for o in ops if o.kind == "linear" and 0 <= o.layer < m.L) / 2
    assert layer_macs == 5_971_968 and abs(layer_macs - 6.0e6) / 6.0e6 < 0.01   # 6·(4·288² + 3·288·768) ≈ 6.0 M
    assert n_params(m) * 4 + m.max_seq * m.D_h * 4 + 28 == (MODELS / "stories15M.bin").stat().st_size


# ------------------------------------------------------------------------------------------- E3 / KV-size table
@pytest.mark.parametrize("name,kv", sorted(S9_KV.items()))
def test_e3_matches_s9_table(name, kv):
    assert kv_bytes_per_token(shape(name), 2) == kv


def test_e3_total_paging_and_window():
    m = shape("mistralai__Mistral-7B-v0.1")
    assert set(m.windows) == {4096}
    per_layer = 2 * m.H_kv * m.D_h * 2
    assert kv_bytes_total(m, [100, 5000], 2) == (100 + 4096) * m.L * per_layer
    assert kv_bytes_total(m, [100], 2, block=16) == 112 * m.L * per_layer
    l8 = shape("unsloth__Meta-Llama-3.1-8B")
    assert kv_bytes_total(l8, [7, 9], 2) == 16 * kv_bytes_per_token(l8, 2)
    assert kv_bytes_total(l8, [7, 9], 2, block=8) == 24 * kv_bytes_per_token(l8, 2)


def test_window_rules():
    assert set(shape("Qwen__Qwen2.5-7B").windows) == {0}               # use_sliding_window: false
    assert set(shape("mistralai__Mixtral-8x7B-v0.1").windows) == {0}   # sliding_window: null
    base = dict(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, intermediate_size=128, vocab_size=10)
    q2 = from_hf_config(dict(base, sliding_window=32, use_sliding_window=True, max_window_layers=2))
    assert q2.windows == (0, 0, 32, 32)                                  # Qwen2Config.__post_init__ rule
    lt = from_hf_config(dict(base, sliding_window=8, layer_types=["full_attention", "sliding_attention"] * 2))
    assert lt.windows == (0, 8, 0, 8)


# ------------------------------------------------------------------------------------------- E5 exact, E1/E2/E4
def test_e5_decode_attention_ai_is_2g_over_q_random_shapes():
    rng = random.Random(5)
    for _ in range(200):
        m = rand_shape(rng)
        q = rng.choice([0.5, 1, 2, 4])
        B, T = rng.randint(1, 32), rng.randint(1, 5000)
        ops = gen_ops(m, decode(B, T), q_w=2, q_a=2, q=q, fused_attention=rng.random() < 0.5)
        assert attention_ai_kv(ops) == pytest.approx(2 * m.g / q, rel=1e-12)
        assert attention_ai_kv([o for o in ops if o.layer == 0]) == pytest.approx(2 * m.g / q, rel=1e-12)


def test_e1_e2_linear_ops():
    rng = random.Random(1)
    for _ in range(100):
        n, din, dout = rng.randint(1, 4096), rng.randint(1, 8192), rng.randint(1, 8192)
        qw, qa = rng.choice([0.5, 1, 2, 4]), rng.choice([1, 2, 4])
        o = opgen.linear(0, "x", n, din, dout, qw, qa)
        assert o.flops == 2 * n * din * dout                                           # E1
        e2 = 2 * n / (qw + qa * n * (1 / din + 1 / dout))                              # E2 exact form
        assert o.ai == pytest.approx(e2, rel=1e-12)


def test_e4_attention_flops():
    rng = random.Random(2)
    for _ in range(50):
        m = rand_shape(rng)
        S, T = rng.randint(1, 300), rng.randint(1, 3000)
        att = lambda ops: sum(o.flops for o in ops if o.kind == "attn" and o.layer == 0)
        assert att(gen_ops(m, prefill(1, S), causal=False)) == 4 * S * S * m.d          # E4 prefill, full
        assert att(gen_ops(m, prefill(1, S), causal=True)) == 4 * m.d * S * (S + 1) // 2  # masked half skipped
        assert att(gen_ops(m, decode(1, T))) == 4 * T * m.d                              # E4 decode
        assert att(gen_ops(m, decode(3, T))) == 3 * 4 * T * m.d


def test_keys_closed_forms_vs_brute_force():
    rng = random.Random(3)
    for _ in range(2000):
        n, ctx, W = rng.randint(1, 60), rng.randint(0, 80), rng.choice([0, 1, 2, 5, 17, 64, 200])
        brute = sum(min(ctx + j + 1, W) if W else ctx + j + 1 for j in range(n))
        assert opgen._keys_sum(n, ctx, W) == brute
        touched = len({k for j in range(n) for k in range(ctx + j + 1) if not W or k > ctx + j - W})
        assert opgen._keys_touched(n, ctx, W) == touched


def test_window_caps_decode_kv_reads():
    m = shape("mistralai__Mistral-7B-v0.1")
    kv = lambda T: totals(gen_ops(m, decode(1, T)))["kv_rd"]
    assert kv(10_000) == kv(4096) == 4096 * kv_bytes_per_token(m, 2)
    assert kv(100) == 100 * kv_bytes_per_token(m, 2)


def test_chunked_prefill_conserves_work():
    """Splitting a prefill into chunks (Sarathi) conserves causal attention pairs, KV writes and linear FLOPs."""
    m = shape("unsloth__Llama-3.2-1B")
    full = gen_ops(m, prefill(1, 1000), layers=[0])
    parts = []
    for ctx in range(0, 1000, 256):
        parts += gen_ops(m, prefill(1, min(256, 1000 - ctx), ctx, emits=ctx + 256 >= 1000), layers=[0])
    for key in ("flops", "kv_wr", "w_rd"):
        f = sum(getattr(o, key) for o in full if o.kind in ("attn", "linear", "elementwise") and o.layer == 0)
        p = sum(getattr(o, key) for o in parts if o.kind in ("attn", "linear", "elementwise") and o.layer == 0)
        if key == "w_rd":
            assert p == 4 * f            # weights are re-read by every chunk
        else:
            assert p == pytest.approx(f, rel=1e-12), key
    assert sum(o.flops for o in parts if o.name == "lm_head") == sum(o.flops for o in full if o.name == "lm_head")


# ------------------------------------------------------------------------------------------- weights vs N_params
@pytest.mark.parametrize("name", ["unsloth__Llama-3.2-1B", "unsloth__Meta-Llama-3.1-8B", "Qwen__Qwen2.5-7B",
                                  "HuggingFaceTB__SmolLM2-135M", "mistralai__Mistral-7B-v0.1"])
def test_weight_bytes_per_step_equal_n_params(name):
    """Dense models: one iteration reads every weight once, except the embedding table, of which only N_tok rows
    are gathered (and which the tied LM head reads in full)."""
    m = shape(name)
    for seqs in (decode(4, 300), prefill(2, 50), mixed([(64, 0)], [(3, 100)])):
        n_tok = sum(s.n_new * s.count for s in seqs)
        w = totals(gen_ops(m, seqs, q_w=2))["w_rd"]
        expect = n_params(m) + n_tok * m.d - (0 if m.tie_embeddings else m.vocab * m.d)
        assert w == 2 * expect


def test_mixed_iteration_is_concatenation():
    m = shape("unsloth__Meta-Llama-3.1-8B")
    mix = gen_ops(m, mixed([(128, 0), (64, 512)], [(5, 1000)]))
    a = gen_ops(m, [Seq("prefill", 128, 0)])
    b = gen_ops(m, [Seq("prefill", 64, 512)])
    c = gen_ops(m, decode(5, 1000))
    attn = lambda ops: [(o.layer, o.name, o.flops, o.kv_rd) for o in ops if o.kind == "attn" or "softmax" in o.name]
    assert sorted(attn(mix)) == sorted(attn(a) + attn(b) + attn(c))
    lin = lambda ops, k: sum(getattr(o, k) for o in ops if o.kind == "linear" and o.name != "lm_head")
    assert lin(mix, "flops") == lin(a, "flops") + lin(b, "flops") + lin(c, "flops")      # FLOPs add
    assert lin(mix, "w_rd") == lin(a, "w_rd")                                           # weights read once
    assert [o.M for o in mix if o.name == "lm_head"] == [7]                               # 2 chunks + 5 decodes


# ------------------------------------------------------------------------------------------- MoE
def test_moe_routing():
    for E, k in ((8, 2), (64, 8), (128, 8)):
        assert opgen.moe_active_experts(1, E, k) == pytest.approx(k)
        assert opgen.moe_active_experts(1e6, E, k) == pytest.approx(E)
        for s in (0.5, 1.0, 2.0):
            pi = opgen.moe_inclusion_probs(E, k, "zipf", s)
            assert sum(pi) == pytest.approx(k) and max(pi) <= 1 + 1e-12 and min(pi) > 0
            assert opgen.moe_active_experts(1, E, k, "zipf", s) == pytest.approx(k)
            # skewed routing touches fewer distinct experts than uniform
            assert opgen.moe_active_experts(16, E, k, "zipf", s) < opgen.moe_active_experts(16, E, k)
    # uniform active-experts formula vs Monte Carlo of k-subsets
    rng = random.Random(4)
    E, k, n = 16, 2, 5
    mc = sum(len({e for _ in range(n) for e in rng.sample(range(E), k)}) for _ in range(20000)) / 20000
    assert opgen.moe_active_experts(n, E, k) == pytest.approx(mc, rel=0.01)


def test_moe_flops_independent_of_routing_and_bytes():
    m = shape("mistralai__Mixtral-8x7B-v0.1")
    u = gen_ops(m, decode(4, 100), layers=[0])
    z = gen_ops(m, decode(4, 100), layers=[0], routing="zipf", zipf_s=1.5)
    f = lambda ops: sum(o.flops for o in ops if o.name.startswith("expert"))
    assert f(u) == f(z) == 4 * 2 * (3 * 2 * m.d * m.d_ff_expert) + 4 * 2 * 4 * m.d_ff_expert + 2 * 4 * 2 * m.d
    one = gen_ops(m, decode(1, 100), layers=[0])
    w = sum(o.w_rd for o in one if o.name.startswith("expert"))
    assert w == pytest.approx(2 * 3 * m.d * m.d_ff_expert * 2)       # B=1: exactly top-2 experts' weights
    ds = shape("deepseek-ai__DeepSeek-V2-Lite")
    assert ds.moe_layers[0] is False and all(ds.moe_layers[1:]) and ds.d_ff_shared == 2 * 1408
    assert sum(shape("Qwen__Qwen3-30B-A3B").moe_layers) == 48


# ------------------------------------------------------------------------------------------- MLA
def test_mla_kv_and_weights():
    m = shape("deepseek-ai__DeepSeek-V2-Lite")
    assert m.is_mla and kv_bytes_per_token(m, 2) == 27 * (512 + 64) * 2 == 31104
    ops = gen_ops(m, decode(2, 1000), layers=[3])
    kv_rd = sum(o.kv_rd for o in ops if o.layer == 3)
    kv_wr = sum(o.kv_wr for o in ops if o.layer == 3)
    assert kv_rd == 2 * 1000 * (512 + 64) * 2 and kv_wr == 2 * (512 + 64) * 2
    # attention-layer weights = q_proj + kv_a + kv_a_norm + kv_b (= W_UK + W_UV) + o_proj
    H, r, rho, nope, v, d = 16, 512, 64, 128, 128, 2048
    w = sum(o.w_rd for o in ops if o.name in ("q_proj", "kv_a_proj", "kv_norm_rope_kv_write", "q_absorb_uk",
                                                "v_up_uv", "o_proj"))
    assert w == 2 * (d * H * (nope + rho) + d * (r + rho) + r + r * H * (nope + v) + H * v * d)
    att = sum(o.flops for o in ops if o.kind == "attn")
    assert att == 2 * (2 * H * (r + rho) * 1000 + 2 * H * r * 1000)


# ------------------------------------------------------------------------------------------- FlopCounterMode
def _read_compare():
    p = ROOT / "results" / "P1.1" / "flopcounter_compare.csv"
    assert p.exists(), "run model/flopcount_xcheck.py through $LL first (see results/P1.1/README.md)"
    lines = [l for l in p.read_text().splitlines() if not l.startswith("#")]
    return {r["item"]: r for r in csv.DictReader(lines)}, p.read_text()


def test_flopcounter_crosscheck_linear_within_1pct():
    """torch FlopCounterMode numbers (meta-device Llama-3.2-1B, B=1, S=128, recorded by model/flopcount_xcheck.py
    through lowload) vs opgen recomputed now: each projection and the linear total within 1 %."""
    rows, text = _read_compare()
    assert "# B: 1" in text and "# S: 128" in text and "Llama-3.2-1B" in text and "device: meta" in text
    m = shape("unsloth__Llama-3.2-1B")
    ops = gen_ops(m, prefill(1, 128), causal=False, lm_head_all_tokens=True)
    by = {}
    for o in ops:
        if o.kind == "linear":
            by[o.name] = by.get(o.name, 0) + o.flops
    assert set(by) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}
    for name, f in by.items():
        t = float(rows[name]["torch_flops"])
        assert abs(f - t) / t <= 0.01, (name, f, t)
    t = float(rows["ALL_LINEAR"]["torch_flops"])
    assert abs(sum(by.values()) - t) / t <= 0.01
    att = sum(o.flops for o in ops if o.kind == "attn")
    assert att == float(rows["ATTENTION_QK+AV (eager, full rectangle; informational)"]["torch_flops"])


# ------------------------------------------------------------------------------------------- docs / IO
def test_formulas_documented():
    src = inspect.getsource(opgen)
    assert "methodology.md` §2" in opgen.__doc__
    for name, fn in inspect.getmembers(opgen, inspect.isfunction):
        if fn.__module__ == "model.opgen" and name not in ("main", "op_rows", "shape"):
            assert fn.__doc__, f"{name} lacks a docstring"
    for eq in ("E1", "E2", "E3", "E4", "E5"):
        assert eq in src
    meth = (ROOT / "docs" / "methodology.md").read_text()
    assert "### Op-trace conventions (`model/opgen.py`)" in meth
    assert "`M_KV/token = L·(kv_lora_rank + qk_rope_head_dim)·q`" in meth


def test_main_writes_csvs(tmp_path):
    opgen.main(["--out", str(tmp_path), "--configs", str(CFG), "--models", str(MODELS)])
    files = sorted(p.name for p in tmp_path.glob("ops_*.csv"))
    assert len(files) == 14 and "ops_stories15M.csv" in files
    with open(tmp_path / "ops_stories15M.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows and set(opgen.CSV_FIELDS) == set(rows[0])
    assert {r["scenario"] for r in rows} == {"prefill_B1_S256", "decode_B1_T256", "decode_B16_T256",
                                            "mixed_chunk256_ctx0+decode_B15_T256"}
    assert all(r["tags"] for r in rows if r["kind"] != "elementwise" or float(r["bytes"]) > 0)
    with open(tmp_path / "summary.csv") as f:
        summ = list(csv.DictReader(f))
    assert len(summ) == 14 * 4
    # representative-layer rows × layers_like reproduce the full-model totals of summary.csv
    for name in ("deepseek-ai__DeepSeek-V2-Lite", "mistralai__Mistral-7B-v0.1", "stories15M"):
        with open(tmp_path / f"ops_{name}.csv") as f:
            rows = list(csv.DictReader(f))
        for s in (r for r in summ if r["model"] == name):
            sel = [r for r in rows if r["scenario"] == s["scenario"]]
            for k in ("flops", "w_rd", "kv_rd", "a_wr"):
                tot = sum(float(r[k]) * int(r["layers_like"]) for r in sel)
                assert tot == pytest.approx(float(s[k]), rel=1e-9), (name, s["scenario"], k)
    ds = opgen.layer_classes(shape("deepseek-ai__DeepSeek-V2-Lite"))
    assert ds == {0: 1, 1: 26}


# ------------------------------------------------------------------------------------------- follow-up checks
def test_mla_decompressed_prefill_form():
    m = shape("deepseek-ai__DeepSeek-V2-Lite")
    H, r, rho, nope, v = 16, 512, 64, 128, 128
    S = 300
    ab = gen_ops(m, prefill(1, S), layers=[2])
    de = gen_ops(m, prefill(1, S), layers=[2], mla_prefill_form="decompressed")
    assert [o.name for o in ab] == [o.name for o in gen_ops(m, prefill(1, S), layers=[2], mla_prefill_form="absorbed")]
    names = [o.name for o in de]
    assert "kv_b_decompress" in names and "q_absorb_uk" not in names and "v_up_uv" not in names
    pairs = S * (S + 1) // 2
    att = sum(o.flops for o in de if o.kind == "attn")
    assert att == 2 * H * (nope + rho) * pairs + 2 * H * v * pairs
    dc = next(o for o in de if o.name == "kv_b_decompress")
    assert dc.flops == 2 * S * r * H * (nope + v) and dc.kv_rd == 2 * S * r and dc.a_wr == 2 * S * H * (nope + v)
    kv = lambda ops: sum(o.kv_rd for o in ops if o.layer == 2)
    assert kv(de) == kv(ab) == 2 * S * (r + rho)                          # latent + RoPE key read once per op path
    w = lambda ops: sum(o.w_rd for o in ops if o.layer == 2)
    assert w(de) == w(ab)                                                 # kv_b read once either way (pure prefill)
    # decode sequences stay absorbed; a mixed iteration has both paths
    mix = gen_ops(m, mixed([(64, 0)], [(3, 500)]), layers=[2], mla_prefill_form="decompressed")
    mn = [o.name for o in mix]
    assert "kv_b_decompress" in mn and "q_absorb_uk" in mn
    assert next(o for o in mix if o.name == "q_absorb_uk").M == 3
    dec_only = gen_ops(m, decode(4, 900), layers=[2])
    assert [(o.name, o.flops) for o in dec_only] == \
        [(o.name, o.flops) for o in gen_ops(m, decode(4, 900), layers=[2], mla_prefill_form="decompressed")]
    with pytest.raises(ValueError):
        gen_ops(m, prefill(1, 8), mla_prefill_form="naive")


def test_zipf_routing_k_equals_e():
    for routing in ("uniform", "zipf"):
        assert opgen.moe_inclusion_probs(8, 8, routing, 1.2) == [1.0] * 8
        assert opgen.moe_active_experts(3, 8, 8, routing, 1.2) == pytest.approx(8)
    pi = opgen.moe_inclusion_probs(8, 7, "zipf", 3.0)                    # near-saturation water-filling
    assert sum(pi) == pytest.approx(7) and max(pi) <= 1
    with pytest.raises(ValueError):
        opgen.moe_inclusion_probs(8, 8, "bogus")
