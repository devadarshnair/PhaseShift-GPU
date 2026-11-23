"""PhaseShift op generator: model config -> per-layer, per-op FLOPs and bytes by class.

Equations are those of `docs/methodology.md` §2 (E1-E5); the op-trace conventions (what each op reads and writes,
how MoE routing, MLA, sliding windows and mixed iterations are modelled) are listed in methodology §2 "Op-trace
conventions" and repeated next to the code below. Pure Python, standard library only.

Inputs
  * a model shape, from a Hugging Face `config.json` (`from_hf_config`) or a llama2.c `.bin` header
    (`from_llama2c_bin`; hidden_dim is read from the header, never inferred);
  * an iteration: a list of `Seq` groups (prefill chunk / decode token, context, multiplicity);
  * bytes per element q_w (weights), q_a (activations), q (KV cache).

Output: a list of `Op` records, one per (layer, op) — or per (layer, op, sequence group) for attention — with the
GEMM-like shape (M, N, K, batch), FLOPs, and compulsory bytes read/written per class (weight / act / KV).
"Compulsory" = every byte the op touches counted once per op; re-reads caused by tiling or cache misses are the
memory-hierarchy model's business, not opgen's.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Iterable, Sequence

# ------------------------------------------------------------------------------------------------- model shapes


@dataclass(frozen=True)
class ModelShape:
    """Transformer shape in methodology §1 notation.

    d = hidden size, L layers, H_q / H_kv query / KV heads, D_h head dim, d_ff dense-MLP width, vocab size.
    windows[l] = sliding-window length of layer l (0 = full attention).
    MoE (n_experts > 0): top_k routed experts per token of width d_ff_expert, plus always-on shared experts of total
    width d_ff_shared; moe_layers[l] says whether layer l uses the MoE block (else the dense d_ff MLP).
    MLA (kv_lora_rank > 0, DeepSeek-V2): latent rank r = kv_lora_rank, decoupled RoPE dim qk_rope_dim, per-head
    no-RoPE query/key dim qk_nope_dim, per-head value dim v_head_dim, optional query LoRA rank q_lora_rank.
    """

    name: str
    d: int
    L: int
    H_q: int
    H_kv: int
    D_h: int
    d_ff: int
    vocab: int
    tie_embeddings: bool
    max_seq: int = 0
    windows: tuple = ()
    n_experts: int = 0
    top_k: int = 0
    d_ff_expert: int = 0
    d_ff_shared: int = 0
    moe_layers: tuple = ()
    kv_lora_rank: int = 0
    qk_rope_dim: int = 0
    qk_nope_dim: int = 0
    v_head_dim: int = 0
    q_lora_rank: int = 0
    source: str = ""

    def __post_init__(self):
        if not self.windows:
            object.__setattr__(self, "windows", (0,) * self.L)
        if not self.moe_layers:
            object.__setattr__(self, "moe_layers", (self.n_experts > 0,) * self.L)
        assert len(self.windows) == self.L and len(self.moe_layers) == self.L
        if self.n_experts:
            assert 0 < self.top_k <= self.n_experts and self.d_ff_expert > 0

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank > 0

    @property
    def g(self) -> float:
        """GQA group size g = H_q / H_KV (methodology §1)."""
        return self.H_q / self.H_kv

    def kv_elems_per_token_layer(self) -> int:
        """KV-cache elements one token stores in one layer (E3 per-layer term).

        Standard / GQA attention: one K and one V vector of H_KV·D_h elements -> 2·H_KV·D_h.
        MLA: the compressed latent c_KV (kv_lora_rank) plus the decoupled RoPE key (qk_rope_dim, shared by all
        heads) -> kv_lora_rank + qk_rope_dim; K and V are reconstructed from c_KV, never cached.
        """
        if self.is_mla:
            return self.kv_lora_rank + self.qk_rope_dim
        return 2 * self.H_kv * self.D_h


def _hf_get(cfg: dict, *keys, default=None):
    """First non-null value among cfg[keys] (HF configs name the same field differently per family)."""
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return default


def from_hf_config(cfg, name: str | None = None) -> ModelShape:
    """Build a ModelShape from a Hugging Face config.json (path or dict).

    Field rules (checked against transformers 5.22 for the pinned configs, configs/configs.lock):
      * head_dim = `head_dim` if given, else hidden_size / num_attention_heads (Qwen3 has head_dim 128 ≠ 2048/32);
      * H_KV = `num_key_value_heads` (GQA), default H_q (MHA);
      * sliding window (methodology E3 / op-trace conventions): Qwen2 semantics — `sliding_window` applies only if
        `use_sliding_window` is true, and then only to layers i ≥ `max_window_layers` (Qwen2Config.__post_init__);
        configs without `use_sliding_window` (Mistral) apply `sliding_window` to every layer; an explicit
        `layer_types` list wins over both;
      * MoE experts: `num_experts` (Qwen3-MoE, OLMoE) / `num_local_experts` (Mixtral) / `n_routed_experts`
        (DeepSeek); expert width `moe_intermediate_size` if given, else `intermediate_size` (Mixtral, OLMoE);
        DeepSeek shared experts = one MLP of width moe_intermediate_size·n_shared_experts, MoE only in layers
        i ≥ first_k_dense_replace with i % moe_layer_freq == 0; Qwen3-MoE MoE in layers not in `mlp_only_layers`
        with (i+1) % decoder_sparse_step == 0;
      * MLA when `kv_lora_rank` is present (DeepSeek-V2/V3).
    Biases (Qwen2 q/k/v) are ignored: (H_q + 2·H_KV)·D_h parameters per layer, < 0.01 % of any pinned model.
    """
    if not isinstance(cfg, dict):
        p = Path(cfg)
        cfg = json.loads(p.read_text())
        name = name or p.stem
    d = int(cfg["hidden_size"])
    L = int(cfg["num_hidden_layers"])
    H_q = int(cfg["num_attention_heads"])
    H_kv = int(_hf_get(cfg, "num_key_value_heads", default=H_q))
    D_h = int(_hf_get(cfg, "head_dim", default=d // H_q))
    # sliding windows
    sw = cfg.get("sliding_window")
    layer_types = cfg.get("layer_types")
    if layer_types:
        windows = tuple(int(sw) if (t == "sliding_attention" and sw) else 0 for t in layer_types)
    elif sw and cfg.get("use_sliding_window", True):
        mwl = cfg.get("max_window_layers") if "use_sliding_window" in cfg else None
        windows = tuple(int(sw) if (mwl is None or i >= mwl) else 0 for i in range(L))
    else:
        windows = (0,) * L
    # MoE
    n_exp = int(_hf_get(cfg, "num_experts", "num_local_experts", "n_routed_experts", default=0) or 0)
    top_k = int(cfg.get("num_experts_per_tok") or 0) if n_exp else 0
    d_ff_exp = int(_hf_get(cfg, "moe_intermediate_size", "intermediate_size")) if n_exp else 0
    d_ff_shared = 0
    if n_exp and cfg.get("n_shared_experts"):
        d_ff_shared = int(cfg["moe_intermediate_size"]) * int(cfg["n_shared_experts"])
    elif n_exp and cfg.get("shared_expert_intermediate_size"):
        d_ff_shared = int(cfg["shared_expert_intermediate_size"])
    if not n_exp:
        moe = (False,) * L
    elif "first_k_dense_replace" in cfg:
        k0, freq = int(cfg["first_k_dense_replace"]), int(cfg.get("moe_layer_freq") or 1)
        moe = tuple(i >= k0 and i % freq == 0 for i in range(L))
    else:
        only = set(cfg.get("mlp_only_layers") or [])
        step = int(cfg.get("decoder_sparse_step") or 1)
        moe = tuple(i not in only and (i + 1) % step == 0 for i in range(L))
    mla = {}
    if cfg.get("kv_lora_rank"):
        mla = dict(kv_lora_rank=int(cfg["kv_lora_rank"]), qk_rope_dim=int(cfg["qk_rope_head_dim"]),
                   qk_nope_dim=int(cfg["qk_nope_head_dim"]), v_head_dim=int(cfg["v_head_dim"]),
                   q_lora_rank=int(cfg.get("q_lora_rank") or 0))
    return ModelShape(name=name or cfg.get("_name_or_path", "hf-model"), d=d, L=L, H_q=H_q, H_kv=H_kv, D_h=D_h,
                      d_ff=int(cfg["intermediate_size"]), vocab=int(cfg["vocab_size"]),
                      tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
                      max_seq=int(cfg.get("max_position_embeddings") or 0), windows=windows,
                      n_experts=n_exp, top_k=top_k, d_ff_expert=d_ff_exp, d_ff_shared=d_ff_shared, moe_layers=moe,
                      source="hf-config", **mla)


LLAMA2C_HEADER = struct.Struct("<7i")   # dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len


def from_llama2c_bin(path, name: str | None = None) -> ModelShape:
    """Read the 28-byte llama2.c checkpoint header (run.c `Config`): dim, hidden_dim, n_layers, n_heads, n_kv_heads,
    vocab_size, seq_len. A negative vocab_size means an unshared classifier (run.c: shared_weights = vocab_size > 0).
    hidden_dim comes from the header (not inferred from dim), head_size = dim / n_heads (run.c)."""
    p = Path(path)
    with open(p, "rb") as f:
        dim, hidden, L, H, H_kv, vocab, seq = LLAMA2C_HEADER.unpack(f.read(LLAMA2C_HEADER.size))
    return ModelShape(name=name or p.stem, d=dim, L=L, H_q=H, H_kv=H_kv, D_h=dim // H, d_ff=hidden,
                      vocab=abs(vocab), tie_embeddings=vocab > 0, max_seq=seq, source="llama2c-bin")


def n_params(m: ModelShape) -> int:
    """Number of weights N_params (methodology §1): embedding, per-layer attention + MLP/MoE + 2 RMSNorm vectors,
    final norm, LM head (not counted again when tied). Routed experts are all counted (total, not active)."""
    if m.is_mla:
        H, r, rho, nope, v = m.H_q, m.kv_lora_rank, m.qk_rope_dim, m.qk_nope_dim, m.v_head_dim
        q = (m.d * m.q_lora_rank + m.q_lora_rank + m.q_lora_rank * H * (nope + rho)) if m.q_lora_rank \
            else m.d * H * (nope + rho)
        attn = q + m.d * (r + rho) + r + r * H * (nope + v) + H * v * m.d
    else:
        attn = m.d * m.D_h * (m.H_q + 2 * m.H_kv) + m.H_q * m.D_h * m.d
    total = m.vocab * m.d + m.d + (0 if m.tie_embeddings else m.vocab * m.d)
    for l in range(m.L):
        if m.moe_layers[l]:
            mlp = m.d * m.n_experts + m.n_experts * 3 * m.d * m.d_ff_expert + 3 * m.d * m.d_ff_shared
        else:
            mlp = 3 * m.d * m.d_ff
        total += attn + mlp + 2 * m.d
    return total


def kv_bytes_per_token(m: ModelShape, q: float) -> float:
    """E3 per token: M_KV/token = 2·L·H_KV·D_h·q (MHA/GQA) or L·(kv_lora_rank + qk_rope_dim)·q (MLA).
    Independent of sliding windows (a window caps how many tokens are kept, not the bytes per token)."""
    return m.L * m.kv_elems_per_token_layer() * q


def kv_bytes_total(m: ModelShape, contexts: Iterable[int], q: float, block: int = 1) -> float:
    """E3 total, generalised (methodology E3): Σ_seq Σ_layers ⌈min(T_i, W_l)/b⌉·b · (per-layer per-token bytes).
    T_i = cached tokens of sequence i, W_l the layer's sliding window (0 = none; a windowed layer keeps only the last
    W_l tokens, rolling buffer), b the paging block size (b = 1: no paging)."""
    per = m.kv_elems_per_token_layer() * q
    tot = 0.0
    for T in contexts:
        for W in m.windows:
            t = min(T, W) if W else T
            tot += math.ceil(t / block) * block * per
    return tot


# ------------------------------------------------------------------------------------------------- iterations


@dataclass(frozen=True)
class Seq:
    """A group of `count` identical sequences in one iteration.

    n_new  new tokens processed this iteration (prefill chunk length S, or 1 for decode);
    ctx    tokens already in the KV cache before this iteration (0 for a full prefill / first chunk);
    emits  True if the group's last token produces logits (decode, and the final prefill chunk).
    Query token j (0-based) of the chunk attends causally to ctx + j + 1 keys (its own K/V included), capped by the
    layer's sliding window W: min(ctx + j + 1, W).
    """

    kind: str
    n_new: int
    ctx: int = 0
    count: int = 1
    emits: bool = True

    def __post_init__(self):
        assert self.kind in ("prefill", "decode"), self.kind
        assert self.n_new >= 1 and self.ctx >= 0 and self.count >= 1
        if self.kind == "decode":
            assert self.n_new == 1


def prefill(B: int, S: int, ctx: int = 0, emits: bool = True) -> list[Seq]:
    """B sequences, each a prefill chunk of S tokens after ctx cached tokens (ctx = 0: full prefill)."""
    return [Seq("prefill", S, ctx, B, emits)]


def decode(B: int, T: int) -> list[Seq]:
    """B sequences decoding one token at context length T, where T = KV entries the new query attends to = the
    cache length after the new token's own K/V were appended (so the step reads exactly E3's B·T tokens of KV and
    performs E4's 4·T·d attention FLOPs per layer and sequence)."""
    assert T >= 1
    return [Seq("decode", 1, T - 1, B)]


def mixed(chunks: Sequence[tuple[int, int]], decodes: Sequence[tuple[int, int]]) -> list[Seq]:
    """Mixed iteration (methodology op-trace conventions): prefill chunks [(S, ctx), …] and decode groups
    [(B, T), …] batched together. Linear layers see the concatenated token rows of all requests; attention is
    per request."""
    out = [Seq("prefill", S, ctx) for S, ctx in chunks]
    for B, T in decodes:
        out += decode(B, T)
    return out


def _keys_sum(n: int, ctx: int, W: int) -> int:
    """Σ_{j=0}^{n-1} min(ctx + j + 1, W) (W = 0: no cap) = Σ of keys attended by the n causal queries.
    Without a window: n·ctx + n(n+1)/2. With a window: the first m = clamp(W − ctx, 0, n) queries see ctx + j + 1
    keys, the rest see W."""
    if not W:
        return n * ctx + n * (n + 1) // 2
    m = max(0, min(n, W - ctx))
    return m * ctx + m * (m + 1) // 2 + (n - m) * W


def _keys_touched(n: int, ctx: int, W: int) -> int:
    """Distinct keys read by an n-token chunk after ctx cached tokens: ctx + n, or min(ctx + n, W + n − 1) with a
    window W (the first query reaches back W − 1 tokens before itself)."""
    return min(ctx + n, W + n - 1) if W else ctx + n


# ------------------------------------------------------------------------------------------------- op records


@dataclass
class Op:
    """One op. Shape: a batch of `batch` independent M×K @ K×N products (linear: batch = 1, M = token rows).
    Bytes are compulsory bytes per class: w_rd weights read; a_rd / a_wr activations read / written; kv_rd / kv_wr
    KV-cache read / written. FLOPs count a multiply-accumulate as 2 (E1); elementwise ops count 1 per arithmetic or
    transcendental operation (conventions in the op's helper)."""

    layer: int          # -1 = before the first layer (embedding), L = after the last (final norm, LM head)
    name: str
    kind: str           # linear | attn | elementwise | embed
    M: int
    N: int
    K: int
    batch: int
    flops: float
    w_rd: float = 0.0
    a_rd: float = 0.0
    a_wr: float = 0.0
    kv_rd: float = 0.0
    kv_wr: float = 0.0
    causal: str = ""    # attention ops: "causal" (masked half skipped) or "full"; "" otherwise
    group: str = ""     # attention ops: which Seq group (e.g. "prefill S=512 ctx=0 x1")
    n_active: float = 0.0   # MoE expert ops: expected number of distinct experts touched

    @property
    def bytes(self) -> float:
        return self.w_rd + self.a_rd + self.a_wr + self.kv_rd + self.kv_wr

    @property
    def ai(self) -> float:
        """Arithmetic intensity FLOPs / compulsory bytes (methodology §1)."""
        b = self.bytes
        return self.flops / b if b else math.inf

    @property
    def tags(self) -> str:
        """Class tags (weight / KV / act) of the byte streams this op touches."""
        t = []
        if self.w_rd:
            t.append("weight")
        if self.kv_rd or self.kv_wr:
            t.append("kv")
        if self.a_rd or self.a_wr:
            t.append("act")
        return "|".join(t)


def linear(layer: int, name: str, n_tok: float, d_in: int, d_out: int, q_w: float, q_a: float) -> Op:
    """E1/E2: FLOPs = 2·N_tok·d_in·d_out; weights q_w·d_in·d_out read once and reused across the N_tok rows;
    activations q_a·N_tok·d_in in, q_a·N_tok·d_out out."""
    return Op(layer, name, "linear", int(n_tok), d_out, d_in, 1, 2.0 * n_tok * d_in * d_out,
              w_rd=q_w * d_in * d_out, a_rd=q_a * n_tok * d_in, a_wr=q_a * n_tok * d_out)


def rmsnorm(layer: int, name: str, n_tok: float, width: int, q_w: float, q_a: float) -> Op:
    """RMSNorm over n_tok rows of `width`: x² (1), running sum (1), scale by rsqrt (1), × weight (1) = 4 FLOPs per
    element (the per-row rsqrt is negligible); reads x and the weight vector, writes y."""
    return Op(layer, name, "elementwise", int(n_tok), width, 1, 1, 4.0 * n_tok * width,
              w_rd=q_w * width, a_rd=q_a * n_tok * width, a_wr=q_a * n_tok * width)


def residual(layer: int, name: str, n_tok: float, width: int, q_a: float) -> Op:
    """Residual add y = x + f(x): 1 FLOP per element; reads two tensors, writes one."""
    return Op(layer, name, "elementwise", int(n_tok), width, 1, 1, 1.0 * n_tok * width,
              a_rd=2 * q_a * n_tok * width, a_wr=q_a * n_tok * width)


def silu_mul(layer: int, name: str, n_tok: float, width: int, q_a: float) -> Op:
    """SiLU-gate h = silu(g)·u with silu(g) = g/(1+exp(−g)): exp (1), add (1), divide (1), multiply by u (1) =
    4 FLOPs per element; reads g and u, writes h."""
    return Op(layer, name, "elementwise", int(n_tok), width, 1, 1, 4.0 * n_tok * width,
              a_rd=2 * q_a * n_tok * width, a_wr=q_a * n_tok * width)


# ------------------------------------------------------------------------------------------------- MoE routing


def moe_inclusion_probs(E: int, k: int, routing: str = "uniform", zipf_s: float = 1.0) -> list[float]:
    """Per-token probability π_e that expert e is among the token's top-k (Σ_e π_e = k).

    uniform: π_e = k/E (every k-subset equally likely).
    zipf:    expert popularity p_e ∝ (e+1)^(−s) (s = zipf_s); π_e = min(1, λ·p_e) with λ chosen so that Σ π_e = k
             (water-filling: an expert cannot be picked more than once per token). Which expert is popular is
             arbitrary and does not matter for the byte/FLOP totals.
    """
    if routing not in ("uniform", "zipf"):
        raise ValueError(f"unknown routing {routing!r}")
    if routing == "uniform" or k >= E:
        return [min(1.0, k / E)] * E            # k = E: every expert is always selected
    p = [(e + 1) ** (-zipf_s) for e in range(E)]
    capped = [False] * E
    pi = [0.0] * E
    for _ in range(E + 1):
        free = k - sum(capped)
        mass = sum(pe for pe, c in zip(p, capped) if not c)
        if free <= 0 or mass <= 0:
            pi = [1.0 if c else 0.0 for c in capped]
            break
        lam = free / mass
        newly = False
        for e in range(E):
            if not capped[e] and lam * p[e] >= 1.0:
                capped[e] = True
                newly = True
        if not newly:
            pi = [1.0 if capped[e] else lam * p[e] for e in range(E)]
            break
    return pi


def moe_active_experts(n_tok: float, E: int, k: int, routing: str = "uniform", zipf_s: float = 1.0) -> float:
    """Expected number of distinct experts touched by n_tok tokens: Σ_e [1 − (1 − π_e)^n_tok], assuming tokens are
    routed independently (real routing is correlated across neighbouring tokens, so this is an upper estimate of
    distinct experts, i.e. of expert-weight bytes). FLOPs do not depend on routing (each token uses exactly k)."""
    return sum(1.0 - (1.0 - pe) ** n_tok for pe in moe_inclusion_probs(E, k, routing, zipf_s))


# ------------------------------------------------------------------------------------------------- generator


def gen_ops(m: ModelShape, seqs: Sequence[Seq], q_w: float = 2, q_a: float = 2, q: float = 2,
            causal: bool = True, fused_attention: bool = True, lm_head_all_tokens: bool = False,
            routing: str = "uniform", zipf_s: float = 1.0, layers: Iterable[int] | None = None,
            mla_prefill_form: str = "absorbed") -> list[Op]:
    """Per-op trace of one iteration (methodology §2 "Op-trace conventions").

    causal            True: attention kernels skip the masked half (FLOPs = Σ_j keys_j, E4's ≈ 2·S²·d for prefill);
                      False: they compute the full n_new × keys rectangle (E4's 4·S²·d).
    fused_attention   True: QKᵀ → softmax → AV is one kernel, scores stay on chip (no score traffic);
                      False: scores and probabilities are written to and read from memory (q_a bytes per element).
    lm_head_all_tokens  True: final norm + LM head on every token row (HF forward); False: only on each emitting
                      sequence's last token (serving engines; non-final prefill chunks emit nothing).
    layers            restrict the per-layer ops to these layer indices (pre/post ops are always emitted).
    mla_prefill_form  MLA models only: "absorbed" (default, every phase in the weight-absorbed form) or "decompressed"
                      (prefill sequences decompress K/V from the cached latent and run per-head attention; decode
                      sequences stay absorbed). See `_mla_attention`.
    """
    if mla_prefill_form not in ("absorbed", "decompressed"):
        raise ValueError(f"mla_prefill_form must be 'absorbed' or 'decompressed', not {mla_prefill_form!r}")
    n_tok = sum(s.n_new * s.count for s in seqs)
    n_lm = n_tok if lm_head_all_tokens else sum(s.count for s in seqs if s.emits)
    d = m.d
    ops: list[Op] = []
    # embedding gather: one row of q_w·d bytes per token (weights), written as activations
    ops.append(Op(-1, "embed", "embed", n_tok, d, 1, 1, 0.0, w_rd=q_w * n_tok * d, a_wr=q_a * n_tok * d))
    sel = range(m.L) if layers is None else layers
    for l in sel:
        W = m.windows[l]
        ops.append(rmsnorm(l, "attn_norm", n_tok, d, q_w, q_a))
        if m.is_mla:
            ops += _mla_attention(m, l, seqs, n_tok, W, q_w, q_a, q, causal, fused_attention, mla_prefill_form)
        else:
            ops += _gqa_attention(m, l, seqs, n_tok, W, q_w, q_a, q, causal, fused_attention)
        ops.append(residual(l, "attn_residual", n_tok, d, q_a))
        ops.append(rmsnorm(l, "ffn_norm", n_tok, d, q_w, q_a))
        if m.moe_layers[l]:
            ops += _moe_block(m, l, n_tok, q_w, q_a, routing, zipf_s)
        else:
            ops += [linear(l, "gate_proj", n_tok, d, m.d_ff, q_w, q_a), linear(l, "up_proj", n_tok, d, m.d_ff, q_w, q_a),
                    silu_mul(l, "silu_mul", n_tok, m.d_ff, q_a), linear(l, "down_proj", n_tok, m.d_ff, d, q_w, q_a)]
        ops.append(residual(l, "ffn_residual", n_tok, d, q_a))
    ops.append(rmsnorm(m.L, "final_norm", n_tok, d, q_w, q_a))
    ops.append(linear(m.L, "lm_head", n_lm, d, m.vocab, q_w, q_a))
    return ops


def _attn_core(l, seqs, W, H_q, dk, dv, kv_k, kv_v, q_a, q, causal, fused, n_head_q_bytes, prefix="attn"):
    """QKᵀ, softmax and AV for every sequence group (E4).

    dk = per-head dot-product length of QKᵀ, dv = per-head length of the AV output; kv_k / kv_v = KV-cache elements
    per token read by QKᵀ / AV (one layer). FLOPs: QKᵀ 2·H_q·dk per (query, key) pair, AV 2·H_q·dv per pair;
    pairs = Σ_j keys_j (causal) or n_new·keys_touched (full). Softmax: max (1), subtract (1), exp (1), sum (1),
    divide (1) = 5 FLOPs per score. KV read once per op = keys_touched tokens (compulsory)."""
    out = []
    for s in seqs:
        kt = _keys_touched(s.n_new, s.ctx, W)
        pairs = _keys_sum(s.n_new, s.ctx, W) if causal else s.n_new * kt
        c = s.count
        grp = f"{s.kind} n={s.n_new} ctx={s.ctx} x{c}"
        tag = "causal" if causal else "full"
        sc = q_a * H_q * pairs * c                       # score (or probability) bytes if they go to memory
        out.append(Op(l, f"{prefix}_qk", "attn", s.n_new, kt, dk, H_q * c, 2.0 * H_q * dk * pairs * c,
                      a_rd=q_a * s.n_new * n_head_q_bytes * c, a_wr=0 if fused else sc,
                      kv_rd=q * kt * kv_k * c, causal=tag, group=grp))
        out.append(Op(l, f"{prefix}_softmax", "elementwise", s.n_new, kt, 1, H_q * c, 5.0 * H_q * pairs * c,
                      a_rd=0 if fused else sc, a_wr=0 if fused else sc, causal=tag, group=grp))
        out.append(Op(l, f"{prefix}_av", "attn", s.n_new, dv, kt, H_q * c, 2.0 * H_q * dv * pairs * c,
                      a_rd=0 if fused else sc, a_wr=q_a * s.n_new * H_q * dv * c,
                      kv_rd=q * kt * kv_v * c, causal=tag, group=grp))
    return out


def _gqa_attention(m, l, seqs, n_tok, W, q_w, q_a, q, causal, fused):
    """MHA/GQA attention block: Q, K, V projections; RoPE fused with the KV-cache write (kernel: reads q, k, v
    activations, rotates q and k — 3 FLOPs per rotated element: two multiplies and one add — writes q back as an
    activation and k, v into the KV cache, 2·H_KV·D_h·q bytes per token); QKᵀ/softmax/AV per sequence group; O
    projection. GQA: K and V have H_KV heads; FLOPs use H_q heads (E4: GQA does not reduce attention FLOPs)."""
    d, Dh, Hq, Hkv = m.d, m.D_h, m.H_q, m.H_kv
    ops = [linear(l, "q_proj", n_tok, d, Hq * Dh, q_w, q_a), linear(l, "k_proj", n_tok, d, Hkv * Dh, q_w, q_a),
           linear(l, "v_proj", n_tok, d, Hkv * Dh, q_w, q_a),
           Op(l, "rope_kv_write", "elementwise", n_tok, (Hq + 2 * Hkv) * Dh, 1, 1, 3.0 * n_tok * (Hq + Hkv) * Dh,
              a_rd=q_a * n_tok * (Hq + 2 * Hkv) * Dh, a_wr=q_a * n_tok * Hq * Dh, kv_wr=q * n_tok * 2 * Hkv * Dh)]
    ops += _attn_core(l, seqs, W, Hq, Dh, Dh, Hkv * Dh, Hkv * Dh, q_a, q, causal, fused, Hq * Dh)
    ops.append(linear(l, "o_proj", n_tok, Hq * Dh, d, q_w, q_a))
    return ops


def _mla_attention(m, l, seqs, n_tok, W, q_w, q_a, q, causal, fused, prefill_form="absorbed"):
    """MLA (DeepSeek-V2). Modeling choice (op-trace convention 10): by default every phase uses the weight-absorbed
    form: the cache holds the latent c_KV (r = kv_lora_rank) and the shared RoPE key (ρ = qk_rope_dim); K and V are
    never materialised. q = W_Q·x (H·(nope+ρ)); [c_KV, k_rope] = W_KVa·x; RMSNorm(c_KV) + RoPE(q_rope, k_rope) + cache
    write (r+ρ elements per token); q_nope is absorbed into the latent space per head (W_UK: nope → r); scores use
    H heads of dot-product length r+ρ against the cached (r+ρ) vectors; AV accumulates H heads of length r over
    c_KV (already read by QKᵀ in the fused kernel: kv_rd = 0 for AV when fused); per-head up-projection W_UV
    (r → v); O projection (H·v → d). W_UK and W_UV together are kv_b_proj.

    prefill_form="decompressed" (opt-in): prefill sequences instead decompress, for every key they touch, the cached
    latent into per-head K_nope and V with kv_b_proj (`kv_b_decompress`: reads r cache elements per key, writes
    H·(nope+v) activations per key, FLOPs 2·keys·r·H·(nope+v), kv_b weights read once), then run ordinary per-head
    attention: QKᵀ of length nope+ρ (reads K_nope activations + k_rope from the cache), AV of length v (reads V
    activations); no W_UK/W_UV on those tokens. Decode sequences keep the absorbed form, so a mixed iteration has
    both paths (and reads W_UK/W_UV and kv_b in separate kernels)."""
    d, H, r, rho, nope, v = m.d, m.H_q, m.kv_lora_rank, m.qk_rope_dim, m.qk_nope_dim, m.v_head_dim
    dec = [s for s in seqs if prefill_form == "absorbed" or s.kind == "decode"]       # absorbed path
    pre = [s for s in seqs if prefill_form == "decompressed" and s.kind == "prefill"]  # decompressed path
    n_abs = sum(s.n_new * s.count for s in dec)
    ops = []
    if m.q_lora_rank:
        ops += [linear(l, "q_a_proj", n_tok, d, m.q_lora_rank, q_w, q_a),
                rmsnorm(l, "q_a_norm", n_tok, m.q_lora_rank, q_w, q_a),
                linear(l, "q_b_proj", n_tok, m.q_lora_rank, H * (nope + rho), q_w, q_a)]
    else:
        ops.append(linear(l, "q_proj", n_tok, d, H * (nope + rho), q_w, q_a))
    ops.append(linear(l, "kv_a_proj", n_tok, d, r + rho, q_w, q_a))
    ops.append(Op(l, "kv_norm_rope_kv_write", "elementwise", n_tok, r + rho, 1, 1,
                  4.0 * n_tok * r + 3.0 * n_tok * (H + 1) * rho, w_rd=q_w * r,
                  a_rd=q_a * n_tok * (r + rho + H * rho), a_wr=q_a * n_tok * H * rho, kv_wr=q * n_tok * (r + rho)))
    if dec:
        ops.append(Op(l, "q_absorb_uk", "linear", n_abs, r, nope, H, 2.0 * n_abs * H * nope * r,
                      w_rd=q_w * H * nope * r, a_rd=q_a * n_abs * H * nope, a_wr=q_a * n_abs * H * r))
        ops += _attn_core(l, dec, W, H, r + rho, r, r + rho, 0 if fused else r, q_a, q, causal, fused, H * (r + rho))
        ops.append(Op(l, "v_up_uv", "linear", n_abs, v, r, H, 2.0 * n_abs * H * r * v, w_rd=q_w * H * r * v,
                      a_rd=q_a * n_abs * H * r, a_wr=q_a * n_abs * H * v))
    if pre:
        keys = sum(_keys_touched(s.n_new, s.ctx, W) * s.count for s in pre)
        ops.append(Op(l, "kv_b_decompress", "linear", keys, H * (nope + v), r, 1, 2.0 * keys * r * H * (nope + v),
                      w_rd=q_w * r * H * (nope + v), a_wr=q_a * keys * H * (nope + v), kv_rd=q * keys * r))
        core = _attn_core(l, pre, W, H, nope + rho, v, rho, 0, q_a, q, causal, fused, H * (nope + rho), prefix="attn")
        for i, s in enumerate(pre):                # K_nope / V come from the decompressed activations
            kt = _keys_touched(s.n_new, s.ctx, W) * s.count
            core[3 * i].a_rd += q_a * kt * H * nope
            core[3 * i + 2].a_rd += q_a * kt * H * v
        ops += core
    ops.append(linear(l, "o_proj", n_tok, H * v, d, q_w, q_a))
    return ops


def _moe_block(m, l, n_tok, q_w, q_a, routing, zipf_s):
    """MoE MLP: router (d → E linear; top-k selection negligible), routed experts as grouped GEMMs over the
    N_tok·k (token, expert) rows — FLOPs 2·N_tok·k·d·d_ff_expert per projection regardless of routing; expert
    weight bytes = E[distinct experts touched]·d·d_ff_expert·q_w (`moe_active_experts`); each routed row reads its
    token's input (gather) — then the weighted combine of the k expert outputs (2 FLOPs per element: multiply by
    the gate weight, add) plus the shared-expert output when present. Shared experts are a dense MLP."""
    d, E, k, f = m.d, m.n_experts, m.top_k, m.d_ff_expert
    rows = n_tok * k
    na = moe_active_experts(n_tok, E, k, routing, zipf_s)
    ops = [linear(l, "router", n_tok, d, E, q_w, q_a)]
    for name, din, dout in (("expert_gate", d, f), ("expert_up", d, f)):
        o = linear(l, name, rows, din, dout, q_w, q_a)
        o.w_rd, o.n_active = na * din * dout * q_w, na
        ops.append(o)
    ops.append(silu_mul(l, "expert_silu_mul", rows, f, q_a))
    o = linear(l, "expert_down", rows, f, d, q_w, q_a)
    o.w_rd, o.n_active = na * f * d * q_w, na
    ops.append(o)
    if m.d_ff_shared:
        fs = m.d_ff_shared
        ops += [linear(l, "shared_gate", n_tok, d, fs, q_w, q_a), linear(l, "shared_up", n_tok, d, fs, q_w, q_a),
                silu_mul(l, "shared_silu_mul", n_tok, fs, q_a), linear(l, "shared_down", n_tok, fs, d, q_w, q_a)]
    extra = n_tok * d if m.d_ff_shared else 0
    ops.append(Op(l, "expert_combine", "elementwise", int(n_tok), d, k, 1, 2.0 * rows * d + extra,
                  a_rd=q_a * (rows * d + extra), a_wr=q_a * n_tok * d))
    return ops


# ------------------------------------------------------------------------------------------------- summaries / IO


def totals(ops: Iterable[Op]) -> dict:
    """Sum FLOPs and bytes by class over ops."""
    t = dict(flops=0.0, w_rd=0.0, a_rd=0.0, a_wr=0.0, kv_rd=0.0, kv_wr=0.0)
    for o in ops:
        for k in t:
            t[k] += getattr(o, k)
    t["bytes"] = t["w_rd"] + t["a_rd"] + t["a_wr"] + t["kv_rd"] + t["kv_wr"]
    t["ai"] = t["flops"] / t["bytes"] if t["bytes"] else math.inf
    return t


def attention_ai_kv(ops: Iterable[Op]) -> float:
    """E5's ratio: attention matmul FLOPs (QKᵀ + AV) ÷ KV-cache bytes read, over the given ops."""
    ops = [o for o in ops if o.kind == "attn"]
    return sum(o.flops for o in ops) / sum(o.kv_rd for o in ops)


CSV_FIELDS = ["model", "scenario", "q_w", "q_a", "q", "layer", "layers_like", "name", "kind", "group", "causal",
              "M", "N", "K",
              "batch", "flops", "w_rd", "a_rd", "a_wr", "kv_rd", "kv_wr", "bytes", "ai", "tags", "n_active"]


def layer_classes(m: ModelShape) -> dict[int, int]:
    """Representative layers: {first layer of each class: number of layers in the class}. Layers with the same
    (sliding window, MoE-or-dense) signature have identical ops, so the CSV lists one of each and `layers_like`
    says how many layers it stands for (Σ over rows of value × layers_like = the full-model total)."""
    reps: dict[tuple, list[int]] = {}
    for l in range(m.L):
        reps.setdefault((m.windows[l], m.moe_layers[l]), []).append(l)
    return {ls[0]: len(ls) for ls in reps.values()}


def op_rows(model: str, scenario: str, ops: Iterable[Op], q_w, q_a, q, like: dict | None = None) -> list[dict]:
    """CSV rows (CSV_FIELDS) for ops; like = layer_classes() multiplicities (pre/post ops count once)."""
    rows = []
    for o in ops:
        r = {f.name: getattr(o, f.name) for f in fields(o)}
        r.update(model=model, scenario=scenario, q_w=q_w, q_a=q_a, q=q, bytes=o.bytes, ai=o.ai, tags=o.tags,
                 layers_like=(like or {}).get(o.layer, 1))
        rows.append({k: r[k] for k in CSV_FIELDS})
    return rows


def standard_scenarios(m: ModelShape) -> dict[str, list[Seq]]:
    """Scenarios written to results/<run>/ops_<model>.csv. Lengths are capped at the model's max context
    (stories15M: 256) so the tiny models get physically possible shapes."""
    cap = m.max_seq or 1 << 30
    S, T1, T2, C = min(512, cap), min(512, cap), min(4096, cap), min(256, cap)
    return {f"prefill_B1_S{S}": prefill(1, S),
            f"decode_B1_T{T1}": decode(1, T1),
            f"decode_B16_T{T2}": decode(16, T2),
            f"mixed_chunk{C}_ctx0+decode_B15_T{T2}": mixed([(C, 0)], [(15, T2)])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/P1.1")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--models", default="data/models")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shapes = []
    for b in ("stories260K", "stories15M"):
        p = Path(a.models) / f"{b}.bin"
        if p.exists():
            shapes.append((from_llama2c_bin(p), 4))           # llama2.c checkpoints are fp32
    for p in sorted(Path(a.configs).glob("*.json")):
        shapes.append((from_hf_config(p), 2))                 # pinned HF configs: bf16
    summary = []
    for m, qb in shapes:
        rows = []
        like = layer_classes(m)
        for sc, seqs in standard_scenarios(m).items():
            ops = gen_ops(m, seqs, q_w=qb, q_a=qb, q=qb)
            rep = [o for o in ops if o.layer in like or o.layer in (-1, m.L)]
            rows += op_rows(m.name, sc, rep, qb, qb, qb, like)
            t = totals(ops)
            summary.append(dict(model=m.name, scenario=sc, q=qb, n_params=n_params(m),
                                kv_bytes_per_token=kv_bytes_per_token(m, qb), **{k: t[k] for k in t}))
        with open(out / f"ops_{m.name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {len(shapes)} ops_*.csv files and summary.csv to {out}")


if __name__ == "__main__":
    main()
