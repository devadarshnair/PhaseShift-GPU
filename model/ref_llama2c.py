"""NumPy reference of llama2.c `run.c`.

A float32 forward pass that mirrors `run.c` (llama2.c [cand:D15], commit pinned in `third_party/LOCK`) op for op,
the v0 `.bin` checkpoint reader, and the BPE tokenizer (`encode`, `decode`, `safe_printf`). It is the golden model
for the Vortex kernel fixtures (`model/make_fixtures.py`) and for the teacher-forced checks.

Nothing here imports or executes llama2.c code; only the file formats it defines are read:

* checkpoint v0: header of 7 little-endian int32 `dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size,
  seq_len`, then float32 tensors in the order of `run.c:memory_map_weights` (token embedding, rms_att, wq, wk, wv,
  wo, rms_ffn, w1, w2, w3, rms_final, 2 × seq_len·head_size/2 legacy RoPE tables that are skipped, then wcls only
  when the classifier is unshared). A negative `vocab_size` means an unshared classifier (`run.c:read_checkpoint`).
* tokenizer: int32 `max_token_length`, then per token float32 score, int32 length, raw bytes (`run.c:build_tokenizer`).

Fidelity notes (what "mirrors run.c" means here):

* every arithmetic op is done in the reference dtype (float32 by default) with float32 constants (`1e-5f`,
  `10000.0f`, `sqrtf(head_size)`), in run.c's order of operations within each element-wise formula;
* reductions (dot products, sums) use NumPy/BLAS summation order, not run.c's sequential loop, so logits differ from
  run.c in the last bits. That is why the Tests compares argmaxes with the near-tie margin rule instead of bits;
* the same code runs in float64 (`dtype=np.float64`) for the float64 cross-check of the test.

Strings are handled as `bytes` because run.c works on raw bytes (UTF-8 pieces, `<0xXX>` byte tokens).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

PS_ROOT = Path(__file__).resolve().parents[1]

# Models known to the pipeline (llama2.c golden runs): checkpoint, tokenizer, golden text, golden token ids.
MODELS = {
    "stories260K": dict(ckpt="data/models/stories260K.bin", tok="data/models/tok512.bin",
                        golden="results/S8/golden_260K.txt", golden_tokens="results/S8/golden_tokens_260K.txt"),
    "stories15M": dict(ckpt="data/models/stories15M.bin", tok="third_party/llama2.c/tokenizer.bin",
                       golden="results/S8/golden_15M.txt", golden_tokens="results/S8/golden_tokens_15M.txt"),
}
GOLDEN_PROMPT = b"Once upon a time"
BOS = 1
NEAR_TIE_REL_MARGIN = 1e-4     # |l1 - l2| / |l1| <= 1e-4 counts as an fp32 near-tie, not a bug
F64_MAX_ABS_TOL = 1e-3         # per-step logits max-abs difference vs a float64 run


# ------------------------------------------------------------------------------------------------ checkpoint
@dataclasses.dataclass(frozen=True)
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    shared_classifier: bool

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:          # run.c: (dim * n_kv_heads) / n_heads
        return (self.dim * self.n_kv_heads) // self.n_heads

    @property
    def kv_mul(self) -> int:          # run.c: n_heads / n_kv_heads (GQA group size g)
        return self.n_heads // self.n_kv_heads


WEIGHT_NAMES = ("token_embedding_table", "rms_att_weight", "wq", "wk", "wv", "wo", "rms_ffn_weight",
                "w1", "w2", "w3", "rms_final_weight", "wcls")


@dataclasses.dataclass
class Weights:
    """Weight tensors with run.c's shapes; matrices are (out, in) row-major, as `run.c:matmul` reads them."""
    token_embedding_table: np.ndarray   # (vocab, dim)
    rms_att_weight: np.ndarray          # (L, dim)
    wq: np.ndarray                      # (L, dim, dim)          = (L, n_heads*head_size, dim)
    wk: np.ndarray                      # (L, kv_dim, dim)
    wv: np.ndarray                      # (L, kv_dim, dim)
    wo: np.ndarray                      # (L, dim, dim)
    rms_ffn_weight: np.ndarray          # (L, dim)
    w1: np.ndarray                      # (L, hidden, dim)
    w2: np.ndarray                      # (L, dim, hidden)
    w3: np.ndarray                      # (L, hidden, dim)
    rms_final_weight: np.ndarray        # (dim,)
    wcls: np.ndarray                    # (vocab, dim); aliases the embedding when shared

    def astype(self, dtype) -> "Weights":
        return Weights(**{n: getattr(self, n).astype(dtype) for n in WEIGHT_NAMES})


def expected_file_bytes(cfg: Config) -> int:
    """Byte size implied by the header, following `run.c:memory_map_weights` (incl. the skipped RoPE tables)."""
    hs, L = cfg.head_size, cfg.n_layers
    n = (cfg.vocab_size * cfg.dim + L * cfg.dim + L * cfg.dim * cfg.n_heads * hs
         + 2 * L * cfg.dim * cfg.n_kv_heads * hs + L * cfg.n_heads * hs * cfg.dim + L * cfg.dim
         + 3 * L * cfg.dim * cfg.hidden_dim + cfg.dim + cfg.seq_len * hs)  # 2 × seq_len·hs/2 legacy tables
    if not cfg.shared_classifier:
        n += cfg.vocab_size * cfg.dim
    return 28 + 4 * n


def load_checkpoint(path) -> tuple[Config, Weights]:
    """Read a llama2.c v0 checkpoint (`run.c:read_checkpoint` + `memory_map_weights`) as float32 arrays."""
    raw = Path(path).read_bytes()
    dim, hidden, L, nh, nkv, vocab, seq = struct.unpack_from("<7i", raw, 0)
    cfg = Config(dim, hidden, L, nh, nkv, abs(vocab), seq, shared_classifier=vocab > 0)
    if cfg.dim % cfg.n_heads or cfg.n_heads % cfg.n_kv_heads:
        raise ValueError(f"{path}: inconsistent head counts {cfg}")
    if len(raw) != expected_file_bytes(cfg):
        raise ValueError(f"{path}: {len(raw)} bytes, header implies {expected_file_bytes(cfg)}")
    data = np.frombuffer(raw, dtype="<f4", offset=28)
    hs, kv = cfg.head_size, cfg.kv_dim
    off = 0

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        a = data[off:off + n].reshape(shape)
        off += n
        return a.astype(np.float32)          # native-endian, writable copy

    emb = take(cfg.vocab_size, dim)
    rms_att = take(L, dim)
    wq = take(L, nh * hs, dim)
    wk = take(L, kv, dim)
    wv = take(L, kv, dim)
    wo = take(L, dim, nh * hs)
    rms_ffn = take(L, dim)
    w1 = take(L, hidden, dim)
    w2 = take(L, dim, hidden)
    w3 = take(L, hidden, dim)
    rms_final = take(dim)
    off += seq * hs                           # legacy freq_cis_real + freq_cis_imag (run.c skips them too)
    wcls = emb if cfg.shared_classifier else take(cfg.vocab_size, dim)
    return cfg, Weights(emb, rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3, rms_final, wcls)


# ------------------------------------------------------------------------------------------------ tokenizer
_BYTE_TOKEN_RE = re.compile(rb"<0x([0-9A-Fa-f]{1,2})")


class Tokenizer:
    """llama2.c BPE tokenizer (`run.c` build_tokenizer / encode / decode / safe_printf), byte-exact."""

    def __init__(self, path, vocab_size: int):
        raw = Path(path).read_bytes()
        (self.max_token_length,) = struct.unpack_from("<i", raw, 0)
        off = 4
        self.vocab: list[bytes] = []
        scores = []
        for _ in range(vocab_size):
            score, n = struct.unpack_from("<fi", raw, off)
            off += 8
            self.vocab.append(raw[off:off + n])
            off += n
            scores.append(score)
        if off != len(raw):
            raise ValueError(f"{path}: {len(raw) - off} trailing bytes after {vocab_size} tokens")
        self.scores = np.asarray(scores, dtype=np.float32)
        # run.c finds strings with bsearch over the sorted vocab; with no duplicate strings (asserted here) a dict
        # lookup returns the same id.
        self.lookup = {s: i for i, s in enumerate(self.vocab)}
        if len(self.lookup) != len(self.vocab):
            raise ValueError(f"{path}: duplicate vocab strings; bsearch semantics would be ambiguous")

    def decode(self, prev_token: int, token: int) -> bytes:
        """`run.c:decode`: strip one leading space after BOS; map `<0xXX>` pieces to the raw byte.

        sscanf("<0x%02hhX>") succeeds as soon as 1-2 hex digits follow "<0x"; the regex mirrors that (all byte tokens
        of the two shipped vocabularies have the exact `<0xXX>` form, so the corner cases never arise).
        """
        piece = self.vocab[token]
        if prev_token == BOS and piece[:1] == b" ":
            piece = piece[1:]
        m = _BYTE_TOKEN_RE.match(piece)
        if m:
            piece = bytes([int(m.group(1), 16)])
        return piece

    @staticmethod
    def safe_piece(piece: bytes) -> bytes:
        """`run.c:safe_printf`: drop empty pieces and single bytes that are neither printable nor whitespace (C locale:
        isprint = 0x20..0x7e, isspace = 0x20 and 0x09..0x0d)."""
        if not piece:
            return b""
        if len(piece) == 1:
            b = piece[0]
            if not (0x20 <= b <= 0x7E or 0x09 <= b <= 0x0D):
                return b""
        return piece

    def render(self, tokens: Sequence[int]) -> bytes:
        """The stdout of `run.c:generate` for a token sequence (first token = the one fed at pos 0): every
        `decode(token, next)` through `safe_printf`, then the final newline. Stops at a BOS, like generate()."""
        out = []
        for prev, nxt in zip(tokens[:-1], tokens[1:]):
            if nxt == BOS:
                break
            out.append(self.safe_piece(self.decode(prev, nxt)))
        return b"".join(out) + b"\n"

    def encode(self, text: bytes, bos: bool = True, eos: bool = False) -> list[int]:
        """`run.c:encode`, byte for byte: optional BOS, dummy-prefix " " token when text is non-empty, UTF-8 code
        points looked up with byte fallback (+3), then greedy merges of the highest-scoring adjacent pair (first pair
        wins ties, because run.c compares with a strict `>`), then optional EOS."""
        tokens: list[int] = [BOS] if bos else []
        if text:
            tokens.append(self.lookup[b" "])
        buf = bytearray()
        n = len(text)
        for i in range(n):
            c = text[i]
            if (c & 0xC0) != 0x80:     # not a continuation byte -> new code point
                buf.clear()
            buf.append(c)
            nxt = text[i + 1] if i + 1 < n else 0
            if (nxt & 0xC0) == 0x80 and len(buf) < 4:
                continue
            tid = self.lookup.get(bytes(buf))
            if tid is not None:
                tokens.append(tid)
            else:
                tokens.extend(b + 3 for b in buf)   # byte_fallback: ids 3..258 are <0x00>..<0xFF>
            buf.clear()
        while True:
            best_score, best_id, best_idx = np.float32(-1e10), -1, -1
            for i in range(len(tokens) - 1):
                tid = self.lookup.get(self.vocab[tokens[i]] + self.vocab[tokens[i + 1]])
                if tid is not None and self.scores[tid] > best_score:
                    best_score, best_id, best_idx = self.scores[tid], tid, i
            if best_idx == -1:
                break
            tokens[best_idx:best_idx + 2] = [best_id]
        if eos:
            tokens.append(2)
        return tokens


# ------------------------------------------------------------------------------------------------ forward pass
Tap = Callable[..., None]   # tap(op, layer, pos, **arrays): receives copies of each op's inputs/outputs


class Llama2cRef:
    """Single-sequence forward pass mirroring `run.c:forward`, with a KV cache (layer, seq_len, kv_dim)."""

    def __init__(self, cfg: Config, weights: Weights, dtype=np.float32):
        self.cfg = cfg
        self.dt = np.dtype(dtype)
        self.w = weights if weights.wq.dtype == self.dt else weights.astype(self.dt)
        self.key_cache = np.zeros((cfg.n_layers, cfg.seq_len, cfg.kv_dim), self.dt)
        self.value_cache = np.zeros_like(self.key_cache)
        c = self.dt.type
        self._eps = c(1e-5)                                     # run.c rmsnorm: ss += 1e-5f
        hs = cfg.head_size
        i = np.arange(0, cfg.dim, 2)
        head_dim = (i % hs).astype(self.dt)                     # run.c: int head_dim = i % head_size
        self._freq = c(1.0) / np.power(c(10000.0), head_dim / c(hs))   # 1.0f / powf(10000.0f, hd / (float)hs)
        self._sqrt_hs = np.sqrt(c(hs))                          # sqrtf(head_size)
        self._kv_head = np.arange(cfg.n_heads) // cfg.kv_mul   # run.c: (h / kv_mul) selects the KV head

    def reset(self):
        self.key_cache[:] = 0
        self.value_cache[:] = 0

    # element-wise blocks (public so fixtures and tests can call them on fixture inputs)
    def rmsnorm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """run.c rmsnorm: o = w * (x / sqrt(mean(x^2) + 1e-5)), computed as ss = 1/sqrtf(ss/size + 1e-5f)."""
        c = self.dt.type
        ss = c(np.dot(x, x)) / c(x.shape[-1]) + self._eps
        ss = c(1.0) / np.sqrt(ss)
        return weight * (ss * x)

    def rope(self, q: np.ndarray, k: np.ndarray, pos: int) -> tuple[np.ndarray, np.ndarray]:
        """run.c RoPE: rotate each (even, odd) pair by angle pos·freq(i % head_size); k only for i < kv_dim."""
        val = self.dt.type(pos) * self._freq
        fcr, fci = np.cos(val), np.sin(val)
        q, k = q.copy(), k.copy()
        q0, q1 = q[0::2].copy(), q[1::2].copy()
        q[0::2], q[1::2] = q0 * fcr - q1 * fci, q0 * fci + q1 * fcr
        nk = self.cfg.kv_dim // 2
        k0, k1 = k[0::2].copy(), k[1::2].copy()
        k[0::2], k[1::2] = k0 * fcr[:nk] - k1 * fci[:nk], k0 * fci[:nk] + k1 * fcr[:nk]
        return q, k

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """run.c softmax over the last axis: subtract the max, exp, divide by the sum."""
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def attention(self, q: np.ndarray, kc: np.ndarray, vc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """GQA decode attention of one query position over kc/vc (T, kv_dim): returns (out (dim,), scores (H, T),
        probs (H, T)). Query head h reads KV head h // kv_mul; scores = q·k / sqrtf(head_size)."""
        cfg = self.cfg
        T = kc.shape[0]
        qh = q.reshape(cfg.n_heads, cfg.head_size)
        K = kc.reshape(T, cfg.n_kv_heads, cfg.head_size)[:, self._kv_head, :]    # (T, H, hs)
        V = vc.reshape(T, cfg.n_kv_heads, cfg.head_size)[:, self._kv_head, :]
        scores = np.einsum("hd,thd->ht", qh, K) / self._sqrt_hs
        probs = self.softmax(scores)
        out = np.einsum("ht,thd->hd", probs, V).reshape(cfg.dim)
        return out, scores, probs

    def silu_gate(self, hb: np.ndarray, hb2: np.ndarray) -> np.ndarray:
        """run.c SwiGLU: val *= 1/(1+expf(-val)); val *= hb2."""
        c = self.dt.type
        return hb * (c(1.0) / (c(1.0) + np.exp(-hb))) * hb2

    def forward(self, token: int, pos: int, tap: Tap | None = None) -> np.ndarray:
        """Logits for `token` at position `pos`; writes K/V of this position into the cache (run.c:forward)."""
        cfg, w = self.cfg, self.w
        if not 0 <= pos < cfg.seq_len:
            raise ValueError(f"pos {pos} outside [0, {cfg.seq_len})")
        T = tap or (lambda *a, **k: None)
        x = w.token_embedding_table[token].copy()
        T("embed", -1, pos, token=np.asarray([token], np.int32), x=x.copy())
        for l in range(cfg.n_layers):
            xb = self.rmsnorm(x, w.rms_att_weight[l])
            T("rmsnorm_att", l, pos, x=x.copy(), weight=w.rms_att_weight[l], out=xb)
            q = w.wq[l] @ xb
            k = w.wk[l] @ xb
            v = w.wv[l] @ xb
            T("matmul_q", l, pos, x=xb, out=q)
            T("matmul_k", l, pos, x=xb, out=k)
            T("matmul_v", l, pos, x=xb, out=v)
            q_r, k_r = self.rope(q, k, pos)
            T("rope", l, pos, q_in=q, k_in=k, q_out=q_r, k_out=k_r)
            self.key_cache[l, pos] = k_r
            self.value_cache[l, pos] = v
            T("kv_write", l, pos, k=k_r, v=v)
            att_out, scores, probs = self.attention(q_r, self.key_cache[l, :pos + 1], self.value_cache[l, :pos + 1])
            T("attention", l, pos, q=q_r, k_cache=self.key_cache[l, :pos + 1].copy(),
              v_cache=self.value_cache[l, :pos + 1].copy(), scores=scores, probs=probs, out=att_out)
            xb2 = w.wo[l] @ att_out
            T("matmul_o", l, pos, x=att_out, out=xb2)
            x_new = x + xb2
            T("residual_att", l, pos, x=x.copy(), branch=xb2, out=x_new)
            x = x_new
            xb = self.rmsnorm(x, w.rms_ffn_weight[l])
            T("rmsnorm_ffn", l, pos, x=x.copy(), weight=w.rms_ffn_weight[l], out=xb)
            hb = w.w1[l] @ xb
            hb2 = w.w3[l] @ xb
            T("matmul_w1", l, pos, x=xb, out=hb)
            T("matmul_w3", l, pos, x=xb, out=hb2)
            g = self.silu_gate(hb, hb2)
            T("silu_gate", l, pos, hb=hb, hb2=hb2, out=g)
            xb = w.w2[l] @ g
            T("matmul_w2", l, pos, x=g, out=xb)
            x_new = x + xb
            T("residual_ffn", l, pos, x=x.copy(), branch=xb, out=x_new)
            x = x_new
        xf = self.rmsnorm(x, w.rms_final_weight)
        T("rmsnorm_final", -1, pos, x=x.copy(), weight=w.rms_final_weight, out=xf)
        logits = w.wcls @ xf
        T("classifier", -1, pos, x=xf, out=logits)
        return logits


# ------------------------------------------------------------------------------------------------ golden data
def read_golden_tokens(path) -> tuple[list[int], list[int]]:
    """S8 `golden_tokens_*.txt` -> (token sequence = step-0 token followed by every `next`, forced flag per step)."""
    seq, forced = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        pos, tok, nxt, f = (int(v) for v in line.split())
        if pos != len(forced):
            raise ValueError(f"{path}: non-contiguous pos {pos}")
        if pos == 0:
            seq.append(tok)
        elif seq[-1] != tok:
            raise ValueError(f"{path}: chain broken at pos {pos}")
        seq.append(nxt)
        forced.append(f)
    return seq, forced


def load_model(name: str, root: Path = PS_ROOT, dtype=np.float32):
    m = MODELS[name]
    cfg, w = load_checkpoint(root / m["ckpt"])
    tok = Tokenizer(root / m["tok"], cfg.vocab_size)
    return cfg, w, tok


def top2(logits: np.ndarray) -> tuple[int, float, float]:
    """(argmax with run.c's first-index tie rule, top-1 logit, top-2 logit)."""
    i = int(np.argmax(logits))
    l1 = float(logits[i])
    rest = np.delete(logits, i)
    return i, l1, float(rest.max())


def rel_margin(l1: float, l2: float) -> float:
    """Relative top-1/top-2 margin |l1 - l2| / |l1| (near-tie rule); inf when l1 == 0 and l1 != l2."""
    if l1 == 0.0:
        return 0.0 if l2 == 0.0 else float("inf")
    return abs(l1 - l2) / abs(l1)


def teacher_forced(cfg: Config, w: Weights, seq: Sequence[int], forced: Sequence[int]) -> list[dict]:
    """Feed the golden sequence; per position record the float32 argmax, top-2 margin, golden next token and the
    max-abs difference of the float32 logits to a float64 run of the same code on the same tokens."""
    r32 = Llama2cRef(cfg, w, np.float32)
    r64 = Llama2cRef(cfg, w, np.float64)
    rows = []
    for pos in range(len(forced)):
        l32 = r32.forward(seq[pos], pos)
        l64 = r64.forward(seq[pos], pos)
        am, l1, l2 = top2(l32)
        am64 = int(np.argmax(l64))
        rows.append(dict(pos=pos, token=seq[pos], golden_next=seq[pos + 1], forced=forced[pos], ref_argmax=am,
                         ref_argmax_f64=am64, top1=l1, top2=l2, rel_margin=rel_margin(l1, l2),
                         agree=int(am == seq[pos + 1]),
                         max_abs_f32_vs_f64=float(np.max(np.abs(l32.astype(np.float64) - l64)))))
    return rows


def free_running(cfg: Config, w: Weights, prompt_tokens: Sequence[int], steps: int) -> list[int]:
    """`run.c:generate` with temperature 0: force the prompt tokens, then greedy argmax; stop at BOS or `steps`."""
    ref = Llama2cRef(cfg, w, np.float32)
    seq = [prompt_tokens[0]]
    for pos in range(steps):
        logits = ref.forward(seq[-1], pos)
        nxt = prompt_tokens[pos + 1] if pos < len(prompt_tokens) - 1 else int(np.argmax(logits))
        seq.append(nxt)
        if nxt == BOS:
            break
    return seq


def evaluate_teacher_forced(rows: Iterable[dict]) -> dict:
    """Apply the pass rule to non-forced positions (forced positions take the next token from the prompt, so
    the model's argmax there is not a prediction run.c made). Returns a summary dict with `passed`."""
    rows = list(rows)
    free = [r for r in rows if not r["forced"]]
    dis = [r for r in free if not r["agree"]]
    bad = [r for r in dis if r["rel_margin"] > NEAR_TIE_REL_MARGIN]
    return dict(positions=len(rows), compared=len(free), disagreements=len(dis),
                disagreements_beyond_near_tie=len(bad), first_near_tie_pos=(dis[0]["pos"] if dis else None),
                min_rel_margin_compared=min((r["rel_margin"] for r in free), default=None),
                max_abs_f32_vs_f64=max((r["max_abs_f32_vs_f64"] for r in rows), default=0.0),
                passed=not bad and max((r["max_abs_f32_vs_f64"] for r in rows), default=0.0) < F64_MAX_ABS_TOL)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ CLI
def _cmd_verify(args) -> int:
    """Teacher-forced + free-running + float64 check against the llama2.c golden; writes results/<run>/."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.models:
        cfg, w, tok = load_model(name)
        m = MODELS[name]
        golden_text = (PS_ROOT / m["golden"]).read_bytes()
        seq, forced = read_golden_tokens(PS_ROOT / m["golden_tokens"])
        rows = teacher_forced(cfg, w, seq, forced)
        with open(out_dir / f"teacher_forced_{name}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            for r in rows:
                wr.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})
        ev = evaluate_teacher_forced(rows)
        prompt = tok.encode(GOLDEN_PROMPT)
        fr = free_running(cfg, w, prompt, len(forced))
        n_match = next((i for i, (a, b) in enumerate(zip(fr, seq)) if a != b), min(len(fr), len(seq)))
        enc = tok.encode(golden_text.rstrip(b"\n"))
        (out_dir / f"encoded_tokens_{name}.txt").write_text(
            f"# run.c-equivalent encode(golden text without the final newline, bos=1) of {m['golden']}\n"
            f"# decodes back to the identical text: {tok.render(enc) == golden_text}; "
            f"equal to the true golden ids: {enc == seq}\n" + " ".join(map(str, enc)) + "\n")
        ev.update(model=name, config=dataclasses.asdict(cfg), prompt_tokens=prompt,
                  prompt_matches_golden=prompt == seq[:len(prompt)],
                  golden_ids_decode_to_golden_text=tok.render(seq) == golden_text,
                  encode_roundtrip_identical=tok.render(enc) == golden_text,
                  encoded_equals_true_ids=enc == seq, encoded_len=len(enc), true_len=len(seq),
                  free_running_tokens_matching=n_match, free_running_text_identical=tok.render(fr) == golden_text,
                  checkpoint_sha256=sha256_file(PS_ROOT / m["ckpt"]))
        summary[name] = ev
        print(json.dumps({k: v for k, v in ev.items() if k != "config"}))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0 if all(v["passed"] and v["free_running_text_identical"] for v in summary.values()) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="teacher-forced/free-running/float64 checks vs the llama2.c golden")
    v.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    v.add_argument("--out", default=str(PS_ROOT / "results" / "P2.1"))
    args = p.parse_args(argv)
    return _cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
