"""NumPy reference of llama2.c `runq.c` Q8_0 arithmetic (methodology §18.5).

What this file is
-----------------
The golden model for the Q8_0 GEMV kernels (kernels/gemv_q80), the activation quantizer kernel `quantize_q80`, the host
C++ quantizer (kernels/common/ps_q80.h) and `ps_llama -q q80`. It mirrors llama2.c's `runq.c` [cand:D15] op for op:

* `quantize` (runq.c:145-171): per group of GS values, wmax = max |x|, scale = wmax / 127.0f (float32), then
  q = (int8_t) round(x / scale), with the division in float32 and `round` the C double `round` (half away from zero).
  The double `round` of a float32 value is exact, so it equals `roundf` of the same value (the device uses musl roundf).
  A group with scale 0 (all zeros): runq.c divides by zero (NaN) and converts it to int8 (undefined behaviour in C; 0 on
  x86-64 after the truncation). Here q = 0 for such a group, which is also what the device and the host quantizer do.
* `dequantize` (runq.c:139-143): x = (float) q * s[i / GS] with a float32 product (the int8 is promoted to int, then
  converted to float).
* `matmul` (runq.c:317-340): per output row, per group j: ival = sum_k xq[k] * wq[k] in int32 (exact: |ival| <=
  GS * 127^2 = 516,128 for GS = 32 and 1,032,256 for GS = 64, both < 2^24, so float(ival) is exact too), then
  val += (float) ival * ws[g] * xs[g]  evaluated left to right, i.e. val = val + ((float(ival) * ws) * xs), every
  operation a separate float32 rounding (no fused multiply-add; the device kernels are compiled with fp-contract off
  and the host driver's comparison loop uses the same order). `matmul_q80` below is vectorised over rows but keeps this
  order exactly, so it is BIT-identical to runq.c's loop for any row.

Error model (modeling choice, also kernels/gemv_q80/main.cpp). For y_i = sum_g t_g, t_g = float(ival_g) * ws_g * xs_g,
the int32 group sums are exact; the remaining float32 work is two products per group and G = n / GS additions. A term t_g
passes through at most G + 2 roundings (2 products + the add into the running sum and the later adds), so
|fl(y_i) - y_i| <= gamma_{G+2} S_i with S_i = sum_g |t_g| and gamma_m = m u / (1 - m u), u = 2^-24. Two correct
implementations with different summation orders differ by <= 2 gamma_{G+2} S_i; the earlier tolerance is
max(1e-4, 2 gamma_{n/GS}), and 2 gamma_{G+2} is used here because it is the exact bound (it is >= the plan's term by 4 u).

KV8 (methodology §18.9; `Llama2cRefKV8`, `python -B -m model.ref_q80 --kv8 ...`): the same forward pass with an int8 KV cache. Per
(position, KV head) the rotated K (and the V) vector of head_size values is quantized with `quantize_q80(x, gs=head_size)` (runq.c's rule, group = the
head: scale = max|x| / 127.0f, q = round(x / scale)) and the attention reads the DEQUANTIZED values `q * scale` (float32 product), which is what the device
kernels `rope_kv_tab_q8` / `attn_gqa_kv8` compute. K is rotated with the angle table (`rope_table_np`: correctly rounded float32 cos / sin of the
float32 angle p * freq, which is what the device table holds except for isolated near-tie neighbours) in NON-fused float32 operations, as the device TU
(fp contract off). `--golden` free-runs this model at temperature 0 for the golden token file `results/<run>/golden_tokens_15M_q80_kv8.txt` (same line
format as the golden files).

Also here: the host RNG port (`Rng`, xorshift32 of kernels/common/ps_common.h) so that synthetic weights and activations
are reproduced exactly, the v2 checkpoint reader (`load_q80_checkpoint`, runq.c `read_checkpoint` /
`memory_map_weights`), and a Q8_0 forward pass (`Llama2cRefQ80`, runq.c `forward`) used to make the logits reference of
`ps_llama -q q80 -r` (`python -B -m model.ref_q80 --positions N`).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS))
from model import ref_llama2c as ref  # noqa: E402

DEFAULT_Q80 = PS / "build" / "models" / "stories15M_q80.bin"   # export (see results/<run>/README.md: not in data/)
DEFAULT_OUT = PS / "build" / "fixtures" / "ep2"
U = 2.0 ** -24
F32 = np.float32


def default_gs(n: int, gs: int = 64) -> int:
    """export.py's rule (`while dim % group_size != 0: group_size //= 2`, export.py:193-195) applied to the reduction
    length n. The kernels' `-G` option defaults to this; stories15M uses it with dim = 288 (-> 32), not with its
    hidden_dim (768 % 64 == 0), so stories15M shapes always pass -G 32 explicitly."""
    while n % gs != 0:
        gs //= 2
    return gs


# ------------------------------------------------------------------------------------------------ quantizer
def _round_half_away(v: np.ndarray) -> np.ndarray:
    """C `round()` on float32 values, evaluated in float64 (exact for float32 inputs of any magnitude < 2^52)."""
    d = v.astype(np.float64)
    return np.trunc(d + np.copysign(0.5, d))


def quantize_q80(x: np.ndarray, gs: int) -> tuple[np.ndarray, np.ndarray]:
    """runq.c `quantize`: x (..., n) float32 -> (q int8 (..., n), s float32 (..., n // gs))."""
    x = np.asarray(x, dtype=F32)
    n = x.shape[-1]
    if n % gs:
        raise ValueError(f"n={n} is not a multiple of GS={gs}")
    g = x.reshape(*x.shape[:-1], n // gs, gs)
    wmax = np.abs(g).max(axis=-1)                                 # float32; `if (val > wmax)` keeps the max
    scale = (wmax / F32(127.0)).astype(F32)                       # float32 division, as `wmax / Q_MAX` (Q_MAX = 127.0f)
    with np.errstate(divide="ignore", invalid="ignore"):
        qv = (g / scale[..., None]).astype(F32)                   # float32 division x / scale
    qr = _round_half_away(qv)
    qr = np.where(scale[..., None] == 0, 0.0, qr)                 # all-zero group: q = 0 (see module doc)
    q = qr.astype(np.int64).astype(np.int8)                       # (int8_t) conversion; |qr| <= 127 for finite groups
    return q.reshape(x.shape), scale


def dequantize_q80(q: np.ndarray, s: np.ndarray, gs: int) -> np.ndarray:
    """runq.c `dequantize`: x = q * s[i / GS] (float32 product)."""
    n = q.shape[-1]
    g = q.reshape(*q.shape[:-1], n // gs, gs).astype(F32)
    return (g * s[..., None]).astype(F32).reshape(q.shape)


def matmul_q80(wq: np.ndarray, ws: np.ndarray, xq: np.ndarray, xs: np.ndarray, gs: int) -> np.ndarray:
    """runq.c `matmul`: W (M, N) int8 with scales (M, N // gs), x (N,) or (B, N) int8 with scales (.., N // gs) ->
    y (M,) or (B, M) float32, in runq's operation order (group loop sequential per row)."""
    squeeze = xq.ndim == 1
    xq2 = xq[None] if squeeze else xq
    xs2 = xs[None] if squeeze else xs
    M, N = wq.shape
    G = N // gs
    w3 = wq.reshape(M, G, gs).astype(np.int32)
    y = np.zeros((xq2.shape[0], M), F32)
    for b in range(xq2.shape[0]):
        x3 = xq2[b].reshape(G, gs).astype(np.int32)
        ival = np.einsum("mgk,gk->mg", w3, x3)                    # exact int32 group sums
        val = np.zeros(M, F32)
        for g in range(G):
            term = (ival[:, g].astype(F32) * ws[:, g]).astype(F32) * xs2[b, g]   # (float(ival) * ws) * xs
            val = (val + term.astype(F32)).astype(F32)
        y[b] = val
    return y[0] if squeeze else y


def err_scale(wq, ws, xq, xs, gs):
    """float64 exact value of the Q8_0 product and S_i = sum_g |t_g| for the error metric of ps_common.h."""
    M, N = wq.shape
    G = N // gs
    ival = np.einsum("mgk,gk->mg", wq.reshape(M, G, gs).astype(np.int64), xq.reshape(G, gs).astype(np.int64))
    t = ival.astype(np.float64) * ws.astype(np.float64) * xs.astype(np.float64)[None, :]
    return t.sum(axis=1), np.abs(t).sum(axis=1)


def tol_q80(G: int) -> float:
    """max(1e-4, 2 gamma_{G+2}) (module doc)."""
    m = G + 2
    return max(1e-4, 2 * m * U / (1 - m * U))


def fnv1a64(buf: bytes) -> int:
    """FNV-1a 64 over bytes (the host's digest of the quantized tensors; the float outputs use ps_common.h's
    fnv1a64_f32, which is the same function over the little-endian bytes of the floats)."""
    h = 0xCBF29CE484222325
    for b in buf:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h



# ------------------------------------------------------------------------------------------------ host RNG
def _xorshift_step(s: int) -> int:
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= s >> 17
    s ^= (s << 5) & 0xFFFFFFFF
    return s


def _gf2_apply(cols, x):
    r, i = 0, 0
    while x:
        if x & 1:
            r ^= cols[i]
        x >>= 1
        i += 1
    return r


def _jump_matrix(k: int):
    """Columns of T^k, T = one xorshift32 step (a linear map over GF(2)), by repeated squaring."""
    cols = [_xorshift_step(1 << i) for i in range(32)]
    res = [1 << i for i in range(32)]
    while k:
        if k & 1:
            res = [_gf2_apply(cols, c) for c in res]    # res = T_cols o res
        cols = [_gf2_apply(cols, c) for c in cols]
        k >>= 1
    return res


def rng_uniform(seed: int, n: int) -> np.ndarray:
    """Vectorised `ps::Rng(seed)`: the stream is split into S parallel blocks whose start states come from the GF(2)
    jump matrix, so the 8M values of a classifier matrix cost ~0.2 s instead of ~15 s. Identical to the scalar loop
    (`rng_uniform_scalar`, tested)."""
    if n <= 4096:
        return rng_uniform_scalar(seed, n)
    S = 4096
    L = -(-n // S)                                   # values per block
    s0 = seed if seed else 0x9E3779B9
    J = _jump_matrix(L)
    starts = np.empty(S, np.uint32)
    st = s0
    for b in range(S):
        starts[b] = st                               # state BEFORE the block's first step
        st = _gf2_apply(J, st)
    s = starts.astype(np.uint32)
    out = np.empty((L, S), np.uint32)
    for i in range(L):
        s ^= s << np.uint32(13)
        s ^= s >> np.uint32(17)
        s ^= s << np.uint32(5)
        out[i] = s >> np.uint32(8)
    flat = out.T.reshape(-1)[:n]                     # block b holds values b*L .. b*L + L - 1
    return (flat.astype(np.float64) * (1.0 / 16777216.0)).astype(F32) * F32(2.0) - F32(1.0)


def rng_uniform_scalar(seed: int, n: int) -> np.ndarray:
    """`ps::Rng(seed)` of kernels/common/ps_common.h (xorshift32, `uniform()` in [-1, 1)), n values as float32.
    `uniform() = (float)((next() >> 8) * (1.0 / 16777216.0)) * 2.0f - 1.0f`; the product is an exact double with 24
    significant bits, so the float cast is exact and the whole expression is exact in float32."""
    s = seed if seed else 0x9E3779B9
    out = np.empty(n, np.uint32)
    for i in range(n):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        out[i] = s >> 8
    return (out.astype(np.float64) * (1.0 / 16777216.0)).astype(F32) * F32(2.0) - F32(1.0)


# ------------------------------------------------------------------------------------------------ v2 checkpoint
@dataclasses.dataclass
class QT:
    """runq.c QuantizedTensor: q (rows, cols) int8 and s (rows, cols // gs) float32."""
    q: np.ndarray
    s: np.ndarray


@dataclasses.dataclass
class Q80Model:
    cfg: ref.Config
    gs: int
    rms_att: np.ndarray     # (L, dim) float32
    rms_ffn: np.ndarray
    rms_final: np.ndarray   # (dim,)
    tok: QT                 # (vocab, dim); also the classifier when shared
    wq: list                # per layer QT (dim, dim)
    wk: list
    wv: list
    wo: list
    w1: list
    w2: list
    w3: list
    wcls: QT
    sha256: str = ""


def load_q80_checkpoint(path) -> Q80Model:
    """runq.c read_checkpoint + memory_map_weights: header 256 B (magic 'ak42', version 2, 7 int32 config, 1 byte
    shared_classifier, int32 group size), fp32 rms weights (att, ffn, final), then Q8_0 tensors each as int8 values
    followed by their float32 scales: token table, wq, wk, wv, wo, w1, w2, w3 (one tensor per layer), [wcls]."""
    raw = Path(path).read_bytes()
    magic, version = struct.unpack_from("<Ii", raw, 0)
    if magic != 0x616B3432 or version != 2:
        raise ValueError(f"{path}: not a llama2.c v2 (Q8_0) file")
    dim, hidden, L, nh, nkv, vocab, seq = struct.unpack_from("<7i", raw, 8)
    shared = raw[36]
    (gs,) = struct.unpack_from("<i", raw, 37)
    cfg = ref.Config(dim, hidden, L, nh, nkv, vocab, seq, bool(shared))
    off = 256
    hs = cfg.head_size

    def f32(n):
        nonlocal off
        a = np.frombuffer(raw, "<f4", n, off).astype(F32)
        off += 4 * n
        return a

    rms_att = f32(L * dim).reshape(L, dim)
    rms_ffn = f32(L * dim).reshape(L, dim)
    rms_final = f32(dim)

    def qt(rows, cols):
        nonlocal off
        q = np.frombuffer(raw, np.int8, rows * cols, off).reshape(rows, cols).copy()
        off += rows * cols
        s = np.frombuffer(raw, "<f4", rows * cols // gs, off).astype(F32).reshape(rows, cols // gs)
        off += 4 * rows * cols // gs
        return QT(q, s)

    tok = qt(vocab, dim)
    mats = {}
    for name, (r, c) in (("wq", (nh * hs, dim)), ("wk", (cfg.kv_dim, dim)), ("wv", (cfg.kv_dim, dim)),
                         ("wo", (dim, nh * hs)), ("w1", (hidden, dim)), ("w2", (dim, hidden)), ("w3", (hidden, dim))):
        mats[name] = [qt(r, c) for _ in range(L)]
    wcls = tok if shared else qt(vocab, dim)
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return Q80Model(cfg, gs, rms_att, rms_ffn, rms_final, tok, **mats, wcls=wcls,
                    sha256=hashlib.sha256(raw).hexdigest())


class Llama2cRefQ80(ref.Llama2cRef):
    """runq.c `forward` in NumPy float32: every matmul is quantize(x) -> matmul_q80, the token embedding is the
    dequantized table (runq.c:203-205), the rest is run.c's float32 math (the reference blocks, inherited)."""

    def __init__(self, m: Q80Model):
        z = np.zeros(1, F32)       # placeholders: the inherited blocks only use the embedding table and the norms
        w = ref.Weights(dequantize_q80(m.tok.q, m.tok.s, m.gs), m.rms_att, z, z, z, z, m.rms_ffn, z, z, z,
                        m.rms_final, z)
        super().__init__(m.cfg, w)
        self.m = m

    def _mm(self, t: QT, x: np.ndarray) -> np.ndarray:
        xq, xs = quantize_q80(x, self.m.gs)
        return matmul_q80(t.q, t.s, xq, xs, self.m.gs)

    def _mm2(self, ts, x):          # one quantization shared by several matmuls (runq.c quantizes once per x)
        xq, xs = quantize_q80(x, self.m.gs)
        return [matmul_q80(t.q, t.s, xq, xs, self.m.gs) for t in ts]

    def _store_kv(self, l: int, pos: int, k_r: np.ndarray, v: np.ndarray) -> None:
        """runq.c / run.c: the rotated k and the v go into the cache as they are (float32). Llama2cRefKV8 overrides this."""
        self.key_cache[l, pos] = k_r
        self.value_cache[l, pos] = v

    def forward(self, token: int, pos: int, tap=None) -> np.ndarray:
        """`tap(name, layer, array)` (optional) receives each matmul input vector: xb_att, ao, xb_ffn, hb, xf."""
        cfg, m = self.cfg, self.m
        T = tap or (lambda *a: None)
        x = self.w.token_embedding_table[token].copy()
        for l in range(cfg.n_layers):
            xb = self.rmsnorm(x, m.rms_att[l])
            T("xb_att", l, xb.copy())
            q, k, v = self._mm2((m.wq[l], m.wk[l], m.wv[l]), xb)
            q_r, k_r = self.rope(q, k, pos)
            self._store_kv(l, pos, k_r, v)
            att_out, _, _ = self.attention(q_r, self.key_cache[l, :pos + 1], self.value_cache[l, :pos + 1])
            T("ao", l, att_out.copy())
            x = x + self._mm(m.wo[l], att_out)
            xb = self.rmsnorm(x, m.rms_ffn[l])
            T("xb_ffn", l, xb.copy())
            hb, hb2 = self._mm2((m.w1[l], m.w3[l]), xb)
            g = self.silu_gate(hb, hb2)
            T("hb", l, g.copy())
            x = x + self._mm(m.w2[l], g)
        xf = self.rmsnorm(x, m.rms_final)
        T("xf", -1, xf.copy())
        return self._mm(m.wcls, xf)


# ------------------------------------------------------------------------------------------------ KV8
def rope_table_np(hs: int, npos: int, theta: float = 10000.0, pos0: int = 0) -> np.ndarray:
    """angle table as the device holds it (kernels/rope/rope_table_ref.h): freq_j = 1.0f / (float) powl(theta, e), e = (float)(2 j) / (float) hs,
    val = (float) p * freq_j (float32), table[p, j] = ((float) cosl(val), (float) sinl(val)) correctly rounded; shape (npos, hs // 2, 2) float32."""
    L = np.longdouble
    half = hs // 2
    fr = np.empty(half, F32)
    for j in range(half):
        e = F32(2 * j) / F32(hs)
        fr[j] = F32(1) / F32(np.power(L(theta), L(float(e))))
    out = np.empty((npos, half, 2), F32)
    for p in range(npos):
        v = F32(pos0 + p) * fr
        for j in range(half):
            out[p, j, 0] = F32(np.cos(L(float(v[j]))))
            out[p, j, 1] = F32(np.sin(L(float(v[j]))))
    return out


def rope_rotate_f32(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """Rotate consecutive pairs of x (n,) by the table row cs (n // 2, 2), in NON-fused float32 operations: (x0 fcr - x1 fci, x0 fci + x1 fcr)."""
    x0, x1 = x[0::2].copy(), x[1::2].copy()
    fcr, fci = cs[:, 0], cs[:, 1]
    y = np.empty_like(x)
    y[0::2] = ((x0 * fcr).astype(F32) - (x1 * fci).astype(F32)).astype(F32)
    y[1::2] = ((x0 * fci).astype(F32) + (x1 * fcr).astype(F32)).astype(F32)
    return y


def kv8_head_quantize(x: np.ndarray, hs: int) -> tuple[np.ndarray, np.ndarray]:
    """x (n_heads * hs,) float32 -> (q int8 (n_heads * hs,), s float32 (n_heads,)): one group per head (runq.c quantize, GS = head_size)."""
    q, s = quantize_q80(x.reshape(-1, hs), hs)
    return q.reshape(-1), s.reshape(-1)


def kv8_dequantize(q: np.ndarray, s: np.ndarray, hs: int) -> np.ndarray:
    """the float32 operand the attention kernel uses: (float) q * scale per head."""
    return dequantize_q80(q.reshape(-1, hs), s.reshape(-1, 1), hs).reshape(-1)


class Llama2cRefKV8(Llama2cRefQ80):
    """Llama2cRefQ80 with the KV8 cache: K rotated with the table angles (non-fused float32), K and V quantized per (position, KV head) and read
    back dequantized. Everything else is the parent's (runq.c forward)."""

    def __init__(self, m: Q80Model):
        super().__init__(m)
        self._tab = rope_table_np(self.cfg.head_size, self.cfg.seq_len)
        self.kv_q = {}     # (layer, pos) -> (kq, ks, vq, vs), kept for tests

    def rope(self, q, k, pos):
        hs = self.cfg.head_size
        hh = hs // 2
        tab = self._tab[pos]
        # q: every head rotates with the same per-head table row; k only has kv_dim // 2 pairs
        qh = np.stack([rope_rotate_f32(q[h * hs:(h + 1) * hs], tab) for h in range(self.cfg.n_heads)]).reshape(-1)
        kh = np.stack([rope_rotate_f32(k[h * hs:(h + 1) * hs], tab) for h in range(self.cfg.n_kv_heads)]).reshape(-1)
        assert hh == tab.shape[0]
        return qh, kh

    def _store_kv(self, l, pos, k_r, v):
        hs = self.cfg.head_size
        kq, ks = kv8_head_quantize(k_r, hs)
        vq, vs = kv8_head_quantize(v, hs)
        self.kv_q[(l, pos)] = (kq, ks, vq, vs)
        self.key_cache[l, pos] = kv8_dequantize(kq, ks, hs)
        self.value_cache[l, pos] = kv8_dequantize(vq, vs, hs)


def make_golden_kv8(path: Path, steps: int, out: Path) -> Path:
    """Free-running greedy (temperature 0) generation of the KV8 model for `steps` positions, the prompt (forced tokens) taken runq golden
    file; written in the format of the golden files (`pos token next forced`)."""
    m = load_q80_checkpoint(path)
    seq0, forced0 = ref.read_golden_tokens(PS / "results" / "EP.2" / "golden_tokens_15M_q80.txt")
    P = 1
    while P - 1 < len(forced0) and forced0[P - 1]:
        P += 1
    prompt = seq0[:P]
    r = Llama2cRefKV8(m)
    lines, tok = [], prompt[0]
    for pos in range(steps):
        lg = r.forward(tok, pos)
        nxt = prompt[pos + 1] if pos + 1 < P else int(np.argmax(lg))   # run.c: first index of the maximum
        lines.append(f"{pos} {tok} {nxt} {1 if pos + 1 < P else 0}")
        tok = nxt
    hdr = ["# KV8 golden token ids, generated by `python -B -m model.ref_q80 --kv8 --golden`",
           f"# model: {path.name} sha256 {m.sha256}; NumPy float32 runq.c forward with int8 KV (per (position, KV head) scale, methodology §18.9), temperature 0, {steps} steps,",
           "# prompt tokens (forced) = those of results/EP.2/golden_tokens_15M_q80.txt (\"Once upon a time\")",
           "# columns: pos token next forced   (forced=1: next taken from the prompt; 0: greedy argmax)"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(hdr + lines) + "\n")
    return out


# ------------------------------------------------------------------------------------------------ logits reference
def make(path: Path, n: int, out: Path, kv8: bool = False, golden: Path | None = None) -> Path:
    """Teacher-forced Q8_0 logits over the llama2.c stories15M golden sequence (the format `ps_llama -r` reads)."""
    m = load_q80_checkpoint(path)
    seq, forced = ref.read_golden_tokens(PS / ref.MODELS["stories15M"]["golden_tokens"])
    if golden is not None:
        seq, forced = ref.read_golden_tokens(golden)
    if not 1 <= n <= min(len(forced), m.cfg.seq_len):
        raise SystemExit(f"--positions must be in [1, {min(len(forced), m.cfg.seq_len)}]")
    if golden is not None:
        seq, forced = ref.read_golden_tokens(golden)
    r = Llama2cRefKV8(m) if kv8 else Llama2cRefQ80(m)
    logits = np.empty((n, m.cfg.vocab_size), F32)
    rows = []
    for p in range(n):
        lg = r.forward(seq[p], p).astype(F32)
        logits[p] = lg
        am, l1, l2 = ref.top2(lg)
        rows.append(dict(pos=p, token=seq[p], golden_next=seq[p + 1], forced=forced[p], argmax=am,
                         rel_margin=ref.rel_margin(l1, l2)))
    out.mkdir(parents=True, exist_ok=True)
    tag = "q80_kv8" if kv8 else "q80"
    f = out / f"stories15M_{tag}_tf{n}.f32"
    f.write_bytes(logits.astype("<f4").tobytes())
    meta = dict(model=f"stories15M_{tag}", positions=n, vocab=m.cfg.vocab_size, gs=m.gs,
                dtype="float32 little endian [positions][vocab]",
                generator="model/ref_q80.py (Llama2cRefQ80: runq.c forward in NumPy float32, teacher-forced over "
                          "results/S8 golden_tokens_15M.txt)",
                checkpoint_sha256=m.sha256, sha256=hashlib.sha256(f.read_bytes()).hexdigest(), rows=rows)
    (out / f"stories15M_{tag}_tf{n}.json").write_text(json.dumps(meta, indent=1) + "\n")
    return f


def real_activations(path: Path, out: Path, pos: int = 7) -> list[Path]:
    """Real stories15M activations at position `pos` of the golden sequence, for the quantizer data-dependence check
   : x288_real.f32 = 4 vectors of dim 288 (attention-norm output of layers 0 and 5, ffn-norm output of
    layers 0 and 5), x768_real.f32 = the SwiGLU output (the w2 input) of layer 5."""
    m = load_q80_checkpoint(path)
    seq, _ = ref.read_golden_tokens(PS / ref.MODELS["stories15M"]["golden_tokens"])
    r = Llama2cRefQ80(m)
    got = {}
    for p in range(pos + 1):
        r.forward(seq[p], p, tap=(lambda n, l, a: got.__setitem__((n, l), a)) if p == pos else None)
    out.mkdir(parents=True, exist_ok=True)
    a288 = np.stack([got[("xb_att", 0)], got[("xb_att", 5)], got[("xb_ffn", 0)], got[("xb_ffn", 5)]]).astype("<f4")
    a768 = got[("hb", 5)].astype("<f4")[None]
    fs = []
    for name, arr in (("x288_real.f32", a288), ("x768_real.f32", a768)):
        (out / name).write_bytes(arr.tobytes())
        fs.append(out / name)
    return fs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, default=DEFAULT_Q80)
    ap.add_argument("--positions", type=int, default=0, help="write the teacher-forced logits reference (ps_llama -r)")
    ap.add_argument("--activations", action="store_true", help="write the real-activation files of the quantizer check")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--kv8", action="store_true", help="int8 KV cache (per (position, KV head) scale), table RoPE; with --positions / --golden")
    ap.add_argument("--golden", type=Path, default=None,
                    help="with --kv8 and no --positions: write the free-running golden token file here (--steps positions); with --positions: the teacher-forced "
                         "sequence is read from this file instead of the llama2.c golden")
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args(argv)
    if a.kv8 and a.golden is not None and not a.positions:
        f = make_golden_kv8(a.model, a.steps, a.golden)
        print(f"wrote {f}")
    if a.positions:
        f = make(a.model, a.positions, a.out, kv8=a.kv8, golden=a.golden)
        print(f"wrote {f} ({f.stat().st_size} bytes)")
    if a.activations:
        for f in real_activations(a.model, a.out):
            print(f"wrote {f} ({f.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
