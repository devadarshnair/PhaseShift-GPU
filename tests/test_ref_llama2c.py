"""Tests: NumPy llama2.c reference vs the run.c golden, tokenizer round trips, fixtures.

TEST rule: teacher-forced, at every (non-forced) position the reference argmax equals the next golden
token, or the relative top-1/top-2 margin |l1 - l2| / |l1| is <= 1e-4 (fp32 near-tie, reported); free-running
greedy reproduces the golden text up to the first such near-tie; per-step logits max-abs difference vs a float64
run < 1e-3. Forced positions (the prompt) are excluded from the argmax comparison because run.c takes the next
token from the prompt there, not from the logits.
"""
import ast
import json
import struct

import numpy as np
import pytest

from model import make_fixtures as mf
from model import ref_llama2c as ref

MODELS = list(ref.MODELS)


@pytest.fixture(scope="module", params=MODELS)
def loaded(request, ps_root):
    name = request.param
    cfg, w, tok = ref.load_model(name, ps_root)
    m = ref.MODELS[name]
    seq, forced = ref.read_golden_tokens(ps_root / m["golden_tokens"])
    golden = (ps_root / m["golden"]).read_bytes()
    return name, cfg, w, tok, seq, forced, golden


@pytest.fixture(scope="module")
def tf_rows(loaded):
    name, cfg, w, tok, seq, forced, golden = loaded
    return ref.teacher_forced(cfg, w, seq, forced)


# ------------------------------------------------------------------------------------------------ checkpoint
EXPECTED_HEADERS = {"stories260K": (64, 172, 5, 8, 4, 512, 512, True),
                    "stories15M": (288, 768, 6, 6, 6, 32000, 256, True)}


def test_checkpoint_header_and_size(loaded, ps_root):
    name, cfg, w, *_ = loaded
    assert (cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len,
            cfg.shared_classifier) == EXPECTED_HEADERS[name]
    assert (ps_root / ref.MODELS[name]["ckpt"]).stat().st_size == ref.expected_file_bytes(cfg)
    assert w.wq.shape == (cfg.n_layers, cfg.dim, cfg.dim)
    assert w.wk.shape == (cfg.n_layers, cfg.kv_dim, cfg.dim)
    assert w.w2.shape == (cfg.n_layers, cfg.dim, cfg.hidden_dim)
    assert w.wcls is w.token_embedding_table          # shared classifier (positive vocab_size)


def _tiny_ckpt(path, vocab_sign):
    """A random v0 checkpoint: dim 8, hidden 12, 2 layers, 4 heads, 2 KV heads, vocab 16, seq 6."""
    dim, hid, L, nh, nkv, V, S = 8, 12, 2, 4, 2, 16, 6
    hs, kv = dim // nh, dim * nkv // nh
    n = V * dim + L * dim + L * dim * dim + 2 * L * kv * dim + L * dim * dim + L * dim + 3 * L * dim * hid + dim + S * hs
    if vocab_sign < 0:
        n += V * dim
    rng = np.random.default_rng(1)
    vals = rng.standard_normal(n).astype("<f4")
    path.write_bytes(struct.pack("<7i", dim, hid, L, nh, nkv, vocab_sign * V, S) + vals.tobytes())
    return vals, V * dim


def test_unshared_classifier_negative_vocab(tmp_path):
    p = tmp_path / "tiny.bin"
    vals, nemb = _tiny_ckpt(p, -1)
    cfg, w = ref.load_checkpoint(p)
    assert cfg.vocab_size == 16 and not cfg.shared_classifier
    assert w.wcls is not w.token_embedding_table
    np.testing.assert_array_equal(w.wcls.ravel(), vals[-nemb:])        # wcls is the trailing tensor
    np.testing.assert_array_equal(w.token_embedding_table.ravel(), vals[:nemb])
    r = ref.Llama2cRef(cfg, w)
    for pos in range(cfg.seq_len):
        logits = r.forward(pos % cfg.vocab_size, pos)
        assert logits.shape == (16,) and logits.dtype == np.float32 and np.isfinite(logits).all()


def test_truncated_checkpoint_rejected(tmp_path):
    p = tmp_path / "tiny.bin"
    _tiny_ckpt(p, 1)
    p.write_bytes(p.read_bytes()[:-4])
    with pytest.raises(ValueError, match="header implies"):
        ref.load_checkpoint(p)


# ------------------------------------------------------------------------------------------------ blocks vs run.c loops
def _runc_loops(cfg, w, tokens):
    """A literal float32 transcription of run.c:forward's scalar loops (sequential sums), for a few positions."""
    f = np.float32
    L, dim, hs, kv = cfg.n_layers, cfg.dim, cfg.head_size, cfg.kv_dim
    kc = np.zeros((L, len(tokens), kv), f)
    vc = np.zeros_like(kc)

    def matmul(W, x):
        out = np.zeros(W.shape[0], f)
        for i in range(W.shape[0]):
            val = f(0)
            for j in range(W.shape[1]):
                val = f(val + W[i, j] * x[j])
            out[i] = val
        return out

    def rmsnorm(x, wt):
        ss = f(0)
        for v in x:
            ss = f(ss + v * v)
        ss = f(f(ss / f(len(x))) + f(1e-5))
        ss = f(f(1) / np.sqrt(ss))
        return (wt * (ss * x)).astype(f)

    outs = []
    for pos, tok in enumerate(tokens):
        x = w.token_embedding_table[tok].copy()
        for l in range(L):
            xb = rmsnorm(x, w.rms_att_weight[l])
            q, k, v = matmul(w.wq[l], xb), matmul(w.wk[l], xb), matmul(w.wv[l], xb)
            for i in range(0, dim, 2):
                freq = f(1) / np.power(f(10000), f(i % hs) / f(hs))
                val = f(pos) * freq
                fcr, fci = np.cos(val), np.sin(val)
                for vec in ((q, k) if i < kv else (q,)):
                    v0, v1 = vec[i], vec[i + 1]
                    vec[i], vec[i + 1] = v0 * fcr - v1 * fci, v0 * fci + v1 * fcr
            kc[l, pos], vc[l, pos] = k, v
            xb = np.zeros(dim, f)
            for h in range(cfg.n_heads):
                kh = h // cfg.kv_mul
                att = np.zeros(pos + 1, f)
                for t in range(pos + 1):
                    s = f(0)
                    for i in range(hs):
                        s = f(s + q[h * hs + i] * kc[l, t, kh * hs + i])
                    att[t] = s / np.sqrt(f(hs))
                att = np.exp(att - att.max())
                att = att / att.sum(dtype=f)
                for t in range(pos + 1):
                    xb[h * hs:(h + 1) * hs] += att[t] * vc[l, t, kh * hs:(kh + 1) * hs]
            x = x + matmul(w.wo[l], xb)
            xb = rmsnorm(x, w.rms_ffn_weight[l])
            hb, hb2 = matmul(w.w1[l], xb), matmul(w.w3[l], xb)
            hb = hb * (f(1) / (f(1) + np.exp(-hb))) * hb2
            x = x + matmul(w.w2[l], hb)
        x = rmsnorm(x, w.rms_final_weight)
        outs.append(matmul(w.wcls, x))
    return outs


def test_forward_matches_literal_runc_loops(ps_root):
    """The vectorised reference equals a literal transcription of run.c's loops (stories260K, 4 positions) to fp32
    summation-order noise; this pins every formula (RoPE frequency, GQA head mapping, rmsnorm eps, SiLU)."""
    cfg, w, _ = ref.load_model("stories260K", ps_root)
    tokens = [1, 403, 407, 261]
    loops = _runc_loops(cfg, w, tokens)
    r = ref.Llama2cRef(cfg, w)
    for pos, tok in enumerate(tokens):
        lg = r.forward(tok, pos)
        np.testing.assert_allclose(lg, loops[pos], rtol=0, atol=2e-5)
        assert int(np.argmax(lg)) == int(np.argmax(loops[pos]))


# ------------------------------------------------------------------------------------------------ tokenizer
def test_golden_ids_decode_to_golden_text(loaded):
    """the recorded token ids, decoded with run.c's decode + safe_printf, give the
    golden stdout byte for byte."""
    name, cfg, w, tok, seq, forced, golden = loaded
    assert len(seq) == 201 and len(forced) == 200
    assert tok.render(seq) == golden


def test_prompt_encoding_matches_golden_prefix(loaded):
    name, cfg, w, tok, seq, forced, golden = loaded
    prompt = tok.encode(ref.GOLDEN_PROMPT)
    n_forced = sum(forced)
    assert forced[:n_forced] == [1] * n_forced and not any(forced[n_forced:])
    assert prompt == seq[:n_forced + 1]


def test_encode_golden_text_roundtrip(loaded):
    """encoding the llama2.c golden text must decode back to the identical string. (Equality with the true
    generated ids is not required by BPE in general; it is reported in results/<run>/summary.json.)"""
    name, cfg, w, tok, seq, forced, golden = loaded
    enc = tok.encode(golden.rstrip(b"\n"))
    assert enc[0] == ref.BOS
    assert tok.render(enc) == golden


def test_encode_golden_text_equals_true_ids(loaded):
    """(minor 2, done): the round trip above also holds for a degenerate encoder (e.g.
    no merges), so additionally require that the run.c-equivalent encoding of the FULL golden text equals the true
    golden token ids (recorded by the instrumented run.c), not only for the 5-token prompt."""
    name, cfg, w, tok, seq, forced, golden = loaded
    assert tok.encode(golden.rstrip(b"\n")) == list(seq)


def test_encode_byte_fallback_and_utf8(ps_root):
    """tok512 has no multi-byte pieces, so UTF-8 code points fall back to byte tokens (id = byte + 3). The raw
    decoded pieces rebuild the text exactly; run.c's safe_printf (C locale) then drops every single non-ASCII or
    control byte, which render() mirrors."""
    cfg, w, tok = ref.load_model("stories260K", ps_root)
    text = "naïve ✓ x\x01".encode()
    enc = tok.encode(text)
    for b in "ï✓".encode():
        assert b + 3 in enc
    raw = b"".join(tok.decode(p, n) for p, n in zip(enc[:-1], enc[1:]))
    assert raw == text
    assert tok.render(enc) == bytes(b for b in text if 0x20 <= b <= 0x7E) + b"\n"
    assert tok.encode(b"") == [ref.BOS]
    assert tok.encode(b"a", bos=False, eos=True)[-1] == 2


# ------------------------------------------------------------------------------------------------ teacher forcing
def test_teacher_forced_exactness(loaded, tf_rows):
    name, *_ = loaded
    ev = ref.evaluate_teacher_forced(tf_rows)
    assert ev["compared"] == sum(1 for r in tf_rows if not r["forced"]) > 150
    assert ev["disagreements_beyond_near_tie"] == 0, ev
    for r in tf_rows:
        if not r["forced"] and not r["agree"]:
            assert r["rel_margin"] <= ref.NEAR_TIE_REL_MARGIN       # near-tie: reported, allowed


def test_logits_vs_float64(tf_rows):
    worst = max(r["max_abs_f32_vs_f64"] for r in tf_rows)
    assert worst < ref.F64_MAX_ABS_TOL, worst


def test_free_running_reproduces_golden(loaded, tf_rows):
    name, cfg, w, tok, seq, forced, golden = loaded
    fr = ref.free_running(cfg, w, tok.encode(ref.GOLDEN_PROMPT), len(forced))
    ev = ref.evaluate_teacher_forced(tf_rows)
    upto = len(seq) if ev["first_near_tie_pos"] is None else ev["first_near_tie_pos"] + 1
    assert fr[:upto] == seq[:upto]
    if ev["first_near_tie_pos"] is None:
        assert tok.render(fr) == golden


def test_near_tie_rule_helpers():
    assert ref.rel_margin(10.0, 9.9995) == pytest.approx(5e-5)
    assert ref.rel_margin(-2.0, -2.0) == 0.0
    rows = [dict(pos=0, forced=1, agree=0, rel_margin=1.0, max_abs_f32_vs_f64=0.0),
            dict(pos=1, forced=0, agree=0, rel_margin=5e-5, max_abs_f32_vs_f64=0.0),
            dict(pos=2, forced=0, agree=1, rel_margin=0.3, max_abs_f32_vs_f64=0.0)]
    ev = ref.evaluate_teacher_forced(rows)
    assert ev["passed"] and ev["disagreements"] == 1 and ev["first_near_tie_pos"] == 1
    rows[1]["rel_margin"] = 2e-4
    assert not ref.evaluate_teacher_forced(rows)["passed"]
    rows[1]["rel_margin"], rows[2]["max_abs_f32_vs_f64"] = 5e-5, 2e-3
    assert not ref.evaluate_teacher_forced(rows)["passed"]


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def fx260k(tmp_path_factory):
    out = tmp_path_factory.mktemp("fx") / "stories260K"
    meta = mf.make_model_fixtures("stories260K", [0, -1], ["0", "17", "max"], out)
    return out, meta


def test_fixture_inventory(fx260k, ps_root):
    out, meta = fx260k
    assert meta["layers"] == [0, 4] and meta["positions"] == [0, 17, 511]
    layer_ops = {"rmsnorm_att", "matmul_q", "matmul_k", "matmul_v", "rope", "kv_write", "attention", "matmul_o",
                 "residual_att", "rmsnorm_ffn", "matmul_w1", "matmul_w3", "silu_gate", "matmul_w2", "residual_ffn"}
    want = {f"L{l}_P{p}_{op}" for l in (0, 4) for p in (0, 17, 511) for op in layer_ops}
    want |= {f"P{p}_{op}" for p in (0, 17, 511) for op in ("embed", "rmsnorm_final", "classifier")}
    assert set(meta["fixtures"]) == want
    seq, _ = ref.read_golden_tokens(ps_root / ref.MODELS["stories260K"]["golden_tokens"])
    assert meta["token_sequence"][:len(seq)] == seq and len(meta["token_sequence"]) == 512
    assert json.loads((out / "model.json").read_text())["checkpoint_sha256"] == \
        "b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696"


def test_fixtures_recompute(fx260k, ps_root):
    """Every fixture's outputs follow from its inputs through the reference ops (raw LE float32 + JSON header)."""
    out, meta = fx260k
    cfg, w, _ = ref.load_model("stories260K", ps_root)
    r = ref.Llama2cRef(cfg, w)
    for name in meta["fixtures"]:
        m, a = mf.load_fixture(out / name)
        assert m["format"] == "ps-fixture-v1"
        op, pos = m["op"], m["pos"]
        for t in m["tensors"].values():
            raw = (out / name / t["file"]).read_bytes()
            assert len(raw) == 4 * int(np.prod(t["shape"])) and t["dtype"] == "<f4"
        if op.startswith("matmul") or op == "classifier":
            np.testing.assert_allclose(a["w"] @ a["x"], a["out"], rtol=1e-6, atol=1e-6)
        elif op.startswith("rmsnorm"):
            np.testing.assert_array_equal(r.rmsnorm(a["x"], a["weight"]), a["out"])
        elif op == "rope":
            q, k = r.rope(a["q_in"], a["k_in"], pos)
            np.testing.assert_array_equal(q, a["q_out"])
            np.testing.assert_array_equal(k, a["k_out"])
        elif op == "attention":
            assert a["k_cache"].shape == (pos + 1, cfg.kv_dim)
            o, s, p = r.attention(a["q"], a["k_cache"], a["v_cache"])
            np.testing.assert_allclose(o, a["out"], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)
            kv = mf.load_fixture(out / f"L{m['layer']}_P{pos}_kv_write")[1]
            np.testing.assert_array_equal(a["k_cache"][-1], kv["k"])
            np.testing.assert_array_equal(a["v_cache"][-1], kv["v"])
        elif op == "silu_gate":
            np.testing.assert_array_equal(r.silu_gate(a["hb"], a["hb2"]), a["out"])
        elif op.startswith("residual"):
            np.testing.assert_array_equal(a["x"] + a["branch"], a["out"])
    # role labels (minor 3, fixed): every fixture except kv_write has an output-role
    # tensor (kv_write's k, v are the values written into the cache, its output is the cache slot); the embedding
    # row `x` is embed's output
    for name in meta["fixtures"]:
        roles = {k: t["role"] for k, t in json.loads((out / name / "meta.json").read_text())["tensors"].items()}
        assert ("out" in roles.values()) != name.endswith("_kv_write"), (name, roles)
        if name.endswith("_embed"):
            assert roles == {"x": "out"}
    # the classifier at a non-forced golden position predicts the golden next token
    seq = meta["token_sequence"]
    _, a = mf.load_fixture(out / "P17_classifier")
    assert int(np.argmax(a["out"])) == seq[18]


def test_fixture_sha_check_detects_corruption(fx260k, tmp_path):
    import shutil
    out, _ = fx260k
    shutil.copytree(out, tmp_path / "copy")
    d = tmp_path / "copy" / "L0_P0_matmul_q"
    mf.load_fixture(d)                                  # intact copy loads
    raw = bytearray((d / "out.f32").read_bytes())
    raw[0] ^= 1
    (d / "out.f32").write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="sha256"):
        mf.load_fixture(d)


def test_stories15m_layer0_wq_fixture(tmp_path):
    """The GEMV test input: stories15M layer-0 wq (288x288) at pos 0."""
    out = tmp_path / "s15m"
    mf.make_model_fixtures("stories15M", [0], ["0"], out)
    m, a = mf.load_fixture(out / "L0_P0_matmul_q")
    assert a["w"].shape == (288, 288) and a["x"].shape == (288,)
    np.testing.assert_allclose(a["w"] @ a["x"], a["out"], rtol=1e-6, atol=1e-6)


def test_random_gemv_deterministic(tmp_path):
    m1 = mf.make_random_gemv(64, 32, 7, tmp_path / "a")
    m2 = mf.make_random_gemv(64, 32, 7, tmp_path / "b")
    assert m1["tensors"] == {k: {**v} for k, v in m2["tensors"].items()}
    _, a = mf.load_fixture(tmp_path / "a")
    np.testing.assert_allclose(a["w"] @ a["x"], a["out"], rtol=1e-6, atol=1e-6)


def test_no_llama2c_python_imports(ps_root):
    """no dependency on llama2.c Python files (only its file formats). llama2.c ships model.py,
    export.py, tokenizer.py, ...; our own package is also called `model`, so only `model.<our module>` is allowed."""
    ours = {"model.ref_llama2c", "model.make_fixtures"}
    llama2c_py = {p.stem for p in (ps_root / "third_party" / "llama2.c").glob("*.py")} | {"llama2c", "third_party"}
    for f in ("model/ref_llama2c.py", "model/make_fixtures.py"):
        src = (ps_root / f).read_text()
        tree = ast.parse(src)
        mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        for m in mods:
            assert m in ours or m.split(".")[0] not in llama2c_py, (f, m)
        assert "sys.path" not in src
