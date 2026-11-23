"""Tests: the ps_llama full-model driver (runtime/llama_vx) and its evaluation.

Non-vortex (always run, no SimX):
  test_token_rule_*                 model/llama_vx_eval.py re-implements the TEST rule: agreement, a near-tie in both
                                    senses (top-1/top-2 AND top-1/golden within 1e-4 -- the carry-over),
                                    a top-2 tie with the golden token far below fails, free mode stops at the first
                                    disagreement.
  test_gemv_rule_matches_errata29   the GEMV-variant rule picks lanes exactly at the strides errata 29 measured as
                                    set-conflicting at NT = 32 (K = 512, 768, 1792) and rows at K = 288, 528 and at NT = 4.
  test_logits_metric                norm-wise relative error, scale-invariant.
  test_app_layout                   kernels/llama -> runtime/llama_vx; every reused source is a file symlink to the
                                    validated kernels/<k>/<file>; each device TU includes exactly one kernel.cpp; app.conf
                                    lists all 12 TUs (added k_q80.cpp, k_ep1.cpp, k_kv8r.cpp / k_kv8a.cpp); the generated Makefile (if built) has them in VX_SRCS.
  test_ref_logits_deterministic     model/llama_vx_ref.py regenerates the stories260K reference bit-identically (first 6
                                    positions) and matches the reference's argmax/margins.
  test_p29_results                  results/<run> (from experiments/<run>.yaml + summarize.py) covers the TEST matrix:
                                    every run ok and re-evaluated as passed, logits within 1e-3, the prefill-mode final
                                    logits within 1e-3, launches = flushes = 14 L + 2 per decode token, launch cycles sum to
                                    the PERF line, b_cap = 2 and perf-7 logits bit-identical, free-running text = golden
                                    prefix, C++ and Python GEMV rules agree, carry-over runs ok.
  test_methodology_section_14_5     the modeling choices are documented.
Vortex-marked (PS_RUN_VORTEX=1; private results dir build/tmp/llama_vx_test, each SimX job via vxrun -> lowload):
  test_ps_llama_short               stories260K, 6 positions teacher-forced (2 compared) -> PASSED, logits checked.
  test_ps_llama_rule_is_live        the same with one golden next token replaced -> FAILED (the comparison is real).
lowload exit 75 -> pytest.skip("DEFERRED").
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS))
from model import llama_vx_eval as ev  # noqa: E402
from model import ref_llama2c as ref  # noqa: E402

RES = PS / "results" / "P2.9"
APP = PS / "runtime" / "llama_vx"
REUSED = {"gemv_f32": ("kernel.cpp", "common.h"), "rmsnorm": ("kernel.cpp", "common.h"),
          "rope": ("kernel.cpp", "common.h"), "silu_gate": ("kernel.cpp", "common.h"),
          "residual": ("kernel.cpp", "common.h"), "decode_attn_gqa": ("kernel.cpp", "common.h", "warp_ops.h"),
          "linear": ("kernel.cpp", "common.h"), "prefill_attn": ("kernel.cpp", "common.h", "warp_ops.h"),
          "gemv_q80": ("kernel.cpp", "common.h")}   # the Q8_0 kernels of `ps_llama -q q80`
# the table / exp-variant entries live in their own TU (k_ep1.cpp), so the nine TUs above (every default-mode launch) are unchanged
EP1_REUSED = {"rope": ("kernel_tab.cpp", "rope_table_ref.h"), "silu_gate": ("kernel_exp.cpp", "warp_ops.h")}
EP1_TU = ("k_ep1.cpp", ["reused/rope/kernel_tab.cpp", "reused/silu_gate/kernel_exp.cpp"])
# the int8-KV entries live in two more TUs (k_kv8r.cpp: rope_kv_tab_q8, compiled with fp contract off; k_kv8a.cpp: attn_*_kv8 / kv16), linked last
EPK_REUSED = {"rope": ("kernel_kv8.cpp", "common_kv8.h"), "decode_attn_gqa": ("kernel_kv.cpp", "common_kv.h")}
EPK_TUS = (("k_kv8r.cpp", ["reused/rope/kernel_kv8.cpp"]), ("k_kv8a.cpp", ["reused/decode_attn_gqa/kernel_kv.cpp"]))
TUS = {"kernel.cpp": "gemv_f32", "k_rmsnorm.cpp": "rmsnorm", "k_rope.cpp": "rope", "k_silu.cpp": "silu_gate",
       "k_residual.cpp": "residual", "k_attn.cpp": "decode_attn_gqa", "k_linear.cpp": "linear",
       "k_pattn.cpp": "prefill_attn", "k_q80.cpp": "gemv_q80"}


def _tok(pos, am, g, top1, top2, lg, forced=0):
    return dict(pos=pos, argmax=am, golden_next=g, forced=forced, top1=top1, top2=top2, l_golden=lg,
                rel_top2=ev.rel_margin(top1, top2))


# --------------------------------------------------------------------------------------------- rules (cheap)
def test_token_rule_agree_and_near_tie():
    rows = [_tok(0, 5, 5, 10.0, 9.0, 10.0, forced=1), _tok(1, 7, 7, 12.0, 11.0, 12.0),
            _tok(2, 3, 4, 10.0, 10.0005, 9.9995)]            # disagreement; both margins 5e-5 <= 1e-4
    e = ev.evaluate_tokens(rows, "tf")
    assert e["compared"] == 2 and e["agree"] == 1 and e["near_ties"] == 1 and e["failures"] == 0 and e["passed"]
    assert e["first_near_tie"] == 2


def test_token_rule_golden_logit_must_be_close():
    """top-1/top-2 within 1e-4 but the golden token's logit far below: a real error, not a near-tie."""
    rows = [_tok(4, 3, 9, 10.0, 9.9995, 8.0)]
    e = ev.evaluate_tokens(rows, "tf")
    assert e["failures"] == 1 and not e["passed"]
    assert not ev.is_near_tie(10.0, 9.9995, 8.0) and ev.is_near_tie(10.0, 9.9995, 9.9995)


def test_token_rule_free_stops_at_first_disagreement():
    rows = [_tok(4, 1, 1, 9.0, 8.0, 9.0), _tok(5, 2, 3, 9.0, 8.9995, 8.9995), _tok(6, 0, 7, 9.0, 1.0, 1.0)]
    assert ev.evaluate_tokens(rows, "free")["passed"]               # pos 6 is after the near-tie: not compared
    assert not ev.evaluate_tokens(rows, "tf")["passed"]             # teacher forcing compares pos 6 too


def test_gemv_rule_matches_errata29():
    # errata 29 (TIER-CFG NT = 32, 16 KiB 4-way L1, 64 B lines): rows/lanes 3.5-10x at K = 512, 768, 1792;
    # rows faster at K = 528 (odd 33-line stride) and on par at K = 288 (0.97x); at NT = 4 rows was faster.
    for K in (512, 768, 1792):
        assert ev.gemv_variant(K, 32) == "lanes", K
    for K in (288, 528):
        assert ev.gemv_variant(K, 32) == "rows", K
    for K in (64, 172, 288, 512, 768, 1792):
        assert ev.gemv_variant(K, 4) == "rows", K
    for lines in (1, 3, 9, 33, 47):                 # an odd number of 64 B lines is always conflict-free (NT <= 64)
        assert ev.rows_set_pressure(64 * lines, 32) == 1


def test_logits_metric():
    r = np.array([10.0, -3.0, 0.0, 2.0], np.float32)
    d = r + np.array([0.01, 0.0, -0.02, 0.0], np.float32)
    assert ev.logits_rel(d, r) == pytest.approx(0.002, rel=1e-5)
    assert ev.logits_rel(4 * d, 4 * r) == pytest.approx(ev.logits_rel(d, r), rel=1e-5)


def test_app_layout():
    link = PS / "kernels" / "llama"
    assert link.is_symlink() and link.resolve() == APP.resolve()
    for k, files in REUSED.items():
        for f in files:
            p = APP / "reused" / k / f
            assert p.is_symlink() and p.resolve() == (PS / "kernels" / k / f).resolve(), p
    for k, files in list(EP1_REUSED.items()) + list(EPK_REUSED.items()):
        for f in files:
            p = APP / "reused" / k / f
            assert p.is_symlink() and p.resolve() == (PS / "kernels" / k / f).resolve(), p
    conf = dict(l.split("=", 1) for l in (APP / "app.conf").read_text().splitlines() if l and not l.startswith("#"))
    assert conf["VX_SRCS"].split() == list(TUS) + [EP1_TU[0]] + [t[0] for t in EPK_TUS]
    for tu, k in TUS.items():
        inc = re.findall(r'^#include "([^"]+)"', (APP / tu).read_text(), re.M)
        assert inc == [f"reused/{k}/kernel.cpp"], (tu, inc)
    assert re.findall(r'^#include "([^"]+)"', (APP / EP1_TU[0]).read_text(), re.M) == EP1_TU[1]
    for tu, inc in EPK_TUS:
        assert re.findall(r'^#include "([^"]+)"', (APP / tu).read_text(), re.M) == inc, tu
    mk = PS / "build/vortex-base/tests/regression/ps_llama/Makefile"
    if mk.is_file():
        vx = next(l for l in mk.read_text().splitlines() if l.startswith("VX_SRCS :="))
        assert vx.split(":=")[1].split() == [f"$(SRC_DIR)/{t}" for t in list(TUS) + [EP1_TU[0]] + [t[0] for t in EPK_TUS]]


def test_ref_logits_deterministic(tmp_path):
    from model import llama_vx_ref
    f = llama_vx_ref.make("stories260K", 6, tmp_path)
    stored = PS / "build/fixtures/llama_vx/stories260K_tf32.f32"
    if not stored.is_file():
        pytest.skip("reference logits not generated (python -B -m model.llama_vx_ref ...)")
    a = np.fromfile(f, "<f4").reshape(6, -1)
    b = np.fromfile(stored, "<f4").reshape(32, -1)[:6]
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    p21 = [l.split(",") for l in (PS / "results/P2.1/teacher_forced_stories260K.csv").read_text().splitlines()[1:7]]
    assert [int(np.argmax(x)) for x in a] == [int(r[4]) for r in p21]   # ref_argmax column


# --------------------------------------------------------------------------------------------- stored results
def _csv(name):
    import csv
    p = RES / name
    assert p.is_file(), f"{p} missing: run experiments/P2.9.yaml + results/P2.9/summarize.py"
    with open(p) as fh:
        return list(csv.DictReader(fh))


def test_p29_results():
    rows = _csv("summary.csv")
    assert all(r["status"] == "ok" or r["superseded"] == "True" for r in rows)
    by = {(r["model"], r["case"], r["perf"]): r for r in rows if r["superseded"] != "True"}
    need = [("stories260K", "tf", "1"), ("stories260K", "free", "1"), ("stories260K", "prefill", "1"),
            ("stories260K", "bcap2", "1"), ("stories15M", "tf", "1"), ("stories15M", "prefill", "1"),
            ("stories260K", "tf", "7"), ("stories15M", "tf", "7"), ("stories260K", "lanes", "1"),
            ("stories15M", "lanes", "1")]
    L = {"stories260K": 5, "stories15M": 6}
    for k in need:
        r = by[k]
        assert r["status"] == "ok" and r["eval_passed"] == "True" and r["failures"] == "0", r
        assert int(r["eval_compared"]) == int(r["compared"]) > 0
        assert float(r["max_logits_rel"]) <= 1e-3
        assert int(r["launches"]) == int(r["flushes"])
        assert int(r["launch_cycle_sum"]) == int(r["cycles"])
        assert int(r["launches_per_decode_token"]) == 14 * L[k[0]] + 2
    assert int(by[("stories260K", "tf", "1")]["positions"]) == 32
    assert int(by[("stories260K", "free", "1")]["positions"]) == 32
    assert int(by[("stories15M", "tf", "1")]["positions"]) >= 8
    r32 = by[("stories15M", "tf32", "1")]                      # extension: 32 positions (tier A/B)
    assert r32["status"] == "ok" and r32["eval_passed"] == "True" and int(r32["compared"]) == 28
    chk = json.loads((RES / "checks.json").read_text())
    for m in ("stories260K", "stories15M"):
        assert chk[f"{m}:prefill_final_logits_rel_vs_numpy"] <= 1e-3        # batched prefill rel tol 1e-3
        assert chk[f"{m}:prefill_then_decode_max_logits_rel_vs_numpy"] <= 1e-3
        assert chk[f"{m}:perf7_bitidentical_to_perf1"] is True
        assert chk[f"{m}:lanes_logits_rel_vs_rows"] <= 1e-3
    assert chk["stories260K:bcap2_bitidentical_to_tf"] is True
    assert chk["stories260K:free:text_is_golden_prefix"] is True
    assert chk["stories260K:free:generated_tokens"] == 33
    toks = _csv("tokens.csv")
    for t in toks:   # every logged near-tie satisfies both criteria
        if t["near_tie"] == "1":
            assert ev.is_near_tie(float(t["top1"]), float(t["top2"]), float(t["l_golden"]))
    # C++ rule (PSGEMV lines) == Python port
    for g in _csv("gemv_variants.csv"):
        assert int(g["rows_set_pressure"]) == ev.rows_set_pressure(int(g["stride_bytes"]), 4)
        assert g["auto_variant"] == ev.gemv_variant(int(g["K"]), 4)
    carry = _csv("carry.csv")
    assert {c["case"] for c in carry} >= {"rope_bcap", "rmsnorm_tol", "rmsnorm_bench"}
    for c in carry:
        assert c["status"] == "ok" and c["check_errors"] == "0", c
    ops = _csv("ops.csv")
    m15 = [o for o in ops if o["model"] == "stories15M" and o["case"] == "tf" and o["perf"] == "1"]
    assert {o["op"] for o in m15} == {"rmsnorm_att", "wq", "wk", "wv", "rope_kv", "attn", "wo", "residual_att",
                                      "rmsnorm_ffn", "w1", "w3", "silu_gate", "w2", "residual_ffn", "rmsnorm_final",
                                      "wcls"}
    assert abs(sum(float(o["share"]) for o in m15) - 1.0) < 1e-9


def test_methodology_section_14_5():
    meth = (PS / "docs" / "methodology.md").read_text()
    sec = meth[meth.index("### 14.5"):meth.index("## 15.")]
    for s in ("14 L + 2", "flush", "rows_set_pressure", "near-tie", "max_i |l_i − r_i| / max_j |r_j|", "musl",
              "seq_len = 0", "MCYCLE"):
        assert s in sec, s


# --------------------------------------------------------------------------------------------- SimX (vortex)
@pytest.fixture(scope="module")
def vx():
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun
    res = PS / "build" / "tmp" / "llama_vx_test"
    shutil.rmtree(res, ignore_errors=True)
    return vxrun.VxRun("P2.9-test", results=res), res


def _run(vx, args):
    v, _res = vx
    r = v.run(dict(tree="base", app="ps_llama", args=args, config="REG-CFG", perf=1), timeout_s=3600, rss_gb=1.0)
    if r["status"] == "deferred":
        pytest.skip("DEFERRED (lowload exit 75)")
    return r, (PS / r["log"]).read_text(errors="replace")


@pytest.mark.vortex
def test_ps_llama_short(vx):
    r, txt = _run(vx, f"-w {PS}/data/models/stories260K.bin -T {PS}/results/S8/golden_tokens_260K.txt -m tf -n 6 "
                      f"-r {PS}/build/fixtures/llama_vx/stories260K_tf32.f32")
    assert r["status"] == "ok", txt[-2000:]
    p = ev.parse_log(txt)
    assert p["passed"] and p["summary"]["compared"] == 2 and p["summary"]["failures"] == 0
    assert len(p["checks"]) == 6 and all(c["errors"] == 0 for c in p["checks"])
    assert sum(int(l["cycles"]) for l in p["launches"]) == p["perf_cycles"]


@pytest.mark.vortex
def test_ps_llama_rule_is_live(vx):
    _v, res = vx
    src = (PS / "results/S8/golden_tokens_260K.txt").read_text().splitlines()
    out = []
    for ln in src:
        f = ln.split()
        if not ln.startswith("#") and f[0] == "5":     # pos 5: next 383 -> 384 (and the chain at pos 6)
            f[2] = str(int(f[2]) + 1)
        if not ln.startswith("#") and f[0] == "6":
            f[1] = str(int(f[1]) + 1)
        out.append(" ".join(f) if not ln.startswith("#") else ln)
    bad = res / "golden_tokens_260K_wrong.txt"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("\n".join(out) + "\n")
    r, txt = _run(vx, f"-w {PS}/data/models/stories260K.bin -T {bad} -m tf -n 6")
    assert r["status"] == "failed" and "FAILED!" in txt
    p = ev.parse_log(txt)
    assert p["summary"]["failures"] == 1
