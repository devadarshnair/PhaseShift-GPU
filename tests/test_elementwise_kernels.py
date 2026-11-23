"""Tests: batched element-wise fixtures and the ps_rmsnorm / ps_rope / ps_silu_gate / ps_residual apps
.

Non-vortex (always run, cheap, no SimX):
  test_batch_fixture_inventory          `make_fixtures batch` for stories260K: one fixture per (batch, layer, op),
                                        shapes (B, n), params (positions, kv offsets), batch.json index.
  test_batch_fixture_rows_are_distinct  the B = 4 batch has 4 distinct positions and pairwise different rows (a B = 4 batch without duplicating a row).
  test_batch_fixtures_recompute         every output follows from its inputs through the reference ops, row by row at
                                        the row's own position; B = 4 rows equal the B = 1 fixtures at that position;
                                        rope_kv's v / k_out equal the reference KV cache contents.
  test_batch_fixtures_deterministic     regenerating gives byte-identical files.
  test_methodology_section_14_prelude   KV layout + offset formula + tolerance table in methodology §14.0.
  test_elementwise_apps_share_host_header   kernels/<app>/elt_host.h are symlinks to kernels/elt_common/elt_host.h
                                        (so vxrun's app-source hash covers the shared driver code).
  test_p23_results_cover_test_matrix    results/<run>/summary.csv (written by results/<run>/summarize.py from the batch
                                        experiments/<run>.yaml) has an ok row with every PSCHECK errors=0 for every
                                        fixture of both models and every synthetic edge case, and the benchmark rows.

Vortex-marked (PS_RUN_VORTEX=1; each SimX run goes through scripts/vxrun.py -> lowload.sh, serial; private results
dir build/tmp/elt_test, deleted at session start, as tests/test_kernels.py does):
  test_kernel_fixture[...]   each app on the stories15M last-layer B = 4 fixture and the stories260K layer-0 B = 1
                             pos = max fixture -> PASSED, every PSCHECK errors=0 at the app's tolerance.
  test_kernel_edge[...]      synthetic edge cases: odd B and n (n % NT != 0), grid-stride (-g), kv_dim < dim with a
                             different head size, in place (-i).
  test_checks_are_live[...]  the same kernels with a tolerance far below their real error -> FAILED (the comparisons
                             are not vacuous); a fixture of the wrong op is refused.
lowload exit 75 -> pytest.skip("DEFERRED").
"""
import csv
import json
import os
import re
import shutil

import numpy as np
import pytest

from model import make_fixtures as mf
from model import ref_llama2c as ref

PS = __import__("pathlib").Path(__file__).resolve().parents[1]
FIX = PS / "build" / "fixtures"
APPS = ("ps_rmsnorm", "ps_rope", "ps_silu_gate", "ps_residual")
TOL = {"ps_rmsnorm": 1e-5, "ps_rope": 1e-4, "ps_silu_gate": 1e-5, "ps_residual": 1e-5}
OP_APP = (("rmsnorm", "ps_rmsnorm"), ("rope_kv", "ps_rope"), ("silu_gate", "ps_silu_gate"), ("residual", "ps_residual"))
EXPECTED_CHECKS = {   # PSCHECK labels per app (fixture runs); synthetic runs have only the device-vs-f64 ones
    "ps_rmsnorm": {"fixture_vs_f64", "dev_vs_f64", "dev_vs_fixture"},
    "ps_rope": {"fixture_q_vs_f64", "fixture_k_vs_f64", "q_vs_f64", "q_vs_fixture", "kcache_vs_f64",
                "kcache_vs_fixture", "vcache_exact", "kv_untouched"},
    "ps_silu_gate": {"fixture_vs_f64", "dev_vs_f64", "dev_vs_fixture"},
    "ps_residual": {"fixture_vs_f64", "dev_vs_f64", "dev_vs_fixture", "dev_bits_vs_fixture"},
}
PSCHECK_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) "
                        r"tol=(\S+) errors=(\d+)", re.M)


def app_of(op: str) -> str:
    return next(a for prefix, a in OP_APP if op.startswith(prefix))


# --------------------------------------------------------------------------------------------- fixtures (cheap)
@pytest.fixture(scope="module")
def bfx(tmp_path_factory):
    out = tmp_path_factory.mktemp("bfx") / "stories260K_batch"
    idx = mf.make_batch_fixtures("stories260K", [0, -1], ["0", "17", "max", "0,17,100,max"], out)
    return out, idx


def test_batch_fixture_inventory(bfx):
    out, idx = bfx
    assert idx["layers"] == [0, 4] and idx["batches"] == [[0], [17], [511], [0, 17, 100, 511]]
    tags = ["B1_P0", "B1_P17", "B1_P511", "B4_P0-17-100-511"]
    want = {f"L{l}_{t}_{op}" for l in (0, 4) for t in tags for op in mf.BATCH_LAYER_OPS}
    want |= {f"{t}_rmsnorm_final" for t in tags}
    assert set(idx["fixtures"]) == want and len(want) == 52
    assert json.loads((out / "batch.json").read_text())["fixtures"] == sorted(want)
    cfg = ref.load_model("stories260K")[0]
    for name in idx["fixtures"]:
        m = json.loads((out / name / "meta.json").read_text())
        B = m["batch"]
        assert m["pos"] == -1 and m["params"]["batch"] == B == len(m["params"]["positions"])
        assert m["params"]["tokens"] == [idx["token_sequence"][p] for p in m["params"]["positions"]]
        for t in m["tensors"].values():
            if t["role"] != "weight":
                assert t["shape"][0] == B and len(t["shape"]) == 2
        if m["op"] == "rope_kv":
            assert m["params"]["kv_layout"] == "[layer][b][pos][kv_head][head_dim]"
            assert m["params"]["kv_cache_offset_elems"] == [((m["layer"] * B + b) * cfg.seq_len + p) * cfg.kv_dim
                                                           for b, p in enumerate(m["params"]["positions"])]
            assert {k: t["role"] for k, t in m["tensors"].items()} == {
                "q_in": "in", "k_in": "in", "v": "in", "q_out": "out", "k_out": "out"}


def test_batch_fixture_rows_are_distinct(bfx):
    out, _ = bfx
    for op in mf.BATCH_LAYER_OPS:
        for l in (0, 4):
            _, a = mf.load_fixture(out / f"L{l}_B4_P0-17-100-511_{op}")
            for name, x in a.items():
                if x.ndim != 2:
                    continue
                rows = {x[b].tobytes() for b in range(4)}
                # pos 0 rope is the identity, so q_in == q_out there, but rows across positions always differ
                assert len(rows) == 4, (op, l, name)


def test_batch_fixtures_recompute(bfx):
    out, idx = bfx
    cfg, w, _ = ref.load_model("stories260K")
    r = ref.Llama2cRef(cfg, w)
    seq = idx["token_sequence"]
    # reference KV cache for layer checks: run the reference over the needed positions
    for p in range(512):
        r.forward(seq[p], p)
    for name in idx["fixtures"]:
        m, a = mf.load_fixture(out / name)
        op, l, pos = m["op"], m["layer"], m["params"]["positions"]
        for b, p in enumerate(pos):
            if op.startswith("rmsnorm"):
                np.testing.assert_array_equal(r.rmsnorm(a["x"][b], a["weight"]), a["out"][b])
            elif op == "rope_kv":
                q, k = r.rope(a["q_in"][b], a["k_in"][b], p)
                np.testing.assert_array_equal(q, a["q_out"][b])
                np.testing.assert_array_equal(k, a["k_out"][b])
                np.testing.assert_array_equal(r.key_cache[l, p], a["k_out"][b])
                np.testing.assert_array_equal(r.value_cache[l, p], a["v"][b])
            elif op == "silu_gate":
                np.testing.assert_array_equal(r.silu_gate(a["hb"][b], a["hb2"][b]), a["out"][b])
            else:
                np.testing.assert_array_equal(a["x"][b] + a["branch"][b], a["out"][b])
        if m["batch"] == 4:   # B = 4 row b == the B = 1 fixture at the same position
            for b, p in enumerate(pos):
                one = name.replace("B4_P0-17-100-511", f"B1_P{p}")
                if (out / one).is_dir():
                    _, a1 = mf.load_fixture(out / one)
                    for t in a1:
                        if a1[t].ndim == 2:
                            np.testing.assert_array_equal(a1[t][0], a[t][b])


def test_batch_fixtures_deterministic(bfx, tmp_path):
    out, _ = bfx
    mf.make_batch_fixtures("stories260K", [0, -1], ["0", "17", "max", "0,17,100,max"], tmp_path / "again")
    for f in sorted(p for p in out.rglob("*") if p.is_file()):
        assert f.read_bytes() == (tmp_path / "again" / f.relative_to(out)).read_bytes(), f


def test_parse_batches_rejects_out_of_range():
    assert mf.parse_batches(["max", "1,2"], 8) == [[7], [1, 2]]
    for bad in (["8"], ["-1"], [""]):
        with pytest.raises(ValueError):
            mf.parse_batches(bad, 8)


def test_elementwise_apps_share_host_header():
    shared = PS / "kernels" / "elt_common" / "elt_host.h"
    assert shared.is_file() and not (PS / "kernels" / "elt_common" / "main.cpp").exists()
    for d in ("rmsnorm", "rope", "silu_gate", "residual"):
        link = PS / "kernels" / d / "elt_host.h"
        assert link.is_symlink() and link.resolve() == shared.resolve()
        for f in ("main.cpp", "kernel.cpp", "common.h"):
            assert (PS / "kernels" / d / f).is_file()


def test_methodology_section_14_prelude():
    """the KV-cache layout is documented in methodology §14 prelude, with the offset formula the fixtures
    and ps_rope use, and every element-wise tolerance is stated there."""
    meth = (PS / "docs" / "methodology.md").read_text()
    sec = meth[meth.index("### 14.0 Prelude"):meth.index("## 15.")]
    assert "`[layer][b][pos][kv_head][head_dim]`" in sec
    assert "`off(l, b, p) = ((l·b_cap + b)·seq_len + p)·kv_dim`" in sec
    for row in ("| RMSNorm |", "| RoPE (+KV) |", "| SiLU-gate |", "| residual |"):
        assert row in sec, row
    assert "§14.0" in meth[meth.index("6. **Kernel correctness metric"):meth.index("### Op-trace conventions")]


def _summary_rows():
    p = PS / "results" / "P2.3" / "summary.csv"
    assert p.is_file(), "results/P2.3/summary.csv missing: run the batch experiments/P2.3.yaml + summarize.py"
    with open(p) as f:
        return list(csv.DictReader(f))


def test_p23_results_cover_test_matrix():
    rows = _summary_rows()
    by = {(r["kind"], r["app"], r["args"], r["config"]): r for r in rows}
    for model in ("stories260K", "stories15M"):
        idx = json.loads((FIX / f"{model}_batch" / "batch.json").read_text())
        assert len(idx["fixtures"]) == 52
        for name in idx["fixtures"]:
            op = json.loads((FIX / f"{model}_batch" / name / "meta.json").read_text())["op"]
            app = app_of(op)
            r = by.get(("test", app, f"-f {model}_batch/{name}", "smoke-default"))
            assert r is not None, (model, name)
            assert r["status"] == "ok" and r["check_errors"] == "0", r
            assert set(r["checks"].split(";")) == EXPECTED_CHECKS[app], r
            assert float(r["max_rel_err"]) <= TOL[app], r
    edge = [r for r in rows if r["kind"] == "test-edge"]
    assert len(edge) == 12 and all(r["status"] == "ok" and r["check_errors"] == "0" for r in edge)
    bench = [r for r in rows if r["kind"] == "bench"]
    assert len(bench) == 16 and all(r["status"] == "ok" and r["check_errors"] == "0" and int(r["cycles"]) > 0
                                    for r in bench)


# --------------------------------------------------------------------------------------------- vortex (SimX)
@pytest.fixture(scope="module")
def vx():
    import subprocess
    import sys
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun  # noqa: E402
    priv = PS / "build" / "tmp" / "elt_test"
    if priv.exists():
        shutil.rmtree(priv)
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], env=dict(os.environ, PS=str(PS)),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun("P2.3-test", results=priv)


def _run(vx, app, args):
    row = vx.run({"tree": "base", "app": app, "args": args, "config": "smoke-default", "perf": 1}, timeout_s=3600)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


def _checks(log):
    return {m.group(1): {"max_rel_err": float(m.group(4)), "tol": float(m.group(6)), "errors": int(m.group(7))}
            for m in PSCHECK_RE.finditer(log)}


def _fixture_cases():
    cases = []
    for op, app in OP_APP:
        op15 = {"rmsnorm": "rmsnorm_ffn", "residual": "residual_ffn"}.get(op, op)
        op26 = {"rmsnorm": "rmsnorm_att", "residual": "residual_att"}.get(op, op)
        cases.append(pytest.param(app, FIX / "stories15M_batch" / f"L5_B4_P0-17-100-255_{op15}", id=f"{app}-15M-L5-B4"))
        cases.append(pytest.param(app, FIX / "stories260K_batch" / f"L0_B1_P511_{op26}", id=f"{app}-260K-L0-B1-P511"))
    cases.append(pytest.param("ps_rmsnorm", FIX / "stories15M_batch" / "B4_P0-17-100-255_rmsnorm_final",
                              id="ps_rmsnorm-15M-final-B4"))
    return cases


@pytest.mark.vortex
@pytest.mark.parametrize("app,fixture", _fixture_cases())
def test_kernel_fixture(vx, app, fixture):
    if not (fixture / "meta.json").is_file():
        pytest.fail(f"fixture {fixture} missing: run `python -B -m model.make_fixtures batch --model <m>`")
    row, log = _run(vx, app, f"-f {fixture}")
    assert row["status"] == "ok", row["reason"]
    c = _checks(log)
    assert set(c) == EXPECTED_CHECKS[app], c
    for k, v in c.items():
        assert v["errors"] == 0 and v["max_rel_err"] <= TOL[app], (k, v)


EDGE = [
    ("ps_rmsnorm", "-b 3 -n 37 -s 5"), ("ps_rmsnorm", "-b 5 -n 288 -s 6 -g 1"),
    ("ps_rope", "-b 3 -d 96 -m 32 -H 16 -S 64 -N 2 -l 1 -s 5"), ("ps_rope", "-b 2 -d 64 -m 32 -H 8 -S 512 -N 5 -l 4 -s 6 -g 1"),
    ("ps_silu_gate", "-b 3 -n 101 -s 5"), ("ps_silu_gate", "-b 3 -n 101 -s 6 -g 2"),
    ("ps_residual", "-b 3 -n 37 -s 5"), ("ps_residual", "-b 3 -n 37 -s 6 -g 1"),
]


@pytest.mark.vortex
@pytest.mark.parametrize("app,args", EDGE + [
    (a, f"-f {FIX}/stories15M_batch/L0_B4_P0-17-100-255_{op} -i")
    for a, op in (("ps_rmsnorm", "rmsnorm_att"), ("ps_rope", "rope_kv"), ("ps_silu_gate", "silu_gate"),
                  ("ps_residual", "residual_att"))], ids=lambda x: x if x in APPS else re.sub(r"\S*/", "", x))
def test_kernel_edge(vx, app, args):
    row, log = _run(vx, app, args)
    assert row["status"] == "ok", row["reason"]
    c = _checks(log)
    assert c and all(v["errors"] == 0 for v in c.values()), c
    if "-g" in args:
        assert "(grid overridden by -g)" in log
    if "-i" in args.split():
        assert "inplace=1" in log


@pytest.mark.vortex
@pytest.mark.parametrize("app,args,needle", [
    ("ps_rmsnorm", f"-f {FIX}/stories15M_batch/L5_B4_P0-17-100-255_rmsnorm_ffn -t 1e-9", "dev_vs_f64"),
    ("ps_rope", f"-f {FIX}/stories15M_batch/L5_B4_P0-17-100-255_rope_kv -t 1e-8", "q_vs_f64"),
    ("ps_silu_gate", f"-f {FIX}/stories15M_batch/L5_B4_P0-17-100-255_silu_gate -t 1e-9", "dev_vs_f64"),
    ("ps_rmsnorm", f"-f {FIX}/stories15M_batch/L5_B4_P0-17-100-255_residual_ffn", "is not rmsnorm_"),
], ids=["rmsnorm-tight-tol", "rope-tight-tol", "silu-tight-tol", "wrong-op"])
def test_checks_are_live(vx, app, args, needle):
    row, log = _run(vx, app, args)
    assert row["status"] == "failed" and "FAILED!" in log and "PASSED!" not in log
    if needle.startswith("is not"):
        assert needle in log
    else:
        assert _checks(log)[needle]["errors"] > 0
