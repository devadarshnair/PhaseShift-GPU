"""(2026-11-16) unit tests: light, no simulator, no build, no tree writes."""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS / "scripts"))
import vxrun  # noqa: E402


def _tok(s):
    return s.split()


def test_dedupe_tokens_last_wins_and_order():
    t = _tok("-DA=1 -DB=2 -DA=3 -DC -DB=4 -fx -fx")
    assert vxrun.dedupe_tokens(t) == ["-DA=3", "-DC", "-DB=4", "-fx"]
    assert vxrun.dedupe_tokens([]) == []


def test_dedupe_separate_key_spaces_bare_token_vs_macro():
    t = _tok("FOO -DFOO=1 FOO -DFOO=2")
    assert vxrun.dedupe_tokens(t) == ["FOO", "-DFOO=2"]
    assert vxrun.dedupe_tokens(_tok("-DFOO FOO")) == ["-DFOO", "FOO"]
    assert vxrun.canonical_configs(vxrun.dedupe_tokens(t)) == vxrun.canonical_configs(t)


def test_dedupe_does_not_change_canonical_key_material():
    """canonical_configs already applied last-wins, so deduping first must give the identical key material."""
    for t in (_tok("-DA=1 -DB=2 -DA=3"), _tok("-DX -DY=1 -DX=2 -DX"), _tok("-DQ=9")):
        assert vxrun.canonical_configs(vxrun.dedupe_tokens(t)) == vxrun.canonical_configs(t)


def test_resolve_named_plus_extra_configs_emits_each_macro_once():
    named = {"N": {"configs": "-DVX_CFG_L2_MSHR_SIZE=16 -DFOO=1", "cores": 1}}
    s = vxrun.resolve({"tree": "ps", "app": "a", "config": "N", "configs": "-DVX_CFG_L2_MSHR_SIZE=32"}, named)
    assert s["configs_env"].split().count("-DVX_CFG_L2_MSHR_SIZE=32") == 1
    assert "-DVX_CFG_L2_MSHR_SIZE=16" not in s["configs_env"]
    assert "-DFOO=1" in s["configs_env"]
    assert s["configs_extra"] == "-DVX_CFG_L2_MSHR_SIZE=32"
    names = [vxrun._macro(t)[0] for t in s["configs_effective"].split()]
    assert len(names) == len(set(names))
    # same key material as without the dedupe: last wins
    assert "-DVX_CFG_L2_MSHR_SIZE=32" in s["configs_canonical"]


def test_stored_rows_keep_canonical_configs_under_dedupe():
    """Every stored row: re-deriving configs_canonical from its (possibly duplicated) env tokens + flag macros with the
    new dedupe gives exactly the stored value, so no run_key of an existing row changes."""
    n = 0
    for f in sorted((PS / "results").glob("*/rows.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if not isinstance(r.get("configs_effective"), str) or not r.get("configs_canonical"):
                continue
            toks = r["configs_effective"].split()
            assert vxrun.canonical_configs(vxrun.dedupe_tokens(toks)) == r["configs_canonical"], (f, r.get("run_key"))
            n += 1
    if n == 0:
        pytest.skip("no stored rows")


def test_vxrun_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "res").mkdir()
    v = vxrun.VxRun("X1", ps=PS, results=Path("res"), named={}, out=open(os.devnull, "w"))
    assert v.results.is_absolute() and v.dir.is_absolute() and v.ps.is_absolute()
    v2 = vxrun.VxRun("X1", ps=Path(os.path.relpath(PS)), named={}, out=open(os.devnull, "w"))
    assert v2.ps.is_absolute()


def test_row_records_cfgx_field_and_tag():
    src = (PS / "scripts/vxrun.py").read_text()
    assert '"configs_extra": s.get("configs_extra"' in src and '"cfgx"' in src


def _sh(*args, **kw):
    return subprocess.run(list(args), capture_output=True, text=True, **kw)


def test_build_vortex_rss_honest_and_tree_pattern():
    t = (PS / "scripts/build_vortex.sh").read_text()
    assert '--rss 1.0 --mem 5 --label "$STEP-tp-$X"' in t
    assert '--rss 0.5 --mem 4 --label "$STEP-sw-$X"' in t
    assert "--rss 2 " not in t
    # usage / tree-name rule equals vxrun's regex
    assert "{1,15}$" in t
    for good in ("base", "ps", "psrtl", "psx", "psdot", "pspf", "psff"):
        assert re.fullmatch(r"[a-z][a-z0-9]{1,15}", good)
    r = _sh("bash", str(PS / "scripts/build_vortex.sh"), "Bad_Tree")
    assert r.returncode == 2 and "usage" in r.stderr
    r = _sh("bash", str(PS / "scripts/build_vortex.sh"), "a")      # one char: vxrun regex requires >= 2
    assert r.returncode == 2
    r = _sh("bash", str(PS / "scripts/build_vortex.sh"), "ps", "bogus")
    assert r.returncode == 2


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bs_t6_make_batch_errors_on_missing_selection(tmp_path, monkeypatch):
    m = _load(PS / "results/BS.T6/make_batch.py", "bs6mb")
    cls = [dict(model="m", matrix="w", B=1, M=1, K=1, variant="v", args="-v x -L 1")]
    monkeypatch.setattr(m, "classes", lambda: cls)
    base = {"status": "ok", "cycles": 10, "tags": {"group": "lat", "role": "lat_cold", "model": "m", "matrix": "w", "B": 1}}
    monkeypatch.setattr(m, "load_rows", lambda step: [base] if step == "BS.T3" else [])
    with pytest.raises(SystemExit, match="no ok sel_cold row"):
        m.best_kernels()
    monkeypatch.setattr(m, "load_rows", lambda step: [])
    with pytest.raises(SystemExit, match="no ok"):
        m.best_kernels()
    sel = {"status": "ok", "cycles": 5, "args": "-v dq_b -L 1",
           "tags": {"role": "sel_cold", "model": "m", "matrix": "w", "B": 1, "variant": "dq_b"}}
    monkeypatch.setattr(m, "load_rows", lambda step: [base] if step == "BS.T3" else [sel])
    assert m.best_kernels()[("m", "w", 1)]["variant"] == "dq_b"


MK = PS / "scripts/mk_vortex_tree.sh"


def _fp(*a):
    r = _sh("bash", str(MK), "--print-fingerprint", *a)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_mk_vortex_tree_fingerprint_backward_compatible_and_extras_included():
    for ps in ("base", "ps"):
        marker = (PS / "vortex-trees" / ps / ".ps_patchset")
        if marker.is_file():
            assert _fp("--patchset", ps) == marker.read_text().strip()
    a = _fp("--patchset", "base")
    b = _fp("--patchset", "base", "--extra", "rtl")
    c = _fp("--patchset", "base", "--extra", "rtl", "--extra", "dot4")
    assert len({a, b, c}) == 3
    # editing an extra patch changes the fingerprint (hash over contents, incl. rtl 0004)
    assert b == _fp("--patchset", "base", "--extra", "rtl")
    r = _sh("bash", str(MK), "--patchset", "base", "--extra", "nosuchset", "--print-fingerprint")
    assert r.returncode == 2


def test_mk_vortex_tree_recreate_reuses_recorded_extras_in_script():
    t = MK.read_text()
    assert ".ps_extra" in t and "mapfile -t EXTRAS" in t


def test_synth_sh_refuses_stale_build_dir(tmp_path):
    sh = PS / "results/BS.T10/synth.sh"
    env = dict(os.environ, PS=str(tmp_path), SYNTH_CHECK_ONLY="1")
    args = ["bash", str(sh), "job", "TOPX", "-I/a", "-DX=1", "500", "1.0"]
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr                      # fresh dir: recorded
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    assert r.returncode == 0                                # no b_<top> dir yet -> nothing to be stale
    out = tmp_path / "build/bs_t10/job"
    (out / "b_TOPX").mkdir()
    assert subprocess.run(args, env=env, capture_output=True, text=True).returncode == 0   # same params -> reuse ok
    changed = args[:5] + ["-DX=2"] + args[6:]
    r = subprocess.run(changed, env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "REFUSING" in r.stderr
    (out / ".synth_params").unlink()                        # unrecorded legacy dir -> refused too
    assert subprocess.run(args, env=env, capture_output=True, text=True).returncode == 3
    env["SYNTH_ALLOW_REUSE"] = "1"
    assert subprocess.run(args, env=env, capture_output=True, text=True).returncode == 0


def test_synth_sh_fingerprint_changes_with_rtl(tmp_path):
    sh = PS / "results/BS.T10/synth.sh"
    tree = tmp_path / "tree"
    (tree / "hw/rtl").mkdir(parents=True)
    (tree / "hw/rtl/a.sv").write_text("module a; endmodule\n")
    env = dict(os.environ, PS=str(tmp_path), SYNTH_CHECK_ONLY="1", PS_RTL_TREE=str(tree), PS_BUILD_TREE=str(tmp_path / "nb"))
    args = ["bash", str(sh), "job", "TOPX", "-I/a", "-DX=1", "500", "1.0"]
    assert subprocess.run(args, env=env, capture_output=True, text=True).returncode == 0
    (tmp_path / "build/bs_t10/job/b_TOPX").mkdir()
    assert subprocess.run(args, env=env, capture_output=True, text=True).returncode == 0
    (tree / "hw/rtl/a.sv").write_text("module a; wire x; endmodule\n")     # RTL patch applied after the build
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "REFUSING" in r.stderr
    env["SYNTH_ALLOW_REUSE"] = "1"
    assert subprocess.run(args, env=env, capture_output=True, text=True).returncode == 0


def test_duplicated_kernel_sources_stay_identical():
    """Drift guard: these copies are byte-identical today (not symlinked:.md); keep them so."""
    groups = [
        ("gemv_q80", ["ps_gemv_q80dot4", "ps_gemv_q80dq", "ps_gemv_q80pf"],
         ["kernel.cpp", "kernel_tile.cpp", "kernel_nsp.cpp", "kernel_sw.cpp", "common.h"]),
        ("ps_gemv_q80dq", ["ps_gemv_q80pf"], ["kernel_dq.cpp"]),
        ("gemv_dxa", ["ps_gemv_dxaghost"], ["kernel.cpp", "common.h"]),
    ]
    for src, dsts, files in groups:
        for d in dsts:
            for f in files:
                a, b = PS / "kernels" / src / f, PS / "kernels" / d / f
                if a.exists() and b.exists():
                    assert a.read_bytes() == b.read_bytes(), f"{b} drifted from {a}"
