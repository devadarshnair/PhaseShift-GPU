"""Tests: ps patch dcr-phase ([dcr_phase] DCR block, host helper, PS_DUMP_PHASE, no-op regression script).

Non-vortex (always run, cheap, no compile):
  test_phase_block_map            vortex-trees/ps/VX_types.toml [dcr_phase]: inside the free range 0x024-0x03F, no
                                  overlap with any other DCR block, contiguous unique registers, and the same numbers
                                  as the constants in kernels/common/ps_phase.h.
  test_str_formats_identical      host (ps_phase.h) and SimX (phase_config.h) print the config with the same format.
  test_persist_never_default      the host helper statically asserts that the default config has persist_l2 = 0.
  test_check_phase_log_*          the log checker used below on synthetic logs (match, mismatch, missing launch).
  test_noop_compare_*             scripts/noop_regression.py's comparator on synthetic rows.
  test_reg_list_has_plan_entries  experiments/noop_regression.yaml holds the earlier REG set on both trees.

Vortex-marked (PS_RUN_VORTEX=1; SimX runs through scripts/vxrun.py -> lowload.sh, serial; private results dir
build/tmp/ps_phase_test, so they never skip on old rows):
  test_encode_decode_roundtrip    host g++ (via lowload.sh) of tests/fixtures/ps_phase_roundtrip.cpp: 2,000 random
                                  configs encoded by ps::PhaseCfg::regs() and decoded by vortex::PhaseConfig::write().
  test_phase_rw_dump[w]          : ps_phase_rw on the ps tree (REG-CFG, PS_DUMP_PHASE=1), 5 launches
                                  with 4 distinct configs, written with w = alt | queue | sync: for every launch, every
                                  consumer's post-reset dump equals the config the host wrote before that launch.
  test_phase_rw_dump_dxa          the same with DXA built in (REG-CFG + -DVX_CFG_EXT_DXA_ENABLE): DxaCore included.
  test_base_ignores_phase_dcrs    base tree: the same app writing non-default configs PASSES, prints no dump, and has
                                  instrs/cycles identical to the run that writes nothing (unknown DCRs are ignored).
  test_default_write_is_noop_ps   ps tree: writing the all-zero block before every launch == writing nothing.
lowload exit 75 -> pytest.skip("DEFERRED").
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

PS = __import__("pathlib").Path(__file__).resolve().parents[1]
TREE = PS / "vortex-trees" / "ps"
HDR = PS / "kernels" / "common" / "ps_phase.h"
SIMX_HDR = TREE / "sim" / "common" / "phase_config.h"
HOST_RE = re.compile(r"^PSPHASE-HOST launch=(\d+) (.*\S)\s*$", re.M)
DUMP_RE = re.compile(r"^PSPHASE-DUMP launch=(\d+) consumer=(\S+) (.*\S)\s*$", re.M)

sys.path.insert(0, str(PS / "scripts"))


def check_phase_log(text: str):
    """-> (n_launches, consumers, n_distinct_configs, errors). Every launch with a host line must have dumps from the
    same, non-empty consumer set, each equal to the host config; dumps for launches without a host line are errors."""
    host, dumps, errors = {}, {}, []
    for m in HOST_RE.finditer(text):
        k = int(m.group(1))
        if k in host:
            errors.append(f"duplicate host line for launch {k}")
        host[k] = m.group(2)
    for m in DUMP_RE.finditer(text):
        k, c = int(m.group(1)), m.group(2)
        if c in dumps.setdefault(k, {}):
            errors.append(f"duplicate dump launch {k} consumer {c}")
        dumps[k][c] = m.group(3)
    if not host:
        errors.append("no PSPHASE-HOST lines")
    if sorted(host) != list(range(len(host))):
        errors.append(f"host launches not 0..n-1: {sorted(host)}")
    if set(dumps) != set(host):
        errors.append(f"launches with dumps {sorted(dumps)} != launches written {sorted(host)}")
    consumers = None
    for k in sorted(host):
        d = dumps.get(k, {})
        if not d:
            continue
        if consumers is None:
            consumers = set(d)
        elif set(d) != consumers:
            errors.append(f"launch {k}: consumer set differs: {sorted(set(d) ^ consumers)}")
        for c, cfg in sorted(d.items()):
            if cfg != host[k]:
                errors.append(f"launch {k} consumer {c}: dump '{cfg}' != host '{host[k]}'")
    return len(host), consumers or set(), len(set(host.values())), errors


# --------------------------------------------------------------------------------------------- cheap tests
def _toml_dcrs():
    vals, blocks = {}, {}
    for line in (TREE / "VX_types.toml").read_text().splitlines():
        m = re.match(r"\s*(VX_DCR_\w+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)\b", line)
        if m:
            vals[m.group(1)] = int(m.group(2), 0)
            b = re.match(r"(VX_DCR_\w+?)_STATE_(BEGIN|END)$", m.group(1))
            if b:
                blocks.setdefault(b.group(1), {})[b.group(2)] = int(m.group(2), 0)
    return vals, blocks


@pytest.mark.skipif(not (TREE / "VX_types.toml").exists(), reason="vortex-trees/ps not created")
def test_phase_block_map():
    vals, blocks = _toml_dcrs()
    ph = blocks["VX_DCR_PHASE"]
    assert 0x024 <= ph["BEGIN"] < ph["END"] <= 0x040
    assert ph["BEGIN"] >= vals["VX_DCR_KMU_STATE_END"]
    for name, b in blocks.items():
        if name != "VX_DCR_PHASE":
            assert b["END"] <= ph["BEGIN"] or b["BEGIN"] >= ph["END"], f"{name} overlaps [dcr_phase]"
    regs = {k: v for k, v in vals.items() if k.startswith("VX_DCR_PHASE_") and not k.endswith(("_BEGIN", "_END"))}
    assert sorted(regs.values()) == list(range(ph["BEGIN"], ph["END"])), "registers not contiguous/unique"
    # host constants (ps_phase.h) agree with the toml
    host = dict(re.findall(r"constexpr uint32_t (\w+)\s*=\s*(0x[0-9A-Fa-f]+);", HDR.read_text()))
    host = {k: int(v, 16) for k, v in host.items()}
    assert host["STATE_BEGIN"] == ph["BEGIN"] and host["STATE_END"] == ph["END"]
    for k, v in host.items():
        if k.startswith("STATE_") or k.startswith("FLAG_"):
            continue
        assert vals["VX_DCR_PHASE_" + k] == v, k


def _fmt(path):
    src = path.read_text()
    i = src.index("snprintf(")
    return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', src[i:src.index(";", i)]))


@pytest.mark.skipif(not SIMX_HDR.exists(), reason="vortex-trees/ps not patched")
def test_str_formats_identical():
    assert _fmt(HDR) == _fmt(SIMX_HDR)
    assert "kv0=0x%llx-0x%llx" in _fmt(HDR)


def test_persist_never_default():
    src = HDR.read_text()
    assert re.search(r"static_assert\(\(PhaseCfg\{\}\.flags & phase_dcr::FLAG_PERSIST_L2\) == 0", src)
    assert re.search(r"uint32_t flags = 0;", src)


_CFG0 = "mode=0x0 flags=0x0"
_CFG1 = "mode=0x1 flags=0x10000"


def test_check_phase_log_match():
    log = "\n".join([f"PSPHASE-HOST launch=0 {_CFG0}", f"PSPHASE-HOST launch=1 {_CFG1}",
                     f"PSPHASE-DUMP launch=0 consumer=l3cache {_CFG0}", f"PSPHASE-DUMP launch=0 consumer=l2 {_CFG0}",
                     f"PSPHASE-DUMP launch=1 consumer=l3cache {_CFG1}", f"PSPHASE-DUMP launch=1 consumer=l2 {_CFG1}"])
    n, cons, distinct, err = check_phase_log(log)
    assert (n, cons, distinct, err) == (2, {"l3cache", "l2"}, 2, [])


def test_check_phase_log_detects_errors():
    base = [f"PSPHASE-HOST launch=0 {_CFG0}", f"PSPHASE-HOST launch=1 {_CFG1}",
            f"PSPHASE-DUMP launch=0 consumer=a {_CFG0}"]
    # launch 1 has no dump
    assert check_phase_log("\n".join(base))[3]
    # a consumer holding the previous launch's config
    err = check_phase_log("\n".join(base + [f"PSPHASE-DUMP launch=1 consumer=a {_CFG0}"]))[3]
    assert any("launch 1 consumer a" in e for e in err)
    # a consumer missing in one launch
    err = check_phase_log("\n".join(base + [f"PSPHASE-DUMP launch=1 consumer=b {_CFG1}"]))[3]
    assert any("consumer set differs" in e for e in err)
    # no host lines at all
    assert check_phase_log(f"PSPHASE-DUMP launch=0 consumer=a {_CFG0}")[3]


def _row(tree, fp, cycles, app="sgemm", args="-n64", key="k", src=""):
    return {"status": "ok", "tree": tree, "tree_fp": fp, "app": app, "args": args, "perf": 1, "env": {},
            "configs_canonical": ["-DPERF_ENABLE"], "instrs": 100, "cycles": cycles, "run_key": key * 12,
            "app_src_hash": src}


def test_noop_compare_identical_and_different():
    import noop_regression as nr
    spec = [{"tree": "ps", "app": "sgemm", "args": "-n64", "perf": 1}]
    fps = {"base": "B", "ps": "P"}
    res = nr.compare([_row("base", "B", 50), _row("ps", "P", 50)], spec, fps, {})
    assert res[0]["identical"] and res[0]["note"] == ""
    res = nr.compare([_row("base", "B", 50), _row("ps", "P", 51)], spec, fps, {})
    assert not res[0]["identical"] and "cycles +1" in res[0]["note"]


def test_noop_compare_ignores_stale_rows():
    import noop_regression as nr
    spec = [{"tree": "ps", "app": "sgemm", "args": "-n64", "perf": 1}]
    fps = {"base": "B", "ps": "P2"}
    # the only ps row was measured on an older ps tree -> missing, not compared
    res = nr.compare([_row("base", "B", 50), _row("ps", "P1", 50)], spec, fps, {})
    assert not res[0]["identical"] and "missing ps" in res[0]["note"]
    # app-source hash mismatch is ignored too
    res = nr.compare([_row("base", "B", 50, src="h1"), _row("ps", "P2", 50, src="h1")], spec, fps, {},
                     app_hash=lambda t, a: "h2")
    assert not res[0]["identical"] and "missing" in res[0]["note"]
    # the newest of two current rows wins
    res = nr.compare([_row("base", "B", 49), _row("base", "B", 50), _row("ps", "P2", 50)], spec, fps, {})
    assert res[0]["identical"]


def test_reg_list_has_plan_entries():
    import vxrun
    import noop_regression as nr
    reg = nr.load_reg(PS / "experiments" / "noop_regression.yaml")
    specs = vxrun.expand_batch(nr.batch_doc(reg, "x"), vxrun.load_named_configs())
    got = {(s["tree"], s["app"], s["args"], s.get("config")) for s in specs}
    for tree in ("base", "ps"):
        for app, args, cfg in [("sgemm", "-n64", "REG-CFG"), ("sgemv", "-m256 -n256", "REG-CFG"),
                               ("softmax", "-n64", "REG-CFG"),
                               ("sgemm_tcu_wg_dxa", "-m 128 -n 128 -k 64", "smoke-wgmma-dxa")]:
            assert (tree, app, args, cfg) in got
    assert all(not s.get("env") for s in specs)   # DCRs and env at defaults


# --------------------------------------------------------------------------------------------- vortex (SimX)
@pytest.fixture(scope="module")
def vx():
    import vxrun
    priv = PS / "build" / "tmp" / "ps_phase_test"
    if priv.exists():
        shutil.rmtree(priv)
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], capture_output=True, text=True,
                       env=dict(os.environ, PS=str(PS)))
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun("P4.1-test", results=priv)


def _run(vx, spec):
    row = vx.run(spec, timeout_s=3600)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


@pytest.mark.vortex
def test_encode_decode_roundtrip():
    out = PS / "build" / "tmp" / "ps_phase_test_rt"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    exe = out / "ps_phase_roundtrip"
    gen = PS / "build" / "vortex-ps" / "sw"
    cmd = ["g++", "-std=c++17", "-Wall", "-Wextra", "-pedantic", "-Werror", f"-I{gen}", f"-I{TREE / 'sim/common'}",
           f"-I{TREE / 'sw/runtime/include'}", f"-I{PS / 'kernels/common'}",
           str(PS / "tests/fixtures/ps_phase_roundtrip.cpp"), "-o", str(exe)]
    env = dict(os.environ, PS_RESULTS_DIR=str(out), PS_STEP="P4.1-test")
    r = subprocess.run([str(PS / "scripts/lowload.sh"), "--rss", "0.5", "--mem", "4", "--label", "P4.1-test:roundtrip",
                        "--log", str(out / "build.log"), "--", *cmd], env=env, capture_output=True, text=True)
    if r.returncode == 75:
        pytest.skip("DEFERRED: lowload.sh exit 75")
    assert r.returncode == 0, (out / "build.log").read_text(errors="replace")
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0 and "roundtrip: OK (0 errors)" in r.stdout, r.stdout


def _assert_dump_ok(row, log, launches=5):
    assert row["status"] == "ok", row["reason"]
    assert re.findall(r"^PSCHECK: launch\d+ errors=(\d+)", log, re.M) == ["0"] * launches
    n, consumers, distinct, err = check_phase_log(log)
    assert err == []
    assert n == launches >= 3 and distinct >= 3
    assert {"l3cache", "dram-dramsim"} <= consumers
    assert any(c.endswith("l2cache") for c in consumers)
    assert any("icache" in c for c in consumers) and any("dcache" in c for c in consumers)
    return consumers


@pytest.mark.vortex
@pytest.mark.parametrize("w", ["alt", "queue", "sync"])
def test_phase_rw_dump(vx, w):
    row, log = _run(vx, {"tree": "ps", "app": "ps_phase_rw", "args": f"-L 5 -c configs -w {w}", "config": "REG-CFG",
                         "env": {"PS_DUMP_PHASE": "1"}})
    consumers = _assert_dump_ok(row, log)
    assert not any("dxa" in c for c in consumers)   # DXA not built at REG-CFG


@pytest.mark.vortex
def test_phase_rw_dump_dxa(vx):
    row, log = _run(vx, {"tree": "ps", "app": "ps_phase_rw", "args": "-L 5 -c configs -w alt", "config": "REG-CFG",
                         "configs": "-DVX_CFG_EXT_DXA_ENABLE", "env": {"PS_DUMP_PHASE": "1"}})
    consumers = _assert_dump_ok(row, log)
    assert any("dxa" in c for c in consumers)


@pytest.mark.vortex
def test_base_ignores_phase_dcrs(vx):
    r_cfg, log = _run(vx, {"tree": "base", "app": "ps_phase_rw", "args": "-L 5 -c configs -w alt",
                           "config": "REG-CFG", "env": {"PS_DUMP_PHASE": "1"}})
    assert r_cfg["status"] == "ok", r_cfg["reason"]
    assert "PSPHASE-DUMP" not in log and len(HOST_RE.findall(log)) == 5
    r_none, _ = _run(vx, {"tree": "base", "app": "ps_phase_rw", "args": "-L 5 -c none", "config": "REG-CFG"})
    assert r_none["status"] == "ok"
    assert (r_cfg["instrs"], r_cfg["cycles"]) == (r_none["instrs"], r_none["cycles"])


@pytest.mark.vortex
def test_default_write_is_noop_ps(vx):
    r_def, _ = _run(vx, {"tree": "ps", "app": "ps_phase_rw", "args": "-L 5 -c default -w alt", "config": "REG-CFG"})
    r_none, _ = _run(vx, {"tree": "ps", "app": "ps_phase_rw", "args": "-L 5 -c none", "config": "REG-CFG"})
    assert r_def["status"] == "ok" and r_none["status"] == "ok"
    assert (r_def["instrs"], r_def["cycles"]) == (r_none["instrs"], r_none["cycles"])
