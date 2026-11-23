"""PS-BASE configurations and measured machine parameters (methodology §11).

Cheap tests (no SimX):
  test_psbase_configs_common          every PS-BASE config has L2 on, TCU + WGMMA, DXA, MUFU, PLRU at L1 and L2,
                                      NT = 32, NW = 8 (notes), and the stated fixed DRAM latency.
  test_psbase_per_core_resources      L2 = 256 KiB per core and DRAM bandwidth = 16 B/cycle per core in S0 and S1
                                      (methodology §7: per-core ratios held across tiers).
  test_machine_params_reproducible    results/<run>/summarize.py --check: machine_params.csv equals what the rows give
                                      and every re-run row (rep 1) equals its rep-0 twin exactly.
  test_rho_matches_a100_reference     the chosen DRAM latency puts rho = L0 * ISSUE_WIDTH / NUM_WARPS of PS-BASE-S1-4c
                                      within 10 % of the a100-like reference (the decision's own criterion).
  test_roofline_psbase_measured       model/roofline.py psbase machines are measured=True and equal the CSV.
  test_psbase_override_rewrites_spec  the tests/conftest.py PS_BASE_CONFIG override keeps only macros the named set
                                      does not define and drops structural keys.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PS = Path(__file__).resolve().parents[1]
PY = sys.executable
BASES = ["PS-BASE-S0", "PS-BASE-S1-4c", "PS-BASE-S1-8c", "PS-BASE-S1-16c", "PS-BASE-S1-4c-lat0",
         "PS-BASE-S1-4c-lat190"]
COMMON = ["-DVX_CFG_ISSUE_WIDTH=2", "-DVX_CFG_EXT_TCU_ENABLE", "-DVX_CFG_TCU_WGMMA_ENABLE", "-DVX_CFG_EXT_DXA_ENABLE",
          "-DWGMMA_NRC=32", "-DVX_CFG_EXT_MUFU_ENABLE", "-DVX_CFG_DCACHE_REPL_POLICY=2", "-DVX_CFG_L2_REPL_POLICY=2"]


def _named():
    return yaml.safe_load((PS / "experiments" / "configs.yaml").read_text())


def _macro(cfg, name, default=None):
    for t in cfg["configs"].split():
        if t.startswith(f"-D{name}="):
            return t.split("=", 1)[1]
    return default


def test_psbase_configs_common():
    nm = _named()
    for b in BASES:
        c = nm[b]
        toks = c["configs"].split()
        assert c["l2"] is True and c["threads"] == 32 and c["warps"] == 8, b
        for t in COMMON:
            assert t in toks, (b, t)
    lat = {b: int(_macro(nm[b], "VX_CFG_PLATFORM_MEMORY_LATENCY")) for b in BASES}
    assert lat == {"PS-BASE-S0": 11, "PS-BASE-S1-4c": 11, "PS-BASE-S1-8c": 11, "PS-BASE-S1-16c": 11,
                   "PS-BASE-S1-4c-lat0": 0, "PS-BASE-S1-4c-lat190": 190}
    # the sensitivity variants differ from the primary config ONLY in the fixed DRAM latency
    strip = lambda s: [t for t in s.split() if not t.startswith("-DVX_CFG_PLATFORM_MEMORY_LATENCY=")]  # noqa: E731
    for v in ("PS-BASE-S1-4c-lat0", "PS-BASE-S1-4c-lat190"):
        assert strip(nm[v]["configs"]) == strip(nm["PS-BASE-S1-4c"]["configs"]) and nm[v]["cores"] == 4


def test_psbase_per_core_resources():
    nm = _named()
    for b in BASES:
        c = nm[b]
        cores = c["cores"]
        assert int(_macro(c, "VX_CFG_L2_SIZE")) == 256 * 1024 * cores, b
        ch = int(_macro(c, "VX_CFG_PLATFORM_MEMORY_NUM_BANKS"))
        rate = float(_macro(c, "VX_CFG_PLATFORM_MEMORY_DATA_RATE", 800))
        bw = ch * 16 * rate / 400.0                      # HBM2: 16 B per transfer, core clock 400 MHz (default)
        assert bw / cores == 16.0, (b, bw, cores)


def test_machine_params_reproducible():
    r = subprocess.run([PY, "-B", str(PS / "results/P3.1/summarize.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _params():
    with open(PS / "results/P3.1/machine_params.csv") as f:
        return {r["config"]: r for r in csv.DictReader(f)}


def test_rho_matches_a100_reference():
    p = _params()
    ref, base, lat0 = p["P3.1-a100lat-ref"], p["PS-BASE-S1-4c"], p["PS-BASE-S1-4c-lat0"]
    for k in ("rho_rand", "rho_seq"):
        assert abs(float(base[k]) / float(ref[k]) - 1) <= 0.10, (k, base[k], ref[k])
    # the default model is further from the reference than the chosen one (otherwise the default would be chosen)
    mean = lambda r: (float(r["rho_rand"]) + float(r["rho_seq"])) / 2   # noqa: E731
    assert abs(mean(base) - mean(ref)) < abs(mean(lat0) - mean(ref))


def test_roofline_psbase_measured():
    from model import roofline
    p = _params()
    ms = roofline.machines()
    for c in (4, 8, 16):
        m = ms[f"psbase-S1-{c}c"]
        row = p[f"PS-BASE-S1-{c}c"]
        assert m.measured
        assert m.flops_per_cycle["tcu"] == pytest.approx(float(row["tcu_flop_per_cycle"]))
        assert m.flops_per_cycle["fpu"] == pytest.approx(float(row["fpu_flop_per_cycle"]))
        assert m.dram_bytes_per_cycle == pytest.approx(float(row["stream_bytes_per_cycle"]))
        assert "results/P3.1/machine_params.csv" in m.sources["dram_bytes_per_cycle"]


def test_psbase_override_rewrites_spec(monkeypatch):
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun
    import importlib.util
    spec = importlib.util.spec_from_file_location("ps_conftest_copy", PS / "tests" / "conftest.py")
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)
    seen = []
    orig_run = vxrun.VxRun.run
    try:
        monkeypatch.setattr(vxrun.VxRun, "run", lambda self, spec, **kw: seen.append(spec) or spec, raising=True)
        conftest._psbase_override("PS-BASE-S1-4c")
        vxrun.VxRun.run(None, {"tree": "base", "app": "x", "config": "smoke-default", "threads": 4, "cores": 1,
                               "configs": "-DVX_CFG_EXT_MUFU_ENABLE -DPS_FOO=1"}, rss_gb=1.0)
    finally:
        vxrun.VxRun.run = orig_run
    s = seen[-1]
    assert s["config"] == "PS-BASE-S1-4c" and s["configs"] == "-DPS_FOO=1"
    assert "threads" not in s and "cores" not in s and s["tags"]["psbase_from"] == "smoke-default"


def test_memhier_psbase_latencies_are_measured():
    from model import memhier as mh
    p = _params()
    for c in (4, 8, 16):
        h, row = mh.hier_for(f"psbase-S1-{c}c"), p[f"PS-BASE-S1-{c}c"]
        assert h.hit_lat == pytest.approx(float(row["l2_load_lat_est"]))
        assert h.miss_lat == pytest.approx(float(row["dram_load_lat_rand"]))
        assert h.l2_bytes == 262144 * c and h.dram_channels == c // 2 and not h.persist_across_launch
