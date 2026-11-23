"""Tests: ps patch kv-insert (mechanism D: KV_STREAM insert-at-LRU / bypass at the L2, plus the rtl_like_fill and
NVIDIA-practice window baselines). Semantics: docs/methodology.md §14.D; evidence: results/<run>/.

Non-vortex (cheap):
  test_patch_files               patches/vortex/ps/0005-kv-insert.patch exists, touches exactly the 5 SimX files, no
                                 .ps_* marker; the ps tree carries the encoding and the three guards (L2-only
                                 capability, PLRU-only + encoding check in Cluster, per-launch-static window).
  test_encoding_constants        INS_* constants of phase_config.h equal the documented encoding.
  test_stored_schedule_files     every results/<run>/sched/*.sched parses with scripts/mkschedule.py.
Vortex-marked (PS_RUN_VORTEX=1; rows of experiments/<run>.yaml in results/$PS_KVINSERT_STEP, default, same specs ->
same run keys -> reused; lowload exit 75 -> skip "DEFERRED"):
  test_wkv_thrash_case           ps_wkv_micro -w 512 -c 1024 -r 16 at PS-BASE-S1-4c: weight-region L2 hit rate
                                 >= 0.9 with D insert-at-LRU, <= 0.2 with stock (MRU) insertion; same instruction
                                 count in every mode (same binary, same work).
  test_bypass_allocates_no_kv    D bypass: KV_STREAM L2 accesses = misses = 0 (no allocation is possible without a
                                 bank miss), every KV_STREAM link read was bypassed, KV DRAM bytes unchanged.
  test_demoted_equals_kv_fills   D insert-at-LRU: l2_fill_demoted_kvs = number of KV_STREAM L2 fills = KV_STREAM DRAM
                                 read bytes / 64 (the L2 is the LLC, one 64-B DRAM read per fill); 0 in stock mode.
  test_stress_all_pass           every P2 kernel PASSES in every D mode incl. rtl_like_fill and the window baseline
                                 with every unhinted request classified KV_STREAM; and D acted in those runs.
  test_two_segment_insert        PS_SCHEDULE switches D on at 300,000 instructions: trace rows before the switch row
                                 are identical to the static run's (every counter), demoted fills are 0 there and
                                 > 0 from the switch row on.
  test_rejections                D on a FIFO L2, reserved encodings and a mid-kernel window change abort SimX.
  test_noop_regression_csv       results/<run>/noop_regression.csv: every REG entry identical (written by
                                 results/<run>/noop_compare.py compare).
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
TREE = PS / "vortex-trees" / "ps"
RES = PS / "results" / "P5.1"
PATCH = PS / "patches" / "vortex" / "ps" / "0005-kv-insert.patch"
sys.path.insert(0, str(PS / "scripts"))
import mkschedule as mk  # noqa: E402
import pstrace  # noqa: E402

STEP = os.environ.get("PS_KVINSERT_STEP", "P5.1")
LINE = 64   # L2 line = DRAM request size in every config (methodology §12.9)


# ------------------------------------------------------------------------------------------------ non-vortex
def test_patch_files():
    t = PATCH.read_text()
    files = sorted(set(re.findall(r"^diff --git a/(\S+) b/", t, re.M)))
    assert files == ["sim/common/phase_config.h", "sim/simx/cluster.cpp", "sim/simx/mem/cache.cpp",
                     "sim/simx/mem/cache.h", "sim/simx/ps_schedule.h"], files
    assert not re.search(r"^diff --git a/(.*/)?\.ps_", t, re.M)
    if not TREE.exists():
        pytest.skip("no ps tree")
    cl = (TREE / "sim/simx/cluster.cpp").read_text()
    assert "l2cache_->ps_set_insert_capable(true);" in cl
    assert "mechanism D needs PLRU (2) at the L2" in cl and "insert_policy_error(value)" in cl
    assert "is static" in (TREE / "sim/simx/ps_schedule.h").read_text()
    ch = (TREE / "sim/simx/mem/cache.h").read_text()
    assert "return ps_insert_capable_ ? phase_cfg_.insert_policy : 0u;" in ch


def test_encoding_constants():
    h = (TREE / "sim/common/phase_config.h").read_text() if TREE.exists() else pytest.skip("no ps tree")
    want = {"INS_D_MASK": 0x3, "INS_D_LRU": 1, "INS_D_BYPASS": 2, "INS_RTL_FILL": 0x10, "INS_WINDOW": 0x100,
            "INS_WIN_STATIC_MASK": 0x1FFFF00, "INS_VALID_MASK": 0x1FFF113}
    for k, v in want.items():
        m = re.search(rf"\b{k} = (0x[0-9a-fA-F]+|\d+)", h)
        assert m and int(m.group(1), 0) == v, k
    # valid mask = D bits | rtl bit | window bit | set-aside ways (12..15) | hitRatio (16..24)
    assert 0x3 | 0x10 | 0x100 | (0xF << 12) | (0x1FF << 16) == 0x1FFF113


def test_stored_schedule_files():
    files = sorted((RES / "sched").glob("*.sched"))
    assert len(files) >= 14
    for f in files:
        mk.parse(f.read_text())


# ------------------------------------------------------------------------------------------------ vortex
def _batch_specs():
    import yaml
    import vxrun
    doc = yaml.safe_load((PS / "experiments" / "P5.1.yaml").read_text())
    return [{k: v for k, v in sp.items() if not k.startswith("_")}
            for sp in vxrun.expand_batch(doc, vxrun.load_named_configs()) if sp.get("rep", 0) == 0]


@pytest.fixture(scope="module")
def vx():
    import vxrun
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), str(PS / "build" / "vortex-ps")],
                       capture_output=True, text=True, env=dict(os.environ, PS=str(PS)))
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun(STEP)


def _specs(**tags):
    return [s for s in _batch_specs() if all(str((s.get("tags") or {}).get(k)) == str(v) for k, v in tags.items())]


def _one(**tags):
    got = _specs(**tags)
    assert len(got) == 1, (tags, len(got))
    return got[0]


def _run(vx, spec, expect_ok=True):
    s = {k: v for k, v in spec.items() if k not in ("timeout", "mem", "reps")}
    s.setdefault("env", {})
    row = vx.run(s, timeout_s=int(spec.get("timeout", 7200)))
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    if expect_ok:
        assert row["status"] == "ok", (row["status"], row.get("reason"), row["log"])
    return row, (PS / row["log"]).read_text(errors="replace")


def _w(group, mode, vx):
    row, txt = _run(vx, _one(group=group, mode=mode))
    return row, pstrace.parse_perf_ps(txt)


def wgt_hit(p):
    return 1.0 - p["l2_miss_wgt"] / p["l2_acc_wgt"]


@pytest.mark.vortex
def test_wkv_thrash_case(vx):
    r0, p0 = _w("W", "0x0", vx)
    r1, p1 = _w("W", "0x1", vx)
    assert p0["l2_acc_wgt"] == p1["l2_acc_wgt"] == 16 * 512 * 1024 // LINE   # 16 rounds x the 512 KiB block
    assert wgt_hit(p0) <= 0.2, wgt_hit(p0)
    assert wgt_hit(p1) >= 0.9, wgt_hit(p1)
    for m in ("0x2", "0x10", "0x11", "0x12", "0x100"):
        r, _ = _w("W", m, vx)
        assert r["instrs"] == r0["instrs"], m


@pytest.mark.vortex
def test_bypass_allocates_no_kv(vx):
    _, p0 = _w("W", "0x0", vx)
    _, p2 = _w("W", "0x2", vx)
    assert p2["l2_acc_kvs"] == 0 and p2["l2_miss_kvs"] == 0
    assert p2["l2_fill_demoted_kvs"] == 0
    assert p2["l2_fill_bypassed_kvs"] == p2["l1l2_rd_bytes_kvs"] // LINE > 0
    assert p2["dram_rd_bytes_kvs"] == p0["dram_rd_bytes_kvs"] == 16 * 1024 * 1024
    assert wgt_hit(p2) >= 0.9


@pytest.mark.vortex
def test_demoted_equals_kv_fills(vx):
    _, p0 = _w("W", "0x0", vx)
    _, p1 = _w("W", "0x1", vx)
    assert p1["l2_fill_demoted_kvs"] == p1["dram_rd_bytes_kvs"] // LINE == p1["l2_miss_kvs"] > 0
    assert sum(v for k, v in p1.items() if k.startswith("l2_fill_demoted_") and k != "l2_fill_demoted_kvs") == 0
    assert sum(v for k, v in p0.items() if k.startswith(("l2_fill_demoted_", "l2_fill_bypassed_"))) == 0


@pytest.mark.vortex
def test_stress_all_pass(vx):
    specs = _specs(group="F")
    assert len(specs) == 14 * 7
    acted = {"0x1": 0, "0x2": 0, "0x11": 0, "0x12": 0, "0x800100": 0}
    for sp in specs:
        row, txt = _run(vx, sp)
        assert "PASSED" in txt and "FAILED" not in txt.split("PASSED")[-1], row["log"]
        p = pstrace.parse_perf_ps(txt)
        m = sp["tags"]["mode"]
        if m in ("0x1", "0x11", "0x800100"):
            acted[m] += p["l2_fill_demoted_kvs"] + p["l2_fill_demoted_wgt"]
        if m in ("0x2", "0x12"):
            acted[m] += p["l2_fill_bypassed_kvs"]
    assert all(v > 0 for v in acted.values()), acted


@pytest.mark.vortex
def test_two_segment_insert(vx):
    sp = _one(group="T", case="twoseg300k")
    st = _one(group="T", case="static")
    W, n = 300000, int(sp["env"]["PS_TRACE_EVERY"])
    _run(vx, st)
    row, txt = _run(vx, sp)
    names, rows = pstrace.read_trace(sp["env"]["PS_TRACE"])
    _, rows_s = pstrace.read_trace(st["env"]["PS_TRACE"])
    w, ws = pstrace.work_rows(rows, 0), pstrace.work_rows(rows_s, 0)
    k_sw = next(i for i, r in enumerate(w) if r["work_end"] >= W)
    assert W % n == 0 and w[k_sw]["work_end"] == W
    sw = [m for m in re.findall(r"^PSSCHED launch=0 line=\d+ work_count=(\d+) work=(\d+) cycle=(\d+)", txt, re.M)
          if int(m[0]) == W]
    assert len(sw) == 1 and int(sw[0][2]) == w[k_sw]["cycle"]   # switch in the cycle that closes row k_sw
    first_new = k_sw + 1
    for i in range(first_new):                                     # identical prefix (every counter)
        assert w[i] == ws[i], i
    dem = [r["l2_fill_demoted_kvs"] for r in w]
    assert all(v == 0 for v in dem[:first_new]) and dem[first_new] > 0
    assert sum(dem[first_new:]) > 0 and sum(r["l2_fill_demoted_kvs"] for r in ws) == 0
    assert row["cycles"] != _run(vx, st)[0]["cycles"]               # the switch changed the kernel


@pytest.mark.vortex
@pytest.mark.parametrize("case,msg", [("fifo", "needs PLRU (2) at the L2"), ("reserved", "D mode 3 is reserved"),
                                      ("win_and_d", "window baseline and D are exclusive"),
                                      ("win_switch", "static per launch")])
def test_rejections(vx, case, msg):
    row, txt = _run(vx, _one(group="X", case=case), expect_ok=False)
    assert row["status"] != "ok" and "PASSED!" not in txt
    assert msg in txt, row["log"]


@pytest.mark.vortex
def test_noop_regression_csv():
    rows = list(csv.DictReader((RES / "noop_regression.csv").open()))
    assert len(rows) >= 8 and all(r["identical"] == "True" for r in rows), rows
