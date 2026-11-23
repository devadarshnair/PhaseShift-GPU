"""Tests: ps patch waymask (mechanism A1: per-launch LMEM size via VX_DCR_KMU_LMEM_SIZE with kernel tile variants;
mechanism A2: [dcr_phase] way masks restricting victim selection at the L1 dcaches and the L2). Semantics:
docs/methodology.md §14.A; evidence: results/<run>/.

Non-vortex (cheap):
  test_patch_files            patches/vortex/ps/0011-waymask.patch touches exactly the 8 SimX files, nothing in third_party/.
  test_schedule_rejects       scripts/mkschedule.py and SimX's PS_SCHEDULE parser reject l1_waymask / l2_waymask / lmem
                              (a mask change inside a launch); the stored reject_* schedules fail to parse.
  test_select_victim_mask     cache.cpp: select_victim takes a way_mask argument and processFill passes the fill's mask.
Vortex-marked (PS_RUN_VORTEX=1; rows of experiments/<run>.yaml in results/<run>; same specs -> same run keys -> reused):
  test_waymask_unit           g++ (via lowload.sh) of tests/fixtures/ps_waymask_unit.cpp: encoding, legality, host helper
                              (refuses a mask change inside a declared schedule), host/SimX register agreement.
  test_victims_allowed_ways   masks {all, 1/2, 1/4} (+ a non-low 1/2) on L1 and on L2, two apps: violations = 0, the
                              victim ways used are exactly the allowed ways (non-vacuous), the other level untouched;
                              the explicit all-ways mask gives cycles identical to no mask (same victims).
  test_weight_hit_rate        ps_wkv_micro -w 512 -c 1024 -r 16: unmasked weight-region L2 hit rate <= 0.2; weights on
                              4 reserved ways (others on the other 4) >= 0.9 (the earlier size, no resize needed); the run
                              also shows the host helper refusing a mask change inside a schedule (-X).
  test_l1_partition_measured  the 4-way L1 analogue (Vantage warning, C41): reported, only violations = 0 asserted.
  test_a1_tile_variants       decode (-t) and mixed (-w) tile variants 8/16/32, unmasked and under L1 1/4 + L2 1/4:
                              PASSED, SimX's PSLMEM lmem_size = the host's lmem_bytes, >= 2 distinct sizes per app.
  test_functional_masked      other kernel variants under L1 1/4 + L2 1/4: PASSED, violations = 0.
  test_rejections             schedule mask switch, mask + insert policy, insert switched on under a mask, mask bits
                              >= associativity: the run fails with the rejection message (since the earlier patch
                              0013-compose the two mask + D-mode cases run and PASS instead).
  test_default_cells_unchanged unmasked default cells equal earlier stored ps rows in instrs and cycles.
  test_noop_regression_csv    results/<run>/noop_regression.csv: every REG entry identical (base, ps 0002-0010).
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
TREE = PS / "vortex-trees" / "ps"
RES = PS / "results" / "P5.4"
PATCH = PS / "patches" / "vortex" / "ps" / "0011-waymask.patch"
sys.path.insert(0, str(PS / "scripts"))
sys.path.insert(0, str(PS / "tests"))
import mkschedule as mk  # noqa: E402
import test_dram_2class as t52  # noqa: E402  (helpers: _run)

FILES = ["sim/common/phase_config.h", "sim/simx/cluster.cpp", "sim/simx/kmu/kmu.h", "sim/simx/mem/cache.cpp",
         "sim/simx/mem/cache.h", "sim/simx/processor.cpp", "sim/simx/processor_impl.h", "sim/simx/socket.cpp"]
FULL = {"l1": 0xF, "l2": 0xFF}
_run = t52._run


# ------------------------------------------------------------------------------------------------ cheap
def test_patch_files():
    assert PATCH.exists()
    got = sorted(set(re.findall(r"^diff --git a/(\S+) b/", PATCH.read_text(), re.M)))
    assert got == sorted(FILES), got
    assert not any(p.startswith("third_party/") for p in got)


def test_schedule_rejects():
    for knob in ("l1_waymask", "l2_waymask", "lmem", "kmu_lmem"):
        with pytest.raises(mk.ScheduleError):
            mk.parse(f"0 {knob}=1\n")
    for f in ("reject_l2_waymask.sched", "reject_l1_waymask.sched"):
        with pytest.raises(mk.ScheduleError):
            mk.parse((RES / "sched" / f).read_text())
    mk.parse((RES / "sched" / "insert_d1_1k.sched").read_text())
    src = (TREE / "sim/simx/ps_schedule.h").read_text()
    assert 'k == "l1_waymask" || k == "l2_waymask"' in src and "needs a flush" in src


def test_select_victim_mask():
    src = (TREE / "sim/simx/mem/cache.cpp").read_text()
    assert re.search(r"int select_victim\(uint8_t policy, uint32_t rand_idx,\s*int \*free_line_id, int \*repl_line_id, "
                     r"uint32_t way_mask = 0xffffffffu\)", src)
    assert "set.select_victim(config_.repl_policy, rand_ctr_, &free_id, &repl_id, wm);" in src
    assert "PhaseConfig::wm_effective(wm_word, set.lines.size(), wgt)" in src


# ------------------------------------------------------------------------------------------------ vortex
def _batch_specs(name="P5.4.yaml"):
    import yaml
    import vxrun
    doc = yaml.safe_load((PS / "experiments" / name).read_text())
    return [{k: v for k, v in sp.items() if not k.startswith("_")}
            for sp in vxrun.expand_batch(doc, vxrun.load_named_configs()) if sp.get("rep", 0) == 0]


@pytest.fixture(scope="module")
def vx():
    import vxrun
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), str(PS / "build" / "vortex-ps")],
                       capture_output=True, text=True, env=dict(os.environ, PS=str(PS)))
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun("P5.4")


def _specs(**tags):
    return [s for s in _batch_specs() if all(str((s.get("tags") or {}).get(k)) == str(v) for k, v in tags.items())]


def _one(**tags):
    got = _specs(**tags)
    assert len(got) == 1, (tags, len(got))
    return got[0]


def wm_lines(txt):
    """{level: dict} of the last launch's PSWAYMASK lines (hex words as ints, lists for reqs/misses)."""
    out = {}
    for line in re.findall(r"^PSWAYMASK launch=\d+ (.*)$", txt, re.M):
        d = {}
        for k, v in re.findall(r"(\w+)=(\S+)", line):
            if k == "level":
                continue
            d[k] = [int(x) for x in v.split(",")] if "," in v else int(v, 0)
        out[re.search(r"level=(\w+)", line).group(1)] = d
    return out


def weight_hit(d):
    return 1.0 - d["misses"][1] / d["reqs"][1]   # class 1 = WEIGHT (VX_PS_CLASS_WEIGHT)


@pytest.mark.vortex
def test_waymask_unit():
    out = PS / "build" / "tmp" / "p54_waymask_unit"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    exe = out / "ps_waymask_unit"
    cmd = ["g++", "-std=c++17", "-Wall", "-Wextra", "-pedantic", "-Werror", f"-I{PS / 'build/vortex-ps/sw'}",
           f"-I{TREE / 'sim/common'}", f"-I{TREE / 'sw/runtime/include'}", f"-I{PS / 'kernels/common'}",
           str(PS / "tests/fixtures/ps_waymask_unit.cpp"), "-o", str(exe)]
    env = dict(os.environ, PS_RESULTS_DIR=str(out), PS_STEP="P5.4-test")
    r = subprocess.run([str(PS / "scripts/lowload.sh"), "--rss", "0.5", "--mem", "4", "--label", "P5.4-test:wm_unit",
                        "--log", str(out / "build.log"), "--", *cmd], env=env, capture_output=True, text=True)
    if r.returncode == 75:
        pytest.skip("DEFERRED: lowload.sh exit 75")
    assert r.returncode == 0, (out / "build.log").read_text(errors="replace")
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0 and "waymask_unit: OK (0 errors)" in r.stdout, r.stdout


@pytest.mark.vortex
def test_victims_allowed_ways(vx):
    specs = _specs(group="V")
    assert len(specs) == 2 * 9
    cyc = {}
    for sp in specs:
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt
        t = sp["tags"]
        cyc[(sp["app"], t["level"], t["mask"])] = row["cycles"]
        wm = wm_lines(txt)
        assert set(wm) == {"l1d", "l2"}
        for lv, d in wm.items():
            assert d["violations"] == 0, (sp["args"], lv, d)
            lev = "l1" if lv == "l1d" else "l2"
            if t["level"] == lev:
                m = d["word"] & 0xFFFF
                assert d["fills"] > 0 and d["ways_used"] == m, (sp["args"], lv, hex(d["ways_used"]), hex(m))
            else:   # the other level: no mask, no counted fills
                assert d["word"] == 0 and d["fills"] == 0 and d["ways_used"] == 0
    for app in {s["app"] for s in specs}:
        for lev in ("l1", "l2"):
            assert cyc[(app, lev, "all")] == cyc[(app, "none", "none")], (app, lev)


@pytest.mark.vortex
def test_weight_hit_rate(vx):
    row0, txt0 = _run(vx, _one(group="H", level="l2", part="none"))
    row1, txt1 = _run(vx, _one(group="H", level="l2", part="w_half"))
    assert "PASSED!" in txt0 and "PASSED!" in txt1
    h0, h1 = weight_hit(wm_lines(txt0)["l2"]), weight_hit(wm_lines(txt1)["l2"])
    assert h0 <= 0.2, h0
    assert h1 >= 0.9, h1
    d1 = wm_lines(txt1)["l2"]
    assert d1["violations"] == 0 and d1["fills_wgt"] > 0
    assert row0["instrs"] == row1["instrs"]
    assert re.search(r"^PSWAYMASK-REFUSED rc=-78 l1=0x0 l2=0xf0000f$", txt1, re.M)


@pytest.mark.vortex
def test_l1_partition_measured(vx):
    for sp in _specs(group="H", level="l1"):
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt
        assert wm_lines(txt)["l1d"]["violations"] == 0


@pytest.mark.vortex
def test_a1_tile_variants(vx):
    specs = _specs(group="A")
    assert len(specs) == 12
    sizes = {}
    for sp in specs:
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt
        host = int(re.search(r"lmem_bytes=(\d+)", txt).group(1))
        sim = [int(x) for x in re.findall(r"^PSLMEM launch=\d+ lmem_size=(\d+)", txt, re.M)]
        assert sim and all(s == host for s in sim), (sp["args"], host, sim)
        sizes.setdefault(sp["tags"]["app_k"], set()).add(host)
        for d in wm_lines(txt).values():
            assert d["violations"] == 0
    assert all(len(v) >= 2 for v in sizes.values()), sizes


@pytest.mark.vortex
def test_functional_masked(vx):
    specs = _specs(group="F")
    assert len(specs) == 4
    for sp in specs:
        row, txt = _run(vx, sp)
        assert "PASSED!" in txt
        wm = wm_lines(txt)
        assert wm["l1d"]["violations"] == wm["l2"]["violations"] == 0
        assert wm["l1d"]["ways_used"] == 0x1 and wm["l2"]["ways_used"] == 0x3


@pytest.mark.vortex
def test_rejections(vx):
    want = {"sched_l2": "'l2_waymask' rejected", "sched_l1": "'l1_waymask' rejected",
            "mask_plus_insert": "exclusive with a non-zero L2 insert policy",
            "sched_insert_under_mask": "rejected (schedule switch): exclusive with a non-zero L2 insert policy",
            "l1_bit_ge_ways": "bits 15:0 name a way >= associativity",
            "l2_wbit_ge_ways": "bits 31:16 name a way >= associativity"}
    specs = _specs(group="X")
    assert {s["tags"]["case"] for s in specs} == set(want)
    # (ps patch 0013-compose, methodology §14.E) made an L2 mask legal together with D's own modes (insert bits
    # 1:0), so the two "mask + insert=1" cases now run (PASS) on a tree with that patch; every other rejection stands.
    # The mask x non-D insert policies (rtl_like / window / SRRIP / SHiP) stay rejected: tests/test_compose.py.
    composed = (PS / "patches/vortex/ps/0013-compose.patch").exists()
    for sp in specs:
        if composed and sp["tags"]["case"] in ("mask_plus_insert", "sched_insert_under_mask"):
            row, txt = _run(vx, sp, expect_ok=True)
            assert "PASSED!" in txt
            continue
        row, txt = _run(vx, sp, expect_ok=False)
        assert row["status"] != "ok", sp["tags"]
        assert want[sp["tags"]["case"]] in txt, (sp["tags"], row["log"])
        assert "PASSED!" not in txt


@pytest.mark.vortex
def test_default_cells_unchanged(vx):
    """The -W option and the new header leave the default path alone: unmasked cells equal the earlier stored rows."""
    ref = [json.loads(l) for l in (PS / "results/P5.1/rows.jsonl").read_text().splitlines() if l.strip()]
    ref = [r for r in ref if r.get("status") == "ok" and r.get("config") == "PS-BASE-S1-4c"
           and "PS_SCHEDULE" not in (r.get("env") or {})]
    n = 0
    for sp in _specs(group="V", level="none") + _specs(group="H", level="l2", part="none"):
        row, txt = _run(vx, sp)
        cands = [r for r in ref if r["app"] == sp["app"] and r["args"] in (sp["args"], sp["args"] + " -I 0x0")]
        assert cands, sp["args"]
        assert all((r["instrs"], r["cycles"]) == (row["instrs"], row["cycles"]) for r in cands), \
            (sp["args"], row["instrs"], row["cycles"], [(r["instrs"], r["cycles"]) for r in cands])
        n += 1
    assert n == 3


def test_noop_regression_csv():
    p = RES / "noop_regression.csv"
    if not p.exists():
        pytest.skip("results/P5.4/noop_regression.csv not yet produced")
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 8 and all(r["identical"] == "True" for r in rows)
