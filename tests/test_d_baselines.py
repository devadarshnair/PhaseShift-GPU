"""Tests: ps patch d-baselines (phase-ignorant D baselines at the L2: SRRIP [C34] and SHiP-Mem-lite [C35]).
Semantics: docs/methodology.md §14.D (part); evidence: results/<run>/.

Non-vortex (cheap):
  test_patch_files               patches/vortex/ps/0006-d-baselines.patch exists, touches exactly the 3 SimX files, no
                                 .ps_* marker.
  test_encoding_constants        INS_REPL_* of phase_config.h equal the documented encoding, disjoint from the earlier fields.
  test_policies_phase_ignorant   static review aid: the SRRIP / SHiP code paths of cache.cpp read no
                                 request class, range, phase-config field or schedule input (the class is used only
                                 to index the demoted-fill COUNTER).
  test_stored_schedule_files     every results/<run>/sched/*.sched parses with scripts/mkschedule.py.
Vortex-marked (PS_RUN_VORTEX=1; rows of experiments/<run>.yaml in results/$PS_DBASE_STEP, default; same specs
-> same run keys -> reused; lowload exit 75 -> skip "DEFERRED"):
  test_scan_resistance           ps_scan_micro -w 512 -c 1024 -r 16 -p 1 (re-used 512 KiB = 1/2 L2 block interleaved
                                 with a one-pass 16 MiB = 16 x L2 scan): weight-region L2 hit rate >= 0.8 with SRRIP,
                                 <= 0.2 with stock LRU; same instruction count in every mode.
  test_ship_shct_kv_zero         SHiP-Mem-lite run: every SHCT entry of the KV regions of the first two scan chunks
                                 (passes) reaches 0 before the third chunk is first touched, and so does every KV
                                 region of chunk r before chunk r+2 starts (r <= rounds-3); the weight-region entries
                                 end > 0 (end-of-launch dump); distant insertions are counted (demoted fills > 0).
  test_functional_all_pass       every P2 kernel PASSES with SRRIP and with SHiP-Mem-lite for the whole launch, and
                                 SHiP used its table in every run.
  test_two_segment               PS_SCHEDULE switches SHiP (resp. SRRIP) on at 300,000 instructions: trace rows before
                                 the switch row are identical to the static run (every counter); SHiP distant fills
                                 are 0 there and > 0 after; the kernel cycles change.
  test_rejections                FIFO L2, reserved baseline 3, and SRRIP/SHiP combined with D, rtl_like or the window
                                 abort SimX.
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
RES = PS / "results" / "P5.1b"
PATCH = PS / "patches" / "vortex" / "ps" / "0006-d-baselines.patch"
sys.path.insert(0, str(PS / "scripts"))
import mkschedule as mk  # noqa: E402
import pstrace  # noqa: E402

STEP = os.environ.get("PS_DBASE_STEP", "P5.1b")
LINE = 64
SHCT_REGION = 1 << 14


# ------------------------------------------------------------------------------------------------ non-vortex
def test_patch_files():
    t = PATCH.read_text()
    files = sorted(set(re.findall(r"^diff --git a/(\S+) b/", t, re.M)))
    assert files == ["sim/common/phase_config.h", "sim/simx/mem/cache.cpp", "sim/simx/mem/cache.h"], files
    assert not re.search(r"^diff --git a/(.*/)?\.ps_", t, re.M)


def test_encoding_constants():
    h = (TREE / "sim/common/phase_config.h").read_text() if TREE.exists() else pytest.skip("no ps tree")
    want = {"INS_REPL_MASK": 0x60, "INS_REPL_SRRIP": 0x20, "INS_REPL_SHIP": 0x40, "INS_VALID_MASK": 0x1FFF113}
    for k, v in want.items():
        m = re.search(rf"\b{k} = (0x[0-9a-fA-F]+|\d+)", h)
        assert m and int(m.group(1), 0) == v, k
    assert 0x60 & 0x1FFF113 == 0                       # disjoint from the kv-insert fields
    assert "replacement baseline 3 is reserved" in h and "exclusive with D, rtl_like_fill and the window" in h


def _added_code(patch_text):
    return [l[1:] for l in patch_text.splitlines() if l.startswith("+") and not l.startswith("+++")]


def test_policies_phase_ignorant():
    """Every added non-comment line of cache.cpp that belongs to the SRRIP / SHiP paths must not read a class, range,
    window or schedule input. The only class use allowed is the demoted-fill counter index."""
    t = PATCH.read_text()
    cpp = t.split("diff --git a/sim/simx/mem/cache.cpp")[1].split("diff --git")[0]
    bad = []
    for l in _added_code(cpp):
        code = l.split("//")[0]
        if not code.strip():
            continue
        if re.search(r"mem_class|range_class|\bkv\[|\.wgt\b|contains\(|ins_window|ins_d\(|PS_SCHEDULE|ps_sched|"
                     r"VX_PS_CLASS", code):
            # allowed: the demoted-fill counter index, and the earlier window dispatch that the patch turned into the
            # `else` branch of `if (ps_repl != 0)` (taken only when neither baseline is selected)
            if "fill_demoted[ps_cls" not in code and code.strip() != "} else if (ps_ins != 0 && ps_owner_->phase_config().ins_window()) {":
                bad.append(l)
    assert not bad, bad
    # the class variable is only an index of the accounting counter
    uses = [l for l in _added_code(cpp) if "ps_cls" in l.split("//")[0]]
    assert uses and all("fill_demoted[ps_cls" in l for l in uses), uses


def test_stored_schedule_files():
    files = sorted((RES / "sched").glob("*.sched"))
    assert len(files) >= 10
    for f in files:
        mk.parse(f.read_text())


# ------------------------------------------------------------------------------------------------ vortex
def _batch_specs():
    import yaml
    import vxrun
    doc = yaml.safe_load((PS / "experiments" / "P5.1b.yaml").read_text())
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
    s = {k: v for k, v in spec.items() if k not in ("timeout", "mem", "reps", "rss")}
    s.setdefault("env", {})
    row = vx.run(s, timeout_s=int(spec.get("timeout", 7200)), rss_gb=spec.get("rss"))
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    if expect_ok:
        assert row["status"] == "ok", (row["status"], row.get("reason"), row["log"])
    return row, (PS / row["log"]).read_text(errors="replace")


def wgt_hit(p):
    return 1.0 - p["l2_miss_wgt"] / p["l2_acc_wgt"]


@pytest.mark.vortex
def test_scan_resistance(vx):
    got = {}
    for m in ("0x0", "0x1", "0x2", "0x20", "0x40"):
        row, txt = _run(vx, _one(group="S", mode=m))
        got[m] = (row, pstrace.parse_perf_ps(txt))
    p0, ps_ = got["0x0"][1], got["0x20"][1]
    assert p0["l2_acc_wgt"] == ps_["l2_acc_wgt"] == 17 * 512 * 1024 // LINE   # 16 rounds + 1 warm-up pass
    assert p0["l2_acc_kvs"] == 16 * 1024 * 1024 // LINE                       # the one-pass scan, 16 x L2
    assert wgt_hit(p0) <= 0.2, wgt_hit(p0)
    assert wgt_hit(ps_) >= 0.8, wgt_hit(ps_)
    assert len({r["instrs"] for r, _ in got.values()}) == 1


def _shct(txt):
    first = {int(a): int(b) for a, b in re.findall(r"^PSSHCT first idx=(\d+) cycle=(\d+)", txt, re.M)}
    zero = {int(a): int(b) for a, b in re.findall(r"^PSSHCT zero idx=(\d+) cycle=(\d+)", txt, re.M)}
    dump = {int(a): int(v) for a, v in re.findall(r"^PSSHCT dump idx=(\d+) val=(\d+)", txt, re.M)}
    return first, zero, dump


@pytest.mark.vortex
def test_ship_shct_kv_zero(vx):
    row, txt = _run(vx, _one(group="S", mode="0x40"))
    m = re.search(r"^PSRANGE: wgt=0x([0-9a-f]+)-0x([0-9a-f]+) kv0=0x([0-9a-f]+)-0x([0-9a-f]+)", txt, re.M)
    wb, wl, kb, kl = (int(x, 16) for x in m.groups())
    chunk, rounds = 1024 * 1024, 16
    assert kl - kb == rounds * chunk and kb % SHCT_REGION == 0 and wb % SHCT_REGION == 0
    first, zero, dump = _shct(txt)
    per = chunk // SHCT_REGION
    idx = [[(kb + r * chunk) // SHCT_REGION + j for j in range(per)] for r in range(rounds)]
    start = [min(first[i] for i in idx[r]) for r in range(rounds)]
    for r in range(rounds - 2):                       # includes the first two passes (r = 0, 1)
        late = [i for i in idx[r] if i not in zero or zero[i] >= start[r + 2]]
        assert not late, (r, late[:5])
    widx = range(wb // SHCT_REGION, wl // SHCT_REGION)
    assert all(dump[i] > 0 for i in widx), [dump[i] for i in widx]
    assert all(dump[i] == 0 for r in range(rounds - 2) for i in idx[r])
    p = pstrace.parse_perf_ps(txt)
    assert p["l2_fill_demoted_kvs"] > 0


@pytest.mark.vortex
def test_functional_all_pass(vx):
    specs = _specs(group="F")
    assert len(specs) == 15 * 2
    for sp in specs:
        row, txt = _run(vx, sp)
        assert "PASSED" in txt and "FAILED" not in txt.split("PASSED")[-1], row["log"]
        if sp["tags"]["mode"] == "0x40":
            assert "PSSHCT first" in txt and "PSSHCT dump" in txt, row["log"]
        else:
            assert "PSSHCT" not in txt, row["log"]


@pytest.mark.vortex
@pytest.mark.parametrize("case", ["twoseg_ship300k", "twoseg_srrip300k"])
def test_two_segment(vx, case):
    sp = _one(group="T", case=case)
    st = _one(group="T", case="static")
    W, n = 300000, int(sp["env"]["PS_TRACE_EVERY"])
    rs, _ = _run(vx, st)
    row, txt = _run(vx, sp)
    _, rows = pstrace.read_trace(sp["env"]["PS_TRACE"])
    _, rows_s = pstrace.read_trace(st["env"]["PS_TRACE"])
    w, ws = pstrace.work_rows(rows, 0), pstrace.work_rows(rows_s, 0)
    k_sw = next(i for i, r in enumerate(w) if r["work_end"] >= W)
    assert W % n == 0 and w[k_sw]["work_end"] == W
    sw = [m for m in re.findall(r"^PSSCHED launch=0 line=\d+ work_count=(\d+) work=(\d+) cycle=(\d+)", txt, re.M)
          if int(m[0]) == W]
    assert len(sw) == 1 and int(sw[0][2]) == w[k_sw]["cycle"]
    first_new = k_sw + 1
    for i in range(first_new):
        assert w[i] == ws[i], i
    assert any(w[i] != ws[i] for i in range(first_new, min(len(w), len(ws))))
    dem = [sum(v for k, v in r.items() if k.startswith("l2_fill_demoted_")) for r in w]
    assert all(v == 0 for v in dem[:first_new])
    if "ship" in case:
        assert sum(dem[first_new:]) > 0 and "PSSHCT first" in txt
    else:
        assert sum(dem) == 0 and "PSSHCT" not in txt
    assert row["cycles"] != rs["cycles"]


@pytest.mark.vortex
@pytest.mark.parametrize("case,msg", [("fifo", "needs PLRU (2) at the L2"),
                                      ("reserved", "replacement baseline 3 is reserved"),
                                      ("srrip_and_d", "exclusive with D, rtl_like_fill and the window"),
                                      ("ship_and_rtl", "exclusive with D, rtl_like_fill and the window"),
                                      ("srrip_and_window", "exclusive with D, rtl_like_fill and the window")])
def test_rejections(vx, case, msg):
    row, txt = _run(vx, _one(group="X", case=case), expect_ok=False)
    assert row["status"] != "ok" and "PASSED!" not in txt
    assert msg in txt, row["log"]


@pytest.mark.vortex
def test_noop_regression_csv():
    rows = list(csv.DictReader((RES / "noop_regression.csv").open()))
    assert len(rows) >= 8 and all(r["identical"] == "True" for r in rows), rows
