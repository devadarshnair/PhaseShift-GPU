"""Tests: ps patch mem-class (request class, mem_hint / ps_mark CSRs, KV address ranges).

Non-vortex (always run, cheap):
  test_csr_numbers_custom_range     vortex-trees/ps/VX_types.toml [csr_ps]: both CSRs inside 0x800-0x8FF (RISC-V custom
                                    user read/write), no other VX_CSR_* in 0x800-0x8FF, ps_hint.h agrees.
  test_class_constants_agree        [ps_mem_class] / [ps_mark_type] / VX_PHASE_FLAG_RANGE_ACCT_ONLY == ps_hint.h /
                                    ps_phase.h / scripts/psclass.py tables.
  test_static_asserts_present       sim/simx/types.h asserts mem_class at bits 21-23 (bit_cast on the field list).
  test_hint_macros_noop             ps_hint.h: without -DPS_HINTS=1 the macros expand to ((void)0) (preprocessor only).
  test_elf_identity_recorded        results/<run>/elf_sha_pre.txt == elf_sha_post.txt (PS_HINTS=0 kernels byte-identical
                                    to the pre-hint build) and the PS_HINTS=1 ELFs differ.
  test_psclass_parser               scripts/psclass.py on a synthetic report.
  test_reg_has_decode_attn          experiments/noop_regression.yaml now holds ps_decode_attn (PS_HINTS=0) in `plan`.
  test_marks_single_lane            every PS_MARK in the kernels' device sources and headers runs on one lane (`if (tid
                                    == 0)`, `const uint32_t tid = threadIdx.x;`, no aliases); hardened (carry-over 4); test apps listed in MARK_LANE_EXEMPT are exempt.

Vortex-marked (PS_RUN_VORTEX=1). SimX runs go through scripts/vxrun.py -> lowload.sh (serial) with the specs of
experiments/<run>.yaml, into results/$PS_MEMCLASS_STEP (default, so rows of the TEST batch are reused; set another
id to re-run everything):
  test_mem_flags_layout_dxa_on_off  host g++ (lowload) of tests/fixtures/mem_flags_layout.cpp with DXA off and on: both
                                    compile (static_asserts) and print the same raw values for every class.
  test_class_bytes_sum[...]         PS_DEBUG_CLASS: per-class DRAM bytes (rd + wr) sum to the independent total
                                    (MPM mem_reads + mem_writes) x block, and the region table sums to it too.
  test_decode_kv_labelled           ps_decode_attn PS_HINTS=1, ranges accounting-only: >= 99 % of the DRAM bytes inside
                                    the KV range are KV_* (here: KV_STREAM); PS_HINTS=0 with ranges classifying: also
                                    >= 99 %; PS_HINTS=0 accounting-only control: 0 % (so the label comes from the hint).
  test_mixed_kv_labelled            ps_mixed_attn PS_HINTS=1: decode KV (kv0) bytes KV_STREAM, prefill KV (kv1) bytes
                                    KV_PERSIST, each >= 99 %; PS_HINTS=0 + ranges likewise.
  test_dxa_bulk[C|D]                gqa_dxa_dbuf: every DXA GMEM request is BULK_DXA (and > 0), every DRAM byte inside
                                    the KV range is BULK_DXA (also when the ranges classify), and with an L2 the L2 sees
                                    exactly the DXA requests as BULK_DXA.
  test_marks_match_host[...]        ps_mark counts (PSMARK-TOTAL) == the host's expectation (PSHINT) for gqa, naive,
                                    dxa, mixed and ragged mixed; PS_HINTS=0 runs count 0.
  test_hint_builds_pass             every hint build run PASSED (part of every test above via _ok).
  test_debug_and_class_no_timing    within one binary, PS_DEBUG_CLASS on/off and -R none/kv/kvacct give identical
                                    instrs/cycles (classes are carried and counted only in this patch).
  test_hint_build_aborts_on_base    the PS_HINTS=1 binary on the base tree aborts on the unknown CSR (by design).
lowload exit 75 -> pytest.skip("DEFERRED").
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
TREE = PS / "vortex-trees" / "ps"
TOML = TREE / "VX_types.toml"
HINT_H = PS / "kernels" / "common" / "ps_hint.h"
PHASE_H = PS / "kernels" / "common" / "ps_phase.h"
sys.path.insert(0, str(PS / "scripts"))
import psclass  # noqa: E402

STEP = os.environ.get("PS_MEMCLASS_STEP", "P4.2")
GQA = "-B 2 -H 8 -K 4 -D 32 -T 128 -v gqa"
NAIVE = "-B 1 -H 8 -K 4 -D 16 -T 64 -v naive"
MIXED = "-H 8 -K 4 -D 32 -c 16 -o 16 -T 64 -n 2"
RAGGED = "-H 8 -K 4 -D 16 -c 8 -T 17,40"
DXA = "-B 4 -H 8 -K 4 -D 16 -T 256 -v dxa -p 2"
GQA_C = "-B 2 -H 8 -K 4 -D 16 -T 128 -v gqa"
DBG = {"PS_DEBUG_CLASS": "1"}
A = {"config": "REG-CFG", "configs": "-DPS_HINTS=1"}
B = {"config": "REG-CFG"}
C = {"config": "smoke-dxa", "configs": "-DPS_HINTS=1"}
D = {"config": "REG-CFG", "configs": "-DVX_CFG_EXT_DXA_ENABLE -DPS_HINTS=1"}


def _toml_section(name: str) -> dict:
    out, cur = {}, None
    for line in TOML.read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if s.startswith("[") and s.endswith("]"):
            cur = s.strip("[]")
            continue
        if cur == name and "=" in s:
            k, v = (x.strip() for x in s.split("=", 1))
            out[k] = int(v, 0)
    return out


def _all_csrs() -> dict:
    out = {}
    for line in TOML.read_text().splitlines():
        m = re.match(r"\s*(VX_CSR_\w+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2), 0)
    return out


def _hdr_define(path: Path, name: str) -> int:
    m = re.search(rf"^#define {name} (0x[0-9A-Fa-f]+|\d+)u?\b", path.read_text(), re.M)
    assert m, f"{name} not defined in {path}"
    return int(m.group(1), 0)


# --------------------------------------------------------------------------------------------- cheap tests
needs_tree = pytest.mark.skipif(not TOML.exists(), reason="vortex-trees/ps not created")


@needs_tree
def test_csr_numbers_custom_range():
    csr = _toml_section("csr_ps")
    assert set(csr) == {"VX_CSR_PS_MEM_HINT", "VX_CSR_PS_MARK"}
    for k, v in csr.items():
        assert 0x800 <= v <= 0x8FF, (k, hex(v))
        assert (v >> 10) & 3 == 2 and (v >> 8) & 3 == 0, f"{k} must be user read/write custom (bits 11:8 = 1000)"
    others = {k: v for k, v in _all_csrs().items() if k not in csr and 0x800 <= v <= 0x8FF}
    assert not others, others
    assert len(set(csr.values())) == 2
    assert _hdr_define(HINT_H, "PS_CSR_MEM_HINT") == csr["VX_CSR_PS_MEM_HINT"]
    assert _hdr_define(HINT_H, "PS_CSR_MARK") == csr["VX_CSR_PS_MARK"]


@needs_tree
def test_class_constants_agree():
    cls = _toml_section("ps_mem_class")
    marks = _toml_section("ps_mark_type")
    for name in ("ORDINARY", "WEIGHT", "KV_PERSIST", "KV_STREAM", "BULK_DXA"):
        assert _hdr_define(HINT_H, f"PS_CLASS_{name}") == cls[f"VX_PS_CLASS_{name}"]
        assert psclass.CLASSES[cls[f"VX_PS_CLASS_{name}"]] == name
    assert cls["VX_PS_NUM_CLASSES"] == 8 == len(psclass.CLASSES)
    assert set(psclass.KV_CLASSES) == {cls["VX_PS_CLASS_KV_PERSIST"], cls["VX_PS_CLASS_KV_STREAM"]}
    for name in ("OTHER", "PREFILL_TILE", "DECODE_TILE"):
        assert _hdr_define(HINT_H, f"PS_MARK_{name}") == marks[f"VX_PS_MARK_{name}"]
        assert psclass.MARKS[marks[f"VX_PS_MARK_{name}"]] == name
    flag = _toml_section("phase_const")["VX_PHASE_FLAG_RANGE_ACCT_ONLY"]
    m = re.search(r"FLAG_RANGE_ACCT_ONLY = (0x[0-9a-fA-F]+);", PHASE_H.read_text())
    assert m and int(m.group(1), 16) == flag == 0x2


@needs_tree
def test_static_asserts_present():
    t = (TREE / "sim" / "simx" / "types.h").read_text()
    assert "MEM_CLASS_SHIFT = 21" in t and "MEM_CLASS_BITS  = 3" in t
    assert "uint32_t ps_pad_dxa        : 17;" in t           # the 17-bit pad of the non-DXA branch
    assert '__builtin_bit_cast(uint32_t, with_class(1)) == (1u << 21)' in t
    assert 'static_assert(sizeof(MemFlagsLayout) == sizeof(uint32_t)' in t


def _preprocess(defs: list[str]) -> str:
    cmd = ["g++", "-E", "-P", "-x", "c++", "-D__VORTEX__", *defs, "-"]
    src = f'#include "{HINT_H}"\nHINT: PS_MEM_HINT(PS_CLASS_KV_STREAM);\nMARK: PS_MARK(PS_MARK_DECODE_TILE);\n'
    r = subprocess.run(cmd, input=src, capture_output=True, text=True, check=True)
    return r.stdout


def test_hint_macros_noop():
    off = _preprocess([])
    assert "HINT: ((void)0);" in off and "MARK: ((void)0);" in off
    off0 = _preprocess(["-DPS_HINTS=0"])
    assert "HINT: ((void)0);" in off0
    on = _preprocess(["-DPS_HINTS=1"])
    assert 'csrw " "0x8C0" ", %0"' in on.replace("\n", " ") or "csrw" in on
    assert "((void)0)" not in on.split("HINT:", 1)[1].split("\n", 1)[0]


def test_elf_identity_recorded():
    pre, post, h1 = (PS / "results" / "P4.2" / f"elf_sha_{t}.txt" for t in ("pre", "post", "hints1"))
    if not pre.exists():
        pytest.skip("results/P4.2/elf_sha_pre.txt not generated yet")
    a, b = pre.read_text().split("\n"), post.read_text().split("\n")
    assert a == b and len([x for x in a if x]) == 6
    hints = {tuple(x.split()[:2]): x.split()[2] for x in h1.read_text().split("\n") if x}
    for x in a:
        if x:
            app, cfg, sha = x.split()[:3]
            assert hints[(app, cfg)] != sha, "PS_HINTS=1 ELF must differ from the PS_HINTS=0 ELF"


SYN = """noise
PSHINT: ps_hints=1 expect_marks=0,0,8
PSCLASS launch=0 when=end block=64 dram_total_bytes=640 dram_rd_bytes=64,0,0,512,0,0,0,0 dram_wr_bytes=64,0,0,0,0,0,0,0
PSCLASS-REGION launch=0 when=end region=kv0 bytes=0,0,0,512,0,0,0,0
PSCLASS-REGION launch=0 when=end region=kv1 bytes=0,0,0,0,0,0,0,0
PSCLASS-REGION launch=0 when=end region=wgt bytes=0,0,0,0,0,0,0,0
PSCLASS-REGION launch=0 when=end region=other bytes=128,0,0,0,0,0,0,0
PSCLASS-CACHE launch=0 when=end level=l2 reqs=3,0,0,8,0,0,0,0 mixed_merges=1 prio_upgrades=0
PSCLASS-DXA launch=0 when=end gmem_reqs=0,0,0,0,0,0,0,0
PSMARK launch=0 when=end core=0 counts=0,0,8,0,0,0,0,0
PSMARK-TOTAL launch=0 when=end counts=0,0,8,0,0,0,0,0
PSCLASS launch=0 when=flush block=64 dram_total_bytes=704 dram_rd_bytes=64,0,0,512,0,0,0,0 dram_wr_bytes=128,0,0,0,0,0,0,0
PSCLASS-REGION launch=0 when=flush region=kv0 bytes=0,0,0,512,0,0,0,0
PSCLASS-REGION launch=0 when=flush region=other bytes=192,0,0,0,0,0,0,0
PSMARK-TOTAL launch=0 when=flush counts=0,0,8,0,0,0,0,0
"""


def test_psclass_parser():
    reps = psclass.parse_all(SYN)
    assert [(r["launch"], r["when"]) for r in reps] == [(0, "end"), (0, "flush")]
    r = psclass.parse(SYN)
    assert r["when"] == "flush" and r["dram_total"] == 704
    d = psclass.derived(r)
    assert d["class_sum"] == 704 == d["region_sum"] and d["kv_labelled_frac"] == 1.0
    assert psclass.parse_pshint(SYN) == {"ps_hints": 1, "expect_marks": [0, 0, 8]}
    e = psclass.derived(reps[0])
    assert e["class_sum"] == 640 == e["region_sum"]
    assert reps[0]["caches"]["l2"]["mixed_merges"] == 1


# Test apps that write VX_CSR_PS_MARK from many lanes ON PURPOSE (not validated kernels); each entry says why.
MARK_LANE_EXEMPT = {
    "ps_marklanes": "carry-over 3: unguarded / divergent marks check the trace->sop count (one per warp instr)",
}


def _strip_comments(text: str) -> str:
    """C/C++ source without // and /* */ comments (string literals in kernels never contain them); newlines kept so
    line numbers still match."""
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def test_marks_single_lane():
    """ (errata 33b), hardened (carry-over 4). Every PS_MARK use in an
    validated kernel's device code executes on exactly one lane of the CTA: `if (tid == 0) PS_MARK(PS_MARK_*);` where
    the enclosing function defines `const uint32_t tid = threadIdx.x;` (no alias such as `threadIdx.x % 4u`). Patch
    ps-counters counts one unit per warp CSR instruction (trace->sop), so the guard is what makes a mark mean one unit
    per (CTA, tile). Scanned: every .cpp/.h/.hpp/.c under kernels/ (sources AND headers, incl. kernels/common), except
    host main.cpp files, the macro definition in kernels/common/ps_hint.h and the test apps in MARK_LANE_EXEMPT.
    Limits: only uses of the PS_MARK macro are checked -- a raw `csrw 0x8c1` / VX_CSR_PS_MARK access
    would not be seen -- and `threadIdx.x == 0` is one lane per CTA only for 1-D CTAs (a 2-D/3-D CTA has one such thread
    per row); neither occurs in the current kernels."""
    found = 0
    files = [f for f in sorted((PS / "kernels").rglob("*")) if f.suffix in (".cpp", ".h", ".hpp", ".c")]
    assert files
    for f in files:
        rel = f.relative_to(PS / "kernels")
        if f.name == "main.cpp" or rel.as_posix() == "common/ps_hint.h" or rel.parts[0] in MARK_LANE_EXEMPT:
            continue
        code = _strip_comments(f.read_text(errors="replace"))
        lines = code.splitlines()
        for n, line in enumerate(lines, 1):
            if "PS_MARK(" not in line:
                continue
            found += 1
            assert re.fullmatch(r"\s*if \(tid == 0\) PS_MARK\(PS_MARK_[A-Z_]+\);\s*", line), f"{rel}:{n}: {line.strip()}"
            # the enclosing function (nearest preceding function head) defines tid as exactly threadIdx.x
            head = "\n".join(lines[:n - 1])
            starts = [m.start() for m in re.finditer(r"\b(always_inline|__kernel)\b", head)]
            assert starts, f"{rel}:{n}: no enclosing device function found"
            fn = head[starts[-1]:]
            # every assignment/initialisation of tid in the function (not `tid == ...`) must be exactly threadIdx.x
            defs = re.findall(r"\btid\s*=(?!=)\s*([^,;)]+)", fn)
            assert defs and all(d.strip() == "threadIdx.x" for d in defs), f"{rel}:{n}: tid is not threadIdx.x: {defs}"
            assert re.search(r"const uint32_t tid = threadIdx\.x\s*[,;]", fn), \
                f"{rel}:{n}: tid is not declared `const uint32_t tid = threadIdx.x`"
    assert found == 4, found     # decode_attn_gqa (1) + decode_attn_gqa kernel_kv.cpp (1, same `if (tid == 0)` form) + mixed_attn prefill and decode (2)
    for app in MARK_LANE_EXEMPT:  # the exemption list names real test apps only
        assert (PS / "kernels" / app / "kernel.cpp").exists(), app


def test_reg_has_decode_attn():
    import yaml
    reg = yaml.safe_load((PS / "experiments" / "noop_regression.yaml").read_text())
    dec = [r for r in reg["runs"] if r["app"] == "ps_decode_attn_gqa"]
    assert dec and dec[0]["tags"]["reg"] == "plan" and dec[0]["config"] == "REG-CFG"
    for r in reg["runs"]:
        assert "PS_HINTS=1" not in str(r.get("configs", "")), "REG must use PS_HINTS=0 builds"


# --------------------------------------------------------------------------------------------- vortex tests
@pytest.fixture(scope="module")
def vx():
    import vxrun
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], capture_output=True, text=True,
                       env=dict(os.environ, PS=str(PS)))
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun(STEP)


def _run(vx, app, args, cfg, env=DBG, tree="ps"):
    spec = {"tree": tree, "app": app, "args": args, "perf": 1, "env": dict(env), **cfg}
    row = vx.run(spec, timeout_s=3600)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


def _ok(row, log):
    assert row["status"] == "ok" and row["passed"], (row["status"], row["log"])
    rep = psclass.parse(log)
    assert rep is not None, f"no PS_DEBUG_CLASS report in {row['log']}"
    return rep


@pytest.mark.vortex
def test_mem_flags_layout_dxa_on_off(tmp_path):
    gen = PS / "build" / "vortex-ps" / "sw"
    outs = {}
    for dxa in (False, True):
        exe = tmp_path / f"layout_{int(dxa)}"
        cmd = ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{gen}", f"-I{TREE / 'sim/simx'}",
               f"-I{TREE / 'sim/common'}", f"-I{TREE / 'sw/common'}", "-DVX_CFG_XLEN=32", "-DVX_CFG_XLEN_32",
               *(["-DVX_CFG_EXT_DXA_ENABLE"] if dxa else []), str(PS / "tests/fixtures/mem_flags_layout.cpp"),
               "-o", str(exe)]
        env = dict(os.environ, PS_RESULTS_DIR=str(tmp_path), PS_STEP="P4.2-test")
        r = subprocess.run([str(PS / "scripts/lowload.sh"), "--rss", "0.8", "--mem", "4", "--label",
                            f"P4.2-test:layout{int(dxa)}", "--", *cmd], capture_output=True, text=True, env=env)
        if r.returncode == 75:
            pytest.skip("DEFERRED: lowload.sh exit 75")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        outs[dxa] = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    lines_off = [x for x in outs[False].splitlines() if not x.startswith("dxa_sideband")]
    lines_on = [x for x in outs[True].splitlines() if not x.startswith("dxa_sideband")]
    assert lines_off == lines_on
    for c in range(8):
        assert f"mem_class={c} raw=0x{c << 21:08x}" in lines_off
    side = re.search(r"dxa_sideband raw=0x([0-9a-f]+)", outs[True])
    assert side and int(side.group(1), 16) == ((1 << 4) | (0xFFFF << 5)) and int(side.group(1), 16) & (7 << 21) == 0


SUM_CASES = [
    ("ps_decode_attn_gqa", f"{GQA} -R kvacct", A), ("ps_decode_attn_gqa", f"{GQA} -R none", A),
    ("ps_decode_attn_gqa", f"{NAIVE} -R kvacct", A), ("ps_mixed_attn", f"{MIXED} -R kvacct", A),
    ("ps_mixed_attn", f"{RAGGED} -R kvacct", A), ("ps_decode_attn_gqa", f"{GQA} -R kv", B),
    ("ps_decode_attn_gqa", GQA, B), ("ps_mixed_attn", f"{MIXED} -R kv", B),
    ("ps_decode_attn_gqa", f"{DXA} -R kvacct", C), ("ps_decode_attn_gqa", f"{GQA_C} -R kvacct", C),
    ("ps_decode_attn_gqa", f"{DXA} -R kvacct", D),
]


@pytest.mark.vortex
@pytest.mark.parametrize("app,args,cfg", SUM_CASES, ids=[f"{a[3:]}:{b}:{c.get('configs', '')}" for a, b, c in SUM_CASES])
def test_class_bytes_sum(vx, app, args, cfg):
    row, log = _run(vx, app, args, cfg)
    rep = _ok(row, log)
    for r in psclass.parse_all(log):          # every report (end of launch and after the flush)
        d = psclass.derived(r)
        assert d["class_sum"] == d["dram_total"] > 0, (r["when"], d)
        assert d["region_sum"] == d["dram_total"], (r["when"], d)
    assert rep["when"] == "flush"


@pytest.mark.vortex
def test_decode_kv_labelled(vx):
    # PS_HINTS=1, ranges accounting-only: the label can only come from the hints
    rep = _ok(*_run(vx, "ps_decode_attn_gqa", f"{GQA} -R kvacct", A))
    d = psclass.derived(rep)
    assert d["kv_region_total"] >= 2 * 2 * 128 * 4 * 32 * 4 * 0.99    # E3 bytes of K and V (B T H_kv D 4 each)
    assert d["kv_labelled_frac"] >= 0.99, d
    assert d["kv_region_bytes"][3] == d["kv_region_total"]            # all KV_STREAM
    # PS_HINTS=0 build, ranges classify
    d0 = psclass.derived(_ok(*_run(vx, "ps_decode_attn_gqa", f"{GQA} -R kv", B)))
    assert d0["kv_labelled_frac"] >= 0.99, d0
    # PS_HINTS=0, accounting only: no label at all (control)
    dc = psclass.derived(_ok(*_run(vx, "ps_decode_attn_gqa", f"{GQA} -R kvacct", B)))
    assert dc["kv_region_total"] == d0["kv_region_total"] and dc["kv_labelled_frac"] == 0.0, dc
    # 1-core config without L2 (smoke-dxa build, gqa variant): the hint also reaches DRAM through L1 alone
    dl = psclass.derived(_ok(*_run(vx, "ps_decode_attn_gqa", f"{GQA_C} -R kvacct", C)))
    assert dl["kv_labelled_frac"] >= 0.99, dl


@pytest.mark.vortex
def test_mixed_kv_labelled(vx):
    for args, cfg in ((f"{MIXED} -R kvacct", A), (f"{RAGGED} -R kvacct", A), (f"{MIXED} -R kv", B)):
        rep = _ok(*_run(vx, "ps_mixed_attn", args, cfg))
        kv0, kv1 = rep["regions"]["kv0"], rep["regions"]["kv1"]   # decode KV, prefill KV
        assert sum(kv0) > 0 and sum(kv1) > 0, (args, kv0, kv1)
        assert kv0[3] >= 0.99 * sum(kv0), (args, "decode KV must be KV_STREAM", kv0)
        assert kv1[2] >= 0.99 * sum(kv1), (args, "prefill KV must be KV_PERSIST", kv1)


@pytest.mark.vortex
@pytest.mark.parametrize("cfg", [C, D], ids=["smoke-dxa", "REG-CFG+DXA"])
def test_dxa_bulk(vx, cfg):
    rep = _ok(*_run(vx, "ps_decode_attn_gqa", f"{DXA} -R kvacct", cfg))
    dxa = rep["dxa"]
    assert dxa is not None and dxa[4] > 0 and sum(dxa) == dxa[4], dxa     # every DXA GMEM request is BULK_DXA
    d = psclass.derived(rep)
    assert d["kv_region_total"] > 0 and d["kv_bulk_dxa_frac"] == 1.0, d   # all KV bytes at DRAM came from DXA
    if cfg is D:
        assert rep["caches"]["l2"]["reqs"][4] == dxa[4], (rep["caches"]["l2"], dxa)
    if cfg is C:
        # ranges that classify do not relabel DXA requests (they only classify ORDINARY ones)
        rk = _ok(*_run(vx, "ps_decode_attn_gqa", f"{DXA} -R kv", cfg))
        dk = psclass.derived(rk)
        assert dk["kv_bulk_dxa_frac"] == 1.0 and sum(rk["dxa"]) == rk["dxa"][4], dk


MARK_CASES = [("ps_decode_attn_gqa", f"{GQA} -R kvacct", A), ("ps_decode_attn_gqa", f"{GQA} -R none", A),
              ("ps_decode_attn_gqa", f"{NAIVE} -R kvacct", A), ("ps_decode_attn_gqa", f"{DXA} -R kvacct", C),
              ("ps_decode_attn_gqa", f"{DXA} -R kvacct", D), ("ps_mixed_attn", f"{MIXED} -R kvacct", A),
              ("ps_mixed_attn", f"{RAGGED} -R kvacct", A), ("ps_decode_attn_gqa", f"{GQA} -R kv", B),
              ("ps_mixed_attn", f"{MIXED} -R kv", B)]


@pytest.mark.vortex
@pytest.mark.parametrize("app,args,cfg", MARK_CASES,
                         ids=[f"{a[3:]}:{b}:{c.get('configs', '')}" for a, b, c in MARK_CASES])
def test_marks_match_host(vx, app, args, cfg):
    row, log = _run(vx, app, args, cfg)
    rep = _ok(row, log)
    hint = psclass.parse_pshint(log)
    assert hint is not None
    hints_on = "PS_HINTS=1" in cfg.get("configs", "")
    assert hint["ps_hints"] == int(hints_on)
    exp = hint["expect_marks"] + [0] * (8 - len(hint["expect_marks"]))
    end = [r for r in psclass.parse_all(log) if r["when"] == "end"][-1]
    assert end["marks_total"] == exp, (end["marks_total"], exp)
    assert rep["marks_total"] == exp
    if hints_on:
        assert sum(exp) > 0
        assert sum(sum(v) for v in end["marks"].values()) == sum(exp)


@pytest.mark.vortex
def test_debug_and_class_no_timing(vx):
    # hint binary: PS_DEBUG_CLASS on vs off, ranges accounting-only
    r1, _ = _run(vx, "ps_decode_attn_gqa", f"{GQA} -R kvacct", A)
    r0, _ = _run(vx, "ps_decode_attn_gqa", f"{GQA} -R kvacct", A, env={})
    r2, _ = _run(vx, "ps_decode_attn_gqa", f"{GQA} -R none", A)
    # PS_HINTS=0 binary: no range / classifying ranges / accounting-only ranges, debug on
    b = [_run(vx, "ps_decode_attn_gqa", a, B)[0] for a in (GQA, f"{GQA} -R kv", f"{GQA} -R kvacct")]
    for rows in ((r0, r1, r2), b):
        for r in rows:
            assert r["status"] == "ok" and r["passed"]
        assert len({(r["instrs"], r["cycles"]) for r in rows}) == 1, [(r["instrs"], r["cycles"]) for r in rows]


@pytest.mark.vortex
def test_hint_build_aborts_on_base(vx):
    row, log = _run(vx, "ps_decode_attn_gqa", f"{GQA} -R none", A, env={}, tree="base")
    assert row["status"] != "ok" and not row.get("passed")
    assert "invalid CSR" in log, row["log"]
