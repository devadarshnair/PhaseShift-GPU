"""Tests: element-wise baseline repair -- RoPE angle table and SiLU exp variants (methodology §18.7).

Non-vortex (always run; no SimX; one tiny g++ compile):
  test_default_kernels_untouched      kernels/{rope,silu_gate}/kernel.cpp (the validated device code) are byte-identical to their
                                      sources (sha256), so the default ps_llama kernels are unchanged; the repair lives in new files
  test_new_entries_and_symlinks       kernel_tab.cpp / kernel_exp.cpp / rope_table_ref.h exist; ps_rope and ps_silu_gate list them in
                                      app.conf VX_SRCS; ps_llama has 12 translation units (appended two), k_ep1.cpp the tenth, and its reused/ symlinks resolve
  test_table_arg_structs              ps_rope_arg_t unchanged in size (88 B); the two new structs are 8-byte aligned
  test_host_table_reference           rope_table_ref.h compiled with g++: pos 0 -> (1, 0) exactly, the host table passes its own check,
                                      a 1-ulp-flipped / NaN / swapped entry FAILS (negative control), the PSTABLE line counts it
  test_batch_file_up_to_date          experiments/<run>.yaml == results/<run>/make_batch.py output
  test_ep1_results                    results/<run>/summary.json (when present): every stored row ok; the claims of results/<run>/README.md
  test_methodology_18_7               modeling choices documented (table, near-tie rule, bounds)

Vortex-marked (PS_RUN_VORTEX=1; each SimX run via scripts/vxrun.py -> lowload.sh, serial, private results dir build/tmp/ep1_test,
deleted at session start; lowload exit 75 -> skip "DEFERRED"). Do not wrap this pytest in lowload.sh (vxrun does it per run):
  test_table_entries[...]             rope_table at REG-CFG for hs = 8 (stories260K, 512 positions) and windows of hs = 48 / 128: PASSED, every
                                      entry within 0.5 + 2^-9 ulp of the exact value (PSCHECK rope_table_vs_host errors = 0)
  test_probe_equals_table[...]        rope_kv's own per-launch angles (`ps_rope -A`) have the SAME digest as the table of the same window
  test_rope_tab_bit_identical         rope_kv_tab outputs: q, K/V cache slots and the whole cache arrays bit-identical to rope_kv on stories260K
                                      fixtures (B = 1 at pos 0/17/max-1 and B = 4), digests compared
  test_table_position_outside_refused a position beyond the table is refused (the table check is live)
  test_silu_variants_fixture          `-E poly` (REG-CFG) and `-E mufu` (PS-BASE-S0) within their derived bounds; mufu refused without the unit;
                                      a tight `-t` fails
  test_ps_llama_table_identical       ps_llama -R table on stories260K: logits dump byte-identical to the default mode, in tf and in prefill mode
  test_ps_llama_exp_variants          -R table -E poly (REG-CFG) and -E mufu (PS-BASE-S0) pass the token rule and the 1e-3 logits bound
"""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS))
FIX = PS / "build" / "fixtures"
RES = PS / "results" / "EP.1"
PSCHECK_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) tol=(\S+) errors=(\d+)", re.M)
PSDIGEST_RE = re.compile(r"^PSDIGEST: (\S+) n=(\d+) fnv1a64=([0-9a-f]{16})$", re.M)

SHA_ROPE = "cc3f2f6217dfe196620752ff56511eb1930407d17dfd43640f4804f954c94b03"      # kernels/rope/kernel.cpp
SHA_SILU = "a4f07bf7503d159b47422a33f3a9a30868df1ffcfbfbcdc366ce5422a085c8a7"      # kernels/silu_gate/kernel.cpp


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_default_kernels_untouched():
    assert _sha(PS / "kernels/rope/kernel.cpp") == SHA_ROPE
    assert _sha(PS / "kernels/silu_gate/kernel.cpp") == SHA_SILU
    # the new entries are not in the default translation units
    assert "rope_table" not in (PS / "kernels/rope/kernel.cpp").read_text()
    assert "silu_gate_expf" not in (PS / "kernels/silu_gate/kernel.cpp").read_text()


def test_new_entries_and_symlinks():
    for f in ("rope/kernel_tab.cpp", "rope/rope_table_ref.h", "silu_gate/kernel_exp.cpp"):
        assert (PS / "kernels" / f).is_file(), f
    assert (PS / "kernels/rope/app.conf").read_text().split() == ["VX_SRCS=kernel.cpp", "kernel_tab.cpp", "kernel_kv8.cpp"]   # appended kernel_kv8.cpp
    assert (PS / "kernels/silu_gate/app.conf").read_text().split() == ["VX_SRCS=kernel.cpp", "kernel_exp.cpp"]
    ll = PS / "runtime/llama_vx"
    conf = (ll / "app.conf").read_text()
    srcs = re.search(r"^VX_SRCS=(.*)$", conf, re.M).group(1).split()
    # 12 (k_kv8r.cpp, k_kv8a.cpp linked after k_ep1.cpp); the earlier claim is that k_ep1.cpp follows the nine stock units
    assert len(srcs) == 12 and srcs[9] == "k_ep1.cpp" and srcs[10:] == ["k_kv8r.cpp", "k_kv8a.cpp"] and srcs[:9] == [
        "kernel.cpp", "k_rmsnorm.cpp", "k_rope.cpp", "k_silu.cpp", "k_residual.cpp", "k_attn.cpp", "k_linear.cpp", "k_pattn.cpp", "k_q80.cpp"]
    assert "reused/*/*.cpp" in conf and "reused/*/*.h" in conf     # DEPS: the new included .cpp files rebuild the app
    for f in ("rope/kernel_tab.cpp", "rope/rope_table_ref.h", "silu_gate/kernel_exp.cpp", "silu_gate/warp_ops.h"):
        link = ll / "reused" / f
        assert link.is_symlink() and link.resolve().is_file(), f
        assert link.resolve() == (PS / "kernels" / f).resolve() or f.endswith("warp_ops.h")
    k = (ll / "k_ep1.cpp").read_text()
    assert '#include "reused/rope/kernel_tab.cpp"' in k and '#include "reused/silu_gate/kernel_exp.cpp"' in k
    # the earlier nine units are unchanged in what they include
    assert '#include "reused/rope/kernel.cpp"' in (ll / "k_rope.cpp").read_text()
    assert '#include "reused/silu_gate/kernel.cpp"' in (ll / "k_silu.cpp").read_text()
    for e in ("rope_table", "rope_kv_tab"):
        assert f"__kernel void {e}(" in (PS / "kernels/rope/kernel_tab.cpp").read_text()
    for e in ("silu_gate_expf_poly", "silu_gate_expf_mufu"):
        assert f"__kernel void {e}(" in (PS / "kernels/silu_gate/kernel_exp.cpp").read_text()


HOST_CPP = textwrap.dedent(r'''
    #include <cstdio>
    #include <cstdlib>
    #include "rope_table_ref.h"
    int main(int argc, char** argv) {
      const uint32_t hs = (uint32_t)std::atoi(argv[1]), npos = (uint32_t)std::atoi(argv[2]);
      const int mode = std::atoi(argv[3]);
      ps_rope::TableRef r = ps_rope::host_table(hs, npos, 10000.0f);
      std::vector<float> dev = r.cs;
      if (mode == 1) { uint32_t b; std::memcpy(&b, &dev[5], 4); b += 4; std::memcpy(&dev[5], &b, 4); }   // 4 ulp up: unambiguous
      if (mode == 2) dev[7] = std::nanf("");
      if (mode == 3) std::swap(dev[2], dev[3]);                                                      // cos <-> sin of pair 1
      bool ok = ps_rope::check_table("t_", dev.data(), hs, npos, 10000.0f);
      std::printf("RESULT %d cs0=%.9g cs1=%.9g\n", (int)ok, (double)r.cs[0], (double)r.cs[1]);
      return 0;
    }''')


@pytest.fixture(scope="module")
def host_exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("ropeh")
    src = d / "h.cpp"
    src.write_text(HOST_CPP)
    exe = d / "h"
    r = subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-I", str(PS / "kernels/common"), "-I", str(PS / "kernels/rope"),
                        str(src), "-o", str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def _host(exe, hs, npos, mode):
    return subprocess.run([str(exe), str(hs), str(npos), str(mode)], capture_output=True, text=True, check=True).stdout


def test_host_table_reference(host_exe):
    for hs, npos in ((8, 64), (48, 40), (128, 9)):
        out = _host(host_exe, hs, npos, 0)
        assert "RESULT 1 cs0=1 cs1=0" in out                      # pos 0: val = 0 -> (cos, sin) = (1, 0) exactly
        assert re.search(r"errors=0\b", out) and re.search(r"mismatches=0\b", out)
    # negative controls: the check is live
    for mode in (1, 2, 3):
        out = _host(host_exe, 8, 64, mode)
        assert "RESULT 0" in out, (mode, out)
        c = PSCHECK_RE.search(out)
        assert c and int(c.group(7)) >= 1
    assert re.search(r"mismatches=1\b", _host(host_exe, 8, 64, 1))
    assert "PSTABLE-MISMATCH" in _host(host_exe, 8, 64, 1)
    # the host model is a correctly rounded float angle chain: compare against a float64 numpy recomputation of the same float32 chain
    hs, npos = 48, 40
    e = (np.arange(0, hs, 2, dtype=np.float32) / np.float32(hs)).astype(np.float32)
    pw = np.float32(10000.0) ** 0 * np.array([np.float32(10000.0 ** float(x)) for x in e], dtype=np.float32)   # float64 pow, rounded
    freq = (np.float32(1.0) / pw).astype(np.float32)
    val = (np.arange(npos, dtype=np.float32)[:, None] * freq[None, :]).astype(np.float32)
    assert val[0].tolist() == [0.0] * (hs // 2) and abs(float(np.cos(val[7, 3]) - np.cos(np.float64(val[7, 3])))) < 1e-6


def test_table_arg_structs(tmp_path):
    src = tmp_path / "s.cpp"
    src.write_text('#include <cstdio>\n#include "common.h"\nint main(){std::printf("%zu %zu %zu %zu\\n", sizeof(ps_rope_arg_t), '
                   'sizeof(ps_rope_table_arg_t), sizeof(ps_rope_tab_arg_t), alignof(ps_rope_tab_arg_t));}\n')
    exe = tmp_path / "s"
    r = subprocess.run(["g++", "-std=c++17", "-I", str(PS / "kernels/rope"), str(src), "-o", str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    sizes = subprocess.run([str(exe)], capture_output=True, text=True).stdout.split()
    assert sizes == ["88", "24", "96", "8"], sizes          # rope_kv's argument struct is unchanged (88 B)


def test_batch_file_up_to_date():
    r = subprocess.run([sys.executable, "-B", str(RES / "make_batch.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, "experiments/EP.1.yaml differs from results/EP.1/make_batch.py yaml"


def test_methodology_18_7():
    meth = (PS / "docs" / "methodology.md").read_text()
    assert "### 18.7 " in meth
    sec = meth[meth.index("### 18.7 "):]
    nxt = re.search(r"\n### 18\.8 |\n## 19\.", sec)
    sec = sec[:nxt.start()] if nxt else sec
    for s in ("rope_table", "rope_kv_tab", "bit-identical", "correctly rounded", "0.5 + 2^-9", "numerics-changing", "near-tie", "expf_poly",
              "expf_mufu", "model-load cost", "c_L", "PSTABLE", "§14.0"):
        assert s in sec, s


def test_ep1_results():
    p = RES / "summary.json"
    if not p.is_file():
        pytest.skip("results/EP.1/summary.json not generated yet (analyze.py)")
    j = json.loads(p.read_text())
    assert j["rows_total"] == j["rows_ok"] and j["rows_total"] >= 100, (j["rows_total"], j["rows_ok"])
    t = j["table"]
    assert t["entries_checked"] > 0 and t["check_errors"] == 0 and t["probe_digest_mismatches"] == 0 and t["cross_machine_digest_mismatches"] == 0
    assert t["tolerated_mismatches_all_within_tol"] is True
    f = j["fixtures"]
    assert f["rope_pairs"] == 16 and f["rope_digest_mismatches"] == 0
    assert all(v["errors"] == 0 for v in j["silu"]["checks"])
    assert j["llama"]["table_logits_identical"] is True and j["llama"]["default_instrs_identical_to_epf_reval"] is True
    # re-based default check (the earlier re-validation rows are the reference, not the pre-ordering-fix rows): identical instructions, cycles within 0.01 %
    d15 = j["llama"]["default_cycles_vs_epf"]
    assert d15["logits_file_identical"] and d15["instrs"] == d15["epf_instrs"] and abs(d15["cycles_delta_pct"]) <= 0.01, d15
    d6 = j["default_rebase_260K"]
    assert d6["instrs_identical"] and d6["pstok_argmax_top1_top2_instrs_identical"] and abs(d6["cycles_delta_pct"]) <= 0.01, d6
    # every PSCHECK line of every stored row has its bit-exact PSDIGEST line
    assert j["pscheck_lines"] > 500 and j["pscheck_without_digest"] == 0, (j["pscheck_lines"], j["pscheck_without_digest"])


# --------------------------------------------------------------------------------------------- vortex (SimX)
@pytest.fixture(scope="module")
def vx():
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun  # noqa: E402
    priv = PS / "build" / "tmp" / "ep1_test"
    if priv.exists():
        shutil.rmtree(priv)
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh")], env=dict(os.environ, PS=str(PS)), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return vxrun.VxRun("EP.1-test", results=priv)


def _run(vx, app, args, config="REG-CFG", tree="ps", perf=1, rss=1.0):
    row = vx.run(dict(tree=tree, perf=perf, app=app, args=args, config=config), timeout_s=7200, rss_gb=rss)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    return row, (PS / row["log"]).read_text(errors="replace")


def _checks(log):
    return {m.group(1): dict(rel=float(m.group(4)), tol=float(m.group(6)), errors=int(m.group(7))) for m in PSCHECK_RE.finditer(log)}


def _digests(log):
    return {m.group(1): m.group(3) for m in PSDIGEST_RE.finditer(log)}


def _tab_digest(log, prefix):
    return _digests(log)[f"{prefix}rope_table_vs_host"]


TAB_CASES = [(8, 512, 0, 512), (48, 256, 0, 32), (128, 2048, 2016, 32)]      # (hs, seq_len, pos0, npos)


@pytest.mark.vortex
@pytest.mark.parametrize("hs,S,p0,n", TAB_CASES, ids=[f"hs{c[0]}_p{c[2]}_n{c[3]}" for c in TAB_CASES])
def test_table_entries(vx, hs, S, p0, n):
    row, log = _run(vx, "ps_rope", f"-R setup -b 1 -d {hs} -m {hs} -H {hs} -S {S} -N 1 -l 0 -Q {p0} -Z {n}")
    assert row["status"] == "ok", row["reason"]
    c = _checks(log)
    assert c["table_rope_table_vs_host"]["errors"] == 0
    m = re.search(rf"^PSTABLE: table_ hs={hs} pos0={p0} npos={n} entries={n * hs} bits_equal_to_correct_rounding=(\d+) mismatches=(\d+) "
                  r"max_mismatch_ulp=(\S+)", log, re.M)
    assert m, log[-1500:]
    assert int(m.group(2)) <= 2 and float(m.group(3)) <= 1.0001     # a mismatch is a near-tie neighbour (<= 1 ulp from the correctly rounded float)
    # probe (rope_kv's per-launch angles) has the same digest as the table
    row2, log2 = _run(vx, "ps_rope", f"-A -b {n} -H {hs} -d {hs} -m {hs} -S {S} -Q {p0}")
    assert row2["status"] == "ok", row2["reason"]
    assert _tab_digest(log2, "probe_") == _tab_digest(log, "table_")


def test_hs8_probe_count():   # sanity of the digest regex against a log line
    assert _digests("PSDIGEST: table_rope_table_vs_host n=4096 fnv1a64=35ac3c531f03f6c6\n") == {"table_rope_table_vs_host": "35ac3c531f03f6c6"}


ROPE_FX = [("L0_B1_P0_rope_kv", 0), ("L0_B1_P17_rope_kv", 17), ("L4_B1_P511_rope_kv", 511), ("L0_B4_P0-17-100-511_rope_kv", 511)]


@pytest.mark.vortex
@pytest.mark.parametrize("name,pm", ROPE_FX, ids=[f[0] for f in ROPE_FX])
def test_rope_tab_bit_identical(vx, name, pm):
    d = FIX / "stories260K_batch" / name
    if not (d / "meta.json").is_file():
        pytest.fail(f"fixture {d} missing")
    r0, l0 = _run(vx, "ps_rope", f"-f {d} -R perlaunch")
    r1, l1 = _run(vx, "ps_rope", f"-f {d} -R table -Z {pm + 1}")
    assert r0["status"] == "ok" and r1["status"] == "ok", (r0["reason"], r1["reason"])
    d0, d1 = _digests(l0), _digests(l1)
    keys = ("q_vs_f64", "kcache_vs_f64", "vcache_exact", "kcache_full", "vcache_full", "q_vs_fixture", "kcache_vs_fixture")
    for k in keys:
        assert d0[k] == d1[k], k
    assert all(v["errors"] == 0 for v in _checks(l1).values())
    assert "table_rope_table_vs_host" in _checks(l1)


@pytest.mark.vortex
def test_table_position_outside_refused(vx):
    d = FIX / "stories260K_batch" / "L4_B1_P511_rope_kv"
    row, log = _run(vx, "ps_rope", f"-f {d} -R table -Z 100")
    assert row["status"] == "failed" and "is outside the table" in log and "PASSED!" not in log


@pytest.mark.vortex
def test_silu_variants_fixture(vx):
    d = FIX / "stories260K_batch" / "L0_B1_P511_silu_gate"
    base, lb = _run(vx, "ps_silu_gate", f"-f {d} -E musl")
    assert base["status"] == "ok", base["reason"]
    poly, lp = _run(vx, "ps_silu_gate", f"-f {d} -E poly")
    assert poly["status"] == "ok", poly["reason"]
    mufu, lm = _run(vx, "ps_silu_gate", f"-f {d} -E mufu", "PS-BASE-S0")
    assert mufu["status"] == "ok", mufu["reason"]
    for lg, var in ((lp, "poly"), (lm, "mufu")):
        assert f"PSEXP: variant={var} numerics_changing=1" in lg
        for k, v in _checks(lg).items():
            if k.startswith("dev_"):
                assert v["errors"] == 0 and v["tol"] < 1e-5 and v["rel"] <= v["tol"], (var, k, v)
    assert "PSEXP: variant=musl numerics_changing=0" in lb
    # musl is untouched (digest independent of the tree's repair code): identical to the stored behaviour = tol 1e-5
    assert _checks(lb)["dev_vs_f64"]["tol"] == 1e-5
    # negative controls: mufu needs the unit; a tight tolerance fails
    nm, lnm = _run(vx, "ps_silu_gate", f"-f {d} -E mufu")
    assert nm["status"] == "failed" and "-E mufu needs a device built with" in lnm
    tight, lt = _run(vx, "ps_silu_gate", f"-f {d} -E poly -t 1e-9")
    assert tight["status"] == "failed" and _checks(lt)["dev_vs_f64"]["errors"] > 0


LL6 = (f"-w {PS}/data/models/stories260K.bin -T {PS}/results/S8/golden_tokens_260K.txt "
       f"-r {PS}/build/fixtures/llama_vx/stories260K_tf32.f32")


@pytest.mark.vortex
def test_ps_llama_table_identical(vx):
    out = PS / "build" / "tmp" / "ep1_test"
    out.mkdir(parents=True, exist_ok=True)
    for mode in ("tf", "prefill"):
        n = 6
        a, la = _run(vx, "ps_llama", f"{LL6} -m {mode} -n {n} -o {out}/{mode}_default.f32", rss=1.5)
        b, lb = _run(vx, "ps_llama", f"{LL6} -m {mode} -n {n} -R table -o {out}/{mode}_table.f32", rss=1.5)
        assert a["status"] == "ok" and b["status"] == "ok", (a["reason"], b["reason"])
        assert (out / f"{mode}_default.f32").read_bytes() == (out / f"{mode}_table.f32").read_bytes(), mode
        assert "PSEP1: rope=table exp=musl numerics_changing=0" in lb
        assert re.search(r"^PSL: pos=-1 layer=-1 op=rope_table kernel=rope_table ", lb, re.M)
        assert _checks(lb)["rope_table_vs_host"]["errors"] == 0
        # every rope launch of the table run is rope_kv_tab, none is rope_kv
        assert "kernel=rope_kv_tab" in lb and "kernel=rope_kv " not in lb
        assert "kernel=rope_kv " in la and "kernel=rope_kv_tab" not in la
        # same logits => same digests at every position
        da = {k: v for k, v in _digests(la).items() if k.startswith("logits_dev:")}
        db = {k: v for k, v in _digests(lb).items() if k.startswith("logits_dev:")}
        assert da and da == db
        # the repair removes the soft-float libm from the per-token path: the RoPE launches execute < 1 % of the default's instructions, and the
        # whole token fewer (SiLU's musl expf is still there, so only RoPE's share goes: 9.80 M -> 6.14 M over 6 positions at REG-CFG)
        rope = lambda t: sum(int(m) for m in re.findall(r"^PSL: pos=\d+ layer=\d+ op=rope_kv kernel=\S+ .*? instrs=(\d+) ", t, re.M))
        ra, rb = rope(la), rope(lb)
        assert ra > 0 and rb < 0.01 * ra, (ra, rb)
        ia = sum(int(m) for m in re.findall(r"^PSTOK: .* instrs=(\d+) ", la, re.M))
        ib = sum(int(m) for m in re.findall(r"^PSTOK: .* instrs=(\d+) ", lb, re.M))
        assert ib < 0.75 * ia and ia - ib > 0.9 * ra, (ia, ib, ra)


@pytest.mark.vortex
def test_ps_llama_exp_variants(vx):
    a, la = _run(vx, "ps_llama", f"{LL6} -m tf -n 6 -R table -E poly", rss=1.5)
    b, lb = _run(vx, "ps_llama", f"{LL6} -m tf -n 6 -R table -E mufu", "PS-BASE-S0", rss=1.5)
    c, lc = _run(vx, "ps_llama", f"{LL6} -m tf -n 6 -R table -E mufu", rss=1.5)
    for r, lg, var in ((a, la, "poly"), (b, lb, "mufu")):
        assert r["status"] == "ok", (var, r["reason"])
        assert f"PSEP1: rope=table exp={var} numerics_changing=1" in lg
        s = re.search(r"^PSSUM: .*failures=(\d+) .*logits_failures=(\d+) max_logits_rel=(\S+)", lg, re.M)
        assert s and s.group(1) == "0" and s.group(2) == "0" and float(s.group(3)) <= 1e-3
        assert f"kernel=silu_gate_expf_{var}" in lg and "kernel=silu_gate " not in lg
    assert c["status"] == "failed" and "-E mufu needs a device" in lc          # REG-CFG has no MUFU unit
