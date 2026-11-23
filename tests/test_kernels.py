"""Tests: out-of-tree kernel app framework, ps_gemv_f32, ps_twolaunch.

Non-vortex (always run, cheap, no compile):
  test_installer_*            scripts/install_kernel_apps.sh into a temporary fake build dir: one Makefile per
                              kernels/<dir>/main.cpp, idempotent, --check, and the generated Makefile equals the
                              stock tests/regression/sgemv/Makefile line for line except for the documented
                              substitutions/additions ("generated Makefile diffed against sgemv's").
  test_installer_rejects_app_name_collision   kernels/foo + kernels/ps_foo -> exit 1.
  test_installer_real_builds_up_to_date   the real build dirs carry the current generated Makefiles.
  test_built_kernels_have_no_fmaf_libcall   every ps_* kernel.elf built with the flags has 0 calls to musl
                              fmaf (errata 30: -fno-math-errno lowers __builtin_fmaf to inline fmadd.s).

Vortex-marked (PS_RUN_VORTEX=1; each SimX run goes through scripts/vxrun.py -> lowload.sh, serial):
  test_gemv_fixture[...]      ps_gemv_f32 on the stories15M layer-0 wq fixture and the 512x512 random fixture
                             both kernel variants -> PASSED, every PSCHECK line errors=0, max_rel_err <= 1e-4
                              (metric/tolerance: kernels/common/ps_common.h, kernels/gemv_f32/gemv_host.h).
  test_gemv_synthetic_tail    M=37, N=51 (N % 4 != 0 -> scalar path, partial last block) for both variants.
  test_gemv_grid_stride[...]  -g overrides the grid so threads/warps loop over several rows (grid-stride path).
  test_twolaunch_cold_cache   ps_twolaunch at REG-CFG, --perf=5 (L2 class): -L 1 and -L 2 PASSED (both launches
                              checked) and L2 read misses(2 launches) within 1 % of 2 x misses(1 launch), i.e. no
                              cross-launch reuse; a warm L2 would give ~misses(1) because wq (324 KiB) fits
                              the 1 MiB L2.
  test_corrupt_fixture_fails  a copy of the wq fixture with one flipped byte in x.f32 -> the app refuses it (sha256
                              mismatch, FAILED, status failed) before any simulation.
Vortex tests use a PRIVATE results dir build/tmp/kernels_test (rows + lowload ledger), deleted at session start,
so they never skip on old rows and never mix into results/<run>. lowload exit 75 -> pytest.skip("DEFERRED").
"""
import os
import re
import shutil
import subprocess

import pytest

PS = __import__("pathlib").Path(__file__).resolve().parents[1]
FIX = PS / "build" / "fixtures"
WQ = FIX / "stories15M" / "L0_P0_matmul_q"
R512 = FIX / "random" / "gemv_512x512_s0"
TOL = 1e-4
PSCHECK_RE = re.compile(r"^PSCHECK: (\S+) n=(\d+) max_abs_err=(\S+) max_rel_err=(\S+) max_elem_rel_err=(\S+) "
                        r"tol=(\S+) errors=(\d+)", re.M)


# --------------------------------------------------------------------------------------------- installer (cheap)
def _fake_build(tmp_path):
    reg = tmp_path / "build" / "tests" / "regression"
    reg.mkdir(parents=True)
    (reg / "common.mk").write_text("# fake\n")
    return tmp_path / "build"


def _install(*args):
    env = dict(os.environ, PS=str(PS))
    return subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), *map(str, args)], env=env,
                          capture_output=True, text=True)


def test_installer_generates_one_makefile_per_app(tmp_path):
    b = _fake_build(tmp_path)
    r = _install(b)
    assert r.returncode == 0, r.stderr
    apps = sorted(p.name for p in (b / "tests/regression").iterdir() if p.is_dir())
    want = sorted((d.name if d.name.startswith("ps_") else "ps_" + d.name)
                  for d in (PS / "kernels").iterdir() if (d / "main.cpp").is_file())
    assert apps == want and {"ps_gemv_f32", "ps_twolaunch"} <= set(apps)
    # idempotent: second run rewrites nothing, --check passes
    mt = {a: (b / "tests/regression" / a / "Makefile").stat().st_mtime_ns for a in apps}
    r2 = _install(b)
    assert r2.returncode == 0 and "wrote" not in r2.stdout
    assert mt == {a: (b / "tests/regression" / a / "Makefile").stat().st_mtime_ns for a in apps}
    assert _install("--check", b).returncode == 0
    (b / "tests/regression/ps_gemv_f32/Makefile").write_text("stale\n")
    assert _install("--check", b).returncode == 1


def test_installer_verifies_after_install_and_detects_stale_makefile(tmp_path):
    """D14: after installing, each Makefile is cmp-verified; a stale file the installer cannot fix (read-only dir)
    makes the run fail instead of leaving a silent vx_module_get_kernel=3 at run time."""
    b = _fake_build(tmp_path)
    assert _install(b).returncode == 0
    mk = b / "tests/regression/ps_gemv_f32/Makefile"
    mk.write_text("stale\n")
    r = _install("--check", b)
    assert r.returncode == 1 and "stale" in r.stderr
    r = _install(b)                                   # install rewrites it; the verification pass then agrees
    assert r.returncode == 0 and "VERIFY FAILED" not in r.stderr
    assert _install("--check", b).returncode == 0
    # the verification pass itself: stub out the writer by making mv a no-op through PATH
    mk.write_text("stale\n")
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    (fakebin / "mv").write_text("#!/bin/sh\nexit 0\n")
    (fakebin / "mv").chmod(0o755)
    env = dict(os.environ, PS=str(PS), PATH=f"{fakebin}:{os.environ['PATH']}")
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), str(b)], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "VERIFY FAILED" in r.stderr


def test_installer_makefile_mirrors_sgemv(tmp_path):
    """Every line of the stock sgemv Makefile appears, in order, in the generated one, after the documented
    substitutions; the only extra lines are comments, PS_KERNELS, and the 5 documented additions (the 5th,
    -fno-math-errno: errata 30)."""
    # ps_linear: an app without app.conf (ps_gemv_f32 / ps_gemv_q80 list an extra device TU: VX_SRCS=kernel.cpp kernel_ilp.cpp)
    b = _fake_build(tmp_path)
    assert _install(b).returncode == 0
    gen = (b / "tests/regression/ps_linear/Makefile").read_text().splitlines()
    stock = (PS / "third_party/vortex/tests/regression/sgemv/Makefile").read_text().splitlines()
    subst = {"PROJECT := sgemv": "PROJECT := ps_linear",
             "SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)": "SRC_DIR := $(PS_KERNELS)/linear",
             "OPTS ?= -n32": "OPTS ?= "}
    expect = [subst.get(l, l) for l in stock if l.strip()]
    body = [l for l in gen if l.strip() and not l.startswith("#")]
    it = iter(body)
    for line in expect:                       # in-order subsequence
        assert any(g == line for g in it), f"missing/out of order: {line!r}"
    extra = [l for l in body if l not in expect]
    assert extra == [f"PS_KERNELS := {PS / 'kernels'}",
                     "CXXFLAGS += -I$(PS_KERNELS)/common", "VX_CFLAGS += -I$(PS_KERNELS)/common",
                     "VX_CFLAGS += -fno-math-errno",
                     "HDRS := $(wildcard $(SRC_DIR)/*.h) $(wildcard $(PS_KERNELS)/common/*.h)",
                     "VX_HDRS := $(wildcard $(SRC_DIR)/*.h) $(wildcard $(PS_KERNELS)/common/*.h)"]
    assert body[-1] == "include ../common.mk"


def test_installer_rejects_app_name_collision(tmp_path):
    """kernels/foo and kernels/ps_foo would both become app ps_foo -> the installer refuses (exit 1, writes nothing)."""
    for d in ("foo", "ps_foo"):
        (tmp_path / "kernels" / d).mkdir(parents=True)
        (tmp_path / "kernels" / d / "main.cpp").write_text("int main() { return 0; }\n")
        (tmp_path / "kernels" / d / "kernel.cpp").write_text("\n")
    b = _fake_build(tmp_path)
    r = subprocess.run(["bash", str(PS / "scripts/install_kernel_apps.sh"), str(b)], env=dict(os.environ, PS=str(tmp_path)),
                       capture_output=True, text=True)
    assert r.returncode == 1 and "collision" in r.stderr
    assert not (b / "tests/regression/ps_foo").exists()


def test_installer_real_builds_up_to_date():
    builds = [PS / "build" / f"vortex-{t}" for t in ("base", "ps")
              if (PS / "build" / f"vortex-{t}" / "tests/regression/common.mk").is_file()]
    if not builds:
        pytest.skip("no Vortex build dir on this machine")
    r = _install("--check", *builds)
    assert r.returncode == 0, r.stdout + r.stderr


def test_built_kernels_have_no_fmaf_libcall():
    """Errata 30 / with -fno-math-errno in the generated Makefile, __builtin_fmaf is the inline fmadd.s, never
    a call to musl's fmaf. Checks every built ps_* kernel.elf whose config.stamp (the VX_CFLAGS it was built with,
    tests/regression/common.mk) carries the flag; ELFs built before the fix are not evidence either way."""
    objdump = __import__("pathlib").Path.home() / "tools/llvm-vortex/bin/llvm-objdump"
    elfs = [e for e in sorted(PS.glob("build/vortex-*/tests/regression/ps_*/kernel.elf"))
            if (e.parent / "config.stamp").is_file() and "-fno-math-errno" in (e.parent / "config.stamp").read_text()]
    if not objdump.is_file() or not elfs:
        pytest.skip("no llvm-objdump or no kernel built with the flags on this machine")
    for e in elfs:
        dis = subprocess.run([str(objdump), "-d", "--no-show-raw-insn", str(e)], capture_output=True, text=True,
                             check=True).stdout
        assert not re.search(r"\b(jal|call|tail)\s+.*<fmaf>", dis), e
        assert "<fmaf>:" not in dis, e        # not even linked in


def test_twolaunch_shares_gemv_sources():
    """ps_twolaunch reuses the gemv kernel/driver through symlinks (so vxrun's app-source hash covers them)."""
    d = PS / "kernels" / "ps_twolaunch"
    for f in ("kernel.cpp", "common.h", "gemv_host.h"):
        assert (d / f).is_symlink() and (d / f).resolve() == (PS / "kernels/gemv_f32" / f).resolve()


# --------------------------------------------------------------------------------------------- vortex (SimX)
@pytest.fixture(scope="module")
def vx():
    import sys
    sys.path.insert(0, str(PS / "scripts"))
    import vxrun  # noqa: E402
    priv = PS / "build" / "tmp" / "kernels_test"
    if priv.exists():
        shutil.rmtree(priv)
    assert _install().returncode == 0
    return vxrun.VxRun("P2.2-test", results=priv)


def _run(vx, spec):
    row = vx.run(spec, timeout_s=3600)
    if row["status"] == "deferred":
        pytest.skip("DEFERRED: lowload.sh exit 75 (not admitted in time); not a failure")
    log = (PS / row["log"]).read_text(errors="replace")
    return row, log


def _checks(log):
    return {m.group(1): {"max_rel_err": float(m.group(4)), "errors": int(m.group(7)), "tol": float(m.group(6))}
            for m in PSCHECK_RE.finditer(log)}


@pytest.mark.vortex
@pytest.mark.parametrize("variant", ["rows", "lanes"])
@pytest.mark.parametrize("fixture", [WQ, R512], ids=["stories15M_L0_wq", "random_512x512"])
def test_gemv_fixture(vx, fixture, variant):
    if not (fixture / "meta.json").is_file():
        pytest.fail(f"fixture {fixture} missing: run `python -B -m model.make_fixtures`")
    row, log = _run(vx, {"tree": "base", "app": "ps_gemv_f32", "args": f"-f {fixture} -v {variant}",
                         "config": "smoke-default", "perf": 1})
    assert row["status"] == "ok", row["reason"]
    c = _checks(log)
    assert set(c) == {"fixture_vs_f64", "launch0:vs_f64", "launch0:vs_fixture"}
    for k, v in c.items():
        assert v["errors"] == 0 and v["max_rel_err"] <= TOL and v["tol"] == TOL, (k, v)


@pytest.mark.vortex
@pytest.mark.parametrize("variant", ["rows", "lanes"])
def test_gemv_synthetic_tail(vx, variant):
    row, log = _run(vx, {"tree": "base", "app": "ps_gemv_f32", "args": f"-m 37 -n 51 -s 7 -v {variant}",
                         "config": "smoke-default", "perf": 1})
    assert row["status"] == "ok", row["reason"]
    c = _checks(log)
    assert set(c) == {"launch0:vs_f64"} and c["launch0:vs_f64"]["errors"] == 0
    assert "M=37 N=51" in log


@pytest.mark.vortex
@pytest.mark.parametrize("variant", ["rows", "lanes"])
@pytest.mark.parametrize("case", [f"-f {WQ} -g 1", "-m 37 -n 51 -s 7 -g 2"], ids=["wq_grid1", "tail_grid2"])
def test_gemv_grid_stride(vx, case, variant):
    """Fewer blocks than needed (-g): with block = NT*NW = 16 threads at the default config, -g 1 makes each thread
    handle 18 wq rows (rows) / each warp 72 rows (lanes); -g 2 on 37 rows leaves an uneven last pass."""
    row, log = _run(vx, {"tree": "base", "app": "ps_gemv_f32", "args": f"{case} -v {variant}",
                         "config": "smoke-default", "perf": 1})
    assert row["status"] == "ok", row["reason"]
    assert "(grid overridden by -g)" in log
    c = _checks(log)
    assert c and all(v["errors"] == 0 and v["max_rel_err"] <= TOL for v in c.values()), c


@pytest.mark.vortex
def test_twolaunch_cold_cache(vx):
    miss = {}
    for n in (1, 2):
        row, log = _run(vx, {"tree": "base", "app": "ps_twolaunch", "args": f"-f {WQ} -L {n}", "config": "REG-CFG",
                             "perf": 5})
        assert row["status"] == "ok", row["reason"]
        c = _checks(log)
        assert {f"launch{i}:vs_fixture" for i in range(n)} <= set(c) and all(v["errors"] == 0 for v in c.values())
        miss[n] = row["perf_kv"]["core0.l2cache.miss_r"]
        if n == 1:
            instrs1 = row["instrs"]
        else:
            assert row["instrs"] == 2 * instrs1          # same work twice
    assert miss[1] > 5184                                # >= one miss per 64 B sector of wq (288*288*4/64)
    assert abs(miss[2] - 2 * miss[1]) <= 0.01 * miss[1], miss


@pytest.mark.vortex
def test_corrupt_fixture_fails(vx, tmp_path):
    bad = PS / "build" / "tmp" / "kernels_test" / "corrupt_wq"
    if bad.exists():
        shutil.rmtree(bad)
    bad.mkdir(parents=True)
    shutil.copy(WQ / "meta.json", bad)
    shutil.copy(WQ / "out.f32", bad)
    x = bytearray((WQ / "x.f32").read_bytes())
    x[5] ^= 0x01
    (bad / "x.f32").write_bytes(bytes(x))
    (bad.parent / "weights").mkdir(exist_ok=True)       # meta.json references ../weights/L0_wq.f32
    shutil.copy(WQ.parent / "weights" / "L0_wq.f32", bad.parent / "weights")
    row, log = _run(vx, {"tree": "base", "app": "ps_gemv_f32", "args": f"-f {bad}", "config": "smoke-default",
                         "perf": 1})
    assert row["status"] == "failed"
    assert "sha256 mismatch (x.f32)" in log and "FAILED!" in log and "PASSED!" not in log

