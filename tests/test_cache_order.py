"""Tests: SimX CacheBank ordering fix (patch 0014-cache-ordering, base + ps; docs/vortex_errata.md item 38).

  test_patch_files            both patches touch exactly sim/simx/mem/cache.cpp, nothing under third_party/
  test_patches_equivalent     the base and ps patches add the same code lines (only context/line numbers differ)
  test_trees_carry_fix        vortex-trees/{base,ps} cache.cpp contain the fix and no dev-only PS_ORDER_FIX switch
  test_base_patch_dir_wired   mk_vortex_tree.sh applies patches/vortex/base/*.patch for the base patchset only
  test_cache_order_unit[...]  (vortex) tests/cache_order/cache_order_test.cpp on the base and ps builds: every load
                              returns the program-order value, final re-reads and flushed memory equal the golden copy,
                              no deadlock, and the bank order checker (PS_ORDER_CHECK=1) reports 0 violations
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
PATCHES = {"ps": PS / "patches/vortex/ps/0014-cache-ordering.patch",
           "base": PS / "patches/vortex/base/0014-cache-ordering.patch"}


def _files(p: Path):
    return sorted(set(re.findall(r"^\+\+\+ b/(\S+)", p.read_text(), re.M)))


def _added(p: Path):
    return [l[1:] for l in p.read_text().splitlines() if l.startswith("+") and not l.startswith("+++")]


def test_patch_files():
    for tree, p in PATCHES.items():
        assert p.is_file(), p
        assert _files(p) == ["sim/simx/mem/cache.cpp"], (tree, _files(p))


def test_patches_equivalent():
    assert _added(PATCHES["ps"]) == _added(PATCHES["base"])
    assert len(_added(PATCHES["ps"])) > 100


def test_trees_carry_fix():
    for tree in ("base", "ps"):
        s = (PS / "vortex-trees" / tree / "sim/simx/mem/cache.cpp").read_text()
        assert "ord_replay_held" in s and "ord_forward_blocked" in s, tree
        assert "PS_ORDER_FIX" not in s, tree          # the run-time off switch exists only dev copy
        assert s.count("pipe_req_->pop();") == 1 and s.count("pipe_req_->push(") == 1, tree   # all via the mirror


def test_base_patch_dir_wired():
    s = (PS / "scripts/mk_vortex_tree.sh").read_text()
    assert 'if [ "$PATCHSET" = base ]; then\n    dirs+=("$PS/patches/vortex/base:.")' in s


@pytest.mark.vortex
@pytest.mark.parametrize("tree", ["base", "ps"])
def test_cache_order_unit(tree):
    b = PS / "build" / f"vortex-{tree}"
    proc = subprocess.run([str(PS / "scripts/cache_order_test.sh"), str(b), "4000", "3", "PS_ORDER_CHECK=1"],
                          capture_output=True, text=True, env={**os.environ, "PS_STEP": "EP.F"})
    log = (PS / "build/tmp/cache_order" / f"vortex-{tree}" / "last_run.log").read_text()
    assert proc.returncode == 0, log[-3000:]
    assert "CACHEORDER PASSED" in log
    viol = [int(v) for v in re.findall(r"^PSORDER bank=.* violations=(\d+)", log, re.M)]
    assert viol and sum(viol) == 0
