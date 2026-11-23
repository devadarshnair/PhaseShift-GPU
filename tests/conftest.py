"""Shared pytest configuration for PhaseShift.

- `ps_root` fixture: absolute path of the phaseshift/ directory (the parent of tests/).
- Tests marked `@pytest.mark.vortex` are skipped unless the environment variable PS_RUN_VORTEX=1 is set,
  because they need built Vortex trees and run heavy simulations.
- config override: with PS_BASE_CONFIG=<named set> (e.g. PS-BASE-S1-4c, experiments/configs.yaml), every SimX
  run a test makes through scripts/vxrun.py VxRun.run is re-targeted to that configuration: the spec's `config` is
  replaced, its structural keys (cores/warps/threads/l2) are dropped (the named set defines them) and from its extra
  `configs` only macros the named set does not define are kept (the P2 tests only add -DVX_CFG_EXT_MUFU_ENABLE,
  which every PS-BASE has). This is how "re-runs every existing P2 kernel test" per PS-BASE config without
  copying the tests. Tests marked `psbase_skip` need a build WITHOUT an extension (the refusal tests) and are
  skipped under the override (reason printed). The override never changes a test's assertions.
"""
import os
import pathlib

import pytest

PS_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Test modules that import analysis scripts or batch files stored with experiment outputs (results/, experiments/).
# Those outputs are not part of the source snapshot; the modules are collected only when the files exist.
_NEEDS_RUN_OUTPUTS = {
    "test_sampling_error.py": "results/P3.3/analyze.py",
    "test_p61_intervals.py": "results/P6.1/costlib.py",
    "test_pstrace.py": "experiments/P4.3.yaml",
}
collect_ignore = [m for m, p in _NEEDS_RUN_OUTPUTS.items() if not (PS_ROOT / p).exists()]


@pytest.fixture(scope="session")
def ps_root() -> pathlib.Path:
    return PS_ROOT


def _psbase_rss(vx, spec) -> float:
    """lowload admission estimate (--rss) for a re-targeted run. The test modules wipe their private row stores, so
    vxrun's own estimate (1.2 x the largest earlier maxrss of the same tree/app/CONFIGS, floor 0.6 GB) would fall back
    to its 2 GB default for every first run and wait for memory the job never uses. Here the earlier rows are the
    module's own plus every row of the same tree and app at ANY configuration (results/<run>/rows.jsonl and
    results/<run>/tests/*/*.rows.jsonl): SimX itself stays <= 20.4 MB even at 16 cores (G-SPEED), and a blackbox job's
    peak is the rebuild (<= 0.49 GB), so the RSS does not depend on the configuration. Same rule as vxrun
    (estimate_rss_gb: max(1.2 x peak, 0.6 GB)); vxrun's default is used if no such row exists."""
    import json
    import sys
    sys.path.insert(0, str(PS_ROOT / "scripts"))
    import vxrun
    s = vx.plan(spec)
    rows = vx.store.load()
    if vxrun.estimate_rss_gb(rows, s, -1.0) >= 0:
        return vxrun.estimate_rss_gb(rows, s, vx.rss_default_gb)
    peak = 0
    files = [PS_ROOT / "results" / "P3.1" / "rows.jsonl", *sorted((PS_ROOT / "results" / "P3.1" / "tests").glob("*/*.rows.jsonl"))]
    for f in files:
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("tree") == s["tree"] and r.get("app") == s["app"] and isinstance(r.get("maxrss_kb"), int):
                peak = max(peak, r["maxrss_kb"])
    if not peak:
        return vx.rss_default_gb
    return round(max(vxrun.RSS_FACTOR * peak / 1048576, vxrun.BUILD_FLOOR_GB), 3)


def _psbase_override(name: str) -> None:
    import shlex
    import sys
    sys.path.insert(0, str(PS_ROOT / "scripts"))
    import vxrun
    if getattr(vxrun.VxRun.run, "_psbase", None) == name:
        return
    named = vxrun.load_named_configs(PS_ROOT / "experiments" / "configs.yaml")
    if name not in named:
        raise pytest.UsageError(f"PS_BASE_CONFIG={name!r} is not a named set in experiments/configs.yaml")
    own = {vxrun._macro(t)[0] for t in shlex.split(named[name].get("configs") or "") if vxrun._macro(t)}
    orig = getattr(vxrun.VxRun.run, "_orig", vxrun.VxRun.run)

    def run(self, spec, **kw):
        s = {k: v for k, v in dict(spec).items() if k not in ("cores", "warps", "threads", "l2")}
        s["config"] = name
        extra = [t for t in shlex.split(s.pop("configs", "") or "") if (vxrun._macro(t) or ("",))[0] not in own]
        if extra:
            s["configs"] = " ".join(extra)
        s.setdefault("tags", {})
        s["tags"] = dict(s["tags"], psbase_from=str(spec.get("config")), psbase=name)
        if kw.get("rss_gb") is None:
            kw["rss_gb"] = _psbase_rss(self, s)
        return orig(self, s, **kw)

    run._psbase, run._orig = name, orig
    vxrun.VxRun.run = run


def pytest_configure(config):
    config.addinivalue_line("markers", "psbase_skip: needs a build without an extension; skipped under "
                                       "PS_BASE_CONFIG (config override)")
    name = os.environ.get("PS_BASE_CONFIG")
    if name:
        _psbase_override(name)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PS_BASE_CONFIG"):
        skip_cfg = pytest.mark.skip(reason=f"psbase_skip: needs a build without the extension; not applicable under "
                                           f"PS_BASE_CONFIG={os.environ['PS_BASE_CONFIG']}")
        for item in items:
            if "psbase_skip" in item.keywords:
                item.add_marker(skip_cfg)
    if os.environ.get("PS_RUN_VORTEX") == "1":
        return
    skip = pytest.mark.skip(reason="vortex-marked test; set PS_RUN_VORTEX=1 to run")
    for item in items:
        if "vortex" in item.keywords:
            item.add_marker(skip)
