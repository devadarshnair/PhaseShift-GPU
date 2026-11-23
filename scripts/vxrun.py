#!/usr/bin/env python3
"""vxrun.py -- PhaseShift SimX run harness.

Usage (run it with the phaseshift env's python; vxrun itself is light, every SimX job goes through lowload.sh):

  vxrun.py run --step <ID> --tree {base,ps} --app <name> [--args "..."] [--config <name> | --configs "-D..."]
               [--cores N] [--warps N] [--threads N] [--l2] [--perf N] [--env K=V ...] [--timeout S]
               [--tag k=v ...] [--rep N] [--direct] [--rss GB] [--mem GB] [--force]
  vxrun.py batch experiments/<file>.yaml [--stop-on-fail] [--dry-run]

What one run does
  `$LL --rss <est> --mem <cap> --label <step>:<app>:<key8>:r<rep> --log results/<step>/logs/<label>.log --
       timeout -k 30 <S> env CONFIGS="<configs>" [K=V ...] ./ci/blackbox.sh --driver=simx --app=<app>
       [--cores=N] [--warps=N] [--threads=N] [--l2cache] --args=<args> --perf=<class>`
  in the build dir build/vortex-<tree> (always with a perf class, default 1, so every build has -DPERF_ENABLE).
  Exactly one JSON row is appended (fsync'd) to results/<step>/rows.jsonl.

  --direct (footprint measurement): instead of blackbox, ONE lowload job runs (1) `make -C <app dir> <app>
  kernel.vxbin runtime-simx` with the SAME CONFIGS string blackbox would build (in the environment, as blackbox
  passes it; nothing is rebuilt if the last blackbox run used this config), then (2) the app binary itself under its
  own `/usr/bin/time -v -o <log>.app.time`: `env LD_LIBRARY_PATH=<build>/sw/runtime VORTEX_DRIVER=simx
  VORTEX_PROFILING=<perf> ./<app> <args>` in the app dir -- the command line of the run-simx make rule
  (tests/regression/common.mk:237-239). One job, so no other job can rebuild the shared build dir between build and
  run (all SimX jobs hold the machine flock). The process timed in (2) IS the simulator (SimX is libsimx.so inside the
  app process), so `wall_s`/`maxrss_kb`/`simx_maxrss_kb` of a direct row are the app's; `job_wall_s`/`job_maxrss_kb`
  are the whole job's (build check included).
  In a blackbox row, `maxrss_kb` is the max over all processes of the job (make, python gen_config.py, compilers when
  a rebuild happens, the app+SimX); it is an UPPER bound on the SimX RSS.

run_key = sha256 over the JSON of (tree, tree fingerprint, canonical CONFIGS, app, app-source hash, args, perf,
  env, rep, direct). The first eight are the earlier list; `rep` (repeat index, default 0) and `direct` are added so
  that "run each case twice" and its direct footprint run get their own rows instead of being skipped.
  tree fingerprint = sha256(<tree>/.ps_patchset + HEAD commits of the tree and its ramulator submodule (also stored
  in the row as `tree_heads`) + `git diff HEAD --binary` of the tree + the same for its ramulator
  submodule + names and contents of untracked, non-ignored files); app-source hash = sha256 over $PS/kernels/common and
  $PS/kernels/<name> (+ the generated app Makefile) for out-of-tree `ps_*` apps, "" otherwise (in-tree apps are
  covered by the tree fingerprint).
  canonical CONFIGS = every -D macro of (named set's configs, --configs, structural flags, -DPERF_ENABLE) as a sorted
  list, last definition of a macro wins (blackbox appends the flags after the env CONFIGS, and gen_config.py reads
  the flags left to right), so two spellings of the same configuration get the same key.
A run whose key already has a row with status "ok" is skipped (--force re-runs it and appends a new row).

Row status: ok (exit 0, PASS string, global `PERF: instrs=..., cycles=...` line, no FAILED) | failed | timeout (exit
  124, or 137 after >= --timeout s: `timeout` fired, R8) | oom (137 without the timeout firing = SIGKILL from
  elsewhere, likely the OOM killer; or a failing run with an allocation failure in its log, e.g. the VA cap) |
  deferred (lowload exit 75: not admitted in time, not a failure).

--rss estimate (admission, lowload.sh): if the row store holds earlier rows of the same (tree, app, canonical
  CONFIGS), est = max(1.2 x their largest maxrss, 0.6 GB); 1.2 is the same safety factor lowload.sh uses for ledger
  estimates and 0.6 GB covers a SimX + app rebuild (build peaks <= 0.49 GB). Without earlier rows the step budget
  --rss-default (2 GB, the S7 value) is used. S7's 18.7 MB rows are not in any rows.jsonl and are never used.

Batch file (YAML):
  step:
  defaults: {tree: base, perf: 1, timeout: 7200, mem: 4}
  runs:
    - {app: sgemm, args: ["-n64", "-n128"], config: smoke-default, reps: 2, tags: {case: sgemm}}
  List values of tree/app/args/config/configs/cores/warps/threads/l2/perf/direct expand as a cartesian product;
  `reps: N` adds rep = 0..N-1. Expanded runs are sorted (stably) by (tree, canonical CONFIGS, direct, rep) so each
  SimX build is made once per batch (K8), then run serially. Failures are recorded and the batch continues,
  unless --stop-on-fail (used by tests). A deferred run stops the batch (the machine is busy; re-run later: done keys
  are skipped). Pattern borrowed from src/bench/smolgpu/lowload_a2a.py (append-as-you-go JSONL, restart skips rows).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import itertools
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

PS = Path(os.environ.get("PS", Path(__file__).resolve().parents[1]))
VXRUN_VERSION = 4   # 4 (2026-11-14): app_src_hash = include-closure hash with comments stripped (v<=3 rows keep the legacy full-text hash, see legacy_hit), `expect: fail`; 3: file content in key, provenance hashes, header guard; 2: HEAD commits in the tree fingerprint, one-job --direct, oom status
DEFERRED = 75
TIMEOUT_EXITS = (124, 137)          # GNU timeout: 124 after TERM, 128+9 if it had to KILL (-k)
DEFAULT_RSS_GB = 2.0                # step budget when no earlier rows exist (S7 used --rss 2)
BUILD_FLOOR_GB = 0.6                # SimX + app rebuild peak (S7: <= 0.49 GB) rounded up
RSS_FACTOR = 1.2                    # same factor lowload.sh applies to ledger maxrss
HEARTBEAT_S = 300
DIRECT_SCRIPT = r"""set -o pipefail
cfg=$1 dir=$2 app=$3 blog=$4 atime=$5; shift 5
env CONFIGS="$cfg" make -C "$dir" "$app" kernel.vxbin runtime-simx > "$blog" 2>&1 \
    || { rc=$?; echo "vxrun-direct: build failed (exit $rc, see $blog)"; tail -20 "$blog"; exit $rc; }
cd "$dir"
exec /usr/bin/time -v -o "$atime" "$@"
"""
R8_TIMEOUT_S = 24 * 3600            # R8: a single required run > 24 h is killed (default --timeout)
STRUCT_MACROS = {"cores": "VX_CFG_NUM_CORES", "warps": "VX_CFG_NUM_WARPS", "threads": "VX_CFG_NUM_THREADS"}
L2_MACRO = "VX_CFG_L2_ENABLE"
PASS_RE = re.compile(r"^PASSED!?\s*$", re.M)
FAIL_RE = re.compile(r"FAILED", re.M)
GLOBAL_PERF_RE = re.compile(r"^PERF: instrs=(\d+), cycles=(\d+), IPC=([0-9.]+)\s*$", re.M)
# sgemm prints "Elapsed: N ms", sgemv / sgemm_tcu_wg_dxa print "Elapsed time: %lg ms"
ELAPSED_RE = re.compile(r"^Elapsed(?: time)?: ([0-9.]+(?:[eE][+-]?\d+)?) ms\s*$", re.M)
# allocation failures: a prlimit --as (VA cap) abort or an out-of-memory condition inside the job
MEM_ERR_RE = re.compile(r"bad_alloc|Cannot allocate memory|[Oo]ut of memory|ENOMEM", re.M)
TIME_FIELDS = {
    "Maximum resident set size (kbytes)": ("maxrss_kb", int),
    "User time (seconds)": ("user_s", float),
    "System time (seconds)": ("sys_s", float),
    "Exit status": ("time_exit", int),
}


# ----------------------------------------------------------------------------------------------- configs / keys
def load_named_configs(path: Path | None = None) -> dict:
    import yaml
    path = path or PS / "experiments" / "configs.yaml"
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _macro(tok: str):
    """'-DNAME=VAL' -> ('NAME', 'VAL'); '-DNAME' -> ('NAME', None); anything else -> None."""
    if not tok.startswith("-D") or len(tok) < 3:
        return None
    name, eq, val = tok[2:].partition("=")
    return name, (val if eq else None)


def canonical_configs(tokens: list[str]) -> list[str]:
    """Sorted, de-duplicated -D list; the last definition of a macro wins. Non -D tokens are kept (sorted) as-is."""
    macros, other = {}, set()
    for t in tokens:
        m = _macro(t)
        if m is None:
            other.add(t)
        else:
            macros[m[0]] = m[1]
    out = [f"-D{k}" if v is None else f"-D{k}={v}" for k, v in sorted(macros.items())]
    return out + sorted(other)


def dedupe_tokens(tokens: list[str]) -> list[str]:
    """Drop repeated config tokens, LAST definition wins (same rule as canonical_configs): a macro that appears in the
    named set and again in the spec's extra `configs` is emitted once, at the position of its last definition, with the
    last value. Non -D tokens are de-duplicated by exact text. Without this the same -DNAME=... twice reaches the
    compiler (-Werror redefinition). The run key is unchanged: it uses canonical_configs(), which already
    applied last-wins, so every stored run_key stays valid (tests/test_bugfix_pass2.py checks all stored rows)."""
    def key(t):   # separate key spaces: a bare token "FOO" must not collide with the macro -DFOO=...
        m = _macro(t)
        return ("D", m[0]) if m else ("T", t)
    last = {key(t): i for i, t in enumerate(tokens)}
    return [t for i, t in enumerate(tokens) if last[key(t)] == i]


def valid_tree(tree) -> bool:
    """base / ps, or an extra experiment tree (..T3, 2026-11-14): a plain name whose vortex-trees/<name>/ was made
    by scripts/mk_vortex_tree.sh (it carries the .ps_patchset marker). Its build dir is build/vortex-<name>."""
    if tree in ("base", "ps"):
        return True
    return (isinstance(tree, str) and re.fullmatch(r"[a-z][a-z0-9]{1,15}", tree) is not None
            and (PS / "vortex-trees" / tree / ".ps_patchset").is_file())


def resolve(spec: dict, named: dict) -> dict:
    """Fill a run spec from its named config set; compute the env CONFIGS string, blackbox flags and canonical list.

    spec keys: tree, app, args, config, configs, cores, warps, threads, l2, perf, env (dict), rep, direct, tags.
    Explicit spec values win over the named set's; structural flags win over the same macro in `configs`.
    """
    s = dict(spec)
    base = {}
    if s.get("config"):
        if s["config"] not in named:
            raise SystemExit(f"vxrun: unknown config set '{s['config']}' (see experiments/configs.yaml)")
        base = named[s["config"]]
    cfg_tokens = shlex.split(base.get("configs", "") or "") + shlex.split(s.get("configs") or "")
    for k in ("cores", "warps", "threads", "l2"):
        if s.get(k) in (None, False) and base.get(k) not in (None, False):
            s[k] = base[k]
    s["perf"] = int(s.get("perf", 1) if s.get("perf") is not None else 1)
    flags, flag_macros = [], []
    # fixed order cores, threads, warps (= smoke case 6/7 argv order): the CONFIGS literal blackbox builds depends on it
    for k in ("cores", "threads", "warps"):
        if s.get(k) not in (None, False):
            s[k] = int(s[k])
            flags.append(f"--{k}={s[k]}")
            flag_macros.append(f"-D{STRUCT_MACROS[k]}={s[k]}")
    if s.get("l2"):
        s["l2"] = True
        flags.append("--l2cache")
        flag_macros.append(f"-D{L2_MACRO}")
    flags.append(f"--perf={s['perf']}")
    overridden = {_macro(t)[0] for t in flag_macros}
    env_configs = dedupe_tokens([t for t in cfg_tokens if (_macro(t) or ("",))[0] not in overridden])
    s["configs_extra"] = " ".join(shlex.split(s.get("configs") or ""))      # the spec's own (--cfgx) macros, for the row
    s["configs_env"] = " ".join(env_configs)
    s["blackbox_flags"] = flags
    # the exact CONFIGS literal blackbox.sh builds (env value, then each flag's macro in argv order, then PERF_ENABLE)
    s["configs_effective"] = " ".join(env_configs + flag_macros + ["-DPERF_ENABLE"])
    s["configs_canonical"] = canonical_configs(env_configs + flag_macros + ["-DPERF_ENABLE"])
    s["args"] = " ".join(shlex.split(s.get("args") or ""))
    s["env"] = {str(k): str(v) for k, v in sorted((s.get("env") or {}).items())}
    s["rep"] = int(s.get("rep", 0) or 0)
    s["direct"] = bool(s.get("direct", False))
    s["tags"] = {str(k): v for k, v in (s.get("tags") or {}).items()}
    if s.get("expect") not in (None, "ok", "fail"):
        raise SystemExit(f"vxrun: expect must be 'ok' or 'fail', got {s.get('expect')!r}")
    if not valid_tree(s.get("tree")):
        raise SystemExit(f"vxrun: --tree must be base, ps or an extra tree with vortex-trees/<name>/.ps_patchset, "
                         f"got {s.get('tree')!r}")
    if not s.get("app"):
        raise SystemExit("vxrun: --app is required")
    return s


def _resolve_file(tok: str, bases: list[Path]) -> Path | None:
    """An existing regular file named by `tok` (absolute, or relative to one of `bases`), else None."""
    if not tok or len(tok) > 1024 or "\0" in tok:
        return None
    p = Path(tok)
    for c in ([p] if p.is_absolute() else [b / p for b in bases]):
        try:
            if c.is_file():
                return c
        except OSError:
            pass
    return None


OUTPUT_ENV = {"PS_TRACE"}        # env values that name files the run WRITES (never key inputs)
OUTPUT_OPTS = {"-o", "--out", "--output"}   # argument options whose value is an output file


def _is_output_attached(tok: str) -> bool:
    """D10: the attached forms `--out=FILE` / `--output=FILE` / `-oFILE` also name an output file (never a key input)."""
    if tok.startswith("--"):
        return tok.split("=", 1)[0] in OUTPUT_OPTS and "=" in tok
    return len(tok) > 2 and tok[:2] in OUTPUT_OPTS and tok[2] != "-"
_HASH_CACHE: dict = {}


def _sha_file(f: Path) -> str:
    st = f.stat()
    ck = (str(f), st.st_mtime_ns, st.st_size)
    if ck not in _HASH_CACHE:
        _HASH_CACHE[ck] = hashlib.sha256(f.read_bytes()).hexdigest()
    return _HASH_CACHE[ck]


def file_hashes(args: str, env: dict, bases: list[Path]) -> dict:
    """sha256 of the CONTENT of every file named in the arguments or env values.

    A token names a file if it (or, for `--opt=value`/`-Xvalue`-free `key=value` tokens, its value part) is an existing
    regular file. Key = "arg<i>" (token index) or "env:<NAME>"; the path itself is not hashed, so renaming a file to
    identical content keeps the content part (the args string in the key still changes with the name). Arguments
    that name no file produce {} and the run key is then bit-identical to the pre-file-hash key. Output files (OUTPUT_ENV,
    the value of OUTPUT_OPTS) are skipped: they may exist from an earlier run and must not make the key depend on it.
    """
    out = {}
    toks = shlex.split(args or "")
    for i, tok in enumerate(toks):
        if (i and toks[i - 1] in OUTPUT_OPTS) or _is_output_attached(tok):
            continue
        for cand in ([tok, tok.split("=", 1)[1]] if "=" in tok else [tok]):
            f = _resolve_file(cand, bases)
            if f is not None:
                out[f"arg{i}"] = _sha_file(f)
                break
    for k, v in (env or {}).items():
        f = None if k in OUTPUT_ENV else _resolve_file(str(v), bases)
        if f is not None:
            out[f"env:{k}"] = _sha_file(f)
    return out


def run_key(s: dict, tree_fp: str, app_hash: str) -> str:
    material = {
        "tree": s["tree"], "tree_fp": tree_fp, "configs": s["configs_canonical"], "app": s["app"],
        "app_src": app_hash, "args": s["args"], "perf": s["perf"], "env": s["env"],
        "rep": s["rep"], "direct": s["direct"],
    }
    if s.get("file_hashes"):          # only when files are named: no-file runs keep their pre-file-hash keys
        material["files"] = s["file_hashes"]
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_FP_CACHE: dict = {}


def tree_heads(tree: str, ps: Path = PS) -> dict:
    """HEAD commits of the working tree and of its ramulator submodule (the diff alone is relative to HEAD)."""
    t = ps / "vortex-trees" / tree
    out = {"tree": _git(t, "rev-parse", "HEAD").decode().strip()}
    ram = t / "third_party" / "ramulator"
    if (ram / ".git").exists():
        out["ramulator"] = _git(ram, "rev-parse", "HEAD").decode().strip()
    return out


def _git(tree: Path, *args: str) -> bytes:
    return subprocess.run(["git", "--no-optional-locks", "-C", str(tree), *args], check=True,
                          capture_output=True).stdout


def tree_fingerprint(tree: str, ps: Path = PS) -> str:
    """sha256 over .ps_patchset, the tree's diff vs HEAD, its ramulator submodule's diff and its untracked files."""
    if tree in _FP_CACHE:
        return _FP_CACHE[tree]
    t = ps / "vortex-trees" / tree
    h = hashlib.sha256()
    h.update((t / ".ps_patchset").read_bytes())
    h.update(b"\0heads\0" + json.dumps(tree_heads(tree, ps), sort_keys=True).encode())
    h.update(b"\0diff\0" + _git(t, "diff", "HEAD", "--binary", "--ignore-submodules=all"))
    ram = t / "third_party" / "ramulator"
    if (ram / ".git").exists():
        h.update(b"\0ramulator\0" + _git(ram, "diff", "HEAD", "--binary"))
    untracked = _git(t, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    for name in sorted(u for u in untracked if u):
        p = t / name.decode()
        h.update(b"\0u\0" + name + b"\0" + (p.read_bytes() if p.is_file() else b""))
    _FP_CACHE[tree] = h.hexdigest()
    return _FP_CACHE[tree]


def app_source_hash(app: str, build_dir: Path, ps: Path = PS) -> str:
    """LEGACY (vxrun_version <= 3) app hash, kept byte-for-byte so stored rows and the closed-step compare scripts
    (results/*/noop_compare.py ...) still resolve. Out-of-tree PhaseShift apps (ps_*): hash ALL of kernels/common +
    kernels/<name> (raw text, comments included) + the generated Makefile. New rows use app_dep_hash()."""
    if not app.startswith("ps_"):
        return ""
    h = hashlib.sha256()
    dirs = [ps / "kernels" / "common"]
    own = ps / "kernels" / app
    dirs.append(own if own.is_dir() else ps / "kernels" / app[3:])
    for d in dirs:
        if d.is_dir():
            for p in sorted(x for x in d.rglob("*") if x.is_file()):
                h.update(str(p.relative_to(ps)).encode() + b"\0" + p.read_bytes())
    mk = build_dir / "tests" / "regression" / app / "Makefile"
    if mk.is_file():
        h.update(b"\0Makefile\0" + mk.read_bytes())
    return h.hexdigest()


_INC_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]', re.M)


def strip_comments(text: str) -> str:
    """C/C++ source with comments removed and whitespace normalised (per-line strip, blank lines dropped). String and
    character literals are respected (`'` right after an alphanumeric is a digit separator). A comment-only or
    whitespace-only edit therefore leaves the result unchanged."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == "//":
            j = text.find("\n", i)
            i = n if j < 0 else j       # keep the newline; a `\` continuation also continues a // comment (rare, ignored)
        elif two == "/*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        elif c == '"' or (c == "'" and not (out and out[-1][-1:].isalnum())):
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    lines = ((" ".join(l.split())) for l in "".join(out).split("\n"))
    return "\n".join(l for l in lines if l)


def app_dep_hash(app: str, build_dir: Path, ps: Path = PS) -> str:
    """D1 (vxrun_version >= 4): hash of the files an out-of-tree ps_* app really compiles, not of whole directories.

    Roots = the app dir's main.cpp, its device sources (VX_SRCS of app.conf, default kernel.cpp) and the files its
    DEPS patterns name; the closure follows every `#include "x"`/`<x>` that resolves inside the app dir or
    kernels/common (system headers are ignored). Each file contributes its comment-stripped, whitespace-normalised
    text (strip_comments), so editing a comment or an unrelated header of kernels/common does not change the key, while
    any code edit of a file that is compiled does. The generated build Makefile is hashed as before."""
    if not app.startswith("ps_"):
        return ""
    kern = ps / "kernels"
    own = kern / app
    own = own if own.is_dir() else kern / app[3:]
    common = kern / "common"
    conf = {}
    if (own / "app.conf").is_file():
        for line in (own / "app.conf").read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                conf[k] = v
    roots = [own / "main.cpp"] + [own / f for f in (conf.get("VX_SRCS") or "kernel.cpp").split()]
    for pat in (conf.get("DEPS") or "").split():
        roots += sorted(own.glob(pat))
    seen, todo = {}, [r for r in roots if r.is_file()]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen[f] = strip_comments(f.read_text(errors="replace"))
        for inc in _INC_RE.findall(f.read_text(errors="replace")):
            for base in (f.parent, own, common):
                c = base / inc
                if c.is_file():
                    todo.append(c)
                    break
    h = hashlib.sha256(b"app_dep_hash/1\0")
    for f in sorted(seen):
        h.update(str(f.relative_to(ps)).encode() + b"\0" + seen[f].encode() + b"\0")
    mk = build_dir / "tests" / "regression" / app / "Makefile"
    if mk.is_file():
        h.update(b"\0Makefile\0" + mk.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------ parsing
def _num(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _split_top(body: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    out, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def parse_perf_line(line: str) -> dict:
    """'PERF: core0: l2cache: reqs=12040, miss_r=2080 (hit=79%)' -> {'core0.l2cache.reqs': 12040,
    'core0.l2cache.miss_r': 2080, 'core0.l2cache.miss_r.hit_pct': 79}. A bare parenthetical like 'occupancy=7.3 (92%)'
    gives '<key>.pct'. Values ending in % are stored as numbers under a '_pct' key suffix."""
    body = line[len("PERF:"):].strip()
    scope = []
    while True:
        m = re.match(r"^([A-Za-z_][\w\-]*):\s+(.*)$", body)
        if not m or "=" in m.group(1):
            break
        scope.append(m.group(1))
        body = m.group(2)
    pre = ".".join(scope)
    out = {}
    for item in _split_top(body):
        m = re.match(r"^([\w\-.]+)=([^\s(]+)\s*(?:\((.*)\))?$", item)
        if not m:
            continue
        key = f"{pre}.{m.group(1)}" if pre else m.group(1)
        val = m.group(2)
        if val.endswith("%"):
            out[key + "_pct"] = _num(val[:-1])
        else:
            out[key] = _num(val)
        inner = m.group(3)
        if inner:
            im = re.match(r"^([\w\-.]+)=(\S+)$", inner.strip())
            if im:
                iv = im.group(2)
                out[f"{key}.{im.group(1)}" + ("_pct" if iv.endswith("%") else "")] = _num(iv.rstrip("%"))
            elif inner.strip().endswith("%"):
                out[f"{key}.pct"] = _num(inner.strip()[:-1])
    return out


def parse_log(text: str) -> dict:
    """PASS/FAIL, the global PERF line (last one), every other PERF line (raw + key=value), app 'Elapsed: N ms'."""
    g = GLOBAL_PERF_RE.findall(text)
    raw = [ln.rstrip() for ln in text.splitlines() if ln.startswith("PERF:")]
    perf = {}
    for ln in raw:
        if GLOBAL_PERF_RE.match(ln):
            continue
        perf.update(parse_perf_line(ln))
    el = ELAPSED_RE.findall(text)
    cfg = re.findall(r"^CONFIGS=(.*)$", text, re.M)
    return {
        "passed": bool(PASS_RE.search(text)),
        "failed_str": bool(FAIL_RE.search(text)),
        "instrs": int(g[-1][0]) if g else None,
        "cycles": int(g[-1][1]) if g else None,
        "ipc": float(g[-1][2]) if g else None,
        "perf": perf,
        "perf_raw": raw,
        "app_elapsed_ms": _num(el[-1]) if el else None,
        "mem_error": bool(MEM_ERR_RE.search(text)),
        "blackbox_configs": cfg[0] if cfg else None,
    }


def parse_time(text: str) -> dict:
    """GNU /usr/bin/time -v output -> wall_s, maxrss_kb, user_s, sys_s, time_exit (missing fields are None)."""
    out = {"wall_s": None, "maxrss_kb": None, "user_s": None, "sys_s": None, "time_exit": None}
    for ln in text.splitlines():
        k, _, v = ln.strip().partition(": ")
        if k in TIME_FIELDS:
            name, conv = TIME_FIELDS[k]
            try:
                out[name] = conv(v.strip())
            except ValueError:
                pass
        elif k.startswith("Elapsed (wall clock) time"):
            parts = v.strip().split(":")
            try:
                secs = 0.0
                for p in parts:
                    secs = secs * 60 + float(p)
                out["wall_s"] = secs
            except ValueError:
                pass
    return out


def classify(exit_code: int, log: dict, wall_s: float | None = None, timeout_s: float | None = None) -> tuple[str, str]:
    """Row status. 124 = GNU timeout fired (TERM). 137 = SIGKILL: a timeout only if the run lasted >= its --timeout
    (timeout's -k escalation); otherwise something else killed it -- most likely the OOM killer -> `oom`. A failing run
    whose log shows an allocation failure (VA cap from lowload's prlimit --as, K2) is also `oom`, so R8's
    "over memory cap -> next tier down" can act on the status."""
    if exit_code == DEFERRED:
        return "deferred", "lowload exit 75 (not admitted within the wait window)"
    if exit_code == 124:
        return "timeout", "exit 124 from timeout (run over its --timeout budget)"
    if exit_code == 137:
        if timeout_s and wall_s is not None and wall_s >= timeout_s:
            return "timeout", f"exit 137 after {wall_s:.0f} s >= --timeout {timeout_s} s (KILL after the grace period, R8)"
        return "oom", "exit 137 = SIGKILL not sent by timeout (likely the kernel OOM killer)"
    if exit_code != 0 and log.get("mem_error"):
        return "oom", f"exit {exit_code} with an allocation failure in the log (VA cap / out of memory)"
    if log["failed_str"]:
        return "failed", f"FAILED in log (exit {exit_code})"
    if exit_code != 0:
        return "failed", f"exit {exit_code}"
    if not log["passed"]:
        return "failed", "exit 0 but no PASSED line (truncated log?)"
    if log["instrs"] is None:
        return "failed", "PASSED but no global 'PERF: instrs=..., cycles=...' line (truncated log or perf off?)"
    return "ok", ""


# -------------------------------------------------------------------------------------------------- row store
class RowStore:
    """Append-only JSONL. Each row is written with ONE os.write of a complete line, then fsync'd. A writer killed
    mid-line leaves a partial last line: load() ignores lines that do not parse, and append() first terminates an
    unterminated last line so the next row starts on its own line."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for ln in self.path.read_text(errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict):
                rows.append(r)
        return rows

    def ok_keys(self) -> set:
        return {r.get("run_key") for r in self.load() if r.get("status") == "ok"}

    def append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(row, sort_keys=True) + "\n").encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            size = os.fstat(fd).st_size
            if size:
                with open(self.path, "rb") as f:
                    f.seek(size - 1)
                    if f.read(1) != b"\n":
                        data = b"\n" + data
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)


def estimate_rss_gb(rows: list[dict], s: dict, default_gb: float = DEFAULT_RSS_GB) -> float:
    same = [r for r in rows if r.get("tree") == s["tree"] and r.get("app") == s["app"]
            and r.get("configs_canonical") == s["configs_canonical"] and isinstance(r.get("maxrss_kb"), int)]
    if not same:
        return default_gb
    peak_gb = max(r["maxrss_kb"] for r in same) / 1048576
    return round(max(RSS_FACTOR * peak_gb, BUILD_FLOOR_GB), 3)


# ------------------------------------------------------------------------------------------------------ runner
HEADER_DIRS = ("sim", "sw", "hw", "third_party/ramulator/src")


def stale_runtime_objects(tree_src: Path, build_dir: Path) -> list[Path]:
    """Errata 36a: `make runtime-simx` compiles into sw/runtime/obj but includes the .d files of sim/simx/obj, so a header
    edit does not rebuild those objects. Return the objects to delete: all of sw/runtime/obj/**/*.o if any header under
    HEADER_DIRS of the tree is newer than libsimx.so (the next build then recompiles them; ccache keeps it cheap)."""
    lib = build_dir / "sw" / "runtime" / "libsimx.so"
    objdir = build_dir / "sw" / "runtime" / "obj"
    if not lib.is_file() or not objdir.is_dir():
        return []
    t = lib.stat().st_mtime
    for d in HEADER_DIRS:
        root = tree_src / d
        if root.is_dir():
            for x in root.rglob("*"):
                if x.suffix in (".h", ".hpp", ".inc") and x.is_file() and x.stat().st_mtime > t:
                    return sorted(objdir.rglob("*.o"))
    return []


def tree_lock_path(bdir: Path, fallback_dir: Path) -> Path:
    """Per-build-tree lock file: <build_dir>/.vxrun.lock (fallback next to the rows if the build dir does not exist yet,
    so a first build does not create the tree dir ahead of make)."""
    return bdir / ".vxrun.lock" if bdir.is_dir() else fallback_dir / f".vxrun.{bdir.name}.lock"


def lock_tree(path: Path):
    """Blocking exclusive flock on `path`; returns the open fd (close it to release; the kernel also drops it if we die)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def stamp_macros(text: str) -> dict:
    """-D macros (last definition wins) of a tests/regression config.stamp (VX_CFLAGS, CXXFLAGS, LDFLAGS lines)."""
    out = {}
    for tok in shlex.split(text.replace("\n", " ")) if text.strip() else []:
        m = _macro(tok)
        if m:
            out[m[0]] = m[1]
    return out


def check_config_stamp(stamp_text: str | None, canonical: list[str]) -> tuple[bool | None, str]:
    """Did the app build that just ran use the requested CONFIGS? config.stamp is written by common.mk from the flags
    (which contain $(CONFIGS)) of the make that built the app, and that make also (re)builds libsimx.so with the same
    CONFIGS. Every requested -D macro must be in the stamp with the same value. None = no stamp to check."""
    if stamp_text is None:
        return None, "no config.stamp"
    have = stamp_macros(stamp_text)
    bad = []
    for tok in canonical:
        m = _macro(tok)
        if m and (m[0] not in have or have[m[0]] != m[1]):
            bad.append(f"{tok} (stamp: {'absent' if m[0] not in have else have[m[0]]})")
    return (not bad), ("; ".join(bad) if bad else "ok")


def _heartbeat(path: Path, stop: threading.Event):
    while not stop.wait(HEARTBEAT_S):
        try:
            path.touch()
        except OSError:
            pass


def lowload_runner(cmd: list[str], cwd: Path, env: dict, log_path: Path, time_path: Path) -> int:
    """Default runner: execute cmd (a full lowload.sh command line), return its exit status. lowload writes the log
    (--log) and the /usr/bin/time file (<log>.time)."""
    return subprocess.run(cmd, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL).returncode


def _iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _file_state(p: Path):
    try:
        st = p.stat()
        return {"mtime_ns": st.st_mtime_ns, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
    except OSError:
        return None


class VxRun:
    def __init__(self, step: str, ps: Path = PS, runner=lowload_runner, named: dict | None = None,
                 results: Path | None = None, rss_default_gb: float = DEFAULT_RSS_GB, out=sys.stdout):
        if not re.match(r"^[A-Za-z0-9._-]+$", step or ""):
            raise SystemExit(f"vxrun: bad step id {step!r}")
        # absolute paths: --log/--time paths handed to lowload (cwd = build dir) must not depend on the caller's cwd
        self.step, self.ps, self.runner, self.out = step, Path(ps).absolute(), runner, out
        self.named = load_named_configs(self.ps / "experiments" / "configs.yaml") if named is None else named
        self.results = Path(results).absolute() if results else self.ps / "results"
        self.dir = self.results / step
        self.store = RowStore(self.dir / "rows.jsonl")
        self.rss_default_gb = rss_default_gb
        # injectable for tests: fingerprint(tree) -> str, app_hash(app, build_dir) -> str
        self.fingerprint = lambda tree: tree_fingerprint(tree, self.ps)
        self.app_hash = lambda app, bdir: app_dep_hash(app, bdir, self.ps)
        self.legacy_app_hash = lambda app, bdir: app_source_hash(app, bdir, self.ps)   # vxrun_version <= 3 rows
        self.heads = lambda tree: tree_heads(tree, self.ps)

    def _say(self, msg):
        print(msg, file=self.out, flush=True)

    def build_dir(self, tree: str) -> Path:
        return self.ps / "build" / f"vortex-{tree}"

    def plan(self, spec: dict) -> dict:
        s = resolve(spec, self.named)
        bdir = self.build_dir(s["tree"])
        s["tree_fp"] = self.fingerprint(s["tree"])
        s["tree_heads"] = self.heads(s["tree"])
        s["app_src_hash"] = self.app_hash(s["app"], bdir)
        s["file_hashes"] = file_hashes(s["args"], s["env"],
                                       [self.ps, bdir / "tests" / "regression" / s["app"]])
        s["run_key"] = run_key(s, s["tree_fp"], s["app_src_hash"])
        # D1: keys of rows written with the pre-v4 whole-directory app hash (vxrun_version <= 3), see legacy_hit()
        s["legacy_app_hash"] = self.legacy_app_hash(s["app"], bdir)
        s["legacy_app_key"] = None
        if s["legacy_app_hash"] != s["app_src_hash"]:
            s["legacy_app_key"] = run_key(s, s["tree_fp"], s["legacy_app_hash"])
        s["legacy_key"] = None
        if s["file_hashes"]:     # the pre-file-hash key of this spec (files not covered), for old rows; see legacy_hit()
            s["legacy_key"] = run_key({**s, "file_hashes": {}}, s["tree_fp"], s["legacy_app_hash"])
        s["build_dir"] = bdir
        return s

    @staticmethod
    def _provenance(app_dir: Path, app: str) -> dict:
        """Hashes of the built artifacts (recorded after the run; NOT part of the pre-run key)."""
        out = {}
        for key, name in (("kernel_vxbin", "kernel.vxbin"), ("kernel_elf", "kernel.elf"), ("host_app", app)):
            p = app_dir / name
            if p.is_file():
                out[key] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    def _env(self) -> dict:
        env = dict(os.environ)
        env["PS"] = str(self.ps)
        env["PS_STEP"] = self.step
        env["TMPDIR"] = str(self.ps / "build" / "tmp")     # /tmp is a small RAM tmpfs (errata 21)
        env["PS_RESULTS_DIR"] = str(self.results)          # lowload ledger + heartbeat live next to the rows
        (self.ps / "build" / "tmp").mkdir(parents=True, exist_ok=True)
        return env

    def _ll(self, s, label, log, rss, mem, inner: list[str], timeout_s) -> list[str]:
        cmd = [str(self.ps / "scripts" / "lowload.sh"), "--rss", f"{rss:g}", "--mem", str(mem), "--label", label,
               "--log", str(log), "--"]
        if timeout_s:
            cmd += ["timeout", "-k", "30", str(int(timeout_s))]
        return cmd + inner

    def legacy_hit(self, s: dict, rows: list[dict]) -> dict | None:
        """Old rows (vxrun_version < 3) were keyed without file content. Accept such an ok row for a spec that names files
        only if no new-key row exists and every named file's mtime is not later than the row's `started` (so the file
        cannot have been edited since). Anything uncertain re-runs (safe direction); new rows always use the new key."""
        if any(r.get("run_key") == s["run_key"] and r.get("status") == "ok" for r in rows):
            return None
        # D1: an ok row keyed with the legacy whole-directory app hash (v<=3) is valid history: its key still matches
        # exactly when the hashed text is unchanged, which is the old (safe) behaviour. Nothing is re-keyed or rewritten.
        if s.get("legacy_app_key"):
            for r in rows:
                if r.get("run_key") == s["legacy_app_key"] and r.get("status") == "ok" and r.get("vxrun_version", 0) <= 3:
                    return r
        if not s.get("legacy_key"):
            return None
        for r in rows:
            if r.get("run_key") == s["legacy_key"] and r.get("status") == "ok" and r.get("vxrun_version", 0) < 3:
                try:
                    t0 = _dt.datetime.fromisoformat(r["started"]).timestamp()
                    if all(f.stat().st_mtime <= t0 for f in self._named_files(s)):
                        return r
                except (KeyError, ValueError, OSError):
                    pass
        return None

    def _named_files(self, s: dict) -> list[Path]:
        bases = [self.ps, s["build_dir"] / "tests" / "regression" / s["app"]]
        toks = shlex.split(s["args"] or "")
        out = [_resolve_file(c, bases) for i, t in enumerate(toks) if f"arg{i}" in s["file_hashes"]
               for c in ([t, t.split("=", 1)[1]] if "=" in t else [t])]
        out += [_resolve_file(str(v), bases) for k, v in s["env"].items() if f"env:{k}" in s["file_hashes"]]
        return [f for f in out if f is not None]

    def run(self, spec: dict, *, timeout_s: int | None = None, rss_gb: float | None = None, mem_gb: int = 4,
            force: bool = False) -> dict:
        s = self.plan(spec)
        key = s["run_key"]
        rows = self.store.load()
        legacy = None if force else self.legacy_hit(s, rows)
        if legacy is not None:
            lk = legacy.get("run_key", "")
            self._say(f"vxrun: skip {s['app']} {s['args']!r} -- legacy (v{legacy.get('vxrun_version', 0)}) key {lk[:12]} ok "
                      f"(pre-v4 app hash or pre-file-hash file rule; content assumed unchanged)")
            return legacy | {"skipped": True, "legacy_key_match": True}
        if not force and s.get("expect") == "fail":     # D2: a stored failed row with this key is the expected result
            for r in rows:
                if r.get("run_key") == key and r.get("status") == "failed":
                    self._say(f"vxrun: skip {s['app']} {s['args']!r} -- key {key[:12]} already failed (expect: fail)")
                    return r | {"skipped": True, "expected_failure": True}
        if not force and key in {r.get("run_key") for r in rows if r.get("status") == "ok"}:
            self._say(f"vxrun: skip {s['app']} {s['args']!r} [{s.get('config') or s['configs_env']}] rep{s['rep']}"
                      f"{' direct' if s['direct'] else ''} -- key {key[:12]} already ok")
            return next(r for r in rows if r.get("run_key") == key and r.get("status") == "ok") | {"skipped": True}
        rss = rss_gb if rss_gb is not None else estimate_rss_gb(rows, s, self.rss_default_gb)
        label = f"{self.step}:{s['app']}:{key[:8]}:r{s['rep']}" + (":direct" if s["direct"] else "")
        logs = self.dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (label.replace(":", "_") + ".log")
        for p in (log, Path(str(log) + ".time")):     # a re-run (failed/forced) starts a fresh log
            if p.exists():
                p.unlink()
        env = self._env()
        bdir = s["build_dir"]
        # BUG-QUEUE 2026-11-14: the memory-admitted lane runs several PhaseShift jobs at once and they share one build
        # tree; a job with other CONFIGS would rebuild libsimx.so/the app between this job's build and run. Hold an
        # exclusive per-tree flock across the whole build+run of this single run (released after the row is built).
        # Taken BEFORE the lowload admission, so we queue on the tree lock without holding a lane slot.
        lockpath = tree_lock_path(bdir, self.dir)
        t_lock = time.time()
        lock_fd = lock_tree(lockpath)
        waited = round(time.time() - t_lock, 3)
        if waited > 1:
            self._say(f"vxrun: waited {waited:g} s for the build-tree lock {lockpath}")
        try:
            return self._run_locked(s, key, rss, label, log, env, bdir, mem_gb, timeout_s, waited)
        finally:
            os.close(lock_fd)

    def _run_locked(self, s, key, rss, label, log, env, bdir, mem_gb, timeout_s, lock_wait_s) -> dict:
        logs = log.parent
        dropped = stale_runtime_objects(self.ps / "vortex-trees" / s["tree"], bdir)
        for o in dropped:
            o.unlink()
        if dropped:
            self._say(f"vxrun: header newer than libsimx.so -> dropped {len(dropped)} stale runtime objects (errata 36a)")
        app_dir = bdir / "tests" / "regression" / s["app"]
        watch = {"libsimx": bdir / "sw" / "runtime" / "libsimx.so", "app_bin": app_dir / s["app"],
                 "config_stamp": app_dir / "config.stamp"}
        before = {k: _file_state(p) for k, p in watch.items()}
        hb = self.dir / ".heartbeat"
        hb.touch()
        stop = threading.Event()
        threading.Thread(target=_heartbeat, args=(hb, stop), daemon=True).start()
        t0, started = time.time(), _iso()
        extra_env = [f"{k}={v}" for k, v in s["env"].items()]
        build = None
        try:
            if not s["direct"]:
                inner = ["env", f"CONFIGS={s['configs_env']}", *extra_env, "./ci/blackbox.sh", "--driver=simx",
                         f"--app={s['app']}", *s["blackbox_flags"][:-1], f"--args={s['args']}",
                         s["blackbox_flags"][-1]]
                cmd = self._ll(s, label, log, rss, mem_gb, inner, timeout_s)
                cwd = bdir
                code = self.runner(cmd, cwd, env, log, Path(str(log) + ".time"))
            else:
                # ONE lowload job (build, then the app), so no other job can rebuild libsimx.so / the app in the shared
                # build dir between the two (every SimX job holds the machine flock while it runs). The app is timed
                # by its own /usr/bin/time (<log>.app.time): that maxrss is the app+SimX process alone.
                # CONFIGS goes in the ENVIRONMENT, as blackbox passes it: a make command-line variable would override
                # common.mk's `CONFIGS := $(CONFIGS) -DWGMMA_RS` default and so build a different model.
                blog = logs / (label.replace(":", "_") + ".build.log")
                atime = Path(str(log) + ".app.time")
                for p in (blog, atime):
                    if p.exists():
                        p.unlink()
                ldp = f"{bdir / 'sw' / 'runtime'}:{os.environ.get('LD_LIBRARY_PATH', '')}"   # as run-simx
                inner = ["env", *extra_env, "bash", "-c", DIRECT_SCRIPT, "vxrun-direct", s["configs_effective"],
                         str(app_dir), s["app"], str(blog), str(atime),
                         "env", f"LD_LIBRARY_PATH={ldp}", "VORTEX_DRIVER=simx", f"VORTEX_PROFILING={s['perf']}",
                         f"./{s['app']}", *shlex.split(s["args"])]
                cmd = self._ll(s, label, log, max(rss, BUILD_FLOOR_GB), mem_gb, inner, timeout_s)
                cwd = bdir
                code = self.runner(cmd, cwd, env, log, Path(str(log) + ".time"))
                ainfo = parse_time(atime.read_text(errors="replace")) if atime.exists() else None
                build = {"log": str(blog.relative_to(self.ps)) if blog.is_relative_to(self.ps) else str(blog),
                         "app_time": ainfo}
        finally:
            stop.set()
            hb.touch()
        ended = _iso()
        after = {k: _file_state(p) for k, p in watch.items()}
        text = log.read_text(errors="replace") if log.exists() else ""
        tpath = Path(str(log) + ".time")
        tinfo = parse_time(tpath.read_text(errors="replace")) if tpath.exists() else parse_time("")
        job_time = tinfo
        if build is not None and build["app_time"] is not None:
            tinfo = build["app_time"]          # the direct row's wall/maxrss = the app (SimX) process alone
        pl = parse_log(text)
        if build is not None and build["app_time"] is None and code not in (DEFERRED, 124, 137):
            status, reason = "failed", f"direct-mode build failed (exit {code}; see {build['log']})"
        else:
            status, reason = classify(code, pl, job_time["wall_s"], timeout_s)
        stamp = watch["config_stamp"]
        cfg_ok, cfg_note = check_config_stamp(stamp.read_text(errors="replace") if stamp.is_file() else None,
                                              s["configs_canonical"])
        if cfg_ok is False and status == "ok":
            status, reason = "failed", f"config mismatch: app/libsimx build does not carry the requested CONFIGS: {cfg_note}"
        rel = (lambda p: str(p.relative_to(self.ps)) if Path(p).is_relative_to(self.ps) else str(p))
        row = {
            "vxrun_version": VXRUN_VERSION, "run_key": key, "status": status, "reason": reason, "step": self.step,
            "tree": s["tree"], "tree_fp": s["tree_fp"], "tree_heads": s["tree_heads"], "build_dir": rel(bdir), "app": s["app"],
            "app_src_hash": s["app_src_hash"], "args": s["args"], "perf": s["perf"],
            "config": s.get("config"), "configs_env": s["configs_env"], "configs_effective": s["configs_effective"],
            "configs_canonical": s["configs_canonical"], "configs_extra": s.get("configs_extra", ""), "cores": s.get("cores"), "warps": s.get("warps"),
            "threads": s.get("threads"), "l2": bool(s.get("l2")), "env": s["env"], "rep": s["rep"],
            "direct": s["direct"], "tags": s["tags"] | ({"cfgx": s["configs_extra"]} if s.get("configs_extra") and "cfgx" not in s["tags"] else {}), "cmd": cmd, "cwd": rel(cwd), "label": label,
            "log": rel(log), "exit": code, "rss_req_gb": rss, "mem_cap_gb": mem_gb, "timeout_s": timeout_s,
            "started": started, "ended": ended, "harness_wall_s": round(time.time() - t0, 3),
            "wall_s": tinfo["wall_s"], "maxrss_kb": tinfo["maxrss_kb"], "user_s": tinfo["user_s"],
            "sys_s": tinfo["sys_s"], "instrs": pl["instrs"], "cycles": pl["cycles"], "ipc": pl["ipc"],
            "passed": pl["passed"], "app_elapsed_ms": pl["app_elapsed_ms"], "blackbox_configs": pl["blackbox_configs"],
            "perf_kv": pl["perf"], "perf_raw": pl["perf_raw"],
            "file_hashes": s["file_hashes"], "stale_objs_dropped": len(dropped),
            "provenance": self._provenance(app_dir, s["app"]),
            "files_before": before, "files_after": after,
            "rebuilt": {k: before[k] != after[k] for k in watch},
            "direct_build": build, "job_wall_s": job_time["wall_s"], "job_maxrss_kb": job_time["maxrss_kb"],
            "config_stamp_ok": cfg_ok, "config_stamp_note": cfg_note, "tree_lock_wait_s": lock_wait_s,
            "host": socket.gethostname(), "expect": s.get("expect"),
            "expected_failure": s.get("expect") == "fail" and status == "failed",
        }
        if s["direct"]:
            row["simx_maxrss_kb"] = tinfo["maxrss_kb"]   # the measured process is the app+SimX process itself
        self.store.append(row)
        self._say(f"vxrun: {status:8s} {s['app']} {s['args']!r} [{s.get('config') or s['configs_env']}] "
                  f"rep{s['rep']}{' direct' if s['direct'] else ''}: instrs={pl['instrs']} cycles={pl['cycles']} "
                  f"wall={tinfo['wall_s']} s maxrss={tinfo['maxrss_kb']} KB {reason}")
        return row


# ------------------------------------------------------------------------------------------------------- batch
GRID_FIELDS = ("tree", "app", "args", "config", "configs", "cores", "warps", "threads", "l2", "perf", "direct")


def expand_batch(doc: dict, named: dict) -> list[dict]:
    defaults = doc.get("defaults") or {}
    out = []
    for i, entry in enumerate(doc.get("runs") or []):
        e = {**defaults, **entry}
        axes = [(f, e[f] if isinstance(e[f], list) else [e[f]]) for f in GRID_FIELDS if f in e]
        reps = int(e.get("reps", 1))
        for combo in itertools.product(*[v for _, v in axes]):
            for rep in range(reps):
                spec = {k: v for k, v in e.items() if k not in GRID_FIELDS and k not in ("reps",)}
                spec.update(dict(zip([f for f, _ in axes], combo)))
                spec["rep"] = rep if reps > 1 else int(e.get("rep", 0))
                spec["_order"] = len(out)
                out.append(spec)
    # sort by (tree, canonical CONFIGS) so a SimX build is made once per config (K8); stable within a group
    def sk(sp):
        r = resolve({k: v for k, v in sp.items() if not k.startswith("_")}, named)
        return (r["tree"], " ".join(r["configs_canonical"]), r["direct"], r["rep"], sp["_order"])
    return sorted(out, key=sk)


def run_batch(path: Path, vx_factory=VxRun, stop_on_fail=False, dry_run=False, out=sys.stdout) -> int:
    import yaml
    doc = yaml.safe_load(Path(path).read_text())
    step = doc.get("step")
    vx = vx_factory(step)
    specs = expand_batch(doc, vx.named)
    n_fail = 0
    for sp in specs:
        run_opts = {"timeout_s": sp.pop("timeout", R8_TIMEOUT_S), "mem_gb": int(sp.pop("mem", 4)),
                    "rss_gb": sp.pop("rss", None)}
        sp.pop("_order", None)
        if dry_run:
            s = vx.plan(sp)
            print(f"{s['tree']} [{' '.join(s['configs_canonical'])}] {s['app']} {s['args']!r} rep{s['rep']}"
                  f"{' direct' if s['direct'] else ''} key={s['run_key'][:12]}", file=out)
            continue
        row = vx.run(sp, **run_opts)
        if row["status"] == "deferred":
            print("vxrun batch: DEFERRED (machine busy); stopping. Re-run later: finished keys are skipped.", file=out)
            return DEFERRED
        if row["status"] != "ok" and not row.get("expected_failure"):
            n_fail += 1
            if stop_on_fail:
                print(f"vxrun batch: stop on failure ({row['status']}: {row['reason']})", file=out)
                return 1
    print(f"vxrun batch: {len(specs)} runs, {n_fail} not ok", file=out)
    return 1 if n_fail else 0


# --------------------------------------------------------------------------------------------------------- CLI
def _kv(items):
    out = {}
    for it in items or []:
        k, eq, v = it.partition("=")
        if not eq:
            raise SystemExit(f"vxrun: expected K=V, got {it!r}")
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--step", required=True)
    r.add_argument("--tree", required=True)
    r.add_argument("--app", required=True)
    r.add_argument("--args", default="")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--config")
    g.add_argument("--configs")
    r.add_argument("--cores", type=int)
    r.add_argument("--warps", type=int)
    r.add_argument("--threads", type=int)
    r.add_argument("--l2", action="store_true")
    r.add_argument("--perf", type=int, default=1)
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--tag", action="append", default=[])
    r.add_argument("--timeout", type=int, default=R8_TIMEOUT_S)
    r.add_argument("--rep", type=int, default=0)
    r.add_argument("--direct", action="store_true")
    r.add_argument("--rss", type=float)
    r.add_argument("--rss-default", type=float, default=DEFAULT_RSS_GB)
    r.add_argument("--mem", type=int, default=4)
    r.add_argument("--force", action="store_true")
    b = sub.add_parser("batch")
    b.add_argument("file")
    b.add_argument("--stop-on-fail", action="store_true")
    b.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        vx = VxRun(a.step, rss_default_gb=a.rss_default)
        spec = {"tree": a.tree, "app": a.app, "args": a.args, "config": a.config, "configs": a.configs,
                "cores": a.cores, "warps": a.warps, "threads": a.threads, "l2": a.l2, "perf": a.perf,
                "env": _kv(a.env), "tags": _kv(a.tag), "rep": a.rep, "direct": a.direct}
        row = vx.run(spec, timeout_s=a.timeout, rss_gb=a.rss, mem_gb=a.mem, force=a.force)
        return {"ok": 0, "deferred": DEFERRED}.get(row["status"], 1)
    return run_batch(Path(a.file), stop_on_fail=a.stop_on_fail, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
