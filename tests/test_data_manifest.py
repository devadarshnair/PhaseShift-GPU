"""Tests: data/MANIFEST.tsv rows match the files (sizes, sha256, 0444, licence, pinned revision),
the llama2.c model files have the planned sizes and headers, the llama2.c clone is pinned in third_party/LOCK and
untouched since its stamp, and the golden runs are non-empty, run-to-run identical and token-consistent.

These tests check the setup state of this machine (data/ and build/ are git-ignored), so missing files FAIL rather
than skip ("no unexpected skips").
"""
import hashlib
import re
import struct
import subprocess

import pytest

HEADER = ["path", "url", "revision", "bytes", "sha256", "licence", "fetched_utc", "step", "verify"]
SHA40 = re.compile(r"^[0-9a-f]{40}$")
# / models note Q1: exact byte sizes from the Hub API.
S8_FILES = {
    "data/models/stories260K.bin": 1_056_540,
    "data/models/tok512.bin": 6_227,
    "data/models/stories15M.bin": 60_816_028,
    "data/models/stories15M.pt": 60_789_090,
}
LLAMA2C_SHA = "350e04fe35433e6d2941dce5a1f53308f87058eb"


def manifest(ps_root):
    lines = (ps_root / "data/MANIFEST.tsv").read_text().splitlines()
    assert lines[0].split("\t") == HEADER
    rows = [dict(zip(HEADER, l.split("\t"))) for l in lines[1:] if l.strip()]
    for r in rows:
        assert len(r) == len(HEADER) and all(r.values()), r
    return rows


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_manifest_rows_match_files(ps_root):
    rows = manifest(ps_root)
    assert rows, "empty manifest"
    paths = [r["path"] for r in rows]
    assert len(paths) == len(set(paths)), "duplicate manifest rows"
    for r in rows:
        p = ps_root / r["path"]
        assert p.is_file(), r["path"]
        assert p.stat().st_size == int(r["bytes"]), r["path"]
        assert sha256(p) == r["sha256"], r["path"]
        assert (p.stat().st_mode & 0o777) == 0o444, f"{r['path']} not 0444"
        assert r["licence"] and r["licence"].lower() not in ("unknown", "-", "none"), r["path"]
        # pinned: the revision is a commit SHA and the URL fetches exactly that revision, not a branch
        assert SHA40.match(r["revision"]), r["revision"]
        assert f"/{r['revision']}/" in r["url"], r["url"]
        assert not re.search(r"/(main|master)/", r["url"]), r["url"]
        assert r["url"].startswith("https://"), r["url"]


def test_s8_models_present_with_planned_sizes(ps_root):
    rows = {r["path"]: r for r in manifest(ps_root)}
    for path, size in S8_FILES.items():
        assert path in rows, path
        assert int(rows[path]["bytes"]) == size, path
        assert rows[path]["licence"] == "MIT", path
        assert rows[path]["url"].startswith("https://huggingface.co/karpathy/tinyllamas/resolve/"), path
        assert "hf-lfs-sha256" in rows[path]["verify"], path


@pytest.mark.parametrize("name,expect", [
    # llama2.c README table (models note Q1): dim, n_layers, n_heads, n_kv_heads, seq_len; vocab 512 / 32000.
    ("stories260K.bin", dict(dim=64, n_layers=5, n_heads=8, n_kv_heads=4, vocab=512, seq_len=512)),
    ("stories15M.bin", dict(dim=288, n_layers=6, n_heads=6, n_kv_heads=6, vocab=32000, seq_len=256)),
])
def test_model_headers(ps_root, name, expect):
    # run.c Config: 7 int32 (dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len); a negative
    # vocab_size means an unshared classifier (run.c read_checkpoint).
    with open(ps_root / "data/models" / name, "rb") as f:
        dim, hidden, nl, nh, nkv, vocab, seq = struct.unpack("<7i", f.read(28))
    assert (dim, nl, nh, nkv, abs(vocab), seq) == (expect["dim"], expect["n_layers"], expect["n_heads"],
                                                    expect["n_kv_heads"], expect["vocab"], expect["seq_len"])
    assert hidden > 0


def test_llama2c_pinned_and_untouched(ps_root):
    lock = (ps_root / "third_party/LOCK").read_text().splitlines()
    rows = [l.split() for l in lock if l.startswith("llama2.c ")]
    assert len(rows) == 1
    name, path, url, sha, date, lic = rows[0]
    assert (path, url, sha, lic) == ("third_party/llama2.c", "https://github.com/karpathy/llama2.c", LLAMA2C_SHA, "MIT")
    head = subprocess.run(["git", "--no-optional-locks", "-C", str(ps_root / path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == sha
    stamp = ps_root / "results/S8/llama2c.stamp"
    assert stamp.is_file()
    # Documented exception (2026-11-14): git rewrites its own metadata (.git/index stat
    # cache, .git dir mtimes) on read-only commands such as `git status`, so the .git/ tree is excluded from the
    # mtime check. Integrity of the checkout is instead asserted by HEAD == pinned sha (above) and a clean
    # `git status --porcelain` (below): no tracked file modified, no untracked file added.
    newer = subprocess.run(["find", str(ps_root / path), "-path", str(ps_root / path / ".git"), "-prune", "-o",
                            "-newer", str(stamp), "-print"], capture_output=True, text=True, check=True).stdout.split()
    assert newer == [], f"files written inside the llama2.c clone: {newer[:5]}"
    dirty = subprocess.run(["git", "--no-optional-locks", "-C", str(ps_root / path), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert dirty == "", f"llama2.c checkout not clean: {dirty[:200]}"


def test_llama2c_binaries_built_outside_clone(ps_root):
    for b in ("run", "runq", "run_tokens"):
        assert (ps_root / "build/llama2c" / b).is_file(), b
    for junk in ("run", "runq", "run_tokens"):
        assert not (ps_root / "third_party/llama2.c" / junk).exists()
    info = (ps_root / "build/llama2c/BUILDINFO").read_text()
    assert LLAMA2C_SHA in info and "CFLAGS: -O3 " in info
    assert "fast-math" not in info and "Ofast" not in info


@pytest.mark.parametrize("m", ["260K", "15M"])
def test_golden_text(ps_root, m):
    g = ps_root / f"results/S8/golden_{m}.txt"
    text = g.read_bytes()
    assert len(text) > 100 and text.startswith(b"Once upon a time")
    logs = ps_root / "results/S8/logs"
    # run-to-run identical (two runs of build/llama2c/run) and identical to the instrumented run_tokens build
    assert (logs / f"golden_{m}.run1.txt").read_bytes() == text
    assert (logs / f"golden_{m}.run2.txt").read_bytes() == text
    assert (logs / f"golden_{m}.tok.txt").read_bytes() == text


@pytest.mark.parametrize("m,vocab", [("260K", 512), ("15M", 32000)])
def test_golden_tokens(ps_root, m, vocab):
    lines = (ps_root / f"results/S8/golden_tokens_{m}.txt").read_text().splitlines()
    assert lines[0].startswith("# llama2.c golden token ids")
    steps = [tuple(map(int, l.split())) for l in lines if not l.startswith("#")]
    assert len(steps) == 200  # -n 200 and no BOS produced before the limit
    assert steps[0][1] == 1  # sequence starts with BOS
    forced = [s[3] for s in steps]
    k = forced.index(0)
    assert k >= 1 and all(forced[:k]) and not any(forced[k:])  # prompt prefix forced, rest generated
    for i, (pos, tok, nxt, _) in enumerate(steps):
        assert pos == i
        assert 0 <= tok < vocab and 0 <= nxt < vocab
        if i + 1 < len(steps):
            assert steps[i + 1][1] == nxt


# ------------------------------------------------------------------ scripts/fetch.py write-once logic (no network)
def _fetch_module(ps_root):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ps_fetch", ps_root / "scripts/fetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_refuses_to_overwrite(ps_root, tmp_path):
    f = _fetch_module(ps_root)
    dest = tmp_path / "x.bin"
    dest.write_bytes(b"original")
    good = hashlib.sha256(b"original").hexdigest()
    rows = [{"path": f.rel(dest), "sha256": good}]
    assert f.existing_ok(dest, rows) is True                      # recorded with the same hash -> no-op
    assert f.existing_ok(tmp_path / "absent.bin", rows) is None   # absent -> download
    assert f.existing_ok(dest, []) is False                       # present, unrecorded -> download and compare
    with pytest.raises(f.FetchError) as e:                        # recorded with a different hash -> refuse
        f.existing_ok(dest, [{"path": f.rel(dest), "sha256": "0" * 64}])
    assert e.value.code == 3
    part = tmp_path / ".x.bin.part"
    part.write_bytes(b"different")
    with pytest.raises(f.FetchError) as e:                        # a fresh download that differs never replaces dest
        f.install(part, dest, hashlib.sha256(b"different").hexdigest())
    assert e.value.code == 3 and dest.read_bytes() == b"original" and not part.exists()


def test_fetch_git_blob_sha1(ps_root, tmp_path):
    f = _fetch_module(ps_root)
    p = tmp_path / "h.txt"
    p.write_bytes(b"hello\n")
    assert f.git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"  # `git hash-object` of "hello\n"
