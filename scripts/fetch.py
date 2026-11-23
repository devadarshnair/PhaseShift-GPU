#!/usr/bin/env python3
"""PhaseShift pinned, write-once downloader (data immutability).

Standard library only (urllib + hashlib + subprocess for `gh`); no new packages. Run it as
`nice -n 19 conda run -n phaseshift --no-capture-output python -I scripts/fetch.py <subcommand> ...`
(the wrapper scripts fetch_models.sh / fetch_configs.sh / fetch_traces.sh do this).

Subcommands
  hf-info   REPO                       print the Hub API record we pin on: {repo, sha, gated, license, files{...}}
  hf-file   REPO PATH --dest D --size N --licence L
                                       pin REPO to its current commit SHA (Hub API `sha`), download
                                       https://huggingface.co/REPO/resolve/<sha>/PATH, check the byte size AND the
                                       LFS sha256 the Hub API reports for PATH (when the file is LFS-tracked)
  hf-config REPO --outdir DIR --lock LOCKFILE
                                       pin REPO, download resolve/<sha>/config.json -> DIR/<org>__<name>.json and
                                       append a lock line `repo sha sha256 licence gated date` (tab-separated).
                                       HTTP 401/403 = gated -> lock line with sha256 "-" and the file skipped.
  gh-raw    OWNER/REPO PATH --dest D --size N --licence L [--branch B]
                                       pin OWNER/REPO to the commit SHA of B (default: the repo's default branch,
                                       looked up, not guessed) via `gh api`, download
                                       https://raw.githubusercontent.com/OWNER/REPO/<sha>/PATH, check the size and
                                       the git blob SHA-1 from `gh api .../contents/PATH?ref=<sha>`; a Git-LFS
                                       pointer is detected and reported, never accepted.

Write-once rules (hf-file, gh-raw, hf-config)
  * The URL always contains the 40-hex commit SHA, never a branch name (asserted).
  * A file is downloaded to `<dest dir>/.<name>.part`, verified, then atomically renamed and chmod 0444.
  * If DEST already exists: its sha256 is computed. If the manifest (or lock) already has a row for DEST with that
    sha256, the call is a no-op (no network) -- re-runs are idempotent even after the upstream repo moved on.
    If DEST exists without a row, the fresh download must hash identically, else the script REFUSES (exit 3) and
    never rewrites DEST. A DEST whose hash differs from its manifest row is also refused (exit 3).
  * data/MANIFEST.tsv (hf-file, gh-raw) columns, tab-separated, header line first:
      path  url  revision  bytes  sha256  licence  fetched_utc  step  verify
    `path` is relative to phaseshift/, `revision` is the pinned commit SHA, `verify` records which upstream check
    passed (e.g. `size+hf-lfs-sha256`, `size+git-blob-sha1`). Rows are appended (fsync'd), never rewritten.
Exit codes: 0 ok, 2 usage, 3 refused (would overwrite / hash differs), 4 verification failed, 5 network/API error.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PS_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PS_ROOT / "data" / "MANIFEST.tsv"
MANIFEST_HEADER = ["path", "url", "revision", "bytes", "sha256", "licence", "fetched_utc", "step", "verify"]
LOCK_HEADER = "# repo\tsha\tsha256\tlicence\tgated\tdate"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
UA = {"User-Agent": "phaseshift-fetch/1.0 (research; low-volume)"}
CHUNK = 1 << 20


class FetchError(Exception):
    def __init__(self, msg: str, code: int):
        super().__init__(msg)
        self.code = code


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(PS_ROOT).as_posix()
    except ValueError:
        return str(p)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    """Git's object id of a file's content: sha1(b"blob <len>\\0" + content)."""
    size = p.stat().st_size
    h = hashlib.sha1(f"blob {size}\0".encode())
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ network
def http_get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise FetchError(f"HTTP {e.code} for {url}", 5) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise FetchError(f"network error for {url}: {last}", 5)


def http_download(url: str, part: Path) -> tuple[int, str]:
    """Stream url into part; return (bytes, sha256). Raises FetchError(code=5, http status in msg)."""
    req = urllib.request.Request(url, headers=UA)
    last = None
    for attempt in range(3):
        try:
            h = hashlib.sha256()
            n = 0
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
                for b in iter(lambda: r.read(CHUNK), b""):
                    f.write(b)
                    h.update(b)
                    n += len(b)
                f.flush()
                os.fsync(f.fileno())
            return n, h.hexdigest()
        except urllib.error.HTTPError as e:
            part.unlink(missing_ok=True)
            err = FetchError(f"HTTP {e.code} for {url}", 5)
            err.http_status = e.code
            raise err from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            part.unlink(missing_ok=True)
            last = e
            time.sleep(5 * (attempt + 1))
    raise FetchError(f"network error for {url}: {last}", 5)


def gh_api(path: str, jq: str | None = None) -> str:
    cmd = ["gh", "api", path] + (["--jq", jq] if jq else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise FetchError(f"gh api {path} failed: {r.stderr.strip()}", 5)
    return r.stdout.strip()


def hf_info(repo: str) -> dict:
    d = http_get_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
    lic = (d.get("cardData") or {}).get("license")
    if lic is None:  # fall back to the Hub's `license:<x>` tag
        lic = next((t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("license:")), None)
    files = {s["rfilename"]: {"size": s.get("size"), "lfs_sha256": (s.get("lfs") or {}).get("sha256")}
             for s in d.get("siblings", [])}
    sha = d.get("sha")
    if not sha or not SHA40.match(sha):
        raise FetchError(f"Hub API for {repo} returned no commit sha", 5)
    return {"repo": repo, "sha": sha, "gated": d.get("gated"), "license": lic or "unknown", "files": files}


# ------------------------------------------------------------------------------------------------ manifest
def manifest_rows(path: Path = MANIFEST) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    if not lines:
        return []
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]


def manifest_append(row: dict, path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    for k, v in row.items():
        if "\t" in str(v) or "\n" in str(v):
            raise FetchError(f"manifest field {k} contains a tab/newline", 2)
    with open(path, "a") as f:
        if new:
            f.write("\t".join(MANIFEST_HEADER) + "\n")
        f.write("\t".join(str(row[k]) for k in MANIFEST_HEADER) + "\n")
        f.flush()
        os.fsync(f.fileno())


def existing_ok(dest: Path, rows: list[dict], key: str = "path") -> bool | None:
    """None: dest absent. True: dest present and recorded with the same sha256. Raises if recorded differently."""
    if not dest.exists():
        return None
    have = sha256_file(dest)
    recs = [r for r in rows if r.get(key) == rel(dest)]
    if not recs:
        return False
    if any(r.get("sha256") == have for r in recs):
        return True
    raise FetchError(f"REFUSE: {rel(dest)} exists with sha256 {have}, manifest says {recs[-1].get('sha256')}", 3)


def install(part: Path, dest: Path, sha: str) -> None:
    """Move a verified .part into place (write-once) and make it read-only."""
    if dest.exists():
        have = sha256_file(dest)
        part.unlink(missing_ok=True)
        if have != sha:
            raise FetchError(f"REFUSE: {rel(dest)} exists with sha256 {have} != downloaded {sha}; not overwriting", 3)
        os.chmod(dest, 0o444)
        return
    os.replace(part, dest)
    os.chmod(dest, 0o444)


def check_lfs_pointer(part: Path) -> None:
    with open(part, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise FetchError(f"{part.name}: downloaded a Git-LFS pointer, not the data; locate the real blob "
                         f"(gh api .../contents/<path>) -- not guessing", 4)


# ------------------------------------------------------------------------------------------------ commands
def cmd_hf_info(a) -> int:
    print(json.dumps(hf_info(a.repo), indent=1, sort_keys=True))
    return 0


def cmd_hf_file(a) -> int:
    dest = Path(a.dest).resolve()
    state = existing_ok(dest, manifest_rows())
    if state is True:
        print(f"ok (already present, manifest row matches): {rel(dest)}")
        os.chmod(dest, 0o444)
        return 0
    info = hf_info(a.repo)
    sha = info["sha"]
    meta = info["files"].get(a.path)
    if meta is None:
        raise FetchError(f"{a.path} not in {a.repo}@{sha}", 4)
    if meta["size"] is not None and meta["size"] != a.size:
        raise FetchError(f"{a.repo}/{a.path}: Hub reports {meta['size']} B, plan expects {a.size} B", 4)
    url = f"https://huggingface.co/{a.repo}/resolve/{sha}/{a.path}"
    assert SHA40.match(sha) and f"/{sha}/" in url
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / f".{dest.name}.part"
    n, got = http_download(url, part)
    if n != a.size:
        part.unlink(missing_ok=True)
        raise FetchError(f"{url}: got {n} B, expected {a.size} B", 4)
    verify = "size"
    if meta["lfs_sha256"]:
        if got != meta["lfs_sha256"]:
            part.unlink(missing_ok=True)
            raise FetchError(f"{url}: sha256 {got} != Hub LFS sha256 {meta['lfs_sha256']}", 4)
        verify = "size+hf-lfs-sha256"
    install(part, dest, got)
    manifest_append(dict(path=rel(dest), url=url, revision=sha, bytes=n, sha256=got, licence=a.licence,
                         fetched_utc=now_utc(), step=a.step, verify=verify))
    print(f"fetched {rel(dest)} {n} B sha256={got} rev={sha} ({verify})")
    return 0


def cmd_gh_raw(a) -> int:
    dest = Path(a.dest).resolve()
    state = existing_ok(dest, manifest_rows())
    if state is True:
        print(f"ok (already present, manifest row matches): {rel(dest)}")
        os.chmod(dest, 0o444)
        return 0
    branch = a.branch or gh_api(f"repos/{a.repo}", ".default_branch")
    sha = gh_api(f"repos/{a.repo}/commits/{branch}", ".sha")
    if not SHA40.match(sha):
        raise FetchError(f"bad commit sha for {a.repo}@{branch}: {sha!r}", 5)
    meta = json.loads(gh_api(f"repos/{a.repo}/contents/{a.path}?ref={sha}"))
    url = f"https://raw.githubusercontent.com/{a.repo}/{sha}/{a.path}"
    assert f"/{sha}/" in url
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / f".{dest.name}.part"
    n, got = http_download(url, part)
    check_lfs_pointer(part)
    if n != a.size:
        part.unlink(missing_ok=True)
        raise FetchError(f"{url}: got {n} B, expected {a.size} B (contents API says {meta.get('size')})", 4)
    blob = git_blob_sha1(part)
    if blob != meta.get("sha"):
        part.unlink(missing_ok=True)
        raise FetchError(f"{url}: git blob sha1 {blob} != contents API {meta.get('sha')}", 4)
    install(part, dest, got)
    manifest_append(dict(path=rel(dest), url=url, revision=sha, bytes=n, sha256=got, licence=a.licence,
                         fetched_utc=now_utc(), step=a.step, verify=f"size+git-blob-sha1({branch})"))
    print(f"fetched {rel(dest)} {n} B sha256={got} rev={sha} branch={branch}")
    return 0


def lock_rows(lock: Path) -> list[dict]:
    if not lock.is_file():
        return []
    out = []
    for l in lock.read_text().splitlines():
        if l.startswith("#") or not l.strip():
            continue
        c = l.split("\t")
        out.append(dict(repo=c[0], sha=c[1], sha256=c[2], licence=c[3], gated=c[4], date=c[5],
                        path=c[6] if len(c) > 6 else ""))
    return out


def cmd_hf_config(a) -> int:
    outdir = Path(a.outdir).resolve()
    lock = Path(a.lock).resolve()
    org, name = a.repo.split("/", 1)
    dest = outdir / f"{org}__{name}.json"
    rows = lock_rows(lock)
    if dest.exists():
        have = sha256_file(dest)
        recs = [r for r in rows if r["repo"] == a.repo]
        if recs and any(r["sha256"] == have for r in recs):
            print(f"ok (already present, lock row matches): {rel(dest)}")
            os.chmod(dest, 0o444)
            return 0
        if recs:
            raise FetchError(f"REFUSE: {rel(dest)} sha256 {have} differs from lock {recs[-1]['sha256']}", 3)
    elif any(r["repo"] == a.repo and r["sha256"] == "-" for r in rows):
        print(f"ok (recorded as gated, skipped): {a.repo}")
        return 0
    info = hf_info(a.repo)
    sha, gated, lic = info["sha"], info["gated"], info["license"]
    url = f"https://huggingface.co/{a.repo}/resolve/{sha}/config.json"
    outdir.mkdir(parents=True, exist_ok=True)
    part = outdir / f".{dest.name}.part"
    try:
        n, got = http_download(url, part)
    except FetchError as e:
        if getattr(e, "http_status", None) in (401, 403):
            _lock_append(lock, [a.repo, sha, "-", lic, f"gated:{gated}(HTTP {e.http_status})", now_utc(), "-"])
            print(f"GATED {a.repo}: {e} -> recorded and skipped")
            return 0
        raise
    try:
        json.loads(part.read_text())
    except (ValueError, UnicodeDecodeError) as e:
        part.unlink(missing_ok=True)
        raise FetchError(f"{url}: not valid JSON ({e})", 4)
    install(part, dest, got)
    _lock_append(lock, [a.repo, sha, got, lic, str(gated).lower(), now_utc(), rel(dest)])
    print(f"fetched {rel(dest)} {n} B sha256={got} rev={sha} licence={lic} gated={gated}")
    return 0


def _lock_append(lock: Path, cols: list) -> None:
    lock.parent.mkdir(parents=True, exist_ok=True)
    new = not lock.exists() or lock.stat().st_size == 0
    with open(lock, "a") as f:
        if new:
            f.write("# PhaseShift pinned HF config.json files. Written by scripts/fetch.py hf-config.\n")
            f.write("# Revision = Hub commit SHA at fetch time; config URL = https://huggingface.co/<repo>/resolve/<sha>/config.json\n")
            f.write(LOCK_HEADER + "\tpath\n")
        f.write("\t".join(str(c) for c in cols) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("hf-info"); s.add_argument("repo"); s.set_defaults(fn=cmd_hf_info)
    s = sub.add_parser("hf-file"); s.add_argument("repo"); s.add_argument("path")
    s.add_argument("--dest", required=True); s.add_argument("--size", type=int, required=True)
    s.add_argument("--licence", required=True); s.add_argument("--step", default="S8"); s.set_defaults(fn=cmd_hf_file)
    s = sub.add_parser("gh-raw"); s.add_argument("repo"); s.add_argument("path")
    s.add_argument("--dest", required=True); s.add_argument("--size", type=int, required=True)
    s.add_argument("--licence", required=True); s.add_argument("--branch"); s.add_argument("--step", default="S9")
    s.set_defaults(fn=cmd_gh_raw)
    s = sub.add_parser("hf-config"); s.add_argument("repo")
    s.add_argument("--outdir", default=str(PS_ROOT / "configs")); s.add_argument("--lock", default=str(PS_ROOT / "configs" / "configs.lock"))
    s.set_defaults(fn=cmd_hf_config)
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except FetchError as e:
        print(f"fetch.py: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
