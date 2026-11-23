#!/usr/bin/env bash
# PhaseShift Vortex working tree builder (§2 "Vortex trees and builds", §2 "Patch discipline").
#
#   mk_vortex_tree.sh --patchset {base|ps} [--name N] [--extra SET]... [--recreate] [--print-fingerprint]
#
# Creates $PS/vortex-trees/<N> (N defaults to the patchset name) as a LOCAL CLONE of the pristine reference clone
# $PS/third_party/vortex at 50cb174 (not a `git worktree`, which would write into the pristine clone's .git).
# A local clone only reads the source repository (objects are hard-linked, immutable). Submodules (and cvfpu's
# three nested modules) are cloned OFFLINE from the pristine clone's .git/modules/... repositories. git >= 2.38
# refuses file:// submodule clones by default, so `-c protocol.file.allow=always` is passed PER COMMAND (global git
# config is never changed; risk K14).
#
# Patches applied in this order with `git apply --index --binary` (empty folders are skipped):
#   patches/vortex/infra/*.patch          both patchsets
#   patches/vortex/base/*.patch           base only (2026-11-10: the SimX cache-ordering correctness fix, the
#                                         base-tree copy of ps patch 0014; "base" = upstream + that fix)
#   patches/vortex/ps/*.patch             ps only
#   patches/ramulator/*.patch             ps only, applied inside third_party/ramulator
# Idempotence: on success $T/.ps_patchset holds "<patchset> <sha256>" where the sha256 covers the patchset name and
# the ordered (file name, contents) of every applied patch. Re-running with the same fingerprint is a no-op (exit 0);
# a different fingerprint, or a tree without the marker (interrupted run), is refused (exit 1) unless --recreate,
# which deletes the tree and rebuilds it. Vortex's .gitignore ignores `/.*`, so .ps_* markers never enter a patch.
#
# --extra SET (repeatable): also apply patches/vortex/SET/*.patch (plain `git apply --binary`, no --index)
# after the patchset (e.g. rtl, dot4, prefetch, fflags). The extra sets are part of the fingerprint and are recorded in
# $T/.ps_extra, so `--recreate` without --extra re-applies the sets the tree had instead of silently dropping them.
# With no extras the fingerprint input is byte-identical to the old one, so existing markers stay valid.
# --print-fingerprint prints "<patchset> <sha256>" and exits (no tree is touched).
#
# Never writes into $PS/third_party/*; every git command that touches the pristine clone only reads it.
set -euo pipefail

PS="${PS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PRI="$PS/third_party/vortex"
PIN=50cb174546ec69a3869b42b8871e932e4440511f

PATCHSET=""; NAME=""; RECREATE=0; PRINTFP=0; EXTRAS=()
usage() { echo "usage: mk_vortex_tree.sh --patchset {base|ps} [--name N] [--extra SET]... [--recreate] [--print-fingerprint]" >&2; exit 2; }
while [ $# -gt 0 ]; do
    case "$1" in
        --patchset) [ $# -ge 2 ] || usage; PATCHSET="$2"; shift 2 ;;
        --name)     [ $# -ge 2 ] || usage; NAME="$2"; shift 2 ;;
        --extra)    [ $# -ge 2 ] || usage; EXTRAS+=("$2"); shift 2 ;;
        --recreate) RECREATE=1; shift ;;
        --print-fingerprint) PRINTFP=1; shift ;;
        *) usage ;;
    esac
done
case "$PATCHSET" in base|ps) ;; *) usage ;; esac
NAME="${NAME:-$PATCHSET}"
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] && [ "$NAME" != . ] && [ "$NAME" != .. ] || { echo "mk_vortex_tree: bad name '$NAME'" >&2; exit 2; }
T="$PS/vortex-trees/$NAME"
G="git -c protocol.file.allow=always"
# --recreate without --extra: keep the extra sets the existing tree was built with
if [ "${#EXTRAS[@]}" = 0 ] && [ "$RECREATE" = 1 ] && [ -f "$T/.ps_extra" ]; then
    mapfile -t EXTRAS < "$T/.ps_extra"
fi
for x in "${EXTRAS[@]}"; do
    [[ "$x" =~ ^[a-z0-9_-]+$ ]] && [ -d "$PS/patches/vortex/$x" ] || { echo "mk_vortex_tree: bad --extra '$x'" >&2; exit 2; }
done

# Ordered list of patch directories for this patchset (and the git dir each applies in, relative to $T).
dirs=("$PS/patches/vortex/infra:.")
if [ "$PATCHSET" = base ]; then
    dirs+=("$PS/patches/vortex/base:.")          # base-only correctness fix (see header)
fi
if [ "$PATCHSET" = ps ]; then
    dirs+=("$PS/patches/vortex/ps:." "$PS/patches/ramulator:third_party/ramulator")
fi
for x in "${EXTRAS[@]}"; do dirs+=("$PS/patches/vortex/$x:.:x"); done   # third field marks an extra (worktree apply)

fingerprint() {
    {
        echo "patchset=$PATCHSET"
        for e in "${dirs[@]}"; do
            local d="${e%%:*}" p
            [ "${e##*:}" = x ] && echo "extra=${d##*/}"
            for p in "$d"/*.patch; do
                [ -e "$p" ] || continue
                echo "== ${p#$PS/}"
                cat "$p"
            done
        done
    } | sha256sum | cut -d' ' -f1
}
FP="$PATCHSET $(fingerprint)"
if [ "$PRINTFP" = 1 ]; then echo "$FP"; exit 0; fi

if [ -e "$T" ]; then
    if [ "$RECREATE" = 0 ]; then
        if [ -f "$T/.ps_patchset" ] && [ "$(cat "$T/.ps_patchset")" = "$FP" ]; then
            echo "mk_vortex_tree: $T up to date ($FP); no-op"
            exit 0
        fi
        echo "mk_vortex_tree: $T exists with a different or missing fingerprint" \
             "(have: $(cat "$T/.ps_patchset" 2>/dev/null || echo none); want: $FP); use --recreate" >&2
        exit 1
    fi
    echo "mk_vortex_tree: --recreate: removing $T"
    rm -rf -- "$T"
fi
mkdir -p "$PS/vortex-trees"

git clone -q --no-checkout "$PRI" "$T"
git -C "$T" checkout -q "$PIN"
git -C "$T" submodule init
for sm in softfloat ramulator cvfpu hardfloat cocogfx; do
    git -C "$T" config "submodule.third_party/$sm.url" "$PRI/.git/modules/third_party/$sm"
done
$G -C "$T" submodule update --quiet                  # offline, from the pristine clone's module repos
# cvfpu's nested modules (names differ from paths: src-sv/common_cells -> src/common_cells, ...)
git -C "$T/third_party/cvfpu" submodule init
for m in src-sv/common_cells src-sv/fpu_div_sqrt_mvp tb/flexfloat; do
    git -C "$T/third_party/cvfpu" config "submodule.$m.url" "$PRI/.git/modules/third_party/cvfpu/modules/$m"
done
$G -C "$T/third_party/cvfpu" submodule update --quiet

apply_dir() {   # apply_dir <git dir> <patch dir> [noindex]; an empty patch dir is skipped (no stdin read by git apply)
    local d=$1 dir=$2 idx=${3:-}
    set -- "$dir"/*.patch
    [ -e "$1" ] || return 0
    local p
    for p; do
        echo "mk_vortex_tree: applying ${p#$PS/} in ${d#$PS/}"
        if [ -n "$idx" ]; then git -C "$d" apply --binary "$p"; else git -C "$d" apply --index --binary "$p"; fi
    done
}
for e in "${dirs[@]}"; do
    rest="${e#*:}"
    if [ "${e##*:}" = x ]; then apply_dir "$T/${rest%%:*}" "${e%%:*}" noindex; else apply_dir "$T/$rest" "${e%%:*}"; fi
done
if [ "${#EXTRAS[@]}" -gt 0 ]; then printf '%s\n' "${EXTRAS[@]}" > "$T/.ps_extra"; fi

echo "$FP" > "$T/.ps_patchset"
echo "mk_vortex_tree: created $T ($FP)"
