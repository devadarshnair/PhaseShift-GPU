#!/usr/bin/env bash
# PhaseShift Vortex build driver .
#
#   build_vortex.sh {base|ps} [configure|third_party|software|all]      (default: all = the three in order)
#   build_vortex.sh times                                               (ledger -> results/S6/build_times.csv)
#
# Builds ONLY what PhaseShift needs, in $PS/build/vortex-<X> from the working tree $PS/vortex-trees/<X>:
#   configure    <tree>/configure --xlen=32 --tooldir=$HOME/tools --osversion=ubuntu/focal   (light, niced)
#   third_party  make -C <build> third_party   via lowload.sh (--rss 1.0 --mem 5; measured 0.67 GB peak): softfloat, Ramulator2 (+GDDR6 ext),
#                cocogfx. Builds IN-SOURCE in vortex-trees/<X>/third_party (never in third_party/vortex).
#   software     make -C <build> software      via lowload.sh (--rss 0.5 --mem 4; measured 0.22 GB peak): sw/kernel + sw/runtime/stub.
# Never runs a top-level `make`/`make all` (that would build hw, every simulator and all tests).
# SimX itself is built on demand by ci/blackbox.sh .
#
# Ramulator is configured with RAMULATOR_CMAKE_FLAGS (infra patch 0000 default + a ccache compiler launcher, so the
# second tree's Ramulator build hits the shared ccache; CCACHE_BASEDIR=$PS from env.sh makes paths tree-independent).
# The launcher only wraps the compiler; it does not change the produced objects.
# Labels/logs: S6-tp-<X>, S6-sw-<X> under results/${PS_STEP:-S6}/logs/ (label prefix = ${PS_STEP:-S6}).
# Exit status: the failing stage's status; 75 = DEFERRED from lowload.sh (not enough memory in the wait window).
set -euo pipefail

PS="${PS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=env.sh
source "$PS/scripts/env.sh"
STEP="${PS_STEP:-S6}"
LOGDIR="$PS/results/$STEP/logs"
# /tmp is a 3.8 GB tmpfs (RAM-backed, shared with other users of the machine): compiler temporaries (cc*.s) there
# cost RAM and hit ENOSPC when /tmp fills. Keep them on disk under the git-ignored build dir.
export TMPDIR="$PS/build/tmp"
mkdir -p "$TMPDIR"

usage() { echo "usage: build_vortex.sh {base|ps|<tree>} [configure|third_party|software|all] | build_vortex.sh times   (<tree>: [a-z][a-z0-9]{1,15}, same rule as vxrun valid_tree)" >&2; exit 2; }
[ $# -ge 1 ] || usage

if [ "$1" = times ]; then
    out="$PS/results/S6/build_times.csv"
    mkdir -p "$(dirname "$out")"
    # One row per S6 run (every attempt, in ledger order, incl. failed ones); exit != 0 marks a failed run.
    awk -F'\t' 'BEGIN{OFS=","; print "label,exit,maxrss_kb,maxrss_mb,wall_s,wall_h,end_time"}
        $1=="end" && $3 ~ /^S6-/ {print $3, $4, $5, sprintf("%.1f",$5/1024), $6, sprintf("%.2f",$6/3600), $7}' \
        "$PS/results/lowload_ledger.tsv" > "$out"
    cat "$out"
    exit 0
fi

X="$1"; what="${2:-all}"
[[ "$X" =~ ^[a-z][a-z0-9]{1,15}$ ]] || usage   # same pattern as vxrun.valid_tree (base, ps and psX trees all match)
case "$what" in configure|third_party|software|all) ;; *) usage ;; esac
T="$PS/vortex-trees/$X"
B="$PS/build/vortex-$X"
[ -f "$T/.ps_patchset" ] || { echo "build_vortex: $T missing or incomplete; run scripts/mk_vortex_tree.sh --patchset $X" >&2; exit 1; }

check_home() {   # the build dir must point at the working tree, never at the pristine clone
    local vh
    vh=$(sed -n 's/^VORTEX_HOME[[:space:]]*?\{0,1\}=[[:space:]]*//p' "$B/config.mk" | head -1)
    [ "$(realpath "$vh")" = "$(realpath "$T")" ] || { echo "build_vortex: $B/config.mk VORTEX_HOME=$vh is not $T" >&2; exit 1; }
}

do_configure() {
    mkdir -p "$B"
    ( cd "$B" && nice -n 19 "$T/configure" --xlen=32 --tooldir="$HOME/tools" --osversion=ubuntu/focal )
    check_home
}
do_third_party() {
    check_home; mkdir -p "$LOGDIR"
    "$LL" --rss 1.0 --mem 5 --label "$STEP-tp-$X" --log "$LOGDIR/tp-$X.log" -- \
        make -C "$B" third_party \
        RAMULATOR_JOBS=1 \
        "RAMULATOR_CMAKE_FLAGS=-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
}
do_software() {
    check_home; mkdir -p "$LOGDIR"
    "$LL" --rss 0.5 --mem 4 --label "$STEP-sw-$X" --log "$LOGDIR/sw-$X.log" -- \
        make -C "$B" software \
        RAMULATOR_JOBS=1 \
        "RAMULATOR_CMAKE_FLAGS=-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
}

case "$what" in
    configure)   do_configure ;;
    third_party) do_third_party ;;
    software)    do_software ;;
    all)         do_configure; do_third_party; do_software ;;
esac
