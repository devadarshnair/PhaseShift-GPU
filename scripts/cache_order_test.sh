#!/usr/bin/env bash
# cache_order_test.sh -- directed/randomized SimX cache ordering test (tests/cache_order/cache_order_test.cpp).
#
#   scripts/cache_order_test.sh <build dir> [ops] [seeds] [K=V ...]
#
# Compiles the test with the exact flags of <build dir>'s last libsimx build (sw/runtime/simx_config.stamp: the same
# -D configuration, so every struct layout matches), links <build dir>/sw/runtime/libsimx.so, and runs it; one
# lowload job (serial, niced, memory-capped). Binary: build/tmp/cache_order/<build dir name>/cache_order_test.
# MODE=directed (env) runs the L2 RAW sweep instead of the randomized stress.
# K=V pairs go into the run's environment (e.g. PS_ORDER_CHECK=1, and PS_ORDER_FIX=0 on the private dev copy).
# Exit = the test's exit (0 = PASSED), 75 = DEFERRED by lowload.
set -euo pipefail
PS="${PS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PS/scripts/env.sh"
B=$(realpath "$1"); shift
ops=${1:-20000}; seeds=${2:-6}; shift $(( $# >= 2 ? 2 : $# ))
STAMP="$B/sw/runtime/simx_config.stamp"
[ -f "$STAMP" ] && [ -f "$B/sw/runtime/libsimx.so" ] || { echo "cache_order_test: no libsimx build in $B" >&2; exit 2; }
OUT="$PS/build/tmp/cache_order/$(basename "$B")"
mkdir -p "$OUT"
# the stamp is the CXXFLAGS line of the libsimx build; drop its dependency-file options
FLAGS=$(sed -e 's/ -MMD -MP -MF [^ ]*//' "$STAMP")
SRC="$PS/tests/cache_order/cache_order_test.cpp"
cat > "$OUT/build_run.sh" <<EOS
set -e
g++ $FLAGS -c "$SRC" -o "$OUT/cache_order_test.o"
g++ "$OUT/cache_order_test.o" -L"$B/sw/runtime" -lsimx -Wl,-rpath,"$B/sw/runtime" -o "$OUT/cache_order_test"
env $* "$OUT/cache_order_test" $ops $seeds ${MODE:-}
EOS
export TMPDIR="$PS/build/tmp"
exec "$LL" --rss 1.0 --mem 3 --label "${PS_STEP:-cache_order}:cache_order_test:$(basename "$B")" --log "$OUT/last_run.log" -- bash "$OUT/build_run.sh"
