#!/usr/bin/env bash
# PhaseShift: build llama2.c's reference binaries from COPIES under build/llama2c.
#
#   $LL --rss 0.5 --mem 2 --label S8-llama2c --log results/S8/logs/build.log -- scripts/build_llama2c.sh
#
# The clone third_party/llama2.c is read-only: run.c / runq.c are copied to build/llama2c/src/ and compiled there.
#   build/llama2c/run         gcc -O3 run.c  -lm   (no -Ofast / -ffast-math: the golden must be IEEE-deterministic)
#   build/llama2c/runq        gcc -O3 runq.c -lm   (Q8_0 path; needs a version-2 quantized export, made)
#   build/llama2c/run_tokens  same as run, built from run.c + patches/llama2c/run_tokens.patch, which only adds one
#                             fprintf(stderr, "PS_TOK pos token next forced") per forward step; it is used to record
#                             the golden token ids, and its stdout must equal run's stdout byte for byte (checked by
#                             scripts/golden_llama2c.sh).
# No -fopenmp (single thread on this shared machine; runq.c's `#pragma omp` lines are then ignored). Idempotent:
# rebuilds every time (seconds). Writes build/llama2c/BUILDINFO (gcc version, flags, source sha256s).
set -euo pipefail
PS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$PS/third_party/llama2.c"
B="$PS/build/llama2c"
mkdir -p "$B/src"
cp -f "$SRC/run.c" "$SRC/runq.c" "$B/src/"
cp -f "$SRC/run.c" "$B/src/run_tokens.c"
patch --quiet --no-backup-if-mismatch "$B/src/run_tokens.c" "$PS/patches/llama2c/run_tokens.patch"
CFLAGS="-O3"
for t in run runq run_tokens; do
    gcc $CFLAGS -o "$B/$t" "$B/src/$t.c" -lm
done
{
    echo "llama2.c commit: $(git --no-optional-locks -C "$SRC" rev-parse HEAD)"
    echo "gcc: $(gcc --version | head -1)"
    echo "CFLAGS: $CFLAGS  LDLIBS: -lm"
    (cd "$B" && sha256sum src/run.c src/runq.c src/run_tokens.c run runq run_tokens)
} > "$B/BUILDINFO"
cat "$B/BUILDINFO"
