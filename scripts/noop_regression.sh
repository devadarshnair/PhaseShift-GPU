#!/usr/bin/env bash
# noop_regression.sh -- PhaseShift no-op regression (created).
#
#   scripts/noop_regression.sh [--step ID] [--reg FILE] [--all] [--dry-run | --compare-only]
#   --all: REG + every P2 kernel of experiments/knobs.yaml on both trees + an "all modes 0" ps run per kernel
#
# Runs the REG set (experiments/noop_regression.yaml) on vortex-trees/base and vortex-trees/ps through vxrun.py (serial,
# every SimX job via lowload.sh, results in results/<ID>/), then requires identical instrs and cycles base vs ps for every
# entry, using only rows measured on the trees' CURRENT fingerprints. Exit 0 = PASS, 1 = FAIL (difference or missing
# row), 75 = DEFERRED (machine busy; re-run, finished runs are skipped). Details: scripts/noop_regression.py.
# ID defaults to $PS_STEP. Python: the phaseshift env's interpreter.
set -euo pipefail
PS="${PS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=env.sh
source "$PS/scripts/env.sh"
PY="${PS_PY:-$HOME/miniconda3/envs/phaseshift/bin/python}"
step="${PS_STEP:-}"; cmd=run; extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --step) step="$2"; shift 2 ;;
    --reg) extra+=(--reg "$2"); shift 2 ;;
    --dry-run) extra+=(--dry-run); shift ;;
    --all) extra+=(--all); shift ;;
    --compare-only) cmd=compare; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "noop_regression: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$step" ] || { echo "noop_regression: give --step ID or set PS_STEP" >&2; exit 2; }
export PS_STEP="$step"
exec nice -n 19 "$PY" -B "$PS/scripts/noop_regression.py" "$cmd" --step "$step" "${extra[@]}"
