#!/usr/bin/env bash
# PhaseShift: pinned serving traces (models note Q3). Small length-only traces (~9.6 MB total).
#
#   scripts/fetch_traces.sh
#
# scripts/fetch.py gh-raw looks up each repo's default branch (`gh api repos/<o>/<r> --jq .default_branch`, not
# guessed), pins it to a commit SHA (`gh api repos/<o>/<r>/commits/<branch> --jq .sha`), downloads
# https://raw.githubusercontent.com/<o>/<r>/<sha>/<path> into data/traces/, checks the byte size and the git blob
# SHA-1 from `gh api repos/<o>/<r>/contents/<path>?ref=<sha>` (a Git-LFS pointer is detected and rejected), chmod
# 0444, and appends a data/MANIFEST.tsv row. Idempotent (manifest-matching files are not re-fetched).
#   Azure LLM inference trace 2023 (Splitwise)  CC-BY-4.0   code 320,117 B, conv 719,188 B
#   Mooncake FAST'25 traces (kvcache-ai)       Apache-2.0  conversation 3,029,533 B, toolagent 4,415,857 B,
#                                                          synthetic 1,136,313 B
# Not fetched: BurstGPT, Azure 2024 one-week traces, text datasets.
set -euo pipefail
PS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=(nice -n 19 conda run -n phaseshift --no-capture-output python -I "$PS/scripts/fetch.py")
T="$PS/data/traces"
AZ=Azure/AzurePublicDataset
MC=kvcache-ai/Mooncake
"${PY[@]}" gh-raw $AZ data/AzureLLMInferenceTrace_code.csv --dest "$T/AzureLLMInferenceTrace_code.csv" --size 320117  --licence CC-BY-4.0 --step S9
"${PY[@]}" gh-raw $AZ data/AzureLLMInferenceTrace_conv.csv --dest "$T/AzureLLMInferenceTrace_conv.csv" --size 719188  --licence CC-BY-4.0 --step S9
for f in conversation:3029533 toolagent:4415857 synthetic:1136313; do
    n=${f%%:*}; sz=${f##*:}
    "${PY[@]}" gh-raw $MC "FAST25-release/traces/${n}_trace.jsonl" --dest "$T/mooncake_${n}_trace.jsonl" --size "$sz" \
        --licence Apache-2.0 --step S9
done
