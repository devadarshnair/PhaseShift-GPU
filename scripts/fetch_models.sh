#!/usr/bin/env bash
# PhaseShift model fetch. Pinned, write-once, idempotent.
#
#   scripts/fetch_models.sh
#
# Pins karpathy/tinyllamas (MIT, ungated) to its current Hub commit SHA and downloads, via scripts/fetch.py hf-file,
#   stories260K/stories260K.bin  1,056,540 B  -> data/models/stories260K.bin
#   stories260K/tok512.bin           6,227 B  -> data/models/tok512.bin
#   stories15M.bin              60,816,028 B  -> data/models/stories15M.bin
#   stories15M.pt               60,789,090 B  -> data/models/stories15M.pt   (PyTorch pickle: never torch.load it
#                                                                              with weights_only=False)
# Each file is size-checked and sha256-checked against the Hub's LFS record, chmod 0444, and recorded in
# data/MANIFEST.tsv (path, URL with the commit SHA, revision, bytes, sha256, licence MIT, date). A re-run whose files
# match their manifest rows makes no network call. SmolLM2 weights are NOT fetched here.
set -euo pipefail
PS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=(nice -n 19 conda run -n phaseshift --no-capture-output python -I "$PS/scripts/fetch.py")
M="$PS/data/models"
R=karpathy/tinyllamas
"${PY[@]}" hf-file $R stories260K/stories260K.bin --dest "$M/stories260K.bin" --size 1056540  --licence MIT --step S8
"${PY[@]}" hf-file $R stories260K/tok512.bin      --dest "$M/tok512.bin"      --size 6227     --licence MIT --step S8
"${PY[@]}" hf-file $R stories15M.bin              --dest "$M/stories15M.bin"  --size 60816028 --licence MIT --step S8
"${PY[@]}" hf-file $R stories15M.pt               --dest "$M/stories15M.pt"   --size 60789090 --licence MIT --step S8
