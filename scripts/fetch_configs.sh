#!/usr/bin/env bash
# PhaseShift: pinned HF config.json files (models note Q2). Config files only -- never weights.
#
#   scripts/fetch_configs.sh
#
# For each repo, scripts/fetch.py hf-config queries https://huggingface.co/api/models/<repo> for `sha` (commit),
# `gated` and `cardData.license`, downloads https://huggingface.co/<repo>/resolve/<sha>/config.json to
# configs/<org>__<name>.json (git-ignored: Llama-family licence terms), chmod 0444, and appends
# `repo sha sha256 licence gated date path` to configs/configs.lock (tracked). HTTP 401/403 = gated -> recorded with
# sha256 "-" and skipped. Idempotent: a file matching its lock row is not re-fetched (no network).
# Llama 3.x shapes come from the ungated unsloth/* mirrors, which carry Meta's Llama 3.x licence terms; the
# canonical source for citation is Meta's (gated) repo.
set -euo pipefail
PS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=(nice -n 19 conda run -n phaseshift --no-capture-output python -I "$PS/scripts/fetch.py")
REQUIRED=(
    unsloth/Llama-3.2-1B
    unsloth/Meta-Llama-3.1-8B
    mistralai/Mistral-7B-v0.1
    Qwen/Qwen2.5-7B
    Qwen/Qwen2.5-7B-Instruct-1M
    Qwen/Qwen2.5-32B
    unsloth/Meta-Llama-3.1-70B
    allenai/OLMoE-1B-7B-0924
    deepseek-ai/DeepSeek-V2-Lite
    HuggingFaceTB/SmolLM2-135M
)
OPTIONAL=(Qwen/Qwen3-30B-A3B mistralai/Mixtral-8x7B-v0.1)   # "optional"; config.json only (KB each)
for r in "${REQUIRED[@]}" "${OPTIONAL[@]}"; do
    "${PY[@]}" hf-config "$r" --outdir "$PS/configs" --lock "$PS/configs/configs.lock"
done
