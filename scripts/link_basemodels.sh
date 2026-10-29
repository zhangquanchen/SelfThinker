#!/usr/bin/env bash
# Attach ThinkMorph tokenizer, configs, and VAE to a trained checkpoint directory.
# The weight file model.safetensors stays the trained EMA.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${SELFTHINKER_ROOT:-}" ]]; then
  ROOT="${SELFTHINKER_ROOT}"
elif [[ -d "${CODE}/models" || -d "${CODE}/data" ]]; then
  ROOT="${CODE}"
else
  ROOT="$(cd "${CODE}/.." && pwd)"
fi
THINKMORPH_MODEL="${THINKMORPH_MODEL:?Set THINKMORPH_MODEL to a local ThinkMorph-7B directory}"
DEST="${1:-${ROOT}/models}"
FILES=(
  ae.safetensors
  config.json
  generation_config.json
  llm_config.json
  vit_config.json
  tokenizer_config.json
  tokenizer.json
  merges.txt
  vocab.json
)
mkdir -p "${DEST}"
test -s "${DEST}/model.safetensors"
for f in "${FILES[@]}"; do
  test -e "${THINKMORPH_MODEL}/${f}"
  ln -sfn "${THINKMORPH_MODEL}/${f}" "${DEST}/${f}"
done
ln -sfn model.safetensors "${DEST}/ema.safetensors"
echo "linked ThinkMorph companion files into ${DEST}"
