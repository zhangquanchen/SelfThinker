#!/usr/bin/env bash
# OpenAI-compatible judge for the nine-benchmark scorer.
set -euo pipefail
PORT="${JUDGE_PORT:-8008}"
NAME="${JUDGE_NAME:-Qwen3-30B-A3B-Instruct-2507}"
MODEL="${JUDGE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
GPU="${JUDGE_GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
unset WORLD_SIZE LOCAL_WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK || true
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${NAME}" \
  --host 127.0.0.1 --port "${PORT}" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.82 \
  --max-model-len 4096 \
  --trust-remote-code
