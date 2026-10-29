#!/usr/bin/env bash
# Seven-benchmark inference and unified local-judge evaluation.
# Usage: scripts/eval_nine.sh infer SelfThinker
#        scripts/eval_nine.sh judge SelfThinker
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${SELFTHINKER_ROOT:-}" ]]; then
  ROOT="${SELFTHINKER_ROOT}"
elif [[ -d "${CODE}/models" || -d "${CODE}/data" ]]; then
  ROOT="${CODE}"
else
  ROOT="$(cd "${CODE}/.." && pwd)"
fi
STAGE="${1:-infer}"
MODEL="${2:-SelfThinker}"
NPROC="${NPROC:-8}"
PORT="${PORT:-29593}"
JUDGE_PORT="${JUDGE_PORT:-8008}"
VLMEVAL_REPO="${VLMEVAL_REPO:-${CODE}/third_party/vlmevalkit}"
OUT="${OUT:-${ROOT}/outputs/eval_${MODEL}}"
export SELFTHINKER_ROOT="${ROOT}"
export THINKMORPH_MODEL="${THINKMORPH_MODEL:-${ROOT}/external/ThinkMorph-7B}"
export BAGEL_MODEL="${BAGEL_MODEL:-${ROOT}/external/BAGEL-7B-MoT}"
export PYTHONPATH="${VLMEVAL_REPO}:${CODE}/scripts${PYTHONPATH:+:$PYTHONPATH}"
export LMUData="${ROOT}/data/lmudata"
export HF_HOME="${HF_HOME:-${ROOT}/external/hf_home}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python "${CODE}/scripts/register_models.py" "${VLMEVAL_REPO}"
unset WORLD_SIZE LOCAL_WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK || true
DATA="${DATA:-VSP_maze_task_main_original ChartQA_h_bar ChartQA_v_bar VStarBench MMVP BLINK SAT_circular CV-Bench-2D CV-Bench-3D}"
cd "${VLMEVAL_REPO}"
if [[ "$STAGE" == "infer" ]]; then
  python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="${NPROC}" --master_addr=127.0.0.1 --master_port="${PORT}" \
    run.py \
    --mode infer --reuse --reuse-aux 1 \
    --data ${DATA} \
    --model "${MODEL}" \
    --work-dir "${OUT}"
elif [[ "$STAGE" == "judge" ]]; then
  export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
  export OPENAI_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1/chat/completions"
  export LOCAL_LLM="${JUDGE_NAME:-Qwen3-30B-A3B-Instruct-2507}"
  export CUDA_VISIBLE_DEVICES="${JUDGE_GPU:-0}"
  export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
  python -u run.py \
    --mode eval --reuse --reuse-aux 0 \
    --data ${DATA} \
    --model "${MODEL}" \
    --work-dir "${OUT}" \
    --judge chatgpt-0125 \
    --verbose
  python "${CODE}/scripts/summarize_nine.py" "${OUT}" --model-key "${MODEL}"
else
  echo "usage: scripts/eval_nine.sh infer|judge ModelName" >&2
  exit 2
fi
