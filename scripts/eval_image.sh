#!/usr/bin/env bash
# Run one image-condition job. Set MUCG_INTV, MUCG_CHART_TSV, MUCG_INTV_OUT, and optionally MUCG_INTV_MODEL.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${SELFTHINKER_ROOT:-}" ]]; then
  ROOT="${SELFTHINKER_ROOT}"
elif [[ -d "${CODE}/models" || -d "${CODE}/data" ]]; then
  ROOT="${CODE}"
else
  ROOT="$(cd "${CODE}/.." && pwd)"
fi
VLMEVAL_REPO="${VLMEVAL_REPO:-${CODE}/third_party/vlmevalkit}"
export PYTHONPATH="${VLMEVAL_REPO}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
if [[ -z "${MUCG_INTV_MODEL:-}" ]]; then
  if [[ -f "${ROOT}/models/model.safetensors" ]]; then
    export MUCG_INTV_MODEL="${ROOT}/models"
  else
    export MUCG_INTV_MODEL="${ROOT}/models/selfthinker"
  fi
fi
export MUCG_MAX_THINK="${MUCG_MAX_THINK:-512}"
NPROC="${NPROC:-8}"
PORT="${PORT:-29571}"
unset WORLD_SIZE LOCAL_WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK || true
python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node="${NPROC}" --master_addr=127.0.0.1 --master_port="${PORT}" \
  "${CODE}/scripts/eval_image_condition.py"
