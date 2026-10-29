#!/usr/bin/env bash
# Train SelfThinker (answer) or SelfThinker-Zero (vote) from ThinkMorph-7B.
# Usage: code/scripts/train_selfthinker.sh answer|vote
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${SELFTHINKER_ROOT:-}" ]]; then
  ROOT="${SELFTHINKER_ROOT}"
elif [[ -d "${CODE}/models" || -d "${CODE}/data" ]]; then
  ROOT="${CODE}"
else
  ROOT="$(cd "${CODE}/.." && pwd)"
fi
VARIANT="${1:-answer}"
NPROC="${NPROC:-8}"
PORT="${PORT:-29653}"
VISUAL_OPSD_REPO="${VISUAL_OPSD_REPO:-${CODE}/third_party/visual_opsd}"
THINKMORPH_MODEL="${THINKMORPH_MODEL:-${ROOT}/external/ThinkMorph-7B}"
if [[ "$VARIANT" == "answer" ]]; then
  DATA="${ROOT}/data/audit_pass_l0_vsearch_trainonly"
  SEED="${SEED:-4411}"
  NAME="selfthinker"
elif [[ "$VARIANT" == "vote" ]]; then
  DATA="${ROOT}/data/audit_pass_l1_vsearch_trainonly"
  SEED="${SEED:-4409}"
  NAME="selfthinker-zero"
else
  echo "usage: code/scripts/train_selfthinker.sh answer|vote" >&2
  exit 2
fi
OUT="${OUT:-${ROOT}/outputs/${NAME}}"
mkdir -p "${OUT}/checkpoints"
export PYTHONPATH="${VISUAL_OPSD_REPO}:${CODE}/scripts"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export SELFTHINKER_ROOT="${ROOT}"
export VISUAL_OPSD_DATA_ROOT="${DATA}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WORLD_SIZE LOCAL_WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK || true
cd "${VISUAL_OPSD_REPO}"
python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node="${NPROC}" --master_addr=127.0.0.1 --master_port="${PORT}" \
  scripts/visual_opsd/train_visual_opsd.py \
  --model_path "${THINKMORPH_MODEL}" \
  --dataset_config_file "${CODE}/configs/visual_opsd_jsd_vs.yaml" \
  --finetune_from_hf True --resume_model_only True --finetune_from_ema False \
  --resume_from "${THINKMORPH_MODEL}" \
  --results_dir "${OUT}" --checkpoint_dir "${OUT}/checkpoints" \
  --global_seed "${SEED}" \
  --lr 1e-5 --lr_scheduler cosine --min_lr 1e-7 --warmup_steps 200 \
  --num_workers 2 --prefetch_factor 2 --gradient_accumulation_steps 1 \
  --cpu_offload False --use_flex False --visual_gen False \
  --total_steps 1000 --log_every 10 --save_every 250 \
  --num_shard "${NPROC}" --num_replicate 1 --sharding_strategy FULL_SHARD \
  --teacher_mode ema --ema_decay 0.995 --max_reference_rounds 2 \
  --sample_max_new_tokens 128 --sample_temperature 1.0 --sample_do_sample True \
  --sample_max_image_skips 1 --max_forward_tokens 10240 \
  --jsd_beta 0.5 --jsd_temperature 1.0 --jsd_top_k 256 --jsd_token_clip 0.05 \
  --jsd_weight 1.0 --ce_weight 0.0 \
  --wandb_project selfthinker --wandb_name "${NAME}" --wandb_offline True
echo "checkpoint: ${OUT}/checkpoints/0000999/ema.safetensors"
