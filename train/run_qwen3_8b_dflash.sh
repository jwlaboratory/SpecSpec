#!/bin/bash
# DFlash 8B->1B draft-model pre-training (Qwen3-8B target).
# Thin wrapper around the official SpecForge recipe. REQUIRES a CUDA box
# (recipe assumes 8 GPUs) with sglang installed — it will NOT run on this Mac.
#
# Pipeline:
#   1. prepare + regenerate the dataset with the target model (see below)
#   2. launch training (this script)
#
# For a no-GPU sanity check of the draft + loss code, use dry_run.py instead.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECFORGE="${SPECFORGE_ROOT:-$HERE/../SpecForge}"
NUM_GPUS="${1:-8}"
ATTENTION_BACKEND="${2:-flex_attention}"

DATA="$SPECFORGE/cache/dataset/perfectblend_qwen3-8b_regen.jsonl"
if [[ ! -f "$DATA" ]]; then
  cat <<EOF
[!] Training data not found: $DATA

Regenerate it first (needs the target model served via sglang):

  python $SPECFORGE/scripts/prepare_data.py --dataset perfectblend
  python3 -m sglang.launch_server --model Qwen/Qwen3-8B --dtype bfloat16 \\
      --mem-frac=0.8 --port 30000
  python $SPECFORGE/scripts/regenerate_train_data.py \\
      --model Qwen/Qwen3-8B --is-reasoning-model --concurrency 128 \\
      --max-tokens 98304 --temperature 0.8 --server-address localhost:30000 \\
      --input-file-path $SPECFORGE/cache/dataset/perfectblend_train.jsonl \\
      --output-file-path $DATA
EOF
  exit 1
fi

export TORCHINDUCTOR_CACHE_DIR="$SPECFORGE/cache/compiled_kernels"
export SPECFORGE_DATA_NUM_PROC=32

torchrun \
  --standalone \
  --nproc_per_node "$NUM_GPUS" \
  "$SPECFORGE/scripts/train_dflash.py" \
  --target-model-path Qwen/Qwen3-8B \
  --draft-config-path "$SPECFORGE/configs/qwen3-8b-dflash.json" \
  --train-data-path "$DATA" \
  --output-dir "$SPECFORGE/outputs/qwen3-8b-perfectblend" \
  --num-epochs 6 \
  --batch-size 4 \
  --learning-rate 6e-4 \
  --warmup-ratio 0.04 \
  --max-grad-norm 1.0 \
  --max-length 3072 \
  --chat-template qwen \
  --attention-backend "$ATTENTION_BACKEND" \
  --loss-decay-gamma 7.0 \
  --log-interval 50 \
  --save-interval 1000 \
  --report-to wandb \
  --wandb-project specforge-qwen3-8b-dflash \
  --target-model-backend sglang \
  --block-size 16 \
  --num-anchors 512 \
  --wandb-name qwen3-8b-dflash-perfectblend
