#!/usr/bin/env bash
# Train on full qwen_annotation dataset: train_captions.jsonl + val_captions.jsonl.
# Uses LoRA + masked-attribute alignment. Single-GPU or multi-GPU via NGPUS.
#
# Usage:
#   ./scripts/run_train_full.sh              # default: 1 GPU (GPU 1), 10 epochs
#   NGPUS=2 ./scripts/run_train_full.sh      # 2 GPUs
#   EPOCHS=5 BATCH=32 ./scripts/run_train_full.sh
#   GPU_ID=0 ./scripts/run_train_full.sh     # use GPU 0 instead of GPU 1

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# Use GPU 1 by default for single-GPU runs (set CUDA_VISIBLE_DEVICES so only that GPU is visible)
GPU_ID="${GPU_ID:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_LOGS="${TORCH_LOGS:--dynamo}"

DATA_DIR="${DATA_DIR:-/home/varsha.redla/ai-search}"
TRAIN_CAPTIONS="${DATA_DIR}/qwen_annotation/train_captions.jsonl"
VAL_CAPTIONS="${DATA_DIR}/qwen_annotation/val_captions_50k.jsonl"
# Single attr file for all image indices. If val uses different images, merge image_train_vectors.jsonl + image_val_vectors.jsonl into one.
IMAGE_ATTR="${IMAGE_ATTR:-${DATA_DIR}/qwen_annotation/image_train_vectors.jsonl}"
LOG_DIR="${DATA_DIR}/open_clip/logs_full"
RUN_NAME="full-no-hard-$(date +%Y%m%d-%H%M%S)"

EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-256}"
NGPUS="${NGPUS:-1}"
# NCCL collective timeout (minutes); increase if you hit "Watchdog caught collective operation timeout"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-120}"

for f in "$TRAIN_CAPTIONS" "$VAL_CAPTIONS" "$IMAGE_ATTR"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing: $f"
    exit 1
  fi
done

if [[ "$NGPUS" -gt 1 ]]; then
  AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
  if [[ "$AVAILABLE_GPUS" -lt "$NGPUS" ]]; then
    echo "Need at least $NGPUS GPUs, found $AVAILABLE_GPUS. Set NGPUS=$AVAILABLE_GPUS or use a machine with more GPUs."
    exit 1
  fi
  echo "Training with $NGPUS GPUs, batch-size $BATCH per GPU, $EPOCHS epochs (NCCL timeout: ${DIST_TIMEOUT_MINUTES} min)..."
  torchrun --nproc_per_node="$NGPUS" -m open_clip_train.main \
    --dist-timeout-minutes "$DIST_TIMEOUT_MINUTES" \
    --train-data "$TRAIN_CAPTIONS" \
    --val-data "$VAL_CAPTIONS" \
    --image-attr-vectors "$IMAGE_ATTR" \
    --dataset-type attr_jsonl \
    --masked-attr-alignment \
    --model hf-hub:timm/ViT-SO400M-14-SigLIP2 \
    --siglip \
    --lora \
    --lora-r 128 \
    --lora-alpha 256 \
    --lora-dropout 0.05 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --workers 4 \
    --logs "$LOG_DIR" \
    --name "$RUN_NAME" \
    --report-to "wandb" \
    --wandb-project-name "ai-search" \
    --viz-batch-matrix \
    --viz-batch-matrix-max-batch "$BATCH" \
    "$@"
else
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  echo "Training with 1 GPU (GPU $GPU_ID), batch-size $BATCH, $EPOCHS epochs..."
  python -m open_clip_train.main \
    --train-data "$TRAIN_CAPTIONS" \
    --val-data "$VAL_CAPTIONS" \
    --image-attr-vectors "$IMAGE_ATTR" \
    --dataset-type attr_jsonl \
    --masked-attr-alignment \
    --model hf-hub:timm/ViT-SO400M-14-SigLIP2 \
    --siglip \
    --lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lr  5e-5 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --workers 4 \
    --logs "$LOG_DIR" \
    --name "$RUN_NAME" \
    --report-to "wandb" \
    --wandb-project-name "ai-search" \
    --viz-batch-matrix \
    --viz-batch-matrix-max-batch "$BATCH" \
    "$@"
fi

echo "Training finished. Logs: $LOG_DIR/$RUN_NAME"
