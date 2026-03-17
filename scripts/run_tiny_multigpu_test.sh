#!/usr/bin/env bash
# Quick multi-GPU test: DDP + LoRA + masked-attribute alignment (tiny data, 1 epoch).
# Requires at least 2 GPUs. Uses same tiny jsonl as run_tiny_attr_test.sh.
#
# Usage:
#   ./scripts/run_tiny_multigpu_test.sh
#   NGPUS=4 ./scripts/run_tiny_multigpu_test.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# Disable NCCL P2P to avoid hangs on systems where GPU-to-GPU direct comm fails
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
# Reduce PyTorch dynamo / FakeTensor log noise (optional; unset to debug)
export TORCH_LOGS="${TORCH_LOGS:--dynamo}"

NGPUS="${NGPUS:-2}"
DATA_DIR="${DATA_DIR:-/home/varsha.redla/ai-search}"
TRAIN_CAPTIONS="${DATA_DIR}/qwen_annotation/train_captions_tiny.jsonl"
VAL_CAPTIONS="${DATA_DIR}/qwen_annotation/val_captions_tiny.jsonl"
IMAGE_ATTR="${DATA_DIR}/qwen_annotation/image_val_vectors_tiny.jsonl"
LOG_DIR="${DATA_DIR}/open_clip/logs_tiny_multigpu_test"
RUN_NAME="tiny-multigpu-lora-test-$(date +%Y%m%d-%H%M%S)"

for f in "$TRAIN_CAPTIONS" "$VAL_CAPTIONS" "$IMAGE_ATTR"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing: $f"
    exit 1
  fi
done

AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [[ "$AVAILABLE_GPUS" -lt "$NGPUS" ]]; then
  echo "Need at least $NGPUS GPUs, found $AVAILABLE_GPUS. Set NGPUS=$AVAILABLE_GPUS or use a machine with more GPUs."
  exit 1
fi

echo "Running multi-GPU test with $NGPUS GPUs (DDP + LoRA + masked-attr), 1 epoch..."
torchrun --nproc_per_node="$NGPUS" -m open_clip_train.main \
  --train-data "$TRAIN_CAPTIONS" \
  --val-data "$VAL_CAPTIONS" \
  --image-attr-vectors "$IMAGE_ATTR" \
  --dataset-type attr_jsonl \
  --masked-attr-alignment \
  --model ViT-SO400M-14-SigLIP2 \
  --siglip \
  --pretrained "" \
  --lora \
  --lora-r 4 \
  --lora-alpha 16 \
  --epochs 1 \
  --batch-size 4 \
  --workers 0 \
  --logs "$LOG_DIR" \
  --name "$RUN_NAME" \
  --report-to "wandb" \
  --wandb-project-name "ai-search" \
  "$@"

echo "Multi-GPU test finished. Logs: $LOG_DIR/$RUN_NAME"
