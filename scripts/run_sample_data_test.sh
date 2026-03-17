#!/usr/bin/env bash
# Test training + eval (including retrieval metrics) on a small sample of data.
# Creates tiny train/val/attr files from the full qwen_annotation data if they don't exist,
# then runs 1 epoch so we hit the retrieval-metrics path quickly.
#
# Usage:
#   ./scripts/run_sample_data_test.sh              # 1 GPU, 2000 samples
#   NGPUS=2 ./scripts/run_sample_data_test.sh     # 2 GPUs
#   SAMPLE_LINES=500 ./scripts/run_sample_data_test.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_LOGS="${TORCH_LOGS:--dynamo}"

DATA_DIR="${DATA_DIR:-/home/varsha.redla/ai-search}"
FULL_TRAIN="${DATA_DIR}/qwen_annotation/train_captions.jsonl"
FULL_VAL="${DATA_DIR}/qwen_annotation/val_captions.jsonl"
FULL_ATTR="${DATA_DIR}/qwen_annotation/image_train_vectors.jsonl"

SAMPLE_LINES="${SAMPLE_LINES:-2000}"
TRAIN_TINY="${DATA_DIR}/qwen_annotation/train_captions_tiny.jsonl"
VAL_TINY="${DATA_DIR}/qwen_annotation/val_captions_tiny.jsonl"
ATTR_TINY="${DATA_DIR}/qwen_annotation/image_train_vectors_tiny.jsonl"

LOG_DIR="${DATA_DIR}/open_clip/logs_sample_test"
RUN_NAME="sample-eval-test-$(date +%Y%m%d-%H%M%S)"
NGPUS="${NGPUS:-1}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-10}"

# Create sample files if missing
for f in "$FULL_TRAIN" "$FULL_VAL" "$FULL_ATTR"; do
  if [[ ! -f "$f" ]]; then
    echo "Full data file not found: $f"
    exit 1
  fi
done

if [[ ! -f "$TRAIN_TINY" ]] || [[ ! -f "$VAL_TINY" ]] || [[ ! -f "$ATTR_TINY" ]]; then
  echo "Creating sample data (first ${SAMPLE_LINES} lines)..."
  head -n "$SAMPLE_LINES" "$FULL_TRAIN" > "$TRAIN_TINY"
  head -n "$SAMPLE_LINES" "$FULL_VAL"   > "$VAL_TINY"
  head -n "$SAMPLE_LINES" "$FULL_ATTR" > "$ATTR_TINY"
  echo "Created: $TRAIN_TINY, $VAL_TINY, $ATTR_TINY"
else
  echo "Using existing tiny files: $TRAIN_TINY, $VAL_TINY, $ATTR_TINY"
fi

echo "Running sample test: 1 epoch, batch 32, NGPUS=$NGPUS (will run eval and retrieval metrics)..."

if [[ "$NGPUS" -gt 1 ]]; then
  AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
  if [[ "$AVAILABLE_GPUS" -lt "$NGPUS" ]]; then
    echo "Need at least $NGPUS GPUs, found $AVAILABLE_GPUS. Set NGPUS=$AVAILABLE_GPUS."
    exit 1
  fi
  torchrun --nproc_per_node="$NGPUS" -m open_clip_train.main \
    --dist-timeout-minutes "$DIST_TIMEOUT_MINUTES" \
    --train-data "$TRAIN_TINY" \
    --val-data "$VAL_TINY" \
    --image-attr-vectors "$ATTR_TINY" \
    --dataset-type attr_jsonl \
    --masked-attr-alignment \
    --model hf-hub:timm/ViT-SO400M-14-SigLIP2 \
    --siglip \
    --lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 \
    --epochs 1 \
    --batch-size 32 \
    --workers 0 \
    --logs "$LOG_DIR" \
    --name "$RUN_NAME" \
    --report-to "none" \
    "$@"
else
  python -m open_clip_train.main \
    --train-data "$TRAIN_TINY" \
    --val-data "$VAL_TINY" \
    --image-attr-vectors "$ATTR_TINY" \
    --dataset-type attr_jsonl \
    --masked-attr-alignment \
    --model hf-hub:timm/ViT-SO400M-14-SigLIP2 \
    --siglip \
    --lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 \
    --epochs 1 \
    --batch-size 32 \
    --workers 0 \
    --logs "$LOG_DIR" \
    --name "$RUN_NAME" \
    --report-to "none" \
    "$@"
fi

echo "Sample test finished. Check logs for 'Retrieval metrics total' timing: $LOG_DIR/$RUN_NAME"
