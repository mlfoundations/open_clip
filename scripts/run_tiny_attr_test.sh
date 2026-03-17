#!/usr/bin/env bash
# Quick test of masked-attribute alignment training flow using tiny train/val subsets.
# Uses: qwen_annotation/train_captions_tiny.jsonl, val_captions_tiny.jsonl, image_val_vectors_tiny.jsonl

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

DATA_DIR="${DATA_DIR:-/home/varsha.redla/ai-search}"
TRAIN_CAPTIONS="${DATA_DIR}/qwen_annotation/train_captions_tiny.jsonl"
VAL_CAPTIONS="${DATA_DIR}/qwen_annotation/val_captions_tiny.jsonl"
IMAGE_ATTR="${DATA_DIR}/qwen_annotation/image_val_vectors_tiny.jsonl"

for f in "$TRAIN_CAPTIONS" "$VAL_CAPTIONS" "$IMAGE_ATTR"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing: $f"
    exit 1
  fi
done

python -m open_clip_train.main \
  --train-data "$TRAIN_CAPTIONS" \
  --val-data "$VAL_CAPTIONS" \
  --image-attr-vectors "$IMAGE_ATTR" \
  --dataset-type attr_jsonl \
  --masked-attr-alignment \
  --model ViT-B-16-SigLIP-256 \
  --siglip \
  --pretrained "" \
  --epochs 1 \
  --batch-size 4 \
  --workers 0 \
  --logs "${DATA_DIR}/open_clip/logs_tiny_test" \
  --name "tiny-attr-test" \
  --report-to "none" \
  "$@"

echo "Tiny attr test finished. Logs: ${DATA_DIR}/open_clip/logs_tiny_test/tiny-attr-test"
