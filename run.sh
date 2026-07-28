#!/usr/bin/env bash
set -euo pipefail

# Create and activate venv
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "Install PyTorch: visit https://pytorch.org and choose the right command for your platform"
# Example CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

# Run baseline evaluation
python baseline_evaluation.py --data-dir data/stanford40 --out baseline_results/baseline.csv

# Run improved prompt evaluation
python action_aware_prompts.py --data-dir data/stanford40 --out action_aware_results/action_aware.csv

# Compare results
python compare_dual_fusion_classes.py --baseline baseline_results/baseline.csv --improved action_aware_results/action_aware.csv --out results/comparison.csv
