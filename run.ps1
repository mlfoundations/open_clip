# Run common experiments on Windows PowerShell

# Activate virtual environment
if (Test-Path .venv) {
    .\.venv\Scripts\Activate.ps1
} else {
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
}

Write-Host "Install PyTorch: choose the correct command from https://pytorch.org"
Write-Host "Example CPU-only: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"

# Install other dependencies
pip install -r requirements.txt

# Run baseline evaluation
python baseline_evaluation.py --data-dir data/stanford40 --out baseline_results/baseline.csv

# Run improved prompt evaluation
python action_aware_prompts.py --data-dir data/stanford40 --out action_aware_results/action_aware.csv

# Compare results
python compare_dual_fusion_classes.py --baseline baseline_results/baseline.csv --improved action_aware_results/action_aware.csv --out results/comparison.csv
