# Improving OpenCLIP for Human Action Recognition on Stanford-40

## Project Overview

This repository adapts the official OpenCLIP implementation to evaluate and improve zero-shot human action recognition on the Stanford-40 Actions dataset.

The project extends the baseline through:

- Stanford-40 dataset integration
- Baseline zero-shot evaluation
- Action-aware prompt engineering
- Body-part-aware prompt engineering
- Prompt ensembles
- Overall and per-class performance analysis
- CSV result export
- Confusion-matrix generation
- Baseline-versus-improved comparison

## Research Objective

The objective of this research is to investigate whether action-aware prompt engineering can improve OpenCLIP’s ability to recognize human actions in still images.

The research focuses particularly on action classes that are difficult to distinguish because they contain similar objects, body movements or visual environments.

## Attribution and Original Sources

This research uses the following open-source projects and dataset:

- [OpenCLIP by ML Foundations](https://github.com/mlfoundations/open_clip)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Stanford-40 Actions Dataset](http://vision.stanford.edu/Datasets/40actions.html)
- [OpenCLIP Paper](https://arxiv.org/abs/2212.07143)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)

The original OpenCLIP source code was used as the baseline framework. Its original licence, copyright notices and attribution have been retained.

The original OpenCLIP README is preserved as:

```text
README_OPENCLIP_ORIGINAL.md
```

## Dataset

The Stanford-40 Actions dataset contains approximately 9,500 images across 40 human-action classes.

Example classes include:

- Applauding
- Cutting vegetables
- Pouring liquid
- Phoning
- Texting message
- Washing dishes
- Writing on a book

The dataset is not included in this repository because of its size. It must be downloaded separately from the official [Stanford-40 Actions Dataset page](http://vision.stanford.edu/Datasets/40actions.html).

## Modifications and Improvements

### 1. Stanford-40 Dataset Integration

- Adapted the evaluation pipeline for the Stanford-40 Actions dataset.
- Added support for all 40 action classes.
- Configured the Stanford-40 test split.
- Added image-path and ground-truth label processing.

### 2. Baseline Zero-Shot Evaluation

- Implemented zero-shot action classification using OpenCLIP.
- Used a basic text prompt for each action class.
- Compared normalized image and text embeddings.
- Recorded the predicted and ground-truth classes.
- Calculated overall and per-class performance.

### 3. Action-Aware Prompt Engineering

- Replaced basic class-name prompts with detailed action descriptions.
- Added information about relevant people, objects, interactions and environments.
- Designed prompts to reduce confusion between visually similar actions.

### 4. Body-Part-Aware Prompt Engineering

- Added descriptions of relevant hand, arm and body movements.
- Included information about physical interactions between people and objects.
- Used body-position information to improve fine-grained action recognition.

### 5. Prompt Ensemble

- Created multiple prompt templates for each action class.
- Encoded each prompt using the OpenCLIP text encoder.
- Normalized the individual text embeddings.
- Averaged the text embeddings to obtain one representation per class.
- Compared prompt-ensemble performance against the baseline.

### 6. Evaluation and Error Analysis

The evaluation pipeline calculates:

- Top-1 accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Per-class accuracy
- Per-class precision, recall and F1-score
- Correct and incorrect prediction counts
- Confusion matrices
- Per-class performance gains and losses
- Net changes in correct predictions

### 7. Results Export

- Exported overall evaluation results to CSV files.
- Exported per-class results to CSV files.
- Generated baseline-versus-improved comparison tables.
- Identified action classes that improved or declined.

## Experimental Methods

| Method | Description |
|---|---|
| Baseline OpenCLIP | Zero-shot classification using a basic prompt |
| Action-aware prompts | Prompts containing action, object and scene information |
| Body-part-aware prompts | Prompts describing body movement and object interaction |
| Prompt ensemble | Multiple normalized prompt embeddings averaged for each class |

## Repository Structure

```text
open_clip/
├── src/                         # Original OpenCLIP source code
├── docs/                        # OpenCLIP documentation
├── tests/                       # OpenCLIP tests
├── baseline_evaluation.py       # Stanford-40 baseline evaluation
├── zero_shot.py                 # Zero-shot classification
├── generate_context_views.py    # Context-view generation
├── compare_dual_fusion_classes.py
├── baseline_results/            # Baseline evaluation results
├── action_aware_results/        # Improved-method results
├── requirements.txt             # Python dependencies
├── README.md                    # Research-project documentation
└── README_OPENCLIP_ORIGINAL.md  # Original OpenCLIP README
```

Remove or rename any entries above that do not match the actual files in the repository.

## Requirements

The main Python packages used include:

- PyTorch
- Torchvision
- OpenCLIP
- NumPy
- Pandas
- Scikit-learn
- Pillow
- tqdm
- Matplotlib
- Seaborn
- timm

A CUDA-compatible GPU is recommended but not strictly required. Evaluation can also run on a CPU, although it will be slower.

## Installation

### 1. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

If the local OpenCLIP package must be installed in editable mode, run this command from the repository’s main directory:

```bash
pip install -e .
```

## Quick start

Follow these minimal steps to reproduce an evaluation run (Windows PowerShell shown first).

PowerShell (Windows):

```powershell
# create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch first (choose the right command from https://pytorch.org)
# Example CPU-only install:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the remaining dependencies
pip install -r requirements.txt

# Run baseline evaluation (adjust --data-dir and --out as needed)
python baseline_evaluation.py --data-dir data/stanford40 --out baseline_results/baseline.csv

# Run improved prompt evaluation
python action_aware_prompts.py --data-dir data/stanford40 --out action_aware_results/action_aware.csv

# Compare results
python compare_dual_fusion_classes.py --baseline baseline_results/baseline.csv --improved action_aware_results/action_aware.csv --out results/comparison.csv
```

macOS / Linux (bash):

```bash
# create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch following https://pytorch.org (choose CUDA or CPU wheel)
# Example CPU-only install:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the remaining dependencies
pip install -r requirements.txt

# Run baseline evaluation
python baseline_evaluation.py --data-dir data/stanford40 --out baseline_results/baseline.csv

# Run improved prompt evaluation
python action_aware_prompts.py --data-dir data/stanford40 --out action_aware_results/action_aware.csv

# Compare results
python compare_dual_fusion_classes.py --baseline baseline_results/baseline.csv --improved action_aware_results/action_aware.csv --out results/comparison.csv
```

Note: if your scripts expect different CLI flags, replace the example flags above with those the script prints at the top of the file.

## Dataset Preparation

Download the Stanford-40 Actions dataset and its official train/test split.

Place the dataset in the location expected by the evaluation scripts. For example:

```text
data/
└── stanford40/
    ├── JPEGImages/
    ├── ImageSplits/
    └── XMLAnnotations/
```

If your dataset is stored elsewhere, update the dataset path in the relevant evaluation script.

The dataset itself should not be uploaded to GitHub.

## Running the Evaluation

Open a terminal in the repository’s main directory.

Run the baseline evaluation:

```bash
python baseline_evaluation.py
```

Run the zero-shot evaluation, if required separately:

```bash
python zero_shot.py
```

Generate the context views:

```bash
python generate_context_views.py
```

Compare the baseline and improved results:

```bash
python compare_dual_fusion_classes.py
```

These commands assume that the scripts use configured paths or request the necessary inputs. Do not add command-line arguments unless the scripts actually support them.

## Output Files

Depending on the experiment, the generated outputs may include:

- Overall evaluation summaries
- Per-class accuracy tables
- Precision, recall and F1-score tables
- Per-image predictions
- Confusion matrices
- Baseline-versus-improved comparisons
- CSV result files

The main result directories are:

```text
baseline_results/
action_aware_results/
```

## Experimental Results

Complete this table using the final results generated by the experiments.

| Method | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Baseline OpenCLIP | To be added | To be added | To be added | To be added |
| Improved prompts | To be added | To be added | To be added | To be added |
| Difference | To be added | To be added | To be added | To be added |

## Reproducibility

For a fair comparison, all experiments should use the same:

- OpenCLIP model architecture
- Pretrained checkpoint
- Stanford-40 test split
- Image preprocessing procedure
- Batch size
- Evaluation metrics
- Random seed, where applicable
- Hardware and software environment

The exact model name, pretrained weights, Python version, PyTorch version and CUDA version should be recorded when reporting the final results.

## Research Contribution

The original OpenCLIP architecture was not developed as part of this research.

The contributions of this project are:

1. Adapting OpenCLIP for the Stanford-40 Actions dataset.
2. Implementing the Stanford-40 zero-shot baseline evaluation.
3. Designing action-aware prompts.
4. Designing body-part-aware prompts.
5. Implementing prompt ensembles.
6. Developing the evaluation and comparison pipeline.
7. Conducting overall and per-class error analysis.
8. Producing and analysing the experimental results.

## Author

- **Student:** Maisarah binti Mubarakad Arshad
- **Student ID:** 20618336
- **Programme:** Computer Science w AI
- **Lecturer:** Dr. Tomas Maul
- **Institution:** University Nottingham Malaysia

## Licence

This repository contains code derived from the OpenCLIP project. The original OpenCLIP licence and copyright notices remain applicable to the relevant source code.

Refer to the repository’s `LICENSE` file and the preserved [original OpenCLIP README](README_OPENCLIP_ORIGINAL.md) for further information.
