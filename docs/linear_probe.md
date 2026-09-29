# Frozen image-encoder linear probes

[`scripts/linear_probe.py`](../scripts/linear_probe.py) includes both sides of the
probe: dataset loading and frozen image embedding extraction, followed by fitting
and evaluating a linear classifier. It accepts single-label and multilabel tasks,
Hugging Face classification datasets, CSV files, WebDataset shards, or existing
feature caches. Both task types use this single entry point.

The image encoder stays frozen. The text encoder and prompts do not determine
probe predictions. All images use the checkpoint's deterministic evaluation
preprocessing; there is no image augmentation or encoder fine-tuning.

## Workflows

| Mode | Input | Action |
| --- | --- | --- |
| `fit` (default) | Raw dataset or `--features` manifest | Extract if necessary, select a linear head on calibration data, then evaluate on a separate evaluation source when supplied |
| `extract` | Raw dataset | Save feature NPZs and a manifest; no head fitting |
| `evaluate` | Raw dataset or feature manifest, plus `--head` | Evaluate a saved head with its existing standardization and thresholds; no refitting |

Without `--cache-features`, raw-dataset fitting keeps embeddings in memory and
writes only the head, score arrays, metrics, and selection record. Add
`--cache-features DIRECTORY` to retain reusable intermediates. Extract mode saves
them under `OUTPUT/features` by default. Use a fresh cache directory; caches are
never silently reused or overwritten. Reuse is explicit through `--features`.

“Directly from the dataset” still means extracting frozen embeddings once and
fitting on them in memory. This is full-batch logistic regression, not minibatch
encoder training. CPU memory must hold the embeddings; the fitting device also
holds float64 training/calibration features and optimizer state. `--device`
defaults to CUDA when available, otherwise CPU; add `--amp` for bfloat16 GPU extraction. NaFlex token sequences
and generative-only models are outside this fixed-resolution global-feature probe.

## Installation and a single-label run

Use this checkout and its optional evaluation dependencies:

```bash
pip install -e . datasets scikit-learn webdataset 'transformers[sentencepiece]'
```

[RESISC45](https://huggingface.co/datasets/timm/resisc45) has scalar `ClassLabel`
targets and train/validation/test splits. The task and class order are inferred
from its schema. This command runs extraction, fitting, and held-out evaluation:

```bash
PYTHONPATH=src python scripts/linear_probe.py \
  --dataset timm/resisc45 \
  --model PE-Core-L-14-336 --pretrained meta \
  --device cuda --amp --batch-size 128 --workers 4 \
  --output results/probe/resisc45
```

HF shorthand defaults to `train` for fitting, `validation` for calibration/model
selection, and `test` for evaluation. Set `--train-split`, `--calibration-split`,
and `--eval-split` to change them; `none` disables a role. `--revision COMMIT`
pins the dataset snapshot. It does not pin model weights. `--target-key` can
select a different classification field, for example Pets' `label_cat_dog`.

[Oxford-IIIT Pet](https://huggingface.co/datasets/timm/oxford-iiit-pet) has train and
test splits. Reserve calibration data from training instead of selecting the head
on the test set:

```bash
PYTHONPATH=src python scripts/linear_probe.py \
  --dataset timm/oxford-iiit-pet \
  --calibration-split none --calibration-fraction 0.1 \
  --model PE-Core-L-14-336 --pretrained meta \
  --device cuda --amp --workers 4 \
  --cache-features results/probe/pets-features \
  --output results/probe/pets
```

Holdout uses a deterministic ordering by SHA256 of `seed:group_id`, or
`seed:image_id` when no groups are supplied. Entire groups stay together. The
fraction is a fraction of groups when grouping is present, and is rounded to a
nonempty holdout. It is not label-stratified. Every class must remain represented
in training; multilabel training also requires a negative example for each class.
If that fails on a small or imbalanced dataset, provide a suitable explicit split.

## Multilabel datasets and split configurations

JSON configurations keep source columns, folds, and dataset revisions reviewable.
The same files work with fit, extract, and evaluate modes:

```bash
PYTHONPATH=src python scripts/linear_probe.py \
  --data-config scripts/probe_configs/nih_chest_xray14.json \
  --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
  --device cuda --amp --batch-size 128 --workers 4 \
  --cache-features results/probe/nih-features \
  --output results/probe/nih
```

Ready-to-run HF configurations:

| Configuration | Task | Calibration/evaluation protocol |
| --- | --- | --- |
| [Plant Pathology](../scripts/probe_configs/plant_pathology_2021.json) | Multilabel | 10% hash holdout from train; official validation for evaluation |
| [BigEarthNet RGB](../scripts/probe_configs/bigearthnet_v2_rgb.json) | Multilabel | Official validation for calibration, test for evaluation |
| [NIH ChestX-ray14](../scripts/probe_configs/nih_chest_xray14.json) | Multilabel | Train patient fold 0 for calibration, remaining folds for fitting, official test for evaluation |
| [RESISC45](../scripts/probe_configs/resisc45.json) | Single-label | Official validation for calibration, test for evaluation |
| [Oxford-IIIT Pet](../scripts/probe_configs/oxford_iiit_pet.json) | Single-label | 10% hash holdout from train; official test for evaluation |

The three multilabel configurations pin specific dataset snapshots. RESISC45/Pets
examples use `main`; pin their JSON `revision` for a published benchmark.

A source config has a `type` and `splits` mapping. Roles are `train`, `calibration`,
and `evaluation`; they need not have the same names as physical dataset splits.
For NIH, the configuration is:

```json
{
  "type": "hfids",
  "dataset": "timm/nih-chest-xray-14",
  "revision": "c1bf579641b3256b4d49924436984b82bee5834d",
  "group_key": "patient_id",
  "splits": {
    "train": {"split": "train", "exclude": {"fold": [0]}},
    "calibration": {"split": "train", "include": {"fold": [0]}},
    "evaluation": {"split": "test"}
  }
}
```

`include` and `exclude` compare metadata fields with the listed values. Filtering
happens before image decoding. Options at the root apply to all sources; per-role
objects can override them. `task` can be `single-label`, `multi-label`, or omitted
for HF schema inference. `ClassLabel` gives single-label targets;
`Sequence(ClassLabel)`/`List(ClassLabel)` gives multilabel targets. Metadata class
order is preserved even when a class is absent from calibration/evaluation.

HF input types:

- `hfds` uses a Hugging Face map-style dataset and timm's `ImageDataset` wrapper,
  with a small reader that preserves IDs and supports a pinned revision. HF prepares
  a local Arrow dataset. This is the default for `--dataset` shorthand.
- `hfids`, or shorthand `--hf-streaming`, reads finite streams. Parquet datasets
  use the existing synchronous reader and cached HF shards when available, avoiding
  Arrow background-scanner shutdown issues seen in the earlier experiments. Other
  HF formats use HF streaming. No complete Arrow materialization is required.

Optional HF fields include `config_name`, `cache_dir`, `data_files` (including
local Parquet fixtures), `image_key`, `target_key`, `id_key`, and `group_key`.
Named physical splits are supported; use filters or holdout options instead of
HF slice expressions. On NIH, `patient_id` is automatically used as the group key
unless explicitly overridden.

Each source can specify a nonnegative `limit` for smoke tests. Zero means all rows;
a positive limit takes a prefix in loader order, which can change with the worker
count. It is not representative sampling. Full benchmark runs should omit limits.

## CSV inputs

Start from the [CSV example](../scripts/probe_configs/csv_multilabel.json), replacing
its paths and class names:

```json
{
  "type": "csv",
  "task": "multi-label",
  "classnames": ["cat", "dog", "bird"],
  "image_key": "filepath",
  "target_key": "labels",
  "id_key": "image_id",
  "splits": {
    "train": "train.csv",
    "calibration": "validation.csv",
    "evaluation": "test.csv"
  }
}
```

A multilabel CSV can use index lists or class-name lists encoded as JSON:

```csv
filepath,labels,image_id
images/a.jpg,"[0, 1]",a
images/b.jpg,[],b
images/c.jpg,"[""bird""]",c
```

An optional `label_separator`, such as `"|"`, accepts `cat|dog` instead of JSON.
For already dense targets, set `"target_format": "multi-hot"` and provide a
binary vector of exactly `len(classnames)` entries. The default is an index/name
list: `[0, 1]` means two positive classes, not a dense binary vector.

For single-label input use `"task": "single-label"`; each target is an integer
class ID or one exact class name. Set `delimiter` to `"\t"` for TSV. CSV file paths
are relative to the JSON config, image paths are relative to the CSV file by
default, and an optional `image_root` is relative to the JSON config. Absolute
paths work. Without `id_key`/an `image_id` column, resolved image paths are IDs.

## WebDataset inputs

The [WebDataset example](../scripts/probe_configs/webdataset_single_label.json)
accepts local shard paths or URLs, lists of shards, and brace patterns such as
`train-{0000..0015}.tar`. Relative paths are resolved against the JSON config.
Each sample uses an image (`jpg`, `png`, `jpeg`, or `webp`) and a target:

- Single-label: `sample.jpg` plus `sample.cls`, containing an integer class ID.
- Multilabel: `sample.jpg` plus `sample.json`, for example
  `{"labels": [0, 2], "image_id": "sample", "patient_id": "p17"}`.

For the second layout, set `task` to `multi-label`, `target_key` to `json.labels`,
`id_key` to `json.image_id`, and, if applicable, `group_key` to `json.patient_id`.
Dotted keys traverse JSON sidecars; the same index/name and dense-target formats
as CSV are supported. Declare `classnames` in the config for both tasks.

If no explicit ID field is configured, tar sample keys (`__key__`) are used. IDs
must be globally unique across shards and partitions; do not reset numeric sample
keys in every shard. `--workers` partitions shards, so workers beyond the number
of shards can be idle. Sources make one finite pass, retain final partial batches,
and fail on malformed samples instead of silently skipping them.

## Saving, reusing, and validating

Extract without fitting:

```bash
PYTHONPATH=src python scripts/linear_probe.py \
  --mode extract --data-config scripts/probe_configs/bigearthnet_v2_rgb.json \
  --model PE-Core-L-14-336 --pretrained meta \
  --device cuda --amp --workers 4 --batch-size 128 \
  --cache-features results/probe/bigearth-features \
  --output results/probe/bigearth-extraction
```

Fit from cached embeddings without loading an encoder:

```bash
PYTHONPATH=src python scripts/linear_probe.py \
  --features results/probe/bigearth-features/features.json \
  --device cuda --output results/probe/bigearth-head
```

Validate a saved head from raw data or caches:

```bash
PYTHONPATH=src python scripts/linear_probe.py \
  --mode evaluate --head results/probe/bigearth-head/head.npz \
  --data-config scripts/probe_configs/bigearthnet_v2_rgb.json \
  --device cuda --amp --output results/probe/bigearth-validation

PYTHONPATH=src python scripts/linear_probe.py \
  --mode evaluate --head results/probe/bigearth-head/head.npz \
  --features results/probe/bigearth-features/features.json \
  --output results/probe/bigearth-cached-validation
```

Evaluate mode opens only the `evaluation` source/cache. New heads record the
encoder identity and preprocessing, so raw validation can restore those settings
without repeating `--model`/`--pretrained`. Validation checks class order, feature
dimensions, and available encoder provenance. It uses saved thresholds and never
fits cutoffs on evaluation labels. Image/group IDs used for fitting or calibration
are stored in the head and rejected if they occur in evaluation.

Relative paths in feature manifests are resolved against the manifest's directory.

Fitting without an evaluation role is supported: omit it in JSON or use
`--eval-split none`. The output is explicitly marked `calibration_only`. Those
metrics reuse data used to select the model and are not a held-out estimate.

## Objectives, selection, and outputs

Training-only feature means and standard deviations are saved with the head.
The bias is unpenalized. Full-batch float64 L-BFGS uses `--max-iter` (default 1,500)
and an L2 grid controlled by `--l2` (default `1 .1 .01 .001 .0001`). Ties retain the
first candidate in the supplied order.

| Task | Objective | Calibration selection | Evaluation metrics |
| --- | --- | --- | --- |
| Single-label | Mean softmax cross-entropy + `lambda * sum(W²) / 2` | Top-1 accuracy | Top-1, top-k (`k=min(5,C)`), balanced accuracy, per-class support/accuracy |
| Multilabel | Mean binary cross-entropy + `lambda * sum(W²) / (2*C)` | Macro AP | Macro/micro AP, macro/per-class AUROC, per-class AP, calibrated macro/micro F1, precision/recall, exact match |

Multilabel thresholds maximize each class's calibration F1; degenerate calibration
classes fall back to the global cutoff. Fixed probability-0.5 results are also
reported. Single-label predictions use argmax, not independent binary heads.
No class weighting or nonlinear classifier is used.

All training/calibration image IDs and group IDs are checked for overlap.
Evaluation is loaded after selection and the head have been saved; its labels do
not choose regularization, standardization, or thresholds. All split vocabularies
must agree. Automatic IDs establish identity within a source, not a content-based
duplicate audit across separately copied images; supply meaningful IDs/group keys.

Outputs are `head.npz`, `selection.json`, `metrics.json`, `calibration.npz`, and
`evaluation.npz` when that role is present. Score NPZs contain logits, targets,
class names, and IDs. Optional feature NPZs also contain normalized float32 image
embeddings. The cache manifest records the source config, task, class order,
encoder/preprocessing, and counts. L-BFGS budget flags and gradient residuals are
preserved in the selection record.

Apply a saved head as
`((features - feature_mean) / feature_std) @ weight + bias`. For multilabel
predictions compare each logit to `thresholds`; for single-label use argmax.
