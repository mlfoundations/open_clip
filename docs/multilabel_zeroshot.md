# Zero-shot multilabel evaluation

CLIP and SigLIP can score several labels for the same image. Reliable label
decisions require more than changing a softmax to a sigmoid: the visual domain,
prompt wording, class prevalence, and decision threshold all matter. This script
measures frozen-model ranking separately from thresholded decisions.

[`scripts/multilabel_zeroshot.py`](../scripts/multilabel_zeroshot.py) supports
fixed-resolution OpenCLIP image/text models, including CLIP, SigLIP, SigLIP2, and
compatible `hf-hub:` models. It does not train model weights or change the existing
ImageNet evaluation. NaFlex and generative models are outside this script's scope.

## Setup and a quick run

Use an environment with this checkout and the optional evaluation dependencies:

```bash
pip install -e . datasets scikit-learn 'transformers[sentencepiece]'

PYTHONPATH=src python scripts/multilabel_zeroshot.py \
  --dataset plant-pathology-2021 \
  --model ViT-B-16-SigLIP2-256 --pretrained webli \
  --device cuda --amp --batch-size 32 --limit 128 \
  --output results/multilabel/plant-siglip2-smoke
```

`PYTHONPATH=src` selects this OpenCLIP checkout over any installed `open_clip`.
For CPU execution, use `--device cpu` and omit `--amp`.

Data is streamed from Hugging Face using synchronous PyArrow Parquet batches and
Hugging Face Datasets feature decoding. This avoids background scanner reads
lingering after a limited iterator closes. The entire dataset need not be downloaded
or decoded for a small run. Parquet reads can still fetch substantial row groups.
`--limit 0` (the default) evaluates the full official evaluation split. A positive
limit takes a **prefix**, not a representative random sample. `--shuffle-buffer N`
provides an optional seeded, bounded streaming shuffle; it is not uniform random
sampling across a large split. Use full splits for reported benchmarks, especially
for geographically or patient-ordered data. Rare labels may be absent in small runs.

## Dataset protocols

| Dataset | Labels | Evaluation | Optional threshold fitting |
| --- | ---: | --- | --- |
| [Plant Pathology 2021](https://huggingface.co/datasets/timm/plant-pathology-2021) | 6 | `validation` (1,864 images) | `train` |
| [BigEarthNet v2 RGB](https://huggingface.co/datasets/timm/bigearthnet-v2-rgb) | 19 | `test` (119,825 images) | `validation` |
| [NIH ChestX-ray14](https://huggingface.co/datasets/timm/nih-chest-xray-14) | 14 | `test` (25,596 images) | `train`, patient-grouped `fold == 0` |

Class order comes from the dataset's `labels` feature, rather than labels observed
in a sample. The script uses the model's evaluation preprocessing and converts
images to RGB, including NIH grayscale images.

- Plant's `healthy` is an explicit class; `complex` is a dataset-specific disease
  category. Its prompt is only an initial interpretation, not a verified visual
  definition. Fine disease distinctions are a domain-transfer challenge.
- BigEarthNet uses the official geographic splits. Do not randomly re-split it.
  This RGB subset has no NIR/SWIR channels, unlike multispectral benchmarks.
- NIH has noisy labels extracted from reports. Empty `labels` means none of the
  14 findings are annotated; it does **not** assert that the image is normal.
  Empty targets are kept. Its calibration fold groups images by patient, and
  the script additionally rejects overlapping evaluation/calibration image or
  patient IDs. Benchmark performance is not clinical validation.

These details are specified in the linked dataset cards. For ranking, report mAP
and per-class AP for all three; NIH also conventionally reports mean per-finding
AUROC. Do not compare an aggregate from a small subset against full-split scores.

## Scores and thresholds

For normalized image embedding `v` and normalized, prompt-ensembled class
prototype `t[c]`, the script supports:

| `--score` | Stored score | Interpretation |
| --- | --- | --- |
| `cosine` (default except paired strategies) | `v @ t[c]` | Independent ranking score; no universal cutoff |
| `logit` | `exp(logit_scale) * (v @ t[c]) + logit_bias` | Uses learned bias when present, including SigLIP |
| `paired` | `exp(logit_scale) * (v @ (t_positive[c] - t_negative[c]))` | Experimental positive versus negative prompt margin |

There is no softmax across candidate classes, forced top-k, or forced positive
prediction. A score for one class does not depend on which other classes are
included. Keeping logits/margins avoids the numerical saturation of a sigmoid
before AP/AUROC calculation. Cosine and native logits have the same per-class
ranking for the same embeddings and positive scalar scale.

[SigLIP's loss](https://arxiv.org/abs/2303.15343) is a pairwise image/text matching
objective. Applying its sigmoid gives an image/text match score; this does not
establish calibrated disease/land-cover presence probabilities on a new dataset.
Prompt ensembling also changes the input to that scoring rule. For CLIP, adding a
sigmoid does not turn its contrastive scores into learned binary probabilities.
[Zero-shot CLIP miscalibration](https://arxiv.org/abs/2303.12748) has been measured
across datasets and prompts.

Ranking-only evaluation uses no target labels for adaptation. To inspect SigLIP's
unfitted 0.5 sigmoid decision rule, use `--score logit --threshold 0`. To inspect
the positive/negative prompt baseline, use `--score paired --threshold 0`.
Negative wording can be poorly understood by CLIP; a paired score is an experiment,
not a guaranteed improvement or calibrated probability.

To fit cutoffs, add `--calibrate global` (maximize calibration micro-F1) or
`--calibrate per-class` (maximize each class's calibration F1). Evaluation labels
are never passed to the threshold fitter. This is **a frozen zero-shot model with
supervised threshold fitting**, not a fully label-free decision system. Classes
with no positives or no negatives in calibration fall back to the global cutoff
and are listed in the report. Classes with only a few positives can still overfit.
`--calibration-limit` bounds this separate pass; its default is the full split/fold.

For deployment, choose thresholds for the desired precision/recall or error cost,
validate them on independent data, and measure performance by class and subgroup.
F1-optimal cutoffs are only one benchmark operating point. Estimate uncertainty
with patient/geographic grouping where appropriate; this script does not compute
confidence intervals or probability calibration metrics.

## Comparing models

Full Plant Pathology evaluation with SigLIP2 and separate threshold fitting:

```bash
PYTHONPATH=src python scripts/multilabel_zeroshot.py \
  --dataset plant-pathology-2021 \
  --model ViT-L-16-SigLIP2-256 --pretrained webli \
  --device cuda --amp --score logit --threshold 0 \
  --calibrate per-class \
  --output results/multilabel/plant-siglip2
```

A CLIP comparison using the same prompts and split:

```bash
PYTHONPATH=src python scripts/multilabel_zeroshot.py \
  --dataset plant-pathology-2021 \
  --model ViT-B-16 --pretrained laion2b_s34b_b88k \
  --device cuda --amp --calibrate per-class \
  --output results/multilabel/plant-clip
```

The different architectures, scales, and pretraining data make this a practical
checkpoint comparison, not a controlled comparison of CLIP versus SigLIP losses.
Use `--dataset bigearthnet-v2-rgb` or `--dataset nih-chest-xray-14` for the other
presets. Full NIH/BigEarthNet evaluation takes substantially more data and compute;
start with a limited run to check your setup.

For full runs, `--workers 4 --batch-size 128` parallelizes image decoding and
preprocessing. Workers use separate processes; their count is bounded by the
number of Parquet shards. `--download-data` caches complete shards for reuse
across checkpoints. Already cached shards are used even without this flag.
NIH's calibration fold is filtered before image decoding. Leave `--workers 0`
when comparing an exact ordered prefix with an earlier run; worker sharding can
change which examples a limited run visits. Full runs keep all examples.

PE-Core provides another CLIP-family comparison:

```bash
PYTHONPATH=src python scripts/multilabel_zeroshot.py \
  --dataset bigearthnet-v2-rgb \
  --model PE-Core-L-14-336 --pretrained meta \
  --device cuda --amp --workers 4 --batch-size 128 --download-data \
  --calibrate per-class --output results/multilabel/bigearth-pecore-full
```

Domain-specific models are useful additional baselines, e.g.
[RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) for satellite imagery and
[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
for biomedical imagery. A biomedical checkpoint is not necessarily specialized
for chest radiographs. Check pretraining overlap before calling any model
zero-shot on a benchmark. A compatible Hub model can be supplied directly:

```bash
PYTHONPATH=src python scripts/multilabel_zeroshot.py \
  --dataset nih-chest-xray-14 \
  --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
  --device cuda --limit 128 \
  --output results/multilabel/nih-biomedclip-smoke
```

## Artifacts and custom prompts

Each run writes:

- `metrics.json`: configuration, versions, class order, split/fold, model scale
  and bias, macro/micro AP, macro AUROC, per-class support/prevalence/AP/AUROC,
  and optional fixed/fitted threshold metrics.
- `prompts.json`: exact positive/negative prompts used.
- `evaluation.npz`: scores, multi-hot targets, image IDs, patient IDs, and class
  names, loadable with `np.load(..., allow_pickle=False)`.
- `calibration.npz`: equivalent calibration data when fitting thresholds.

With `--save-features`, NPZ files also contain normalized float32 image `features`.
These allow prompt comparisons without repeating image inference. Reuse them only
with the same checkpoint and preprocessing, and select prompts on development data.

Use distinct output directories for comparisons. Pin `--revision` to a dataset
commit SHA and preserve checkpoint identity for reproducible results. The script
records the revision argument; `main` is a moving reference. It does not establish
absence of pretraining contamination.

AP is `null` for classes with no positives; AUROC is `null` when positives or
negatives are absent. Macro averages exclude those undefined values and explicitly
report their denominators (`ap_classes`, `auroc_classes`). Macro F1 includes all
classes with zero-division set to zero. All metric values are fractions, not percentages.
Inspect class coverage, especially when comparing limited runs.

To change prompts, copy a run's `prompts.json`, edit its class-specific lists, and
pass `--prompts my_prompts.json`. Keys must match the exact dataset class names.
Each class needs a nonempty `positive` list; `paired` scoring also needs `negative`.
Prototypes average normalized text embeddings and then normalize again. The
baseline was untuned; the additional built-in candidates were compared on labeled
development data. If you select prompts using labeled development data, disclose
that supervision and reserve the evaluation split for the final comparison.

## Prompt strategies

Use `--prompt-strategy` to select any of the six tested families. Every strategy
supports all three dataset presets, preserving the exact experimental phrases.

| Option | Positive texts per label | Default scoring |
| --- | ---: | --- |
| `baseline` | 2 original domain phrases | `cosine` |
| `baseline-paired` | 2 original domain phrases | `paired` |
| `domain-templates` | 6 domain phrases around the label | `cosine` |
| `visual-descriptions` | 3 class-specific descriptions/synonyms | `cosine` |
| `baseline-plus-descriptions` | 2 original + 3 descriptive phrases | `cosine` |
| `descriptions-paired` | 2 original + 3 descriptive phrases | `paired` |

Paired strategies use the original two negative phrases per label. In particular,
`descriptions-paired` uses the **five-text mixture** for positives, matching the
experiment; it does not generate new negative descriptions. All five texts are
averaged with equal weight before the final normalization.

```bash
PYTHONPATH=src python scripts/multilabel_zeroshot.py \
  --dataset nih-chest-xray-14 \
  --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
  --prompt-strategy descriptions-paired \
  --device cuda --amp --workers 4 --batch-size 128 \
  --calibrate per-class --output results/multilabel/nih-biomedclip-paired
```

`baseline` remains the default. Existing `--score paired` commands still work.
An explicit `--score` can change a positive-only strategy's scoring, for example
`--prompt-strategy visual-descriptions --score paired`. Named paired strategies
require paired scoring and reject conflicting overrides. `--prompts custom.json`
and `--prompt-strategy` are mutually exclusive; custom JSON defaults to cosine,
so pass `--score paired` explicitly when desired. Each run records the resolved
strategy and score in `metrics.json` and writes the exact strings to `prompts.json`.
Selecting a named strategy does not automatically select it based on dataset scores.

CLIP-style models embed text descriptions; they do not execute instructions to
reason through an image or output a list of labels. Try short phrases describing
the visible concept, and score each label independently.

- **Domain templates:** average several phrasings such as “a satellite image
  containing {label}” and “an aerial view showing {label}.” For multilabel scenes,
  “containing” need not describe the entire image as one class.
- **Visual descriptions and synonyms:** augment a class name with a few separate
  concrete descriptions. For example, `permanent_crops` can include “orchards and
  vineyards in regular rows”; leaf rust can include “an apple leaf with bright
  orange spots.” Descriptions are candidate cues, not exact label definitions.
  This is inspired by [classification by description](https://arxiv.org/abs/2210.07183),
  not a reproduction of that paper's complete method.
- **Positive/negative pairs:** `--score paired` uses each class's positive versus
  negative embedding margin. This is rank-equivalent to a two-way softmax for
  each label, as used by [CheXzero](https://www.nature.com/articles/s41551-022-00936-9).
  It does not apply a softmax across classes. CheXzero also trained on chest
  X-rays and reports; its results cannot be attributed to prompting alone.
  Negation is unreliable in many VLMs, so compare against positive-only scoring
  rather than assuming “no {label}” is useful
  ([NegBench](https://arxiv.org/abs/2501.09425)).
- **Co-occurrence prompts:** [DualPrompt](https://github.com/xiemk/DualPrompt)
  combines class-specific prompts with label co-occurrence prompts. This is a
  further experiment, not implemented here. Its data-derived priors consume
  training labels even though the encoder remains frozen.

Keep individual descriptions within the model's context length: the local
`PE-Core-L-14-336` config has only 32 tokens including special tokens. Encode
several short descriptions separately rather than concatenating a long report.
Prompt averaging can also dilute a useful cue, so measure it against the original
baseline on the same images. Prompt selection with labeled calibration data is
development supervision, even when model weights remain frozen. Thresholds must
be recalibrated after changing prompts, and held-out evaluation should follow
only after the prompt strategy is fixed.

## Supervised linear probes

[`scripts/linear_probe.py`](../scripts/linear_probe.py) supports raw Hugging Face,
CSV, and WebDataset inputs as well as cached image embeddings, with optional
feature caching and saved-head validation. It handles both single-label softmax
and multilabel logistic probes. See the [linear-probe guide](linear_probe.md) for
dataset configurations and complete workflows.

For multilabel tasks it fits independent binary logistic classifiers.
The image encoder stays frozen; the text encoder and
prompts do not determine probe predictions. This is supervised adaptation using
training labels, not zero-shot classification or full encoder fine-tuning.

Provide a JSON feature manifest with `paths.train`, `paths.calibration`, and
`paths.evaluation` pointing to NPZ caches containing `features`, `targets`,
`image_ids`, `group_ids`, `patient_ids`, and `classnames`. Features can be saved with the
zero-shot evaluator's `--save-features` option. Each cache must use the same
checkpoint and preprocessing. Empty patient IDs are permitted for datasets
without patient metadata.

```bash
PYTHONPATH=src:scripts python scripts/linear_probe.py \
  --features results/my_features.json \
  --output results/my_linear_probe --device cuda:0
```

The probe standardizes each feature dimension using training-only statistics,
then optimizes binary cross-entropy plus L2-regularized weights with an unpenalized
bias. It uses float64 L-BFGS optimization; no class balancing, data augmentation,
nonlinear head, or class softmax is used. Five regularization values are compared
by calibration macro AP. Per-class decision thresholds maximize calibration F1.
Only the selected head is evaluated on the held-out split; fixed probability-0.5
decisions are also reported. Train/calibration/evaluation image IDs and patient IDs
must be pairwise disjoint.

Outputs include `selection.json` (written before loading evaluation data),
`metrics.json`, evaluation/calibration score NPZs, and `head.npz`. Apply a saved head
as `((features - feature_mean) / feature_std) @ weight + bias`, then compare each
logit with its saved `thresholds` value. When input caches also include zero-shot
`scores`, the report compares both methods on identical images and calibration data.
