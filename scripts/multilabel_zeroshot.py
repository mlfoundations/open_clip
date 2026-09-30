"""Frozen CLIP/SigLIP multilabel evaluation; see docs/multilabel_zeroshot.md.

Requires datasets, scikit-learn and the dependencies of the chosen OpenCLIP model.
Scores are independent across labels. No class softmax or forced top-k is used.
"""

import argparse
import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

PRESETS = {
    'plant-pathology-2021': {'evaluation': 'validation', 'calibration': 'train'},
    'bigearthnet-v2-rgb': {'evaluation': 'test', 'calibration': 'validation'},
    'nih-chest-xray-14': {'evaluation': 'test', 'calibration': 'train', 'fold': 0},
}


# These six presets reproduce the development prompt-comparison sweep.
# Paired presets include the scoring rule as well as the prompt ensemble.
PROMPT_STRATEGIES = {
    'baseline': 'cosine',
    'baseline-paired': 'paired',
    'domain-templates': 'cosine',
    'visual-descriptions': 'cosine',
    'baseline-plus-descriptions': 'cosine',
    'descriptions-paired': 'paired',
}

# Candidate visual cues, not expert-validated definitions or diagnostic criteria.
VISUAL_DESCRIPTIONS = {
    'plant-pathology-2021': {
        'complex': [
            'an apple leaf with several types of disease damage',
            'an apple leaf with a mixture of spots and discoloration',
            'an apple leaf with complex disease symptoms',
        ],
        'frog_eye_leaf_spot': [
            'an apple leaf with frog eye leaf spot',
            'an apple leaf with round tan spots and dark purple edges',
            'an apple leaf with small circular brown lesions',
        ],
        'healthy': [
            'a healthy green apple leaf',
            'an apple leaf with a smooth green surface',
            'an apple leaf with clean unblemished foliage',
        ],
        'powdery_mildew': [
            'an apple leaf with powdery mildew',
            'an apple leaf covered in white powdery fungal growth',
            'an apple leaf with dusty white patches',
        ],
        'rust': [
            'an apple leaf with rust disease',
            'an apple leaf with bright orange spots',
            'an apple leaf with yellow orange circular lesions',
        ],
        'scab': [
            'an apple leaf with apple scab',
            'an apple leaf with olive brown fungal spots',
            'an apple leaf with dark irregular scabby lesions',
        ],
    },
    'bigearthnet-v2-rgb': {
        'agriculture_with_natural_vegetation': [
            'farmland interspersed with natural vegetation',
            'agricultural fields mixed with patches of trees and shrubs',
            'cropland and natural vegetation',
        ],
        'agro_forestry_areas': [
            'agroforestry land',
            'agricultural land with scattered trees',
            'trees growing among crops or grazing fields',
        ],
        'arable_land': [
            'arable cropland',
            'large rectangular cultivated fields',
            'ploughed fields and annual crops',
        ],
        'beaches_dunes_sands': [
            'sandy beaches and dunes',
            'pale sand along a coastline',
            'bare sandy land',
        ],
        'broad_leaved_forest': [
            'broadleaf forest',
            'dense deciduous woodland',
            'forest with broadleaf tree canopies',
        ],
        'coastal_wetlands': [
            'coastal wetlands',
            'salt marshes and tidal flats',
            'wetland vegetation along the sea',
        ],
        'complex_cultivation_patterns': [
            'a patchwork of small cultivated plots',
            'a mosaic of mixed agricultural fields',
            'small irregular fields with different crops',
        ],
        'coniferous_forest': [
            'coniferous forest',
            'dense evergreen needleleaf woodland',
            'pine and spruce forest',
        ],
        'industrial_commercial_units': [
            'industrial and commercial buildings',
            'large warehouses with paved yards',
            'factories and commercial complexes',
        ],
        'inland_waters': [
            'inland water bodies',
            'lakes and rivers',
            'reservoirs and freshwater channels',
        ],
        'inland_wetlands': [
            'inland wetlands',
            'marshes and waterlogged vegetation',
            'peat bogs and freshwater marshes',
        ],
        'marine_waters': [
            'marine water',
            'open sea and ocean',
            'coastal seawater',
        ],
        'mixed_forest': [
            'mixed forest',
            'woodland with both coniferous and broadleaf trees',
            'a mixture of evergreen and deciduous tree canopies',
        ],
        'moors_heathland_sclerophyllous_vegetation': [
            'moorland and heathland',
            'low shrubs and heath vegetation',
            'dry scrub and sclerophyllous vegetation',
        ],
        'natural_grassland_sparse_vegetation': [
            'natural grassland and sparse vegetation',
            'open grassland with scattered vegetation',
            'sparsely vegetated land',
        ],
        'pastures': [
            'pasture land',
            'managed grassy grazing fields',
            'green meadows used for livestock',
        ],
        'permanent_crops': [
            'permanent crop plantations',
            'orchards and vineyards in regular rows',
            'fruit tree plantations and olive groves',
        ],
        'transitional_woodland_shrub': [
            'transitional woodland and shrubs',
            'regenerating woodland with shrubs and young trees',
            'patchy scrub and scattered trees',
        ],
        'urban_fabric': [
            'urban residential areas',
            'dense buildings and streets',
            'towns and housing neighborhoods',
        ],
    },
    'nih-chest-xray-14': {
        'Atelectasis': [
            'atelectasis',
            'atelectasis with lung volume loss',
            'linear opacity from partial lung collapse',
        ],
        'Cardiomegaly': [
            'cardiomegaly',
            'an enlarged cardiac silhouette',
            'an abnormally enlarged heart',
        ],
        'Consolidation': [
            'lung consolidation',
            'dense airspace opacity',
            'airspace consolidation with air bronchograms',
        ],
        'Edema': [
            'pulmonary edema',
            'bilateral hazy lung opacities',
            'interstitial fluid and perihilar opacities',
        ],
        'Effusion': [
            'pleural effusion',
            'pleural fluid blunting a costophrenic angle',
            'fluid at the lung base with a meniscus',
        ],
        'Emphysema': [
            'emphysema',
            'hyperinflated lungs with flattened diaphragms',
            'hyperlucent lungs with reduced markings',
        ],
        'Fibrosis': [
            'pulmonary fibrosis',
            'coarse linear scarring in the lungs',
            'reticular lung opacities and scarring',
        ],
        'Hernia': [
            'a hiatal hernia',
            'a retrocardiac air fluid level',
            'a stomach herniation above the diaphragm',
        ],
        'Infiltration': [
            'pulmonary infiltrates',
            'patchy ill defined lung opacities',
            'diffuse infiltrative lung shadows',
        ],
        'Mass': [
            'a lung mass',
            'a large focal rounded lung opacity',
            'a bulky pulmonary mass lesion',
        ],
        'Nodule': [
            'a pulmonary nodule',
            'a small rounded lung opacity',
            'a small focal nodular lung lesion',
        ],
        'Pleural_Thickening': [
            'pleural thickening',
            'thickened pleura along the chest wall',
            'apical pleural scarring',
        ],
        'Pneumonia': [
            'pneumonia',
            'patchy lung consolidation from pneumonia',
            'a focal airspace opacity from lung infection',
        ],
        'Pneumothorax': [
            'a pneumothorax',
            'a pleural line with absent peripheral lung markings',
            'air in the pleural space beside a collapsed lung',
        ],
    },
}

DOMAIN_TEMPLATES = {
    'plant-pathology-2021': (
        'a photo of {}.',
        'a close-up photo of {}.',
        'a detailed image of {}.',
        'a photograph showing {}.',
        'a cropped photo of {}.',
        'an image showing {}.',
    ),
    'bigearthnet-v2-rgb': (
        'an RGB satellite image containing {}.',
        'a satellite view showing some {}.',
        'a remote sensing image containing {}.',
        'an overhead view of an area with {}.',
        'an aerial photograph showing {}.',
        'a Sentinel-2 image of {}.',
    ),
    'nih-chest-xray-14': (
        'a chest x-ray showing {}.',
        'a chest radiograph showing {}.',
        'a frontal chest x-ray with {}.',
        'a radiograph of the chest with {}.',
        'there is {} in this chest x-ray.',
        'a chest x-ray with evidence of {}.',
    ),
}

DESCRIPTION_TEMPLATES = {
    'plant-pathology-2021': 'a close-up photo of {}.',
    'bigearthnet-v2-rgb': 'a satellite image showing {}.',
    'nih-chest-xray-14': 'a chest x-ray showing {}.',
}


def make_prompts(dataset, classnames, strategy='baseline'):
    """Build a named ensemble; score defaults are resolved separately by the CLI.

    Baseline prompts were untuned. The other candidates were compared on labeled
    development data. Each class retains baseline negatives for paired scoring.
    """
    if dataset not in PRESETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    if strategy not in PROMPT_STRATEGIES:
        raise ValueError(f'Unknown prompt strategy: {strategy}')
    baseline = _baseline_prompts(dataset, classnames)
    if strategy in ('baseline', 'baseline-paired'):
        return baseline

    prompts = {}
    for label in classnames:
        if strategy == 'domain-templates':
            subject = (
                baseline[label]['positive'][0].removeprefix('a photo of ').removesuffix('.')
                if dataset == 'plant-pathology-2021'
                else label.replace('_', ' ').lower()
            )
            positive = [template.format(subject) for template in DOMAIN_TEMPLATES[dataset]]
        else:
            positive = [
                DESCRIPTION_TEMPLATES[dataset].format(description)
                for description in VISUAL_DESCRIPTIONS[dataset][label]
            ]
            if strategy in ('baseline-plus-descriptions', 'descriptions-paired'):
                positive = baseline[label]['positive'] + positive
        prompts[label] = {'positive': positive, 'negative': baseline[label]['negative']}
    return prompts


def _baseline_prompts(dataset, classnames):
    """Original two-template baseline, retained verbatim for reproducibility."""
    plant = {
        'complex': ('an apple leaf with complex disease symptoms', 'a healthy apple leaf'),
        'frog_eye_leaf_spot': ('an apple leaf with frog eye leaf spot', 'an apple leaf without frog eye leaf spot'),
        'healthy': ('a healthy apple leaf', 'a diseased apple leaf'),
        'powdery_mildew': ('an apple leaf with powdery mildew', 'an apple leaf without powdery mildew'),
        'rust': ('an apple leaf with rust disease', 'an apple leaf without rust disease'),
        'scab': ('an apple leaf with apple scab', 'an apple leaf without apple scab'),
    }
    prompts = {}
    for label in classnames:
        name = label.replace('_', ' ').lower()
        if dataset == 'plant-pathology-2021':
            positive, negative = plant[label]
            templates = ('a photo of {}.', 'a close-up photo of {}.')
        elif dataset == 'bigearthnet-v2-rgb':
            positive, negative = name, 'land without ' + name
            templates = ('a satellite image of {}.', 'an aerial image of {}.')
        else:
            positive, negative = name, 'no ' + name
            templates = ('a chest x-ray showing {}.', 'a chest radiograph showing {}.')
        prompts[label] = {
            'positive': [t.format(positive) for t in templates],
            'negative': [t.format(negative) for t in templates],
        }
    return prompts


def validate_prompts(prompts, classnames, paired):
    if set(prompts) != set(classnames):
        raise ValueError('Prompt JSON keys must exactly match the dataset ClassLabel names.')
    for label in classnames:
        for kind in ('positive', 'negative') if paired else ('positive',):
            texts = prompts[label].get(kind)
            if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
                raise ValueError(f'{label}: {kind} must be a nonempty list of prompt strings.')


@torch.inference_mode()
def encode_prompts(model, tokenizer, prompts, classnames, kind, device):
    # Normalize each text, average templates, then normalize the prototype, as in
    # OpenCLIP's zero-shot classifier. This is an ensemble heuristic, not calibration.
    prototypes = []
    for label in classnames:
        tokens = tokenizer(prompts[label][kind]).to(device)
        features = model.encode_text(tokens, normalize=True).float()
        prototype = torch.nn.functional.normalize(features.mean(dim=0), dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes, dim=1)


def score_features(features, positive, mode, logit_scale, logit_bias=None, negative=None):
    cosine = features.float() @ positive.float()
    if mode == 'cosine':
        return cosine
    if mode == 'paired':
        if negative is None:
            raise ValueError('Paired scoring requires negative prototypes.')
        # Positive versus negative two-way softmax has sigmoid(scale * (pos-neg)).
        # Keep the margin to avoid sigmoid saturation and preserve ranking ties.
        return logit_scale * (cosine - features.float() @ negative.float())
    if mode == 'logit':
        return logit_scale * cosine + (0.0 if logit_bias is None else logit_bias)
    raise ValueError(f'Unknown score mode: {mode}')


def multihot(labels, num_classes):
    target = np.zeros(num_classes, dtype=np.uint8)
    for index in labels:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < num_classes:
            raise ValueError(f'Invalid label index {index} for {num_classes} classes.')
        target[index] = 1
    return target


def validate_arrays(targets, scores):
    targets, scores = np.asarray(targets), np.asarray(scores)
    if targets.ndim != 2 or targets.shape != scores.shape or 0 in targets.shape:
        raise ValueError('Targets and scores must be nonempty matching [images, classes] arrays.')
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Targets must be binary and scores must be finite.')
    return targets, scores


def ranking_metrics(targets, scores, classnames):
    targets, scores = validate_arrays(targets, scores)
    per_class, aps, aucs = {}, [], []
    for j, name in enumerate(classnames):
        y, s = targets[:, j], scores[:, j]
        count = int(y.sum())
        ap = float(average_precision_score(y, s)) if count else None
        auc = float(roc_auc_score(y, s)) if 0 < count < len(y) else None
        if ap is not None:
            aps.append(ap)
        if auc is not None:
            aucs.append(auc)
        per_class[name] = {'positives': count, 'prevalence': count / len(y), 'ap': ap, 'auroc': auc}
    return {
        'num_images': len(targets),
        'num_classes': len(classnames),
        'mean_labels': float(targets.sum(axis=1).mean()),
        'macro_ap': float(np.mean(aps)) if aps else None,
        'micro_ap': float(average_precision_score(targets.ravel(), scores.ravel())) if targets.any() else None,
        'macro_auroc': float(np.mean(aucs)) if aucs else None,
        'ap_classes': len(aps),
        'auroc_classes': len(aucs),
        'per_class': per_class,
    }


def decision_metrics(targets, scores, thresholds):
    targets, scores = validate_arrays(targets, scores)
    predictions = scores >= thresholds
    truth = targets.astype(bool)
    tp = (predictions & truth).sum(axis=0)
    fp = (predictions & ~truth).sum(axis=0)
    fn = (~predictions & truth).sum(axis=0)

    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    per_class_f1 = divide(2 * tp, 2 * tp + fp + fn)
    return {
        'micro_f1': float(divide(2 * tp.sum(), (2 * tp + fp + fn).sum())),
        'macro_f1': float(per_class_f1.mean()),
        'micro_precision': float(divide(tp.sum(), (tp + fp).sum())),
        'micro_recall': float(divide(tp.sum(), (tp + fn).sum())),
        'exact_match': float((predictions == truth).all(axis=1).mean()),
        'mean_predicted_labels': float(predictions.sum(axis=1).mean()),
        'per_class_f1': per_class_f1.tolist(),
    }


def best_f1_threshold(targets, scores):
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    # Prefer the more conservative threshold when F1 ties. Tied scores are never split.
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])


def fit_thresholds(targets, scores, method):
    targets, scores = validate_arrays(targets, scores)
    if method not in ('global', 'per-class'):
        raise ValueError('Threshold method must be global or per-class.')
    if not targets.any() or targets.all():
        raise ValueError('Threshold fitting requires both positive and negative calibration labels.')
    global_threshold = best_f1_threshold(targets.ravel(), scores.ravel())
    thresholds = np.full(targets.shape[1], global_threshold)
    fallback = []
    if method == 'per-class':
        for j in range(targets.shape[1]):
            count = targets[:, j].sum()
            if 0 < count < len(targets):
                thresholds[j] = best_f1_threshold(targets[:, j], scores[:, j])
            else:
                fallback.append(j)
    return thresholds, fallback


def iter_parquet_rows(files, columns, fold=None, download=False):
    import pyarrow.compute as pc
    from huggingface_hub import HfFileSystem, hf_hub_download, try_to_load_from_cache
    from pyarrow import parquet

    filesystem = HfFileSystem()
    for path in files:
        cached = path if Path(path).is_file() else None
        if cached is None:
            resolved = filesystem.resolve_path(path)
            cache_kwargs = {
                'repo_id': resolved.repo_id,
                'filename': resolved.path_in_repo,
                'revision': resolved.revision,
                'repo_type': 'dataset',
            }
            cached = hf_hub_download(**cache_kwargs) if download else try_to_load_from_cache(**cache_kwargs)
        with (
            open(cached, 'rb') if isinstance(cached, str) else filesystem.open(path, 'rb') as source,
            parquet.ParquetFile(source) as reader,
        ):
            # Avoid Arrow scanner background tasks holding Python HF file
            # handles after .take() closes a partial stream (can hang at exit).
            for batch in reader.iter_batches(batch_size=64, columns=columns, use_threads=False):
                if fold is not None:
                    # Filter before feature decoding so unused NIH images never
                    # incur JPEG decoding/resizing costs.
                    batch = batch.filter(pc.equal(batch.column('fold'), fold))
                yield from batch.to_pylist()


def load_split(args, calibration=False):
    from datasets import Features, IterableDataset, get_dataset_config_info
    from huggingface_hub import HfFileSystem

    preset = PRESETS[args.dataset]
    split = preset['calibration' if calibration else 'evaluation']
    fold = preset.get('fold') if calibration else None
    info = get_dataset_config_info('timm/' + args.dataset, revision=args.revision, cache_dir=args.cache_dir)
    # Use the schema, never the observed subset of labels, to keep absent classes.
    label_feature = info.features['labels'].feature
    classnames = list(label_feature.names)
    columns = ['image', 'labels', 'image_id']
    columns += [name for name in ('patient_id', 'fold') if name in info.features]
    files = sorted(HfFileSystem().glob(f'datasets/timm/{args.dataset}@{args.revision}/data/{split}-*.parquet'))
    if not files:
        raise ValueError(f'No Parquet files found for {args.dataset} split {split}.')
    ds = IterableDataset.from_generator(
        iter_parquet_rows,
        gen_kwargs={
            'files': files,
            'columns': tuple(columns),
            'fold': fold,
            'download': getattr(args, 'download_data', False),
        },
        features=Features({name: info.features[name] for name in columns}),
    )
    if args.shuffle_buffer:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    limit = args.calibration_limit if calibration else args.limit
    if limit:
        ds = ds.take(limit)
    return ds, classnames, {'split': split, 'fold': fold, 'limit': limit, 'shuffle_buffer': args.shuffle_buffer}


def collate_images(rows, preprocess, num_classes):
    return {
        'images': torch.stack([preprocess(row['image'].convert('RGB')) for row in rows]),
        'targets': torch.from_numpy(np.stack([multihot(row['labels'], num_classes) for row in rows])),
        'image_ids': [str(row['image_id']) for row in rows],
        'patient_ids': [str(row['patient_id']) if 'patient_id' in row else '' for row in rows],
    }


@torch.inference_mode()
def collect_scores(model, preprocess, ds, classnames, positive, negative, args):
    all_scores, all_targets, ids, patients = [], [], [], []
    all_features = [] if getattr(args, 'save_features', False) else None
    scale = float(model.logit_scale.detach().float().exp().item())
    bias = getattr(model, 'logit_bias', None)
    bias = float(bias.detach().float().item()) if bias is not None else None

    workers = getattr(args, 'workers', 0)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=workers,
        collate_fn=partial(collate_images, preprocess=preprocess, num_classes=len(classnames)),
        pin_memory=torch.device(args.device).type == 'cuda',
        multiprocessing_context='spawn' if workers else None,
    )
    with tqdm(desc='Scoring', unit='image', mininterval=5) as progress:
        for batch in loader:
            images = batch['images'].to(args.device, non_blocking=True)
            autocast = torch.autocast('cuda', dtype=torch.bfloat16) if args.amp else nullcontext()
            with autocast:
                features = model.encode_image(images, normalize=True)
            # Re-normalize in float32 so scoring precision is independent of autocast.
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            if all_features is not None:
                all_features.append(features.cpu().numpy())
            scores = score_features(features, positive, args.score, scale, bias, negative)
            all_scores.append(scores.cpu().numpy())
            all_targets.append(batch['targets'].numpy())
            ids.extend(batch['image_ids'])
            patients.extend(batch['patient_ids'])
            progress.update(len(images))
    if not all_scores:
        raise ValueError('The selected dataset split contains no images.')
    result = {
        'scores': np.concatenate(all_scores),
        'targets': np.concatenate(all_targets),
        'image_ids': np.asarray(ids),
        'patient_ids': np.asarray(patients),
        # Lets --save-features NPZs double as linear_probe.py feature caches.
        'group_ids': np.asarray(patients),
    }
    if all_features is not None:
        result['features'] = np.concatenate(all_features)
    return result


def check_disjoint(evaluation, calibration):
    if set(evaluation['image_ids']) & set(calibration['image_ids']):
        raise ValueError('Calibration and evaluation image IDs overlap.')
    eval_patients = set(evaluation['patient_ids']) - {''}
    calibration_patients = set(calibration['patient_ids']) - {''}
    if eval_patients & calibration_patients:
        raise ValueError('Calibration and evaluation patient IDs overlap.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=list(PRESETS) + ['timm/' + s for s in PRESETS])
    parser.add_argument('--model', default='ViT-B-16-SigLIP2-256')
    parser.add_argument('--pretrained', help='Pretrained tag or checkpoint; required unless --model is hf-hub:...')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action='store_true', help='CUDA bfloat16 image inference (text/scoring use float32).')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0, help='Parallel image decoding/preprocessing workers.')
    parser.add_argument(
        '--save-features', action='store_true', help='Include normalized float32 image embeddings in NPZ outputs.'
    )
    parser.add_argument(
        '--download-data', action='store_true', help='Cache each complete Parquet shard before reading it.'
    )
    parser.add_argument(
        '--score',
        choices=['cosine', 'logit', 'paired'],
        help='Default: paired for paired prompt strategies, otherwise cosine.',
    )
    parser.add_argument('--threshold', type=float, help='Fixed cutoff in score units; no fitted target labels.')
    parser.add_argument('--calibrate', choices=['none', 'global', 'per-class'], default='none')
    parser.add_argument('--limit', type=int, default=0, help='Evaluation images, 0 for the whole split.')
    parser.add_argument('--calibration-limit', type=int, default=0, help='Calibration images, 0 for the whole split.')
    parser.add_argument(
        '--shuffle-buffer', type=int, default=0, help='Optional bounded streaming shuffle, not uniform sampling.'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--revision', default='main', help='HF dataset revision; use a commit SHA for reproducibility.')
    parser.add_argument('--cache-dir', help='Dataset metadata cache; HF_DATASETS_CACHE sets the generator cache.')
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        '--prompt-strategy',
        choices=list(PROMPT_STRATEGIES),
        default='baseline',
        help='Built-in prompt ensemble (default: baseline); paired strategies also select paired scoring.',
    )
    prompt_group.add_argument(
        '--prompts', type=Path, help='JSON mapping each ClassLabel name to positive/negative string lists.'
    )
    parser.add_argument('--output', type=Path, required=True, help='Directory for metrics, prompts and NPZ scores.')
    args = parser.parse_args(argv)
    args.dataset = args.dataset.removeprefix('timm/')
    if args.prompts:
        args.prompt_strategy = 'custom'
    default_score = PROMPT_STRATEGIES.get(args.prompt_strategy, 'cosine')
    if default_score == 'paired' and args.score not in (None, 'paired'):
        parser.error(f'--prompt-strategy {args.prompt_strategy} requires --score paired.')
    args.score = args.score or default_score
    if not args.pretrained and not args.model.startswith('hf-hub:'):
        parser.error('Provide --pretrained (e.g. webli for SigLIP2); random weights are not a zero-shot baseline.')
    if args.batch_size <= 0 or min(args.limit, args.calibration_limit, args.shuffle_buffer, args.workers) < 0:
        parser.error('Batch size must be positive; limits and shuffle buffer must be nonnegative.')
    if args.amp and torch.device(args.device).type != 'cuda':
        parser.error('--amp requires a CUDA device.')
    if args.threshold is not None and not np.isfinite(args.threshold):
        parser.error('--threshold must be finite.')
    return args


def main(argv=None):
    import datasets
    import sklearn

    import open_clip

    args = parse_args(argv)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    ds, classnames, eval_spec = load_split(args)
    prompts = (
        json.loads(args.prompts.read_text())
        if args.prompts
        else make_prompts(args.dataset, classnames, args.prompt_strategy)
    )
    validate_prompts(prompts, classnames, args.score == 'paired')
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=args.device
    )
    if getattr(preprocess, 'is_naflex_eval_transform_factory', False) or not hasattr(model, 'encode_text'):
        raise ValueError('This evaluator requires a fixed-resolution contrastive image/text model.')
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    positive = encode_prompts(model, tokenizer, prompts, classnames, 'positive', args.device)
    negative = (
        encode_prompts(model, tokenizer, prompts, classnames, 'negative', args.device)
        if args.score == 'paired'
        else None
    )
    evaluation = collect_scores(model, preprocess, ds, classnames, positive, negative, args)
    report = {
        'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'versions': {
            'torch': torch.__version__,
            'open_clip': open_clip.__version__,
            'datasets': datasets.__version__,
            'sklearn': sklearn.__version__,
        },
        'classnames': classnames,
        'evaluation': eval_spec,
        'logit_scale': float(model.logit_scale.detach().float().exp().item()),
        'logit_bias': float(model.logit_bias.detach().float().item())
        if getattr(model, 'logit_bias', None) is not None
        else None,
        'ranking': ranking_metrics(evaluation['targets'], evaluation['scores'], classnames),
    }
    np.savez_compressed(args.output / 'evaluation.npz', **evaluation, classnames=np.asarray(classnames))
    if args.threshold is not None:
        report['fixed_threshold'] = {
            'threshold': args.threshold,
            'metrics': decision_metrics(evaluation['targets'], evaluation['scores'], args.threshold),
        }
    if args.calibrate != 'none':
        calibration_ds, calibration_names, calibration_spec = load_split(args, calibration=True)
        if classnames != calibration_names:
            raise ValueError('Calibration and evaluation label vocabularies differ.')
        calibration = collect_scores(model, preprocess, calibration_ds, classnames, positive, negative, args)
        check_disjoint(evaluation, calibration)
        thresholds, fallback = fit_thresholds(calibration['targets'], calibration['scores'], args.calibrate)
        report['calibrated_thresholds'] = {
            'method': args.calibrate,
            'data': calibration_spec,
            'num_images': len(calibration['targets']),
            'thresholds': thresholds.tolist(),
            'global_fallback_classes': [classnames[j] for j in fallback],
            'calibration_positives': calibration['targets'].sum(axis=0).tolist(),
            'metrics': decision_metrics(evaluation['targets'], evaluation['scores'], thresholds),
        }
        np.savez_compressed(args.output / 'calibration.npz', **calibration, classnames=np.asarray(classnames))
    (args.output / 'prompts.json').write_text(json.dumps(prompts, indent=2) + '\n')
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    summary = {k: v for k, v in report['ranking'].items() if k != 'per_class'}
    print(json.dumps(summary, indent=2))
    if 'calibrated_thresholds' in report:
        print('Frozen model with thresholds fitted on labeled calibration data:')
        print(json.dumps(report['calibrated_thresholds']['metrics'], indent=2))
    print(f'Results: {args.output / "metrics.json"}')


if __name__ == '__main__':
    main()
