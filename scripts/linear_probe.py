"""Frozen OpenCLIP linear probes from datasets or cached embeddings.

Fit or validate single-label softmax / multilabel logistic heads. Use --mode extract
for reusable feature caches, or --mode evaluate --head head.npz for validation.
See docs/linear_probe.md for HF, CSV, and WebDataset configuration examples.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from linear_probe_data import (
    ROLES,
    TASKS,
    check_disjoint,
    compatible,
    extract_features,
    holdout,
    load_features,
    make_source,
    read_config,
    validate_features,
)
from multilabel_zeroshot import decision_metrics, fit_thresholds, ranking_metrics
from torch.nn import functional as F


def fit_probe(features, targets, l2, max_iter=1500, log_interval=0, task='multi-label', num_classes=None):
    """Fit an unpenalized bias and L2-regularized weights in float64.

    Multilabel: mean BCE + l2 * sum(W**2) / (2 * C).
    Single-label: mean softmax CE + l2 * sum(W**2) / 2.
    """
    if features.ndim != 2 or not all(features.shape) or len(features) != len(targets):
        raise ValueError('Features and targets must have matching nonempty image counts.')
    if not np.isfinite(l2) or l2 <= 0 or max_iter <= 0:
        raise ValueError('Regularization and iteration limit must be positive and finite.')
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError('Features and targets must be finite.')
    if task == 'multi-label':
        if targets.ndim != 2 or not torch.all((targets == 0) | (targets == 1)):
            raise ValueError('Multilabel targets must be a binary matrix.')
        num_classes = targets.shape[1]
        counts = targets.sum(dim=0)
        if not torch.all((counts > 0) & (counts < len(targets))):
            raise ValueError('Every training class needs both positive and negative examples.')
        initial_bias = torch.logit(counts / len(targets))
    elif task == 'single-label':
        if targets.ndim != 1 or targets.dtype != torch.long or num_classes is None or num_classes < 2:
            raise ValueError('Single-label training requires integer targets and num_classes >= 2.')
        if (targets < 0).any() or (targets >= num_classes).any():
            raise ValueError('Single-label target outside the class vocabulary.')
        counts = targets.bincount(minlength=num_classes).to(features.dtype)
        if not (counts > 0).all():
            raise ValueError('Every class must occur in training; adjust the split or label vocabulary.')
        initial_bias = (counts / len(targets)).log()
        initial_bias -= initial_bias.mean()
    else:
        raise ValueError(f'Unknown task: {task}')
    weights = torch.nn.Parameter(features.new_zeros((features.shape[1], num_classes)))
    bias = torch.nn.Parameter(initial_bias.to(features.dtype))
    optimizer = torch.optim.LBFGS(
        [weights, bias],
        lr=1,
        max_iter=max_iter,
        history_size=100,
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
        line_search_fn='strong_wolfe',
    )
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        logits = features @ weights + bias
        loss = (
            F.binary_cross_entropy_with_logits(logits, targets)
            if task == 'multi-label'
            else F.cross_entropy(logits, targets)
        )
        loss = loss + l2 * weights.square().sum() / (2 * num_classes if task == 'multi-label' else 2)
        loss.backward()
        evaluations += 1
        if log_interval and evaluations % log_interval == 0:
            print(f'l2={l2:g} function_evaluations={evaluations} objective={loss.item():.8f}', flush=True)
        return loss

    optimizer.step(closure)
    objective = closure()
    if not torch.isfinite(objective):
        raise RuntimeError('Linear probe optimization produced a nonfinite objective.')
    state = optimizer.state[weights]
    diagnostics = {
        'iterations': int(state['n_iter']),
        'function_evaluations': int(state['func_evals']),
        'objective': objective.item(),
        'max_abs_gradient': max(weights.grad.abs().max().item(), bias.grad.abs().max().item()),
        'max_iter': max_iter,
        'hit_iteration_limit': int(state['n_iter']) >= max_iter,
        'hit_function_evaluation_limit': int(state['func_evals']) >= optimizer.param_groups[0]['max_eval'],
    }
    return weights.detach(), bias.detach(), diagnostics


def single_label_metrics(targets, scores, classnames):
    if scores.shape != (len(targets), len(classnames)) or not len(targets) or not np.isfinite(scores).all():
        raise ValueError('Expected nonempty, finite [images, classes] scores.')
    targets = np.asarray(targets)
    if (
        targets.ndim != 1
        or not np.issubdtype(targets.dtype, np.integer)
        or (targets < 0).any()
        or (targets >= len(classnames)).any()
    ):
        raise ValueError('Expected integer class indices for single-label metrics.')
    # Stable ties follow class order; top-1 agrees with argmax.
    order = np.argsort(-scores, axis=1, kind='stable')
    predicted = order[:, 0]
    k = min(5, len(classnames))
    per_class = {}
    for index, name in enumerate(classnames):
        selected = targets == index
        per_class[name] = {
            'support': int(selected.sum()),
            'accuracy': float((predicted[selected] == index).mean()) if selected.any() else None,
        }
    accuracies = [v['accuracy'] for v in per_class.values() if v['accuracy'] is not None]
    return {
        'num_images': len(targets),
        'num_classes': len(classnames),
        'top1': float((predicted == targets).mean()),
        'top_k': k,
        'top_k_accuracy': float((order[:, :k] == targets[:, None]).any(axis=1).mean()),
        'balanced_accuracy': float(np.mean(accuracies)),
        'per_class': per_class,
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_encoder(expected, actual):
    for key in ('model', 'pretrained', 'preprocess', 'feature_kind'):
        if key in expected and key in actual and expected[key] != actual[key]:
            raise ValueError(f'Encoder {key} differs from saved head/cache provenance.')


class FeatureProvider:
    """Lazy extraction/loading: evaluation is not opened until head selection is complete."""

    def __init__(self, args, head=None):
        self.args, self.model, self.preprocess = args, None, None
        self.head = head
        self.names = head['classnames'].tolist() if head is not None else None
        self.task = head['metadata']['task'] if head is not None else args.task
        self.cache = {}
        if args.features:
            self.manifest = json.loads(args.features.read_text())
            self.config = None
            self.paths = {}
            for key, path in self.manifest['paths'].items():
                path = Path(path)
                self.paths[key] = path if path.is_absolute() else args.features.resolve().parent / path
            self.roles = set(self.paths)
            if self.manifest.get('task') and self.task not in ('auto', self.manifest['task']):
                raise ValueError('Requested task differs from feature manifest.')
            if self.task == 'auto':
                self.task = self.manifest.get('task', 'auto')
        else:
            self.config = read_config(args.data_config) if args.data_config else hf_config(args)
            self.roles = set(self.config['splits'])
            if self.task != 'auto':
                declared = self.config.get('task', 'auto')
                if declared not in ('auto', self.task):
                    raise ValueError('Requested task differs from dataset config.')
                self.config['task'] = self.task
            self.manifest = {'format_version': 1, 'source': self.config, 'paths': {}, 'counts': {}}
        if not self.roles or not self.roles <= set(ROLES):
            raise ValueError(f'Splits must use these role names: {ROLES}.')
        if 'calibration' in self.roles and (
            args.calibration_fraction is not None
            or (self.config and self.config.get('calibration_fraction') is not None)
        ):
            raise ValueError('Use a calibration source or a training holdout fraction, not both.')
        if head is not None:
            if args.task not in ('auto', self.task):
                raise ValueError('Requested task differs from saved head.')
            check_encoder(head['metadata'].get('encoder', {}), self.manifest.get('encoder', {}))
        if args.cache_features and any(
            (args.cache_features / name).exists()
            for name in ('features.json', 'train.npz', 'calibration.npz', 'evaluation.npz')
        ):
            raise ValueError('Feature cache already exists; choose a fresh directory or reuse --features.')

    def load_encoder(self):
        if self.model is not None:
            return
        import open_clip

        expected = self.head['metadata'].get('encoder', {}) if self.head is not None else {}
        model_name = self.args.model or expected.get('model')
        pretrained = self.args.pretrained if self.args.pretrained is not None else expected.get('pretrained')
        if not model_name or (not pretrained and not model_name.startswith('hf-hub:')):
            raise ValueError('Raw datasets require --model and --pretrained (or an hf-hub: model).')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.args.device
        )
        if getattr(self.preprocess, 'is_naflex_eval_transform_factory', False):
            raise ValueError('This linear probe currently requires fixed-resolution global image embeddings.')
        self.model.eval().requires_grad_(False)
        self.manifest['encoder'] = {
            'model': model_name,
            'pretrained': pretrained,
            'preprocess': repr(self.preprocess),
            'feature_kind': 'normalized projected global image embedding',
            'image_amp': 'bfloat16' if self.args.amp else 'float32',
            'features_dtype': 'float32',
            'open_clip_version': open_clip.__version__,
        }
        check_encoder(expected, self.manifest['encoder'])

    def get(self, role):
        if role in self.cache:
            return self.cache[role]
        if self.config is None:
            data = load_features(self.paths[role], self.task)
        else:
            source = make_source(self.config, role, self.names, self.task if self.task != 'auto' else None)
            self.task, self.names = source.task, source.classnames
            self.load_encoder()
            data = extract_features(
                self.model,
                self.preprocess,
                source,
                self.args.device,
                self.args.batch_size,
                self.args.workers,
                self.args.amp,
            )
        self.task = validate_features(data, self.task)
        if self.names is not None and not np.array_equal(data['classnames'], self.names):
            raise ValueError('Class vocabulary/order differs from previous source or saved head.')
        self.names = data['classnames'].tolist()
        self.cache[role] = data
        self.manifest.update(task=self.task, classnames=self.names)
        self.manifest.setdefault('counts', {})[role] = len(data['targets'])
        return data

    def training_data(self):
        train = self.get('train')
        if 'calibration' in self.roles:
            calibration = self.get('calibration')
        else:
            fraction = self.args.calibration_fraction
            if fraction is None and self.config:
                fraction = self.config.get('calibration_fraction')
            if fraction is None:
                raise ValueError('Provide calibration data or --calibration-fraction to hold out training data.')
            train, calibration = holdout(train, fraction, self.args.seed)
            self.cache.update(train=train, calibration=calibration)
            self.manifest['holdout'] = {
                'fraction': fraction,
                'seed': self.args.seed,
                'method': 'lowest SHA256(seed:group_id), else image_id; no stratification',
            }
            self.manifest['counts'].update(train=len(train['targets']), calibration=len(calibration['targets']))
        compatible(train, calibration)
        return train, calibration

    def stash(self, roles):
        if not self.args.cache_features:
            return
        directory = self.args.cache_features.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        for role in roles:
            data = self.cache[role]
            path = directory / f'{role}.npz'
            if path.exists():
                raise ValueError(f'Feature cache exists: {path}; choose a fresh directory or reuse --features.')
            np.savez(path, **data)
            self.manifest.setdefault('paths', {})[role] = path.name
            self.manifest.setdefault('counts', {})[role] = len(data['targets'])
        self.manifest.update(task=self.task, classnames=self.names)
        write_json(directory / 'features.json', self.manifest)


def load_head(path):
    with np.load(path, allow_pickle=False) as data:
        head = dict(data)
    if 'metadata_json' not in head:
        raise ValueError('Saved head lacks metadata_json; refit it with this script.')
    head['metadata'] = json.loads(str(head.pop('metadata_json')))
    weight, bias, mean, std = [head[k] for k in ('weight', 'bias', 'feature_mean', 'feature_std')]
    if (
        weight.ndim != 2
        or bias.shape != (weight.shape[1],)
        or mean.shape != (weight.shape[0],)
        or std.shape != mean.shape
        or len(head['classnames']) != weight.shape[1]
        or not all(np.isfinite(v).all() for v in (weight, bias, mean, std))
        or not (std > 0).all()
    ):
        raise ValueError('Invalid saved head dimensions or numeric values.')
    if head['metadata']['task'] not in TASKS:
        raise ValueError('Unknown saved head task.')
    if head['metadata']['task'] == 'multi-label' and (
        head.get('thresholds', np.array([])).shape != bias.shape or not np.isfinite(head['thresholds']).all()
    ):
        raise ValueError('Multilabel head needs one finite saved threshold per class.')
    return head


def predict(data, head, device):
    if not np.array_equal(data['classnames'], head['classnames']):
        raise ValueError('Saved head class order differs from evaluation data.')
    if data['features'].shape[1] != head['weight'].shape[0]:
        raise ValueError('Saved head feature dimension differs from evaluation data.')
    mean, std, weight, bias = [
        torch.as_tensor(head[k], dtype=torch.float64, device=device)
        for k in ('feature_mean', 'feature_std', 'weight', 'bias')
    ]
    scores = []
    for start in range(0, len(data['features']), 4096):
        x = torch.as_tensor(data['features'][start : start + 4096], dtype=torch.float64, device=device)
        scores.append((((x - mean) / std) @ weight + bias).cpu().numpy())
    return np.concatenate(scores)


def evaluate(data, scores, names, task, thresholds=None):
    if task == 'single-label':
        return {'classification': single_label_metrics(data['targets'], scores, names)}
    return {
        'ranking': ranking_metrics(data['targets'], scores, names),
        'calibrated_thresholds': {
            'thresholds': thresholds.tolist(),
            'metrics': decision_metrics(data['targets'], scores, thresholds),
        },
        'fixed_probability_0_5': decision_metrics(data['targets'], scores, 0.0),
    }


def save_scores(path, data, scores):
    np.savez_compressed(
        path, scores=scores, **{k: data[k] for k in ('targets', 'image_ids', 'group_ids', 'patient_ids', 'classnames')}
    )


def hf_config(args):
    splits = {}
    for role, split in [
        ('train', args.train_split),
        ('calibration', args.calibration_split),
        ('evaluation', args.eval_split),
    ]:
        if split.lower() != 'none' and (args.mode != 'evaluate' or role == 'evaluation'):
            splits[role] = {'split': split}
    config = {
        'type': 'hfids' if args.hf_streaming else 'hfds',
        'dataset': args.dataset,
        'revision': args.revision,
        'task': args.task,
        'splits': splits,
    }
    for key in ('target_key', 'id_key', 'group_key'):
        if getattr(args, key):
            config[key] = getattr(args, key)
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--features', type=Path, help='Existing feature manifest; no image model is loaded.')
    inputs.add_argument('--data-config', type=Path, help='JSON source config for HF, CSV, or WebDataset.')
    inputs.add_argument('--dataset', help='HF dataset shorthand, e.g. timm/resisc45.')
    parser.add_argument('--mode', choices=['fit', 'extract', 'evaluate'], default='fit')
    parser.add_argument('--task', choices=['auto', *TASKS], default='auto')
    parser.add_argument('--head', type=Path, help='Saved head.npz, required for evaluate mode.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-features', type=Path, help='Optionally save embeddings plus features.json here.')
    parser.add_argument('--model', help='OpenCLIP model used to extract raw-dataset image embeddings.')
    parser.add_argument('--pretrained')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action='store_true', help='CUDA bfloat16 image extraction.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--l2', type=float, nargs='+', default=[1.0, 0.1, 0.01, 0.001, 0.0001])
    parser.add_argument('--max-iter', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--calibration-fraction',
        type=float,
        help='Hold out this fraction of training groups/images when no calibration source is given.',
    )
    parser.add_argument('--train-split', default='train', help='HF shorthand only; none disables this role.')
    parser.add_argument(
        '--calibration-split', default='validation', help='HF shorthand only; none enables a training holdout.'
    )
    parser.add_argument('--eval-split', default='test', help='HF shorthand only; none skips held-out evaluation.')
    parser.add_argument(
        '--revision', default='main', help='HF shorthand dataset revision; pin a commit for reproducibility.'
    )
    parser.add_argument(
        '--hf-streaming',
        action='store_true',
        help='Use finite streaming HF reads instead of the timm map-style wrapper.',
    )
    parser.add_argument('--target-key', help='HF shorthand target column; otherwise infer label or labels.')
    parser.add_argument('--id-key', help='HF shorthand stable image ID column.')
    parser.add_argument('--group-key', help='HF shorthand group column; patient_id is used automatically when present.')
    args = parser.parse_args(argv)
    if args.mode == 'evaluate' and not args.head:
        parser.error('--mode evaluate requires --head.')
    if args.mode != 'evaluate' and args.head:
        parser.error('--head is only used with --mode evaluate.')
    if args.mode == 'extract':
        if args.features:
            parser.error('Extract mode requires a raw dataset, not --features.')
        args.cache_features = args.cache_features or args.output / 'features'
    if args.features and args.cache_features:
        parser.error('--cache-features is for raw dataset extraction; --features already supplies a cache.')
    if (
        args.batch_size <= 0
        or args.workers < 0
        or args.max_iter <= 0
        or any(not np.isfinite(v) or v <= 0 for v in args.l2)
    ):
        parser.error('Batch size, iteration limit, and L2 values must be positive; workers must be nonnegative.')
    if args.calibration_fraction is not None and not 0 < args.calibration_fraction < 1:
        parser.error('--calibration-fraction must be strictly between 0 and 1.')
    if args.amp and torch.device(args.device).type != 'cuda':
        parser.error('--amp requires CUDA.')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error('Run one process; extraction workers are supported, distributed training is not.')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    head = load_head(args.head) if args.head else None
    provider = FeatureProvider(args, head)
    if args.mode == 'evaluate':
        if 'evaluation' not in provider.roles:
            raise ValueError('Evaluate mode requires an evaluation source/cache.')
        data = provider.get('evaluation')
        for role in ('train', 'calibration'):
            previous = {k: head.get(role + '_' + k, []) for k in ('image_ids', 'group_ids', 'patient_ids')}
            check_disjoint(previous, data)
        scores = predict(data, head, args.device)
        report = evaluate(data, scores, provider.names, provider.task, head.get('thresholds'))
        report.update(
            task=provider.task, mode='evaluate', head=str(args.head), features=provider.manifest, split='evaluation'
        )
        provider.stash(['evaluation'])
        save_scores(args.output / 'evaluation.npz', data, scores)
        write_json(args.output / 'metrics.json', report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('features',)}, indent=2))
        return
    if args.mode == 'extract':
        roles = []
        if 'train' in provider.roles:
            if (
                'calibration' in provider.roles
                or args.calibration_fraction is not None
                or provider.config.get('calibration_fraction')
            ):
                provider.training_data()
                roles.extend(['train', 'calibration'])
            else:
                provider.get('train')
                roles.append('train')
        for role in ROLES:
            if role in provider.roles and role not in roles:
                data = provider.get(role)
                for previous in roles:
                    compatible(provider.cache[previous], data)
                roles.append(role)
        provider.stash(roles)
        print(f'Features: {args.cache_features / "features.json"}')
        return
    if 'train' not in provider.roles:
        raise ValueError('Fit mode requires training data.')
    train, calibration = provider.training_data()
    provider.stash(['train', 'calibration'])
    names, task = provider.names, provider.task
    x = torch.as_tensor(train['features'], dtype=torch.float64, device=args.device)
    mean, std = x.mean(dim=0), x.std(dim=0, correction=0).clamp_min(1e-6)
    x = (x - mean) / std
    y = torch.as_tensor(train['targets'], dtype=torch.long if task == 'single-label' else x.dtype, device=args.device)
    x_cal = (torch.as_tensor(calibration['features'], dtype=x.dtype, device=args.device) - mean) / std
    candidates, best, best_value = [], None, -float('inf')
    for l2 in args.l2:
        weight, bias, diagnostics = fit_probe(
            x, y, l2, args.max_iter, log_interval=100, task=task, num_classes=len(names)
        )
        scores = (x_cal @ weight + bias).cpu().numpy()
        metrics = (
            ranking_metrics(calibration['targets'], scores, names)
            if task == 'multi-label'
            else single_label_metrics(calibration['targets'], scores, names)
        )
        value = metrics['macro_ap'] if task == 'multi-label' else metrics['top1']
        if value is None:
            raise ValueError('Calibration has no positives; cannot select a multilabel head.')
        candidate = {'l2': l2, 'optimization': diagnostics, 'calibration_score': value}
        candidate['calibration_macro_ap' if task == 'multi-label' else 'calibration_top1'] = value
        if task == 'multi-label':
            candidate['calibration_macro_auroc'] = metrics['macro_auroc']
        candidates.append(candidate)
        print(json.dumps(candidate), flush=True)
        if value > best_value:
            best_value, best = value, (weight, bias, scores, l2)
    weight, bias, cal_scores, chosen_l2 = best
    thresholds, fallback = (
        fit_thresholds(calibration['targets'], cal_scores, 'per-class') if task == 'multi-label' else (None, [])
    )
    selection = {
        'selected_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_metric': 'calibration macro AP' if task == 'multi-label' else 'calibration top-1 accuracy',
        'selected_l2': chosen_l2,
        'candidates': candidates,
        'training_images': len(y),
        'calibration_images': len(cal_scores),
        'threshold_method': 'per-class calibration F1' if task == 'multi-label' else 'argmax',
        'training_positives': (
            train['targets'].sum(axis=0)
            if task == 'multi-label'
            else np.bincount(train['targets'], minlength=len(names))
        ).tolist(),
        'global_fallback_classes': [names[i] for i in fallback],
    }
    write_json(args.output / 'selection.json', selection)
    head = {
        'weight': weight.cpu().numpy(),
        'bias': bias.cpu().numpy(),
        'feature_mean': mean.cpu().numpy(),
        'feature_std': std.cpu().numpy(),
        'classnames': np.asarray(names),
    }
    if thresholds is not None:
        head['thresholds'] = thresholds
    metadata = {'task': task, 'encoder': provider.manifest.get('encoder', {}), 'format_version': 1}
    head['metadata_json'] = np.asarray(json.dumps(metadata))
    for role, data in [('train', train), ('calibration', calibration)]:
        for key in ('image_ids', 'group_ids', 'patient_ids'):
            head[role + '_' + key] = data[key]
    np.savez(args.output / 'head.npz', **head)
    save_scores(args.output / 'calibration.npz', calibration, cal_scores)
    report = {
        'task': task,
        'mode': 'fit',
        'selection': selection,
        'features': provider.manifest,
        'optimization_dtype': 'float64',
        'calibration': evaluate(calibration, cal_scores, names, task, thresholds),
    }
    if 'evaluation' in provider.roles:
        # The held-out source is opened only after selecting and saving the head.
        data = provider.get('evaluation')
        compatible(train, data)
        compatible(calibration, data)
        scores = predict(data, head, args.device)
        report.update(evaluate(data, scores, names, task, thresholds))
        report['split'] = 'evaluation'
        if task == 'multi-label':
            report['calibrated_thresholds']['global_fallback_classes'] = [names[i] for i in fallback]
            if 'scores' in data and 'scores' in calibration:
                zero_thresholds, _ = fit_thresholds(calibration['targets'], calibration['scores'], 'per-class')
                report['zero_shot_same_encoder'] = {
                    'ranking': ranking_metrics(data['targets'], data['scores'], names),
                    'calibrated_metrics': decision_metrics(data['targets'], data['scores'], zero_thresholds),
                }
        save_scores(args.output / 'evaluation.npz', data, scores)
        provider.stash(['evaluation'])
    else:
        report['split'] = 'calibration_only'
        report['note'] = 'Calibration metrics reuse model-selection data; no held-out evaluation was provided.'
    write_json(args.output / 'metrics.json', report)
    key = 'ranking' if task == 'multi-label' else 'classification'
    print(
        json.dumps({k: v for k, v in report.get(key, report['calibration'][key]).items() if k != 'per_class'}, indent=2)
    )
    print(f'Results: {args.output / "metrics.json"}')


if __name__ == '__main__':
    main()
