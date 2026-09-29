"""Classification data and feature caches for scripts/linear_probe.py.

All sources make one finite pass. No resampling, dropped batches, or skipped errors.
"""

import csv
import hashlib
import io
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

TASKS = ('single-label', 'multi-label')
ROLES = ('train', 'calibration', 'evaluation')


def field(row, key):
    """Read a column or a dotted JSON field (e.g. json.labels in a tar sample)."""
    value = row
    for part in key.split('.'):
        if isinstance(value, bytes):
            value = json.loads(value)
        value = value[part]
    return value


def label_index(value, classnames):
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if isinstance(value, str):
        value = value.strip()
        if value in classnames:
            return classnames.index(value)
        try:
            value = int(value)
        except ValueError as error:
            raise ValueError(f'Unknown class {value!r}.') from error
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'Expected an integer class ID or class name, got {value!r}.')
    if not 0 <= value < len(classnames):
        raise ValueError(f'Class ID {value} outside [0, {len(classnames)}).')
    return int(value)


def parse_target(value, task, classnames, target_format='indices', label_separator=None):
    if task == 'single-label':
        return label_index(value, classnames)
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if isinstance(value, str):
        if not value.strip():
            value = []
        elif label_separator:
            value = value.split(label_separator)
        else:
            value = json.loads(value)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError('Multilabel targets require a list (JSON in CSV), including [] for no labels.')
    if target_format == 'multi-hot':
        target = np.asarray(value)
        if target.shape != (len(classnames),) or not np.isin(target, [0, 1]).all():
            raise ValueError('Multi-hot targets must be binary vectors with one entry per class.')
        return target.astype(np.uint8)
    if target_format != 'indices':
        raise ValueError(f'Unknown target_format: {target_format}')
    target = np.zeros(len(classnames), dtype=np.uint8)
    for item in value:
        target[label_index(item, classnames)] = 1
    return target


def decode_image(value):
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    if isinstance(value, dict):
        value = value.get('bytes') or value.get('path')
    if isinstance(value, bytes):
        value = io.BytesIO(value)
    with Image.open(value) as image:
        return image.convert('RGB')


def selected(row, spec):
    for rule, include in [('include', True), ('exclude', False)]:
        for key, values in spec.get(rule, {}).items():
            values = values if isinstance(values, list) else [values]
            if (str(field(row, key)) in {str(v) for v in values}) != include:
                return False
    return True


def _parquet_rows(files):
    # Reuse the synchronous reader: it uses cached HF shards and avoids background
    # Arrow scanner threads surviving a limited stream's shutdown.
    from multilabel_zeroshot import iter_parquet_rows

    for path in files:
        for index, row in enumerate(iter_parquet_rows([path], columns=None)):
            yield row, f'{path}:{index}'


class ClassificationSource(IterableDataset):
    def __init__(self, spec, classnames, task, files=None, streaming=None):
        self.spec, self.classnames, self.task = spec, classnames, task
        self.files, self.streaming = files, streaming

    def rows(self):
        spec = self.spec
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        if spec['type'] == 'csv':
            path = Path(spec['path'])
            root = Path(spec.get('image_root', path.parent))
            with path.open(newline='', encoding='utf-8') as stream:
                for index, row in enumerate(csv.DictReader(stream, delimiter=spec.get('delimiter', ','))):
                    if index % workers != worker_id:
                        continue
                    image_key = spec.get('image_key', 'image')
                    row[image_key] = str((root / row[image_key]).resolve())
                    yield row, row[image_key]
        elif spec['type'] == 'webdataset':
            import webdataset as wds

            # WebDataset handles worker sharding itself, including URLs/brace expansion.
            stream = wds.DataPipeline(
                wds.SimpleShardList(spec['path']),
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.reraise_exception),
            )
            for row in stream:
                yield row, row['__key__']
        elif self.files is not None:
            yield from _parquet_rows(self.files[worker_id::workers])
        else:
            # HF IterableDataset handles its own worker sharding.
            for row in self.streaming:
                # Non-Parquet datasets without IDs need an explicit id_key: a
                # worker-local row counter would not be a stable identifier.
                yield row, None

    def __iter__(self):
        for row, fallback_id in self.rows():
            if selected(row, self.spec):
                image, target, image_id, group_id = sample_values(
                    row, self.spec, self.classnames, self.task, fallback_id
                )
                yield {'image': decode_image(image), 'target': target, 'image_id': image_id, 'group_id': group_id}


def sample_values(row, spec, classnames, task, fallback_id):
    image_keys = spec.get('image_key', 'jpg;png;jpeg;webp' if spec['type'] == 'webdataset' else 'image')
    image_key = next((key for key in image_keys.split(';') if key in row), image_keys)
    id_key, group_key = spec.get('id_key'), spec.get('group_key')
    image_id = field(row, id_key) if id_key else row.get('image_id', fallback_id)
    group_id = field(row, group_key) if group_key else ''
    if image_id is None or str(image_id) == '':
        raise ValueError('Every sample needs a stable image ID; configure id_key.')
    if group_key and (group_id is None or str(group_id) == ''):
        raise ValueError(f'Missing group ID for {image_id}.')
    target = parse_target(
        field(row, spec['target_key']),
        task,
        classnames,
        spec.get('target_format', 'indices'),
        spec.get('label_separator'),
    )
    return field(row, image_key), target, str(image_id), str(group_id)


class MetadataReader:
    """Small reader for timm ImageDataset; retains IDs alongside classification targets."""

    def __init__(self, rows, spec, names, task):
        self.rows, self.spec, self.names, self.task = rows, spec, names, task
        self.indices = (
            [i for i, row in enumerate(rows) if selected(row, spec)]
            if any(spec.get(k) for k in ('include', 'exclude'))
            else range(len(rows))
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        row = self.rows[index]
        fallback = f"{self.spec['dataset']}:{self.spec['split']}:{index}"
        image, target, image_id, group_id = sample_values(row, self.spec, self.names, self.task, fallback)
        if isinstance(image, dict):
            image = image.get('bytes') or image.get('path')
        image = io.BytesIO(image) if isinstance(image, bytes) else open(image, 'rb')  # noqa: SIM115
        return image, target, image_id, group_id


class TimmClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, rows, spec, classnames, task):
        from timm.data.dataset import ImageDataset

        self.spec, self.classnames, self.task = spec, classnames, task
        reader = MetadataReader(rows, spec, classnames, task)
        self.dataset = ImageDataset(
            root=None, reader=reader, input_img_mode='RGB', additional_features=['image_id', 'group_id']
        )
        # Benchmark failures must surface, rather than silently substituting the next image.
        # timm has no public option for this; _max_retries is a private attribute.
        self.dataset._max_retries = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target, image_id, group_id = self.dataset[index]
        return {'image': image, 'target': target, 'image_id': image_id, 'group_id': group_id}


def make_source(config, role, expected_names=None, expected_task=None):
    """Resolve a source from metadata, never from labels observed in evaluation."""
    stage = config['splits'][role]
    kind = config['type']
    if isinstance(stage, str):
        stage = {'split' if kind in ('hfds', 'hfids') else 'path': stage}
    spec = {k: v for k, v in config.items() if k not in ('splits', 'calibration_fraction')}
    spec.update(stage)
    files, streaming, metadata_names, metadata_task, mapped = None, None, None, None, None
    if kind in ('hfds', 'hfids'):
        import datasets

        kwargs = {k: spec[k] for k in ('revision', 'data_files', 'cache_dir') if k in spec}
        builder = datasets.load_dataset_builder(spec['dataset'], name=spec.get('config_name'), **kwargs)
        split = spec['split']
        if not builder.config.data_files or split not in builder.config.data_files:
            raise ValueError(f'Unknown or unsupported split {split!r}; use a named split and include/exclude filters.')
        paths = list(builder.config.data_files[split])
        is_parquet = paths and all(str(path).split('?')[0].endswith('.parquet') for path in paths)
        if kind == 'hfds':
            if is_parquet:
                # Prepare only this physical split, retaining its HF label schema.
                mapped = datasets.load_dataset(
                    'parquet',
                    data_files={split: paths},
                    split=split,
                    features=builder.info.features,
                    cache_dir=spec.get('cache_dir'),
                )
            else:
                builder.download_and_prepare()
                mapped = builder.as_dataset(split=split)
            features = mapped.features
            mapped = mapped.cast_column(spec.get('image_key', 'image'), datasets.Image(decode=False))
        elif is_parquet:
            files = [str(p) for p in paths]
            features = builder.info.features
            if features is None:
                from pyarrow import parquet

                # Local Parquet fixtures and local datasets can obtain metadata
                # directly from the Arrow schema, without download_and_prepare.
                if not Path(files[0]).is_file():
                    raise ValueError('Remote Parquet needs HF feature metadata with a ClassLabel vocabulary.')
                features = datasets.Features.from_arrow_schema(parquet.read_schema(files[0]))
        else:
            streaming = builder.as_streaming_dataset(split=split)
            features = streaming.features
            image_key = spec.get('image_key', 'image')
            if features and isinstance(features.get(image_key), datasets.Image):
                streaming = streaming.cast_column(image_key, datasets.Image(decode=False))
        if features is None:
            raise ValueError('HF source needs feature metadata; provide a dataset with a declared schema.')
        if 'target_key' not in spec:
            spec['target_key'] = 'label' if 'label' in features else 'labels'
        feature = field(features, spec['target_key'])
        if isinstance(feature, datasets.ClassLabel):
            metadata_names, metadata_task = feature.names, 'single-label'
        elif isinstance(getattr(feature, 'feature', None), datasets.ClassLabel):
            metadata_names, metadata_task = feature.feature.names, 'multi-label'
        if 'group_key' not in spec and 'patient_id' in features:
            spec['group_key'] = 'patient_id'
    elif kind not in ('csv', 'webdataset'):
        raise ValueError(f'Unsupported source type: {kind}')
    else:
        spec.setdefault('target_key', 'cls' if kind == 'webdataset' else 'label')
    names = spec.get('classnames', metadata_names if metadata_names is not None else expected_names)
    if not names or not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
        raise ValueError('Provide a nonempty, unique classnames list or HF ClassLabel metadata.')
    if metadata_names is not None and list(names) != list(metadata_names):
        raise ValueError('Configured class order does not match HF ClassLabel metadata.')
    task = spec.get('task', 'auto')
    if task == 'auto':
        task = metadata_task or expected_task
    if task not in TASKS:
        raise ValueError('Set task to single-label or multi-label for sources without ClassLabel metadata.')
    if metadata_task and task != metadata_task:
        raise ValueError('Requested task conflicts with HF label schema.')
    if expected_names is not None and list(names) != list(expected_names):
        raise ValueError('Class vocabularies/order differ across sources or saved head.')
    if expected_task and task != expected_task:
        raise ValueError('Task differs across sources or saved head.')
    if mapped is not None:
        return TimmClassificationDataset(mapped, spec, list(names), task)
    return ClassificationSource(spec, list(names), task, files, streaming)


class ImageCollator:
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, rows):
        return (
            torch.stack([self.preprocess(row['image']) for row in rows]),
            np.stack([row['target'] for row in rows]),
            [row['image_id'] for row in rows],
            [row['group_id'] for row in rows],
        )


@torch.inference_mode()
def extract_features(model, preprocess, source, device='cpu', batch_size=64, workers=0, amp=False):
    features, targets, image_ids, group_ids = [], [], [], []
    loader = DataLoader(
        source,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=ImageCollator(preprocess),
        pin_memory=torch.device(device).type == 'cuda',
        multiprocessing_context='spawn' if workers else None,
    )
    limit = source.spec.get('limit', 0)
    if limit < 0:
        raise ValueError('Source limit must be nonnegative.')
    with tqdm(desc='Extracting frozen features', unit='image', mininterval=5) as progress:
        for images, target, ids, groups in loader:
            remaining = min(len(images), limit - len(image_ids)) if limit else len(images)
            images = images[:remaining].to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
                value = model.encode_image(images, normalize=True)
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError('Probe requires one global image embedding per image.')
            features.append(torch.nn.functional.normalize(value.float(), dim=-1).cpu().numpy())
            targets.append(target[:remaining])
            image_ids.extend(ids[:remaining])
            group_ids.extend(groups[:remaining])
            progress.update(remaining)
            if limit and len(image_ids) >= limit:
                break
    if not features:
        raise ValueError('Source produced no images after filtering.')
    data = {
        'features': np.concatenate(features),
        'targets': np.concatenate(targets),
        'image_ids': np.asarray(image_ids),
        'group_ids': np.asarray(group_ids),
        'patient_ids': np.asarray(group_ids)
        if source.spec.get('group_key') == 'patient_id'
        else np.full(len(image_ids), ''),
        'classnames': np.asarray(source.classnames),
    }
    validate_features(data, source.task)
    return data


def validate_features(data, task=None):
    x, y = data['features'], data['targets']
    names = data['classnames']
    if x.ndim != 2 or not all(x.shape) or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError('Features must be finite, nonempty [images, dimensions] arrays matching targets.')
    if (
        names.ndim != 1
        or not len(names)
        or names.dtype.kind not in ('U', 'S')
        or len(set(names.tolist())) != len(names)
    ):
        raise ValueError('Feature cache classnames must be a nonempty unique vector.')
    data['classnames'] = names = names.astype(str)
    if (names == '').any():
        raise ValueError('Class names must not be empty.')
    inferred = 'single-label' if y.ndim == 1 else 'multi-label'
    task = inferred if task in (None, 'auto') else task
    if task == 'single-label':
        if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= len(names)).any():
            raise ValueError('Single-label targets must be integer class IDs in [0, num_classes).')
    elif task != 'multi-label' or y.shape != (len(x), len(names)) or not np.isin(y, [0, 1]).all():
        raise ValueError('Multilabel targets must be binary [images, classes] arrays.')
    for key in ('image_ids', 'group_ids', 'patient_ids'):
        if key not in data or data[key].shape != (len(x),) or data[key].dtype.kind not in ('U', 'S'):
            raise ValueError(f'{key} must be one string per image.')
        data[key] = data[key].astype(str)
    if (data['image_ids'] == '').any() or len(set(data['image_ids'].tolist())) != len(x):
        raise ValueError('Duplicate or missing image IDs in feature cache/source.')
    if 'scores' in data and (data['scores'].shape != (len(x), len(names)) or not np.isfinite(data['scores']).all()):
        raise ValueError('Invalid cached zero-shot scores.')
    return task


def load_features(path, task=None):
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    validate_features(data, task)
    return data


def check_disjoint(left, right):
    for key in ('image_ids', 'group_ids', 'patient_ids'):
        a, b = set(left.get(key, [])) - {''}, set(right.get(key, [])) - {''}
        if a & b:
            raise ValueError(f'Dataset {key} overlap ({len(a & b)} shared IDs).')


def compatible(left, right):
    if not np.array_equal(left['classnames'], right['classnames']):
        raise ValueError('Class vocabularies/order differ between feature caches.')
    if left['features'].shape[1] != right['features'].shape[1]:
        raise ValueError('Feature dimensions differ between caches.')
    check_disjoint(left, right)


def holdout(data, fraction, seed):
    """Deterministic holdout by group when present, otherwise image ID; no label stratification."""
    if not 0 < fraction < 1:
        raise ValueError('calibration_fraction must be strictly between 0 and 1.')
    groups = data['group_ids']
    if (groups != '').any() and (groups == '').any():
        raise ValueError('Cannot split a source with partially missing group IDs.')
    units = groups if (groups != '').any() else data['image_ids']
    unique = sorted(set(units), key=lambda value: hashlib.sha256(f'{seed}:{value}'.encode()).digest())
    if len(unique) < 2:
        raise ValueError('Need at least two independent groups/images for calibration holdout.')
    selected_units = set(unique[: max(1, min(len(unique) - 1, round(len(unique) * fraction)))])
    mask = np.asarray([value in selected_units for value in units])

    def subset(selection):
        return {key: value if key == 'classnames' else value[selection] for key, value in data.items()}

    return subset(~mask), subset(mask)


def read_config(path):
    config = json.loads(Path(path).read_text())
    base = Path(path).resolve().parent
    # Local paths in a raw dataset config are relative to that config, including
    # CSV image roots and local WDS brace patterns. URLs are preserved.
    if config.get('type') in ('csv', 'webdataset'):
        for role, stage in config['splits'].items():
            if isinstance(stage, str):
                stage = {'path': stage}
                config['splits'][role] = stage
            paths = stage['path']
            paths = paths if isinstance(paths, list) else [paths]
            resolved = [str(base / p) if '://' not in p and not p.startswith('pipe:') else p for p in paths]
            stage['path'] = resolved if isinstance(stage['path'], list) else resolved[0]
            if 'image_root' in stage:
                stage['image_root'] = str(base / stage['image_root'])
        if 'image_root' in config:
            config['image_root'] = str(base / config['image_root'])
    return config
