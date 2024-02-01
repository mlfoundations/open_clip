import os
import random
from itertools import product
from typing import Any, Optional, Union

from training.distributed import world_info_from_env

from clip_benchmark.dataset import dataset_collection, get_dataset_collection_from_file
from clip_benchmark.evaluate import CLIPBenchmarkModel, run_evaluation_task
from clip_benchmark.models import model_collection


def get_model_collection_from_file(path: str):
    return [line.strip().split(',') for line in open(path).readlines()]


def _as_list(s: Any) -> list[Any]:
    if not s:
        return []
    return [s] if not isinstance(s, list) else s


def _single_option_to_multiple_datasets(cur_option, datasets, name):
    cur_len = len(cur_option)
    ds_len = len(datasets)
    if cur_len != ds_len:
        # If user wants to use same value for all datasets
        if cur_len == 1:
            return [cur_option[0]] * ds_len
        else:
            raise ValueError(f'The incommensurable number of {name}')
    else:
        return cur_option


def run_benchmark(
    datasets: Union[str, list[str]],
    models: Union[str, CLIPBenchmarkModel, list[str], list[CLIPBenchmarkModel]],
    task: str = 'auto',
    output: Optional[str] = None,
    language: str = 'en',
    dataset_root: str = 'root',
    feature_root: str = 'features',
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 0,
    amp: bool = False,
    distributed: bool = False,
    skip_existing: bool = False,
    recall_ks: Optional[list[int]] = None,
    save_clf: Optional[str] = None,
    load_clfs: Optional[list[str]] = None,
    annotation_file: Optional[str] = None,
    custom_template_file: Optional[str] = None,
    custom_classname_file: Optional[str] = None,
    dump_classnames: bool = False,
    dump_templates: bool = False,
    model_cache_dir: Optional[str] = None,
    wds_cache_dir: Optional[str] = None,
    normalize: bool = True,
    split: str = 'test',
    linear_probe_train_splits: str = 'train',
    linear_probe_val_splits: Optional[Union[str, list[str]]] = None,
    linear_probe_val_proportions: Union[float, list[float]] = 0.2,
    linear_probe_fewshot_k: int = -1,
    linear_probe_fewshot_lr: float = 0.1,
    linear_probe_fewshot_epochs: int = 10,
):
    _models = []
    for name in _as_list(models):
        if isinstance(name, str):
            if os.path.isfile(name):
                _models.extend(
                    [
                        CLIPBenchmarkModel(name=a, pretrained=b)
                        for a, b in get_model_collection_from_file(name)
                    ]
                )
            elif name in model_collection:
                _models.extend(
                    [
                        CLIPBenchmarkModel(name=a, pretrained=b)
                        for a, b in model_collection[name]
                    ]
                )
            else:
                model, pretrained = name.split(',')
                _models.append((model, pretrained))
        elif isinstance(name, CLIPBenchmarkModel):
            _models.append(name)
        else:
            raise TypeError(
                f'Expected type `{str.__name__}` or `{CLIPBenchmarkModel.__name__}` '
                f'for argument `models`'
            )

    if len(_models) == 0:
        raise ValueError('No models provided')

    _datasets = []
    for name in _as_list(datasets):
        if os.path.isfile(name):
            _datasets.extend(get_dataset_collection_from_file(name))
        elif name in dataset_collection:
            _datasets.extend(dataset_collection[name])
        else:
            _datasets.append(name)

    if len(_datasets) == 0:
        raise ValueError('No datasets provided')

    linear_probe_train_splits = _as_list(linear_probe_train_splits)
    linear_probe_train_splits = _single_option_to_multiple_datasets(
        linear_probe_train_splits, datasets, 'train_split'
    )

    _linear_probe_val_proportions, _linear_probe_val_splits = None, None
    if linear_probe_val_splits is not None:
        _linear_probe_val_splits = _as_list(linear_probe_val_splits)
        _linear_probe_val_splits = _single_option_to_multiple_datasets(
            _linear_probe_val_splits, datasets, 'linear_probe_val_splits'
        )
    if linear_probe_val_proportions is not None:
        _linear_probe_val_proportions = _as_list(linear_probe_val_proportions)
        _linear_probe_val_proportions = _single_option_to_multiple_datasets(
            _linear_probe_val_proportions, datasets, 'linear_probe_val_proportions'
        )

    _dataset_info = {}
    for i in range(len(datasets)):
        _dataset_info[datasets[i]] = {
            'linear_probe_train_split': linear_probe_train_splits[i],
            'linear_probe_val_split': (
                _linear_probe_val_splits[i]
                if _linear_probe_val_splits is not None
                else None
            ),
            'linear_probe_val_proportion': (
                _linear_probe_val_proportions[i]
                if _linear_probe_val_proportions is not None
                else None
            ),
        }

    languages = _as_list(language)

    print('Starting OpenCLIP benchmark ...')
    print(f'Models: {[model.name for model in _models]}')
    print(f'Datasets: {_datasets}')
    print(f'Languages: {languages}')

    runs = product(_models, _datasets, languages)

    print(f'Number of runs: {len(runs)}')

    if distributed:
        local_rank, rank, world_size = world_info_from_env()
        runs = list(runs)
        random.seed(seed)
        random.shuffle(runs)
        runs = [r for i, r in enumerate(runs) if i % world_size == rank]

    results = []
    for i, (model, dataset, language) in enumerate(runs):
        print('-----------------------------------------------------------------------')
        print(f'Running task {i+1}/{len(runs)} ...')
        print(
            f'Model: {model.name},{model.pretrained} - Dataset: {dataset} - '
            f'Language: {language}'
        )
        print('-----------------------------------------------------------------------')

        metrics = run_evaluation_task(
            model=model,
            dataset=dataset,
            task=task,
            output=output,
            language=language,
            dataset_root=dataset_root,
            feature_root=feature_root,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            amp=amp,
            distributed=distributed,
            skip_existing=skip_existing,
            recall_ks=recall_ks,
            save_clf=save_clf,
            load_clfs=load_clfs,
            annotation_file=annotation_file,
            custom_template_file=custom_template_file,
            custom_classname_file=custom_classname_file,
            dump_classnames=dump_classnames,
            dump_templates=dump_templates,
            model_cache_dir=model_cache_dir,
            wds_cache_dir=wds_cache_dir,
            normalize=normalize,
            split=split,
            linear_probe_train_split=_dataset_info[dataset]['linear_probe_train_split'],
            linear_probe_val_split=_dataset_info[dataset]['linear_probe_val_split'],
            linear_probe_val_proportion=(
                _dataset_info[dataset]['linear_probe_val_proportion']
            ),
            linear_probe_fewshot_k=linear_probe_fewshot_k,
            linear_probe_fewshot_lr=linear_probe_fewshot_lr,
            linear_probe_fewshot_epochs=linear_probe_fewshot_epochs,
        )
        results.append(metrics)

    return results
