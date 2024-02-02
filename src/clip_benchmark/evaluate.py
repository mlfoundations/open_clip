import json
import os
from typing import Any, Optional, Union

import torch
import torch.utils.data
from torch import nn
from training.distributed import world_info_from_env

from clip_benchmark.dataset import (
    build_dataset,
    get_dataset_collate_fn,
    get_dataset_default_task,
)
from clip_benchmark.metrics import (
    captioning,
    image_caption_selection,
    linear_probe,
    zeroshot_classification,
    zeroshot_retrieval,
)
from clip_benchmark.models import load_openclip_model

DEFAULT_RECALL_KS = [5]


class CLIPBenchmarkModel:
    def __init__(
        self,
        name: str,
        pretrained: str,
        module: Union[None, nn.Module] = None,
        transform: Any = None,
        tokenizer: Any = None,
    ):
        self.name = name
        self.pretrained = pretrained
        self.module = module
        self.transform = transform
        self.tokenizer = tokenizer
        if self.module is not None:
            assert self.tokenizer is not None and self.transform is not None

    def load(
        self,
        model_cache_dir: Optional[str] = None,
        device: Union[str, torch.device] = 'cpu',
    ):
        if self.module is None:
            self.module, self.transform, self.tokenizer = load_openclip_model(
                model=self.name,
                pretrained=self.pretrained,
                cache_dir=model_cache_dir,
                device=device,
            )


def run_evaluation_task(
    dataset: str,
    model: CLIPBenchmarkModel,
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
    linear_probe_train_split: str = 'train',
    linear_probe_val_split: Optional[str] = None,
    linear_probe_val_proportion: float = 0.2,
    linear_probe_fewshot_k: int = -1,
    linear_probe_fewshot_lr: float = 0.1,
    linear_probe_fewshot_epochs: int = 10,
):
    """Run a single evaluation task."""

    if torch.cuda.is_available():
        if distributed:
            local_rank, rank, world_size = world_info_from_env()
            device = 'cuda:%d' % local_rank
            torch.cuda.set_device(device)
        else:
            device = 'cuda'
    else:
        device = 'cpu'

    torch.manual_seed(seed)

    if dataset.startswith('wds/'):
        _dataset_name = dataset.replace('wds/', '', 1)
    else:
        _dataset_name = dataset

    if task == 'auto':
        task = get_dataset_default_task(_dataset_name)

    _pretrained_slug = (
        os.path.basename(model.pretrained)
        if os.path.isfile(model.pretrained)
        else model.pretrained
    )
    _pretrained_slug_full_path = (
        model.pretrained.replace('/', '_')
        if os.path.isfile(model.pretrained)
        else model.pretrained
    )
    _dataset_slug = _dataset_name.replace('/', '_')

    if output:
        output = output.format(
            model=model.name,
            pretrained=_pretrained_slug,
            pretrained_full_path=_pretrained_slug_full_path,
            task=task,
            dataset=_dataset_slug,
            language=language,
        )
        if os.path.exists(output) and skip_existing:
            print(f'Skipping {output}, exists already')
            return

    print(
        f"Running '{task}' on '{_dataset_name}' with model '{model.pretrained}' "
        f"on language '{language}'"
    )
    dataset_root = dataset_root.format(
        dataset=_dataset_name, dataset_cleaned=_dataset_name.replace('/', '-')
    )

    model.load(model_cache_dir=model_cache_dir, device=device)
    model.module.eval()

    _dataset = build_dataset(
        dataset_name=dataset,
        root=dataset_root,
        transform=model.transform,
        split=split,
        annotation_file=annotation_file,
        download=True,
        language=language,
        task=task,
        custom_template_file=custom_template_file,
        custom_classname_file=custom_classname_file,
        wds_cache_dir=wds_cache_dir,
    )
    collate_fn = get_dataset_collate_fn(dataset)

    try:
        print(f'Dataset size: {len(_dataset)}')
    except TypeError:
        print('IterableDataset has no len()')
    print(f'Dataset split: {split}')
    if hasattr(_dataset, 'classes') and _dataset.classes:
        try:
            print(f'Dataset classes: {_dataset.classes}')
            print(f'Dataset number of classes: {len(_dataset.classes)}')
        except AttributeError:
            print('Dataset has no classes.')

    if dataset.startswith('wds/'):
        dataloader = torch.utils.data.DataLoader(
            _dataset.batched(batch_size),
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            _dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    if task == 'zeroshot_classification':
        zeroshot_templates = (
            _dataset.templates if hasattr(_dataset, 'templates') else None
        )
        print(f'Zero-shot templates: {zeroshot_templates}')

        classnames = _dataset.classes if hasattr(_dataset, 'classes') else None
        assert (
            zeroshot_templates is not None and classnames is not None
        ), 'Dataset does not support classification'

        metrics = zeroshot_classification.evaluate(
            model.module,
            dataloader,
            model.tokenizer,
            classnames,
            zeroshot_templates,
            device=device,
            amp=amp,
            verbose=True,
            save_clf=save_clf,
            load_clfs=load_clfs or [],
        )

    elif task == 'zeroshot_retrieval':
        metrics = zeroshot_retrieval.evaluate(
            model.module,
            dataloader,
            model.tokenizer,
            recall_k_list=recall_ks or DEFAULT_RECALL_KS,
            device=device,
            amp=amp,
        )
    elif task == 'image_caption_selection':
        metrics = image_caption_selection.evaluate(
            model.module,
            dataloader,
            model.tokenizer,
            device=device,
            amp=amp,
        )
    elif task == 'linear_probe':
        assert linear_probe_train_split
        train_dataset = build_dataset(
            dataset_name=dataset,
            root=dataset_root,
            transform=model.transform,
            split=linear_probe_train_split,
            annotation_file=annotation_file,
            download=True,
        )
        if linear_probe_val_split is not None:
            val_dataset = build_dataset(
                dataset_name=dataset,
                root=dataset_root,
                transform=model.transform,
                split=linear_probe_val_split,
                annotation_file=annotation_file,
                download=True,
            )
        elif linear_probe_val_proportion is not None:
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [1 - linear_probe_val_proportion, linear_probe_val_proportion],
            )
        else:
            val_dataset = None

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        if val_dataset is not None:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
        else:
            val_dataloader = None

        metrics = linear_probe.evaluate(
            model.module,
            train_dataloader,
            dataloader,
            linear_probe_fewshot_k,
            batch_size,
            num_workers,
            linear_probe_fewshot_lr,
            linear_probe_fewshot_epochs,
            (model.name + '-' + model.pretrained + '-' + dataset).replace('/', '_'),
            seed,
            feature_root,
            val_dataloader=val_dataloader,
            device=device,
            normalize=normalize,
            amp=amp,
            verbose=True,
        )
    elif task == 'captioning':
        metrics = captioning.evaluate(
            model=model.module,
            dataloader=dataloader,
            device=device,
        )
    else:
        raise ValueError(
            f'Unsupported task: {task}. Task should be `zeroshot_classification`, '
            f'`zeroshot_retrieval`, `linear_probe`, or `captioning`'
        )

    dump = {
        'dataset': dataset,
        'model': model.name,
        'pretrained': model.pretrained,
        'task': task,
        'metrics': metrics,
        'language': language,
    }
    if hasattr(_dataset, 'classes') and _dataset.classes and dump_classnames:
        dump['classnames'] = _dataset.classes
    if hasattr(_dataset, 'templates') and _dataset.templates and dump_templates:
        dump['templates'] = _dataset.templates

    if output:
        print(f'Dumping results to: {output}')
        with open(output, 'w') as f:
            json.dump(dump, f)

    return dump
