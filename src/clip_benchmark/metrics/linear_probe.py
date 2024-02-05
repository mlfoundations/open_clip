import os
import time
from contextlib import suppress
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as f
from sklearn.metrics import balanced_accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clip_benchmark.metrics.zeroshot_classification import accuracy


def _assign_learning_rate(param_group, new_lr):
    param_group['lr'] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def _cosine_lr(optimizer, base_lrs: float | list[float], warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            _assign_learning_rate(param_group, lr)

    return _lr_adjuster


class Featurizer(torch.nn.Module):
    def __init__(self, model, normalize=True):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, _input):
        image_features = self.model.encode_image(_input)
        if self.normalize:
            image_features = f.normalize(image_features, dim=-1)
        return image_features


class FeatureDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.targets[i]


def _train(
    dataloader, input_shape, output_shape, weight_decay, lr, epochs, autocast, seed
):
    torch.manual_seed(seed)
    model = torch.nn.Linear(input_shape, output_shape)
    devices = [x for x in range(torch.cuda.device_count())]
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=devices)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    len_loader = len(dataloader)
    scheduler = _cosine_lr(optimizer, lr, 0.0, epochs * len_loader)

    for epoch in range(epochs):
        end = time.time()
        for i, (x, y) in enumerate(dataloader):
            x, y = x.cuda(), y.cuda()
            step = i + epoch * len_loader
            data_time = time.time() - end
            scheduler(step)

            optimizer.zero_grad()
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            if (i % 20) == 1:
                num_samples = i * len(x)
                try:
                    samples_per_epoch = len(dataloader)
                    percent_complete = 100.0 * i / len(dataloader)
                    progress_message = (
                        f'[{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]'
                    )
                except TypeError:
                    progress_message = f'[{num_samples} samples]'
                print(
                    f"Train Epoch: {epoch} {progress_message}\t"
                    f"Loss: {loss.item():.6f}\t"
                    f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t"
                    f"LR {optimizer.param_groups[0]['lr']:.5f}"
                )
    return model


def _infer(model, dataloader, autocast, device):
    true, pred = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)

            with autocast():
                logits = model(x)

            pred.append(logits.cpu())
            true.append(y.cpu())

    logits = torch.cat(pred)
    target = torch.cat(true)
    return logits, target


def _find_peak(
    wd_list,
    idxs,
    train_loader,
    val_loader,
    input_shape,
    output_shape,
    lr,
    epochs,
    autocast,
    device,
    verbose,
    seed,
):
    best_wd_idx, max_acc = 0, 0
    for idx in idxs:
        weight_decay = wd_list[idx]
        model = _train(
            train_loader,
            input_shape,
            output_shape,
            weight_decay,
            lr,
            epochs,
            autocast,
            seed,
        )
        logits, target = _infer(model, val_loader, autocast, device)
        (acc1,) = accuracy(logits.float(), target.float(), topk=(1,))
        if verbose:
            print(f'Valid accuracy with weight_decay {weight_decay}: {acc1}')
        if max_acc < acc1:
            best_wd_idx, max_acc = idx, acc1

    return best_wd_idx


def evaluate(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    dataloader: torch.utils.data.DataLoader,
    fewshot_k: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    epochs: int,
    model_id: str,
    seed: int,
    feature_root: str,
    device: Union[str, torch.device],
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    normalize: bool = True,
    amp: bool = True,
    verbose: bool = False,
):
    assert device == 'cuda', 'Need to use cuda for linear probing, otherwise too slow'

    # first we need to featurize the dataset, and store the result in feature_root
    if not os.path.exists(feature_root):
        os.mkdir(feature_root)

    feature_dir = os.path.join(feature_root, model_id)
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)

    featurizer = Featurizer(model, normalize).cuda()
    autocast = torch.cuda.amp.autocast if amp else suppress

    if not os.path.exists(os.path.join(feature_dir, 'targets_train.pt')):
        # now we have to cache the features
        devices = [x for x in range(torch.cuda.device_count())]
        featurizer = torch.nn.DataParallel(featurizer, device_ids=devices)

        splits = ['_train', '_val', '_test']

        for save_str, loader in zip(
            splits, [train_dataloader, val_dataloader, dataloader]
        ):
            if loader is None:
                continue
            features = []
            targets = []
            num_batches_tracked = 0
            num_cached = 0
            with torch.no_grad():
                for images, target in tqdm(loader):
                    images = images.to(device)

                    with autocast():
                        feature = featurizer(images)

                    features.append(feature.cpu())
                    targets.append(target)

                    num_batches_tracked += 1
                    if (num_batches_tracked % 100) == 0:
                        features = torch.cat(features)
                        targets = torch.cat(targets)

                        torch.save(
                            features,
                            os.path.join(
                                feature_dir, f'features{save_str}_cache_{num_cached}.pt'
                            ),
                        )
                        torch.save(
                            targets,
                            os.path.join(
                                feature_dir, f'targets{save_str}_cache_{num_cached}.pt'
                            ),
                        )
                        num_cached += 1
                        features = []
                        targets = []

            if len(features) > 0:
                features = torch.cat(features)
                targets = torch.cat(targets)
                torch.save(
                    features,
                    os.path.join(
                        feature_dir, f'features{save_str}_cache_{num_cached}.pt'
                    ),
                )
                torch.save(
                    targets,
                    os.path.join(
                        feature_dir, f'targets{save_str}_cache_{num_cached}.pt'
                    ),
                )
                num_cached += 1

            features = torch.load(
                os.path.join(feature_dir, f'features{save_str}_cache_0.pt')
            )
            targets = torch.load(
                os.path.join(feature_dir, f'targets{save_str}_cache_0.pt')
            )

            for k in range(1, num_cached):
                next_features = torch.load(
                    os.path.join(feature_dir, f'features{save_str}_cache_{k}.pt')
                )
                next_targets = torch.load(
                    os.path.join(feature_dir, f'targets{save_str}_cache_{k}.pt')
                )
                features = torch.cat((features, next_features))
                targets = torch.cat((targets, next_targets))

            for k in range(num_cached):
                os.remove(os.path.join(feature_dir, f'features{save_str}_cache_{k}.pt'))
                os.remove(os.path.join(feature_dir, f'targets{save_str}_cache_{k}.pt'))

            torch.save(features, os.path.join(feature_dir, f'features{save_str}.pt'))
            torch.save(targets, os.path.join(feature_dir, f'targets{save_str}.pt'))

    features = torch.load(os.path.join(feature_dir, 'features_train.pt'))
    targets = torch.load(os.path.join(feature_dir, 'targets_train.pt'))

    # second, make a dataloader with k features per class. if k = -1, use all features.
    length = len(features)
    perm = [p.item() for p in torch.randperm(length)]
    idxs = []
    counts = {}
    num_classes = 0

    for p in perm:
        target = targets[p].item()
        if target not in counts:
            counts[target] = 0
            num_classes += 1

        if fewshot_k < 0 or counts[target] < fewshot_k:
            counts[target] += 1
            idxs.append(p)

    for c in counts:
        if 0 < fewshot_k != counts[c]:
            print('Insufficient data for this eval')
            return

    train_features = features[idxs]
    train_labels = targets[idxs]

    feature_val_loader = None
    feature_train_val_loader = None
    if val_dataloader is not None:
        features_val = torch.load(os.path.join(feature_dir, 'features_val.pt'))
        targets_val = torch.load(os.path.join(feature_dir, 'targets_val.pt'))
        feature_val_dset = FeatureDataset(features_val, targets_val)
        feature_val_loader = DataLoader(
            feature_val_dset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        feature_train_val_dset = FeatureDataset(
            np.concatenate((train_features, features_val)),
            np.concatenate((train_labels, targets_val)),
        )
        feature_train_val_loader = DataLoader(
            feature_train_val_dset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    feature_train_dset = FeatureDataset(train_features, train_labels)
    feature_train_loader = DataLoader(
        feature_train_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    features_test = torch.load(os.path.join(feature_dir, 'features_test.pt'))
    targets_test = torch.load(os.path.join(feature_dir, 'targets_test.pt'))

    feature_test_dset = FeatureDataset(features_test, targets_test)
    feature_test_loader = DataLoader(
        feature_test_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    input_shape, output_shape = features[0].shape[0], targets.max().item() + 1

    if val_dataloader is not None:
        # perform openAI-like hyperparameter sweep
        # https://arxiv.org/pdf/2103.00020.pdf A.3
        # instead of scikit-learn LBFGS use FCNNs with AdamW
        wd_list = np.logspace(-6, 2, num=97).tolist()
        wd_list_init = np.logspace(-6, 2, num=7).tolist()
        wd_init_idx = [i for i, val in enumerate(wd_list) if val in wd_list_init]
        peak_idx = _find_peak(
            wd_list,
            wd_init_idx,
            feature_train_loader,
            feature_val_loader,
            input_shape,
            output_shape,
            lr,
            epochs,
            autocast,
            device,
            verbose,
            seed,
        )
        step_span = 8
        while step_span > 0:
            left = max(peak_idx - step_span, 0)
            right = min(peak_idx + step_span, len(wd_list) - 1)
            peak_idx = _find_peak(
                wd_list,
                [left, peak_idx, right],
                feature_train_loader,
                feature_val_loader,
                input_shape,
                output_shape,
                lr,
                epochs,
                autocast,
                device,
                verbose,
                seed,
            )
            step_span //= 2

        best_wd = wd_list[peak_idx]
        train_loader = feature_train_val_loader

    else:
        best_wd = 0
        train_loader = feature_train_loader

    final_model = _train(
        train_loader,
        input_shape,
        output_shape,
        best_wd,
        lr,
        epochs,
        autocast,
        seed,
    )
    logits, target = _infer(final_model, feature_test_loader, autocast, device)
    pred = logits.argmax(axis=1)

    # measure accuracy
    if target.max() >= 5:
        acc1, acc5 = accuracy(logits.float(), target.float(), topk=(1, 5))
    else:
        (acc1,) = accuracy(logits.float(), target.float(), topk=(1,))
        acc5 = float('nan')

    mean_per_class_recall = balanced_accuracy_score(target, pred)

    fair_info = {
        'weight_decay': best_wd,
        'acc1': acc1,
        'acc5': acc5,
        'mean_per_class_recall': mean_per_class_recall,
        'classification_report': classification_report(target, pred, digits=3),
    }
    if verbose:
        print(fair_info['classification_report'])
        print(f'Test acc1: {acc1} with weight_decay: {best_wd}')

    return {
        'lp_acc1': fair_info['acc1'],
        'lp_acc5': fair_info['acc5'],
        'lp_mean_per_class_recall': fair_info['mean_per_class_recall'],
        'weight_decay': fair_info['weight_decay'],
        'epochs': epochs,
        'seed': seed,
        'fewshot_k': fewshot_k,
        'normalized': normalize,
    }
