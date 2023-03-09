import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import open_clip
from open_clip.factory import get_tokenizer
from training.data import get_imagenet
from training.scheduler import cosine_lr
from training.train import AverageMeter
from training.zero_shot import zero_shot_eval

try:
    import evaluate
except ImportError:
    raise ImportError("Please install HF evaluate: pip install evaluate")


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hateful-memes",
        type=str,
        default=None,
        help="Path to Hateful Memes dataset directory.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=2000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=30, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=5, help="Early stopping patience."
    )
    parser.add_argument(
        "--early-stop-threshold", type=float, default=0.0, help="Early stopping threshold."
    )
    parser.add_argument(
        "--early-stop-metric-name", type=str, default="roc_auc", help="Early stopping metric name."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )

    args = parser.parse_args(args)
    return args


class EarlyStopping:

    def __init__(self, patience=5, threshold=0.0, metric_name="roc_auc"):
        self.patience = patience
        self.threshold = threshold
        self.patience_counter = 0
        self.best_score = None
        self.best_metrics = None
        self.metric_name = metric_name

    def step(self, metrics):
        score = metrics[self.metric_name]
        if self.best_score is None:
            self.best_score = score
            self.best_metrics = metrics
        elif score < self.best_score + self.threshold:
            self.patience_counter += 1
        else:
            self.best_score = score
            self.best_metrics = metrics
            self.patience_counter = 0

        if self.patience_counter >= self.patience:
            return True
        return False


class HatefulMemesDataset(Dataset):
    """Hateful Memes dataset."""

    def __init__(self, path, split, transforms, tokenizer=None):
        super().__init__()

        if split == "train":
            jsonl = os.path.join(path, "train.jsonl")
        elif split == "validation":
            jsonl = os.path.join(path, "dev_unseen.jsonl")
        elif split == "test":
            jsonl = os.path.join(path, "test_unseen.jsonl")
        else:
            raise ValueError(f"Invalid split {split}")

        self.path = path
        self.df = pd.read_json(jsonl, lines=True)
        self.length = len(self.df)
        self.transforms = transforms
        self.tokenize = tokenizer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_path = os.path.join(self.path, item["img"])
        image = self.transforms(Image.open(str(img_path)))
        text = item["text"]
        label = item["label"]
        return {
            'image': image,
            'text': self.tokenize([text])[0],
            'label': label
        }


class FLAVAMultimodalClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, num_labels):
        super().__init__()

        self.encoder = encoder
        self.logits_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, num_labels),
        )

    def forward(self, image, text):
        multimodal_features = self.encoder.encode_multimodal(image, text)
        logits = self.logits_proj(multimodal_features)
        return logits


class CLIPMultimodalClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, num_labels):
        super().__init__()

        self.encoder = encoder
        self.logits_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, num_labels),
        )

    def forward(self, image, text):
        # CLIP doesn't have a multimodal encoder, so we concatenate the features
        text_features = self.encoder.encode_text(text)
        image_features = self.encoder.encode_image(image)
        multimodal_features = torch.cat([image_features, text_features], dim=-1)
        logits = self.logits_proj(multimodal_features)
        return logits


def get_task_dataloaders(transforms, args):
    tokenizer = get_tokenizer(args.model)
    hm_path = os.path.expanduser(args.hateful_memes)

    dataloaders = {}
    for split_name in ["train", "validation", "test"]:
        is_train = (split_name == "train")
        dataset = HatefulMemesDataset(
            hm_path,
            split_name,
            transforms,
            tokenizer=tokenizer,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=is_train,
            num_workers=args.workers,
            drop_last=is_train,
        )
        dataloaders[split_name] = dataloader

    return dataloaders


def compute_metrics(model, dataloader, device, args):
    model.eval()
    metric = evaluate.load("roc_auc")
    val_loss = 0
    samples_seen = 0
    for batch in dataloader:
        with torch.no_grad():
            image = batch["image"].to(device)
            text = batch["text"].to(device)
            label = batch["label"].to(device)
            samples_seen += text.shape[0]
            logits = model(image, text)
            logits = logits.view(-1)
            label = label.view(-1).float()
            pred_scores = torch.sigmoid(logits)
            batch_val_loss = nn.functional.binary_cross_entropy_with_logits(logits, label, reduction='sum')
        val_loss += batch_val_loss.item()
        metric.add_batch(
            prediction_scores=pred_scores.cpu().numpy(),
            references=label.cpu().numpy(),
        )
    model.train()
    metrics = metric.compute()
    metrics["loss"] = val_loss / samples_seen
    return metrics


def train_one_epoch(model, data, epoch, optimizer, scheduler, early_stop, device, args):
    model.train()
    progress_bar = tqdm(total=len(data["train"]))
    for i, batch in enumerate(data["train"]):
        step = epoch * len(data["train"]) + i
        scheduler(step)

        image = batch["image"].to(device)
        text = batch["text"].to(device)
        label = batch["label"].to(device)

        logits = model(image, text)
        logits = logits.view(-1)
        label = label.view(-1).float()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        progress_bar.update(1)

        if (i % args.val_frequency) == 0 and i > 0:
            metrics = compute_metrics(model, data["validation"], device, args)
            print(metrics)
            end_training = early_stop.step(metrics)
            if end_training:
                progress_bar.close()
                return metrics, end_training

    progress_bar.close()
    metrics = compute_metrics(model, data["validation"], device, args)
    print(metrics)
    end_training = early_stop.step(metrics)
    return metrics, end_training


def main(args):
    args = parse_args(args)
    random_seed(args.seed, 0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, preprocess_train, preprocess_val = open_clip.factory.create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        pretrained_hf=False,
    )
    model_cfg = open_clip.factory.get_model_config(args.model)
    embed_dim = model_cfg["embed_dim"]

    data = get_task_dataloaders(preprocess_val, args)
    clf_cls = FLAVAMultimodalClassifier if args.model.startswith("flava") else CLIPMultimodalClassifier
    clf = clf_cls(model, embed_dim, 1).to(device)
    optim = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.wd)

    total_steps = len(data["train"]) * args.epochs
    scheduler = cosine_lr(optim, args.lr, args.warmup, total_steps)
    early_stop = EarlyStopping(  # greater metric value is better
        patience=args.early_stop_patience,
        threshold=args.early_stop_threshold,
        metric_name=args.early_stop_metric_name,
    )

    for epoch in range(args.epochs):
        val_metrics, end_training = train_one_epoch(clf, data, epoch, optim, scheduler, early_stop, device, args)
        if end_training:
            print("Stopped early to prevent overfitting.")
            break

    print(early_stop.best_metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
