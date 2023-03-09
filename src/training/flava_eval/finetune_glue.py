import argparse
import os
import sys

import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import open_clip
from open_clip.factory import get_tokenizer
from training.scheduler import cosine_lr
from training.train import AverageMeter

try:
    import evaluate
except ImportError:
    raise ImportError("Please install HF evaluate: pip install evaluate")


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="GLUE task name.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=1000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=100, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=5, help="Early stopping patience."
    )
    parser.add_argument(
        "--early-stop-threshold", type=float, default=0.0, help="Early stopping threshold."
    )
    parser.add_argument(
        "--early-stop-metric-name", type=str, default="accuracy", help="Early stopping metric name."
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

    def __init__(self, patience=5, threshold=0.0, metric_name="accuracy"):
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


class GLUEDataset(Dataset):
    """GLUE dataset."""

    def __init__(self, task, split, text_key, label_key, tokenizer=None):
        super().__init__()

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install HF datasets: pip install datasets")

        self.dataset = load_dataset("glue", task, split=split)
        self.label_key = label_key
        self.text_key = text_key
        self.length = len(self.dataset)
        self.tokenize = tokenizer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[self.text_key]
        label = item[self.label_key]
        return {
            'text': self.tokenize([text])[0],
            'label': label
        }


class TextClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, num_labels):
        super().__init__()

        self.encoder = encoder
        self.logits_proj = nn.Linear(embed_dim, num_labels)

    def forward(self, text):
        text_features = self.encoder.encode_text(text)
        logits = self.logits_proj(text_features)
        return logits


def get_task_metric(task_name):
    if task_name == "cola":
        metric = evaluate.load("glue", "stsb")
    else:
        metric = evaluate.load("glue", task_name)
    return metric


def get_task_dataloaders(args):
    tokenizer = get_tokenizer(args.model)
    task_name = args.task_name

    dataloaders = {}
    for split_name in ["train", "validation", "test"]:
        dataset = GLUEDataset(
            task_name,
            split_name,
            text_key="sentence",
            label_key="label",
            tokenizer=tokenizer,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        dataloaders[split_name] = dataloader

    return dataloaders


def compute_metrics(model, dataloader, device, args):
    model.eval()
    metric = get_task_metric(args.task_name)
    val_loss = 0
    samples_seen = 0
    for batch in dataloader:
        with torch.no_grad():
            text = batch["text"].to(device)
            label = batch["label"].to(device)
            samples_seen += text.shape[0]
            logits = model(text)
            logits = logits.view(-1)
            label = label.view(-1).float()
            predictions = torch.sigmoid(logits) > 0.5
            batch_val_loss = nn.functional.binary_cross_entropy_with_logits(logits, label, reduction='sum')
        val_loss += batch_val_loss.item()
        metric.add_batch(
            predictions=predictions.cpu().numpy(),
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

        text = batch["text"].to(device)
        label = batch["label"].to(device)

        logits = model(text)
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
            end_training = early_stop.step(metrics)
            if end_training:
                progress_bar.close()
                return metrics, end_training

    progress_bar.close()
    metrics = compute_metrics(model, data["validation"], device, args)
    end_training = early_stop.step(metrics)
    return metrics, end_training


def main(args):
    args = parse_args(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, preprocess_train, preprocess_val = open_clip.factory.create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
    )
    model_cfg = open_clip.factory.get_model_config(args.model)
    embed_dim = model_cfg["embed_dim"]

    data = get_task_dataloaders(args)
    clf = TextClassifier(model, embed_dim, 1).to(device)
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
