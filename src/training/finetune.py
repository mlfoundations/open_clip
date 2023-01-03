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
        "--workers", type=int, default=1, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=1000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=50, help="How often to run evaluation with val data."
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


class SequenceClassifier(nn.Module):
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
    import pdb; pdb.set_trace()
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

    metrics = metric.compute()
    metrics["loss"] = val_loss / samples_seen
    return metrics


def train_one_epoch(model, data, epoch, optimizer, scheduler, device, args):
    model.train()
    progress_bar = tqdm(data["train"])
    for step, batch in enumerate(progress_bar):
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

        if (step % args.val_frequency) == 0 and step > 0:
            eval_metrics = compute_metrics(model, data["validation"], device, args)
            model.train()
            # TODO: add early stopping

    eval_metrics = compute_metrics(model, data["validation"], device, args)
    print(eval_metrics)


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
    clf = SequenceClassifier(model, embed_dim, 1).to(device)
    optim = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epochs):
        scheduler = None
        train_one_epoch(clf, data, epoch, optim, scheduler, device, args)

    model.eval()
    import pdb; pdb.set_trace()
    test_metrics = compute_metrics(clf, data["test"], device, args)
    print(test_metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
