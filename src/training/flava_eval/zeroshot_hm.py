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


def zero_shot_classifier(model, device, args):
    tokenizer = get_tokenizer(args.model)
    prompts = [
        'a meme.',
        'a hatespeech meme.'
    ]
    texts = tokenizer(prompts).to(device)
    class_embeddings = model.encode_text(texts, normalize=True)

    def image_clf(x):
        image_features = model.encode_image(x, normalize=True)
        logits = 100. * image_features @ class_embeddings.T
        return logits

    return image_clf


def run(model, classifier, dataloader, device, args):
    model.eval()
    metric = evaluate.load("roc_auc")
    accuracy, n = 0., 0.

    with torch.no_grad():
        for batch in tqdm(dataloader, unit_scale=args.batch_size):
            images = batch['image'].to(device)
            target = batch['label'].to(device)
            logits = classifier(images)

            # update ROC AUC
            pred_hateful = torch.softmax(logits, dim=-1)[:, 1]
            metric.add_batch(
                prediction_scores=pred_hateful.cpu().numpy(),
                references=target.cpu().numpy(),
            )

            # update accuracy
            pred_cmp_label = logits[:, 0] < logits[:, 1]
            accuracy += (pred_cmp_label == target).sum()
            n += images.size(0)

    metrics = metric.compute()
    roc_auc = metrics["roc_auc"]
    accuracy /= n
    return roc_auc, accuracy.item()


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
    data = get_task_dataloaders(preprocess_val, args)

    classifier = zero_shot_classifier(model, device, args)
    roc_auc, accuracy = run(model, classifier, data["test"], device, args)
    print(f'ROC AUC: {roc_auc:.6f}\nAccuracy: {accuracy:.6f}')


if __name__ == "__main__":
    main(sys.argv[1:])
