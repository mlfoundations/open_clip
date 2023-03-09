import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
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


class WinogroundDataset(Dataset):
    """Winoground dataset."""

    def __init__(self, split, transforms, tokenizer=None):
        super().__init__()

        self.df = load_dataset("facebook/winoground", split=split, use_auth_token=True)
        self.length = len(self.df)
        self.transforms = transforms
        self.tokenize = tokenizer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.df[idx]
        image_0 = self.transforms(item['image_0'].convert("RGB"))
        image_1 = self.transforms(item['image_1'].convert("RGB"))
        caption_0 = self.tokenize([item['caption_0']])[0]
        caption_1 = self.tokenize([item['caption_1']])[0]
        return {
            'image_0': image_0,
            'image_1': image_1,
            'caption_0': caption_0,
            'caption_1': caption_1,
        }


def get_task_dataloaders(transforms, args):
    tokenizer = get_tokenizer(args.model)
    dataloaders = {}
    for split_name in ["test"]:
        dataset = WinogroundDataset(
            split_name,
            transforms,
            tokenizer=tokenizer,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=False,
        )
        dataloaders[split_name] = dataloader

    return dataloaders


def run(model, dataloader, device, args):
    model.eval()

    contrastive_text_accuracy = 0.
    contrastive_image_accuracy = 0.
    contrastive_group_accuracy = 0.
    itm_text_accuracy = 0.
    itm_image_accuracy = 0.
    itm_group_accuracy = 0.
    n = 0.

    with torch.no_grad():
        for batch in tqdm(dataloader, unit_scale=args.batch_size):
            i0 = batch['image_0'].to(device)
            c0 = batch['caption_0'].to(device)
            i1 = batch['image_1'].to(device)
            c1 = batch['caption_1'].to(device)

            # compute contrastive scores
            i0_feat = model.encode_image(i0, normalize=True)
            i1_feat = model.encode_image(i1, normalize=True)
            c0_feat = model.encode_text(c0, normalize=True)
            c1_feat = model.encode_text(c1, normalize=True)

            contrastive_score_c0_i0 = (100 * i0_feat[:, None, :] @ c0_feat[:, :, None]).view(-1)
            contrastive_score_c1_i0 = (100 * i0_feat[:, None, :] @ c1_feat[:, :, None]).view(-1)
            contrastive_score_c0_i1 = (100 * i1_feat[:, None, :] @ c0_feat[:, :, None]).view(-1)
            contrastive_score_c1_i1 = (100 * i1_feat[:, None, :] @ c1_feat[:, :, None]).view(-1)

            contrastive_text_correct = torch.logical_and(contrastive_score_c0_i0 > contrastive_score_c1_i0,
                                                         contrastive_score_c1_i1 > contrastive_score_c0_i1)
            contrastive_image_correct = torch.logical_and(contrastive_score_c0_i0 > contrastive_score_c0_i1,
                                                          contrastive_score_c1_i1 > contrastive_score_c1_i0)
            contrastive_group_correct = torch.logical_and(contrastive_text_correct, contrastive_image_correct)

            contrastive_text_accuracy += contrastive_text_correct.sum()
            contrastive_image_accuracy += contrastive_image_correct.sum()
            contrastive_group_accuracy += contrastive_group_correct.sum()

            # compute ITM scores (for FLAVA)
            if args.model.startswith('flava'):
                itm_score_c0_i0 = torch.sigmoid(model.forward_itm(i0, c0))
                itm_score_c1_i0 = torch.sigmoid(model.forward_itm(i0, c1))
                itm_score_c0_i1 = torch.sigmoid(model.forward_itm(i1, c0))
                itm_score_c1_i1 = torch.sigmoid(model.forward_itm(i1, c1))

                itm_text_correct = torch.logical_and(itm_score_c0_i0 > itm_score_c1_i0,
                                                     itm_score_c1_i1 > itm_score_c0_i1)
                itm_image_correct = torch.logical_and(itm_score_c0_i0 > itm_score_c0_i1,
                                                      itm_score_c1_i1 > itm_score_c1_i0)
                itm_group_correct = torch.logical_and(itm_text_correct, itm_image_correct)

                itm_text_accuracy += itm_text_correct.sum()
                itm_image_accuracy += itm_image_correct.sum()
                itm_group_accuracy += itm_group_correct.sum()

            n += i0.size(0)

    contrastive_text_accuracy /= n
    contrastive_image_accuracy /= n
    contrastive_group_accuracy /= n
    itm_text_accuracy /= n
    itm_image_accuracy /= n
    itm_group_accuracy /= n
    return contrastive_text_accuracy.item(), \
           contrastive_image_accuracy.item(), \
           contrastive_group_accuracy.item(), \
           itm_text_accuracy.item(), \
           itm_image_accuracy.item(), \
           itm_group_accuracy.item()


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

    contrastive_text_accuracy, \
    contrastive_image_accuracy, \
    contrastive_group_accuracy, \
    itm_text_accuracy, \
    itm_image_accuracy, \
    itm_group_accuracy = run(model, data["test"], device, args)
    print(f'Contrastive Text Accuracy: {contrastive_text_accuracy:.6f}')
    print(f'Contrastive Image Accuracy: {contrastive_image_accuracy:.6f}')
    print(f'Contrastive Group Accuracy: {contrastive_group_accuracy:.6f}')
    print(f'ITM Text Accuracy: {itm_text_accuracy:.6f}')
    print(f'ITM Image Accuracy: {itm_image_accuracy:.6f}')
    print(f'ITM Group Accuracy: {itm_group_accuracy:.6f}')

if __name__ == "__main__":
    main(sys.argv[1:])
