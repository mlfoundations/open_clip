import argparse

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import open_clip
from open_clip.factory import get_tokenizer
from training.scheduler import cosine_lr
from training.train import AverageMeter
import evaluate
from sklearn import preprocessing
import numpy as np
import sys

from datasets import load_dataset_builder
from datasets import load_dataset

class VQATextDataset(Dataset):
    def __init__(self, df, split, transforms, label_encoder, answer_set, tokenizer=None):
        self.df = df
        self.transforms = transforms
        self.tokenize = tokenizer
        self.num_classes = len(answer_set)
        self.label_encoder = label_encoder
        self.answer_set = answer_set
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_path = item["image"]["path"]
        image = Image.open(str(img_path))
        text = item["question"]

        answer_count = {}
        for answer in item['answers']:
            answer_ = answer["answer"]
            answer_count[answer_] = answer_count.get(answer_, 0) + 1
        labels = []
        scores = []
        for answer in answer_count:
            if answer not in self.answer_set:
                continue
            labels.append(self.label_encoder.transform([answer])[0])
            score = get_score(answer_count[answer])
            scores.append(score)
        target = np.zeros(self.num_classes)
        for label, score in zip(labels, scores):
            target[label] = score
        
        return {
            'image': self.transforms(image),
            'text': self.tokenize([text])[0],
            'target': torch.tensor(target)
        }

def get_score(count: int) -> float:
    return min(1.0, count / 3.0)

def get_task_dataloaders(path, transforms, labelencoder, answer_set, args):
    tokenizer = get_tokenizer(args.model)
    dataloaders = {}
    
    for split in ["train", "validation"]:
        dataset_train = load_dataset(path, split=split, cache_dir = "./vqa_data")
        dataset_df = dataset_train.to_pandas()
        dataset_df = dataset_df[dataset_df.apply(lambda item: True if item['multiple_choice_answer'] in answer_set else False, axis=1)]
        
        b_size = args.batch_size
        if(split == "validation"):
            b_size = args.batch_size * 10
            dataset_df = dataset_df[0:12800]
        dataset = VQATextDataset(dataset_df,
            split,
            transforms, 
            labelencoder,   
            answer_set,       
            tokenizer=tokenizer,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=b_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        dataloaders[split] = dataloader

    return dataloaders

class CLIPMultimodalClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, num_labels):
        super().__init__()

        self.encoder = encoder
        self.layers = nn.Sequential(
            nn.Linear(embed_dim * 2, 1536), #size of answer space
            nn.ReLU(inplace=True),
            nn.LayerNorm(1536),
            nn.Linear(1536, num_labels)
        )
        
    def forward(self, image, text):
        # CLIP doesn't have a multimodal encoder, so we concatenate the features
        text_features = self.encoder.encode_text(text)
        image_features = self.encoder.encode_image(image)
        multimodal_features = torch.cat([image_features, text_features], dim=-1)
        logits = self.layers(multimodal_features)
        return logits

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

def compute_metrics(model, dataloader, device, args):
    model.eval()
    metric = evaluate.load("accuracy")
    val_loss = 0
    samples_seen = 0
    total_correct = 0

    for batch in dataloader:
        with torch.no_grad():
            image = batch["image"].to(device)
            text = batch["text"].to(device)
            label = batch["target"].to(device)
            samples_seen += text.shape[0]
            logits = model(image, text)
            predictions = torch.argmax(logits, dim=-1)
            batch_val_loss = nn.functional.binary_cross_entropy_with_logits(logits, label, reduction="mean")
        predictions=predictions.cpu().numpy()
        references= label.cpu().numpy()
        val_loss += batch_val_loss.item()
        for i, pred in enumerate(predictions):
            total_correct += references[i][pred]
    #print("total correct", total_correct, "seen", samples_seen)

    model.train()
    metrics = {}
    metrics["accuracy"] = total_correct/ samples_seen
    metrics["loss"] = val_loss / samples_seen
    
    return metrics

#Remove in final commit
def train_single_epoch(model, data, optimizer, args):
    model.train()
    for i, batch in enumerate(data["train"]):
        image = batch["image"].to(device)
        text = batch["text"].to(device)
        label = batch["target"].to(device)
        
        logits = model(image, text)
        print(label.shape)
        print(logits.shape)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, label, reduction="mean")
        print(loss)
        loss.backward()

        
def train_one_epoch(model, data, epoch, optimizer, scheduler, early_stop, device, args):
    model.train()
    progress_bar = tqdm(total=len(data["train"]))
    for i, batch in enumerate(data["train"]):
        step = epoch * len(data["train"]) + i
        scheduler(step)
        
        image = batch["image"].to(device)
        text = batch["text"].to(device)
        label = batch["target"].to(device)
        logits = model(image, text)

        loss = nn.functional.binary_cross_entropy_with_logits(logits, label, reduction = "mean") #should be cross entropy 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        progress_bar.update(1)
        
        if (i % args.val_frequency) == 0 and i > 5:
            print(loss)
            metrics = compute_metrics(model, data["validation"], device, args)
            print("accuracy", metrics["accuracy"])
            end_training = early_stop.step(metrics)
            #if end_training:
            #    progress_bar.close()
            #    return metrics, end_training

    progress_bar.close()
    metrics = compute_metrics(model, data["validation"], device, args)
    end_training = early_stop.step(metrics)
    return metrics, end_training

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hateful-memes",
        type=str,
        default=None,
        help="Path to Hateful Memes dataset directory.",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=200, help="Number of steps to warmup for."
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
        default="ViT-B-32-quickgelu",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='laion400m_e32',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )

    args = parser.parse_args(args)
    return args

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
    
    answer_space = []
    with open('src/training/answers_vqa.txt') as f:
        for line in f:
          answer_space.append(line.strip())
    answer_space = np.array(answer_space)
    
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(answer_space)
    num_classes = len(list(labelencoder.classes_))

    answer_set = set(labelencoder.classes_)
    
    data = get_task_dataloaders("HuggingFaceM4/VQAv2", preprocess_val, labelencoder, answer_set, args)

    clf_cls = CLIPMultimodalClassifier
    clf = clf_cls(model, embed_dim, num_classes).to(device)
    optim = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.wd)

    total_steps = len(data["train"]) * args.epochs
    scheduler = cosine_lr(optim, args.lr, args.warmup, total_steps)
    early_stop = EarlyStopping(  # greater metric value is better
        patience=args.early_stop_patience,
        threshold=args.early_stop_threshold,
        metric_name=args.early_stop_metric_name,
    )

    for epoch in range(20):
        val_metrics, end_training = train_one_epoch(clf, data, epoch, optim, scheduler, early_stop, device, args)

if __name__ == "__main__":
    main(sys.argv[1:])
