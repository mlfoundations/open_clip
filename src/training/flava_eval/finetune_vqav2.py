import argparse
import os
import random
import re
import sys

import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import ViltConfig

import open_clip
from open_clip.factory import get_tokenizer
from training.data import get_imagenet
from training.scheduler import cosine_lr
from training.train import AverageMeter
from training.zero_shot import zero_shot_eval
from training.precision import get_autocast

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
        "--epochs", type=int, default=1000, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
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
        "--early-stop-metric-name", type=str, default="accuracy", help="Early stopping metric name."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp_bfloat16",
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


class VQAEval(object):
    """VQA evaluation."""

    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                    "youll": "you'll", "youre": "you're", "youve": "you've"}
    manualMap = {
        'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    articles = ['a', 'an', 'the']
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [';', r"/", '[', ']', '"', '{', '}',
            '(', ')', '=', '+', '\\', '_', '-',
            '>', '<', '@', '`', ',', '?', '!']

    def __init__(self):
        self.acc_qa = []

    def process_punctuation(self, text):
        out_text = text
        for p in self.punct:
            if (p + ' ' in text or ' ' + p in text) or (re.search(self.commaStrip, text) != None):
                out_text = out_text.replace(p, '')
            else:
                out_text = out_text.replace(p, ' ')
        out_text = self.periodStrip.sub("", out_text, re.UNICODE)
        return out_text

    def process_article(self, text):
        out_text = []
        tmp_text = text.lower().split()
        for word in tmp_text:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.contractions:
                out_text[word_id] = self.contractions[word]
        out_text = ' '.join(out_text)
        return out_text

    def add_batch(self, *, answers, annotations):
        assert len(answers) == len(annotations)

        for i, annotation in enumerate(annotations):
            for ansDic in annotation["answers"]:
                ansDic['answer'] = ansDic['answer'].replace('\n', ' ')
                ansDic['answer'] = ansDic['answer'].replace('\t', ' ')
                ansDic['answer'] = ansDic['answer'].strip()
            resAns = answers[i]
            resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            gtAcc = []
            gtAnswers = [ans['answer'] for ans in annotation["answers"]]

            if len(set(gtAnswers)) > 1:
                for ansDic in annotation["answers"]:
                    ansDic['answer'] = self.process_punctuation(ansDic['answer'])
                    ansDic['answer'] = self.process_article(ansDic['answer'])
                resAns = self.process_punctuation(resAns)
                resAns = self.process_article(resAns)

            for gtAnsDatum in annotation["answers"]:
                otherGTAns = [item for item in annotation["answers"] if item!=gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item['answer']==resAns]
                acc = min(1, float(len(matchingAns))/3)
                gtAcc.append(acc)
            quesType    = annotation['question_type']
            ansType     = annotation['answer_type']
            avgGTAcc = float(sum(gtAcc))/len(gtAcc)
            self.acc_qa.append(avgGTAcc)
            # if quesType not in accQuesType:
            #     accQuesType[quesType] = []
            # accQuesType[quesType].append(avgGTAcc)
            # if ansType not in accAnsType:
            #     accAnsType[ansType] = []
            # accAnsType[ansType].append(avgGTAcc)
            # self.setEvalQA(quesId, avgGTAcc)
            # self.setEvalQuesType(quesId, quesType, avgGTAcc)
            # self.setEvalAnsType(quesId, ansType, avgGTAcc)
            # if step%100 == 0:
            #     self.updateProgress(step/float(len(quesIds)))
            # step = step + 1

    def compute(self):
        acc = round(100*float(sum(self.acc_qa))/len(self.acc_qa), 2)
        self.acc_qa = []
        return {'accuracy': acc}


class VQAv2Dataset(Dataset):
    """VQAv2 dataset."""

    def __init__(self, split, label2id, id2label, transforms, tokenizer=None):
        super().__init__()

        self.df = load_dataset("HuggingFaceM4/VQAv2", split=split)
        self.length = len(self.df)
        self.label2id = label2id
        self.id2label = id2label
        self.transforms = transforms
        self.tokenize = tokenizer

    def __len__(self):
        return self.length

    def _get_score(self, count):
        return min(1.0, count / 3)

    def __getitem__(self, idx):
        item = self.df[idx]
        image = self.transforms(item["image"])
        text = self.tokenize([item["question"]])[0]

        answers = item["answers"]
        answer_count = {}
        for answer in answers:
            answer_ = answer["answer"]
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels = []
        scores = []
        for answer in answer_count:
            if answer not in self.label2id: continue
            labels.append(self.label2id[answer])
            score = self._get_score(answer_count[answer])
            scores.append(score)

        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(self.label2id))
        for label, score in zip(labels, scores):
              targets[label] = score

        item.pop("image")

        return {
            'image': image,
            'text': text,
            'label': targets,
            'annotation': item,
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


def vqa_collate(example_list):
    images = []
    texts = []
    labels = []
    annotations = []
    for example in example_list:
        images.append(example["image"])
        texts.append(example["text"])
        labels.append(example["label"])
        annotations.append(example["annotation"])
    images = torch.stack(images)
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    return {
        'image': images,
        'text': texts,
        'label': labels,
        'annotation': annotations,
    }


def get_task_dataloaders(id_mappings, transforms, args):
    tokenizer = get_tokenizer(args.model)
    label2id, id2label = id_mappings

    dataloaders = {}
    for split_name in ["train", "validation", "test"]:
        is_train = (split_name == "train")
        dataset = VQAv2Dataset(
            split_name,
            label2id,
            id2label,
            transforms,
            tokenizer=tokenizer,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=is_train,
            num_workers=args.workers,
            collate_fn=vqa_collate,
            drop_last=is_train,
        )
        dataloaders[split_name] = dataloader

    return dataloaders


def compute_metrics(model, dataloader, device, args):
    model.eval()
    metric = VQAEval()
    id2label = dataloader.dataset.id2label
    samples_seen = 0
    step = 0
    val_batches = 20
    for batch in tqdm(dataloader, total=val_batches):
        if step == val_batches: break
        with torch.no_grad():
            image = batch["image"].to(device)
            text = batch["text"].to(device)
            logits = model(image, text)
        samples_seen += text.shape[0]
        step += 1
        answer_ids = logits.argmax(dim=-1).cpu().numpy()
        pred_answers = [id2label[aid] for aid in answer_ids]
        gt_annotations = batch["annotation"]
        metric.add_batch(
            answers=pred_answers,
            annotations=gt_annotations,
        )
    model.train()
    metrics = metric.compute()
    return metrics


def train_one_epoch(model, data, epoch, optimizer, scheduler, early_stop, device, args):
    autocast = get_autocast(args.precision)
    model.train()
    progress_bar = tqdm(total=len(data["train"]))
    for i, batch in enumerate(data["train"]):
        step = epoch * len(data["train"]) + i
        scheduler(step)
        with autocast():
            image = batch["image"].to(device)
            text = batch["text"].to(device)
            label = batch["label"].to(device)
            logits = model(image, text)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, label) * label.shape[1]

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

    config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    label2id = config.label2id
    id2label = config.id2label
    data = get_task_dataloaders((label2id, id2label), preprocess_val, args)
    clf_cls = FLAVAMultimodalClassifier if args.model.startswith("flava") else CLIPMultimodalClassifier
    clf = clf_cls(model, embed_dim, len(label2id)).to(device)
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
