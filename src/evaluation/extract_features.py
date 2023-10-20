
"""
Do quantitative trait-level evaluation of different CLIP models on CUB-2011 dataset.
"""
from argparse import ArgumentParser

import datetime
import logging
import math
import os
import re
import sys
import warnings
import json

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from imageomics import cub2011
from open_clip import (
    create_model_and_transforms,
    get_cast_dtype,
    get_tokenizer,
    trace_model,
)
from training.imagenet_zeroshot_data import openai_imagenet_template
from training.logger import setup_logging
from training.precision import get_autocast
from imageomics.naming import iNat21NameLookup, Taxon

from .params import parse_args
from .utils import init_device, random_seed

def extract_image_features(model, dloader, device, class_embeddings=None):
    feats = None
    lbls = None
    for img, lbl in tqdm(dloader, desc="Extracting Features"):
        img = img.to(device)
        lbl = lbl.cpu().detach().numpy()
        f = model.encode_image(img)
        f = F.normalize(f, dim=-1)
        if class_embeddings is not None:
            f = 100.0 * f @ class_embeddings
        f = f.cpu().detach().numpy()
        if feats is None:
            feats = f
            lbls = lbl
        else:
            feats = np.concatenate((feats, f), axis=0)
            lbls = np.concatenate((lbls, lbl), axis=0)

    return feats, lbls

def _load_model(args):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem/uri use
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
                "zero_shot_iid",
            ]
        )

    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    os.makedirs(log_base_path, exist_ok=True)
    log_filename = f"out-{args.rank}" if args.log_local else "out.log"
    args.log_path = os.path.join(log_base_path, log_filename)

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup checkpoint logging
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    os.makedirs(args.checkpoint_path, exist_ok=True)

    if (
        isinstance(args.force_image_size, (tuple, list))
        and len(args.force_image_size) == 1
    ):
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)
    model, _, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained, # This is the pretrained weights
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=None,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
    )

    return model, device, preprocess_val

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_text_embeddings_batched(tokenizer, in_texts, model, device, text_batch_size=1, tmp_save_loc="tmp/"):
    os.makedirs(tmp_save_loc, exist_ok=True)
    class_embeddings = []
    text_templates = []
    for txt in tqdm(in_texts, desc="gathering templates"):
        text_templates.extend([template(txt) for template in openai_imagenet_template])

    num_templates = len(openai_imagenet_template)
    batch_size = num_templates * text_batch_size
    num_batches = math.ceil(len(text_templates) / batch_size)
    for i in tqdm(range(num_batches), desc="gathering text embeddings"):
        if (i+1) <= 9000: continue
        s = i * batch_size
        e = min((i+1) * batch_size, len(text_templates)+1)
        batch = text_templates[s:e]
        texts = tokenizer(batch).to(device)  # tokenize
        class_embedding = model.encode_text(texts)
        class_embedding = F.normalize(class_embedding, dim=-1)
        num_feats = class_embedding.shape[-1]
        class_embedding = class_embedding.view(-1, num_templates, num_feats).mean(dim=1)
        norm = class_embedding.norm(dim=1, keepdim=True)
        class_embedding /= norm.repeat(1, class_embedding.shape[-1])
        class_embeddings.append(class_embedding.cpu().detach().numpy())
        if (i+1) % 1000 == 0:
            np.save(os.path.join(tmp_save_loc, f"{i}.npy"), np.concatenate(class_embeddings, axis=0))
            class_embeddings = []
            print(f"Saving text features at {i+1}/{num_batches}")
    class_embeddings = np.concatenate(class_embeddings, axis=0)

    return class_embeddings

def get_text_embeddings(tokenizer, in_texts, model, device):
    class_embeddings = []
    for txt in tqdm(in_texts, desc="gathering text embeddings"):
        texts = [template(txt) for template in openai_imagenet_template]
        texts = tokenizer(texts).to(device)  # tokenize
        class_embedding = model.encode_text(texts)
        class_embedding = F.normalize(class_embedding, dim=-1).mean(dim=0)
        class_embedding /= class_embedding.norm()
        class_embeddings.append(class_embedding)
    class_embeddings = torch.stack(class_embeddings, dim=1).to(device)

    return class_embeddings

def get_taxon_names(taxon_file_path):
    taxon_list = load_json(taxon_file_path).values()
    taxon_names = []
    for x in taxon_list:
        parts = x.split("_")
        if len(parts) != 7: continue
        taxon_names.append(Taxon(*parts).taxonomic)
    print(f"Total taxonomical names: {len(taxon_names)}")
    return taxon_names

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    parser = ArgumentParser()
    parser.add_argument("--exp_type", type=str, default='openai')
    parser.add_argument("--val_root", type=str, default="/local/scratch/cv_datasets/inat21/raw/val")
    parser.add_argument("--feature_output", type=str, default="/local/scratch/carlyn.1/clip_paper_bio/features")
    parser.add_argument("--extract_logits", action="store_true", default=False)
    parser.add_argument("--extract_text", action="store_true", default=False)
    parser.add_argument("--text_batch_size", type=int, default=1)
    parser.add_argument("--taxon_file", type=str, default="/local/scratch/carlyn.1/clip_paper_bio/taxons/taxon-merged.json")
    # Be sure to set the --pretrained [path to weights]

    args, _ = parser.parse_known_args(namespace=args)

    model, device, preprocess_val = _load_model(args)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    logit_suffix = "_logits" if args.extract_logits else ""
    if args.extract_text:
        logit_suffix = "_text" if args.extract_logits else ""
    data_save_path = os.path.join(args.feature_output, args.exp_type + "_features" + logit_suffix + ".npy")
    lbl_save_path = os.path.join(args.feature_output, args.exp_type + "_labels" + logit_suffix + ".npy")

    model.eval()
    if args.extract_text:
        taxon_names = get_taxon_names(args.taxon_file)

        with torch.no_grad():
            tokenizer = get_tokenizer(args.model)
            taxon_embeddings = get_text_embeddings_batched(tokenizer, taxon_names, model, 
                args.device, text_batch_size=args.text_batch_size, tmp_save_loc=os.path.join(args.feature_output, args.exp_type + "_text_tmps"))
            np.save(data_save_path, taxon_embeddings)
            np.save(lbl_save_path, np.arange(len(taxon_names)))
        print("Text features extracted")
        exit()

    # initialize datasets
    dset = ImageFolder(args.val_root, transform=preprocess_val)
    classes = dset.classes
    inat_name_lookup = iNat21NameLookup(inat21_root=args.val_root)
    classnames = [inat_name_lookup.taxonomic(cls) for cls in classes]
    dloader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size,
        num_workers=8,
        sampler=None,
    )

    # logging
    args.save_logs = args.logs and args.logs.lower() != "none"


    with torch.no_grad():
        class_embeddings = None
        if args.extract_logits:
            tokenizer = get_tokenizer(args.model)
            class_embeddings = get_text_embeddings(tokenizer, classnames, model, args.device)
        features, lbls = extract_image_features(model, dloader, device, class_embeddings=class_embeddings)

    np.save(data_save_path, features)
    np.save(lbl_save_path, lbls)

    print("Features saved!")

