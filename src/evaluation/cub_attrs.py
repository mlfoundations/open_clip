"""
Do quantitative trait-level evaluation of different CLIP models on CUB-2011 dataset.
"""

import datetime
import logging
import os
import re
import sys
import warnings

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

from .params import parse_args
from .utils import init_device, random_seed

cub_img_root = "/local/scratch/cv_datasets/cub2011/traintest224"
cub_label_root = "/local/scratch/cv_datasets/cub2011/original/CUB_200_2011"
natural_attrs_path = "data/cub/template_attributes.txt"

num_workers = 8

# Can be one of 'bird', 'taxon' or 'common'.
# bird: just use the word 'bird'
# taxon: use the taxonomic name
# common: use the common name
template_option = "bird"


def get_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=None,
    )


def get_text_features(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        all_features = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            text_features = model.encode_text(texts)
            text_features = F.normalize(text_features, dim=-1).mean(dim=0)
            text_features /= text_features.norm()
            all_features.append(text_features)
        all_features = torch.stack(all_features, dim=1).to(args.device)
    return all_features


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def get_logits(model, text_features, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    all_logits = []
    with torch.no_grad():
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                all_logits.append(100.0 * image_features @ text_features)

    return torch.concat(all_logits)


pattern = re.compile(r"\d+ (.*)")


def get_cub_attributes():
    natural_attr_templates = []
    with open(natural_attrs_path) as fd:
        for line in fd:
            match = pattern.match(line)
            assert match is not None
            natural_attr_templates.append(match.group(1))

    if template_option == "bird":
        for attr in natural_attr_templates:
            return [tmpl.format("bird") for tmpl in natural_attr_templates]
    else:
        raise NotImplementedError(template_option)


def zero_shot_eval(model, data, args):
    results = {}

    logging.info("Starting zero-shot cub-2011 attributes.")

    for split in data:
        logging.info("Building text features for %s.", split)
        texts = get_cub_attributes()
        text_features = get_text_features(model, texts, openai_imagenet_template, args)

        logging.info("Built text features.")

        logits = get_logits(model, text_features, data[split], args)
        for key, value in eval_logits(logits, split).items():
            results[f"{split}/{key}"] = value

        logging.info("Finished classifying %s.", split)

    logging.info("Finished classifying attributes.")

    return results


def eval_logits(logits, split):
    logits = logits.cpu().numpy()

    results = {}

    if split == "train":
        attr_labels = cub_attr_labels.binary_labels[cub_attr_labels.train_mask]
    elif split == "test":
        attr_labels = cub_attr_labels.binary_labels[cub_attr_labels.test_mask]
    else:
        raise ValueError(split)

    assert attr_labels.shape == logits.shape

    # Using one threshold for entire dataset
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
        attr_labels.flatten(), logits.flatten()
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        f1 = 2 * (precision * recall) / (precision + recall)
        f1[(precision + recall) == 0] = 0

    results["all"] = f1.max()

    # Using one threshold for each category
    all_preds = np.zeros_like(attr_labels)

    for i, _ in enumerate(cub_attr_labels.categories):
        mask = cub_attr_labels.category_mask[i]
        masked_labels = attr_labels[:, mask]
        masked_preds = logits[:, mask]

        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
            masked_labels.flatten(), masked_preds.flatten()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            f1 = 2 * (precision * recall) / (precision + recall)
            f1[(precision + recall) == 0] = 0

        best_threshold = thresholds[f1.argmax()]

        category_preds = np.zeros_like(all_preds[:, mask])
        category_preds[masked_preds >= best_threshold] = 1
        all_preds[:, mask] = category_preds

    results["category"] = sklearn.metrics.f1_score(
        attr_labels.flatten(), all_preds.flatten(), zero_division=0.0
    )

    # Using one threshold for each *macro*-category
    all_preds = np.zeros_like(attr_labels)

    for i, _ in enumerate(cub_attr_labels.macrocategories):
        mask = cub_attr_labels.macrocategory_mask[i]
        masked_labels = attr_labels[:, mask]
        masked_preds = logits[:, mask]

        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
            masked_labels.flatten(), masked_preds.flatten()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            f1 = 2 * (precision * recall) / (precision + recall)
            f1[(precision + recall) == 0] = 0

        best_threshold = thresholds[f1.argmax()]

        category_preds = np.zeros_like(all_preds[:, mask])
        category_preds[masked_preds >= best_threshold] = 1
        all_preds[:, mask] = category_preds

    results["macrocategory"] = sklearn.metrics.f1_score(
        attr_labels.flatten(), all_preds.flatten(), zero_division=0.0
    )

    return results


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

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
        args.pretrained,
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

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    logging.info("Model:")
    logging.info(f"{str(model)}")
    logging.info("Params:")
    params_file = os.path.join(args.logs, args.name, "params.txt")
    # TODO: include non-arg-based configuration here
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")

    # initialize datasets
    data = {
        "train": get_dataloader(
            ImageFolder(os.path.join(cub_img_root, "train"), transform=preprocess_val),
            batch_size=args.batch_size,
        ),
        "test": get_dataloader(
            ImageFolder(os.path.join(cub_img_root, "test"), transform=preprocess_val),
            batch_size=args.batch_size,
        ),
    }

    cub_attr_labels = cub2011.CubAttributeLabels.from_root(cub_label_root)

    # logging
    args.save_logs = args.logs and args.logs.lower() != "none"

    model.eval()
    metrics = zero_shot_eval(model, data, args)

    logging.info("Results:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value:.5f}")
    logging.info("Done.")
