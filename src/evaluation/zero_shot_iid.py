"""
Do zero-shot classification on IID data (iNat21) with both seen and unseen classes.

Single-process. If you want to run all evaluations of a single model at once, look
in scripts/.

Writes the output to a plaintext and JSON format in the logs directory.
"""

import datetime
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm

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
from .utils import random_seed, init_device

val_seen_path = "/local/scratch/cv_datasets/inat21/clip/eval/val-seen"
val_unseen_path = "/local/scratch/cv_datasets/inat21/clip/eval/val-unseen"


def get_imagenet(args, preprocess_fn, data_path):
    dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
    )


def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100.0 * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    return top1, top5


def inat21_class_preprocess(cls):
    num, *tiers = cls.split("_")
    return " ".join(tiers)


def zero_shot_eval(model, data, args):
    results = {}

    logging.info("Starting zero-shot inat21.")

    for split in ("inat21-val-seen", "inat21-val-unseen"):
        # To do zero-shot eval on iNat21 classes, we use the class names directly.
        # They are stored in the dataset in the .classes attribute
        logging.info("Building zero-shot %s classifier.", split)
        classnames = [inat21_class_preprocess(c) for c in data[split].dataset.classes]
        classifier = zero_shot_classifier(
            model, classnames, openai_imagenet_template, args
        )

        logging.info("Using classifier")
        top1, top5 = run(model, classifier, data[split], args)
        results[f"{split}-top1"] = top1
        results[f"{split}-top5"] = top5

        logging.info("Finished zero-shot %s.", split)

    logging.info("Finished zero-shot inat21.")

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
    model, preprocess_train, preprocess_val = create_model_and_transforms(
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
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")

    # initialize datasets
    data = {
        "inat21-val-seen": get_imagenet(args, preprocess_val, val_seen_path),
        "inat21-val-unseen": get_imagenet(args, preprocess_val, val_unseen_path),
    }

    args.save_logs = args.logs and args.logs.lower() != "none"
    writer = None

    model.eval()
    metrics = zero_shot_eval(model, data, args)

    logging.info("Results:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value:.5f}")
    logging.info("Done.")
