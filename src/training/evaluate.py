import json
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as f
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import (
    get_input_dtype,
    get_tokenizer,
    build_zero_shot_classifier,
    IMAGENET_CLASSNAMES,
    OPENAI_IMAGENET_TEMPLATES,
)
from .distributed import is_master
from .precision import get_autocast


def _get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image-to-text": logits_per_image, "text-to-image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}-mean-rank"] = preds.mean() + 1
        metrics[f"{name}-median-rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}-R@{k}"] = np.mean(preds < k)

    return metrics


def _maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return f.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


def _run_validation(model, data, epoch, args):
    if 'val' not in data:
        return {}
    if args.val_frequency == 0:
        return {}
    if (epoch % args.val_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting evaluation on the validation set ...')

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    device = torch.device(args.device)

    dataloader = data['val'].dataloader
    num_samples = 0
    samples_per_val = dataloader.num_samples

    # FIXME this does not scale past small eval datasets
    # all_image_features @ all_text_features will blow up memory and compute
    # very quickly
    cumulative_loss = 0.0
    cumulative_gen_loss = 0.0
    all_image_features, all_text_features = [], []

    metrics = {}

    logging.info('Infering text and image features ...')

    with torch.no_grad():

        for i, batch in enumerate(dataloader):
            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            with autocast():
                model_out = model(images, texts)
                image_features = model_out["image_features"]
                text_features = model_out["text_features"]
                logit_scale = model_out["logit_scale"]
                # features are accumulated in CPU tensors, otherwise GPU memory is
                # exhausted quickly
                # however, system RAM is easily exceeded and compute time becomes
                # problematic
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
                logit_scale = logit_scale.mean()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                batch_size = images.shape[0]
                labels = torch.arange(batch_size, device=device).long()
                total_loss = (
                         f.cross_entropy(logits_per_image, labels) +
                         f.cross_entropy(logits_per_text, labels)
                 ) / 2

                gen_loss = _maybe_compute_generative_loss(model_out)

            cumulative_loss += total_loss * batch_size
            num_samples += batch_size
            if is_master(args) and (i % 100) == 0:
                logging.info(
                    f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                    f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                if gen_loss is not None:
                    cumulative_gen_loss += gen_loss * batch_size
                    logging.info(
                        f"Generative Loss: "
                        f"{cumulative_gen_loss / num_samples:.6f}\t"
                    )

        logging.info('Calculating CLIP metrics, mean/median rank and recall ...')

        val_metrics = _get_clip_metrics(
            image_features=torch.cat(all_image_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale.cpu(),
        )
        loss = cumulative_loss / num_samples
        metrics.update(
            {
                **val_metrics,
                "clip_loss": loss.item(),
                "epoch": epoch,
                "num_samples": num_samples,
            }
        )
        if gen_loss is not None:
            gen_loss = cumulative_gen_loss / num_samples
            metrics.update({"generative_loss": gen_loss.item()})

        logging.info('Finished!')

    return metrics


def _accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def _run_classifier(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = (
                    output['image_features'] if isinstance(output, dict) else output[0]
                )
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = _accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def _run_zeroshot_evaluation(model, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot evaluation on Imagenet ...')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier ...')
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier ...')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = _run_classifier(
            model, classifier, data['imagenet-val'].dataloader, args
        )
        results['imagenet-zeroshot-top1'] = top1
        results['imagenet-zeroshot-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = _run_classifier(
            model, classifier, data['imagenet-v2'].dataloader, args
        )
        results['imagenetv2-zeroshot-top1'] = top1
        results['imagenetv2-zeroshot-top5'] = top5

    logging.info('Finished zero-shot evaluation!')

    return results


def _run_clip_benchmark(model, tokenizer, transform, epoch, args):
    if args.clip_benchmark_frequency == 0:
        return {}
    if (epoch % args.clip_benchmark_frequency) != 0 and epoch != args.epochs:
        return {}

    from clip_benchmark.run import run_benchmark, CLIPBenchmarkModel

    results = run_benchmark(
        datasets=args.clip_benchmark_datasets,
        models=[
            CLIPBenchmarkModel(
                name=args.model,
                pretrained=args.name + f'-epoch#{epoch}',
                module=model,
                tokenizer=tokenizer,
                transform=transform,
            )
        ],
        task='auto',
        output=None,
        dataset_root=args.clip_benchmark_dataset_root,
        distributed=False,
        recall_ks=[int(k) for k in args.clip_benchmark_recall_ks.split(',')]
    )
    metrics = {}
    for result in results:
        dataset = result['dataset']
        for k, v in result['metrics'].items():
            metrics[f'{dataset}-{k}'] = v

    return metrics


def evaluate(
    model: torch.nn.Module,
    transform: Any,
    tokenizer: Any,
    data,
    epoch: int,
    args,
    tb_writer: Any = None,
):
    metrics = {}
    if not is_master(args):
        return metrics

    model.eval()

    zero_shot_metrics = _run_zeroshot_evaluation(
        model, data, epoch, args, tokenizer=tokenizer
    )
    metrics.update(zero_shot_metrics)

    val_metrics = _run_validation(model, data, epoch, args)
    metrics.update(val_metrics)

    clip_benchmark_metrics = _run_clip_benchmark(
        model, tokenizer, transform, epoch, args
    )
    metrics.update(clip_benchmark_metrics)

    if not metrics:
        return {}

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    logdata = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in logdata.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as fd:
            fd.write(json.dumps(metrics))
            fd.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        logdata['epoch'] = epoch
        wandb.log(logdata, step=step)

    return metrics
