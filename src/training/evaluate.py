import json
import logging
import os
from collections.abc import MutableMapping
from typing import Any, Union

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

MTEB_LOGGING_METRICS = ['ndcg_at_10', 'cos_sim']

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

    logging.info('--------------------------------------------------------------------')
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
    logging.info('--------------------------------------------------------------------')

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

    logging.info('--------------------------------------------------------------------')
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
    logging.info('--------------------------------------------------------------------')

    return results


def _run_clip_benchmark(model, tokenizer, transform, epoch, args):
    if args.clip_benchmark_frequency == 0:
        return {}
    if (epoch % args.clip_benchmark_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('--------------------------------------------------------------------')
    logging.info('Starting the CLIP benchmark ...')

    from clip_benchmark.run import run_benchmark, CLIPBenchmarkModel

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    else:
        module = model

    results = run_benchmark(
        datasets=[t for t in args.clip_benchmark_datasets.split(',')],
        models=[
            CLIPBenchmarkModel(
                name=args.model,
                pretrained=args.name + f'-epoch#{epoch}',
                module=module,
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
    
    logging.info('Finished CLIP benchmark!')
    logging.info('--------------------------------------------------------------------')

    return metrics


def _run_mteb_benchmark(model, tokenizer, epoch, args):

    if args.mteb_frequency == 0:
        return {}
    if (epoch % args.mteb_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('--------------------------------------------------------------------')
    logging.info('Starting the MTEB benchmark ...')

    from mteb import MTEB
    from transformers import AutoTokenizer

    from open_clip.model import CLIP

    class _MTEBModel(torch.nn.Module):

        def __init__(
            self,
            clip_model: torch.nn.Module,
            _tokenizer: Any = None,
            hf_tokenizer_name: str = '',
            batch_size: int = 4,
            max_seq_length: int = 8192,
            device: Union[str, torch.device] = 'cpu',
        ):
            super(_MTEBModel, self).__init__()

            self._tokenizer = None
            self._batch_size = batch_size
            self._max_seq_length = max_seq_length
            self._device = device

            if isinstance(clip_model, torch.nn.parallel.DistributedDataParallel):
                _model = clip_model.module
            else:
                _model = clip_model

            self._model = _model

            if isinstance(_model, CLIP):
                assert _tokenizer is not None
                self._tokenizer = _tokenizer
                self._embed = self._clip_embed

            else:
                assert hf_tokenizer_name
                self._tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=hf_tokenizer_name,
                    trust_remote_code=True,
                    force_download=True
                )
                self._embed = self._hf_embed

        @staticmethod
        def _mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        def _hf_embed(self, sentences: list[str]):
            encoded_input = self._tokenizer(
                sentences,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self._max_seq_length
            ).to(self._device)

            model_output = self._model.text.transformer(**encoded_input)
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input['attention_mask']
            )
            sentence_embeddings = f.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.cpu().numpy()

        def _clip_embed(self, sentences: list[str]):
            x = self._tokenizer(sentences).to(self._device)
            sentence_embeddings = self._model.encode_text(x)
            return sentence_embeddings.cpu().numpy()

        @torch.no_grad()
        def encode(self, sentences: list[str], batch_size: int = 1, **_):
            embeddings = []
            with torch.inference_mode():
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i: i + batch_size]
                    embeddings.append(self._embed(batch))

            return np.concatenate(embeddings, axis=0)

    def flatten(dictionary, parent_key='', separator='_'):
        items = []
        for key, value in dictionary.items():
            new_key = parent_key + separator + key if parent_key else key
            if isinstance(value, MutableMapping):
                items.extend(flatten(value, new_key, separator=separator).items())
            else:
                items.append((new_key, value))
        return dict(items)
    
    _mteb_model = _MTEBModel(
        clip_model=model,
        _tokenizer=tokenizer,
        hf_tokenizer_name=args.mteb_tokenizer_name,
        max_seq_length=args.mteb_max_seq_length,
        device=args.device,
    )

    metrics = {}
    for task in args.mteb_tasks.split(','):
        evaluation = MTEB(tasks=[task], task_langs=['en'])
        results = evaluation.run(
            _mteb_model,
            batch_size=4,
            output_folder=None,
            eval_splits=['dev'] if task == 'MSMARCO' else ['test'],
            ignore_identical_ids=False,
        )
        metrics.update({
            k: v
            for k, v in flatten(results, separator='-').items() if isinstance(v, float) and any(sub in k for sub in MTEB_LOGGING_METRICS)
        })
    
    logging.info('Finished MTEB benchmark!')
    logging.info('--------------------------------------------------------------------')

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

    logging.info('--------------------------- EVALUATION -----------------------------')

    zero_shot_metrics = _run_zeroshot_evaluation(
        model, data, epoch, args, tokenizer=tokenizer
    )
    metrics.update({f'zeroshot-{k}': v for k, v in zero_shot_metrics.items()})

    val_metrics = _run_validation(model, data, epoch, args)
    metrics.update({f'valset-{k}': v for k, v in val_metrics.items()})

    clip_benchmark_metrics = _run_clip_benchmark(
        model, tokenizer, transform, epoch, args
    )
    metrics.update({f'clipbenchmark-{k}': v for k, v in clip_benchmark_metrics.items()})

    mteb_metrics = _run_mteb_benchmark(model, tokenizer, epoch, args)
    metrics.update({f'mteb-{k}': v for k, v in mteb_metrics.items()})

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

    logging.info('------------------------------ DONE --------------------------------')

    return metrics
