"""Evaluation mechanics shared by modality-specific runners and trainer entrypoints."""
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import get_input_dtype
from open_clip_train.precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum().item() for k in topk]


def iter_eval_batches(dataloader, device, rank=0, use_fsdp_eval=False, unit_scale=None):
    """Let rank zero drive exhaustion while every FSDP rank performs the same number of forwards."""
    if not use_fsdp_eval:
        yield from (tqdm(dataloader, unit_scale=unit_scale) if unit_scale is not None else dataloader)
        return
    signal = torch.zeros(1, device=device, dtype=torch.long)
    iterator = iter(dataloader) if rank == 0 else None
    while True:
        batch = next(iterator, None) if rank == 0 else None
        if rank == 0:
            signal.fill_(batch is not None)
        dist.broadcast(signal, src=0)
        if signal.item() == 0:
            return
        yield batch


def run_classification_eval(
        model, classifier, dataloader, args, *, input_key, prepare_batch, create_dummy, use_fsdp_eval=False,
):
    device = torch.device(args.device)
    input_dtype = get_input_dtype(args.precision)
    autocast = get_autocast(args.precision, device_type=device.type, fsdp=getattr(args, 'fsdp', False))
    score = not use_fsdp_eval or args.rank == 0
    dummy = create_dummy(device, input_dtype) if not score else None
    top1, top5, n = 0., 0., 0
    with torch.inference_mode():
        for batch in iter_eval_batches(dataloader, device, args.rank, use_fsdp_eval, args.batch_size):
            if score:
                inputs, target = prepare_batch(batch, device, input_dtype)
            else:
                inputs = dummy
            with autocast():
                output = model(**{input_key: inputs})
                features = output[f'{input_key}_features'] if isinstance(output, dict) else output[0]
                if score:
                    logits = 100. * features @ classifier
            if score:
                acc1, acc5 = accuracy(logits, target, topk=(1, min(5, classifier.shape[1])))
                top1 += acc1
                top5 += acc5
                n += features.shape[0]
    return (top1 / n, top5 / n) if n else (0., 0.)


def maybe_compute_generative_loss(model_out, texts=None, text_valid=None, pad_id=0):
    """Next-token validation CE, with explicit validity taking precedence over the padding id."""
    if 'logits' in model_out and texts is not None:
        logits = model_out['logits'][:, :-1]
        labels = texts[:, 1:]
        valid = text_valid[:, 1:].bool() if text_valid is not None else labels != pad_id
        labels = labels.masked_fill(~valid, -100)
        return F.cross_entropy(logits.permute(0, 2, 1), labels, ignore_index=-100)


def log_eval_metrics(metrics, data, epoch, args, logger, tb_writer=None, backend=None):
    logger.info(f'Eval Epoch: {epoch} ' + '\t'.join(f'{k}: {v:.4f}' for k, v in metrics.items()))
    log_data = {'val/' + name: val for name, val in metrics.items()}
    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)
        with open(os.path.join(args.checkpoint_path, 'results.jsonl'), 'a+') as f:
            f.write(json.dumps(metrics) + '\n')
    if backend is not None:
        step = (data['train'].dataloader.num_batches // args.accum_freq) * epoch if 'train' in data else None
        log_data['epoch'] = epoch
        backend.log(log_data, step=step)
