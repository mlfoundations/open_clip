import json
import logging
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip.task import get_model_from_task
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(task, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, 'fsdp', False),
    )
    input_dtype = get_input_dtype(args.precision)

    task.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                losses = task(images, texts)
                total_loss = losses["loss"]

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = task.trainable_module(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]

                # Disable gradient sync for all but the last accumulation step.
                # FSDP2: set_requires_gradient_sync; DDP: no_sync context manager.
                is_last_step = (j == args.accum_freq - 1)
                use_fsdp_no_sync = (
                    not is_last_step
                    and hasattr(task.trainable_module, 'set_requires_gradient_sync')
                )
                use_ddp_no_sync = (
                    not is_last_step
                    and not use_fsdp_no_sync
                    and isinstance(task.trainable_module, DistributedDataParallel)
                )
                if use_fsdp_no_sync:
                    task.trainable_module.set_requires_gradient_sync(False)

                ddp_context = task.trainable_module.no_sync() if use_ddp_no_sync else nullcontext()
                with ddp_context:
                    with autocast():
                        model_out = task.trainable_module(images, texts)

                        inputs_no_accum = {}
                        inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                        if "logit_bias" in model_out:
                            inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                        losses = task.compute_accum_loss(inputs, inputs_no_accum, accum_texts)
                        del inputs
                        del inputs_no_accum
                        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
                        losses["loss"] = total_loss
                        losses["logit_scale"] = logit_scale

                    backward(total_loss, scaler)

                if use_fsdp_no_sync:
                    task.trainable_module.set_requires_gradient_sync(True)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    task.trainable_module.parameters(), args.grad_clip_norm, norm_type=2.0,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    task.trainable_module.parameters(), args.grad_clip_norm, norm_type=2.0,
                )
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        task.clamp_logit_scale()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale = losses.get("logit_scale", None)
            logit_scale_scalar = logit_scale.item() if logit_scale is not None else 0.0
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model_or_task, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    use_fsdp_eval = getattr(args, 'fsdp', False) and getattr(args, 'distributed', False)
    is_rank0 = is_master(args)

    if not use_fsdp_eval and not is_rank0:
        return metrics

    device = torch.device(args.device)
    model_or_task.eval()
    model = get_model_from_task(model_or_task)

    zero_shot_metrics = zero_shot_eval(model_or_task, data, epoch, args, tokenizer=tokenizer)
    if is_rank0:
        metrics.update(zero_shot_metrics)

    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, 'fsdp', False),
    )
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        num_samples = 0
        samples_per_val = 0

        if is_rank0:
            dataloader = data['val'].dataloader
            samples_per_val = dataloader.num_samples
            dataloader_iter = iter(dataloader)

        if use_fsdp_eval:
            # Pre-allocate dummy tensors for non-master ranks
            image_size = model.visual.image_size
            if not isinstance(image_size, tuple):
                image_size = (image_size, image_size)
            dummy_images = torch.zeros(1, 3, *image_size, device=device, dtype=input_dtype)
            dummy_texts = torch.zeros(1, model.context_length, device=device, dtype=torch.long)
            signal = torch.zeros(1, device=device, dtype=torch.long)

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            i = 0
            while True:
                if use_fsdp_eval:
                    if is_rank0:
                        batch = next(dataloader_iter, None)
                        signal.fill_(0 if batch is None else 1)
                    dist.broadcast(signal, src=0)
                    if signal.item() == 0:
                        break

                    if is_rank0:
                        images, texts = batch
                        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                        texts = texts.to(device=device, non_blocking=True)
                    else:
                        images, texts = dummy_images, dummy_texts
                else:
                    batch = next(dataloader_iter, None)
                    if batch is None:
                        break
                    images, texts = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model_or_task(images, texts)

                if is_rank0:
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out, texts=texts)

                    cumulative_loss += total_loss * batch_size
                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                    num_samples += batch_size
                    if (i % 100) == 0:
                        logging.info(
                            f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                            f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                        if gen_loss is not None:
                            logging.info(
                                f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

                i += 1

            if is_rank0 and num_samples > 0:
                val_metrics = get_clip_metrics(
                    image_features=torch.cat(all_image_features),
                    text_features=torch.cat(all_text_features),
                    logit_scale=logit_scale.cpu(),
                )
                loss = cumulative_loss / num_samples
                metrics.update(
                    {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
                )
                if gen_loss is not None:
                    gen_loss = cumulative_gen_loss / num_samples
                    metrics.update({"val_generative_loss": gen_loss.item()})

    if not is_rank0:
        return metrics

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out, texts=None, pad_id=0):
    if "logits" in model_out and texts is not None:
        logits = model_out["logits"][:, :-1]
        labels = texts[:, 1:]
        return F.cross_entropy(logits.permute(0, 2, 1), labels, ignore_index=pad_id)
