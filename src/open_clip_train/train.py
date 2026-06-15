import json
import logging
import math

_logger = logging.getLogger(__name__)
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from open_clip_train.distributed import is_master
from open_clip_train.metrics import DEFAULT_RETRIEVAL_CHUNK_SIZE
from open_clip_train.metrics import get_clip_metrics
from open_clip_train.scheduler import get_learning_rate
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast


@dataclass
class TrainState:
    """Runtime training state.

    Checkpoint serialization remains owned by the existing task helpers.
    ``global_step`` and ``samples_seen`` are optional checkpoint metadata;
    ``compiled_train_step`` is runtime-only and is not saved.
    """
    task: Any
    optimizer: Optional[torch.optim.Optimizer] = None
    scaler: Optional[Any] = None
    scheduler: Optional[Callable[[int], None]] = None
    epoch: int = 0
    global_step: int = 0
    samples_seen: int = 0
    compiled_train_step: Optional[Callable] = None
    # Persistent cross-epoch loss EMA meters (runtime logging state; intentionally not checkpointed).
    losses_ema: dict = field(default_factory=dict)


def estimate_train_state_counters(epoch: int, data: dict, args) -> tuple[int, int]:
    """Estimate train counters for legacy checkpoints without explicit counters."""
    if epoch <= 0 or 'train' not in data:
        return 0, 0

    dataloader = data['train'].dataloader
    global_step = (dataloader.num_batches // args.accum_freq) * epoch
    samples_seen = dataloader.num_samples * epoch
    return global_step, samples_seen


def restore_train_state_counters(
        state: TrainState,
        metadata: Optional[dict],
        data: dict,
        args,
) -> None:
    """Restore counters from checkpoint metadata, estimating them for old checkpoints."""
    estimated_global_step, estimated_samples_seen = estimate_train_state_counters(state.epoch, data, args)
    state.global_step = estimated_global_step
    state.samples_seen = estimated_samples_seen
    if metadata is not None:
        if "global_step" in metadata:
            state.global_step = int(metadata["global_step"])
        if "samples_seen" in metadata:
            state.samples_seen = int(metadata["samples_seen"])


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


class SampleWeightedEMA:
    """Sample-count-weighted EMA of a scalar.

    The smoothing horizon is expressed in samples (``ema_samples``), so it is invariant to batch size, gradient
    accumulation, world size, and NaFlex packing -- ``decay = exp(-n / ema_samples)`` per update of ``n`` samples.
    The first observation seeds ``value`` directly (no cold-start ramp from 0), which keeps logs clean after a
    ``--resume`` since this is runtime-only logging state and is not checkpointed.
    """

    def __init__(self, ema_samples: float):
        self.ema_samples = float(ema_samples)
        self.value = None

    def update(self, val, n):
        decay = math.exp(-n / self.ema_samples)
        self.value = val if self.value is None else decay * self.value + (1.0 - decay) * val
        return self.value


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


def _torch_compile_kwargs(args):
    kwargs = {}
    backend = getattr(args, "torchcompile_backend", None)
    mode = getattr(args, "torchcompile_mode", None)
    if backend is not None:
        kwargs["backend"] = backend
    if mode is not None:
        kwargs["mode"] = mode
    return kwargs


def _make_train_step_no_accum_no_scaler(task, optimizer, autocast, args):
    grad_clip_norm = getattr(args, "grad_clip_norm", None)
    # Parameters are snapshotted when the step is created/compiled. Do not swap
    # or re-wrap task.trainable_module after this point.
    clip_params = tuple(task.trainable_module.parameters()) if grad_clip_norm is not None else ()

    def train_step(batch):
        # Keep zero_grad and logit-scale clamp outside this compiled closure.
        # Dynamo intentionally graph-breaks on optimizer.zero_grad(), and the
        # clamp is tiny eager bookkeeping after the optimizer update.
        loss_scale = get_naflex_loss_scale(batch, args, task)
        with autocast():
            losses, report = task(batch)
            total_loss = losses["loss"]
        if loss_scale != 1.0:
            total_loss = total_loss * loss_scale
        total_loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(clip_params, grad_clip_norm, norm_type=2.0)
        optimizer.step()
        return losses, report

    return train_step


def _get_compiled_train_step(state: TrainState, autocast, args):
    assert state.optimizer is not None, "_get_compiled_train_step requires state.optimizer."
    if state.compiled_train_step is not None:
        return state.compiled_train_step

    compiled_train_step = torch.compile(
        _make_train_step_no_accum_no_scaler(state.task, state.optimizer, autocast, args),
        **_torch_compile_kwargs(args),
    )
    state.compiled_train_step = compiled_train_step
    return state.compiled_train_step


def _finish_eager_train_step(task, optimizer, scaler, args):
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

    # Note: we clamp to 4.6052 = ln(100), as in the original paper.
    task.clamp_logit_scale()


def _train_step_eager(task, batch, accum_state, optimizer, scaler, autocast, args):
    if args.accum_freq == 1:
        optimizer.zero_grad()
        with autocast():
            losses, report = task(batch)
            total_loss = losses["loss"]

        loss_scale = get_naflex_loss_scale(batch, args, task)
        if loss_scale != 1.0:
            total_loss = total_loss * loss_scale
        backward(total_loss, scaler)

        _finish_eager_train_step(task, optimizer, scaler, args)
        return losses, report, task.batch_size(batch), accum_state

    accum_batches, accum_features = accum_state

    # First, cache the features without any gradient tracking.
    with torch.no_grad():
        with autocast():
            model_out = task.trainable_module(**batch)

            for f in ("logit_scale", "logit_bias"):
                model_out.pop(f, None)

            for key, val in model_out.items():
                if key in accum_features:
                    accum_features[key].append(val)
                else:
                    accum_features[key] = [val]

        accum_batches.append(batch)

    if len(accum_batches) < args.accum_freq:
        # FIXME this makes data time logging unreliable when accumulating
        return None

    # Now, ready to take gradients for the last accum_freq batches.
    # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
    # Call backwards each time, but only step optimizer at the end.
    optimizer.zero_grad()
    for j in range(args.accum_freq):
        batch_j = accum_batches[j]

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
                model_out = task.trainable_module(**batch_j)

                inputs_no_accum = {}
                inputs_no_accum["logit_scale"] = model_out.pop("logit_scale")
                if "logit_bias" in model_out:
                    inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                inputs = {}
                for key, val in accum_features.items():
                    accumulated = accum_features[key]
                    inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                losses, report = task.compute_accum_loss(inputs, inputs_no_accum, accum_batches)
                del inputs
                del inputs_no_accum
                total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
                losses["loss"] = total_loss

            loss_scale = get_naflex_loss_scale(batch_j, args, task)
            if loss_scale != 1.0:
                total_loss = total_loss * loss_scale
            backward(total_loss, scaler)

        if use_fsdp_no_sync:
            task.trainable_module.set_requires_gradient_sync(True)

    step_batch_size = sum(task.batch_size(accum_batch) for accum_batch in accum_batches)
    _finish_eager_train_step(task, optimizer, scaler, args)
    return losses, report, step_batch_size, ([], {})


def is_naflex_batch(batch):
    image = batch.get("image")
    return isinstance(image, dict) and "patches" in image


def get_naflex_loss_scale(batch, args, task):
    loss_scale = getattr(args, "naflex_loss_scale", "none")
    if loss_scale in (None, "none") or not is_naflex_batch(batch):
        return 1.0

    batch_size = task.batch_size(batch)
    reference_batch_size = getattr(args, "batch_size", None)
    if reference_batch_size is None or reference_batch_size <= 0:
        raise ValueError("NaFlex loss scaling requires a positive --batch-size reference.")

    scale = batch_size / reference_batch_size
    if loss_scale == "linear":
        return scale
    if loss_scale == "sqrt":
        return math.sqrt(scale)
    raise ValueError(f"Unsupported NaFlex loss scale: {loss_scale}")


def train_one_epoch(state: TrainState, data, args, tb_writer=None):
    task = state.task
    optimizer = state.optimizer
    scaler = state.scaler
    scheduler = state.scheduler
    epoch = state.epoch
    assert optimizer is not None, "train_one_epoch requires state.optimizer."
    if not args.skip_scheduler:
        assert scheduler is not None, "train_one_epoch requires state.scheduler unless --skip-scheduler is set."

    device = torch.device(args.device)
    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, 'fsdp', False),
    )
    input_dtype = get_input_dtype(args.precision)
    compile_step = (
        getattr(args, "torchcompile", False)
        and getattr(args, "torchcompile_strategy", "task") == "step"
    )
    eager_step = args.accum_freq > 1 or scaler is not None
    if compile_step and eager_step:
        raise ValueError(
            "--torchcompile-strategy step requires --accum-freq 1 and a precision without GradScaler."
        )
    if eager_step:
        train_step = None
    elif compile_step:
        train_step = _get_compiled_train_step(state, autocast, args)
    else:
        train_step = _make_train_step_no_accum_no_scaler(task, optimizer, autocast, args)

    task.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    accum_state = ([], {}) if args.accum_freq > 1 else None

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    num_samples = 0
    # Log the global-batch loss: under --local-loss each rank's loss is a 1/world_size slice (generative LM losses
    # are likewise per-rank), so all-reduce-mean at log time. Mean is a no-op when ranks already agree.
    reduce_loss = args.distributed and args.world_size > 1
    ema_samples = getattr(args, "train_loss_ema_samples", 0)
    metric_every = max(1, getattr(args, "log_metric_every_n_steps", args.log_every_n_steps))
    # Global samples observed at the previous EMA update (seeded from the persistent counter so the EMA horizon
    # stays continuous across epochs); the EMA decays by the sample delta between metric logs.
    prev_metric_samples = state.samples_seen
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        batch = task.prepare_batch(batch, device=device, input_dtype=input_dtype)

        data_time_m.update(time.time() - end)
        if train_step is not None:
            optimizer.zero_grad()
            losses, report = train_step(batch)
            task.clamp_logit_scale()
            step_batch_size = task.batch_size(batch)
        else:
            result = _train_step_eager(
                task,
                batch,
                accum_state,
                optimizer,
                scaler,
                autocast,
                args,
            )
            if result is None:
                continue
            losses, report, step_batch_size, accum_state = result

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        num_samples += step_batch_size * args.world_size
        state.global_step = step + 1
        state.samples_seen += step_batch_size * args.world_size
        last_batch = batch_count == num_batches_per_epoch
        is_console_step = (i_accum % args.log_every_n_steps == 0) or last_batch
        is_metric_step = (i_accum % metric_every == 0) or is_console_step

        if is_metric_step:
            # Reduce the loss across ranks so the logged value is the true global-batch loss. Collective -> ALL
            # ranks call it (the rank-synced schedule keeps them in lockstep); detach()+clone() since backward
            # already ran and all_reduce mutates in place. Mean is a no-op when ranks already agree.
            reduced = {key: val.detach().float().clone() for key, val in losses.items()}
            if reduce_loss:
                for v in reduced.values():
                    dist.all_reduce(v)
                    v /= args.world_size

            if is_master(args):
                n_ema = state.samples_seen - prev_metric_samples  # global samples since the previous EMA update
                prev_metric_samples = state.samples_seen
                for key, v in reduced.items():
                    vi = v.item()
                    losses_m.setdefault(key, AverageMeter()).update(vi, step_batch_size)
                    if ema_samples and n_ema > 0:
                        state.losses_ema.setdefault(key, SampleWeightedEMA(ema_samples)).update(vi, n_ema)

                # logit_scale / logit_bias are per-step report scalars (not loss terms), logged once from `report`.
                logit_scale = report.get("logit_scale", None)
                logit_scale_scalar = logit_scale.item() if logit_scale is not None else 0.0
                logit_bias = report.get("logit_bias", None)
                logit_bias_scalar = logit_bias.item() if logit_bias is not None else None
                samples_per_second = step_batch_size * args.world_size / batch_time_m.val
                samples_per_second_per_gpu = step_batch_size / batch_time_m.val
                learning_rate = get_learning_rate(optimizer)

                # Raw scalars to tensorboard/wandb at the (dense) metric cadence (see the per-loss block below).
                log_data = {
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "logit_scale": logit_scale_scalar,
                    "lr": learning_rate,
                }
                if logit_bias_scalar is not None:
                    log_data["logit_bias"] = logit_bias_scalar
                # Raw current value only -- dashboards do their own smoothing, so the EMA stays console-only (add
                # train/<loss>_ema behind an explicit flag if a deterministic logged series is ever wanted). The
                # epoch running average is NOT logged per-step (half-formed mid-epoch); it goes out once per epoch.
                for name, m in losses_m.items():
                    log_data[name] = m.val
                log_data = {"train/" + name: val for name, val in log_data.items()}

                if tb_writer is not None:
                    for name, val in log_data.items():
                        tb_writer.add_scalar(name, val, step)

                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    log_data['step'] = step  # for backwards compatibility
                    wandb.log(log_data, step=step)

                # Console at the (sparse) cadence. Parentheses show the cross-epoch EMA trend (the epoch average
                # moves to the End-epoch summary line); falls back to the epoch avg when the EMA is disabled.
                if is_console_step:
                    samples_per_epoch = dataloader.num_samples
                    percent_complete = 100.0 * batch_count / num_batches_per_epoch
                    loss_log = " ".join(
                        f"{name.capitalize()}: {m.val:#.5g} "
                        f"({(state.losses_ema[name].value if ema_samples else m.avg):#.5g})"
                        for name, m in losses_m.items()
                    )
                    _logger.info(
                        f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                        f"Data (t): {data_time_m.avg:.3f} "
                        f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                        f"LR: {learning_rate:5f} "
                        f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
                    )

                # reset batch / data time meters per metric window
                batch_time_m.reset()
                data_time_m.reset()
    # end for
    if is_master(args) and losses_m:
        summary = "  ".join(f"Avg {name.capitalize()}: {m.avg:#.5g}" for name, m in losses_m.items())
        _logger.info(f"End epoch {epoch} | {summary}")
        # One epoch-average point per epoch (distinct, sparse series from the dense per-step curve). Logged at the
        # epoch's final step so it merges with that step's record rather than starting a new one.
        last_step = num_batches_per_epoch * (epoch + 1) - 1
        epoch_log = {f"train/{name}_epoch_avg": m.avg for name, m in losses_m.items()}
        if tb_writer is not None:
            for name, val in epoch_log.items():
                tb_writer.add_scalar(name, val, last_step)
        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            wandb.log({**epoch_log, "epoch": epoch}, step=last_step)


def zero_shot_eval_all(task, data, epoch, args, tokenizer=None):
    """Run requested zero-shot evaluators based on entries in the data dict."""
    zero_shot_metrics = {}
    if "imagenet-val" in data or "imagenet-v2" in data:
        zero_shot_metrics.update(zero_shot_eval(task, data, epoch, args, tokenizer=tokenizer))
    if "audio-zeroshot" in data:
        from open_clip_train.audio_zero_shot import audio_zero_shot_eval

        zero_shot_metrics.update(
            audio_zero_shot_eval(task, data["audio-zeroshot"], epoch, args, tokenizer=tokenizer)
        )
    return zero_shot_metrics


def evaluate(task, data, epoch, args, tb_writer=None, tokenizer=None):
    """Run paired feature validation and supported zero-shot eval for a task."""
    metrics = {}
    use_fsdp_eval = getattr(args, 'fsdp', False) and getattr(args, 'distributed', False)
    is_rank0 = is_master(args)

    if not use_fsdp_eval and not is_rank0:
        return metrics

    device = torch.device(args.device)
    task.eval()

    primary_key = task.primary_key
    primary_features_key = f"{primary_key}_features"

    zero_shot_metrics = zero_shot_eval_all(task, data, epoch, args, tokenizer=tokenizer)
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
            # Pre-allocate dummy batch for non-master ranks
            dummy_batch = task.create_dummy_batch(
                batch_size=1,
                device=device,
                dtype=input_dtype,
            )
            signal = torch.zeros(1, device=device, dtype=torch.long)

        # Retrieval metrics are computed in score chunks below, but feature
        # accumulation and exact pair scoring remain O(N * D) memory and O(N^2)
        # compute respectively.
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_primary_features, all_text_features = [], []
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
                        batch = task.prepare_batch(batch, device, input_dtype)
                    else:
                        batch = dummy_batch
                else:
                    batch = next(dataloader_iter, None)
                    if batch is None:
                        break
                    batch = task.prepare_batch(batch, device, input_dtype)

                with autocast():
                    model_out = task(batch)

                if is_rank0:
                    batch_size = task.batch_size(batch)
                    # Contrastive tasks expose paired features for retrieval; generative-only tasks (e.g. GenLIP)
                    # return just an LM/caption loss. Branch so generative eval doesn't KeyError on features.
                    paired = (primary_features_key in model_out) and ("text_features" in model_out)
                    if paired:
                        primary_features = model_out[primary_features_key]
                        text_features = model_out["text_features"]
                        logit_scale = model_out["logit_scale"].mean()
                        # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                        # however, system RAM is easily exceeded and compute time becomes problematic
                        all_primary_features.append(primary_features.cpu())
                        all_text_features.append(text_features.cpu())
                        logits_per_primary = logit_scale * primary_features @ text_features.t()
                        logits_per_text = logits_per_primary.t()
                        labels = torch.arange(batch_size, device=device).long()
                        total_loss = (
                            F.cross_entropy(logits_per_primary, labels) +
                            F.cross_entropy(logits_per_text, labels)
                        ) / 2
                        cumulative_loss += total_loss * batch_size
                        gen_loss = maybe_compute_generative_loss(model_out, texts=batch.get("text"))
                    else:
                        gen_loss = model_out.get("caption_loss", model_out.get("loss"))

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                    num_samples += batch_size
                    if (i % 100) == 0:
                        if paired:
                            _logger.info(
                                f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                                f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")
                        if gen_loss is not None:
                            _logger.info(
                                f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

                i += 1

            if is_rank0 and num_samples > 0:
                if all_primary_features:
                    retrieval_chunk_size = getattr(
                        args,
                        "val_retrieval_chunk_size",
                        DEFAULT_RETRIEVAL_CHUNK_SIZE,
                    )
                    retrieval_precision = getattr(args, "val_retrieval_precision", "fp32")
                    retrieval_device = device if retrieval_chunk_size and retrieval_chunk_size > 0 else None
                    val_metrics = get_clip_metrics(
                        image_features=all_primary_features,
                        text_features=all_text_features,
                        logit_scale=logit_scale.cpu(),
                        image_key=primary_key,
                        text_key="text",
                        retrieval_chunk_size=retrieval_chunk_size,
                        retrieval_device=retrieval_device,
                        retrieval_dtype="model" if retrieval_precision == "model" else torch.float32,
                    )
                    loss = cumulative_loss / num_samples
                    # Preserve the legacy dashboard key for image-text validation.
                    loss_key = "clip_val_loss" if primary_key == "image" else f"{primary_key}_val_loss"
                    metrics.update(
                        {**val_metrics, loss_key: loss.item(), "epoch": epoch, "num_samples": num_samples}
                    )
                else:
                    # Generative-only task (e.g. GenLIP): no retrieval metrics, just the LM/caption loss.
                    metrics.update({"epoch": epoch, "num_samples": num_samples})
                if isinstance(cumulative_gen_loss, torch.Tensor):
                    metrics.update({"val_generative_loss": (cumulative_gen_loss / num_samples).item()})

    if not is_rank0:
        return metrics

    if not metrics:
        return metrics

    _logger.info(
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

def maybe_compute_generative_loss(model_out, texts=None, pad_id=0):
    if "logits" in model_out and texts is not None:
        logits = model_out["logits"][:, :-1]
        labels = texts[:, 1:]
        return F.cross_entropy(logits.permute(0, 2, 1), labels, ignore_index=pad_id)
