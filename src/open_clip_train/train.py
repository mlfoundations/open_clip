import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip.loss import compute_mask_weight_matrix
from open_clip_train.distributed import is_master, broadcast_object
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast

# Keys returned by loss(..., output_dict=True) that are metrics, not loss terms (excluded from total_loss)
_LOSS_METRIC_KEYS = frozenset({"masked_pairs", "frac_masked"})


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


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def _get_logit_scale_param(model):
    """Return the logit_scale parameter for clamping (handles DDP + PEFT)."""
    m = unwrap_model(model)
    if hasattr(m, 'logit_scale'):
        return m.logit_scale
    if hasattr(m, 'base_model') and hasattr(m.base_model, 'model') and hasattr(m.base_model.model, 'logit_scale'):
        return m.base_model.model.logit_scale
    return None


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def _viz_batch_matrix(images, img_attr, txt_attr, txt_mask, save_dir, step, epoch, max_batch=8, captions=None):
    """
    Visualize the batch label/weight matrix as used in SigLipMaskedAttrLoss.
    Matrix values: 1 = positive (diagonal), -1 = negative, 0 = neutral (masked).
    Saves a figure with image thumbnails, the (B,B) matrix colored by -1/0/1, and optional captions.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available; skipping batch matrix visualization.")
        return
    device = images.device
    B = images.shape[0]
    n = min(B, max_batch)
    W = compute_mask_weight_matrix(img_attr[:n], txt_attr[:n], txt_mask[:n], is_diag_block=True)
    # Effective label matrix: 1 = positive, -1 = negative, 0 = neutral
    labels = 2 * torch.eye(n, device=device, dtype=torch.float32) - 1
    V = torch.where(W > 0.5, labels, torch.zeros_like(labels))
    V_np = V.cpu().numpy()
    imgs = images[:n].cpu().float()
    # Denormalize for display (assume mean=0.5, std=0.5 as in SigLIP)
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    imgs = imgs.permute(0, 2, 3, 1).numpy()
    os.makedirs(save_dir, exist_ok=True)
    # Image grid: multiple rows so thumbnails stay square (avoid stretch)
    n_cols = min(16, n)
    n_rows = (n + n_cols - 1) // n_cols
    cell_in = 1.2
    img_w = n_cols * cell_in
    img_h = n_rows * cell_in
    matrix_w = 6.0
    fig_w = img_w + matrix_w
    has_captions = captions is not None and len(captions) >= n
    # Attribute tensors (n, 9) for display
    img_attr_np = img_attr[:n].cpu().numpy()
    txt_attr_np = txt_attr[:n].cpu().numpy()
    txt_mask_np = txt_mask[:n].cpu().numpy()
    # More vertical space for captions and especially attribute panel (larger text)
    fig_h = max(img_h, 5.0) + (min(5.0, 0.2 * n) if has_captions else 0) + min(6.0, 0.25 * n)
    fig = plt.figure(figsize=(fig_w, fig_h))
    if has_captions:
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1.8], width_ratios=[img_w, matrix_w], hspace=0.4)
    else:
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.8], width_ratios=[img_w, matrix_w], hspace=0.4)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    # Left: square grid of image thumbnails
    ax0.set_aspect("equal")
    for idx in range(n):
        row, col = idx // n_cols, idx % n_cols
        # y: row 0 at top (flip so imshow origin upper looks right)
        y0 = n_rows - 1 - row
        y1 = y0 + 1
        ax0.imshow(imgs[idx], extent=(col, col + 1, y0, y1), aspect="equal", origin="upper", interpolation="bilinear")
    ax0.set_xlim(0, n_cols)
    ax0.set_ylim(0, n_rows)
    ax0.set_xticks(np.arange(n_cols) + 0.5)
    ax0.set_xticklabels([str(i) for i in range(n_cols)])
    ax0.set_yticks(np.arange(n_rows) + 0.5)
    ax0.set_yticklabels([str(i) for i in range(n_rows)])
    ax0.set_xlabel("Batch index (mod {})".format(n_cols))
    ax0.set_ylabel("Batch index")
    ax0.set_title("Batch images (index i = row*{} + col)".format(n_cols))
    # Right: matrix heatmap (-1=red, 0=gray, 1=green)
    cmap = plt.matplotlib.colors.ListedColormap(["#e74c3c", "#95a5a6", "#27ae60"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax1.imshow(V_np, cmap=cmap, norm=norm)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(range(n))
    ax1.set_yticklabels(range(n))
    ax1.set_xlabel("Text index j")
    ax1.set_ylabel("Image index i")
    ax1.set_title("(i,j): 1=pos, -1=neg, 0=neutral")
    if n <= 24:
        for i in range(n):
            for j in range(n):
                ax1.text(j, i, int(V_np[i, j]), ha="center", va="center", color="black", fontsize=max(4, 10 - n // 8))
    plt.colorbar(im, ax=ax1, ticks=[-1, 0, 1], label="Label")
    if has_captions:
        ax_cap = fig.add_subplot(gs[1, :])
        ax_cap.set_axis_off()
        max_len = 80
        lines = []
        for idx in range(n):
            cap = (captions[idx] or "")[:max_len]
            if len(captions[idx] or "") > max_len:
                cap += "..."
            lines.append(f"{idx}: {cap}")
        ax_cap.text(0, 1, "\n".join(lines), transform=ax_cap.transAxes, fontsize=8, verticalalignment="top", family="monospace", wrap=True)
    # Attribute values and mask (img_attr, txt_attr, txt_mask) per batch index — larger font and panel
    ax_attr = fig.add_subplot(gs[2, :] if has_captions else gs[1, :])
    ax_attr.set_axis_off()
    attr_lines = ["i | img_attr(9) | txt_attr(9) | txt_mask(9)"]
    for idx in range(n):
        ia = ",".join(str(int(x)) for x in img_attr_np[idx])
        ta = ",".join(str(int(x)) for x in txt_attr_np[idx])
        tm = "".join("1" if txt_mask_np[idx][k] else "0" for k in range(9))
        attr_lines.append(f"{idx:2d} | {ia} | {ta} | {tm}")
    ax_attr.text(0, 1, "\n".join(attr_lines), transform=ax_attr.transAxes, fontsize=10, verticalalignment="top", family="monospace", wrap=True)
    plt.suptitle(f"Epoch {epoch} step {step}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"batch_matrix_epoch{epoch}_step{step}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved batch matrix viz to {path}")


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}
        accum_img_attr, accum_txt_attr, accum_txt_mask = [], [], []

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if len(batch) >= 5:
            images, texts, img_attr, txt_attr, txt_mask = batch[0], batch[1], batch[2], batch[3], batch[4]
            captions_batch = batch[5] if len(batch) > 5 else None
            img_attr = img_attr.to(device=device, non_blocking=True)
            txt_attr = txt_attr.to(device=device, non_blocking=True)
            txt_mask = txt_mask.to(device=device, non_blocking=True)
            loss_kw = dict(img_attr=img_attr, txt_attr=txt_attr, txt_mask=txt_mask)
        else:
            images, texts = batch[0], batch[1]
            captions_batch = None
            loss_kw = {}

        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        use_attr_loss = getattr(args, 'masked_attr_alignment', False)
        loss_kw_for_loss = loss_kw if use_attr_loss else {}
        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, **loss_kw_for_loss, output_dict=True)

                total_loss = sum(v for k, v in losses.items() if k not in _LOSS_METRIC_KEYS)
                losses["loss"] = total_loss

            backward(total_loss, scaler)
            # Optional: visualize batch matrix (1=positive, -1=negative, 0=neutral)
            if (
                is_master(args)
                and getattr(args, "viz_batch_matrix", False)
                and len(loss_kw) > 0
            ):
                every = getattr(args, "viz_batch_matrix_every", 0)
                if (every == 0 and i_accum == 0) or (every > 0 and i_accum % every == 0):
                    save_dir = os.path.join(args.checkpoint_path, "batch_matrix_viz")
                    _viz_batch_matrix(
                        images,
                        loss_kw["img_attr"],
                        loss_kw["txt_attr"],
                        loss_kw["txt_mask"],
                        save_dir,
                        step=step,
                        epoch=epoch,
                        max_batch=getattr(args, "viz_batch_matrix_max_batch", 8),
                        captions=captions_batch,
                    )
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)
                if loss_kw:
                    accum_img_attr.append(loss_kw["img_attr"])
                    accum_txt_attr.append(loss_kw["txt_attr"])
                    accum_txt_mask.append(loss_kw["txt_mask"])

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            if loss_kw:
                full_img_attr = torch.cat(accum_img_attr, dim=0)
                full_txt_attr = torch.cat(accum_txt_attr, dim=0)
                full_txt_mask = torch.cat(accum_txt_mask, dim=0)
                accum_loss_kw = dict(img_attr=full_img_attr, txt_attr=full_txt_attr, txt_mask=full_txt_mask)
            else:
                accum_loss_kw = {}
            accum_loss_kw_for_loss = accum_loss_kw if use_attr_loss else {}

            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, **accum_loss_kw_for_loss, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(v for k, v in losses.items() if k not in _LOSS_METRIC_KEYS)
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}
            accum_img_attr, accum_txt_attr, accum_txt_mask = [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            logit_scale_param = _get_logit_scale_param(model)
            if logit_scale_param is not None:
                logit_scale_param.clamp_(0, math.log(100))

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
                scalar = val.item() if torch.is_tensor(val) else val
                losses_m[key].update(scalar, batch_size)

            logit_scale_scalar = logit_scale.item()
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


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    """Run evaluation on all ranks so no rank stalls at the post-eval barrier.
    Logging and writing (tb, results.jsonl, wandb) remain rank-0 only."""
    metrics = {}
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        # unwrap DDP for single process eval
        if args.distributed and not args.horovod:
            model = model.module
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch[0], batch[1]
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
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

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            # Only rank 0 computes retrieval metrics (expensive); then broadcast to all ranks.
            if is_master(args):
                val_metrics = get_clip_metrics(
                    image_features=torch.cat(all_image_features),
                    text_features=torch.cat(all_text_features),
                    logit_scale=logit_scale.cpu(),
                    device=device,
                    log_progress=True,
                )
            else:
                val_metrics = {}
            if args.distributed and not args.horovod:
                val_metrics = broadcast_object(args, val_metrics)
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    if is_master(args):
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


def get_clip_metrics(image_features, text_features, logit_scale, chunk_size=1000, device=None, log_progress=False):
    """Compute retrieval metrics (mean rank, R@k). Uses batched similarity to avoid OOM on large N.
    When device is CUDA, runs matmuls on GPU (and uses larger chunk_size for speed); otherwise CPU."""
    n = image_features.shape[0]
    assert n == text_features.shape[0], "Paired eval requires same number of images and texts"
    image_features = image_features.detach().float()
    text_features = text_features.detach().float()
    logit_scale = logit_scale.cpu().float().item() if hasattr(logit_scale, "cpu") else float(logit_scale)
    use_cuda = device is not None and device.type == "cuda"
    # Larger chunks on GPU = fewer iterations; 2000*575k*4 bytes ~ 4.6 GB (safe on 8GB+ GPUs)
    if use_cuda and chunk_size == 1000:
        chunk_size = 2000
    num_chunks = (n + chunk_size - 1) // chunk_size
    if log_progress:
        logging.info(f"Computing retrieval metrics for N={n} (chunk_size={chunk_size}, device={'cuda' if use_cuda else 'cpu'})...")
    t0 = time.perf_counter()

    def compute_ranks_image_to_text():
        # Keep text_features on device; stream image chunks. Do argsort on GPU and only transfer rank indices to CPU.
        text_g = text_features.to(device) if use_cuda else text_features
        preds_list = []
        for c, start in enumerate(range(0, n, chunk_size)):
            if log_progress and (c % 50 == 0 or c == num_chunks - 1):
                logging.info(f"Computing retrieval metrics: image_to_text {c + 1}/{num_chunks}")
            end = min(start + chunk_size, n)
            batch_img = image_features[start:end].to(device) if use_cuda else image_features[start:end]
            logits_batch = (logit_scale * batch_img @ text_g.t())
            gt = torch.arange(start, end, device=logits_batch.device)
            ranking = torch.argsort(logits_batch, descending=True, dim=1)
            preds_batch = torch.where(ranking == gt.unsqueeze(1))[1]
            preds_list.append(preds_batch.cpu())
        return torch.cat(preds_list).numpy()

    def compute_ranks_text_to_image():
        image_g = image_features.to(device) if use_cuda else image_features
        preds_list = []
        for c, start in enumerate(range(0, n, chunk_size)):
            if log_progress and (c % 50 == 0 or c == num_chunks - 1):
                logging.info(f"Computing retrieval metrics: text_to_image {c + 1}/{num_chunks}")
            end = min(start + chunk_size, n)
            batch_txt = text_features[start:end].to(device) if use_cuda else text_features[start:end]
            logits_batch = (logit_scale * batch_txt @ image_g.t())
            gt = torch.arange(start, end, device=logits_batch.device)
            ranking = torch.argsort(logits_batch, descending=True, dim=1)
            preds_batch = torch.where(ranking == gt.unsqueeze(1))[1]
            preds_list.append(preds_batch.cpu())
        return torch.cat(preds_list).numpy()

    preds_i2t = compute_ranks_image_to_text()
    if log_progress:
        logging.info(f"Retrieval metrics image_to_text done in {time.perf_counter() - t0:.1f}s")
    t1 = time.perf_counter()
    preds_t2i = compute_ranks_text_to_image()
    if log_progress:
        logging.info(f"Retrieval metrics text_to_image done in {time.perf_counter() - t1:.1f}s")

    metrics = {}
    for name, preds in [
        ("image_to_text", preds_i2t),
        ("text_to_image", preds_t2i),
    ]:
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    if log_progress:
        logging.info(f"Retrieval metrics total: {time.perf_counter() - t0:.1f}s")
    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
