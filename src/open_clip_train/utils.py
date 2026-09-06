"""Utilities shared by the task and legacy trainers."""
import random

import numpy as np
import torch


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def torch_compile_kwargs(args):
    """Translate trainer options consistently for model, task, and whole-step compilation."""
    return {key: value for key in ("backend", "mode", "dynamic")
            if (value := getattr(args, f"torchcompile_{key}", None)) is not None}


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


def pop_accum_scalars(model_out, is_last_step):
    """Scalars see the full effective batch on every replay; differentiate them only once."""
    scalars = {"logit_scale": model_out.pop("logit_scale")}
    if "logit_bias" in model_out:
        scalars["logit_bias"] = model_out.pop("logit_bias")
    return scalars if is_last_step else {key: value.detach() for key, value in scalars.items()}


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()
