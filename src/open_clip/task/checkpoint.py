"""Checkpoint save/load utilities for TrainingTask.

Full checkpoints: single .pt file, rank 0 writes, torch.save/load.
Sharded checkpoints: DCP directory, all ranks write, per-rank shards.
"""
import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from .task import TrainingTask, unwrap_model


def _get_optim_state_dict(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        fsdp_enabled: bool,
) -> dict:
    """Get optimizer state dict, handling FSDP2 sharded state."""
    if fsdp_enabled:
        from torch.distributed.checkpoint.state_dict import (
            get_optimizer_state_dict,
            StateDictOptions,
        )
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return get_optimizer_state_dict(model, optimizer, options=options)
    return optimizer.state_dict()


def _load_optim_state_dict(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        state_dict: dict,
        fsdp_enabled: bool,
):
    """Load optimizer state dict, handling FSDP2 sharded state."""
    if fsdp_enabled:
        from torch.distributed.checkpoint.state_dict import (
            set_optimizer_state_dict,
            StateDictOptions,
        )
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        set_optimizer_state_dict(
            model, optimizer,
            optim_state_dict=state_dict,
            options=options,
        )
    else:
        optimizer.load_state_dict(state_dict)


def save_checkpoint(
        task: TrainingTask,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        scaler: Optional[torch.amp.GradScaler] = None,
        name: Optional[str] = None,
) -> dict:
    """Save full checkpoint as a single .pt file.

    Under FSDP, this is a collective operation (all ranks must call).
    Returns the checkpoint dict (caller decides when/where to torch.save).
    """
    # task.state_dict() handles FSDP gather + normalize + EMA internally
    task_sd = task.state_dict()
    model = unwrap_model(task.trainable_module)
    optim_sd = _get_optim_state_dict(model, optimizer, task._fsdp_enabled)

    checkpoint_dict = {
        "epoch": epoch,
        "name": name,
        "state_dict": task_sd["state_dict"],
        "optimizer": optim_sd,
    }
    if 'state_dict_ema' in task_sd:
        checkpoint_dict["state_dict_ema"] = task_sd["state_dict_ema"]
    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()
    return checkpoint_dict


def load_checkpoint(
        task: TrainingTask,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        is_distributed: bool = False,
) -> int:
    """Load full checkpoint from a .pt file. Returns start_epoch.

    Under FSDP, this is a collective operation (all ranks must call).
    """
    from open_clip_train.file_utils import pt_load
    checkpoint = pt_load(path, map_location='cpu')

    if 'epoch' in checkpoint:
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if not is_distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        task.load_state_dict({"state_dict": sd})
        if optimizer is not None and "optimizer" in checkpoint:
            model = unwrap_model(task.trainable_module)
            _load_optim_state_dict(model, optimizer, checkpoint["optimizer"], task._fsdp_enabled)
        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        logging.info(f"=> resuming checkpoint '{path}' (epoch {start_epoch})")
    else:
        start_epoch = 0
        task.load_state_dict({"state_dict": checkpoint})
        logging.info(f"=> loaded checkpoint '{path}' (epoch {start_epoch})")

    return start_epoch


def save_sharded_checkpoint(
        task: TrainingTask,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str,
        epoch: int,
        scaler: Optional[torch.amp.GradScaler] = None,
        name: Optional[str] = None,
        is_master: bool = True,
) -> None:
    """Save sharded DCP checkpoint to a directory. All ranks must call.

    Each rank writes its own shard files. Metadata (epoch, scaler) is
    saved as a small .pt file by the master rank only.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model = unwrap_model(task.trainable_module)
    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.save({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=checkpoint_dir)

    # Metadata on master rank only
    if is_master:
        metadata = {"epoch": epoch, "name": name}
        if scaler is not None:
            metadata["scaler"] = scaler.state_dict()
        torch.save(metadata, os.path.join(checkpoint_dir, "_metadata_extra.pt"))


def load_sharded_checkpoint(
        task: TrainingTask,
        checkpoint_dir: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
) -> int:
    """Load sharded DCP checkpoint from a directory. All ranks must call.

    Returns start_epoch.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model = unwrap_model(task.trainable_module)

    # Build empty DTensor scaffolding, then let DCP fill it
    if optimizer is not None:
        model_sd, optim_sd = get_state_dict(model, optimizer)
        dcp.load({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=checkpoint_dir)
        set_state_dict(
            model, optimizer,
            model_state_dict=model_sd,
            optim_state_dict=optim_sd,
        )
    else:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
        model_sd = get_model_state_dict(model)
        dcp.load({"model": model_sd}, checkpoint_id=checkpoint_dir)
        set_model_state_dict(model, model_sd)

    # Load metadata
    start_epoch = 0
    metadata_path = os.path.join(checkpoint_dir, "_metadata_extra.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, map_location='cpu', weights_only=True)
        start_epoch = metadata.get("epoch", 0)
        if scaler is not None and "scaler" in metadata:
            scaler.load_state_dict(metadata["scaler"])

    logging.info(f"=> resuming sharded checkpoint '{checkpoint_dir}' (epoch {start_epoch})")
    return start_epoch
