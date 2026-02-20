from typing import Dict, Optional

import torch
import torch.nn as nn

from .base_task import TrainingTask


class CLIPTask(TrainingTask):
    """Standard CLIP training task wrapping model + ClipLoss."""

    def __init__(
            self,
            model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            local_loss: bool = False,
            gather_with_grad: bool = False,
            cache_labels: bool = True,
            rank: int = 0,
            world_size: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)
        self.trainable_module = model
        if loss is not None:
            self.loss = loss
        else:
            from open_clip.loss import ClipLoss
            self.loss = ClipLoss(
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=cache_labels,
                rank=rank,
                world_size=world_size,
            )

    def training_forward(self, images: torch.Tensor, texts: torch.Tensor) -> Dict[str, torch.Tensor]:
        model_out = self.trainable_module(images, texts)
        logit_scale = model_out["logit_scale"]
        losses = self.loss(**model_out, output_dict=True)
        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
        losses["loss"] = total_loss
        losses["logit_scale"] = logit_scale
        return losses
