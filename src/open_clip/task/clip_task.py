from typing import Dict, Optional

import torch
import torch.nn as nn

from .task import CLIPTrainingTask


class CLIPTask(CLIPTrainingTask):
    """Standard CLIP training task wrapping model + ClipLoss."""

    def __init__(
            self,
            model: nn.Module,
            loss: nn.Module,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)
        self.trainable_module = model
        self.loss = loss

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Dict[str, torch.Tensor]:
        model_out = self.trainable_module(images, texts)
        logit_scale = model_out["logit_scale"]
        losses = self.loss(**model_out, output_dict=True)
        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
        losses["loss"] = total_loss
        losses["logit_scale"] = logit_scale
        return losses
