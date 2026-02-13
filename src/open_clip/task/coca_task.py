from typing import Dict, Optional

import torch
import torch.nn as nn

from .task import CLIPTrainingTask


class CoCaTask(CLIPTrainingTask):
    """CoCa training task wrapping model + CoCaLoss."""

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
        # Filter to keys CoCaLoss expects (it doesn't accept logit_bias)
        loss_keys = ('image_features', 'text_features', 'logits', 'labels', 'logit_scale')
        loss_input = {k: v for k, v in model_out.items() if k in loss_keys}
        losses = self.loss(**loss_input, output_dict=True)
        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
        losses["loss"] = total_loss
        losses["logit_scale"] = logit_scale
        return losses
