from typing import Dict, Optional

import torch
import torch.nn as nn

from .task import CLIPTrainingTask


class CoCaTask(CLIPTrainingTask):
    """CoCa training task wrapping model + CoCaLoss."""

    def __init__(
            self,
            model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            caption_loss_weight: float = 2.0,
            clip_loss_weight: float = 1.0,
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
            from open_clip.loss import CoCaLoss
            self.loss = CoCaLoss(
                caption_loss_weight=caption_loss_weight,
                clip_loss_weight=clip_loss_weight,
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=cache_labels,
                rank=rank,
                world_size=world_size,
            )

    def _build_loss_inputs(self, model_out, texts):
        """Build CoCaLoss inputs with autoregressive shift."""
        return {
            "image_features": model_out["image_features"],
            "text_features": model_out["text_features"],
            "logits": model_out["logits"][:, :-1],
            "labels": texts[:, 1:],
            "logit_scale": model_out["logit_scale"],
        }

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Dict[str, torch.Tensor]:
        model_out = self.trainable_module(images, texts)
        loss_input = self._build_loss_inputs(model_out, texts)
        losses = self.loss(**loss_input, output_dict=True)
        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
        losses["loss"] = total_loss
        losses["logit_scale"] = loss_input["logit_scale"]
        return losses

    def compute_accum_loss(self, inputs, inputs_no_accum, accum_texts):
        all_texts = torch.cat(accum_texts)
        inputs["labels"] = all_texts[:, 1:]
        inputs["logits"] = inputs["logits"][:, :-1]
        # CoCaLoss doesn't accept logit_bias
        inputs_no_accum.pop("logit_bias", None)
        return self.loss(**inputs, **inputs_no_accum, output_dict=True)
