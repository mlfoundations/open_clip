from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .image_text_task import ImageTextTask


class CoCaTask(ImageTextTask):
    """CoCa training task wrapping model + CoCaLoss."""

    def __init__(
            self,
            model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            default_loss: bool = True,
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
        super().__init__(model, device=device, dtype=dtype, verbose=verbose)
        if loss is not None:
            self.loss = loss
        elif default_loss:
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
        # else: eval-only construction, no self.loss attribute

    def _build_loss_inputs(self, model_out, batch):
        """Build CoCaLoss inputs with autoregressive shift."""
        return {
            "image_features": model_out["image_features"],
            "text_features": model_out["text_features"],
            "logits": model_out["logits"][:, :-1],
            "labels": batch["text"][:, 1:],
            "logit_scale": model_out["logit_scale"],
        }

    def training_forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        model_out = self.trainable_module(**batch)
        loss_input = self._build_loss_inputs(model_out, batch)
        losses = self.loss(**loss_input, output_dict=True)
        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
        losses["loss"] = total_loss
        # Report from model_out (not loss_input): _build_loss_inputs drops logit_bias, which CoCaLoss can't take
        # but we still want to log. Matches the accum path, which captures bias from inputs_no_accum before dropping.
        return losses, self._report(model_out)

    def compute_accum_loss(self, inputs, inputs_no_accum, accum_batches):
        all_texts = torch.cat([b["text"] for b in accum_batches])
        inputs["labels"] = all_texts[:, 1:]
        inputs["logits"] = inputs["logits"][:, :-1]
        report = self._report(inputs_no_accum)  # capture before dropping logit_bias for the loss call
        # CoCaLoss doesn't accept logit_bias
        inputs_no_accum.pop("logit_bias", None)
        losses = self.loss(**inputs, **inputs_no_accum, output_dict=True)
        return losses, report
