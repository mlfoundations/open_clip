"""Training task for GenLIP generative vision-language pretraining.

GenLIP has no contrastive objective, dual tower, or paired ``*_features``; it produces next-token logits
over the concatenated ``[image_patches ; caption_tokens]`` sequence and is trained with a single LM loss.
This task therefore derives from :class:`ImageTextTask` only to reuse its NaFlex data-config plumbing and
dummy-batch scaffolding -- it overrides the forward/loss path entirely and never touches logit scale.
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base_task import unwrap_model
from .image_text_task import ImageTextTask


class GenLipTask(ImageTextTask):
    """GenLIP training task wrapping model + GenLipLoss (autoregressive caption loss)."""

    def __init__(
            self,
            model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            default_loss: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(model, device=device, dtype=dtype, verbose=verbose)
        self.pad_id = unwrap_model(model).pad_id
        if loss is not None:
            self.loss = loss
        elif default_loss:
            from open_clip.loss import GenLipLoss
            self.loss = GenLipLoss()
        # else: eval-only construction, no self.loss attribute

    def _loss_forward(self, module: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # The model computes the autoregressive caption loss internally via a memory-efficient fused
        # linear cross-entropy (over text positions only). Doing it inside the single module forward keeps
        # it DDP-safe (gradient sync hooks fire) and avoids materializing full-vocabulary logits.
        out = module(
            image=batch["image"],
            text=batch["text"],
            text_valid=batch.get("text_valid"),
            compute_loss=True,
        )
        return {"caption_loss": out["loss"], "loss": out["loss"]}

    def training_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._loss_forward(self.trainable_module, batch)

    def eval_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._loss_forward(self.get_trainable_module(use_ema=True), batch)

    def create_dummy_batch(
            self,
            image_size=None,
            context_length: Optional[int] = None,
            batch_size: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        batch = super().create_dummy_batch(
            image_size=image_size,
            context_length=context_length,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        batch["text_valid"] = torch.ones_like(batch["text"], dtype=torch.bool)
        return batch
