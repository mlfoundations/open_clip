"""Training task for GenLIP generative vision-language pretraining.

GenLIP has no contrastive objective, dual tower, or paired ``*_features``; it produces next-token logits
over the concatenated ``[image_patches ; caption_tokens]`` sequence and is trained with a single LM loss.
This task therefore derives from :class:`ImageTextTask` only to reuse its NaFlex data-config plumbing and
dummy-batch scaffolding -- it overrides the forward/loss path entirely and never touches logit scale.
"""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base_task import unwrap_model
from .image_text_task import ImageTextTask


class GenLipTask(ImageTextTask):
    """GenLIP training task wrapping model + GenLipLoss (autoregressive caption loss)."""

    # Modality wiring: which batch key holds the (NaFlex) prefix dict and which kwarg the model forward
    # expects. Defaults keep GenLIP behaviour ("image"); GenLAP overrides both to "audio".
    _modality_key: str = "image"
    _modality_kwarg: str = "image"

    @property
    def data_keys(self) -> Tuple[str, ...]:
        return (self._modality_key, "text")

    def __init__(
            self,
            model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            fused_loss: bool = True,
            default_loss: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        """Wrap a GenLIP model with its training objective.

        Args:
            loss: Optional external loss module ``loss(logits, labels, output_dict=True)``. Supplying one
                forces the external-loss path (``fused_loss`` ignored) so the same model can be trained under
                a different / alternate objective.
            fused_loss: When True (default) the model computes its own memory-efficient, DDP-safe
                autoregressive loss inside ``forward(compute_loss=True)`` and the task holds NO loss module
                (nothing to construct, nothing left unused). When False, the model is driven as a plain logits
                producer and an external loss computes the objective in the task.
            default_loss: When the external-loss path is taken (``loss`` given or ``fused_loss=False``) and no
                ``loss`` was supplied, build a default :class:`GenLipLoss`. Set False for eval-only
                construction that should hold no loss module at all.
        """
        super().__init__(model, device=device, dtype=dtype, verbose=verbose)
        self.pad_id = unwrap_model(model).pad_id
        if loss is not None:
            # Explicit external objective: model is just a logits producer, loss owns the objective.
            self.loss = loss
            self.fused_loss = False
        elif fused_loss:
            # Default: model computes the fused loss internally; task carries no (would-be unused) loss.
            self.fused_loss = True
        elif default_loss:
            from open_clip.loss import GenLipLoss
            self.loss = GenLipLoss()
            self.fused_loss = False
        else:
            # Eval-only construction: no loss module, no fused loss path.
            self.fused_loss = False

    def _loss_forward(self, module: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        modality = batch[self._modality_key]
        if self.fused_loss:
            # The model computes the autoregressive caption loss internally via a memory-efficient fused
            # linear cross-entropy (over text positions only). Doing it inside the single module forward keeps
            # it DDP-safe (gradient sync hooks fire) and avoids materializing full-vocabulary logits.
            out = module(
                text=batch["text"],
                text_valid=batch.get("text_valid"),
                compute_loss=True,
                **{self._modality_kwarg: modality},
            )
            return {"caption_loss": out["loss"], "loss": out["loss"]}

        # External-loss path: drive the model as a plain logits producer and apply the autoregressive shift
        # here (CoCa-style), then let the external loss module compute the objective. NOTE: this materializes
        # the full ``[B, S, vocab]`` logits -- intended for small-vocab configs, eval/debugging, or alternate
        # objectives, NOT large-vocab (~100k) training where the fused path is required to fit in memory.
        text = batch["text"]
        text_valid = batch.get("text_valid")
        if text_valid is None:
            text_valid = text != self.pad_id
        out = module(
            text=text,
            text_valid=text_valid,
            compute_loss=False,
            **{self._modality_kwarg: modality},
        )
        ni = modality["patches"].shape[1]
        # Caption token text[:, j] (sequence position ni+j) is predicted by the logits at position ni-1+j,
        # so the text-predicting window is logits[:, ni-1:-1] -> (B, Lt, vocab).
        logits = out["logits"][:, ni - 1:-1]
        labels = torch.where(text_valid, text, torch.full_like(text, -100))
        losses = self.loss(logits, labels, output_dict=True)
        losses["loss"] = sum(v for k, v in losses.items() if k.endswith("_loss"))
        return losses

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
