from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base_task import unwrap_model
from .image_text_task import ImageTextTask


class CoCaTask(ImageTextTask):
    """CoCa / MaMMUT training task wrapping model + CoCaLoss.

    Caption labels are AR-shifted and masked to -100 at invalid positions using the batch
    ``text_valid`` ([B, L] bool/int, True/1 = real token). Batches without a mask fall back to
    the pad-value derivation ``text != model.pad_id`` -- exact for tokenizers with a reserved pad,
    and the historical convention (which also drops genuine tokens equal to the fill id, e.g.
    SimpleTokenizer id 0) otherwise.
    """

    def __init__(
            self,
            model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            default_loss: bool = True,
            caption_loss_weight: float = 2.0,
            clip_loss_weight: float = 1.0,
            fused_caption_loss: bool = False,
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
        # pad-value fallback for batches without a text_valid mask (see _caption_labels)
        self.pad_id = getattr(unwrap_model(model), 'pad_id', 0)
        # fused: labels are passed INTO the model forward, which computes caption_loss via
        # fused_linear_cross_entropy (no [B, L, V] logits materialized); CoCaLoss then only
        # applies the loss weighting. Legacy (False): model returns logits, CoCaLoss computes CE.
        self.fused_caption_loss = bool(fused_caption_loss)
        if loss is not None:
            self.loss = loss
        elif default_loss:
            from open_clip.loss import CoCaLoss
            self.loss = CoCaLoss(
                caption_loss_weight=caption_loss_weight,
                clip_loss_weight=clip_loss_weight,
                pad_id=None,  # labels arrive pre-masked to -100 (built in _caption_labels)
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=cache_labels,
                rank=rank,
                world_size=world_size,
            )
        # else: eval-only construction, no self.loss attribute

    def _caption_labels(
            self,
            text: torch.Tensor,
            text_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """AR-shifted caption labels with invalid positions masked to -100."""
        valid = text_valid.bool() if text_valid is not None else text != self.pad_id
        return text[:, 1:].masked_fill(~valid[:, 1:], -100)

    def create_dummy_batch(self, *args, **kwargs):
        batch = super().create_dummy_batch(*args, **kwargs)
        # explicit all-valid mask: dummy tokens are random, so pad-value fallback would be noise
        batch["text_valid"] = torch.ones_like(batch["text"], dtype=torch.bool)
        return batch

    def _build_loss_inputs(self, model_out, batch):
        """Build CoCaLoss inputs with autoregressive shift and -100 label masking."""
        inputs = {
            "image_features": model_out["image_features"],
            "text_features": model_out["text_features"],
            "logit_scale": model_out["logit_scale"],
        }
        if "caption_loss" in model_out:
            # fused path: the model already reduced the caption term (labels went into forward)
            inputs["caption_loss"] = model_out["caption_loss"]
        else:
            inputs["logits"] = model_out["logits"][:, :-1]
            inputs["labels"] = self._caption_labels(batch["text"], batch.get("text_valid"))
        return inputs

    def training_forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        if self.fused_caption_loss:
            labels = self._caption_labels(batch["text"], batch.get("text_valid"))
            model_out = self.trainable_module(**batch, labels=labels)
        else:
            model_out = self.trainable_module(**batch)
        loss_input = self._build_loss_inputs(model_out, batch)
        losses = self.loss(**loss_input, output_dict=True)
        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
        losses["loss"] = total_loss
        # Report from model_out (not loss_input): _build_loss_inputs drops logit_bias, which CoCaLoss can't take
        # but we still want to log. Matches the accum path, which captures bias from inputs_no_accum before dropping.
        return losses, self._report(model_out)

    def compute_accum_loss(self, inputs, inputs_no_accum, accum_batches):
        if self.fused_caption_loss:
            # The accum feature-cache replays micro-batches against cached contrastive features; the
            # fused caption term would need per-micro-batch loss accumulation weighted by valid-token
            # counts to stay exact. Not wired yet -- use the legacy (logits) path with --accum-freq.
            raise NotImplementedError(
                "fused caption loss does not support --accum-freq > 1 yet; "
                "drop --fused-caption-loss or set --accum-freq 1")
        all_texts = torch.cat([b["text"] for b in accum_batches])
        # derive validity per batch (masks may be present for some accum batches and not others)
        all_valid = torch.cat([
            b["text_valid"].bool() if b.get("text_valid") is not None else b["text"] != self.pad_id
            for b in accum_batches
        ])
        inputs["labels"] = self._caption_labels(all_texts, all_valid)
        inputs["logits"] = inputs["logits"][:, :-1]
        report = self._report(inputs_no_accum)  # capture before dropping logit_bias for the loss call
        # CoCaLoss doesn't accept logit_bias
        inputs_no_accum.pop("logit_bias", None)
        losses = self.loss(**inputs, **inputs_no_accum, output_dict=True)
        return losses, report
