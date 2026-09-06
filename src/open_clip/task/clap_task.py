import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base_task import TrainingTask, unwrap_model
from ..audio.transform import create_dummy_audio


class CLAPTask(TrainingTask):
    """Audio + text contrastive task wrapping CLAP + ClipLoss."""

    @property
    def data_keys(self):
        return ("audio", "text")

    def __init__(
            self,
            model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            default_loss: bool = True,
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
            from open_clip.loss import ClipLoss

            self.loss = ClipLoss(
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=cache_labels,
                rank=rank,
                world_size=world_size,
            )

    def _loss_inputs(self, model_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = {
            "image_features": model_out["audio_features"],
            "text_features": model_out["text_features"],
            "logit_scale": model_out["logit_scale"],
        }
        if "logit_bias" in model_out:
            inputs["logit_bias"] = model_out["logit_bias"]
        return inputs

    def training_forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        model_out = self.trainable_module(audio=batch["audio"], text=batch["text"])
        loss_inputs = self._loss_inputs(model_out)
        losses = self.loss(**loss_inputs, output_dict=True)
        total_loss = sum(v for k, v in losses.items() if k.endswith("_loss"))
        losses["loss"] = total_loss
        # Report from raw model_out (the source of truth for logit_scale/logit_bias) — uniform with clip/coca.
        return losses, self._report(model_out)

    def eval_forward(self, batch: Dict[str, torch.Tensor]):
        inputs = {key: batch[key] for key in self.data_keys if key in batch}
        return self.get_trainable_module(use_ema=True)(**inputs)

    def compute_accum_loss(self, inputs, inputs_no_accum, accum_batches):
        loss_inputs = {
            "image_features": inputs["audio_features"],
            "text_features": inputs["text_features"],
            **inputs_no_accum,
        }
        losses = self.loss(**loss_inputs, output_dict=True)
        return losses, self._report(inputs_no_accum)

    def create_dummy_batch(
            self,
            batch_size: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        model = unwrap_model(self.trainable_module)
        return {
            "audio": create_dummy_audio(model.audio.cfg, batch_size=batch_size, device=device, dtype=dtype),
            "text": torch.zeros(batch_size, model.context_length, dtype=torch.long, device=device),
        }

    def clamp_logit_scale(self, max_val: float = math.log(100)):
        model = unwrap_model(self.trainable_module)
        if hasattr(model, "logit_scale"):
            with torch.no_grad():
                model.logit_scale.clamp_(0, max_val)

    def ddp_extra_kwargs(self):
        # HTSAT feature-fusion towers exercise their fusion modules only for batches containing 'longer' clips,
        # so those params receive no grads on all-short batches and DDP must search for unused parameters.
        # Static audio towers (no fusion, NaFlexClap) skip the search -- it costs an extra graph traversal
        # every step (and PyTorch warns when nothing unused is found). Fusion is declared on the audio tower
        # cfg; the module-attribute scan is a fallback for towers built without a cfg (HTSAT sets both).
        model = unwrap_model(self.trainable_module)
        audio_cfg = getattr(getattr(model, "audio", None), "cfg", None)
        fusion = bool(getattr(audio_cfg, "enable_fusion", False))
        fusion = fusion or any(getattr(m, "enable_fusion", False) for m in model.modules())
        return {"find_unused_parameters": True} if fusion else {}
