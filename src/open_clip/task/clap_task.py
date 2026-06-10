import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base_task import TrainingTask, unwrap_model


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

    def training_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_out = self.trainable_module(audio=batch["audio"], text=batch["text"])
        loss_inputs = self._loss_inputs(model_out)
        logit_scale = loss_inputs["logit_scale"]
        losses = self.loss(**loss_inputs, output_dict=True)
        total_loss = sum(v for k, v in losses.items() if k.endswith("_loss"))
        losses["loss"] = total_loss
        losses["logit_scale"] = logit_scale
        return losses

    def eval_forward(self, batch: Dict[str, torch.Tensor]):
        inputs = {key: batch[key] for key in self.data_keys if key in batch}
        return self.get_trainable_module(use_ema=True)(**inputs)

    def compute_accum_loss(self, inputs, inputs_no_accum, accum_batches):
        loss_inputs = {
            "image_features": inputs["audio_features"],
            "text_features": inputs["text_features"],
            **inputs_no_accum,
        }
        return self.loss(**loss_inputs, output_dict=True)

    def create_dummy_batch(
            self,
            batch_size: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        model = unwrap_model(self.trainable_module)
        audio_cfg = model.audio.cfg

        if getattr(audio_cfg, "model_type", "").lower() == "naflexvit":
            # NaFlexClap: the audio tower consumes a NaFlex patch dict, not a raw waveform.
            n = 16  # arbitrary token count for the dummy
            dummy_audio = {
                "patches": torch.zeros(batch_size, n, audio_cfg.in_chans * audio_cfg.patch_freq * audio_cfg.patch_time,
                                       device=device, dtype=dtype),
                "patch_coord": torch.zeros(batch_size, n, 2, dtype=torch.long, device=device),
                "patch_valid": torch.ones(batch_size, n, dtype=torch.bool, device=device),
            }
            dummy_audio["patch_coord"][..., 1] = torch.arange(n, device=device)  # time index
            return {
                "audio": dummy_audio,
                "text": torch.zeros(batch_size, model.context_length, dtype=torch.long, device=device),
            }

        dummy_audio = {
            "waveform": torch.zeros(batch_size, audio_cfg.clip_samples, device=device, dtype=dtype),
            "longer": torch.zeros(batch_size, dtype=torch.bool, device=device),
        }
        if audio_cfg.enable_fusion:
            from open_clip.audio.transform import get_audio_frame_count

            audio_frames = get_audio_frame_count(audio_cfg)
            dummy_audio["mel_fusion"] = torch.zeros(
                batch_size,
                4,
                audio_frames,
                audio_cfg.mel_bins,
                device=device,
                dtype=dtype,
            )
        return {
            "audio": dummy_audio,
            "text": torch.zeros(batch_size, model.context_length, dtype=torch.long, device=device),
        }

    def clamp_logit_scale(self, max_val: float = math.log(100)):
        model = unwrap_model(self.trainable_module)
        if hasattr(model, "logit_scale"):
            with torch.no_grad():
                model.logit_scale.clamp_(0, max_val)

    def ddp_extra_kwargs(self):
        return {"find_unused_parameters": True}
