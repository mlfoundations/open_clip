"""Training task for GenLAP generative audio-language pretraining.

Identical generative LM objective and wiring as :class:`GenLipTask` -- the only difference is the prefix
modality: a NaFlex log-mel spectrogram supplied under ``batch['audio']`` and forwarded as ``model(audio=...)``.
All of the loss logic (fused vs external, autoregressive shift) is inherited unchanged.
"""
from typing import Any, Dict, Optional

import torch

from .base_task import unwrap_model
from .genlip_task import GenLipTask


class GenLapTask(GenLipTask):
    """GenLAP training task: GenLIP's generative objective over a NaFlex spectrogram prefix."""

    _modality_key: str = "audio"
    _modality_kwarg: str = "audio"

    def create_dummy_batch(
            self,
            image_size=None,
            context_length: Optional[int] = None,
            batch_size: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            audio_tokens: int = 16,
    ) -> Dict[str, Any]:
        """Audio-prefix dummy batch for FSDP eval scaffolding (analog of the image version)."""
        model = unwrap_model(self.trainable_module)
        if context_length is None:
            context_length = model.context_length

        cfg = model.audio_cfg
        patches = torch.zeros(batch_size, audio_tokens, cfg.patch_dim, device=device, dtype=dtype)
        coord = torch.zeros(batch_size, audio_tokens, 2, dtype=torch.long, device=device)
        coord[..., 1] = torch.arange(audio_tokens, device=device)  # time index along the variable axis
        patch_valid = torch.ones(batch_size, audio_tokens, dtype=torch.bool, device=device)
        text = torch.zeros(batch_size, context_length, device=device, dtype=torch.long)
        return {
            "audio": {"patches": patches, "patch_coord": coord, "patch_valid": patch_valid},
            "text": text,
            "text_valid": torch.ones_like(text, dtype=torch.bool),
        }
