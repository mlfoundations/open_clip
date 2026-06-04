import logging
import pickle
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLIPAudioCfg

_logger = logging.getLogger(__name__)

_HTSAT_CONFIGS = {
    "tiny": dict(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32]),
    "base": dict(embed_dim=128, depths=[2, 2, 12, 2], num_heads=[4, 8, 16, 32]),
    "large": dict(embed_dim=256, depths=[2, 2, 12, 2], num_heads=[4, 8, 16, 32]),
}


def _htsat_output_dim(swin_embed_dim, num_layers=4):
    return int(swin_embed_dim * 2 ** (num_layers - 1))


def _load_audio_checkpoint(path: Union[str, Path], weights_only: bool = True):
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except pickle.UnpicklingError as e:
        if weights_only and "Weights only load failed" in str(e):
            _logger.warning(
                "Security warning: loading audio encoder checkpoint %s with weights_only=False because "
                "weights_only=True rejected a non-tensor checkpoint payload.",
                path,
            )
            return torch.load(path, map_location="cpu", weights_only=False)
        raise


class AudioTower(nn.Module):
    """Audio encoder wrapper for CLAP models."""

    def __init__(self, audio_cfg: CLIPAudioCfg, embed_dim: int):
        super().__init__()
        if isinstance(audio_cfg, dict):
            audio_cfg = CLIPAudioCfg(**audio_cfg)
        self.cfg = audio_cfg
        self.pre_norm = audio_cfg.pre_norm
        self.training_head = audio_cfg.training_head
        self._is_whisper = audio_cfg.model_type.lower() == "whisper"

        if audio_cfg.model_type == "HTSAT":
            from .htsat import HTSATEncoder

            if audio_cfg.model_name not in _HTSAT_CONFIGS:
                raise ValueError(f"Unknown HTSAT variant: {audio_cfg.model_name}")
            htsat_cfg = _HTSAT_CONFIGS[audio_cfg.model_name]
            self.encoder = HTSATEncoder(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=audio_cfg.class_num,
                window_size=8,
                config=audio_cfg,
                enable_fusion=audio_cfg.enable_fusion,
                fusion_type=audio_cfg.fusion_type,
                **htsat_cfg,
            )
            audio_width = _htsat_output_dim(htsat_cfg["embed_dim"], num_layers=len(htsat_cfg["depths"]))
        elif self._is_whisper:
            from .whisper import create_whisper_model

            self.encoder = create_whisper_model(audio_cfg, output_dim=embed_dim)
            audio_width = embed_dim
        elif getattr(audio_cfg, "model_type", "").lower() == "naflexvit":
            from .naflex_tower import NaFlexAudioEncoder

            self.encoder = NaFlexAudioEncoder(audio_cfg)
            audio_width = self.encoder.embed_dim
        else:
            raise ValueError(f"Unsupported audio model type: {audio_cfg.model_type}")

        act_layer = nn.ReLU() if audio_cfg.proj_act == "relu" else nn.GELU()
        self.proj = nn.Sequential(
            nn.Linear(audio_width, embed_dim),
            act_layer,
            nn.Linear(embed_dim, embed_dim),
        )

    def set_grad_checkpointing(self, enable: bool = True, impl: str = "inline"):
        """Enable checkpointing, falling back to HTSAT-style ``use_checkpoint`` attrs."""
        if hasattr(self.encoder, "set_grad_checkpointing"):
            self.encoder.set_grad_checkpointing(enable, impl=impl)
            return
        for module in self.encoder.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = enable

    def no_weight_decay(self):
        no_wd = set()
        if hasattr(self.encoder, "no_weight_decay"):
            for name in self.encoder.no_weight_decay():
                no_wd.add("encoder." + name)
        return no_wd

    def layer_groups(self, pooler_in_head: bool = True):
        """Ordered ``(name, [members])`` groups for layer-wise LR decay / lock: the encoder's own
        ``layer_groups`` (trunk, input -> output) followed by this tower's projection head as the top group.
        Only encoders that expose ``layer_groups`` (e.g. the NaFlex spectrogram-ViT) are supported; HTSAT/Whisper
        do not, so layer-wise LR decay on the audio tower raises a clear error for them upstream.
        """
        if not hasattr(self.encoder, "layer_groups"):
            raise ValueError(
                f"audio layer-wise LR decay is not supported for encoder {type(self.encoder).__name__} "
                f"(no layer_groups); only the NaFlex audio encoder supports it."
            )
        groups = list(self.encoder.layer_groups(pooler_in_head))
        if next(self.proj.parameters(), None) is not None:
            groups.append(("proj", [self.proj]))
        return groups

    def load_pretrained_encoder(self, path: Union[str, Path], weights_only: bool = True):
        checkpoint = _load_audio_checkpoint(path, weights_only=weights_only)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Audio checkpoint {path} did not contain a state dict.")

        state_dict = {}
        prefix = "encoder." if self._is_whisper else "sed_model."
        for key, value in checkpoint.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith(prefix):
                key = key[len(prefix) :]
            state_dict[key] = value
        return self.encoder.load_state_dict(state_dict, strict=False)

    def forward(self, audio, apply_proj=True):
        device = self.proj[0].weight.device
        if self._is_whisper:
            audio_input = dict(audio)
            audio_input["waveform"] = audio["waveform"].to(device=device, non_blocking=True)
            out = self.encoder(audio_input)
            features = out["embedding"].mean(dim=1)
        else:
            out = self.encoder(audio, device=device)
            features = out["embedding"]

        if self.pre_norm:
            features = F.normalize(features, dim=-1)
        if apply_proj:
            features = self.proj(features)
        return features
