"""Audio encoder configuration and tower wrapper for CLAP models.

Supports HTSAT and Whisper audio encoders.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .htsat import HTSAT_Swin_Transformer


@dataclass
class CLIPAudioCfg:
    model_type: str = "HTSAT"   # "HTSAT" or "whisper"
    model_name: str = "tiny"    # HTSAT: tiny/base/large; Whisper: tiny/base/small/medium/large
    audio_length: int = 1024
    clip_samples: int = 480000
    sample_rate: int = 48000
    mel_bins: int = 64
    window_size: int = 1024
    hop_size: int = 480
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    enable_fusion: bool = False
    fusion_type: str = "aff_2d"
    pre_norm: bool = False  # L2-normalize encoder output before projection
    proj_act: str = "gelu"  # activation in projection MLP: "gelu" or "relu"
    training_head: bool = False  # if True, projection is a training-only head (not used in encode_audio)
    pretrained: bool = False  # load pretrained encoder weights (Whisper: OpenAI weights)


_HTSAT_CONFIGS = {
    "tiny":  dict(embed_dim=96,  depths=[2, 2, 6, 2],  num_heads=[4, 8, 16, 32]),
    "base":  dict(embed_dim=128, depths=[2, 2, 12, 2], num_heads=[4, 8, 16, 32]),
    "large": dict(embed_dim=256, depths=[2, 2, 12, 2], num_heads=[4, 8, 16, 32]),
}


def _htsat_output_dim(swin_embed_dim, num_layers=4):
    """Compute HTSAT output dimension from Swin embed_dim."""
    return int(swin_embed_dim * 2 ** (num_layers - 1))


class AudioTower(nn.Module):
    """Wrapper around an audio encoder (HTSAT or Whisper) with projection to CLAP embedding space."""

    def __init__(self, audio_cfg: CLIPAudioCfg, embed_dim: int):
        super().__init__()
        self.pre_norm = audio_cfg.pre_norm
        self.training_head = audio_cfg.training_head
        self._is_whisper = (audio_cfg.model_type == "whisper")

        if audio_cfg.model_type == "HTSAT":
            assert audio_cfg.model_name in _HTSAT_CONFIGS, f"Unknown HTSAT variant: {audio_cfg.model_name}"
            htsat_cfg = _HTSAT_CONFIGS[audio_cfg.model_name]
            self.encoder = HTSAT_Swin_Transformer(
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

        elif audio_cfg.model_type == "whisper":
            from .whisper import create_whisper_model
            self.encoder = create_whisper_model(audio_cfg, output_dim=embed_dim)
            audio_width = embed_dim  # WhisperEncoder.proj maps n_state → output_dim

        else:
            raise ValueError(f"Unsupported audio model type: {audio_cfg.model_type}")

        act_layer = nn.ReLU() if audio_cfg.proj_act == "relu" else nn.GELU()
        self.proj = nn.Sequential(
            nn.Linear(audio_width, embed_dim),
            act_layer,
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, audio, apply_proj=True):
        device = self.proj[0].weight.device

        if self._is_whisper:
            # Whisper expects {"waveform": tensor} — move waveform to encoder device
            audio_input = dict(audio)
            audio_input["waveform"] = audio["waveform"].to(device=device, non_blocking=True)
            out = self.encoder(audio_input)
            features = out["embedding"]  # (B, T, D) — sequence output
            features = features.mean(dim=1)  # (B, D) — clip-level pooling
        else:
            # HTSAT
            out = self.encoder(audio, device=device)
            features = out["embedding"]  # (B, audio_width)

        if self.pre_norm:
            features = F.normalize(features, dim=-1)
        if apply_proj:
            return self.proj(features)   # (B, embed_dim)
        return features
