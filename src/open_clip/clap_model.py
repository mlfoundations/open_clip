"""CLAP model."""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .audio.config import CLIPAudioCfg
from .model import CLIPTextCfg, _build_text_tower


def _build_audio_tower(
        embed_dim: int,
        audio_cfg: CLIPAudioCfg,
):
    from .audio.tower import AudioTower

    return AudioTower(audio_cfg, embed_dim)


class CLAP(nn.Module):
    """Contrastive Language-Audio Pretraining model."""

    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            audio_cfg: CLIPAudioCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.audio = _build_audio_tower(embed_dim, audio_cfg)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self._training_head = getattr(self.audio, 'training_head', False)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        self.audio.set_grad_checkpointing(enable, impl=impl)
        self.text.set_grad_checkpointing(enable, impl=impl)

    def no_weight_decay(self):
        no_wd = set()
        if hasattr(self.audio, 'no_weight_decay'):
            for n in self.audio.no_weight_decay():
                no_wd.add('audio.' + n)
        if hasattr(self.text, 'no_weight_decay'):
            for n in self.text.no_weight_decay():
                no_wd.add('text.' + n)
        return no_wd

    def encode_audio(self, audio, normalize: bool = False):
        if self._training_head:
            features = self.audio(audio, apply_proj=False)
        else:
            features = self.audio(audio)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, audio, text):
        audio_features = self.encode_audio(audio, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        audio_logits = self.logit_scale.exp() * audio_features @ text_features.T
        if self.logit_bias is not None:
            audio_logits += self.logit_bias
        text_logits = audio_logits.T
        return audio_logits, text_logits

    def forward(
            self,
            audio=None,
            text: Optional[torch.Tensor] = None,
    ):
        if self._training_head and audio is not None:
            audio_features = self.encode_audio(audio, normalize=True)
            audio_features = self.audio.proj(audio_features)
            audio_features = F.normalize(audio_features, dim=-1)
        else:
            audio_features = self.encode_audio(audio, normalize=True) if audio is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "audio_features": audio_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias.clone()
            return out_dict

        if self.logit_bias is not None:
            return audio_features, text_features, self.logit_scale.exp(), self.logit_bias.clone()
        return audio_features, text_features, self.logit_scale.exp()
