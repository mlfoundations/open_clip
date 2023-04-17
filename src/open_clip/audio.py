from typing import Callable, Optional, Sequence, Tuple, Optional

import numpy as np
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from torchaudio.transforms import Spectrogram, TimeStretch, FrequencyMasking, TimeMasking

from .utils import to_2tuple
from .model import CLIPTextCfg, CLIPVisionCfg, _build_text_tower

from .transformer import (
    VisionTransformer,
    LayerNormFp32,
    LayerNorm,
    QuickGELU
)

# audio spectrogram transformer

class AudioSpectrogramTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            attentional_pool: bool = False,
            n_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            spec_n_fft: int = 128,
            spec_power: int = 2,
            spec_win_length: int = 24,
            spec_hop_length: Optional[int] = None,
            spec_pad: int = 0,
            spec_center: bool = True,
            spec_pad_mode: str = 'reflect',
            aug_stretch_factor: float = 0.8,
            aug_freq_mask: int = 80,
            aug_time_mask: int = 80,
    ):
        super().__init__()

        self.patch_size = to_2tuple(patch_size)

        self.spec = Spectrogram(
            n_fft=spec_n_fft,
            power=spec_power,
            win_length=spec_win_length,
            hop_length=spec_hop_length,
            pad=spec_pad,
            center=spec_center,
            pad_mode=spec_pad_mode
        )

        # spec augment - https://arxiv.org/abs/1904.08779

        self.aug = torch.nn.Sequential(
            TimeStretch(aug_stretch_factor, fixed_rate=True),
            FrequencyMasking(freq_mask_param=aug_freq_mask),
            TimeMasking(time_mask_param=aug_time_mask),
        )

        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            global_average_pool=global_average_pool,
            attentional_pool=attentional_pool,
            n_queries=n_queries,
            attn_pooler_heads=attn_pooler_heads,
            output_dim=output_dim,
            patch_dropout=patch_dropout,
            act_layer=act_layer,
            norm_layer=norm_layer,
            output_tokens=output_tokens,
            channels=1
        )

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        self.vit.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def init_parameters(self):
        self.vit.init_parameters()

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.vit.set_grad_checkpointing(enable=enable)

    def forward(self, x: torch.Tensor, should_augment: bool = True):
        assert x.ndim in {2, 3, 4}   # can be either wave (batch, time) or spectrogram (batch, freq, time) | (batch, 1, freq, time)
        is_spectrogram = x.ndim >= 3

        if not is_spectrogram:
            x = self.spec(x)

        if self.training and should_augment:
            x = self.aug(x)

        # automatically crop if audio does not yield a 2d spectrogram that is divisible by patch sizes

        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_size

        rounded_height = height // patch_height * patch_height
        rounded_width = width // patch_width * patch_width

        if (height, width) != (rounded_height, rounded_width):
            print(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        x = x[..., :rounded_height, :rounded_width]

        # pass maybe cropped spectrogram to vit

        if x.ndim == 3:
            x = x[:, None, ...]

        return self.vit(x)

# audio class config

@dataclass
class CLIPAudioCfg(CLIPVisionCfg):
    spec_n_fft: int = 128
    spec_power: int = 2
    spec_win_length: int = 24
    spec_hop_length: Optional[int] = None
    spec_pad: int = 0
    spec_center: bool = True
    spec_pad_mode: str = 'reflect'
    aug_stretch_factor: float = 0.8
    aug_freq_mask: int = 80
    aug_time_mask: int = 80

# factory method for building audio tower

def _build_audio_tower(
        embed_dim: int,
        audio_cfg: CLIPAudioCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(audio_cfg, dict):
        audio_cfg = CLIPAudioCfg(**audio_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    audio_heads = audio_cfg.width // audio_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    audio = AudioSpectrogramTransformer(
        image_size=audio_cfg.image_size,
        patch_size=audio_cfg.patch_size,
        width=audio_cfg.width,
        layers=audio_cfg.layers,
        heads=audio_heads,
        mlp_ratio=audio_cfg.mlp_ratio,
        ls_init_value=audio_cfg.ls_init_value,
        patch_dropout=audio_cfg.patch_dropout,
        global_average_pool=audio_cfg.global_average_pool,
        attentional_pool=audio_cfg.attentional_pool,
        n_queries=audio_cfg.n_queries,
        attn_pooler_heads=audio_cfg.attn_pooler_heads,
        output_tokens=audio_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
        spec_n_fft=audio_cfg.spec_n_fft,
        spec_power=audio_cfg.spec_power,
        spec_win_length=audio_cfg.spec_win_length,
        spec_hop_length=audio_cfg.spec_hop_length,
        spec_pad=audio_cfg.spec_pad,
        spec_center=audio_cfg.spec_center,
        spec_pad_mode=audio_cfg.spec_pad_mode,
        aug_stretch_factor=audio_cfg.aug_stretch_factor,
        aug_freq_mask=audio_cfg.aug_freq_mask,
        aug_time_mask=audio_cfg.aug_time_mask
    )

    return audio

# audio clip

class AudioCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim,
            text_cfg: CLIPTextCfg,
            audio_cfg: CLIPAudioCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        audio_cfg = CLIPAudioCfg(**audio_cfg) if isinstance(audio_cfg, dict) else audio_cfg

        self.visual = _build_audio_tower(
            embed_dim=embed_dim,
            audio_cfg=audio_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False, should_augment: bool = True):
        features = self.visual(image, should_augment=should_augment)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, audio, text, audio_latent=None, augment_audio=True):
        text_latent = self.encode_text(text)

        if audio_latent is None:
            audio_latent = self.encode_image(audio, should_augment=augment_audio)

        logit_scale = self.logit_scale.exp()

        if self.output_dict:
            return {
                "image_features": audio_latent,
                "text_features": text_latent,
                "logit_scale": logit_scale
            }

        return audio_latent, text_latent, logit_scale
