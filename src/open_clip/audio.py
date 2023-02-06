from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn

from torchaudio.transforms import Spectrogram, TimeStretch, FrequencyMasking, TimeMasking

from .transformer import VisionTransformer, LayerNorm
from .utils import to_2tuple


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

    def forward(self, x: torch.Tensor):
        x = self.spec(x)

        if self.training:
            x = self.aug(x)

        # automatically crop if audio does not yield a 2d spectrogram that is divisible by patch sizes

        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_size

        rounded_height = height // patch_height * patch_height
        rounded_width = width // patch_width * patch_width

        if (height, width) != (rounded_height, rounded_width):
            print(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        x = x[..., None, :rounded_height, :rounded_width]

        # pass maybe cropped spectrogram to vit

        return self.vit(x)
