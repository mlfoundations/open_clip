from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from .model import _build_text_tower, _build_vision_tower, CLIPTextCfg, CLIPVisionCfg


class ThreeTowerCustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        teacher_cfg: Union[CLIPVisionCfg, CLIPTextCfg],
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        if 'hf_tokenizer_name' in teacher_cfg:
            self.teacher = _build_text_tower(
                embed_dim, teacher_cfg, quick_gelu, cast_dtype
            )
            self.teacher.lock(unlocked_layers=0, freeze_layer_norm=True)
            self.teacher_type = 'text'
        else:
            self.teacher = _build_vision_tower(
                embed_dim, teacher_cfg, quick_gelu, cast_dtype
            )
            self.teacher.lock(unlocked_groups=0, freeze_bn_stats=False)
            self.teacher_type = 'vision'
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return f.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return f.normalize(features, dim=-1) if normalize else features

    def encode_teacher(self, image, normalize: bool = False):
        features = self.teacher(image)
        return f.normalize(features, dim=-1) if normalize else features

    # def get_logits(self, image, text):
    #     image_features = self.encode_image(image, normalize=True)
    #     text_features = self.encode_text(text, normalize=True)
    #     image_logits = self.logit_scale.exp() * image_features @ text_features.T
    #     if self.logit_bias is not None:
    #         image_logits += self.logit_bias
    #     text_logits = image_logits.T
    #     return image_logits, text_logits

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        image_features = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )
        if self.teacher_type == 'vision':
            teacher_features = (
                self.encode_teacher(image, normalize=True)
                if image is not None else None
            )
        else:
            teacher_features = (
                self.encode_text(text, normalize=True) if text is not None else None
            )

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "teacher_features": teacher_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return (
                image_features,
                text_features,
                teacher_features,
                self.logit_scale.exp(),
                self.logit_bias,
            )
        return image_features, text_features, teacher_features, self.logit_scale.exp()
