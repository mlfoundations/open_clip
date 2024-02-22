from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from .hf_model import HFTextEncoder
from .transformer import VisionTransformer, TextTransformer
from .model import (
    _build_text_tower, _build_vision_tower, CLIPTextCfg, CLIPVisionCfg, CustomTextCLIP
)


class ThreeTowersCustomTextCLIP(CustomTextCLIP):

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
        cache_dir: Optional[str] = None,
        tie_projections: bool = False
    ):
        super(ThreeTowersCustomTextCLIP, self).__init__(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            init_logit_scale=init_logit_scale,
            init_logit_bias=init_logit_bias,
            cast_dtype=cast_dtype,
            output_dict=output_dict,
            cache_dir=cache_dir,
        )
        if isinstance(teacher_cfg, CLIPTextCfg) or (
            isinstance(teacher_cfg, dict)
            and any(
                field in teacher_cfg
                for field in [
                    'context_length', 'vocab_size', 'hf_tokenizer_name', 'hf_model_name'
                ]
            )
        ):
            self.teacher = _build_text_tower(
                embed_dim, teacher_cfg, quick_gelu, cast_dtype, cache_dir
            )
            self.teacher.lock(unlocked_layers=0, freeze_layer_norm=True)
            self.teacher_type = 'text'
        else:
            self.teacher = _build_vision_tower(
                embed_dim, teacher_cfg, quick_gelu, cast_dtype, cache_dir
            )
            self.teacher.lock(unlocked_groups=0, freeze_bn_stats=False)
            self.teacher_type = 'vision'

        if tie_projections:
            self._tie_projections()

    @staticmethod
    def _tie_linears(linear_a: nn.Linear, linear_b: nn.Linear):
        linear_a.weight = linear_a.weight
        if hasattr(linear_a, 'bias') and hasattr(linear_a, 'bias'):
            linear_a.bias = linear_b.bias

    def _tie_projections(self):
        if self.teacher_type == 'text':
            assert type(self.teacher) is type(self.text), (
                'Unable to tie the projections layers, '
                f'teacher model is of type `{type(self.teacher)}` while text '
                f'model is of type `{type(self.text)}`'
            )
            if isinstance(self.teacher, TextTransformer):
                self._tie_linears(
                    self.teacher.text_projection, self.text.text_projection
                )
            elif isinstance(self.teacher, HFTextEncoder):
                if isinstance(self.teacher.proj, nn.Linear):
                    self._tie_linears(self.teacher.proj, self.text.proj)
                elif isinstance(self.teacher.proj, nn.Sequential):
                    for module_a, module_b in zip(
                        self.teacher.proj.children(), self.text.proj.children()
                    ):
                        if (
                            isinstance(module_a, nn.Linear)
                            and isinstance(module_b, nn.Linear)
                        ):
                            self._tie_linears(module_a, module_b)
            else:
                raise TypeError(
                    f'Teacher model type `{type(self.teacher)}` is not '
                    f'compatible with the 3towers architecture!'
                )
        else:
            assert type(self.teacher) is type(self.visual), (
                'Unable to tie the projections layers, '
                f'teacher model is of type `{type(self.teacher)}` while vision '
                f'model is of type `{type(self.visual)}`'
            )
            if isinstance(self.teacher, VisionTransformer):
                self.teacher.proj = self.visual.proj
            else:
                raise TypeError(
                    f'Teacher model type `{type(self.teacher)}` is not '
                    f'compatible with the 3towers architecture!'
                )

    def encode_teacher(self, image_or_text, normalize: bool = False):
        features = self.teacher(image_or_text)
        return f.normalize(features, dim=-1) if normalize else features

    def forward(
        self, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None,
    ):
        image_features = (
            self.encode_image(image, normalize=True)
            if image is not None else None
        )
        text_features = (
            self.encode_text(text, normalize=True)
            if text is not None else None
        )
        if self.teacher_type == 'vision':
            _teacher_input = image
        else:
            _teacher_input = text

        teacher_features = (
            self.encode_teacher(_teacher_input, normalize=True)
            if _teacher_input is not None else None
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
