from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower
from .generation_utils import Generator


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8
    cross_attn_ratio: int = 1
    does_full_decoding: bool = False
    output_tokens: bool = False
    has_mlp: bool = True


def _build_text_decoder_tower(
        embed_dim,
        multimodal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        is_decoder=True,
):
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    decoder = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        cross_attn_ratio=multimodal_cfg.cross_attn_ratio,
        has_mlp=multimodal_cfg.has_mlp,
        output_dim=embed_dim,
        output_tokens=multimodal_cfg.output_tokens,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return decoder


class CoCa(nn.Module, Generator):
    def __init__(
            self,
            embed_dim,
            multimodal_cfg: MultimodalCfg,
            text_cfg: CLIPTextCfg,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        vocab_size = (
            text_cfg.vocab_size  # for hf models
            if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
            else text_cfg.vocab_size
        )

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.text_decoder = _build_text_decoder_tower(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = pad_id

        self.context_length = multimodal_cfg.context_length

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images, normalize: bool = True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, normalize: bool = True):
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize: bool = True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize: bool = True):
        text_latent, _ = self._encode_text(text, normalize=normalize)
        return text_latent

    def forward(self, image=None, text=None, embed_cls=True, image_latent=None, image_embs=None):
        text_latent, token_embs = self._encode_text(text, embed_cls=embed_cls)
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)

        if text is None:
            return {"image_features": image_latent, "image_embs": image_embs}

        text_latent, token_embs = self._encode_text(text)

        # TODO: add assertion to avoid bugs?
        labels = text[:, 1:]
        if is_training:
            token_embs = token_embs[:, :-1]
        print(token_embs.shape)

        logits = self.text_decoder(image_embs, token_embs)
        return {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "logit_scale": self.logit_scale.exp()
        }

