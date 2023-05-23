from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower
from .generation_utils import Generator


class MaMMUT(nn.Module, Generator):
    def __init__(
        self,
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        pad_id: int = 0,
    ):
        super().__init__()
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = (
            CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        )

        vocab_size = (
            self.text.config.vocab_size  # for hf models
            if text_cfg.__dict__.get("hf_model_name", None) is not None
            else text_cfg.vocab_size
        )

        self.text = _build_text_tower(
            vocab_size,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            language_modeling=True,
            is_decoder=False,
        )

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.map_viz2txt_kv = nn.Parameter(torch.randn(vision_cfg.width, text_cfg.width))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = pad_id
        self._encode_image = self.encode_image

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_text(
        self,
        text,
        image_embs=None,
        normalize=True,
        attn_mask=None,
        cross_attn_mask=None,
        output_logits=False
    ):
        text_latent, token_logits = self.text(
            text,
            cross_embs=image_embs,
            attn_mask=attn_mask,
            cross_attn_mask=cross_attn_mask,
        )

        if output_logits:
            return token_logits

        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent

    def _encode_text(self, text, image_embs, attn_mask, cross_attn_mask):
        return self.text(
            text,
            cross_embs=image_embs,
            attn_mask=attn_mask,
            cross_attn_mask=cross_attn_mask,
        )

    def encode_image(self, image, normalize: bool = True):
        pooled, tokens = self.visual(image)
        if normalize:
            pooled = F.normalize(pooled, dim=-1) if normalize else pooled
        return pooled, tokens

    def _forward(self, text, out, image_embs=None, contrastive=True, embed_cls=True):

        image_embs = image_embs @ self.map_viz2txt_kv
        if contrastive:
            text_features = self.encode_text(text)
            out["text_features"] = text_features
            return out

        # TODO: add assertion to avoid bugs?
        out["labels"] = text[:, 1:]  # shift labels
        text = text[:, :-1] if embed_cls else text# drop last tok because it has no label

        # adjust image output size for cross_attn
        out["logits"] = self.encode_text(text, image_embs=image_embs, output_logits=True)

        return out

    def forward(self, image, text, image_latent=None, image_embs=None, embed_cls=False):
        out = {"logit_scale": self.logit_scale.exp()}
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self.encode_image(image)
        out["image_features"] = image_latent
        out = self._forward(text, out, image_embs=image_embs)
        out = self._forward(text, out, image_embs=image_embs, contrastive=False, embed_cls=embed_cls)
        print(out["logits"].shape)
        return out
