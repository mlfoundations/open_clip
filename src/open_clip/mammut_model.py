from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower



class MaMMUT(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            mixed_text_cfg: CLIPTextCfg,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            pad_id: int = 0,
            cross_attn_ratio: int = 1
    ):
        super().__init__()
        text_cfg = CLIPTextCfg(**mixed_text_cfg) if isinstance(mixed_text_cfg, dict) else mixed_text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        self.text = _build_text_tower(
            vocab_size,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            cross_attn_ratio = cross_attn_ratio,
            is_mammut = True
        )

        vocab_size = (
            self.text.config.vocab_size # for hf models
            if mixed_text_cfg.get("hf_model_name", None) is not None
            else mixed_text_cfg.vocab_size
        )

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = pad_id

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)
        
    def encode_text(self, text, cross_embs=None, normalize=True, contrastive=True):
        text_latent, token_emb = self.text(text, cross_embs=cross_embs)
        
        if contrastive:
            text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
            return text_latent
         
        return token_emb

    def encode_image(self, image, normalize: bool = True):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def _forward(self, image, text, out, image_features=None, contrastive=True):
        
        if image_features is None:
            image_features = self.encode_image(image)
        
        if contrastive:
            text_features = self.encode_text(text, contrastive=contrastive)
            out["image_features"] = image_features
            out["text_features"] = text_features
            return out, image_features

        # TODO: add assertion to avoid bugs?
        text = text[:, :-1] # drop last tok because it has not label
        out["labels"] = text[:, 1:] # shift labels
        out["logits"] = self.encode_text(text, cross_embs=image_features, normalize=False, contrastive=False)

        return out
    
    def forward(self, image, text):
        out = {"logit_scale": self.logit_scale.exp()}
        out, image_features = self._forward(image, text, out)
        out = self._forward(image, text, out, image_features=image_features, contrastive=False)
        return out