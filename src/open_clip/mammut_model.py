from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower
from .coca_model import MultimodalCfg
from .transformer import QuickGELU, LayerNormFp32, LayerNorm, MultimodalTransformer
from .generation_utils import Generator



def _build_multimodal_decoder_tower(
        embed_dim,
        multimodal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        is_decoder=True
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
        does_full_decoding=multimodal_cfg.does_full_decoding,
        output_tokens=multimodal_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return decoder

class MaMMUT(nn.Module, Generator):
    def __init__(
        self,
        embed_dim: int,
        text_cfg: MultimodalCfg,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = (
            CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        )

        vocab_size = (
            self.text.config.vocab_size  # for hf models
            if multimodal_cfg.__dict__.get("hf_model_name", None) is not None
            else multimodal_cfg.vocab_size
        )

        self.text = _build_multimodal_decoder_tower(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            is_decoder=False,
        )

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.map_viz2txt_kv = nn.Parameter(torch.randn(vision_cfg.width, multimodal_cfg.width))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = pad_id

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def _encode_text(self, text, image_embs):
        token_logits, text_latent = self.text(
            text_embs=text,
            image_embs=image_embs,
        )
        return token_logits, text_latent
        

    def encode_text(
        self,
        text,
        image_embs=None,
        normalize=True,
        output_logits=False
    ):
        token_logits, text_latent = self._encode_text(
            text=text,
            image_embs=image_embs,
        )

        if output_logits:
            return token_logits

        text_latent = text_latent.mean(1)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent

    def _encode_image(self, image, normalize: bool=True):
        image_latent, image_embs = self.visual(image)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, image_embs

    def encode_image(self, image, normalize: bool=True):
        image_latent, _ = self._encode_image(image, normalize=normalize)
        return image_latent

    def _forward(self, text, out, image_embs=None, contrastive=True, embed_cls=True):

        if contrastive:
            text_features = self.encode_text(text)
            out["text_features"] = text_features
            return out

        # adjust image output size for cross_attn
        image_embs = image_embs @ self.map_viz2txt_kv

        # TODO: add assertion to avoid bugs?
        out["labels"] = text[:, 1:]  # shift labels
        text = text[:, :-1] if embed_cls else text # drop last tok because it has no label
        out["logits"] = self.encode_text(text, image_embs=image_embs, output_logits=True)

        return out

    def forward(self, image, text, image_latent=None, image_embs=None, embed_cls=True):
        out = {"logit_scale": self.logit_scale.exp()}
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)
        out["image_features"] = image_latent
        out = self._forward(text, out)
        out = self._forward(text, out, image_embs=image_embs, contrastive=False, embed_cls=embed_cls)
        return out
