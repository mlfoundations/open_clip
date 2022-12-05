from typing import Optional

import torch
from torch import nn
import numpy as np
from dataclasses import dataclass

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    TransformerDecoder,
    AttentionPooler,
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


@dataclass
class CoCaCfg:
    model_name: str = "CoCa_base"
    context_length:int = 76
    width: int = 512
    image_dim: int = 512
    mlp_ratio: int = 4
    ls_init_value: Optional[float] = None
    layers: int = 12
    dim_head: int = 64
    heads: int = 8
    contrastive_loss_weight: float = 1.0
    caption_loss_weight: float = 2.0
    n_queries: int = 256


def _build_text_decoder_tower(
    embed_dim: int,
    coca_cfg: CoCaCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(coca_cfg, dict):
        coca_cfg = CoCaCfg(**coca_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    text = TransformerDecoder(
        context_length=coca_cfg.context_length,
        width=coca_cfg.width,
        heads=coca_cfg.heads,
        layers=coca_cfg.layers,
        ls_init_value=coca_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return text


class CoCa(nn.Module):
    def __init__(
        self,
        embed_dim,
        coca_cfg: CoCaCfg,
        text_cfg: CLIPTextCfg,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        self.img_encoder = _build_vision_tower(
            embed_dim, vision_cfg, quick_gelu, cast_dtype
        )



        self.multimodal_decoder = _build_text_decoder_tower(
            embed_dim, coca_cfg, quick_gelu, cast_dtype
        )

        self.width = coca_cfg.width

        self.img_attn_pool = AttentionPooler(
            coca_cfg.width, coca_cfg.heads, n_queries=coca_cfg.n_queries + 1
        )

        self.img_attn_pool_norm = norm_layer(self.width)
        self.text_cls_norm = norm_layer(self.width)

        # contrastive learning temperature

        self.temperature = nn.Parameter(torch.Tensor([1.0]))

        # to logits
        self.to_logits = nn.Sequential(
            norm_layer(self.width), nn.Linear(self.width, text_cfg.vocab_size, bias=False)
        )

        # they used embedding weight tied projection out to logits, not common, but works
        self.to_logits[-1].weight = self.token_embedding.weight

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, text):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), :] @ self.text_projection

        # looking at the tokenizer this seems ok
        cls_emb = self.text_cls_norm(x[torch.arange(x.shape[0]), -1])
        token_emb = x[torch.arange(x.shape[0]), :-1]
        return cls_emb, token_emb

    def encode_image(self, images=None):
        x = self.img_encoder.conv1(images)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.img_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.img_encoder.positional_embedding.to(x.dtype)
        x = self.img_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.img_encoder.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.img_encoder.ln_post(x)

        x = self.img_attn_pool(x)
        x = self.img_attn_pool_norm(x)

        return x[:, 0], x[:, 1:]

    def forward(
        self,
        text,
        images=None,
        image_tokens=None,
        labels=None,
    ):

        if labels is None:
            text, labels = text[:, :-1], text[:, 1:]

        text_embeds, text_tokens = self.encode_text(text)
        image_embeds, image_tokens = self.encode_image(images)

        text_tokens = self.multimodal_decoder(
            text_tokens, image_tokens, eot_token_mask=text.argmax(dim=-1)
        )
        logits = self.to_logits(text_tokens)

        return text_embeds, image_embeds, logits, labels, self.logit_scale
