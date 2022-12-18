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
    AttentionalPooler,
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    dim_latents: int = None


def _build_input_dependent_text_tower(
    embed_dim: int,
    multimodal_cfg: MultimodalCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
    multimodal:bool = True
):

    if not multimodal:
        return _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype
        )

    if isinstance(multimodal_cfg, dict):
        multimodal_cfg = MultimodalCfg(**multimodal_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    text = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return text, multimodal_cfg


class CoCa(nn.Module):
    def __init__(
        self,
        embed_dim,
        multimodal_cfg: MultimodalCfg,
        text_cfg: CLIPTextCfg,
        vision_cfg: CLIPVisionCfg,
        n_queries: int = 256,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()


        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )

        text = _build_input_dependent_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype, multimodal=False)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        self.cls_token = nn.Parameter(torch.randn(embed_dim))
        self.visual = _build_vision_tower(
            embed_dim, vision_cfg, quick_gelu, cast_dtype
        )

        self.multimodal_decoder, multimodal_cfg = _build_input_dependent_text_tower(
            embed_dim, multimodal_cfg, quick_gelu, cast_dtype
        )

        self.img_attn_pool = AttentionalPooler(
            multimodal_cfg.width, multimodal_cfg.heads, n_queries=n_queries + 1
        )

        self.img_attn_pool_norm = norm_layer(embed_dim)

        self.dim_latents = multimodal_cfg.dim_latents if multimodal_cfg.dim_latents else multimodal_cfg.width
        self.to_text_latent = nn.Linear(embed_dim, self.dim_latents, bias=False)

        self.to_logits = nn.Sequential(
            norm_layer(embed_dim), nn.Linear(embed_dim, self.vocab_size, bias=False)
        )

        # tie embedding weights and projection
        self.to_logits[-1].weight = self.token_embedding.weight

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable
        self.multimodal_decoder.grad_checkpointing = enable

    def encode_image(self, images, normalize=True, return_tokens=False):
        x = self.visual.conv1(images)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.visual.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.visual.ln_post(x)

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        x = self.img_attn_pool(x, x)
        x = self.img_attn_pool_norm(x)

        image_latent = x[:, 0]
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent

        return (image_latent, x[:, 1:]) if return_tokens else image_latent

    def _repeat(self, t, N):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def encode_text(self, text, normalize=True, return_tokens=False):
        text = text[:, :-1] # make space for CLS token
        cast_dtype = self.transformer.get_cast_dtype()

        # cls_mask = (text!=self.pad_id).unsqueeze(1)
        # attn_mask = F.pad(cls_mask, (0, 1, text.shape[1], 0), value=True)
        # attn_mask = F.pad(self.attn_mask, (0, 1, 0, 1), value=0.0)

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = torch.cat([x, self._repeat(self.cls_token, x.shape[0])], dim=1)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), :] @ self.text_projection

        cls_emb = x[torch.arange(x.shape[0]), -1]
        token_emb = x[torch.arange(x.shape[0]), :-1]

        cls_emb = self.ln_final(cls_emb)
        text_latent = self.to_text_latent(cls_emb)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent

        return (text_latent, token_emb) if return_tokens else text_latent

    def forward(self, image, text):
        labels = text[:, 1:]

        text_latents, text_tokens = self.encode_text(text, return_tokens=True)
        image_latents, image_tokens = self.encode_image(image, return_tokens=True)

        text_tokens = self.multimodal_decoder(image_tokens, text_tokens)
        logits = self.to_logits(text_tokens)

        return image_latents, text_latents, logits, labels, self.logit_scale.exp()
