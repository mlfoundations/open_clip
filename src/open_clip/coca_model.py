from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from dataclasses import dataclass

from .transformer import LayerNormFp32, LayerNorm, QuickGELU, CoCaMultimodalTransformer, ResidualAttentionBlock
from .coca_layers import ParallelTransformerBlock, CrossAttention
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


@dataclass
class CoCaCfg:
    model_name: str = "CoCa_base"
    context_length = 77
    width: int = 768
    image_dim: int = 768
    mlp_ratio: int = 4
    ls_init_value: Optional[float] = None
    layers: int = 12
    dim_head: int = 64
    heads: int = 12
    num_image_queries: int = 256
    contrastive_loss_weight: float = 1.0
    caption_loss_weight: float = 2.0

    # vit_image_size: int = 288
    # vit_patch_size: int = 18
    # vit_dim: int = 768
    # vit_depth: int = 12
    # vit_heads: int = 12
    # vit_mlp_dim: int = 3072


def _build_coca_multimodal_tower(
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

    text = CoCaMultimodalTransformer(
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

        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        act_layer = QuickGELU if quick_gelu else nn.GELU

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

        self.multimodal_decoder = _build_coca_multimodal_tower(
            embed_dim, coca_cfg, quick_gelu, cast_dtype
        )
        num_img_queries = coca_cfg.num_image_queries
        self.width = coca_cfg.width
        num_tokens = text_cfg.vocab_size
        self.text_cls_token = nn.Parameter(torch.randn(self.width))

        # num image queries for multimodal, but 1 extra CLS for contrastive learning
        self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, self.width))
        self.img_attn_pool = ResidualAttentionBlock(
            d_model=coca_cfg.width,
            n_head=coca_cfg.heads,
            mlp_ratio=coca_cfg.mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            is_cross_attention=True,
            is_pooler=True,
        )

        self.img_attn_pool_norm = norm_layer(self.width)
        self.text_cls_norm = norm_layer(self.width)

        # contrastive learning temperature

        self.temperature = nn.Parameter(torch.Tensor([1.0]))

        # to logits

        self.to_logits = nn.Sequential(
            norm_layer(self.width), nn.Linear(self.width, num_tokens, bias=False)
        )

        # get the token embeddings whether the encoder is HF or custom
        for mod in self.transformer.state_dict():
            if any((emb_name in mod) and ("weight" in mod) for emb_name in ["word_embeddings", "token_embeddings"]):
                token_embeddings = self.transformer.get_parameter(mod)
                break

        # they used embedding weight tied projection out to logits, not common, but works
        self.to_logits[-1].weight = token_embeddings
        nn.init.normal_(token_embeddings, std=0.02)

    def embed_text(self, text):
        batch, device = text.shape[0], text.device

        seq = text.shape[1]

        text_tokens = self.token_emb(text)

        # append text cls tokens

        text_cls_tokens = repeat(self.text_cls_token, "d -> b 1 d", b=batch)
        text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)

        # create specific mask for text cls token at the end
        # to prevent it from attending to padding

        cls_mask = rearrange(text != self.pad_id, "b j -> b 1 j")
        attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)

        # go through unimodal layers

        for attn_ff in self.unimodal_layers:
            text_tokens = attn_ff(text_tokens, attn_mask=attn_mask)

        # get text cls token

        text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
        text_embeds = self.text_cls_norm(text_cls_tokens)
        return text_embeds, text_tokens

    def embed_image(self, images=None, image_tokens=None):
        # encode images into embeddings
        # with the img_encoder passed in at init
        # it can also accept precomputed image tokens

        assert images is None or image_tokens is None

        if images is not None:
            assert (
                self.img_encoder is not None
            ), "img_encoder must be passed in for automatic image encoding"
            image_tokens = self.img_encoder(images)

        # attention pool image tokens

        img_queries = repeat(self.img_queries, "n d -> b n d", b=image_tokens.shape[0])
        img_queries = self.img_attn_pool(img_queries, image_tokens)
        img_queries = self.img_attn_pool_norm(img_queries)

        return img_queries[:, 0], img_queries[:, 1:]

    def forward(
        self,
        text,
        images=None,
        image_tokens=None,
        labels=None,
        return_loss=False,
        return_embeddings=False,
    ):
        batch, device = text.shape[0], text.device

        if return_loss and labels is None:
            text, labels = text[:, :-1], text[:, 1:]

        text_embeds, text_tokens = self.embed_text(text)

        image_embeds, image_tokens = self.embed_image(
            images=images, image_tokens=image_tokens
        )

        for attn_ff, cross_attn in self.multimodal_layers:
            text_tokens = attn_ff(text_tokens)
            text_tokens = cross_attn(text_tokens, image_tokens)

        logits = self.to_logits(text_tokens)

        return text_embeds, image_embeds, logits, labels
