import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from dataclasses import dataclass

from .transformer import LayerNorm
from .coca_layers import ParallelTransformerBlock, CrossAttention
from .model import CLIPTextCfg, _build_vision_tower


@dataclass
class CoCaCfg:
    model_name: str = "CoCa_base"
    dim: int = 768
    image_dim: int = 768
    ff_mult: int = 4
    unimodal_depth: int = 12
    multimodal_depth: int = 12
    dim_head: int = 64
    heads: int = 12
    contrastive_loss_weight: float = 1.0
    caption_loss_weight: float = 2.0

    # vit_image_size: int = 288
    # vit_patch_size: int = 18
    # # vit_num_classes: int = 1000
    # vit_dim: int = 768
    # vit_depth: int = 12
    # vit_heads: int = 12
    # vit_mlp_dim: int = 3072


class CoCa(nn.Module):
    def __init__(self, coca_cfg: CoCaCfg, vit_cfg: CLIPTextCfg, tokenizer):
        super().__init__()

        unimodal_depth = coca_cfg.unimodal_depth
        multimodal_depth = coca_cfg.multimodal_depth
        image_dim = coca_cfg.image_dim
        num_img_queries = 256
        dim_head = coca_cfg.dim_head
        heads = coca_cfg.heads
        ff_mult = coca_cfg.ff_mult

        self.dim = coca_cfg.dim
        self.caption_loss_weight = coca_cfg.caption_loss_weight
        self.contrastive_loss_weight = coca_cfg.contrastive_loss_weight
        self.pad_id = coca_cfg.pad_id

        self.tokenizer = tokenizer
        num_tokens = len(self.tokenizer)
        self.img_encoder = _build_vision_tower(vit_cfg)
        self.token_emb = nn.Embedding(num_tokens, self.dim)
        self.text_cls_token = nn.Parameter(torch.randn(self.dim))

        # num image queries for multimodal, but 1 extra CLS for contrastive learning
        self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, self.dim))
        self.img_attn_pool = CrossAttention(
            dim=self.dim,
            context_dim=image_dim,
            dim_head=dim_head,
            heads=heads,
            norm_context=True,
        )

        self.img_attn_pool_norm = LayerNorm(self.dim)
        self.text_cls_norm = LayerNorm(self.dim)

        # contrastive learning temperature

        self.temperature = nn.Parameter(torch.Tensor([1.0]))

        # unimodal layers

        self.unimodal_layers = nn.ModuleList([])
        for ind in range(unimodal_depth):
            self.unimodal_layers.append(
                ParallelTransformerBlock(
                    dim=self.dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult
                ),
            )

        # multimodal layers

        self.multimodal_layers = nn.ModuleList([])
        for ind in range(multimodal_depth):
            self.multimodal_layers.append(
                nn.ModuleList(
                    [
                        ParallelTransformerBlock(
                            dim=self.dim,
                            dim_head=dim_head,
                            heads=heads,
                            ff_mult=ff_mult,
                        ),
                        CrossAttention(
                            dim=self.dim,
                            dim_head=dim_head,
                            heads=heads,
                            residual=True,
                            parallel_ff=True,
                            ff_mult=ff_mult,
                        ),
                    ]
                )
            )

        # to logits

        self.to_logits = nn.Sequential(
            LayerNorm(self.dim), nn.Linear(self.dim, num_tokens, bias=False)
        )

        # they used embedding weight tied projection out to logits, not common, but works
        self.to_logits[-1].weight = self.token_emb.weight
        nn.init.normal_(self.token_emb.weight, std=0.02)

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
