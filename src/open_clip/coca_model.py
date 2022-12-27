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
from .generation_utils import top_a, top_k, top_p

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
        self.text = text
        if "hf_model_name" not in text_cfg or text_cfg["hf_model_name"] is None:
            self.text_projection = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        self.visual = _build_vision_tower(
            embed_dim, vision_cfg, quick_gelu, cast_dtype
        )

        self.multimodal_decoder, multimodal_cfg = _build_input_dependent_text_tower(
            embed_dim, multimodal_cfg, quick_gelu, cast_dtype
        )

        self.img_attn_pool = AttentionalPooler(
            multimodal_cfg.width, multimodal_cfg.heads, n_queries=n_queries + 1 # extra query for contrastive_loss
        )

        self.img_attn_pool_norm = norm_layer(embed_dim)
        vocab_size = (
            text_cfg["vocab_size"] 
            if "vocab_size" in text_cfg
            else self.text.config.vocab_size # for hf models
        )

        self.to_logits = nn.Sequential(
            norm_layer(embed_dim), nn.Linear(embed_dim, vocab_size, bias=False)
        )

        # tie embedding weights and projection
        # self.to_logits[-1].weight = self.text.token_embedding.weight

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = 0

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.multimodal_decoder.grad_checkpointing = enable

    def encode_image(self, images, normalize=True, return_tokens=False):
        x = self.visual(images, output_tokens=True)

        if hasattr(self.visual, "ln_post"):
            x = self.visual.ln_post(x)

        if hasattr(self.visual, "proj") and self.visual.proj is not None:
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
        text_latent, token_emb = self.text(text, output_tokens=True)

        # not HF model
        if hasattr(self, "text_projection") and self.text_projection is not None:
            text_latent = text_latent @ self.text_projection

        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent

        return (text_latent, token_emb) if return_tokens else text_latent

    def forward(self, image, text, output_dict=False):
        labels = text[:, 1:]

        text_latents, text_tokens = self.encode_text(text, return_tokens=True)
        image_latents, image_tokens = self.encode_image(image, return_tokens=True)

        text_tokens = self.multimodal_decoder(image_tokens, text_tokens)
        logits = self.to_logits(text_tokens)
        if output_dict:
            return {
                "image_features":image_latents,
                "text_features":text_latents,
                "logits":logits,
                "labels":labels,
                "logit_scale":self.logit_scale.exp()
            }

        return image_latents, text_latents, logits, labels, self.logit_scale.exp()

    def generate(
        self,
        image,
        text,
        seq_len,
        max_seq_len=77,
        mask_prob = 0.0,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        min_p_pow = 2.0,
        min_p_ratio = 0.02,
        ):

        assert mask_prob < 1, "mask_prob must be smaller than 1."

        was_training = self.training
        num_dims = len(text.shape)

        if num_dims == 1:
            text = text[None, :]

        _, t = text.shape
        self.eval()
        out = text

        for _ in range(seq_len):
            x = out[:, -max_seq_len:]

            # TODO: adjust for dict output
            logits = self(image, x)[2][:, -1]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is top_a:
                filtered_logits = filter_logits_fn(
                    logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio
                )
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)


        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.train(was_training)
        return out
