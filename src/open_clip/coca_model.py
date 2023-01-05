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
from .generation_utils import top_a, top_k, top_p

@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8
    latent_dim: int = 512

class CoCaEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.encoder.set_grad_checkpointing(enable)
        self.decoder.set_grad_checkpointing(enable)

def _build_encoder_decoder_tower(
    embed_dim,
    multimodal_cfg,
    text_cfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):

    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        
    encoder = _build_text_tower(
        multimodal_cfg.latent_dim, 
        text_cfg=text_cfg, 
        quick_gelu=quick_gelu, 
        cast_dtype=cast_dtype
    )
    
    vocab_size = (
        encoder.config.vocab_size # for hf models
        if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
        else multimodal_cfg.vocab_size
    )

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
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    
    return CoCaEncoderDecoder(encoder, decoder), multimodal_cfg, vocab_size
 
class CoCa(nn.Module):
    def __init__(
        self,
        embed_dim,
        multimodal_cfg: MultimodalCfg,
        text_cfg: CLIPTextCfg,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        pad_id: int = 0
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )

        self.text, multimodal_cfg, vocab_size = _build_encoder_decoder_tower(
            embed_dim, multimodal_cfg, text_cfg, quick_gelu, cast_dtype
        )
        self.visual = _build_vision_tower(
            multimodal_cfg.latent_dim, vision_cfg, quick_gelu, cast_dtype
        )

        self.to_logits = nn.Sequential(
            norm_layer(multimodal_cfg.width), nn.Linear(multimodal_cfg.width, vocab_size, bias=False)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = pad_id

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, images, normalize=True, return_tokens=False):
        image_latent, tokens_embs = self.visual(images, output_tokens=True)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return (image_latent, tokens_embs) if return_tokens else image_latent

    def encode_text(self, text, normalize=True, return_tokens=False):
        text = text[:, :-1] # make space for CLS token
        text_latent, token_emb = self.text.encoder(text, output_tokens=True)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return (text_latent, token_emb) if return_tokens else text_latent

    def forward(self, image, text, output_dict=False):

        text_latent, token_embs = self.encode_text(text, return_tokens=True)
        image_latent, image_embs = self.encode_image(image, return_tokens=True)
        
        # TODO: add assertion to avoid bugs?
        labels = text[:, -token_embs.shape[1]:]
        
        token_embs = self.text.decoder(image_embs, token_embs)
        logits = self.to_logits(token_embs)
        if output_dict:
            return {
                "image_features":image_latent,
                "text_features":text_latent,
                "logits":logits,
                "labels":labels,
                "logit_scale":self.logit_scale.exp()
            }

        return image_latent, text_latent, logits, labels, self.logit_scale.exp()

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
