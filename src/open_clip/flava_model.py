"""FLAVA model"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from.model import _build_text_tower, _build_vision_tower
from .transformer import Transformer, LayerNorm
from .utils import to_2tuple


class EmbeddingTransformer(nn.Module):
    def __init__(
            self,
            context_length: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.output_dim = output_dim

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.ln_pre = norm_layer(width)

        self.num_heads = heads
        self.positional_embedding = nn.Parameter(scale * torch.randn(context_length + 1, width))
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, 1 + context_length, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # always attend to CLS_M token
        attend_cls = torch.ones(x.shape[0], 1, dtype=torch.long, device=attn_mask.device)
        attn_mask = torch.cat([attend_cls, attn_mask], dim=1).bool()
        key_padding_mask = ~attn_mask

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, key_padding_mask=key_padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        x = x @ self.proj

        return x  # x[:, 0, :] is [CLS]


@dataclass
class FLAVAVisionCfg:
    # TODO: does not support native vision transformer yet

    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = None
    pooler_type: str = 'identity_pooler'

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection


@dataclass
class FLAVATextCfg:
    context_length: int = 128
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = None
    pooler_type: str = 'identity_pooler'


@dataclass
class FLAVAMultimodalCfg:
    context_length: int = 127 + 49
    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value


class FLAVA(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: FLAVAVisionCfg,
            text_cfg: FLAVATextCfg,
            multimodal_cfg: FLAVAMultimodalCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if isinstance(vision_cfg, dict):
            vision_cfg = FLAVAVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = FLAVATextCfg(**text_cfg)
        if isinstance(multimodal_cfg, dict):
            multimodal_cfg = FLAVAMultimodalCfg(**multimodal_cfg)

        # Vision encoder
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        # Text encoder
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        # self.text_masked_head = nn.Linear(embed_dim, self.text.config.vocab_size)

        # Multimodal encoder
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        self.multimodal = EmbeddingTransformer(
            context_length=multimodal_cfg.context_length,
            width=multimodal_cfg.width,
            heads=multimodal_cfg.heads,
            layers=multimodal_cfg.layers,
            mlp_ratio=multimodal_cfg.mlp_ratio,
            ls_init_value=multimodal_cfg.ls_init_value,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.mm_masked_head = nn.Linear(embed_dim, self.text.config.vocab_size)
        self.itm_head = nn.Linear(embed_dim, 1)

        self.image_to_mm_projection = nn.Linear(embed_dim, multimodal_cfg.width)
        self.text_to_mm_projection = nn.Linear(embed_dim, multimodal_cfg.width)

        self.image_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)
        self.mm_projection = nn.Linear(embed_dim, embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.language.grad_checkpointing(enable)
        self.multimodal.grad_checkpointing(enable)

    def encode_image(self, image, normalize=False):
        features = self.visual(image)
        features = self.image_projection(features[:, 0, :])
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize=False):
        features = self.text(text)
        features = self.text_projection(features[:, 0, :])
        return F.normalize(features, dim=-1) if normalize else features

    def encode_multimodal(self, image, text, normalize=False):
        text_hidden = self.language(text)
        image_hidden = self.visual(image)
        mm_image_hidden = self.image_to_mm_projection(image_hidden[:, 1:, :])
        mm_text_hidden = self.text_to_mm_projection(text_hidden[:, 1:, :])

        pad_token_id = self.text.config.pad_token_id
        # TODO: RoBERTa prepends CLS token to text, so we need to remove it
        mm_text_attn_mask = (text[:, 1:] != self.text.config.pad_token_id).long()
        image_hidden_length = image_hidden.shape[1] - 1  # ignore CLS_I token
        mm_vis_attn_mask = torch.ones(
            mm_text_attn_mask.shape[0],
            image_hidden_length,
            dtype=torch.long,
            device=mm_text_attn_mask.device,
        )
        mm_attn_mask = torch.cat([mm_vis_attn_mask, mm_text_attn_mask], dim=1)

        mm_inputs = torch.cat([mm_image_hidden, mm_text_hidden], dim=1)
        mm_hidden = self.multimodal(mm_inputs, attn_mask=mm_attn_mask)
        mm_features = self.mm_projection(mm_hidden[:, 0, :])
        return F.normalize(mm_features, dim=-1) if normalize else mm_features

    def forward(
        self,
        *,
        image,
        text,
        text_masked,
        itm_text,
        # passthrough
        text_masked_labels,
        itm_labels,
    ):
        ####################
        ##### LANGUAGE #####
        ####################

        # Contrastive task
        text_hidden = self.text(text)
        text_features = self.text_projection(text_hidden[:, 0, :])
        text_features = F.normalize(text_features, dim=-1)

        # MLM task
        text_masked_hidden = self.text(text_masked)
        # text_masked_recon_logits = self.text_masked_head(text_masked_hidden[:, 1:, :])
        # # TODO: RoBERTa prepends CLS token to text, so we need to remove it
        text_masked_labels = text_masked_labels[:, 1:]

        ####################
        ###### VISION ######
        ####################

        # Contrastive task
        image_hidden = self.visual(image)
        image_features = self.image_projection(image_hidden[:, 0, :])
        image_features = F.normalize(image_features, dim=-1)

        # TODO: MAE task

        ####################
        #### MULTIMODAL ####
        ####################

        pad_token_id = self.text.config.pad_token_id
        # TODO: RoBERTa prepends CLS token to text, so we need to remove it
        mm_text_attn_mask = (text[:, 1:] != self.text.config.pad_token_id).long()
        image_hidden_length = image_hidden.shape[1] - 1  # ignore CLS_I token
        mm_vis_attn_mask = torch.ones(
            mm_text_attn_mask.shape[0],
            image_hidden_length,
            dtype=torch.long,
            device=mm_text_attn_mask.device,
        )
        mm_attn_mask = torch.cat([mm_vis_attn_mask, mm_text_attn_mask], dim=1)

        # ITM task (uncorrupted images, corrupted text)
        mm_image_hidden = self.image_to_mm_projection(image_hidden[:, 1:, :])
        itm_text_hidden = self.text(itm_text)
        mm_itm_text_hidden = self.text_to_mm_projection(itm_text_hidden[:, 1:, :])

        mm_inputs = torch.cat([mm_image_hidden, mm_itm_text_hidden], dim=1)  # [*, image_ctx + text_ctx, d_mm]
        mm_hidden = self.multimodal(mm_inputs, attn_mask=mm_attn_mask)
        mm_features = self.mm_projection(mm_hidden[:, 0, :])
        itm_pred = self.itm_head(mm_features)

        # MLM task (uncorrupted images, corrupted text)
        mm_text_masked_hidden = self.text_to_mm_projection(text_masked_hidden[:, 1:, :])
        mm_inputs_masked = torch.cat([mm_image_hidden, mm_text_masked_hidden], dim=1)
        mm_hidden_masked = self.multimodal(mm_inputs_masked, attn_mask=mm_attn_mask)
        mm_text_hidden = mm_hidden_masked[:, 1+image_hidden_length:, :]
        mm_text_masked_recon_logits = self.mm_masked_head(mm_text_hidden)

        # TODO: MVLM task (https://openreview.net/pdf?id=ZhuXksSJYWn)

        return {
            'image_features': image_features,
            'text_features': text_features,
            'logit_scale': self.logit_scale.exp(),
            # 'text_masked_logits': text_masked_recon_logits,
            'itm_logits': itm_pred,
            'mm_masked_logits': mm_text_masked_recon_logits,
            'text_masked_labels': text_masked_labels,
            'itm_labels': itm_labels,
        }
