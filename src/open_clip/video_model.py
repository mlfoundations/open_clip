from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    Transformer,
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


@dataclass
class TemporalCfg:
    context_length: int = 32 # number of input frames
    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: int = 4
    pooler_type: str = "cls_pooler"


# TODO: ViViT class makes this function a bit pointless
# still thinking about how to organize this better
def _build_video_tower(
        embed_dim,
        vision_cfg,
        temporal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):

        model = ViViT(
            embed_dim,
            vision_cfg,
            temporal_cfg,
            quick_gelu,
            cast_dtype,
        )

        return model

# TODO: finish option for mean pooling (no cls token if global_average_pool == True)
class ViViT(nn.Module):
    """ViViT model (https://arxiv.org/abs/2103.15691), factorised encoder variant"""
    def __init__(
        self,
        embed_dim,
        vision_cfg,
        temporal_cfg,
        global_average_pool: bool = False,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        temporal_cfg = TemporalCfg(**temporal_cfg) if isinstance(temporal_cfg, dict) else temporal_cfg

        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = (
            LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        )
        self.context_length = temporal_cfg.context_length

        '''
        # class embeddings and positional embeddings
        scale = temporal_cfg.width ** -0.5
        self.video_class_embedding = nn.Parameter(scale * torch.randn(temporal_cfg.width))
        self.video_positional_embedding = nn.Parameter(scale * torch.randn(temporal_cfg.context_length, temporal_cfg.width))

        self.ln_pre = norm_layer(temporal_cfg.width)
        self.ln_post = norm_layer(temporal_cfg.width)
        self.proj = nn.Parameter(scale * torch.randn(temporal_cfg.width, embed_dim))
        '''

        self.spatial = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )
        '''
        self.temporal = Transformer(
            width=temporal_cfg.width,
            layers=temporal_cfg.layers,
            heads=temporal_cfg.heads,
            mlp_ratio=temporal_cfg.mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        '''

        self.global_average_pool = global_average_pool
        self.global_average_pool = True

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.spatial.set_grad_checkpointing(enable)
        # self.temporal.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    # TODO: add patch dropout as suggested by lucidrains
    def forward(self, video):
        print("VIDEO SHAPE")
        print(video.shape)
        # video = video[:, 1:] # make space for temporal CLS token
        batch_size = video.shape[0]

        # Flatten all frames in batch across time and encode with ViT
        frames = video.flatten(start_dim=0, end_dim=1)
        f_e = self.spatial(frames)
        # Put frame embeddings back into correct temporal sequences
        f_e = f_e.view(*video.shape[:2], -1)
        print("FRAME EMBEDDINGS SHAPE")
        print(f_e.shape)

        '''
        # class embeddings and positional embeddings
        f_e = torch.cat(
            [self.video_class_embedding.to(f_e.dtype) + torch.zeros(f_e.shape[0], 1, f_e.shape[-1], dtype=f_e.dtype, device=f_e.device),
             f_e], dim=1)  # shape = [b, cl, w]
        f_e = f_e + self.video_positional_embedding.to(f_e.dtype)

        # TODO: need to look at paper again, section 3, equations (4,5,6)
        # do we need the residual connections?
        f_e = self.ln_pre(f_e)

        f_e = f_e.permute(1, 0, 2)
        v_e = self.temporal(f_e)
        v_e = v_e.permute(1, 0, 2)

        pooled, tokens = self._global_pool(v_e)
        pooled = self.ln_post(pooled)
        pooled = pooled @ self.proj

        # print("POOLED")
        # print(pooled[:10, :10])
        # print(torch.mean(torch.var(pooled, dim=0)))
        '''
        pooled, tokens = self._global_pool(f_e)

        return pooled


# TODO: turn into VideoCoCa
# TODO: set_grad_checkpointing
class VideoCLIP(nn.Module):
    def __init__(
        self,
        embed_dim,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        temporal_cfg: TemporalCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        temporal_cfg = TemporalCfg(**temporal_cfg) if isinstance(temporal_cfg, dict) else temporal_cfg

        self.visual = _build_video_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            temporal_cfg=temporal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        vocab_size = (
            text_cfg.vocab_size  # for hf models
            if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
            else text_cfg.vocab_size
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_video(self, video, normalize: bool = False):
        features = self.visual(video)
        return F.normalize(features, dim=-1) if normalize else features
    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features
    def forward(self, video, text):
        video_features = self.encode_video(video, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        # TODO: make loss functions generalize to all types of modality pairs
        # i.e. make keys more general, for now keeping as image_features
        return {
            "image_features": video_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp()
        }
