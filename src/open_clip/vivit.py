"""
ViViT model (https://arxiv.org/abs/2103.15691)
"""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


@dataclass
class TemporalCfg:
    context_length: int = 32 # number of input frames
    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: int = 4
    pooler_type: str = "cls_pooler"


def _build_video_tower(
        embed_dim,
        vision_cfg,
        temporal_cfg,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        temporal_cfg = TemporalCfg(**temporal_cfg) if isinstance(temporal_cfg, dict) else temporal_cfg


# TODO: implement
class ViViT(nn.Module):
    def __init__(self):
        pass


# TODO: turn into VideoCoCa
# TODO: implement
# TODO: do we need quickgelu? 
class VideoCLIP(nn.Module):
    def __init__(
        self,
        embed_dim,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        temporal_cfg: TemporalCfg,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        temporal_cfg = TemporalCfg(**temporal_cfg) if isinstance(temporal_cfg, dict) else temporal_cfg

        '''
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
        '''


