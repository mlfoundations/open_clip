""" MaMMUT model (https://arxiv.org/abs/2303.16839)

A single vision encoder paired with a single text decoder that is used in two passes:
a bi-directional pass without cross-attention for contrastive learning, and a causally
masked pass with cross-attention over image tokens for caption generation.

Ported from the LAION fork (https://github.com/LAION-AI/open_clip_mammut) with fixes:
masked mean text pooling (pads excluded from pooling and attention), a properly scaled
init for the image->decoder projection, a distinct lm_head vs contrastive projection,
and no wasted vocab-head matmul in the contrastive pass. The original behaviour remains
reachable via config flags (pool_type='avg_all', use_pad_mask=False) for compatibility
with released openMaMMUT weights.
"""
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .coca_model import MultimodalCfg
from .loss import fused_linear_cross_entropy
from .model import CLIPVisionCfg, _build_vision_tower
from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    ModernMultimodalDecoder,
    MultimodalDecoder,
)


def _build_multimodal_decoder_tower(
        embed_dim: int,
        multimodal_cfg: MultimodalCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    if multimodal_cfg.proj_type == 'none' and embed_dim != multimodal_cfg.width:
        raise ValueError(
            f"MaMMUT with proj_type='none' requires embed_dim == decoder width, "
            f"got embed_dim={embed_dim}, width={multimodal_cfg.width}. "
            f"Set multimodal_cfg.proj_type='linear' to decouple them."
        )

    if multimodal_cfg.text_arch == 'modern':
        # modern decoder is cfg-driven and ignores quick_gelu / cast_dtype norm selection
        # (act/norm come from mlp_type / norm_type, matching ModernTextTransformer)
        return ModernMultimodalDecoder(multimodal_cfg, output_dim=embed_dim)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    decoder = MultimodalDecoder(
        context_length=multimodal_cfg.context_length,
        vocab_size=multimodal_cfg.vocab_size,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        mlp_ratio=multimodal_cfg.mlp_ratio,
        ls_init_value=multimodal_cfg.ls_init_value,
        cross_attn_ratio=multimodal_cfg.cross_attn_ratio,
        output_dim=embed_dim,
        proj_type=multimodal_cfg.proj_type,
        pool_type=multimodal_cfg.pool_type,
        use_pad_mask=multimodal_cfg.use_pad_mask,
        pad_id=multimodal_cfg.pad_id,
        bos_id=multimodal_cfg.bos_id,
        eos_id=multimodal_cfg.eos_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return decoder


class MaMMUT(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            multimodal_cfg: MultimodalCfg,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )
        if not getattr(self.visual, 'output_tokens', False):
            raise ValueError("MaMMUT requires vision_cfg.output_tokens=True for caption cross-attention.")

        self.text = _build_multimodal_decoder_tower(
            embed_dim=embed_dim,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        # projects image tokens to decoder width for cross-attention k/v. Token width comes
        # from the trunk when the tower is timm-based (vision_cfg.width is unreliable there),
        # else from the config.
        vision_width = getattr(getattr(self.visual, 'trunk', None), 'num_features', None) or vision_cfg.width
        self.map_viz2txt_kv = nn.Parameter(torch.empty(vision_width, multimodal_cfg.width))
        nn.init.normal_(self.map_viz2txt_kv, std=vision_width ** -0.5)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None
        # pad id is derived from the text tower (which gets it from multimodal_cfg.pad_id) so
        # masking/pooling, generation defaults, and the caption loss ignore_index (via create_task)
        # all share one source -- same pattern as CoCa / GenLIP
        self.pad_id = self.text.pad_id
        self.bos_id = getattr(self.text, 'bos_id', None)
        self.eos_id = getattr(self.text, 'eos_id', None)

        self.context_length = multimodal_cfg.context_length

    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        self.visual.set_grad_checkpointing(enable, impl=impl)
        self.text.set_grad_checkpointing(enable, impl=impl)

    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = set()
        if hasattr(self.visual, 'no_weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.' + n)
        for n in self.text.no_weight_decay():
            no_wd.add('text.' + n)
        return no_wd

    def _encode_image(self, images, normalize: bool = True):
        # native towers (output_tokens) return the (pooled, tokens) tuple; timm token-mode
        # towers return {'pooled', 'patch_tokens', 'patch_valid'} -- normalize to a 3-tuple
        # with None validity for native
        out = self.visual(images)
        if isinstance(out, dict):
            required = {'pooled', 'patch_tokens', 'patch_valid'}
            missing = required.difference(out)
            if missing:
                raise KeyError(f"MaMMUT vision output is missing required keys: {sorted(missing)}")
            image_latent, image_embs, image_embs_valid = out['pooled'], out['patch_tokens'], out['patch_valid']
        elif isinstance(out, (tuple, list)) and len(out) == 2:
            image_latent, image_embs = out
            image_embs_valid = None
        else:
            raise TypeError(
                "MaMMUT vision tower must return (pooled, tokens) or a timm token-output dictionary."
            )
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, image_embs, image_embs_valid

    def _encode_text(self, text, text_valid=None, normalize: bool = True):
        # the multimodal decoder boundary uses modality names: text_valid for its intrinsic text
        # sequence (alongside context/context_valid for the generic cross-attention side)
        text_latent = self.text(text, text_valid=text_valid, mode='contrastive')
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent

    def encode_image(self, images, normalize: bool = True):
        image_latent, _, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, text_valid=None, normalize: bool = True):
        return self._encode_text(text, text_valid=text_valid, normalize=normalize)

    def _generation_image_context(self, images):
        """Image context for generation: (projected K/V, patch validity or None) from one encode."""
        _, image_embs, image_embs_valid = self._encode_image(images)
        return image_embs @ self.map_viz2txt_kv, image_embs_valid

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            text_valid: Optional[torch.Tensor] = None,
            image_latent: Optional[torch.Tensor] = None,
            image_embs: Optional[torch.Tensor] = None,
            image_embs_valid: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ):
        """text_valid: optional [B, L] bool/int text validity (True/1 = real token), consumed by the
        contrastive pass (attention + pooling, passed straight through to the decoder's
        ``text_valid``); validity falls back to ``text != pad_id`` when absent, and legacy
        configs ignore it (see MultimodalDecoder). The caption pass is causal over right-padded
        text; label masking happens task-side.

        image_embs_valid: optional [B, N_img] bool/int validity for ``image_embs`` (True/1 = real
        patch token). Produced by NaFlex token-mode towers (padded patch batches) and threaded to
        the caption pass as the decoder's ``context_valid``; supply it alongside precomputed
        ``image_embs`` or padded K/V silently join cross-attention. None = dense tokens.

        labels: optional [B, L-1] AR-shifted caption labels (-100 = ignore, task-built). When
        given, the caption pass returns ``caption_loss`` computed via the fused linear
        cross-entropy (full-vocab logits are never materialized) instead of ``logits``."""
        if image is not None and (image_latent is None or image_embs is None):
            image_latent, image_embs, image_embs_valid = self._encode_image(image)

        if text is None:
            return {
                "image_features": image_latent,
                "image_embs": image_embs,
                "image_embs_valid": image_embs_valid,
            }

        text_latent = self._encode_text(text, text_valid=text_valid)

        if image_latent is None:
            return {"text_features": text_latent}

        # caption pass: causal self-attention w/ cross-attention over projected image tokens
        image_kv = image_embs @ self.map_viz2txt_kv

        out_dict = {
            "image_features": image_latent,
            "text_features": text_latent,
            "logit_scale": self.logit_scale.exp(),
        }
        if labels is not None:
            # fused caption loss: hidden positions [0, L-1) predict tokens [1, L) (same shift the
            # task applies to logits on the legacy path)
            hidden = self.text(
                text, context=image_kv, context_valid=image_embs_valid,
                mode='caption', return_hidden=True)
            pred = hidden[:, :-1]
            weight, bias = self.text.lm_head_params
            out_dict["caption_loss"] = fused_linear_cross_entropy(
                pred.reshape(-1, pred.shape[-1]),
                weight,
                labels.reshape(-1),
                bias=bias,
                ignore_index=-100,
            )
        else:
            out_dict["logits"] = self.text(
                text, context=image_kv, context_valid=image_embs_valid, mode='caption')
        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias
        return out_dict

    def generate(
        self,
        image,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.,
        generation_type="beam_search",
        top_p=0.1,
        top_k=1,
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
            min_seq_len=5,
            stopping_criteria=None,
            repetition_penalty=1.0,
            fixed_output_length=False,
            generation_config=None,
            text_valid=None,
    ):
        try:
            from .generation import generate_multimodal
        except (ImportError, Exception) as e:
            raise RuntimeError(
                "Please install transformers for generate functionality. "
                "`pip install transformers`."
            ) from e

        return generate_multimodal(
            self,
            image=image,
            image_embs_fn=self._generation_image_context,
            # the decoder embeds token ids internally, so pass ids straight through
            text_encoder_fn=lambda ids: ids,
            text_decoder_fn=lambda img_kv, ids, valid=None: self.text(
                ids, context=img_kv, context_valid=valid, mode='caption'),
            decoder=self.text,
            text=text,
            text_valid=text_valid,
            seq_len=seq_len,
            max_seq_len=max_seq_len,
            temperature=temperature,
            generation_type=generation_type,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            sot_token_id=sot_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            min_seq_len=min_seq_len,
            stopping_criteria=stopping_criteria,
            repetition_penalty=repetition_penalty,
            fixed_output_length=fixed_output_length,
            generation_config=generation_config,
        )
