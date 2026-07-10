from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    ModernMultimodalTransformer,
    MultimodalTransformer,
)
from .loss import fused_linear_cross_entropy
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8
    # MaMMUT decoder (MultimodalDecoder) fields, ignored by the CoCa decoder builder
    cross_attn_ratio: int = 1  # one cross-attn block per N self-attn layers (2 -> after layers 0, 2, 4, ...)
    use_pad_mask: bool = True  # mask pad tokens from attention in the bi-directional contrastive pass (legacy: False)
    pool_type: str = 'avg'  # contrastive pool: 'avg' masked mean excl pads | 'avg_all' mean incl pads (legacy)
    proj_type: str = 'none'  # contrastive text projection; 'none' (paper/legacy) requires embed_dim == width
    tie_lm_head: bool = False  # share lm_head weight with token_embedding (modern decoder only)


def _build_text_decoder_tower(
        embed_dim,
        multimodal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    # NOTE: CoCa passes the vocab size as ``embed_dim`` -- the decoder's output projection is the vocab head.
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg

    if multimodal_cfg.text_arch == 'modern':
        # modern decoder is cfg-driven; act/norm come from mlp_type / norm_type (quick_gelu N/A)
        return ModernMultimodalTransformer(multimodal_cfg, vocab_size=embed_dim)

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

    return decoder


class CoCa(nn.Module):
    def __init__(
            self,
            embed_dim,
            multimodal_cfg: MultimodalCfg,
            text_cfg: CLIPTextCfg,
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
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        if vision_cfg.timm_model_name:
            raise ValueError(
                "CoCa does not support timm vision towers: caption cross-attention requires token projection "
                "and validity handling that only MaMMUT's timm path currently provides."
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

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.text_decoder = _build_text_decoder_tower(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

        # pad id is derived from the text tower (the id it masks with: text_cfg.pad_id for native
        # towers, the transformers config pad_token_id for HF towers) so loss ignore_index and
        # generation defaults stay consistent with tower masking and tokenizer padding. The 0
        # fallback is the historical CLIP fill convention (SimpleTokenizer reserves no pad token;
        # 0 is a real vocab token).
        pad_id = getattr(self.text, 'pad_id', None)
        self.pad_id = 0 if pad_id is None else int(pad_id)
        self.bos_id = getattr(self.text, 'bos_id', None)
        self.eos_id = getattr(self.text, 'eos_id', None)

        self.context_length = multimodal_cfg.context_length

    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        self.visual.set_grad_checkpointing(enable, impl=impl)
        self.text.set_grad_checkpointing(enable, impl=impl)
        self.text_decoder.set_grad_checkpointing(enable, impl=impl)

    def _encode_image(self, images, normalize: bool = True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, text_valid=None, normalize: bool = True):
        # text towers keep the HF-style attention_mask kwarg (single-sequence scope); the parent
        # multimodal interface names the mask by modality (text_valid, alongside NaFlex patch_valid)
        text_latent, token_emb = self.text(text, attention_mask=text_valid)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize: bool = True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, text_valid=None, normalize: bool = True):
        """Encode text, optionally using an exact validity mask.

        ``text_valid`` was inserted before ``normalize``. Legacy positional calls such as
        ``encode_text(text, False)`` must use ``normalize=False`` after this breaking API change.
        """
        text_latent, _ = self._encode_text(text, text_valid=text_valid, normalize=normalize)
        return text_latent

    def forward_intermediates(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            text_valid: Optional[torch.Tensor] = None,
            image_indices: Optional[Union[int, List[int]]] = None,
            text_indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize: bool = True,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            image_output_fmt: str = 'NCHW',
            image_output_extra_tokens: bool = False,
            text_output_fmt: str = 'NLC',
            text_output_extra_tokens: bool = False,
            output_logits: bool = False,
            output_logit_scale_bias: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Breaking positional API note: ``text_valid`` was inserted after ``text``; callers passing
        ``image_indices`` or later arguments positionally must switch those arguments to keywords.

        Args:
            image: Input image tensor
            text: Input text tensor
            text_valid: Optional [B, L] bool/int text validity (True/1 = real token); pad-value fallback when absent
            image_indices: For image tower, Take last n blocks if int, all if None, select matching indices if sequence
            text_indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize: L2 Normalize final image and text features (if present)
            normalize_intermediates: Apply final encoder norm layer to all intermediates (if possible)
            intermediates_only: Only return intermediate features, do not return final features
            image_output_fmt: Shape of intermediate image feature outputs
            image_output_extra_tokens: Return both prefix and spatial intermediate tokens
            text_output_fmt: Shape of intermediate text feature outputs
            text_output_extra_tokens: Return both prefix and spatial intermediate tokens
            output_logits: Include logits in output
            output_logit_scale_bias: Include the logit scale bias in the output
        Returns:

        """
        output = {}
        if intermediates_only:
            # intermediates only disables final feature normalization, and include logits
            normalize = False
            output_logits = False
        if output_logits:
            assert False, 'FIXME, needs implementing'

        if image is not None:
            image_output = self.visual.forward_intermediates(
                image,
                indices=image_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=image_output_fmt,
                output_extra_tokens=image_output_extra_tokens,
            )
            if normalize and "image_features" in image_output:
                image_output["image_features"] = F.normalize(image_output["image_features"], dim=-1)
            output.update(image_output)

        if text is not None:
            text_output = self.text.forward_intermediates(
                text,
                attention_mask=text_valid,
                indices=text_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=text_output_fmt,
                output_extra_tokens=text_output_extra_tokens,
            )
            if normalize and "text_features" in text_output:
                text_output["text_features"] = F.normalize(text_output["text_features"], dim=-1)
            output.update(text_output)

        # FIXME text decoder
        logit_scale_exp = self.logit_scale.exp() if output_logits or output_logit_scale_bias else None
        if output_logit_scale_bias:
            output["logit_scale"] = logit_scale_exp
            if self.logit_bias is not None:
                output['logit_bias'] = self.logit_bias

        return output

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            text_valid: Optional[torch.Tensor] = None,
            image_latent: Optional[torch.Tensor] = None,
            image_embs: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ):
        """text_valid: optional [B, L] bool/int text validity (True/1 = real token), consumed by the
        text tower's pad/cls masking (passed down as its HF-style ``attention_mask``); validity falls
        back to ``text != pad_id`` when absent. Caption logits are causal over right-padded text and
        need no mask; label masking for the caption loss happens task-side.

        Breaking positional API note: ``text_valid`` was inserted after ``text``; callers passing
        ``image_latent`` or later arguments positionally must switch those arguments to keywords.

        labels: optional [B, L-1] AR-shifted caption labels (-100 = ignore, task-built). When given,
        returns ``caption_loss`` via the fused linear cross-entropy (full-vocab logits are never
        materialized) instead of ``logits``."""
        if image is not None and (image_latent is None or image_embs is None):
            image_latent, image_embs = self._encode_image(image)

        if text is None:
            return {"image_features": image_latent, "image_embs": image_embs}

        text_latent, token_embs = self._encode_text(text, text_valid=text_valid)

        if image_latent is None:
            return {"text_features": text_latent}

        out_dict = {
            "image_features": image_latent,
            "text_features": text_latent,
            "logit_scale": self.logit_scale.exp(),
        }
        if labels is not None:
            # fused caption loss: hidden positions [0, L-1) predict tokens [1, L) (same shift the
            # task applies to logits on the legacy path)
            hidden = self.text_decoder(image_embs, token_embs, return_hidden=True)
            pred = hidden[:, :-1]
            weight, bias = self.text_decoder.lm_head_params
            out_dict["caption_loss"] = fused_linear_cross_entropy(
                pred.reshape(-1, pred.shape[-1]),
                weight,
                labels.reshape(-1),
                bias=bias,
                ignore_index=-100,
            )
        else:
            out_dict["logits"] = self.text_decoder(image_embs, token_embs)
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
            image_embs_fn=lambda images: self._encode_image(images)[1],
            text_encoder_fn=lambda ids: self._encode_text(ids)[1],
            text_decoder_fn=self.text_decoder,
            decoder=self.text_decoder,
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
