from typing import Dict, List, Optional, Union

import logging
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8


def _build_text_decoder_tower(
        embed_dim,
        multimodal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
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
            pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

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
        self.pad_id = pad_id

        self.context_length = multimodal_cfg.context_length

    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        self.visual.set_grad_checkpointing(enable, impl=impl)
        self.text.set_grad_checkpointing(enable, impl=impl)
        self.text_decoder.set_grad_checkpointing(enable, impl=impl)

    def _encode_image(self, images, normalize: bool = True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, normalize: bool = True):
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize: bool = True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize: bool = True):
        text_latent, _ = self._encode_text(text, normalize=normalize)
        return text_latent

    def forward_intermediates(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
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

        Args:
            image: Input image tensor
            text: Input text tensor
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
            image_latent: Optional[torch.Tensor] = None,
            image_embs: Optional[torch.Tensor] = None,
    ):
        if image is not None and (image_latent is None or image_embs is None):
            image_latent, image_embs = self._encode_image(image)

        if text is None:
            return {"image_features": image_latent, "image_embs": image_embs}

        text_latent, token_embs = self._encode_text(text)

        if image_latent is None:
            return {"text_features": text_latent}

        logits = self.text_decoder(image_embs, token_embs)

        out_dict = {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "logit_scale": self.logit_scale.exp(),
        }
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
    ):
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"
        if stopping_criteria is not None:
            import warnings
            warnings.warn(
                "stopping_criteria is deprecated and ignored. Use "
                "generation_config=GenerationConfig(...) for full control.",
                DeprecationWarning,
                stacklevel=2,
            )
        try:
            from .generation import MultimodalGenerationWrapper
            from transformers import GenerationConfig as GC
        except (ImportError, Exception) as e:
            raise RuntimeError(
                "Please install transformers for generate functionality. "
                "`pip install transformers`."
            ) from e

        device = image.device
        sot_token_id = 49406 if sot_token_id is None else sot_token_id
        eos_token_id = 49407 if eos_token_id is None else eos_token_id
        pad_token_id = self.pad_id if pad_token_id is None else pad_token_id

        with torch.no_grad():
            image_latent, image_embs = self._encode_image(image)

            squeeze_output = False
            if text is None:
                text = torch.full(
                    (image.shape[0], 1), sot_token_id,
                    device=device, dtype=torch.long,
                )
            elif text.dim() == 1:
                text = text.unsqueeze(0)
                squeeze_output = True

            was_training = self.training
            self.eval()

            vocab_size = self.text.token_embedding.weight.shape[0]
            wrapper = MultimodalGenerationWrapper(
                text_encoder_fn=lambda ids: self._encode_text(ids)[1],
                text_decoder_fn=self.text_decoder,
                image_embs=image_embs,
                vocab_size=vocab_size,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                bos_token_id=sot_token_id,
            )

            if generation_config is None:
                # seq_len / min_seq_len are *total* sequence lengths (including
                # the prompt) to match the original API semantics.
                gen_kwargs = dict(
                    max_length=seq_len,
                    min_length=min_seq_len,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    use_cache=False,
                )
                if generation_type == "beam_search":
                    if num_beam_groups > 1:
                        _logger.warning(
                            "Group beam search (num_beam_groups > 1) requires the "
                            "transformers community extension. Falling back to "
                            "standard beam search (num_beam_groups=1). Pass a "
                            "GenerationConfig directly for full control."
                        )
                        num_beam_groups = 1
                    gen_kwargs.update(
                        num_beams=num_beams,
                        num_beam_groups=num_beam_groups,
                    )
                elif generation_type == "top_p":
                    gen_kwargs.update(do_sample=True, top_p=top_p, temperature=temperature)
                elif generation_type == "top_k":
                    gen_kwargs.update(do_sample=True, top_k=top_k, temperature=temperature)
                else:
                    raise ValueError(
                        f"generation_type must be one of 'beam_search', 'top_p', 'top_k', "
                        f"got {generation_type!r}"
                    )
                generation_config = GC(**gen_kwargs)
            else:
                # KV-cache is not supported yet; force off regardless of what
                # the caller set to avoid cache-related errors.
                generation_config.use_cache = False

            output = wrapper.generate(
                text,
                generation_config=generation_config,
                image_embs=image_embs,
            )

            if fixed_output_length and output.shape[1] < seq_len:
                pad_len = seq_len - output.shape[1]
                output = torch.cat(
                    (output, torch.full(
                        (output.shape[0], pad_len), pad_token_id,
                        device=device, dtype=output.dtype,
                    )),
                    dim=1,
                )

            if squeeze_output:
                output = output.squeeze(0)

            self.train(was_training)
            return output

