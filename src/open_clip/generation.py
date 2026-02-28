"""Multimodal text generation via HuggingFace GenerationMixin.

This module requires ``transformers`` and is imported lazily by model
``generate()`` methods — ``import open_clip`` does not require transformers.
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
from transformers import GenerationConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class _SimpleConfig:
    """Minimal config stub satisfying GenerationMixin's attribute access."""

    def __init__(self, vocab_size: int = 49408):
        self.is_encoder_decoder = False
        self.vocab_size = vocab_size
        self._attn_implementation = "eager"

    def get_text_config(self, decoder=False):
        return self


class MultimodalGenerationWrapper(nn.Module, GenerationMixin):
    """Thin adapter making an encode-text + cross-attend-decode pipeline
    compatible with ``GenerationMixin.generate()``.

    Created transiently by a model's ``generate()`` method.  Holds pre-computed
    image embeddings and delegates ``forward()`` to the text encoder and decoder.

    Args:
        text_encoder_fn: Callable (text_ids) -> token_embs ``(B, S, D)``.
        text_decoder_fn: Callable (image_embs, token_embs) -> logits ``(B, S, V)``.
        image_embs: Pre-computed image context for cross-attention ``(B, N, D)``.
        vocab_size: Vocabulary size for config.
        pad_token_id: Pad token id.
        eos_token_id: End-of-sequence token id.
        bos_token_id: Start-of-sequence token id.
    """
    main_input_name = "input_ids"
    _is_stateful = False

    def __init__(
            self,
            text_encoder_fn: Callable,
            text_decoder_fn: Callable,
            image_embs: torch.Tensor,
            vocab_size: int = 49408,
            pad_token_id: int = 0,
            eos_token_id: int = 49407,
            bos_token_id: int = 49406,
    ):
        super().__init__()
        self._text_encoder_fn = text_encoder_fn
        self._text_decoder_fn = text_decoder_fn
        # Register as buffer so .device / expansion for beam search works.
        self.register_buffer("_image_embs", image_embs)
        self.config = _SimpleConfig(vocab_size=vocab_size)
        self.generation_config = GenerationConfig(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    @property
    def device(self) -> torch.device:
        return self._image_embs.device

    def prepare_inputs_for_generation(
            self,
            input_ids,
            image_embs=None,
            **kwargs,
    ):
        # TODO(kv-cache): When past_key_values is not None, slice input_ids
        # to only the last token and forward past_key_values + cache_position.
        return {
            "input_ids": input_ids,
            "image_embs": image_embs if image_embs is not None else self._image_embs,
        }

    def forward(
            self,
            input_ids: torch.Tensor,
            image_embs: Optional[torch.Tensor] = None,
            **kwargs,  # absorb cache_position, attention_mask, etc.
    ) -> CausalLMOutputWithPast:
        if image_embs is None:
            image_embs = self._image_embs
        # TODO(kv-cache): Accept past_key_values, pass to decoder, return
        # updated cache.  With KV-cache, only encode the new token positions
        # and concatenate cached K/V in the self-attention layers of the
        # decoder.  The cross-attention K/V (image_embs) are constant and can
        # be cached once.
        token_embs = self._text_encoder_fn(input_ids)
        logits = self._text_decoder_fn(image_embs, token_embs)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)

    def _reorder_cache(self, past_key_values, beam_idx):
        # TODO(kv-cache): Reorder cached K/V for beam search beam reordering.
        return past_key_values
