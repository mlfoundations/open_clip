"""Multimodal text generation via HuggingFace GenerationMixin.

This module requires ``transformers`` and is imported lazily by model
``generate()`` methods — ``import open_clip`` does not require transformers.
"""
import copy
import logging
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import GenerationConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

_logger = logging.getLogger(__name__)

_LEGACY_SOT_TOKEN_ID = 49406
_LEGACY_EOS_TOKEN_ID = 49407
_LEGACY_PAD_TOKEN_ID = 0


class _SimpleConfig:
    """Minimal config stub satisfying GenerationMixin's attribute access."""

    def __init__(
            self,
            vocab_size: int = 49408,
            pad_token_id: int = _LEGACY_PAD_TOKEN_ID,
            eos_token_id: int = _LEGACY_EOS_TOKEN_ID,
            bos_token_id: int = _LEGACY_SOT_TOKEN_ID,
    ):
        self.is_encoder_decoder = False
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self._attn_implementation = "eager"
        self._experts_implementation = None  # probed by transformers >= 5.x decode optimization

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
        self.config = _SimpleConfig(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )
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
            context_valid=None,
            **kwargs,
    ):
        # TODO(kv-cache): When past_key_values is not None, slice input_ids
        # to only the last token and forward past_key_values + cache_position.
        inputs = {
            "input_ids": input_ids,
            "image_embs": image_embs if image_embs is not None else self._image_embs,
        }
        if context_valid is not None:
            inputs["context_valid"] = context_valid
        return inputs

    def forward(
            self,
            input_ids: torch.Tensor,
            image_embs: Optional[torch.Tensor] = None,
            context_valid: Optional[torch.Tensor] = None,
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
        if context_valid is not None:
            # 3-arg form only when masking is in play; 2-arg decoder fns stay compatible
            logits = self._text_decoder_fn(image_embs, token_embs, context_valid)
        else:
            logits = self._text_decoder_fn(image_embs, token_embs)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)

    def _reorder_cache(self, past_key_values, beam_idx):
        # TODO(kv-cache): Reorder cached K/V for beam search beam reordering.
        return past_key_values


def _normalize_token_id(value: Any, allow_sequence: bool = False):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, (list, tuple)):
        values = [int(v) for v in value if v is not None]
        if not values:
            return None
        return values if allow_sequence else values[0]
    return int(value)


def _iter_generation_sources(model: nn.Module):
    seen = set()
    for source in (
            model,
            getattr(model, 'text', None),
            getattr(model, 'text_decoder', None),
            getattr(model, 'decoder', None),
    ):
        if source is None or id(source) in seen:
            continue
        seen.add(id(source))
        yield source
        config = getattr(source, 'config', None)
        if config is not None and id(config) not in seen:
            seen.add(id(config))
            yield config


def _resolve_from_sources(sources, names, allow_sequence: bool = False):
    for source in sources:
        for name in names:
            value = getattr(source, name, None)
            value = _normalize_token_id(value, allow_sequence=allow_sequence)
            if value is not None:
                return value
    return None


def resolve_generation_token_ids(
        model: nn.Module,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        sot_token_id: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Resolve generation special tokens from explicit args, then built model/tower attrs, then legacy CLIP ids."""
    sources = tuple(_iter_generation_sources(model))
    pad_token_id = _normalize_token_id(pad_token_id)
    if pad_token_id is None:
        pad_token_id = _resolve_from_sources(sources, ('pad_id', 'pad_token_id'))
    if pad_token_id is None:
        pad_token_id = _LEGACY_PAD_TOKEN_ID

    eos_token_id = _normalize_token_id(eos_token_id, allow_sequence=True)
    if eos_token_id is None:
        eos_token_id = _resolve_from_sources(
            sources,
            ('eos_id', 'eos_token_id', 'eot_token_id', 'sep_token_id'),
            allow_sequence=True,
        )
    if eos_token_id is None:
        eos_token_id = _LEGACY_EOS_TOKEN_ID

    sot_token_id = _normalize_token_id(sot_token_id)
    if sot_token_id is None:
        sot_token_id = _resolve_from_sources(
            sources,
            ('bos_id', 'bos_token_id', 'sot_id', 'sot_token_id', 'cls_token_id'),
        )
    if sot_token_id is None:
        sot_token_id = _LEGACY_SOT_TOKEN_ID

    return pad_token_id, eos_token_id, sot_token_id


def resolve_generation_vocab_size(decoder: nn.Module) -> int:
    """Resolve the logits vocabulary size from the decoder side of a multimodal generator."""
    value = getattr(decoder, 'vocab_size', None)
    if value is not None:
        return int(value)

    lm_head = getattr(decoder, 'lm_head', None)
    if isinstance(lm_head, nn.Linear):
        return int(lm_head.out_features)
    if isinstance(lm_head, nn.Parameter):
        return int(lm_head.shape[-1])

    text_projection = getattr(decoder, 'text_projection', None)
    if isinstance(text_projection, nn.Parameter):
        return int(text_projection.shape[-1])
    if isinstance(text_projection, nn.Linear):
        return int(text_projection.out_features)

    # last resort: on some decoders (e.g. ModernMultimodalDecoder) output_dim is the contrastive
    # embed dim rather than the vocab, so only trust it once vocab_size and head introspection miss
    value = getattr(decoder, 'output_dim', None)
    if value is not None:
        return int(value)

    raise AttributeError("Could not resolve generation vocab size from decoder.")


def _right_padded(valid: torch.Tensor) -> bool:
    seen_pad = (~valid).cummax(dim=1).values
    return not bool((seen_pad & valid).any())


def prepare_generation_prompt(
        text: Optional[torch.Tensor],
        text_valid: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        sot_token_id: int,
) -> Tuple[torch.Tensor, bool]:
    """Create/validate the common-length prompt consumed by GenerationMixin."""
    squeeze_output = False
    if text is None:
        return torch.full((batch_size, 1), sot_token_id, device=device, dtype=torch.long), squeeze_output

    if text.dim() == 1:
        text = text.unsqueeze(0)
        if text_valid is not None:
            text_valid = text_valid.unsqueeze(0)
        squeeze_output = batch_size == 1
    if text.dim() != 2:
        raise ValueError(f"generation prompt must be a 1D or 2D token tensor, got shape {tuple(text.shape)}.")
    if text.shape[0] != batch_size:
        raise ValueError(
            f"generation prompt batch ({text.shape[0]}) must match image batch ({batch_size})."
        )

    text = text.to(device=device, dtype=torch.long)
    if text_valid is not None:
        text_valid = text_valid.to(device=device, dtype=torch.bool)
        if text_valid.shape != text.shape:
            raise ValueError(
                f"text_valid shape {tuple(text_valid.shape)} must match text shape {tuple(text.shape)}."
            )
        if not _right_padded(text_valid):
            raise ValueError("generation text_valid must describe right-padded prompts.")
        if text_valid.any():
            keep_len = int(text_valid.any(dim=0).nonzero()[-1].item()) + 1
        else:
            keep_len = 1
        text = text[:, :keep_len]
        text_valid = text_valid[:, :keep_len]
        if not bool(text_valid.all()):
            raise ValueError(
                "Generation prompts with text_valid must have the same valid length in every row; "
                "pass same-length unpadded prompts or generate each prompt length separately."
            )

    return text, squeeze_output


def build_generation_config(
        generation_config: Optional[GenerationConfig],
        generation_type: str,
        seq_len: int,
        min_seq_len: int,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: int,
        num_beam_groups: int,
        repetition_penalty: float,
        pad_token_id: int,
        eos_token_id: int,
        sot_token_id: int,
) -> GenerationConfig:
    if generation_config is not None:
        generation_config = copy.deepcopy(generation_config)
        if getattr(generation_config, 'pad_token_id', None) is None:
            generation_config.pad_token_id = pad_token_id
        if getattr(generation_config, 'eos_token_id', None) is None:
            generation_config.eos_token_id = eos_token_id
        if getattr(generation_config, 'bos_token_id', None) is None:
            generation_config.bos_token_id = sot_token_id
        generation_config.use_cache = False
        return generation_config

    if seq_len <= min_seq_len:
        raise ValueError("seq_len must be larger than min_seq_len.")

    gen_kwargs = dict(
        max_length=seq_len,
        min_length=min_seq_len,
        repetition_penalty=repetition_penalty,
        bos_token_id=sot_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=False,
    )
    if generation_type == "beam_search":
        if num_beam_groups > 1:
            _logger.warning(
                "Group beam search (num_beam_groups > 1) requires the transformers community "
                "extension. Falling back to standard beam search (num_beam_groups=1). Pass a "
                "GenerationConfig directly for full control."
            )
            num_beam_groups = 1
        gen_kwargs.update(num_beams=num_beams, num_beam_groups=num_beam_groups)
    elif generation_type == "top_p":
        gen_kwargs.update(do_sample=True, top_p=top_p, temperature=temperature)
    elif generation_type == "top_k":
        gen_kwargs.update(do_sample=True, top_k=top_k, temperature=temperature)
    else:
        raise ValueError(
            f"generation_type must be one of 'beam_search', 'top_p', 'top_k', got {generation_type!r}"
        )
    return GenerationConfig(**gen_kwargs)


def validate_generation_lengths(
        prompt_len: int,
        generation_config: GenerationConfig,
        seq_len: int,
        max_seq_len: Optional[int],
        context_length: Optional[int],
) -> int:
    max_new_tokens = getattr(generation_config, 'max_new_tokens', None)
    if max_new_tokens is not None:
        max_length = prompt_len + int(max_new_tokens)
    else:
        max_length = getattr(generation_config, 'max_length', None) or seq_len
    if max_seq_len is not None and max_length > max_seq_len:
        raise ValueError(f"generation max_length ({max_length}) cannot exceed max_seq_len ({max_seq_len}).")
    if context_length is not None and context_length > 0 and max_length > context_length:
        raise ValueError(
            f"generation max_length ({max_length}) cannot exceed decoder context_length ({context_length})."
        )
    if prompt_len > max_length:
        raise ValueError(f"generation prompt length ({prompt_len}) cannot exceed max_length ({max_length}).")
    return max_length


def generate_multimodal(
        model: nn.Module,
        image: torch.Tensor,
        image_embs_fn: Callable[[torch.Tensor], torch.Tensor],
        text_encoder_fn: Callable[[torch.Tensor], torch.Tensor],
        text_decoder_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        decoder: nn.Module,
        text: Optional[torch.Tensor] = None,
        text_valid: Optional[torch.Tensor] = None,
        seq_len: int = 30,
        max_seq_len: int = 77,
        temperature: float = 1.,
        generation_type: str = "beam_search",
        top_p: float = 0.1,
        top_k: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        sot_token_id: Optional[int] = None,
        num_beams: int = 6,
        num_beam_groups: int = 3,
        min_seq_len: int = 5,
        stopping_criteria=None,
        repetition_penalty: float = 1.0,
        fixed_output_length: bool = False,
        generation_config: Optional[GenerationConfig] = None,
) -> torch.Tensor:
    if stopping_criteria is not None:
        import warnings
        warnings.warn(
            "stopping_criteria is deprecated and ignored. Use "
            "generation_config=GenerationConfig(...) for full control.",
            DeprecationWarning,
            stacklevel=2,
        )

    pad_token_id, eos_token_id, sot_token_id = resolve_generation_token_ids(
        model,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        sot_token_id=sot_token_id,
    )
    generation_config = build_generation_config(
        generation_config=generation_config,
        generation_type=generation_type,
        seq_len=seq_len,
        min_seq_len=min_seq_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        repetition_penalty=repetition_penalty,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        sot_token_id=sot_token_id,
    )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            # image_embs_fn may return either embs or (embs, context_valid) -- NaFlex token-mode
            # towers surface patch validity so padded K/V get masked in cross-attention
            image_embs = image_embs_fn(image)
            context_valid = None
            if isinstance(image_embs, tuple):
                image_embs, context_valid = image_embs
            prompt, squeeze_output = prepare_generation_prompt(
                text=text,
                text_valid=text_valid,
                batch_size=image_embs.shape[0],
                device=image_embs.device,
                sot_token_id=sot_token_id,
            )
            target_len = validate_generation_lengths(
                prompt_len=prompt.shape[1],
                generation_config=generation_config,
                seq_len=seq_len,
                max_seq_len=max_seq_len,
                context_length=getattr(decoder, 'context_length', None),
            )

            wrapper = MultimodalGenerationWrapper(
                text_encoder_fn=text_encoder_fn,
                text_decoder_fn=text_decoder_fn,
                image_embs=image_embs,
                vocab_size=resolve_generation_vocab_size(decoder),
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                bos_token_id=sot_token_id,
            )
            generate_kwargs = dict(image_embs=image_embs)
            if context_valid is not None:
                # passed as a model kwarg (not a closure) so HF beam search expands it
                # alongside image_embs via _expand_inputs_for_generation
                generate_kwargs['context_valid'] = context_valid
            output = wrapper.generate(
                prompt,
                generation_config=generation_config,
                **generate_kwargs,
            )

            if fixed_output_length and output.shape[1] < target_len:
                pad_len = target_len - output.shape[1]
                output = torch.cat(
                    (output, torch.full(
                        (output.shape[0], pad_len), pad_token_id,
                        device=image_embs.device, dtype=output.dtype,
                    )),
                    dim=1,
                )

            if squeeze_output:
                output = output.squeeze(0)
            return output
    finally:
        model.train(was_training)
