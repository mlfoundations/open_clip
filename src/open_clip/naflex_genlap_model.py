"""GenLAP: generative audio-language pretraining (the audio sibling of NaFlexGenLip).

Same single-transformer, autoregressive-LM-on-captions design as GenLIP, but the prefix is a NaFlex log-mel
**spectrogram** instead of an image. The audio front-end (``audio/naflex_audio.py``) patchifies the
spectrogram into the same ``{patches, patch_coord=(freq,time), patch_valid}`` contract, so this model reuses
the entire GenLIP trunk stack -- ``GenLipTrunk``, ``GenLipRotaryEmbedding``, the prefix-LM mask, the untied LM
head and the memory-efficient fused linear cross-entropy -- unchanged. Only the patch embed (``MelPatchEmbed``)
and the position-id construction differ.

RoPE has two modes, chosen by patch geometry:
  * **1-D time** (``rope_1d=True``; full-height freq strips, ``F == 1``): the time index is broadcast to all
    three MRoPE axes so every channel rotates by time -- a full-capacity 1-D RoPE, no wasted channels.
  * **2-D axial** (``rope_1d=False``; multi-row patches, ``F > 1``): ``t=0, h=freq, w=time`` -- the existing
    interleaved MRoPE with ``(freq, time)`` in the two spatial axes.

NOTE: this currently duplicates the small forward/encode glue from ``NaFlexGenLip`` rather than sharing a
base class, to avoid refactoring the checkpoint-bearing image model while its PR is in flight. Once that
lands, hoist the shared trunk/forward/loss into a ``_NaFlexGenModel`` base and make both models thin
subclasses (patch embed + position builder + modality adapter).
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio.naflex_audio import AudioNaFlexCfg, MelPatchEmbed
from .loss import fused_linear_cross_entropy
from .naflex_genlip_model import (
    GenLipRotaryEmbedding,
    GenLipTrunk,
    NaFlexGenLipTextCfg,
    NaFlexGenLipTrunkCfg,
    build_image_attn_mask,
    build_prefix_lm_mask,
    init_genlm_weights,
)


def build_audio_position_ids(
        patch_coord: torch.Tensor,
        patch_valid: torch.Tensor,
        text_valid: Optional[torch.Tensor] = None,
        rope_1d: bool = False,
) -> torch.Tensor:
    """3-axis MRoPE position ids for ``[audio_patches ; text]`` rows (audio-only if ``text_valid`` is None).

    ``patch_coord = (freq_idx, time_idx)``. With ``rope_1d=True`` the time index is broadcast to all three
    axes (full-capacity 1-D time RoPE); otherwise the 2-D layout ``t=0, h=freq, w=time`` is used. Text tokens,
    when present, take a single running index starting just past the audio's spatial extent.

    Args:
        patch_coord: ``(B, Ni, 2)`` of ``(freq, time)`` per patch.
        patch_valid: ``(B, Ni)`` bool.
        text_valid: Optional ``(B, Lt)`` bool; omit for the audio-only encoder path.
        rope_1d: Broadcast time to all axes (full-height-strip geometry) vs 2-D ``(freq, time)``.

    Returns:
        Long tensor ``(3, B, Ni + Lt)`` (``Lt = 0`` when ``text_valid`` is None).
    """
    b, ni, _ = patch_coord.shape
    device = patch_coord.device
    freq = patch_coord[..., 0].long()
    time = patch_coord[..., 1].long()
    lt = text_valid.shape[1] if text_valid is not None else 0

    pos = torch.zeros(3, b, ni + lt, dtype=torch.long, device=device)
    pv = patch_valid.bool()
    if rope_1d:
        pos[0, :, :ni] = time
        pos[1, :, :ni] = time
        pos[2, :, :ni] = time
        max_pos = torch.where(pv, time, torch.zeros_like(time)).amax(dim=1)  # (B,)
    else:
        pos[1, :, :ni] = freq
        pos[2, :, :ni] = time
        f_valid = torch.where(pv, freq, torch.zeros_like(freq))
        t_valid = torch.where(pv, time, torch.zeros_like(time))
        max_pos = torch.maximum(f_valid.amax(dim=1), t_valid.amax(dim=1))  # (B,)

    if lt > 0:
        text_pos = (max_pos[:, None] + 1) + torch.arange(lt, device=device)[None, :]  # (B, Lt)
        pos[:, :, ni:] = text_pos[None].expand(3, b, lt)
    return pos


class NaFlexGenLap(nn.Module):
    """GenLAP unified audio-language model: NaFlex spectrogram patches through the shared GenLIP trunk."""

    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            audio_naflex_cfg: Union[AudioNaFlexCfg, Dict],
            text_cfg: Union[NaFlexGenLipTextCfg, Dict],
            genlap_cfg: Union[NaFlexGenLipTrunkCfg, Dict],
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = True,
            rope_1d: Optional[bool] = None,
            **kwargs,
    ):
        super().__init__()
        if isinstance(audio_naflex_cfg, dict):
            audio_naflex_cfg = AudioNaFlexCfg(**audio_naflex_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = NaFlexGenLipTextCfg(**text_cfg)
        if isinstance(genlap_cfg, dict):
            genlap_cfg = NaFlexGenLipTrunkCfg(**genlap_cfg)

        self.audio_cfg = audio_naflex_cfg
        self.text_cfg = text_cfg
        self.trunk_cfg = genlap_cfg
        self.output_dict = output_dict
        self.embed_dim = embed_dim
        # 1-D time vs 2-D axial rope; default derives from patch geometry (full-height strips -> 1-D).
        self.rope_1d = audio_naflex_cfg.is_1d_time if rope_1d is None else rope_1d

        width = genlap_cfg.width
        text_embed_dim = genlap_cfg.text_embed_dim
        self.pad_id = text_cfg.pad_id
        self.context_length = text_cfg.context_length

        # audio side
        self.audio_embed = MelPatchEmbed(audio_naflex_cfg, width)
        # text side (identical to GenLIP)
        self.text_embed = nn.Embedding(text_cfg.vocab_size, text_embed_dim, padding_idx=text_cfg.pad_id)
        self.in_proj = nn.Linear(text_embed_dim, width) if text_embed_dim != width else nn.Identity()
        self.out_proj = nn.Linear(width, text_embed_dim) if text_embed_dim != width else nn.Identity()
        self.lm_head = nn.Linear(text_embed_dim, text_cfg.vocab_size, bias=False)  # untied
        # downstream audio-encoder projector (Identity when embed_dim == width)
        self.audio_proj = nn.Linear(width, embed_dim) if embed_dim != width else nn.Identity()

        # shared trunk + rotary
        self.rotary = GenLipRotaryEmbedding(genlap_cfg)
        self.trunk = GenLipTrunk(genlap_cfg)

        self.init_weights()
        if cast_dtype is not None:
            self.to(dtype=cast_dtype)

    @torch.no_grad()
    def init_weights(self, std: float = 0.02):
        init_genlm_weights(self, std)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        if impl == 'composable' and enable:
            from torch.distributed._composable import checkpoint as composable_checkpoint
            for block in self.trunk.layers:
                composable_checkpoint(block)
        else:
            self.trunk.grad_checkpointing = enable

    def fsdp_shard_modules(self) -> List[Tuple[str, nn.Module]]:
        """``(name, module)`` pairs to wrap for FSDP / activation checkpointing (trunk blocks)."""
        return [(f"trunk.layers.{i}", block) for i, block in enumerate(self.trunk.layers)]

    def encode_audio(self, audio: Dict[str, torch.Tensor], normalize: bool = False) -> torch.Tensor:
        """Audio-only bidirectional pass -> pooled features (the downstream encoder; no text)."""
        patch_valid = audio['patch_valid']
        x = self.audio_embed(audio['patches'])
        attn_mask = build_image_attn_mask(patch_valid)
        pos = build_audio_position_ids(audio['patch_coord'], patch_valid, rope_1d=self.rope_1d)
        cos, sin = self.rotary(x, pos)
        x = self.trunk(x, attn_mask, cos, sin)  # ln_post applied inside trunk
        pv = patch_valid.to(x.dtype)
        pooled = (x * pv.unsqueeze(-1)).sum(dim=1) / pv.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = self.audio_proj(pooled)
        return F.normalize(pooled, dim=-1) if normalize else pooled

    def _encode(
            self,
            audio: Dict[str, torch.Tensor],
            text: torch.Tensor,
            text_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Run the shared trunk over ``[audio_patches ; caption_tokens]``; returns ``(hidden, audio_seq_len)``."""
        patch_valid = audio['patch_valid']
        aud_emb = self.audio_embed(audio['patches'])           # (B, Ni, width)
        txt_emb = self.in_proj(self.text_embed(text))          # (B, Lt, width)
        h = torch.cat([aud_emb, txt_emb], dim=1)               # (B, S, width)

        attn_mask = build_prefix_lm_mask(patch_valid, text_valid)
        pos = build_audio_position_ids(audio['patch_coord'], patch_valid, text_valid, rope_1d=self.rope_1d)
        cos, sin = self.rotary(h, pos)
        h = self.trunk(h, attn_mask, cos, sin)
        return self.out_proj(h), aud_emb.shape[1]

    def forward(
            self,
            audio: Dict[str, torch.Tensor],
            text: torch.Tensor,
            text_valid: Optional[torch.Tensor] = None,
            compute_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generative forward over ``[audio_patches ; caption_tokens]`` (see :class:`NaFlexGenLip`).

        Args:
            audio: Dict with ``patches`` ``(B, Ni, C*p_f*p_t)``, ``patch_coord`` ``(B, Ni, 2)`` of
                ``(freq, time)`` and ``patch_valid`` ``(B, Ni)``.
            text: Caption token ids ``(B, Lt)`` padded with ``pad_id``.
            text_valid: Optional ``(B, Lt)`` bool mask; derived from ``text != pad_id`` when omitted.
            compute_loss: When True, return the memory-efficient fused autoregressive ``loss`` over the
                text-predicting positions only; otherwise return full ``logits``.
        """
        if text_valid is None:
            text_valid = text != self.pad_id

        hidden, ni = self._encode(audio, text, text_valid)  # (B, S, D), Ni

        if compute_loss:
            pred = hidden[:, ni - 1:-1, :]  # (B, Lt, D): position p predicts token p+1
            target = torch.where(text_valid, text, torch.full_like(text, -100))
            loss = fused_linear_cross_entropy(
                pred.reshape(-1, pred.shape[-1]),
                self.lm_head.weight,
                target.reshape(-1),
                bias=self.lm_head.bias,
                ignore_index=-100,
            )
            # return only tensors (torch.compile/DDP graph-splitter safety) -- see NaFlexGenLip.forward.
            return {'loss': loss}

        logits = self.lm_head(hidden)
        return {'logits': logits, 'audio_seq_len': ni}
