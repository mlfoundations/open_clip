"""NaFlex spectrogram-ViT audio encoder for NaFlexClap (the contrastive sibling of GenLAP).

Wraps a timm ``NaFlexVit`` operating on pre-patchified mel-spectrogram tokens ``{patches, patch_coord,
patch_valid}`` (linear patch embed, attentive pooling) and exposes the :class:`AudioTower` encoder contract:
``forward(audio, device=None) -> {"embedding": pooled}``. Position scheme is timm's native 2-D **axial RoPE**
(``freq <-> H``, ``time <-> W``); gated attention and MRoPE stay opt-in via ``audio_cfg.naflexvit_cfg`` because
the multimodal-sequence machinery isn't needed for a standalone contrastive tower.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import CLIPAudioCfg


class NaFlexAudioEncoder(nn.Module):
    """timm ``NaFlexVit`` over spectrogram patches -> pooled audio features."""

    def __init__(self, audio_cfg: CLIPAudioCfg):
        super().__init__()
        from timm.models.naflexvit import NaFlexVit, NaFlexVitCfg

        # Single merged kwargs dict so explicit keys and user overrides can't collide into a duplicate-kwarg
        # TypeError; attn_gated is intentionally NOT set here (defaults off -> vanilla NaFlexVit).
        vit_kwargs = {
            "patch_size": (audio_cfg.patch_freq, audio_cfg.patch_time),
            "embed_proj_type": "linear",   # pre-patchified mel patches (no conv stem)
            "pos_embed": "none",           # position comes from RoPE
            "rope_type": audio_cfg.rope_type,   # default 'axial' (2-D freq/time RoPE)
            "global_pool": "map",               # attentive pooling
            **dict(audio_cfg.naflexvit_cfg),    # embed_dim/depth/num_heads/... (and attn_gated/mrope if opted in)
        }
        vit_cfg = NaFlexVitCfg(**vit_kwargs)
        self.vit = NaFlexVit(vit_cfg, in_chans=audio_cfg.in_chans, num_classes=0)
        self.embed_dim = vit_cfg.embed_dim

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True, impl: str = "inline"):
        if hasattr(self.vit, "set_grad_checkpointing"):
            self.vit.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        inner = self.vit.no_weight_decay() if hasattr(self.vit, "no_weight_decay") else set()
        return {f"vit.{name}" for name in inner}

    @torch.jit.ignore
    def layer_groups(self, pooler_in_head: bool = True):
        """Ordered ``(name, [members])`` groups of the spectrogram ViT trunk for layer-wise LR decay / lock,
        built from the timm ViT's own ``group_matcher`` (the same enumeration timm's native layer-decay uses).
        Members are the trunk parameters at each depth. The ``AudioTower`` wraps this and appends its projection
        head. ``pooler_in_head`` is accepted for a common signature but unused (no text pooler here).
        """
        from timm.models.helpers import group_parameters
        gparams = group_parameters(self.vit, self.vit.group_matcher())  # {layer_id: [param_name, ...]}
        groups = []
        for layer_id in sorted(gparams.keys()):
            members = [self.vit.get_parameter(name) for name in gparams[layer_id]]
            groups.append((f"layer.{layer_id}", members))
        return groups

    def forward(
            self,
            audio: Dict[str, torch.Tensor],
            device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        keys = ("patches", "patch_coord", "patch_valid")
        if device is not None:
            patch = {k: audio[k].to(device) for k in keys}
        else:
            patch = {k: audio[k] for k in keys}
        return {"embedding": self.vit(patch)}  # NaFlexVit(dict) -> [B, embed_dim] pooled
