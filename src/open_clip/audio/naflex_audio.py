"""NaFlex-style variable-length audio front-end (shared by generative GenLAP and contrastive NaFlexClap).

Replaces CLAP's fixed-clip + fusion preprocessing (see ``audio/transform.py``) with native variable-length
handling: a log-mel spectrogram is patchified into a variable number of ``(freq, time)`` tokens, producing
exactly the ``{patches, patch_coord, patch_valid}`` per-sample contract that the image NaFlex pipeline,
batch scheduler, total-token budgeting and length bucketing already consume -- so the whole NaFlex batching
stack (collate, scheduler, bucketer) works for audio with no changes.

Patch geometry is configurable via ``(patch_freq, patch_time)``:
  * full-height freq strips (``patch_freq == n_mels`` -> a single freq row, ``F == 1``) give a 1-D
    time-token sequence, consumed with a 1-D time RoPE;
  * multi-row patches (``patch_freq < n_mels`` -> ``F > 1``) give a 2-D ``(freq, time)`` grid for axial MRoPE.

Position ids are NOT built here: ``patch_coord = (freq, time)`` plays the exact role of image ``(h, w)``, so
the existing ``build_image_position_ids`` / ``build_mrope_position_ids`` (naflex_genlip_model) and the timm
NaFlexVit rope path consume audio coords directly. The freq axis is all-zeros when ``F == 1``, which is what
makes the 1-D case fall out of the same builder without a separate code path.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class AudioNaFlexCfg:
    """Self-contained config for the NaFlex audio front-end: mel extraction + patch geometry + embed.

    Mel-extraction fields use CLAP-compatible names (``CLIPAudioCfg``) so ``mel_cfg_dict()`` maps straight onto
    the reused ``audio/transform._get_mel`` with no synonyms.
    """

    # mel extraction (CLAP defaults)
    sample_rate: int = 48000
    window_size: int = 1024    # MelSpectrogram n_fft / win_length
    hop_size: int = 480
    fmin: int = 50
    fmax: int = 14000
    # patch geometry
    n_mels: int = 64           # spectrogram height (freq bins); must be divisible by patch_freq; == mel_bins
    patch_freq: int = 64       # p_f: freq bins per patch (== n_mels -> full-height strips -> 1-D time tokens)
    patch_time: int = 4        # p_t: time frames per patch
    in_chans: int = 1          # mel channels (1 = log-mel energy; >1 to stack delta / delta-delta)
    input_norm: bool = False   # LayerNorm over the flattened patch before projection
    proj_bias: bool = True
    # How mel_to_patches fills the final partial time patch (T not a multiple of patch_time, for variable-length
    # audio). "floor" = per-clip mel min (quiet relative to the clip); "silence" = absolute dB silence floor
    # (-100, true acoustic silence); "repeat" = repeat the last real frame.
    patch_pad_mode: str = "floor"

    @classmethod
    def from_clip_audio_cfg(cls, clip_cfg) -> "AudioNaFlexCfg":
        """Build the NaFlex audio front-end cfg from a CLAP ``CLIPAudioCfg`` (NaFlexClap reuses its mel fields)."""
        return cls(
            sample_rate=clip_cfg.sample_rate,
            window_size=clip_cfg.window_size,
            hop_size=clip_cfg.hop_size,
            fmin=clip_cfg.fmin,
            fmax=clip_cfg.fmax,
            n_mels=clip_cfg.mel_bins,
            patch_freq=clip_cfg.patch_freq,
            patch_time=clip_cfg.patch_time,
            in_chans=clip_cfg.in_chans,
            patch_pad_mode=clip_cfg.patch_pad_mode,
        )

    def mel_cfg_dict(self) -> Dict[str, int]:
        """Return the dict keys ``audio/transform._get_mel`` reads (CLAP key names)."""
        return {
            "sample_rate": self.sample_rate,
            "window_size": self.window_size,
            "hop_size": self.hop_size,
            "mel_bins": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
        }

    @property
    def patch_dim(self) -> int:
        return self.in_chans * self.patch_freq * self.patch_time

    @property
    def freq_tokens(self) -> int:
        """Number of freq patches ``F`` along the mel axis."""
        if self.n_mels % self.patch_freq != 0:
            raise ValueError(f"n_mels={self.n_mels} not divisible by patch_freq={self.patch_freq}")
        return self.n_mels // self.patch_freq

    @property
    def is_1d_time(self) -> bool:
        """True when patches span the full mel height (``F == 1``) -> a 1-D time-token sequence."""
        return self.freq_tokens == 1


# Absolute dB-mel silence floor for the `_get_mel` path: AmplitudeToDB(top_db=None) on zero power bottoms out at
# 10*log10(1e-10) = -100 dB (verified: an all-zero waveform yields exactly -100 everywhere). It also matches the
# mel of the zero-padded region used for sub-window clips, so "silence"-padded tails are consistent with those.
_MEL_SILENCE_DB = -100.0


def _pad_mel_time(mel: torch.Tensor, pad_frames: int, pad_mode: str = "floor") -> torch.Tensor:
    """Append ``pad_frames`` time frames to a ``(C, T, n_mels)`` mel to complete the final time patch.

    Pad fill by ``pad_mode``:
      * ``"floor"``: the per-clip mel min (``amin``) -- quiet *relative to this clip*, so a loud, dense clip's
        floor can sit well above true silence.
      * ``"silence"``: the absolute dB-mel silence floor (``-100`` for ``_get_mel``) -- what an all-zero waveform
        produces, independent of clip loudness.
      * ``"repeat"``: repeat the last real mel frame.
    The padding is patch-internal: real audio stays front-contiguous and the completed patch still carries real
    frames, so it remains ``patch_valid=True``.
    """
    c, t, n_mels = mel.shape
    if pad_mode == "repeat":
        if t > 0:
            tail = mel[:, -1:, :].expand(c, pad_frames, n_mels)
            return torch.cat([mel, tail], dim=1)
        fill = _MEL_SILENCE_DB  # empty clip: nothing to repeat -> absolute silence
    elif pad_mode == "floor":
        fill = mel.amin().item() if t > 0 else _MEL_SILENCE_DB
    elif pad_mode == "silence":
        fill = _MEL_SILENCE_DB
    else:
        raise ValueError(f"unknown patch_pad_mode={pad_mode!r} (expected 'floor', 'silence', or 'repeat')")
    pad = mel.new_full((c, pad_frames, n_mels), fill)
    return torch.cat([mel, pad], dim=1)


def mel_to_patches(
        mel: torch.Tensor,
        patch_freq: int,
        patch_time: int,
        in_chans: int = 1,
        pad_mode: str = "floor",
) -> Dict[str, torch.Tensor]:
    """Patchify one log-mel spectrogram into NaFlex ``(freq, time)`` patch tokens.

    Audio is variable-length, so unlike the image path the time axis is **padded up** to a whole number of
    patches rather than cropped: ``T`` is rounded UP to a multiple of ``patch_time`` (minimum one patch), so no
    time frames are dropped and a sub-patch clip (``T < patch_time``) still yields one patch. The completing
    frames are patch-internal fill (see ``pad_mode``); the patches themselves carry real audio and are all valid.

    Args:
        mel: ``(T, n_mels)`` (single channel) or ``(C, T, n_mels)`` log-mel spectrogram, time-major as produced
            by the CLAP mel extractor. ``T`` is variable; ``n_mels`` must be divisible by ``patch_freq``.
        patch_freq: ``p_f`` -- freq bins per patch (``== n_mels`` for full-height strips).
        patch_time: ``p_t`` -- time frames per patch.
        in_chans: channel count ``C`` (1 when ``mel`` is 2-D).
        pad_mode: fill for the final partial time patch -- ``"floor"`` (per-clip min), ``"silence"`` (absolute
            dB floor), or ``"repeat"`` (last real frame).

    Returns:
        Per-sample dict ``{patches, patch_coord, patch_valid}`` (the image-NaFlex contract): ``patches``
        ``(N, C*p_f*p_t)``, ``patch_coord`` ``(N, 2)`` of ``(freq_idx, time_idx)``, ``patch_valid`` all-True
        ``(N,)``. ``N = F*Tt`` with ``F = n_mels//p_f`` and ``Tt = ceil(T/p_t)``; rows are ordered freq-outer,
        time-inner.
    """
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)  # (1, T, n_mels)
    if mel.ndim != 3:
        raise ValueError(f"mel must be (T, n_mels) or (C, T, n_mels), got shape {tuple(mel.shape)}")
    c, t, n_mels = mel.shape
    if c != in_chans:
        raise ValueError(f"mel channel dim {c} != in_chans {in_chans}")
    if n_mels % patch_freq != 0:
        raise ValueError(f"n_mels={n_mels} not divisible by patch_freq={patch_freq}")
    f = n_mels // patch_freq
    tt = max(1, math.ceil(t / patch_time))   # round UP: no time frames dropped; >= 1 patch even for a sub-patch clip
    pad_frames = tt * patch_time - t
    if pad_frames > 0:
        mel = _pad_mel_time(mel, pad_frames, pad_mode)   # complete the final time patch (floor/silence/repeat fill)

    mel = mel.reshape(c, tt, patch_time, f, patch_freq)  # (C, Tt, p_t, F, p_f)
    mel = mel.permute(3, 1, 0, 2, 4).contiguous()        # (F, Tt, C, p_t, p_f): freq outer, time inner
    patches = mel.reshape(f * tt, c * patch_time * patch_freq)

    freq_idx = torch.arange(f, device=patches.device).repeat_interleave(tt)
    time_idx = torch.arange(tt, device=patches.device).repeat(f)
    patch_coord = torch.stack([freq_idx, time_idx], dim=1).long()
    patch_valid = torch.ones(f * tt, dtype=torch.bool, device=patches.device)
    return {"patches": patches, "patch_coord": patch_coord, "patch_valid": patch_valid}


class MelPatchEmbed(nn.Module):
    """Linear embedding of flattened mel patches -> trunk width (audio analog of ``GenLipPatchEmbed``)."""

    def __init__(self, cfg: AudioNaFlexCfg, width: int):
        super().__init__()
        self.norm_input = nn.LayerNorm(cfg.patch_dim) if cfg.input_norm else None
        self.proj = nn.Linear(cfg.patch_dim, width, bias=cfg.proj_bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if self.norm_input is not None:
            patches = self.norm_input(patches)
        return self.proj(patches)


class AudioNaFlexPatchify:
    """Picklable transform: ``(waveform, sample_rate)`` -> ``{patches, patch_coord, patch_valid}``.

    Reuses the CLAP log-mel extractor (``audio/transform._get_mel``; torchaudio is imported lazily). Unlike
    ``AudioPreprocess`` there is no fixed-clip truncation or fusion -- the time axis stays native and variable,
    and the NaFlex collate pads the batch to the seq-len bucket with a valid mask. An optional
    ``max_audio_tokens`` caps the per-sample token count by keeping whole time columns (see below).
    """

    def __init__(self, cfg: AudioNaFlexCfg, max_audio_tokens: Optional[int] = None):
        if max_audio_tokens is not None and max_audio_tokens < cfg.freq_tokens:
            raise ValueError(
                f"max_audio_tokens={max_audio_tokens} is smaller than freq_tokens={cfg.freq_tokens}: a NaFlex "
                f"audio seq-len bucket must fit at least one whole freq column ({cfg.freq_tokens} tokens), else "
                f"the collate would truncate the flattened patches and drop freq rows. Set the seq-len bucket "
                f"(--naflex-seq-lens) >= {cfg.freq_tokens}."
            )
        self.cfg = cfg
        self.mel_cfg = cfg.mel_cfg_dict()
        self.target_sr = cfg.sample_rate
        self.max_audio_tokens = max_audio_tokens

    def __call__(self, audio_data: Tuple[torch.Tensor, int]) -> Dict[str, torch.Tensor]:
        import torchaudio

        from .transform import _get_mel

        waveform, sr = audio_data
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        waveform = waveform.squeeze(0)
        # The mel STFT (center=True, reflect pad) needs at least one analysis window of samples, so pad sub-window
        # clips up to window_size with zeros -- otherwise torch.stft errors on a tiny clip. Patch-level completion
        # for clips still shorter than one time patch is handled in mel_to_patches via patch_pad_mode.
        if waveform.shape[-1] < self.cfg.window_size:
            waveform = torch.cat([waveform, waveform.new_zeros(self.cfg.window_size - waveform.shape[-1])])
        mel = _get_mel(waveform, self.mel_cfg)  # (T, n_mels)

        if self.max_audio_tokens is not None:
            # Total tokens = freq_tokens * time_tokens, so cap by WHOLE time columns (not a flat patch count):
            # keep the first max_time time-patches -> freq_tokens * max_time <= max_audio_tokens, with every
            # freq row preserved. Truncating the flattened patch list instead would drop entire freq rows
            # under the freq-outer/time-inner ordering.
            max_time = max(1, self.max_audio_tokens // self.cfg.freq_tokens)
            max_frames = max_time * self.cfg.patch_time
            if mel.shape[0] > max_frames:
                mel = mel[:max_frames]

        return mel_to_patches(
            mel, self.cfg.patch_freq, self.cfg.patch_time, self.cfg.in_chans, pad_mode=self.cfg.patch_pad_mode,
        )


class AudioNaFlexTransformFactory:
    """NaFlex transform-factory for audio: produces the audio patchify transform on demand.

    Matches the contract the NaFlex scheduler/eval helper call -- ``transform_factory(max_seq_len, patch_size)``.
    Audio geometry comes from the :class:`AudioNaFlexCfg` so ``patch_size`` is ignored; ``max_seq_len`` (the
    seq-len bucket) becomes the per-sample audio-token cap. Picklable for forkserver workers.
    """

    is_naflex_transform_factory = True
    is_naflex_eval_transform_factory = True

    def __init__(self, cfg: AudioNaFlexCfg, pack_prefix: bool = False):
        self.cfg = cfg
        # GenLAP's packed-row layout (resolved from the actual model, not a config-name lookup) so the data path
        # can pick the right length-bucketing key off the transform it already receives -- see AudioLengthBucketer.
        self.pack_prefix = pack_prefix

    def __call__(self, max_seq_len: Optional[int] = None, patch_size: Any = None) -> AudioNaFlexPatchify:
        return AudioNaFlexPatchify(self.cfg, max_audio_tokens=max_seq_len)
