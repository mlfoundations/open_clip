"""Plain-torch audio front-end ops: band-limited resampling and log-mel spectrograms.

Numerically equivalent to ``torchaudio.functional.resample`` (``sinc_interp_hann``) and
``torchaudio.transforms.MelSpectrogram`` (HTK mel scale, no filter norm) + ``AmplitudeToDB(top_db=None)``,
which are themselves plain torch. Kernels, windows and filter banks are cached per configuration.
"""
import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@lru_cache(maxsize=16)
@torch.inference_mode(False)  # cached tensors must stay usable outside inference mode
def _sinc_resample_kernel(
        orig_freq: int,
        new_freq: int,
        dtype: torch.dtype,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
) -> Tuple[torch.Tensor, int]:
    # orig_freq / new_freq are already divided by their gcd
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = torch.arange(-width, width + orig_freq, dtype=dtype)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=dtype)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    kernel = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernel *= window * (base_freq / orig_freq)
    return kernel, width


def resample(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """Resample ``waveform`` (..., time) from ``orig_freq`` to ``new_freq`` with a Hann-windowed sinc filter."""
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError(f"Sample rates must be positive, got {orig_freq} -> {new_freq}.")
    if int(orig_freq) != orig_freq or int(new_freq) != new_freq:
        raise ValueError(f"Sample rates must be integers, got {orig_freq} -> {new_freq}.")
    if orig_freq == new_freq:
        return waveform
    if not waveform.is_floating_point():
        raise TypeError(f"Expected a floating point waveform, got {waveform.dtype}.")
    gcd = math.gcd(int(orig_freq), int(new_freq))
    orig, new = int(orig_freq) // gcd, int(new_freq) // gcd
    kernel, width = _sinc_resample_kernel(orig, new, waveform.dtype)
    kernel = kernel.to(waveform.device)

    shape = waveform.shape
    waveform = waveform.reshape(-1, shape[-1])
    length = waveform.shape[-1]
    waveform = F.pad(waveform, (width, width + orig))
    out = F.conv1d(waveform[:, None], kernel, stride=orig)
    out = out.transpose(1, 2).reshape(waveform.shape[0], -1)
    # float32 ceil (not math.ceil) to match torchaudio's output length exactly, e.g. 1110642 @ 44.1k -> 32k
    # is 805908.03 but rounds to 805908.0 in float32
    out = out[..., :int(torch.ceil(torch.as_tensor(new * length / orig, dtype=torch.float32)))]
    return out.reshape(shape[:-1] + out.shape[-1:])


def _hz_to_mel(freq: float) -> float:
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


@lru_cache(maxsize=16)
@torch.inference_mode(False)
def mel_filterbank(n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int) -> torch.Tensor:
    """HTK-scale triangular mel filter bank (``n_freqs``, ``n_mels``), unnormalized."""
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_pts = torch.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    f_pts = _mel_to_hz(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down_slopes = -slopes[:, :-2] / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    return torch.clamp(torch.min(down_slopes, up_slopes), min=0.0)


@lru_cache(maxsize=16)
@torch.inference_mode(False)
def _hann_window(win_length: int) -> torch.Tensor:
    return torch.hann_window(win_length)


def log_mel_spectrogram(
        waveform: torch.Tensor,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        win_length: Optional[int] = None,
) -> torch.Tensor:
    """Power mel spectrogram in dB (``10 * log10(max(x, 1e-10))``), shape (..., n_mels, frames).

    Centered, reflect-padded STFT with a periodic Hann window.
    """
    win_length = win_length or n_fft
    f_max = float(sample_rate // 2) if f_max is None else f_max
    window = _hann_window(win_length).to(device=waveform.device, dtype=waveform.dtype)
    fb = mel_filterbank(n_fft // 2 + 1, float(f_min), float(f_max), n_mels, int(sample_rate))
    fb = fb.to(device=waveform.device, dtype=waveform.dtype)

    shape = waveform.shape
    spec = torch.stft(
        waveform.reshape(-1, shape[-1]),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    ).abs().pow(2.0)
    spec = spec.reshape(shape[:-1] + spec.shape[-2:])
    mel = torch.matmul(spec.transpose(-1, -2), fb).transpose(-1, -2)
    return 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
