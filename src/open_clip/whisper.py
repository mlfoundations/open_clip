"""Whisper audio encoder for CLAP models.

Adapted from openai/whisper and open_clap_exp_gijs. Contains the WhisperEncoder
(Conv1d + Transformer blocks) that computes log-mel spectrograms on-the-fly from
raw waveforms and outputs projected sequence embeddings.
"""
import logging
import os
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from torch.nn.functional import scaled_dot_product_attention
    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


# Hard-coded audio hyperparameters (from openai/whisper)
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames in a mel spectrogram input


def _exact_div(x, y):
    assert x % y == 0
    return x // y


def get_T_after_cnn(L_in, dilation=1):
    """Compute output length after the two Conv1d layers in WhisperEncoder."""
    for (padding, kernel_size, stride) in [(1, 3, 1), (1, 3, 2)]:
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out


# ---------------------------------------------------------------------------
# Mel spectrogram computation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """Load the mel filterbank matrix for projecting STFT into a Mel spectrogram.

    The filterbank is stored as mel_filters.npz alongside this file.
    Generated via: np.savez_compressed("mel_filters.npz", mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80))
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Compute the log-Mel spectrogram of a waveform.

    Parameters
    ----------
    audio : np.ndarray or torch.Tensor, shape = (*, T)
        The audio waveform(s) at 16 kHz.
    n_mels : int
        Number of Mel-frequency filters (only 80 supported).
    padding : int
        Number of zero samples to pad to the right.
    device : optional
        Device to move the audio tensor to before STFT.

    Returns
    -------
    torch.Tensor, shape = (*, 80, n_frames)
    """
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels).to(dtype=magnitudes.dtype)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


# ---------------------------------------------------------------------------
# Whisper building blocks (dtype-casting variants for mixed precision)
# ---------------------------------------------------------------------------

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding."""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# ---------------------------------------------------------------------------
# Attention and transformer blocks
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)
        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()
            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


# ---------------------------------------------------------------------------
# WhisperEncoder
# ---------------------------------------------------------------------------

class WhisperEncoder(nn.Module):
    """Whisper audio encoder: Conv1d stem + Transformer blocks + projection.

    Takes raw waveform as input (via dict with "waveform" key), computes
    log-mel spectrogram on-the-fly, and returns projected sequence embeddings.
    """

    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        output_dim: int = 512,
        avg_pool: bool = True,
        add_audio_bos_eos_token: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

        if avg_pool:
            self.avg_pooler = nn.AvgPool1d(2, stride=2)
        else:
            self.avg_pooler = None
        self.proj = nn.Linear(n_state, output_dim)
        if add_audio_bos_eos_token:
            self.audio_bos_eos_token = nn.Embedding(2, output_dim)
        else:
            self.audio_bos_eos_token = None
        self.output_dim = output_dim
        self.n_head = n_head

    def forward(self, x, padding_mask=None, audio_lengths=None):
        """Forward pass.

        Parameters
        ----------
        x : dict
            Must contain "waveform" key with tensor of shape (B, T).
        padding_mask : optional Tensor
        audio_lengths : optional Tensor

        Returns
        -------
        dict with "embedding" (B, T', output_dim), "audio_bos", "audio_eos"
        """
        x = x["waveform"]
        x = log_mel_spectrogram(
            x,
            n_mels=self.conv1.in_channels,
            padding=0,
            device=self.conv1.weight.device,
        )
        x = x.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        if audio_lengths is not None:
            input_mel_len = audio_lengths[:, 0] * 2
            max_mel_len_in_batch = input_mel_len.max()
            x = x[:, :, :max_mel_len_in_batch]

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, L, D)

        src_len = x.size(1)
        pos_emb = self.positional_embedding[:src_len]
        assert x.shape[1:] == pos_emb.shape, (
            f"incorrect audio shape: {x.shape[1:]} vs positional_embedding {pos_emb.shape}"
        )
        x = (x + pos_emb).to(x.dtype)

        if padding_mask is not None:
            bsz = x.size(0)
            padding_mask = padding_mask.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
            batch_src_len = padding_mask.size(1)
            x = x[:, :batch_src_len, :]
            padding_mask = padding_mask.view(bsz, -1, batch_src_len)
            padding_mask_ = padding_mask.all(1)
            x[padding_mask_] = 0
            key_padding_mask = (
                padding_mask_.view(bsz, 1, 1, batch_src_len)
                .expand(-1, self.n_head, -1, -1)
                .reshape(bsz, self.n_head, 1, batch_src_len)
            )
            new_padding_mask = torch.zeros_like(key_padding_mask, dtype=x.dtype)
            padding_mask = new_padding_mask.masked_fill(key_padding_mask, float("-inf"))

        for block in self.blocks:
            x = block(x, mask=padding_mask)

        if self.avg_pooler:
            x = x.permute(0, 2, 1)
            x = self.avg_pooler(x)
            x = x.permute(0, 2, 1)

        x = self.ln_post(x)
        x = self.proj(x)

        if self.audio_bos_eos_token is not None:
            bos = self.audio_bos_eos_token.weight[0][None, :]
            eos = self.audio_bos_eos_token.weight[1][None, :]
        else:
            bos, eos = None, None

        return {
            "embedding": x,
            "audio_bos": bos,
            "audio_eos": eos,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_whisper_model(audio_cfg, output_dim, **kwargs) -> WhisperEncoder:
    """Create a WhisperEncoder based on audio_cfg.model_name.

    Supports: tiny, base, small, medium, large.
    If audio_cfg.pretrained is True, loads OpenAI Whisper weights.
    """
    _WHISPER_CONFIGS = {
        "tiny":   dict(n_layer=4,  width=384,  heads=6),
        "base":   dict(n_layer=6,  width=512,  heads=8),
        "small":  dict(n_layer=12, width=768,  heads=12),
        "medium": dict(n_layer=24, width=1024, heads=16),
        "large":  dict(n_layer=32, width=1280, heads=20),
    }

    name = audio_cfg.model_name
    if name not in _WHISPER_CONFIGS:
        raise ValueError(f"Unknown whisper model name: {name}. Available: {list(_WHISPER_CONFIGS.keys())}")

    cfg = _WHISPER_CONFIGS[name]
    model = WhisperEncoder(
        n_mels=N_MELS,
        n_ctx=get_T_after_cnn(N_FRAMES),
        n_state=cfg["width"],
        n_head=cfg["heads"],
        n_layer=cfg["n_layer"],
        output_dim=output_dim,
        avg_pool=True,
        add_audio_bos_eos_token=True,
        **kwargs,
    )

    if getattr(audio_cfg, 'pretrained', False):
        logging.info(f"Loading pretrained Whisper weights for model '{name}'")
        try:
            import whisper
            whisper_model = whisper.load_model(name)
            model_state_dict = model.state_dict()
            whisper_state_dict = whisper_model.encoder.state_dict()
            loaded = 0
            for k in model_state_dict.keys():
                if k in whisper_state_dict and model_state_dict[k].shape == whisper_state_dict[k].shape:
                    model_state_dict[k] = whisper_state_dict[k]
                    loaded += 1
            model.load_state_dict(model_state_dict)
            logging.info(f"Loaded {loaded}/{len(model_state_dict)} Whisper encoder weights")
        except ImportError:
            logging.warning(
                "openai-whisper package not installed. Cannot load pretrained Whisper weights. "
                "Install via: pip install openai-whisper"
            )
        except Exception as e:
            logging.warning(f"Failed to load pretrained Whisper weights: {e}")

    return model
