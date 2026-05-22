import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Union

import numpy as np
import torch

from .config import CLIPAudioCfg


@dataclass
class AudioAugmentationCfg:
    data_trunc: str = "rand_trunc"
    data_fill: str = "repeatpad"
    enable_fusion: bool = False
    int16_normalize: bool = False


def _as_audio_cfg_dict(audio_cfg: Union[CLIPAudioCfg, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(audio_cfg, CLIPAudioCfg):
        return asdict(audio_cfg)
    return dict(audio_cfg)


def get_audio_frame_count(audio_cfg: Union[CLIPAudioCfg, Dict[str, Any]]) -> int:
    cfg = _as_audio_cfg_dict(audio_cfg)
    return cfg.get("clip_samples", 480000) // cfg.get("hop_size", 480) + 1


def int16_to_float32_torch(x: torch.Tensor) -> torch.Tensor:
    return (x / 32767.0).type(torch.float32)


def float32_to_int16_torch(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=-1.0, max=1.0)
    return (x * 32767.0).type(torch.int16)


def _get_mel(audio_data: torch.Tensor, audio_cfg: Dict[str, Any]) -> torch.Tensor:
    import torchaudio

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg.get("sample_rate", 48000),
        n_fft=audio_cfg.get("window_size", 1024),
        win_length=audio_cfg.get("window_size", 1024),
        hop_length=audio_cfg.get("hop_size", 480),
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        n_mels=audio_cfg.get("mel_bins", 64),
        f_min=audio_cfg.get("fmin", 50),
        f_max=audio_cfg.get("fmax", 14000),
    )
    mel = mel_tf(audio_data)
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T


class AudioPreprocess:
    """Picklable audio preprocessing callable for DataLoader spawn/forkserver workers."""

    def __init__(
            self,
            audio_cfg: Union[CLIPAudioCfg, Dict[str, Any]],
            data_fill: str = "repeatpad",
            data_trunc: str = "rand_trunc",
            int16_normalize: bool = False,
    ):
        self.cfg = _as_audio_cfg_dict(audio_cfg)
        self.data_fill = data_fill
        self.data_trunc = data_trunc
        self.int16_normalize = int16_normalize
        self.target_sr = self.cfg.get("sample_rate", 48000)
        self.clip_samples = self.cfg.get("clip_samples", 480000)
        self.hop_size = self.cfg.get("hop_size", 480)

    def _fill_waveform(self, waveform):
        if len(waveform) >= self.clip_samples:
            return waveform[:self.clip_samples]
        if self.data_fill == "repeat":
            repeats = int(np.ceil(self.clip_samples / len(waveform)))
            waveform = waveform.repeat(repeats)[:self.clip_samples]
        elif self.data_fill == "repeatpad":
            repeats = self.clip_samples // len(waveform)
            waveform = waveform.repeat(repeats)
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_samples - len(waveform)))
        elif self.data_fill == "pad":
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_samples - len(waveform)))
        else:
            raise ValueError(f"Unsupported audio fill mode: {self.data_fill}")
        return waveform

    def __call__(self, audio_data):
        import torchaudio

        waveform, sr = audio_data
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        waveform = waveform.squeeze(0)
        if self.int16_normalize:
            waveform = int16_to_float32_torch(float32_to_int16_torch(waveform))

        result = {}
        if len(waveform) > self.clip_samples:
            if self.data_trunc == "fusion":
                mel = _get_mel(waveform, self.cfg)
                chunk_frames = self.clip_samples // self.hop_size + 1
                total_frames = mel.shape[0]
                if chunk_frames >= total_frames:
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    longer = False
                else:
                    ranges = [
                        (0, chunk_frames),
                        ((total_frames - chunk_frames) // 2, (total_frames - chunk_frames) // 2 + chunk_frames),
                        (total_frames - chunk_frames, total_frames),
                    ]
                    local_mels = [mel[start:end] for start, end in ranges]
                    if any(m.shape[0] < chunk_frames for m in local_mels):
                        local_mels = [
                            torch.nn.functional.pad(m, (0, 0, 0, chunk_frames - m.shape[0])) for m in local_mels
                        ]
                    global_mel = mel
                    if global_mel.shape[0] < chunk_frames:
                        global_mel = torch.nn.functional.pad(global_mel, (0, 0, 0, chunk_frames - global_mel.shape[0]))
                    elif global_mel.shape[0] > chunk_frames:
                        overflow = global_mel.shape[0] - chunk_frames
                        idx = random.randint(0, overflow)
                        global_mel = global_mel[idx:idx + chunk_frames]
                    mel_fusion = torch.stack([global_mel] + local_mels, dim=0)
                    longer = True
                result["mel_fusion"] = mel_fusion
                waveform = waveform[:self.clip_samples]
            elif self.data_trunc == "rand_trunc":
                overflow = len(waveform) - self.clip_samples
                idx = random.randint(0, overflow)
                waveform = waveform[idx:idx + self.clip_samples]
                longer = True
            elif self.data_trunc == "trunc":
                waveform = waveform[:self.clip_samples]
                longer = True
            else:
                raise ValueError(f"Unsupported audio truncation mode: {self.data_trunc}")
        else:
            waveform = self._fill_waveform(waveform)
            longer = False
            if self.data_trunc == "fusion":
                mel = _get_mel(waveform, self.cfg)
                result["mel_fusion"] = torch.stack([mel, mel, mel, mel], dim=0)

        result["waveform"] = waveform
        result["longer"] = longer
        return result


def make_audio_preprocess(
        audio_cfg: Union[CLIPAudioCfg, Dict[str, Any]],
        data_fill: str = "repeatpad",
        data_trunc: str = "rand_trunc",
        int16_normalize: bool = False,
):
    return AudioPreprocess(
        audio_cfg,
        data_fill=data_fill,
        data_trunc=data_trunc,
        int16_normalize=int16_normalize,
    )


def audio_transform_v2(
        audio_cfg: Union[CLIPAudioCfg, Dict[str, Any]],
        is_train: bool = False,
        audio_aug_cfg: Union[AudioAugmentationCfg, Dict[str, Any], None] = None,
):
    cfg = _as_audio_cfg_dict(audio_cfg)
    if isinstance(audio_aug_cfg, dict):
        audio_aug_cfg = AudioAugmentationCfg(**audio_aug_cfg)
    elif audio_aug_cfg is None:
        audio_aug_cfg = AudioAugmentationCfg()

    enable_fusion = bool(audio_aug_cfg.enable_fusion or cfg.get("enable_fusion", False))
    data_trunc = audio_aug_cfg.data_trunc if is_train else "trunc"
    if enable_fusion:
        data_trunc = "fusion"

    return make_audio_preprocess(
        cfg,
        # Eval and train now share the fill policy; eval previously hardcoded zero-padding.
        data_fill=audio_aug_cfg.data_fill,
        data_trunc=data_trunc,
        int16_normalize=audio_aug_cfg.int16_normalize,
    )
