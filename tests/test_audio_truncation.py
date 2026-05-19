import importlib.util

import pytest
import torch

from open_clip.audio import AudioAugmentationCfg, CLIPAudioCfg, audio_transform_v2
from open_clip.audio.transform import get_audio_frame_count, make_audio_preprocess


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchaudio") is None,
    reason="audio preprocessing tests require torchaudio",
)


def _cfg():
    return CLIPAudioCfg(
        sample_rate=8,
        clip_samples=8,
        window_size=4,
        hop_size=2,
        mel_bins=3,
        fmin=0,
        fmax=4,
    )


def _waveform(values):
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def test_audio_preprocess_truncates_deterministically():
    preprocess = make_audio_preprocess(_cfg(), data_truncating="trunc")
    out = preprocess((_waveform(range(12)), 8))

    assert out["longer"] is True
    assert torch.equal(out["waveform"], torch.arange(8, dtype=torch.float32))
    assert "mel_fusion" not in out


def test_audio_preprocess_random_truncation_uses_sampled_offset(monkeypatch):
    monkeypatch.setattr("open_clip.audio.transform.random.randint", lambda start, end: 2)
    preprocess = make_audio_preprocess(_cfg(), data_truncating="rand_trunc")
    out = preprocess((_waveform(range(12)), 8))

    assert out["longer"] is True
    assert torch.equal(out["waveform"], torch.arange(2, 10, dtype=torch.float32))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("pad", [1, 2, 3, 0, 0, 0, 0, 0]),
        ("repeat", [1, 2, 3, 1, 2, 3, 1, 2]),
        ("repeatpad", [1, 2, 3, 1, 2, 3, 0, 0]),
    ],
)
def test_audio_preprocess_fill_modes(mode, expected):
    preprocess = make_audio_preprocess(_cfg(), data_filling=mode)
    out = preprocess((_waveform([1, 2, 3]), 8))

    assert out["longer"] is False
    assert torch.equal(out["waveform"], torch.tensor(expected, dtype=torch.float32))


def test_audio_preprocess_fusion_shape_for_long_audio():
    cfg = _cfg()
    preprocess = make_audio_preprocess(cfg, data_truncating="fusion")
    out = preprocess((_waveform(range(16)), 8))

    assert out["longer"] is True
    assert out["waveform"].shape == (8,)
    assert out["mel_fusion"].shape == (4, get_audio_frame_count(cfg), cfg.mel_bins)


def test_audio_transform_eval_uses_fixed_truncation_policy():
    cfg = _cfg()
    transform = audio_transform_v2(
        cfg,
        is_train=False,
        audio_aug_cfg=AudioAugmentationCfg(data_truncating="rand_trunc", data_filling="repeat"),
    )
    out = transform((_waveform(range(12)), 8))

    assert torch.equal(out["waveform"], torch.arange(8, dtype=torch.float32))
