"""Tests for the GenLAP audio NaFlex data pipeline (transform factory + shared NaFlex batching + wiring)."""
from types import SimpleNamespace

import pytest
import torch

import open_clip
from open_clip import get_tokenizer
from open_clip.audio.naflex_audio import AudioNaFlexCfg, AudioNaFlexPatchify, AudioNaFlexTransformFactory

CONFIG_1D = "naflexgenlap_test_1d"
CONFIG_2D = "naflexgenlap_test_2d"


def _tokens(tok, text):
    return tok(text, pad=False)[0]


def test_audio_naflex_batch_via_scheduler_feeds_model():
    """End-to-end: AudioNaFlexTransformFactory + the modality-agnostic scheduler (image_key='audio') produce
    a `{audio, text, text_valid}` batch the GenLAP model consumes."""
    pytest.importorskip("torchaudio")
    from open_clip_train.naflex_data import NaFlexBatchScheduler

    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=64, patch_time=4)  # 1-D full-height strips
    tok = get_tokenizer(CONFIG_1D)
    sched = NaFlexBatchScheduler(
        train_num_samples=100,
        patch_size=(cfg.patch_time, cfg.patch_freq),  # cosmetic (audio transform returns the dict directly)
        seq_lens=(256,),
        max_tokens_per_batch=4096,
        transform_factory=AudioNaFlexTransformFactory(cfg),
        shuffle=False,
        image_key="audio",
        pad_id=tok.pad_token_id,
        per_row_text_tokens=64,
    )
    sr = cfg.sample_rate
    samples = [
        {"audio": (torch.randn(1, sr * 2), sr), "text": _tokens(tok, "a dog barking in the distance")},
        {"audio": (torch.randn(1, sr * 1), sr), "text": _tokens(tok, "short clip")},
    ]
    batch = sched.collate_batch(samples, seq_len=256, patch_idx=0)

    audio = batch["audio"]
    assert set(("patches", "patch_coord", "patch_valid")).issubset(audio)
    assert audio["patches"].shape[0] == 2 and audio["patches"].shape[2] == cfg.patch_dim == 64 * 4
    assert "text" in batch and "text_valid" in batch
    assert audio["patch_valid"][0].sum() > audio["patch_valid"][1].sum()  # 2 s clip > 1 s clip

    model = open_clip.create_model(CONFIG_1D).eval()
    loss = model(audio=audio, text=batch["text"], text_valid=batch["text_valid"], compute_loss=True)["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_audio_naflex_2d_token_cap_preserves_freq_rows():
    """2-D cap must crop whole time columns (max_time = max_tokens // freq_tokens), keeping every freq row."""
    pytest.importorskip("torchaudio")
    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=32, patch_time=4)  # F = 2
    patchify = AudioNaFlexPatchify(cfg, max_audio_tokens=64)      # max_time = 64 // 2 = 32 columns
    sr = cfg.sample_rate
    out = patchify((torch.randn(1, sr * 10), sr))  # long clip would overflow without the cap
    n = out["patches"].shape[0]
    assert n <= 64
    assert n % cfg.freq_tokens == 0                      # whole time columns only
    assert out["patch_coord"][:, 0].max().item() == 1    # BOTH freq rows present (no dropped row)


def test_create_model_and_transforms_returns_audio_naflex_factory():
    """GenLAP must not crash create_model_and_transforms (no .audio/.visual) and gets the audio factory."""
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(CONFIG_1D)
    assert type(model).__name__ == "NaFlexGenLap"
    assert isinstance(preprocess_train, AudioNaFlexTransformFactory)
    assert isinstance(preprocess_val, AudioNaFlexTransformFactory)


def test_params_genlap_enables_naflex():
    from open_clip_train.params import parse_args

    args = parse_args(["--model", CONFIG_1D, "--train-num-samples", "100"])
    assert args.genlap is True
    assert args.use_naflex is True
    assert args.force_naflex_vision is False
