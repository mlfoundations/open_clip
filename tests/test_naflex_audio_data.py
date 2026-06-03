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


def test_audio_length_bucketer_caption_only_when_block():
    """pack_prefix OFF (block layout): audio pads to the seq-len bucket, so the key is caption length only."""
    from open_clip_train.naflex_data import AudioLengthBucketer

    sr = 16000
    durs = [3, 1, 2, 1]   # seconds (waveform length / sr) -- must NOT influence the key here
    caps = [5, 9, 2, 3]   # caption token counts
    samples = [
        {"audio": (torch.randn(1, sr * d), sr), "text": torch.zeros(c, dtype=torch.long)}
        for d, c in zip(durs, caps)
    ]
    bucketer = AudioLengthBucketer(pool=10, chunk=10, seed=0, epoch=0)  # packed defaults off
    assert bucketer._length(samples[0]) == 5  # caption length; duration ignored in block mode

    out = list(bucketer(iter(samples)))
    assert len(out) == len(samples)  # permutation, nothing dropped/duplicated
    assert [s["text"].shape[0] for s in out] == sorted(caps)  # sorted by caption length


def test_audio_length_bucketer_token_sum_when_packed():
    """pack_prefix ON (packed layout): bucket by the compacted row length = est. audio tokens + caption."""
    from open_clip_train.naflex_data import AudioLengthBucketer

    sr = 16000
    durs = [3, 1, 2, 1]
    caps = [5, 9, 2, 4]
    samples = [
        {"audio": (torch.randn(1, sr * d), sr), "text": torch.zeros(c, dtype=torch.long)}
        for d, c in zip(durs, caps)
    ]
    # freq_tokens=1, patch_time=1, hop_size=sr -> audio_tokens = (samples // sr) + 1 = duration_sec + 1
    bucketer = AudioLengthBucketer(
        pool=10, chunk=10, seed=0, epoch=0,
        packed=True, freq_tokens=1, patch_time=1, hop_size=sr, sample_rate=sr,
    )
    assert bucketer._length(samples[0]) == (3 + 1) + 5  # audio_tokens (dur+1) + caption

    sums = [(d + 1) + c for d, c in zip(durs, caps)]  # [9, 11, 5, 6]
    out = list(bucketer(iter(samples)))
    assert len(out) == len(samples)
    assert [bucketer._length(s) for s in out] == sorted(sums)  # sorted by the token sum


def test_audio_length_bucketer_clamps_long_clips_when_packed():
    """The estimate is clamped to max_audio_tokens (the largest bucket): clips that both overflow it are
    indistinguishable after the transform truncates them, so they sort by caption -- no outlier over-weighting."""
    from open_clip_train.naflex_data import AudioLengthBucketer

    sr = 16000
    bucketer = AudioLengthBucketer(
        pool=10, chunk=10, seed=0, epoch=0,
        packed=True, freq_tokens=1, patch_time=1, hop_size=sr, sample_rate=sr, max_audio_tokens=5,
    )
    long_short_cap = {"audio": (torch.randn(1, sr * 100), sr), "text": torch.zeros(2, dtype=torch.long)}
    long_long_cap = {"audio": (torch.randn(1, sr * 50), sr), "text": torch.zeros(4, dtype=torch.long)}
    # both audio estimates (101, 51) clamp to 5 -> keys are 5+2 and 5+4, NOT 103 vs 55
    assert bucketer._length(long_short_cap) == 5 + 2
    assert bucketer._length(long_long_cap) == 5 + 4


def test_audio_transform_factory_carries_pack_prefix():
    """Concern 1: pack_prefix rides on the transform factory (resolved from the model in _build_preprocess), so
    the data path reads it off the transform it already gets -- robust to local-dir:/hf-hub:/override configs."""
    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=64, patch_time=4)
    assert AudioNaFlexTransformFactory(cfg).pack_prefix is False                  # default off
    assert AudioNaFlexTransformFactory(cfg, pack_prefix=True).pack_prefix is True

    from open_clip.factory import _build_preprocess
    model = open_clip.create_model(CONFIG_1D)
    model.pack_prefix = True  # a packed GenLAP, regardless of how its config was sourced
    train, val = _build_preprocess(model)
    assert train.pack_prefix is True and val.pack_prefix is True


def test_audio_length_bucketer_estimate_matches_actual_patch_count():
    """Packed-mode estimate must equal the patch count the transform actually emits (ceil), for short AND
    remainder clips -- otherwise packed bucketing mis-sizes T = max(k+m)."""
    pytest.importorskip("torchaudio")
    from open_clip_train.naflex_data import AudioLengthBucketer

    # patch_time=2 (not 4) so a dropped window_size would actually diverge on the sub-window clip below; and pass
    # window_size like production does, so this mirrors the real bucketer wiring (not an accidental match).
    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=32, patch_time=2)  # F = freq_tokens = 2 (2-D grid)
    patchify = AudioNaFlexPatchify(cfg)
    bucketer = AudioLengthBucketer(
        pool=10, chunk=10, seed=0, epoch=0, packed=True, freq_tokens=cfg.freq_tokens, patch_time=cfg.patch_time,
        hop_size=cfg.hop_size, window_size=cfg.window_size, sample_rate=cfg.sample_rate,
    )
    sr = cfg.sample_rate
    for n_samples in (200, sr // 10, sr, sr * 3 + 137):  # sub-patch, short, 1s, 3s + remainder
        wav = torch.randn(1, n_samples)
        actual = patchify((wav, sr))["patches"].shape[0]
        est = bucketer._audio_tokens({"audio": (wav, sr)})
        assert est == actual, f"n_samples={n_samples}: estimate {est} != actual {actual}"


def test_audio_naflex_seq_len_below_freq_tokens_raises():
    """A seq-len bucket smaller than freq_tokens fails fast (else collate truncation drops freq rows)."""
    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=16, patch_time=16)  # freq_tokens = 4
    factory = AudioNaFlexTransformFactory(cfg)
    with pytest.raises(ValueError, match="freq_tokens"):
        factory(max_seq_len=3, patch_size=None)  # 3 < 4 -> raises in AudioNaFlexPatchify.__init__


def test_clip_audio_cfg_propagates_patch_pad_mode():
    """NaFlexClap selects patch_pad_mode via CLIPAudioCfg; from_clip_audio_cfg must carry it to the transform cfg."""
    from open_clip.audio.config import CLIPAudioCfg

    assert AudioNaFlexCfg.from_clip_audio_cfg(CLIPAudioCfg()).patch_pad_mode == "floor"  # default
    cfg = AudioNaFlexCfg.from_clip_audio_cfg(CLIPAudioCfg(patch_pad_mode="silence"))
    assert cfg.patch_pad_mode == "silence"
    assert AudioNaFlexTransformFactory(cfg).cfg.patch_pad_mode == "silence"  # reaches the actual patchify transform


def test_synthetic_audio_rejects_naflex_transform():
    """synthetic-audio is NOT supported for NaFlex audio models (GenLAP/NaFlexClap) -> fail loudly, not with a
    confusing 'AudioNaFlexPatchify object is not subscriptable' collate error."""
    from open_clip_train.audio_data import get_synthetic_audio_dataset

    factory = AudioNaFlexTransformFactory(AudioNaFlexCfg(n_mels=64, patch_freq=64, patch_time=4))
    args = SimpleNamespace(model=CONFIG_1D, train_num_samples=4, batch_size=2, workers=0, distributed=False)
    with pytest.raises(NotImplementedError, match="NaFlex"):
        get_synthetic_audio_dataset(args, factory, is_train=True)


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
