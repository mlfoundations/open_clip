"""End-to-end tests for the audio WebDataset path: a tiny on-the-fly tar through get_wds_audio_dataset.

Covers both branches the function dispatches: the standard CLAP path ({audio: {waveform, longer}, text}) and
the NaFlex audio path (GenLAP/NaFlexClap -> {audio: {patches, patch_coord, patch_valid}, text[, text_valid]}).

We stub the file decoder (``_decode_audio``) so the test exercises OUR pipeline wiring (filter/rename/collate/
NaFlex batching) rather than torchaudio's file-decode backend (which may be unavailable in CI). The mel
transform (``MelSpectrogram``, a pure torch op) is still real, so torchaudio is required.
"""
import io
import os
import tarfile

import pytest
import torch
import util_test

pytest.importorskip("torchaudio")  # mel transform (not file decode) needs torchaudio

import open_clip
import open_clip_train.audio_data as audio_data
from open_clip import create_model, get_tokenizer
from open_clip.audio.config import CLIPAudioCfg
from open_clip.audio.naflex_audio import AudioNaFlexTransformFactory
from open_clip.audio.transform import audio_transform_v2
from open_clip.naflex_config import NaFlexDataConfig
from open_clip_train.audio_data import get_wds_audio_dataset
from open_clip_train.params import parse_args

SR = 48000


def _fake_decode(key, data):
    """Stand in for torchaudio.load: (key, bytes) -> (waveform, sr) for audio members, None otherwise."""
    ext = key.rsplit(".", 1)[-1] if "." in key else key
    if ext in ("wav", "flac", "mp3", "ogg"):
        return torch.randn(1, SR), SR
    return None


@pytest.fixture(autouse=True)
def _audio_wds_env(monkeypatch):
    # Tiny shuffle buffers (public env knob) so the first batch doesn't wait on a 1000-sample buffer fill,
    # and a stub decoder so the test doesn't depend on a torchaudio file-decode backend.
    for name in ("OPENCLIP_WDS_SHARD_SHUFFLE_SIZE", "OPENCLIP_WDS_SHARD_SHUFFLE_INITIAL",
                 "OPENCLIP_WDS_SAMPLE_SHUFFLE_SIZE", "OPENCLIP_WDS_SAMPLE_SHUFFLE_INITIAL"):
        monkeypatch.setenv(name, "4" if name.endswith("SIZE") else "1")
    monkeypatch.setattr(audio_data, "_decode_audio", _fake_decode)


def _build_audio_tar(test_name, num_samples=12):
    """Tar of {idx}.wav + {idx}.txt samples (wav bytes are dummy — the decoder is stubbed)."""
    base_input_dir, _ = util_test.get_data_dirs()
    input_dir = os.path.join(base_input_dir, test_name)
    os.makedirs(input_dir, exist_ok=True)
    shard = os.path.join(input_dir, "audio_000.tar")
    with tarfile.open(shard, "w") as tar:
        for i in range(num_samples):
            for name, payload in ((f"{i}.wav", b"\x00" * 64), (f"{i}.txt", f"sound {i}".encode("utf-8"))):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return shard


def _base_args(shard, num_samples=12, batch_size=4):
    args = parse_args([])
    args.train_data = shard
    args.train_num_samples = num_samples
    args.dataset_resampled = True       # single shard -> resampled head (skips num_shards>=workers assert)
    args.seed = 0
    args.workers = 0                    # decode inline; avoids a fork-after-torchaudio-threads deadlock
    args.world_size = 1
    args.rank = 0                       # normally set by distributed init; NaFlexBatcher reads it
    args.batch_size = batch_size
    args.audio_ext = "wav"
    args.distributed = False
    return args


def test_get_wds_audio_dataset_clap_waveform():
    """Standard CLAP path: yields {audio: {waveform, longer}, text}."""
    shard = _build_audio_tar("audio_wds_clap")
    args = _base_args(shard)
    audio_cfg = CLIPAudioCfg(clip_samples=SR, sample_rate=SR)  # 1 s clips, no fusion
    preprocess_audio = audio_transform_v2(audio_cfg, is_train=False)
    tokenizer = lambda s: torch.zeros(1, 16, dtype=torch.long)  # fixed-length contrastive text stub

    info = get_wds_audio_dataset(args, preprocess_audio, is_train=True, tokenizer=tokenizer)
    batch = next(iter(info.dataloader))
    assert set(batch) == {"audio", "text"}
    assert set(batch["audio"]) >= {"waveform", "longer"}
    assert batch["audio"]["waveform"].shape == (args.batch_size, SR)
    assert batch["audio"]["longer"].shape == (args.batch_size,)
    assert batch["text"].shape == (args.batch_size, 16)


def test_audio_loader_forkserver_context_wiring():
    """Audio loaders attach a forkserver multiprocessing context when workers>0 (and only then)."""
    from types import SimpleNamespace

    from open_clip_train.audio_data import _audio_loader_kwargs
    from open_clip_train.params import parse_args

    assert _audio_loader_kwargs(SimpleNamespace(workers=0)) == {}  # 0 workers -> DataLoader rejects the kwarg
    assert _audio_loader_kwargs(SimpleNamespace(workers=2)) == {"multiprocessing_context": "forkserver"}  # default
    assert _audio_loader_kwargs(
        SimpleNamespace(workers=4, audio_multiprocessing_context="spawn")
    ) == {"multiprocessing_context": "spawn"}  # configurable
    assert parse_args([]).audio_multiprocessing_context == "forkserver"  # arg default


def test_get_wds_audio_dataset_naflex_genlap_feeds_model():
    """NaFlex generative path (args.genlap): yields a patch dict + variable text the GenLAP model consumes."""
    shard = _build_audio_tar("audio_wds_genlap")
    model = create_model("naflexgenlap_test_1d").eval()
    tokenizer = get_tokenizer("naflexgenlap_test_1d")  # tiktoken, has pad_token_id

    args = _base_args(shard)
    args.genlap = True  # triggers the NaFlex audio branch (generative: variable text + pad_id)
    args.naflex_length_bucketing = False
    preprocess_audio = AudioNaFlexTransformFactory(model.audio_cfg)  # genlap audio_cfg is already AudioNaFlexCfg
    naflex_data_config = NaFlexDataConfig.resolve(
        patch_sizes=[16], seq_lens=(256,), max_tokens_per_batch=2560, batch_divisor=1,
    )

    info = get_wds_audio_dataset(
        args, preprocess_audio, is_train=True, tokenizer=tokenizer, naflex_data_config=naflex_data_config,
    )
    batch = next(iter(info.dataloader))
    assert set(batch["audio"]) >= {"patches", "patch_coord", "patch_valid"}
    assert "text" in batch and "text_valid" in batch
    pd = model.audio_cfg.in_chans * model.audio_cfg.patch_freq * model.audio_cfg.patch_time
    assert batch["audio"]["patches"].shape[-1] == pd
    b = batch["audio"]["patches"].shape[0]
    assert batch["text"].shape[0] == b and batch["audio"]["patch_valid"].shape[0] == b

    loss = model(
        audio=batch["audio"], text=batch["text"], text_valid=batch["text_valid"], compute_loss=True,
    )["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)
