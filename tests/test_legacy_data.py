"""Smoke tests for the frozen legacy (decode-first) wds pipeline assembly in legacy_data.py.

These exist so the legacy fallback does not silently rot; behavior must match the default builders for the
paths legacy supports (fixed + variable text, json captions; no bucketing, no NaFlex).
"""
import io
import os
import tarfile
import types

import pytest
import torch
import util_test

import open_clip_train.audio_data as audio_data
from open_clip_train.legacy_data import (
    get_data_legacy,
    get_wds_audio_dataset_legacy,
    get_wds_dataset_legacy,
)
from test_audio_wds import SR, _base_args, _build_audio_tar  # noqa: F401  (audio tar + args helpers)
from test_wds import build_inputs, build_params


def test_legacy_wds_image_fixed_text():
    input_dir = build_inputs('legacy_fixed')
    input_shards = os.path.join(input_dir, 'test_data_000.tar')
    args, preprocess_img, _ = build_params(input_shards)
    args.train_num_samples = 4
    args.workers = 0
    args.batch_size = 2

    info = get_wds_dataset_legacy(args, preprocess_img, is_train=True, tokenizer=util_test.VariableTokenizer())
    batch = next(iter(info.dataloader))
    assert batch["image"].shape[0] == 2
    # Fixed-length contract: every caption padded to the tokenizer context length.
    assert batch["text"].shape == (2, util_test.VariableTokenizer.context_length)


def test_legacy_wds_image_variable_text():
    input_dir = build_inputs('legacy_variable')
    input_shards = os.path.join(input_dir, 'test_data_000.tar')
    args, preprocess_img, _ = build_params(input_shards)
    args.train_num_samples = 4
    args.workers = 0
    args.batch_size = 2
    args.variable_text = True

    info = get_wds_dataset_legacy(args, preprocess_img, is_train=True, tokenizer=util_test.VariableTokenizer())
    batch = next(iter(info.dataloader))
    valid = batch["text"] != util_test.VariableTokenizer.pad_token_id
    assert batch["text"].shape[1] == int(valid.sum(dim=1).max())  # padded to batch max
    assert torch.equal(batch["text_valid"], valid)


def test_legacy_wds_audio(monkeypatch):
    # Stub the extension-keyed decoder (the legacy assembly decodes first via wds.decode) so the test does
    # not depend on a torchaudio file backend; mirrors the fixture in test_audio_wds.
    monkeypatch.setattr(
        audio_data, "_decode_audio",
        lambda key, data: (torch.randn(1, SR), SR) if key.rsplit(".", 1)[-1] in ("wav", "flac") else None,
    )
    from open_clip.audio.config import CLIPAudioCfg
    from open_clip.audio.transform import audio_transform_v2

    shard = _build_audio_tar("legacy_audio")
    args = _base_args(shard)
    preprocess_audio = audio_transform_v2(CLIPAudioCfg(clip_samples=SR, sample_rate=SR), is_train=False)
    tokenizer = lambda s: torch.zeros(1, 16, dtype=torch.long)

    info = get_wds_audio_dataset_legacy(args, preprocess_audio, is_train=True, tokenizer=tokenizer)
    batch = next(iter(info.dataloader))
    assert set(batch) == {"audio", "text"}
    assert batch["audio"]["waveform"].shape == (args.batch_size, SR)
    assert batch["text"].shape == (args.batch_size, 16)


def test_legacy_get_data_rejects_naflex_and_bucketing():
    args = types.SimpleNamespace(use_naflex=True, length_bucketing=False)
    with pytest.raises(ValueError, match="NaFlex"):
        get_data_legacy(args, (None, None))
    args = types.SimpleNamespace(use_naflex=False, length_bucketing=True)
    with pytest.raises(ValueError, match="length-bucketing"):
        get_data_legacy(args, (None, None))
