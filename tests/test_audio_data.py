from types import SimpleNamespace

import torch

from open_clip.audio.naflex_audio import AudioNaFlexCfg
from open_clip_train.audio_data import SyntheticAudioDataset, _audio_collate, get_synthetic_audio_dataset
from open_clip_train.naflex_data import AudioTokenLength


def test_audio_token_length_accepts_raw_bytes():
    """Bucketed pipelines reorder before decode, so the length-fn sees raw bytes: it must never raise, and
    falls back to 0 (sort-key only) when the container header cannot be parsed."""
    fn = AudioTokenLength(freq_tokens=2, patch_time=4, hop_size=160, sample_rate=16000)
    decoded = fn({"audio": (torch.zeros(1, 16000), 16000)})
    assert decoded > 0
    raw = fn({"audio": b"\x00" * 64})  # not a valid container -> graceful 0
    assert isinstance(raw, int) and raw >= 0
    assert fn({"audio": None}) == 0


def test_audio_collate_returns_nested_dict():
    batch = [
        {
            "audio": {
                "waveform": torch.ones(8),
                "longer": False,
                "mel_fusion": torch.ones(4, 5, 3),
            },
            "text": torch.ones(5, dtype=torch.long),
        },
        {
            "audio": {
                "waveform": torch.zeros(8),
                "longer": True,
                "mel_fusion": torch.zeros(4, 5, 3),
            },
            "text": torch.zeros(5, dtype=torch.long),
        },
    ]
    out = _audio_collate(batch)
    assert set(out) == {"audio", "text"}
    assert out["audio"]["waveform"].shape == (2, 8)
    assert out["audio"]["longer"].dtype == torch.bool
    assert out["audio"]["mel_fusion"].shape == (2, 4, 5, 3)
    assert out["text"].shape == (2, 5)


def test_audio_collate_variable_text_pads_to_batch_max():
    batch = [
        {
            "audio": {"waveform": torch.ones(8), "longer": False},
            "text": torch.tensor([1, 2], dtype=torch.long),
        },
        {
            "audio": {"waveform": torch.zeros(8), "longer": True},
            "text": torch.tensor([1, 3, 4, 2], dtype=torch.long),
        },
    ]
    out = _audio_collate(batch, pad_id=99)

    assert set(out) == {"audio", "text", "text_valid"}
    assert out["text"].shape == (2, 4)
    assert out["text"][0].tolist() == [1, 2, 99, 99]
    assert out["text_valid"].tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]


def test_synthetic_audio_dataset_uses_transform_and_dict_contract():
    tokenizer = lambda texts: torch.ones(len(texts), 5, dtype=torch.long)

    def transform(audio_data):
        waveform, sr = audio_data
        assert sr == 16000
        return {"waveform": waveform.squeeze(0), "longer": False}

    dataset = SyntheticAudioDataset(
        audio_cfg={"clip_samples": 8, "sample_rate": 16000},
        transform=transform,
        dataset_size=2,
        tokenizer=tokenizer,
    )
    sample = dataset[0]
    assert set(sample) == {"audio", "text"}
    assert sample["audio"]["waveform"].shape == (8,)
    assert sample["text"].shape == (5,)


def test_synthetic_audio_dataset_uses_resolved_dataclass_cfg():
    tokenizer = lambda texts: torch.ones(len(texts), 5, dtype=torch.long)

    class Transform:
        cfg = AudioNaFlexCfg(sample_rate=16000)

        def __call__(self, audio_data):
            waveform, sr = audio_data
            assert sr == 16000
            return {"waveform": waveform.squeeze(0), "longer": False}

    args = SimpleNamespace(
        train_num_samples=2,
        batch_size=2,
        workers=0,
        distributed=False,
        audio_multiprocessing_context="forkserver",
    )
    info = get_synthetic_audio_dataset(args, Transform(), is_train=True, tokenizer=tokenizer)
    dataset = info.dataloader.dataset
    assert dataset.sample_rate == 16000
    assert dataset.clip_samples == 16000

    batch = next(iter(info.dataloader))
    assert batch["audio"]["waveform"].shape == (2, 16000)
