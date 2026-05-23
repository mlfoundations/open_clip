from types import SimpleNamespace

import torch

from open_clip_train.audio_data import SyntheticAudioDataset, _audio_collate


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
