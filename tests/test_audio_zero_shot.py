from types import SimpleNamespace

import torch
from torch import nn

from open_clip_train import audio_zero_shot


class _FeatureDataset:
    features = {}

    def __init__(self):
        self.samples = [
            {"audio": {"array": [1.0, 0.0], "sampling_rate": 2}, "target": 0, "category": "dog_bark"},
            {"audio": {"array": [0.0, 1.0], "sampling_rate": 2}, "target": 1, "category": "vacuum_cleaner"},
        ]

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _AudioDecoderLike:
    def __init__(self, array, sampling_rate):
        self.array = array
        self.sampling_rate = sampling_rate

    def __getitem__(self, key):
        if key == "array":
            return self.array
        if key == "sampling_rate":
            return self.sampling_rate
        raise KeyError(key)


class _TinyAudioTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))

    def prepare_batch(self, batch, device, input_dtype=None):
        return {
            "audio": {
                "waveform": batch["audio"]["waveform"].to(device=device, dtype=input_dtype),
                "longer": batch["audio"]["longer"].to(device=device),
            }
        }

    def create_dummy_batch(self, batch_size=1, device=None, dtype=None):
        return {
            "audio": {
                "waveform": torch.zeros(batch_size, 2, device=device, dtype=dtype),
                "longer": torch.zeros(batch_size, dtype=torch.bool, device=device),
            }
        }

    def forward(self, audio=None, text=None):
        if audio is not None:
            features = torch.nn.functional.normalize(audio["waveform"].float(), dim=-1)
            return {"audio_features": features, "logit_scale": self.logit_scale.exp()}
        return {"text_features": torch.empty(0)}


def test_hf_audio_dataset_wrapper_and_classnames():
    dataset = _FeatureDataset()
    classnames = audio_zero_shot._get_classnames(dataset, target_key="target", class_key="category")
    wrapped = audio_zero_shot.HFAudioClassificationDataset(
        dataset,
        transform=lambda audio: {"waveform": audio[0].squeeze(0), "longer": False},
    )
    batch = audio_zero_shot._collate_audio_zero_shot([wrapped[0], wrapped[1]])

    assert classnames == ["dog bark", "vacuum cleaner"]
    assert batch["audio"]["waveform"].shape == (2, 2)
    assert torch.equal(batch["target"], torch.tensor([0, 1]))


def test_hf_audio_dataset_wrapper_accepts_decoder_like_audio():
    dataset = [
        {"audio": _AudioDecoderLike([1.0, 0.0], 2), "target": 0},
    ]
    wrapped = audio_zero_shot.HFAudioClassificationDataset(
        dataset,
        transform=lambda audio: {"waveform": audio[0].squeeze(0), "longer": False},
    )

    sample = wrapped[0]

    assert sample["audio"]["waveform"].shape == (2,)
    assert sample["target"] == 0


def test_run_audio_zero_shot_classifier():
    task = _TinyAudioTask()
    dataloader = [
        {
            "audio": {
                "waveform": torch.eye(2),
                "longer": torch.zeros(2, dtype=torch.bool),
            },
            "target": torch.tensor([0, 1]),
        }
    ]
    args = SimpleNamespace(
        device="cpu",
        precision="fp32",
        fsdp=False,
        rank=0,
        batch_size=2,
    )

    top1, top5 = audio_zero_shot.run_audio_zero_shot_classifier(
        task,
        classifier=torch.eye(2),
        dataloader=dataloader,
        args=args,
    )

    assert top1 == 1.0
    assert top5 == 1.0


def test_audio_zero_shot_template_validation():
    audio_zero_shot._validate_audio_templates(["a sound of {}"])
    try:
        audio_zero_shot._validate_audio_templates(["a sound"])
    except ValueError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("Expected template without placeholder to fail")


def test_run_audio_zero_shot_classifier_uses_bare_model_dummy_audio(monkeypatch):
    class _AudioCfg:
        clip_samples = 8
        enable_fusion = False

    class _Audio:
        cfg = _AudioCfg()

    class _Model:
        audio = _Audio()

        def __call__(self, audio=None):
            raise AssertionError("rank 1 should stop before forward when broadcast signal is zero")

    monkeypatch.setattr(audio_zero_shot.dist, "broadcast", lambda *_args, **_kwargs: None)
    args = SimpleNamespace(
        device="cpu",
        precision="fp32",
        fsdp=True,
        rank=1,
        batch_size=2,
    )

    top1, top5 = audio_zero_shot.run_audio_zero_shot_classifier(
        _Model(),
        classifier=torch.zeros(2, 5),
        dataloader=None,
        args=args,
        use_fsdp_eval=True,
    )

    assert (top1, top5) == (0.0, 0.0)
