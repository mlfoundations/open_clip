import json
import pickle
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

import open_clip
import open_clip.clap_model
from open_clip.audio import CLIPAudioCfg
from open_clip.clap_model import CLAP
from open_clip.model import CLIPTextCfg
from open_clip.task import CLAPTask


class TinyAudioTower(nn.Module):
    def __init__(self, embed_dim=4):
        super().__init__()
        self.cfg = CLIPAudioCfg(clip_samples=8, mel_bins=3, hop_size=2, enable_fusion=True)
        self.training_head = False
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, audio, apply_proj=True):
        x = audio["waveform"].float().mean(dim=1, keepdim=True)
        return self.proj(x) if apply_proj else x.expand(-1, self.proj.out_features)

    def set_grad_checkpointing(self, enable=True, impl="inline"):
        self.grad_checkpointing = enable

    def no_weight_decay(self):
        return set()


class TinyCLAPLike(nn.Module):
    output_dict = True

    def __init__(self, embed_dim=4, context_length=5):
        super().__init__()
        self.audio = TinyAudioTower(embed_dim)
        self.context_length = context_length
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.text_embedding = nn.Embedding(16, embed_dim)

    def forward(self, audio=None, text=None):
        audio_features = self.audio(audio) if audio is not None else None
        text_features = self.text_embedding(text).mean(dim=1) if text is not None else None
        if audio_features is not None:
            audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
        if text_features is not None:
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
        return {
            "audio_features": audio_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp(),
        }


def _batch(batch_size=2, context_length=5):
    return {
        "audio": {
            "waveform": torch.randn(batch_size, 8),
            "longer": torch.zeros(batch_size, dtype=torch.bool),
        },
        "text": torch.randint(0, 16, (batch_size, context_length)),
    }


def _patch_audio_tower(monkeypatch):
    monkeypatch.setattr(
        open_clip.clap_model,
        "_build_audio_tower",
        lambda embed_dim, audio_cfg: TinyAudioTower(embed_dim),
    )


def test_clap_task_training_forward_returns_loss():
    task = CLAPTask(TinyCLAPLike())
    losses = task(_batch())
    assert losses["loss"].isfinite()
    assert "contrastive_loss" in losses
    assert "logit_scale" in losses


def test_clap_task_eval_forward_keeps_audio_features_key():
    task = CLAPTask(TinyCLAPLike())
    task.eval()
    out = task(_batch())
    assert "audio_features" in out
    assert "image_features" not in out


def test_clap_task_dummy_batch_and_batch_size():
    task = CLAPTask(TinyCLAPLike())
    dummy = task.create_dummy_batch(batch_size=3, device=torch.device("cpu"), dtype=torch.float32)
    assert dummy["audio"]["waveform"].shape == (3, 8)
    assert dummy["audio"]["longer"].shape == (3,)
    assert dummy["audio"]["mel_fusion"].shape == (3, 4, 5, 3)
    assert dummy["text"].shape == (3, 5)
    assert task.batch_size(dummy) == 3
    assert task.primary_key == "audio"


def test_clap_task_accum_loss_maps_audio_features_to_clip_loss():
    task = CLAPTask(TinyCLAPLike())
    inputs = {
        "audio_features": torch.nn.functional.normalize(torch.randn(4, 4), dim=-1),
        "text_features": torch.nn.functional.normalize(torch.randn(4, 4), dim=-1),
    }
    losses = task.compute_accum_loss(inputs, {"logit_scale": torch.tensor(1.0)}, [])
    assert losses["contrastive_loss"].isfinite()


def test_create_task_dispatches_clap(monkeypatch):
    _patch_audio_tower(monkeypatch)
    model = CLAP(
        embed_dim=4,
        audio_cfg=CLIPAudioCfg(),
        text_cfg=CLIPTextCfg(context_length=5, vocab_size=16, width=4, heads=1, layers=1),
        output_dict=True,
    )
    args = SimpleNamespace(
        rank=0,
        world_size=1,
        distill=False,
        model="CLAP-test",
        siglip=False,
        local_loss=False,
        gather_with_grad=False,
    )
    task = open_clip.create_task(args, model)
    assert isinstance(task, CLAPTask)
    assert task.ddp_extra_kwargs() == {"find_unused_parameters": True}


def test_factory_dispatches_audio_config_to_clap(tmp_path, monkeypatch):
    _patch_audio_tower(monkeypatch)
    import open_clip.factory as factory

    monkeypatch.setattr(factory, "_MODEL_CONFIG_PATHS", list(factory._MODEL_CONFIG_PATHS))
    monkeypatch.setattr(factory, "_MODEL_CONFIGS", dict(factory._MODEL_CONFIGS))
    config_path = tmp_path / "CLAP-test.json"
    config_path.write_text(json.dumps({
        "embed_dim": 4,
        "audio_cfg": {
            "model_type": "HTSAT",
            "model_name": "tiny",
            "clip_samples": 8,
            "mel_bins": 3,
            "hop_size": 2,
            "enable_fusion": True,
        },
        "text_cfg": {
            "context_length": 5,
            "vocab_size": 16,
            "width": 4,
            "heads": 1,
            "layers": 1,
        },
    }))
    open_clip.add_model_config(config_path)

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "CLAP-test",
        pretrained=None,
        load_weights=False,
        pretrained_text=False,
        output_dict=True,
    )

    assert isinstance(model, CLAP)
    assert callable(preprocess_train)
    assert callable(preprocess_val)


def test_evaluate_uses_task_primary_key_for_audio():
    from open_clip_train.train import evaluate

    task = CLAPTask(TinyCLAPLike())
    dataloader = mock.MagicMock()
    dataloader.__iter__ = mock.MagicMock(return_value=iter([_batch(batch_size=2)]))
    dataloader.num_samples = 2
    data = {"val": mock.MagicMock(dataloader=dataloader)}
    args = SimpleNamespace(
        device="cpu",
        precision="fp32",
        rank=0,
        local_rank=0,
        world_size=1,
        distributed=False,
        fsdp=False,
        val_frequency=1,
        epochs=1,
        save_logs=False,
        wandb=False,
    )

    metrics = evaluate(task, data, epoch=1, args=args)

    assert "audio_val_loss" in metrics
    assert "clip_val_loss" not in metrics
    assert "audio_to_text_R@1" in metrics
    assert "text_to_audio_R@1" in metrics


def test_audio_checkpoint_retries_only_weights_only_payload_errors(monkeypatch):
    from open_clip.audio.tower import _load_audio_checkpoint

    calls = []

    def fake_load(path, map_location=None, weights_only=True):
        calls.append(weights_only)
        if weights_only:
            raise pickle.UnpicklingError("Weights only load failed. Unsupported global.")
        return {"state_dict": {}}

    monkeypatch.setattr(torch, "load", fake_load)
    assert _load_audio_checkpoint("checkpoint.pt") == {"state_dict": {}}
    assert calls == [True, False]


def test_audio_checkpoint_does_not_retry_unrelated_unpickle_errors(monkeypatch):
    from open_clip.audio.tower import _load_audio_checkpoint

    calls = []

    def fake_load(path, map_location=None, weights_only=True):
        calls.append(weights_only)
        raise pickle.UnpicklingError("corrupt checkpoint")

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(pickle.UnpicklingError):
        _load_audio_checkpoint("checkpoint.pt")
    assert calls == [True]
