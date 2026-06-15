"""Unit tests for TrainingTask + ImageTextTask methods.

Covers (via concrete CLIPTask): prepare_batch, create_dummy_batch,
forward() calling-convention normalization, state_dict_for_inference,
clamp_logit_scale, and data_keys.
"""
import copy
import math
import types

import torch
import torch.nn as nn

from open_clip.naflex_config import NaFlexDataConfig
from open_clip.task import CLIPTask


# ---------------------------------------------------------------------------
# Tiny model / loss stubs (no real CLIP weights needed)
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_proj = nn.Linear(3, 3, bias=False)
        self.text_proj = nn.Linear(3, 3, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        self.visual = types.SimpleNamespace(image_size=8)
        self.context_length = 5

    def encode_image(self, image):
        return self.image_proj(image.float().mean(dim=(2, 3)))

    def encode_text(self, text):
        return self.text_proj(text[:, :3].float())

    def forward(self, image, text):
        return {
            "image_features": self.encode_image(image),
            "text_features": self.encode_text(text),
            "logit_scale": self.logit_scale,
        }


class DummyLoss(nn.Module):
    def forward(self, image_features, text_features, logit_scale, output_dict=False, **kw):
        assert output_dict
        return {
            "contrastive_loss": image_features.mean() * 0 + 1.0,
        }


def _batch(bs=2):
    return {
        "image": torch.randn(bs, 3, 4, 4),
        "text": torch.randint(0, 10, (bs, 5), dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# prepare_batch
# ---------------------------------------------------------------------------


def test_prepare_batch_float_dtype():
    """Float tensors get input_dtype, integer tensors stay unchanged."""
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    batch = _batch()
    out = task.prepare_batch(batch, device=torch.device("cpu"), input_dtype=torch.float16)
    assert out["image"].dtype == torch.float16
    assert out["text"].dtype == torch.long


def test_prepare_batch_nested_dict():
    """Recurses into nested dicts."""
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    batch = {
        "image": torch.randn(2, 3, 4, 4),
        "text": torch.randint(0, 10, (2, 5)),
        "nested": {
            "audio": torch.randn(2, 8),
            "ids": torch.tensor([1, 2], dtype=torch.int32),
        },
    }
    out = task.prepare_batch(batch, device=torch.device("cpu"), input_dtype=torch.float16)
    assert out["nested"]["audio"].dtype == torch.float16
    assert out["nested"]["ids"].dtype == torch.int32


def test_prepare_batch_passthrough_non_tensor():
    """Non-tensor values pass through unchanged."""
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    batch = {"image": torch.randn(2, 3, 4, 4), "meta": ["x", "y"]}
    out = task.prepare_batch(batch, device=torch.device("cpu"))
    assert out["meta"] == ["x", "y"]


def test_prepare_batch_none_dtype():
    """input_dtype=None preserves original float dtype."""
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    batch = {"image": torch.randn(2, 3, 4, 4, dtype=torch.float32)}
    out = task.prepare_batch(batch, device=torch.device("cpu"), input_dtype=None)
    assert out["image"].dtype == torch.float32


# ---------------------------------------------------------------------------
# create_dummy_batch
# ---------------------------------------------------------------------------


def test_create_dummy_batch_int_image_size():
    """Integer image_size is expanded to (h, w) tuple."""
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    batch = task.create_dummy_batch(image_size=32, context_length=10, batch_size=2)
    assert batch["image"].shape == (2, 3, 32, 32)
    assert batch["text"].shape == (2, 10)
    assert batch["text"].dtype == torch.long


def test_create_dummy_batch_tuple_image_size():
    """Tuple image_size is used directly."""
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    batch = task.create_dummy_batch(image_size=(16, 24), context_length=7)
    assert batch["image"].shape == (1, 3, 16, 24)
    assert batch["text"].shape == (1, 7)


def test_create_dummy_batch_respects_dtype():
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    batch = task.create_dummy_batch(image_size=8, context_length=5, dtype=torch.float16)
    assert batch["image"].dtype == torch.float16
    assert batch["text"].dtype == torch.long  # always long regardless of dtype


def test_create_dummy_batch_uses_configured_naflex_shape():
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    task.set_naflex_data_config(NaFlexDataConfig.resolve(patch_sizes=[16], seq_lens=[4]))

    batch = task.create_dummy_batch(batch_size=2, dtype=torch.float16)

    assert batch["image"]["patches"].shape == (2, 4, 16 * 16 * 3)
    assert batch["image"]["patches"].dtype == torch.float16
    assert batch["image"]["patch_coord"].shape == (2, 4, 2)
    assert batch["image"]["patch_valid"].shape == (2, 4)
    assert batch["image"]["seq_len"] == 4
    assert batch["text"].shape == (2, 5)


# ---------------------------------------------------------------------------
# forward() normalization: dict, positional, kwargs
# ---------------------------------------------------------------------------


def test_forward_dict_arg():
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    task.train()
    losses, _ = task(_batch())
    assert "loss" in losses and "contrastive_loss" in losses


def test_forward_positional_args():
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    task.train()
    b = _batch()
    losses, _ = task(b["image"], b["text"])
    assert "loss" in losses


def test_forward_kwargs():
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    task.train()
    b = _batch()
    losses, _ = task(image=b["image"], text=b["text"])
    assert "loss" in losses


def test_forward_mixed_positional_and_kwargs():
    """Mixed positional + keyword args should not silently drop kwargs."""
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    task.train()
    b = _batch()
    losses, _ = task(b["image"], text=b["text"])
    assert "loss" in losses


def test_forward_all_modes_equivalent():
    """All calling conventions produce the same loss."""
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    task.train()
    b = _batch()
    d = task(b)[0]["loss"]
    t = task(b["image"], b["text"])[0]["loss"]
    k = task(image=b["image"], text=b["text"])[0]["loss"]
    assert torch.allclose(d, t) and torch.allclose(t, k)


def test_forward_eval_dict():
    """Eval mode with dict arg calls model(**batch)."""
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    task.eval()
    out = task(_batch())
    assert "image_features" in out


def test_forward_eval_positional():
    """Eval mode with positional args passes through."""
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    task.eval()
    b = _batch()
    out = task(b["image"], b["text"])
    assert "image_features" in out


# ---------------------------------------------------------------------------
# data_keys
# ---------------------------------------------------------------------------


def test_data_keys_default():
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    assert task.data_keys == ("image", "text")


# ---------------------------------------------------------------------------
# state_dict_for_inference
# ---------------------------------------------------------------------------


def test_state_dict_for_inference_no_ema():
    model = TinyModel()
    task = CLIPTask(model, loss=DummyLoss())
    sd = task.state_dict_for_inference()
    assert "logit_scale" in sd
    assert torch.equal(sd["logit_scale"], model.logit_scale.data)


def test_state_dict_for_inference_prefers_ema():
    model = TinyModel()
    task = CLIPTask(model, loss=DummyLoss())
    ema_model = copy.deepcopy(model)
    with torch.no_grad():
        ema_model.logit_scale.fill_(123.0)
    task.trainable_module_ema = types.SimpleNamespace(module=ema_model)
    sd = task.state_dict_for_inference()
    assert torch.equal(sd["logit_scale"], ema_model.state_dict()["logit_scale"])


# ---------------------------------------------------------------------------
# clamp_logit_scale
# ---------------------------------------------------------------------------


def test_clamp_logit_scale_clamps_high():
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    with torch.no_grad():
        task.trainable_module.logit_scale.fill_(1000.0)
    task.clamp_logit_scale(max_val=math.log(100))
    assert task.trainable_module.logit_scale.item() <= math.log(100) + 1e-6


def test_clamp_logit_scale_noop_when_in_range():
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    with torch.no_grad():
        task.trainable_module.logit_scale.fill_(2.0)
    task.clamp_logit_scale()
    assert abs(task.trainable_module.logit_scale.item() - 2.0) < 1e-6


def test_clamp_logit_scale_clamps_negative():
    task = CLIPTask(TinyModel(), loss=DummyLoss())
    with torch.no_grad():
        task.trainable_module.logit_scale.fill_(-5.0)
    task.clamp_logit_scale()
    assert task.trainable_module.logit_scale.item() >= 0.0
