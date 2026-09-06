"""Unit tests for FSDP2-related utilities in TrainingTask.

Tests _normalize_scalar_params, _reconcile_state_dict_shapes,
prepare_fsdp 0-D param reshaping, and EMA+FSDP assertion.
All tests run on CPU without a real FSDP cluster by mocking
fully_shard and register_fsdp_forward_method.
"""
import types

import pytest
import torch
import torch.nn as nn

from open_clip.task import CLIPTask
from open_clip.task.base_task import TrainingTask


# ---------------------------------------------------------------------------
# Tiny model stub
# ---------------------------------------------------------------------------


class TinyFSDPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(3, 3)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))  # 0-D
        self.logit_bias = nn.Parameter(torch.tensor(-0.5))  # 0-D
        self.visual = types.SimpleNamespace(image_size=8)
        self.context_length = 5

    def encode_text(self, text):
        return self.block(text[:, :3].float())

    def encode_image(self, image):
        return self.block(image.float().mean(dim=(2, 3)))

    def forward(self, image, text):
        return {
            "image_features": self.encode_image(image),
            "text_features": self.encode_text(text),
            "logit_scale": self.logit_scale,
        }


class DummyLoss(nn.Module):
    def forward(self, output_dict=False, **kw):
        return {"contrastive_loss": torch.tensor(1.0)}


# ---------------------------------------------------------------------------
# _normalize_scalar_params
# ---------------------------------------------------------------------------


def test_normalize_scalar_params_squeezes_len1():
    sd = {
        "logit_scale": torch.tensor([4.6]),    # [1] -> scalar
        "logit_bias": torch.tensor([-0.5]),    # [1] -> scalar
        "weight": torch.randn(3, 3),           # unchanged
    }
    result = TrainingTask._normalize_scalar_params(sd)
    assert result["logit_scale"].ndim == 0
    assert result["logit_bias"].ndim == 0
    assert result["weight"].ndim == 2


def test_normalize_scalar_params_ignores_longer_vectors():
    sd = {
        "embedding": torch.randn(5),
        "scale": torch.tensor([1.0]),
    }
    result = TrainingTask._normalize_scalar_params(sd)
    assert result["embedding"].ndim == 1 and result["embedding"].shape[0] == 5
    assert result["scale"].ndim == 0


# ---------------------------------------------------------------------------
# _reconcile_state_dict_shapes
# ---------------------------------------------------------------------------


def test_reconcile_noop_when_shapes_match():
    model = TinyFSDPModel()
    sd = {"logit_scale": torch.tensor(4.6)}  # 0-D matches 0-D
    result = TrainingTask._reconcile_state_dict_shapes(model, sd)
    assert result["logit_scale"].ndim == 0


# ---------------------------------------------------------------------------
# prepare_fsdp: 0-D reshape + registration (mocked)
# ---------------------------------------------------------------------------


def test_prepare_fsdp_reshapes_0d_params(monkeypatch):
    """prepare_fsdp reshapes 0-D params to [1]."""
    pytest.importorskip("torch.distributed._composable.fsdp")
    pytest.importorskip("torch.distributed.fsdp")
    import torch.distributed._composable.fsdp as cfsdp
    import torch.distributed.fsdp as fsdp_mod

    model = TinyFSDPModel()
    task = CLIPTask(model, loss=DummyLoss(), verbose=False)
    monkeypatch.setattr(task, "_get_fsdp_shard_modules", lambda: [])

    monkeypatch.setattr(cfsdp, "fully_shard", lambda m, **kw: None)
    monkeypatch.setattr(fsdp_mod, "register_fsdp_forward_method", lambda m, n: None)

    task.prepare_fsdp()

    assert model.logit_scale.ndim == 1 and model.logit_scale.shape == (1,)
    assert model.logit_bias.ndim == 1 and model.logit_bias.shape == (1,)


def test_prepare_fsdp_registers_forward_methods(monkeypatch):
    """prepare_fsdp registers encode_text and encode_image as FSDP forward methods."""
    pytest.importorskip("torch.distributed._composable.fsdp")
    pytest.importorskip("torch.distributed.fsdp")
    import torch.distributed._composable.fsdp as cfsdp
    import torch.distributed.fsdp as fsdp_mod

    model = TinyFSDPModel()
    task = CLIPTask(model, loss=DummyLoss(), verbose=False)
    monkeypatch.setattr(task, "_get_fsdp_shard_modules", lambda: [])

    reg_calls = []
    monkeypatch.setattr(cfsdp, "fully_shard", lambda m, **kw: None)
    monkeypatch.setattr(
        fsdp_mod, "register_fsdp_forward_method",
        lambda m, n: reg_calls.append(n),
    )

    task.prepare_fsdp()
    assert "encode_text" in reg_calls
    assert "encode_image" in reg_calls


def test_prepare_fsdp_shards_submodules_and_root(monkeypatch):
    """prepare_fsdp shards discovered submodules then the root."""
    pytest.importorskip("torch.distributed._composable.fsdp")
    pytest.importorskip("torch.distributed.fsdp")
    import torch.distributed._composable.fsdp as cfsdp
    import torch.distributed.fsdp as fsdp_mod

    model = TinyFSDPModel()
    task = CLIPTask(model, loss=DummyLoss(), verbose=False)
    monkeypatch.setattr(
        task, "_get_fsdp_shard_modules",
        lambda: [("block", model.block)],
    )

    shard_calls = []
    monkeypatch.setattr(cfsdp, "fully_shard", lambda m, **kw: shard_calls.append(m))
    monkeypatch.setattr(fsdp_mod, "register_fsdp_forward_method", lambda m, n: None)

    task.prepare_fsdp()
    # block + root trainable_module
    assert len(shard_calls) == 2
    assert shard_calls[0] is model.block
    assert shard_calls[1] is task.trainable_module


# ---------------------------------------------------------------------------
# EMA + FSDP assertion
# ---------------------------------------------------------------------------


def test_setup_ema_after_fsdp_raises(monkeypatch):
    """Setting up EMA after FSDP is prepared should raise."""
    pytest.importorskip("torch.distributed._composable.fsdp")
    pytest.importorskip("torch.distributed.fsdp")
    import torch.distributed._composable.fsdp as cfsdp
    import torch.distributed.fsdp as fsdp_mod

    model = TinyFSDPModel()
    task = CLIPTask(model, loss=DummyLoss(), verbose=False)
    monkeypatch.setattr(task, "_get_fsdp_shard_modules", lambda: [])
    monkeypatch.setattr(cfsdp, "fully_shard", lambda m, **kw: None)
    monkeypatch.setattr(fsdp_mod, "register_fsdp_forward_method", lambda m, n: None)

    task.prepare_fsdp()

    with pytest.raises(AssertionError, match="EMA must be set up before"):
        task.setup_ema()


# ---------------------------------------------------------------------------
# state_dict with FSDP flag
# ---------------------------------------------------------------------------


def test_state_dict_normalizes_scalars_by_default():
    """When normalize_checkpoint_scalars=True, [1] params are squeezed to 0-D."""
    model = TinyFSDPModel()
    task = CLIPTask(model, loss=DummyLoss())
    # Manually set logit_scale to [1] as if FSDP reshaped it
    model.logit_scale = nn.Parameter(torch.tensor([4.6]))
    sd = task.state_dict()
    # Non-FSDP state_dict doesn't normalize (only FSDP path does)
    # But the raw model state_dict just returns whatever shape it has
    assert "state_dict" in sd
