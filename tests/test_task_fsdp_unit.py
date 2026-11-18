"""Unit tests for FSDP2-related utilities in TrainingTask.

Tests _normalize_scalar_params, _reconcile_state_dict_shapes,
prepare_fsdp 0-D param reshaping, EMA+FSDP setup, and FSDPModelEma.
Tests run on CPU, mocking
fully_shard and register_fsdp_forward_method except for one real 2-rank test.
"""
import copy
import os
import subprocess
import sys
import types
from pathlib import Path

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


def test_resolve_mixed_param_dtypes():
    """Mixed models store fp32 and run in their original dtypes; uniform models are left alone."""
    cfsdp = pytest.importorskip("torch.distributed._composable.fsdp")
    policy = cfsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16)

    uniform = nn.Linear(2, 2)
    assert TrainingTask._resolve_mixed_param_dtypes(uniform, policy) is policy
    assert uniform.weight.dtype == torch.float32

    mixed = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
    mixed[0].to(torch.bfloat16)
    mixed.register_parameter("logit_scale", nn.Parameter(torch.ones(1)))
    if not hasattr(policy, "param_dtype_override_fn"):
        with pytest.raises(RuntimeError, match="PyTorch 2.15"):
            TrainingTask._resolve_mixed_param_dtypes(mixed, policy)
        assert mixed[0].weight.dtype == torch.bfloat16
        return

    resolved = TrainingTask._resolve_mixed_param_dtypes(mixed, policy)
    assert {p.dtype for p in mixed.parameters()} == {torch.float32}
    assert resolved._resolve_for_param(mixed[0].weight).param_dtype == torch.bfloat16
    assert resolved._resolve_for_param(mixed[1].weight).param_dtype == torch.float32
    assert resolved._resolve_for_param(mixed.logit_scale).param_dtype == torch.bfloat16

    mismatched = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
    mismatched[0].to(torch.bfloat16)
    with pytest.raises(ValueError, match="param_dtype"):
        TrainingTask._resolve_mixed_param_dtypes(mismatched, cfsdp.MixedPrecisionPolicy(param_dtype=torch.float16))


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


# ---------------------------------------------------------------------------
# FSDPModelEma
# ---------------------------------------------------------------------------


class TinyEmaModel(nn.Module):
    """Tied params plus each kind of buffer the EMA treats differently."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 2, bias=False)
        self.b = nn.Linear(2, 2, bias=False)
        self.b.weight = self.a.weight
        self.register_buffer("running", torch.zeros(2))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("attn_mask", torch.full((2, 2), float("-inf")), persistent=False)


def test_fsdp_model_ema_update():
    from open_clip.task import FSDPModelEma

    model = TinyEmaModel()
    ema = FSDPModelEma(copy.deepcopy(model), decay=0.9)
    with torch.no_grad():
        model.a.weight.add_(1.0)
        model.running.add_(1.0)
        model.count.add_(3)
        model.attn_mask.fill_(0.)
    start = ema.module.a.weight.clone()
    ema.update(model)

    # tied params move once, not 1 - decay ** 2
    assert torch.allclose(ema.module.a.weight, start + 0.1)
    assert ema.module.a.weight is ema.module.b.weight
    assert torch.allclose(ema.module.running, torch.full((2,), 0.1))
    assert ema.module.count.item() == 3
    # non-persistent buffers are left alone: lerping -inf would give NaN
    assert torch.equal(ema.module.attn_mask, torch.full((2, 2), float("-inf")))

    ema.set(model)
    assert torch.equal(ema.module.a.weight, model.a.weight)
    assert torch.equal(ema.module.running, model.running)


def test_fsdp_model_ema_decay_matches_timm():
    timm_utils = pytest.importorskip("timm.utils")
    from open_clip.task import FSDPModelEma

    model = TinyEmaModel()
    for kwargs in (
        dict(decay=0.9),
        dict(decay=0.9999, use_warmup=True),
        dict(decay=0.99, use_warmup=True, warmup_gamma=2.0, warmup_power=0.75, min_decay=0.1, update_after_step=2),
    ):
        ema = FSDPModelEma(copy.deepcopy(model), **kwargs)
        timm_ema = timm_utils.ModelEmaV3(model, **kwargs)
        for step in (None, 0, 1, 2, 3, 4, 100, 10 ** 7):
            assert ema.get_decay(step) == timm_ema.get_decay(step)


def test_fsdp_model_ema_update_uses_step():
    from open_clip.task import FSDPModelEma

    model = TinyEmaModel()
    ema = FSDPModelEma(copy.deepcopy(model), decay=0.9, use_warmup=True)
    with torch.no_grad():
        model.a.weight.add_(1.0)
    ema.update(model, step=1)
    assert torch.equal(ema.module.a.weight, model.a.weight)

    with torch.no_grad():
        model.a.weight.add_(1.0)
    start = ema.module.a.weight.clone()
    ema.update(model, step=2)
    assert torch.allclose(ema.module.a.weight, start + 1 - ema.get_decay(2))


def test_fsdp_model_ema_matches_compiled_block_names():
    """The trained model's blocks may be wrapped by torch.compile while the EMA's are not."""
    from open_clip.task import FSDPModelEma

    class Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    ema = FSDPModelEma(copy.deepcopy(model), decay=0.)
    model[0] = Wrapped(model[0])
    with torch.no_grad():
        model[0]._orig_mod.weight.add_(1.0)
    ema.update(model)
    assert torch.allclose(ema.module[0].weight, model[0]._orig_mod.weight)


def test_prepare_fsdp_converts_ema(monkeypatch):
    pytest.importorskip("torch.distributed._composable.fsdp")
    pytest.importorskip("torch.distributed.fsdp")
    import torch.distributed._composable.fsdp as cfsdp
    import torch.distributed.fsdp as fsdp_mod
    from open_clip.task import FSDPModelEma

    shard_kwargs = []
    monkeypatch.setattr(cfsdp, "fully_shard", lambda m, **kw: shard_kwargs.append((m, kw)))
    monkeypatch.setattr(fsdp_mod, "register_fsdp_forward_method", lambda m, n: None)

    task = CLIPTask(TinyFSDPModel(), loss=DummyLoss(), verbose=False)
    monkeypatch.setattr(task, "_get_fsdp_shard_modules", lambda: [])
    task.setup_ema(decay=0.5, warmup_gamma=2.0, warmup_power=0.75)
    ema_module = task.trainable_module_ema.module
    timm_ema = task.trainable_module_ema
    task.prepare_fsdp(reshard_after_forward=False)

    assert isinstance(task.trainable_module_ema, FSDPModelEma)
    assert task.trainable_module_ema.module is ema_module
    assert task.trainable_module_ema.get_decay(3) == timm_ema.get_decay(3) != 0.5
    assert ema_module.logit_scale.shape == (1,)
    assert not any(p.requires_grad for p in ema_module.parameters())
    # eval-only: the EMA reshards regardless of the training setting
    assert [kw["reshard_after_forward"] for m, kw in shard_kwargs if m is ema_module] == [True]
    assert [kw["reshard_after_forward"] for m, kw in shard_kwargs if m is task.trainable_module] == [False]


FSDP_EMA_SCRIPT = """
import logging, sys
import torch, torch.distributed as dist
logging.disable(logging.WARNING)
sys.path.insert(0, sys.argv[2])
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from open_clip.task import CLIPTask, FSDPModelEma, load_sharded_checkpoint, save_sharded_checkpoint
from open_clip.task.base_task import unwrap_model
from util_test import create_tiny_model

dist.init_process_group(backend="gloo")
rank = dist.get_rank()

def make_task():
    torch.manual_seed(0)
    task = CLIPTask(create_tiny_model("clip"), rank=rank, world_size=dist.get_world_size())
    task.setup_ema(decay=0.9)
    return task.prepare_fsdp()

def full(module):
    return get_model_state_dict(module, options=StateDictOptions(full_state_dict=True, cpu_offload=False))

task = make_task()
ema, model = task.trainable_module_ema, unwrap_model(task.trainable_module)
assert isinstance(ema, FSDPModelEma)
optimizer = torch.optim.SGD(task.trainable_module.parameters(), lr=0.1)
batch = {"image": torch.rand(4, 3, 32, 32), "text": torch.randint(1, 64, (4, 8))}
replay = full(ema.module)
for step in (1, 2):
    task.train()
    optimizer.zero_grad()
    task(batch)[0]["loss"].backward()
    optimizer.step()
    task.update_ema(step=step)
    replay = {k: v.lerp(p, 1 - ema.get_decay(step)) for (k, v), p in zip(replay.items(), full(model).values())}
    # The first EMA forward runs in inference_mode, then the next update writes to the sharded params
    task.eval()
    with torch.inference_mode():
        output = task(batch)
    assert torch.isfinite(output["text_features"]).all() and torch.isfinite(output["image_features"]).all()

ema_sd = full(ema.module)
assert all(torch.allclose(ema_sd[k], replay[k]) for k in replay)
assert not all(torch.equal(ema_sd[k], v) for k, v in full(model).items())

save_sharded_checkpoint(task, optimizer, sys.argv[1], epoch=1, is_master=rank == 0)
dist.barrier()
resumed = make_task()
load_sharded_checkpoint(resumed, sys.argv[1])
assert all(torch.equal(v, ema_sd[k]) for k, v in full(resumed.trainable_module_ema.module).items())
dist.destroy_process_group()
"""


@pytest.mark.skipif(sys.platform != "linux", reason="gloo CPU sharding is only exercised on Linux")
def test_fsdp_model_ema_two_rank_cpu(tmp_path):
    """Real FSDP on 2 CPU ranks: everything above mocks fully_shard."""
    pytest.importorskip("torch.distributed._composable.fsdp")
    tests_dir = Path(__file__).parent
    script = tmp_path / "fsdp_ema.py"
    script.write_text(FSDP_EMA_SCRIPT)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join([str(tests_dir.parent / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
         str(script), str(tmp_path / "ckpt"), str(tests_dir)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
