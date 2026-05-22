"""Round-trip tests for task-level checkpoint save/load (non-FSDP path).

Covers:
- save_checkpoint/load_checkpoint round-trip preserves model, optimizer,
  scaler, and EMA state
- Raw weights checkpoint (no epoch wrapper) loads via the `else` branch
- 'module.' prefix stripping when loading a DDP-trained checkpoint
  non-distributed
- state_dict_for_inference prefers EMA when present
- _reconcile_state_dict_shapes handles 0-D <-> 1-D (FSDP boundary) mismatch
"""
import os

import pytest
import torch
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import open_clip
from open_clip.task import CLIPTask, save_checkpoint, load_checkpoint
from open_clip.task.base_task import TrainingTask


def _make_task(model_name='RN50'):
    model = open_clip.create_model(model_name)
    return CLIPTask(model, rank=0, world_size=1)


def _mutate_model(model):
    """Perturb every parameter so round-trip equality is meaningful."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)


def test_save_load_roundtrip_restores_model_weights(tmp_path):
    task_a = _make_task()
    _mutate_model(task_a.trainable_module)
    optim_a = torch.optim.SGD(task_a.parameters(), lr=0.01)
    ckpt = save_checkpoint(task_a, optim_a, epoch=5)
    path = tmp_path / 'ckpt.pt'
    torch.save(ckpt, path)

    task_b = _make_task()
    optim_b = torch.optim.SGD(task_b.parameters(), lr=0.01)
    start_epoch = load_checkpoint(task_b, str(path), optimizer=optim_b)

    assert start_epoch == 5
    for (na, pa), (nb, pb) in zip(
        task_a.trainable_module.named_parameters(),
        task_b.trainable_module.named_parameters(),
    ):
        assert na == nb
        assert torch.equal(pa, pb), f'mismatch at {na}'


def test_save_load_roundtrip_restores_optimizer_state(tmp_path):
    task_a = _make_task()
    optim_a = torch.optim.SGD(task_a.parameters(), lr=0.01, momentum=0.9)
    # Take one step so optimizer has non-trivial state (momentum buffers).
    for p in task_a.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    optim_a.step()

    ckpt = save_checkpoint(task_a, optim_a, epoch=1)
    path = tmp_path / 'ckpt.pt'
    torch.save(ckpt, path)

    task_b = _make_task()
    optim_b = torch.optim.SGD(task_b.parameters(), lr=0.01, momentum=0.9)
    load_checkpoint(task_b, str(path), optimizer=optim_b)

    # Momentum buffers should round-trip
    state_a = optim_a.state_dict()['state']
    state_b = optim_b.state_dict()['state']
    assert set(state_a.keys()) == set(state_b.keys())
    for k in state_a:
        if 'momentum_buffer' in state_a[k]:
            assert torch.equal(
                state_a[k]['momentum_buffer'], state_b[k]['momentum_buffer']
            )


def test_save_load_roundtrip_preserves_train_counters(tmp_path):
    task_a = _make_task()
    optim_a = torch.optim.SGD(task_a.parameters(), lr=0.01)
    ckpt = save_checkpoint(
        task_a,
        optim_a,
        epoch=4,
        global_step=123,
        samples_seen=4567,
    )
    path = tmp_path / 'ckpt.pt'
    torch.save(ckpt, path)

    task_b = _make_task()
    metadata = {}
    start_epoch = load_checkpoint(task_b, str(path), metadata=metadata)

    assert start_epoch == 4
    assert metadata == {"global_step": 123, "samples_seen": 4567}


def test_load_checkpoint_ignores_unknown_top_level_metadata(tmp_path):
    task_a = _make_task()
    optim_a = torch.optim.SGD(task_a.parameters(), lr=0.01)
    ckpt = save_checkpoint(task_a, optim_a, epoch=1)
    ckpt["future_counter"] = 99
    path = tmp_path / 'ckpt.pt'
    torch.save(ckpt, path)

    task_b = _make_task()
    start_epoch = load_checkpoint(task_b, str(path))

    assert start_epoch == 1


def test_save_load_roundtrip_restores_scaler_state(tmp_path):
    task_a = _make_task()
    optim_a = torch.optim.SGD(task_a.parameters(), lr=0.01)
    scaler_a = torch.amp.GradScaler(device='cpu', init_scale=2048.)

    ckpt = save_checkpoint(task_a, optim_a, epoch=1, scaler=scaler_a)
    path = tmp_path / 'ckpt.pt'
    torch.save(ckpt, path)

    task_b = _make_task()
    optim_b = torch.optim.SGD(task_b.parameters(), lr=0.01)
    scaler_b = torch.amp.GradScaler(device='cpu', init_scale=1.)
    load_checkpoint(task_b, str(path), optimizer=optim_b, scaler=scaler_b)

    assert scaler_b.state_dict()['scale'] == scaler_a.state_dict()['scale']


def test_save_load_roundtrip_preserves_ema(tmp_path):
    task_a = _make_task()
    task_a.setup_ema(decay=0.99)
    # Perturb EMA weights so they diverge from the trainable module.
    with torch.no_grad():
        for p in task_a.trainable_module_ema.module.parameters():
            p.add_(torch.randn_like(p) * 0.02)

    optim_a = torch.optim.SGD(task_a.parameters(), lr=0.01)
    ckpt = save_checkpoint(task_a, optim_a, epoch=2)
    assert 'state_dict_ema' in ckpt
    path = tmp_path / 'ckpt.pt'
    torch.save(ckpt, path)

    task_b = _make_task()
    task_b.setup_ema(decay=0.99)
    optim_b = torch.optim.SGD(task_b.parameters(), lr=0.01)
    load_checkpoint(task_b, str(path), optimizer=optim_b)

    for (na, pa), (nb, pb) in zip(
        task_a.trainable_module_ema.module.named_parameters(),
        task_b.trainable_module_ema.module.named_parameters(),
    ):
        assert torch.equal(pa, pb), f'ema mismatch at {na}'


def test_load_checkpoint_raw_state_dict_no_epoch(tmp_path):
    """Bare state_dict (no 'epoch' key) loads via else-branch, returns 0."""
    task_a = _make_task()
    _mutate_model(task_a.trainable_module)
    raw_sd = task_a.trainable_module.state_dict()
    path = tmp_path / 'raw.pt'
    torch.save(raw_sd, path)

    task_b = _make_task()
    start_epoch = load_checkpoint(task_b, str(path))
    assert start_epoch == 0

    for (na, pa), (nb, pb) in zip(
        task_a.trainable_module.named_parameters(),
        task_b.trainable_module.named_parameters(),
    ):
        assert torch.equal(pa, pb)


def test_load_checkpoint_strips_module_prefix_when_not_distributed(tmp_path):
    """DDP saves keys with 'module.' prefix; non-distributed load should strip it."""
    task_a = _make_task()
    _mutate_model(task_a.trainable_module)
    optim_a = torch.optim.SGD(task_a.parameters(), lr=0.01)
    ckpt = save_checkpoint(task_a, optim_a, epoch=3)
    # Simulate DDP-saved checkpoint by prefixing every model key with 'module.'
    ckpt['state_dict'] = {f'module.{k}': v for k, v in ckpt['state_dict'].items()}
    path = tmp_path / 'ddp_ckpt.pt'
    torch.save(ckpt, path)

    task_b = _make_task()
    start_epoch = load_checkpoint(task_b, str(path), is_distributed=False)
    assert start_epoch == 3
    # Model state should match despite the prefix in the on-disk checkpoint
    for (na, pa), (nb, pb) in zip(
        task_a.trainable_module.named_parameters(),
        task_b.trainable_module.named_parameters(),
    ):
        assert torch.equal(pa, pb)


def test_state_dict_for_inference_without_ema_returns_trainable():
    task = _make_task()
    sd = task.state_dict_for_inference()
    raw = task.trainable_module.state_dict()
    assert sd.keys() == raw.keys()
    for k in sd:
        assert torch.equal(sd[k], raw[k])


def test_state_dict_for_inference_with_ema_returns_ema():
    task = _make_task()
    task.setup_ema(decay=0.99)
    with torch.no_grad():
        for p in task.trainable_module_ema.module.parameters():
            p.add_(0.5)
    sd = task.state_dict_for_inference()
    ema_sd = task.trainable_module_ema.module.state_dict()
    for k in ema_sd:
        assert torch.equal(sd[k], ema_sd[k])


# ──────────────────────────────────────────────────────────────────────
# _reconcile_state_dict_shapes: handles FSDP <-> non-FSDP boundary
# ──────────────────────────────────────────────────────────────────────

def test_reconcile_loads_1d_scalar_into_0d_model():
    """Checkpoint from FSDP training (logit_scale is [1]) loaded into a
    non-FSDP model (logit_scale is 0-D) — the scalar should be squeezed."""
    task = _make_task()
    model = task.trainable_module
    assert model.logit_scale.ndim == 0  # sanity: non-FSDP has 0-D scalar

    # Build a checkpoint with 1-D logit_scale (as FSDP would produce
    # pre-normalization, or after saving with normalize_checkpoint_scalars=False).
    sd = model.state_dict()
    sd['logit_scale'] = torch.tensor([2.5])  # 1-D [1] shape
    task.load_state_dict({'state_dict': sd})

    # After reconcile + load, model still has a 0-D scalar with the new value.
    assert task.trainable_module.logit_scale.ndim == 0
    assert torch.isclose(
        task.trainable_module.logit_scale, torch.tensor(2.5),
    )


def test_reconcile_loads_0d_scalar_into_1d_model():
    """Standard checkpoint (0-D logit_scale) loaded into an FSDP-reshaped
    model (1-D logit_scale) — the scalar should be unsqueezed."""
    task = _make_task()
    # Simulate the reshape prepare_fsdp() does: 0-D param -> [1] param.
    with torch.no_grad():
        old = task.trainable_module.logit_scale
        task.trainable_module.logit_scale = nn.Parameter(
            old.data.unsqueeze(0), requires_grad=old.requires_grad,
        )
    assert task.trainable_module.logit_scale.ndim == 1

    # Load a 0-D scalar from a standard (non-FSDP) checkpoint.
    sd = task.trainable_module.state_dict()
    sd['logit_scale'] = torch.tensor(1.7)  # 0-D
    task.load_state_dict({'state_dict': sd})

    assert task.trainable_module.logit_scale.ndim == 1
    assert torch.isclose(
        task.trainable_module.logit_scale, torch.tensor([1.7]),
    )
