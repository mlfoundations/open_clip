"""Tests for factory.create_task() dispatch logic.

Verifies the right TrainingTask subclass is constructed for each combination
of (args.distill, args.model, args.siglip), and that task-specific loss
defaults are wired up correctly.
"""
import os
import sys
import types

import pytest
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import open_clip
from open_clip import create_task
from open_clip.naflex_config import NaFlexDataConfig
from open_clip.task import CLIPTask, SigLIPTask, CoCaTask, DistillCLIPTask
from open_clip.loss import ClipLoss, SigLipLoss, CoCaLoss, DistillClipLoss


def _make_args(**overrides):
    defaults = dict(
        model='RN50',
        distill=False,
        siglip=False,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        coca_caption_loss_weight=2.0,
        coca_contrastive_loss_weight=1.0,
        loss_dist_impl=None,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_create_task_default_returns_clip_task():
    model = open_clip.create_model('RN50')
    args = _make_args(model='RN50')
    task = create_task(args, model=model)
    assert isinstance(task, CLIPTask)
    assert not isinstance(task, (SigLIPTask, CoCaTask, DistillCLIPTask))
    assert isinstance(task.loss, ClipLoss)


def test_create_task_siglip_returns_siglip_task():
    model = open_clip.create_model('RN50')
    args = _make_args(model='RN50', siglip=True)
    task = create_task(args, model=model)
    assert isinstance(task, SigLIPTask)
    assert isinstance(task.loss, SigLipLoss)


def test_create_task_coca_model_returns_coca_task():
    model = open_clip.create_model('coca_ViT-B-32')
    args = _make_args(model='coca_ViT-B-32')
    task = create_task(args, model=model)
    assert isinstance(task, CoCaTask)
    assert isinstance(task.loss, CoCaLoss)


def test_create_task_coca_model_name_case_insensitive():
    model = open_clip.create_model('coca_ViT-B-32')
    args = _make_args(model='CoCa_ViT-B-32')
    task = create_task(args, model=model)
    assert isinstance(task, CoCaTask)


def test_create_task_distill_returns_distill_task():
    student = open_clip.create_model('RN50')
    teacher = open_clip.create_model('RN50')
    args = _make_args(model='RN50', distill=True)
    task = create_task(args, model=student, dist_model=teacher)
    assert isinstance(task, DistillCLIPTask)
    assert isinstance(task.loss, DistillClipLoss)
    # Teacher is stored and frozen
    assert task.teacher is teacher
    assert all(not p.requires_grad for p in task.teacher.parameters())


def test_create_task_distill_takes_precedence_over_coca():
    """When --distill is set with a coca model, distill wins (first branch)."""
    student = open_clip.create_model('coca_ViT-B-32')
    teacher = open_clip.create_model('coca_ViT-B-32')
    args = _make_args(model='coca_ViT-B-32', distill=True)
    task = create_task(args, model=student, dist_model=teacher)
    assert isinstance(task, DistillCLIPTask)


def test_create_task_distill_precedence_over_siglip():
    student = open_clip.create_model('RN50')
    teacher = open_clip.create_model('RN50')
    args = _make_args(model='RN50', distill=True, siglip=True)
    task = create_task(args, model=student, dist_model=teacher)
    assert isinstance(task, DistillCLIPTask)


def test_create_task_plumbs_coca_loss_weights():
    model = open_clip.create_model('coca_ViT-B-32')
    args = _make_args(
        model='coca_ViT-B-32',
        coca_caption_loss_weight=3.5,
        coca_contrastive_loss_weight=0.25,
    )
    task = create_task(args, model=model)
    assert task.loss.caption_loss_weight == 3.5
    assert task.loss.clip_loss_weight == 0.25


def test_create_task_plumbs_rank_world_size():
    model = open_clip.create_model('RN50')
    args = _make_args(rank=3, world_size=8)
    task = create_task(args, model=model)
    assert task.loss.rank == 3
    assert task.loss.world_size == 8


def test_create_task_plumbs_local_loss_and_gather():
    model = open_clip.create_model('RN50')
    args = _make_args(local_loss=True, gather_with_grad=True)
    task = create_task(args, model=model)
    assert task.loss.local_loss is True
    assert task.loss.gather_with_grad is True


@pytest.mark.parametrize(
    ("torchcompile", "strategy", "expected_cache"),
    [
        (False, "task", True),
        (True, "model", True),
        (True, "task", False),
        (True, "step", False),
    ],
)
def test_create_task_sets_cache_labels_for_compile_strategy(torchcompile, strategy, expected_cache):
    model = open_clip.create_model('RN50')
    args = _make_args(torchcompile=torchcompile, torchcompile_strategy=strategy)
    task = create_task(args, model=model)

    assert task.loss.cache_labels is expected_cache


@pytest.mark.parametrize(
    ("torchcompile", "strategy", "expected_cache"),
    [
        (False, "task", True),
        (True, "model", True),
        (True, "task", False),
        (True, "step", False),
    ],
)
def test_create_loss_sets_cache_labels_for_compile_strategy(torchcompile, strategy, expected_cache):
    args = _make_args(torchcompile=torchcompile, torchcompile_strategy=strategy)
    loss = open_clip.create_loss(args)

    assert loss.cache_labels is expected_cache


def test_create_task_siglip_plumbs_dist_impl():
    model = open_clip.create_model('RN50')
    args = _make_args(siglip=True, loss_dist_impl='gather')
    task = create_task(args, model=model)
    assert isinstance(task, SigLIPTask)
    assert task.loss.dist_impl == 'gather'


def test_create_task_attaches_model_as_trainable_module():
    """Regardless of task type, the passed-in model is trainable_module."""
    model = open_clip.create_model('RN50')
    args = _make_args()
    task = create_task(args, model=model)
    assert task.trainable_module is model


def test_create_task_configures_naflex_dummy_shape():
    model = open_clip.create_model('RN50')
    args = _make_args()
    config = NaFlexDataConfig.resolve(patch_sizes=[16, 32], seq_lens=[4, 8])
    task = create_task(args, model=model, naflex_data_config=config)
    batch = task.create_dummy_batch(batch_size=2)

    assert batch["image"]["patches"].shape == (2, 8, 16 * 16 * 3)
    assert batch["text"].shape == (2, model.context_length)
