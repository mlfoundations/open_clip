"""Tests for factory.create_task() dispatch logic.

Verifies dispatch uses the built model plus distillation/SigLIP settings, rejects
unsupported combinations, and wires up task-specific loss defaults correctly.
"""
import os
import types

import pytest

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import open_clip
from open_clip import create_task
from open_clip.naflex_config import NaFlexDataConfig
from open_clip.model_traits import CLIP_TRAITS
from open_clip_train.loss import create_loss_from_args
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


def test_create_task_mammut_model_returns_coca_task():
    """MaMMUT shares the CoCa output contract, so it trains via CoCaTask/CoCaLoss."""
    model = open_clip.create_model('mammut2_ViT-B-32')
    args = _make_args(model='mammut2_ViT-B-32', coca_caption_loss_weight=1.0)
    task = create_task(args, model=model)
    assert isinstance(task, CoCaTask)
    assert isinstance(task.loss, CoCaLoss)
    assert task.loss.caption_loss_weight == 1.0


def test_create_task_dispatches_on_model_type_not_name():
    """hf-hub:/local-dir:/renamed configs don't carry an arch hint in args.model; dispatch must
    key on the built model instance or the caption loss silently disappears."""
    model = open_clip.create_model('coca_ViT-B-32')
    args = _make_args(model='hf-hub:someorg/my-renamed-captioner')
    task = create_task(args, model=model)
    assert isinstance(task, CoCaTask)
    assert isinstance(task.loss, CoCaLoss)


def test_create_task_and_loss_dispatch_unwrap_wrapped_models():
    """isinstance dispatch must see through torch.compile (and DDP) wrappers."""
    import torch

    model = open_clip.create_model('coca_ViT-B-32')
    compiled = torch.compile(model)
    args = _make_args(model='hf-hub:someorg/my-renamed-captioner')
    task = create_task(args, model=compiled)
    assert isinstance(task, CoCaTask)
    loss = create_loss_from_args(args, model=compiled)
    assert isinstance(loss, CoCaLoss)
    assert loss.pad_id == 0  # pad attr read through the wrapper


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


def test_create_task_rejects_distilling_coca():
    """Distillation must not silently bypass the captioning task."""
    student = open_clip.create_model('coca_ViT-B-32')
    args = _make_args(model='coca_ViT-B-32', distill=True)
    with pytest.raises(ValueError, match="distillation is not supported for generative models"):
        create_task(args, model=student)


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


def test_create_task_plumbs_caption_loss_options():
    model = open_clip.create_model('coca_ViT-B-32')
    args = _make_args(
        model='coca_ViT-B-32',
        caption_z_loss_weight=1e-4,
        caption_loss_compute_dtype='model',
        caption_loss_chunk_size=512,
    )
    task = create_task(args, model=model)

    assert task.caption_z_loss_weight == 1e-4
    assert task.caption_loss_compute_dtype == 'model'
    assert task.caption_loss_chunk_size == 512
    assert task.loss.z_loss_weight == 1e-4
    assert task.loss.compute_dtype is None


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
def test_legacy_loss_sets_cache_labels_for_compile_strategy(torchcompile, strategy, expected_cache):
    args = _make_args(torchcompile=torchcompile, torchcompile_strategy=strategy)
    loss = create_loss_from_args(args, model=types.SimpleNamespace(traits=CLIP_TRAITS))

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
    # Model geometry known: the (base-size) eval patch is flattened, like the real eval transform.
    config = NaFlexDataConfig.resolve(
        patch_sizes=[16, 32], seq_lens=[4, 8], model_patch_size=16, supports_patch_interpolation=True)
    task = create_task(args, model=model, naflex_data_config=config)
    batch = task.create_dummy_batch(batch_size=2)

    assert batch["image"]["patches"].shape == (2, 8, 16 * 16 * 3)
    assert batch["text"].shape == (2, model.context_length)

    # Data-only config (no model geometry): the historical multi-size rule keeps patches spatial, and the dummy
    # must match what the eval transform emits under that same rule.
    config = NaFlexDataConfig.resolve(patch_sizes=[16, 32], seq_lens=[4, 8])
    task = create_task(args, model=model, naflex_data_config=config)
    assert task.create_dummy_batch(batch_size=2)["image"]["patches"].shape == (2, 8, 16, 16, 3)
