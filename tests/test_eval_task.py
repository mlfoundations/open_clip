"""Evaluation and model extraction work with plain and compiled tasks."""
import types
from unittest import mock

import pytest
import torch

from open_clip.task import (
    CLIPTask,
    SigLIPTask,
    CoCaTask,
    DistillCLIPTask,
    get_model_from_task,
)
from util_test import create_tiny_model


def _make_args():
    """Build a minimal args namespace matching what evaluate() needs."""
    return types.SimpleNamespace(
        device='cpu',
        precision='fp32',
        rank=0,
        local_rank=0,
        world_size=1,
        distributed=False,
        val_frequency=1,
        epochs=1,
        zeroshot_frequency=0,  # disable zero-shot to keep test fast
        model='tiny-clip',
        save_logs=False,
        wandb=False,
    )


@pytest.mark.parametrize("task_cls,family", [(CLIPTask, "clip"), (SigLIPTask, "clip"), (CoCaTask, "coca")])
@pytest.mark.parametrize("compiled", [False, True], ids=["raw", "compiled"])
def test_get_model_from_task(task_cls, family, compiled):
    model = create_tiny_model(family)
    task = task_cls(model, rank=0, world_size=1)
    if compiled:
        task = torch.compile(task, backend="eager")
    assert get_model_from_task(task) is model


@pytest.mark.parametrize("family", ["clip", "coca"])
def test_get_model_from_plain_model(family):
    model = create_tiny_model(family)
    assert get_model_from_task(model) is model


@pytest.mark.parametrize("compiled", [False, True], ids=["raw", "compiled"])
def test_get_model_from_distill_task(compiled):
    student = create_tiny_model()
    teacher = create_tiny_model()
    task = DistillCLIPTask(student, teacher, rank=0, world_size=1)
    if compiled:
        task = torch.compile(task, backend="eager")
    assert get_model_from_task(task) is student


def _make_val_dataloader(model, batch_size=2, num_batches=2):
    """Create an in-memory iterable of image/text batch dicts; no worker processes."""
    image_size = model.visual.image_size
    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)
    batches = [
        {
            "image": torch.randn(batch_size, 3, *image_size),
            "text": torch.randint(1, model.vocab_size, (batch_size, model.context_length)),
        }
        for _ in range(num_batches)
    ]
    dl = mock.MagicMock()
    dl.__iter__.return_value = iter(batches)
    dl.num_samples = batch_size * num_batches
    return dl


@pytest.mark.parametrize("task_cls", [CLIPTask, SigLIPTask])
@pytest.mark.parametrize("compiled", [False, True], ids=["raw", "compiled"])
def test_evaluate_with_task(task_cls, compiled):
    from open_clip_train.train import evaluate

    model = create_tiny_model()
    task = task_cls(model, rank=0, world_size=1)
    if compiled:
        task = torch.compile(task, backend="eager")
    data = {'val': types.SimpleNamespace(dataloader=_make_val_dataloader(model))}

    metrics = evaluate(task, data, epoch=1, args=_make_args())
    assert 'clip_val_loss' in metrics
    assert torch.isfinite(torch.as_tensor(metrics['clip_val_loss']))


@pytest.mark.parametrize("trainer", ["legacy", "task"])
def test_generative_eval_accumulates_every_batch(trainer):
    import math
    from open_clip_train import legacy_train, train

    class CaptionModel(torch.nn.Module):
        pad_id = 0

        def forward(self, image, text):
            batch_size = text.shape[0]
            return {
                "image_features": torch.ones(batch_size, 4),
                "text_features": torch.ones(batch_size, 4),
                "logit_scale": torch.tensor(1.),
                "logits": torch.zeros(batch_size, text.shape[1], 4),
            }

    batches = [
        {"image": torch.zeros(n, 3, 4, 4), "text": torch.tensor([[1, 2, 3]]).repeat(n, 1)}
        for n in (2, 3)
    ]
    loader = mock.MagicMock()
    loader.__iter__.return_value = iter(batches)
    loader.num_samples = 5
    model = CaptionModel()
    evaluate = legacy_train.evaluate if trainer == "legacy" else train.evaluate
    target = model if trainer == "legacy" else CoCaTask(model)
    metrics = evaluate(target, {"val": types.SimpleNamespace(dataloader=loader)}, 1, _make_args())
    assert metrics["num_samples"] == 5
    assert metrics["val_generative_loss"] == pytest.approx(math.log(4))
