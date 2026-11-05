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
    GenLipTask,
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


def _run_eval(evaluate, target, batches):
    loader = mock.MagicMock()
    loader.__iter__.return_value = iter(batches)
    loader.num_samples = sum(len(next(iter(batch.values()))) for batch in batches)
    return evaluate(target, {"val": types.SimpleNamespace(dataloader=loader)}, 1, _make_args())


@pytest.mark.parametrize("trainer,masked", [
    ("legacy", False),
    ("task", False),
    ("task", True),
    ("genlip", False),
    ("genlip", True),
])
def test_generative_eval_weights_by_valid_tokens(trainer, masked):
    import math
    from open_clip_train import legacy_train, train

    class CaptionModel(torch.nn.Module):
        pad_id = 0

        def forward(self, image, text, text_valid=None, compute_loss=False, **kwargs):
            batch_size = text.shape[0]
            # Uniform over 4 tokens, or over 2 when the image flag is set: per-token CE of ln(4) or ln(2).
            logits = torch.zeros(batch_size, text.shape[1], 4)
            logits[..., 2:] = image[:, 0, 0, 0].view(-1, 1, 1) * -1e3
            if compute_loss:
                valid = text_valid if text_valid is not None else text != self.pad_id
                labels = text.masked_fill(~valid, -100)
                return {"caption_loss_ce": torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), labels)}
            return {
                "image_features": torch.ones(batch_size, 4),
                "text_features": torch.ones(batch_size, 4),
                "logit_scale": torch.tensor(1.),
                "logits": logits,
            }

    # A short caption, a long caption, and an empty one.
    full = {
        "image": torch.tensor([1., 0., 1.]).view(3, 1, 1, 1),
        "text": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]),
    }
    if masked:
        full["text_valid"] = full["text"] != 0
        full["text"][1, 1] = 0  # a real target equal to pad_id
        full["text"][0, 2] = 1  # an invalid token not equal to pad_id
    singles = [{key: value[i:i + 1] for key, value in full.items()} for i in range(3)]

    model = CaptionModel()
    evaluate = legacy_train.evaluate if trainer == "legacy" else train.evaluate
    target = {"legacy": model, "task": CoCaTask(model), "genlip": GenLipTask(model)}[trainer]

    # GenLIP also predicts the first caption token from its prefix.
    short, long = (2, 4) if trainer == "genlip" else (1, 3)
    expected = (short * math.log(2) + long * math.log(4)) / (short + long)
    for batches, num_samples in ((singles[:2], 2), (singles, 3), ([full], 3)):
        metrics = _run_eval(evaluate, target, batches)
        assert metrics["num_samples"] == num_samples
        assert metrics["val_generative_loss"] == pytest.approx(expected)

    metrics = _run_eval(evaluate, target, singles[2:])
    assert metrics["num_samples"] == 1
    assert "val_generative_loss" not in metrics


def test_generic_eval_loss_weights_by_samples():
    from open_clip_train import train

    class LossTask(torch.nn.Module):
        primary_key = "audio"

        def prepare_batch(self, batch, device, input_dtype=None):
            return batch

        def batch_size(self, batch):
            return len(batch["audio"])

        def forward(self, batch):
            return {"loss": batch["audio"].mean()}

    batches = [{"audio": torch.tensor([1.])}, {"audio": torch.tensor([3., 3., 3.])}]
    metrics = _run_eval(train.evaluate, LossTask(), batches)
    assert metrics["num_samples"] == 4
    assert metrics["val_generative_loss"] == pytest.approx(2.5)
