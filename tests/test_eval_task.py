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


@pytest.mark.parametrize("trainer,masked,dtype", [
    ("legacy", False, torch.float64),
    ("task", False, torch.float64),
    ("task", True, torch.float64),
    ("genlip", False, torch.float64),
    ("genlip", True, torch.float64),
    ("genlap", True, torch.float64),
    ("genlip", False, torch.float16),
    ("genlip", False, torch.bfloat16),
])
def test_generative_eval_accumulates_every_batch(trainer, masked, dtype):
    import math
    from open_clip.task import GenLipTask, GenLapTask
    from open_clip_train import legacy_train, train

    prefix_lm = trainer in ("genlip", "genlap")
    input_key = "audio" if trainer == "genlap" else "image"
    # Large but finite half-precision means would overflow when multiplied by token counts in fp16.
    offset = 30000.0 if dtype in (torch.float16, torch.bfloat16) else math.log(3)

    class CaptionModel(torch.nn.Module):
        pad_id = 0

        def forward(self, image=None, text=None, audio=None, text_valid=None, compute_loss=False, **kwargs):
            batch_size = text.shape[0]
            inputs = image if image is not None else audio
            logits = torch.zeros(batch_size, text.shape[1], 2, dtype=torch.float64)
            logits[..., 0] = inputs[:, 0, 0, 0].double().unsqueeze(1) * offset
            if compute_loss:
                valid = text_valid if text_valid is not None else text != self.pad_id
                labels = text.masked_fill(~valid, -100)
                ce = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), labels).to(dtype)
                return {"caption_loss_ce": ce, "caption_loss_z": ce.new_tensor(10.)}
            return {
                "image_features": torch.ones(batch_size, 4),
                "text_features": torch.ones(batch_size, 4),
                "logit_scale": torch.tensor(1.),
                "logits": logits,
            }

    full = {
        input_key: torch.tensor([0., 1., float("nan")]).view(3, 1, 1, 1),
        "text": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]),
    }
    if masked:
        full["text_valid"] = full["text"] != 0
        full["text"][1, 1] = 0  # a real target equal to pad_id
        full["text"][0, 2] = 1  # an invalid target not equal to pad_id
    batches = [{key: value[i:i + 1] for key, value in full.items()} for i in range(3)]
    model = CaptionModel()
    evaluate = legacy_train.evaluate if trainer == "legacy" else train.evaluate
    if prefix_lm:
        task_cls = GenLapTask if trainer == "genlap" else GenLipTask
        target = task_cls(model, caption_z_loss_weight=0.5)
    else:
        target = model if trainer == "legacy" else CoCaTask(model)

    def run(batches):
        loader = mock.MagicMock()
        loader.__iter__.return_value = iter(batches)
        loader.num_samples = sum(len(batch["text"]) for batch in batches)
        return evaluate(target, {"val": types.SimpleNamespace(dataloader=loader)}, 1, _make_args())

    short_tokens, long_tokens = (2, 4) if prefix_lm else (1, 3)
    long_ce = math.log1p(math.exp(-offset)) + offset
    expected = (short_tokens * math.log(2) + long_tokens * long_ce - (offset if masked else 0))
    expected /= short_tokens + long_tokens
    tolerance = 0.005 if dtype in (torch.float16, torch.bfloat16) else 1e-12
    nonempty = {key: value[:2] for key, value in full.items()}
    for grouped in (batches[:2], [nonempty], batches, [full]):
        metrics = run(grouped)
        assert metrics["num_samples"] == sum(len(batch["text"]) for batch in grouped)
        assert metrics["val_generative_loss"] == pytest.approx(expected, rel=tolerance)

    # An empty-target batch is harmless even with nonfinite logits / a NaN pre-reduced mean.
    metrics = run([batches[-1]])
    assert metrics["num_samples"] == 1
    assert metrics["val_generative_loss"] == 0.0
    # Nonfinite losses on valid targets must still be visible.
    bad = dict(batches[0], **{input_key: torch.full_like(batches[0][input_key], float("nan"))})
    assert math.isnan(run([bad])["val_generative_loss"])
