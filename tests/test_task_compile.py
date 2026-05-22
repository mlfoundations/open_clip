from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_clip.task import TrainingTask


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x=None, target=None, text=None, image=None):
        if text is not None:
            return {"text_features": self.linear(text.float())}
        if image is not None:
            return {"image_features": self.linear(image.float())}
        return {"logits": self.linear(x)}


class TinyTask(TrainingTask):
    @property
    def data_keys(self):
        return ("x", "target")

    def training_forward(self, batch):
        out = self.trainable_module(**batch)
        loss = F.cross_entropy(out["logits"], batch["target"])
        return {
            "loss": loss,
            "ce_loss": loss,
            "logits": out["logits"],
        }

    def clamp_logit_scale(self):
        pass


def _batch():
    return {
        "x": torch.randn(4, 3),
        "target": torch.tensor([0, 1, 0, 1]),
    }


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="requires torch.compile")
def test_task_compile_keeps_task_methods_and_compiles_train_eval_forward():
    task = TinyTask(TinyModel())
    task.compile(target="task", backend="eager")

    assert task._compiled_training_forward is not None
    assert task._compiled_eval_forward is not None
    assert task.batch_size(_batch()) == 4

    losses = task(_batch())
    losses["loss"].backward()

    task.eval()
    out = task(_batch())
    assert out["logits"].shape == (4, 2)

    zeroshot_out = task(text=torch.randn(4, 3))
    assert zeroshot_out["text_features"].shape == (4, 2)
    zeroshot_out = task(image=torch.randn(4, 3))
    assert zeroshot_out["image_features"].shape == (4, 2)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="requires torch.compile")
def test_task_compile_model_compiles_trainable_module_only():
    task = TinyTask(TinyModel())
    task.compile(target="model", backend="eager")

    assert hasattr(task.trainable_module, "_orig_mod")
    assert task._compiled_training_forward is None
    assert task(_batch())["loss"].isfinite()


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="requires torch.compile")
def test_compiled_train_step_runs_forward_backward_and_optimizer_step():
    from open_clip_train.train import _get_compiled_train_step

    task = TinyTask(TinyModel())
    optimizer = torch.optim.SGD(task.parameters(), lr=0.1)
    args = SimpleNamespace(torchcompile_backend="eager", torchcompile_mode=None, grad_clip_norm=None)
    compiled_step = _get_compiled_train_step(task, optimizer, nullcontext, args)

    before = task.trainable_module.linear.weight.detach().clone()
    losses = compiled_step(_batch())

    assert losses["loss"].isfinite()
    assert not torch.allclose(before, task.trainable_module.linear.weight.detach())


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="requires torch.compile")
def test_compiled_train_step_handles_grad_clip():
    from open_clip_train.train import _get_compiled_train_step

    task = TinyTask(TinyModel())
    optimizer = torch.optim.SGD(task.parameters(), lr=0.1)
    args = SimpleNamespace(torchcompile_backend="eager", torchcompile_mode=None, grad_clip_norm=0.01)
    compiled_step = _get_compiled_train_step(task, optimizer, nullcontext, args)

    losses = compiled_step(_batch())
    grad_norms = [
        param.grad.detach().norm(2)
        for param in task.trainable_module.parameters()
        if param.grad is not None
    ]
    total_norm = torch.linalg.vector_norm(torch.stack(grad_norms), ord=2)

    assert losses["loss"].isfinite()
    assert total_norm <= 0.011
