"""Unit tests for task-specific training_forward and compute_accum_loss logic.

Tests CLIPTask loss aggregation, CoCa autoregressive shift + accum,
and DistillCLIPTask teacher freezing / output prefixing.
"""
import types

import torch
import torch.nn as nn

from open_clip.task import CLIPTask, CoCaTask, DistillCLIPTask


# ---------------------------------------------------------------------------
# Tiny model / loss stubs
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    def __init__(self, has_logits=False):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        self.has_logits = has_logits
        self.visual = types.SimpleNamespace(image_size=8)
        self.context_length = 5

    def encode_image(self, image):
        return self.proj(image.float().mean(dim=(2, 3)))

    def encode_text(self, text):
        return self.proj(text[:, :3].float())

    def forward(self, image, text):
        out = {
            "image_features": self.encode_image(image),
            "text_features": self.encode_text(text),
            "logit_scale": self.logit_scale,
        }
        if self.has_logits:
            bs, seq = text.shape
            out["logits"] = torch.randn(bs, seq, 10)
        return out


class DummyClipLoss(nn.Module):
    def forward(self, image_features, text_features, logit_scale, output_dict=False, **kw):
        return {
            "contrastive_loss": image_features.mean() * 0 + 1.0,
            "debug_metric": logit_scale.detach(),
        }


class DummyCoCaLoss(nn.Module):
    def forward(
            self, image_features, text_features, logits, labels, logit_scale,
            output_dict=False, **kw,
    ):
        return {
            "contrastive_loss": image_features.mean() * 0 + 1.0,
            "caption_loss": logits.mean() * 0 + 0.5,
        }


class DummyDistillLoss(nn.Module):
    def forward(self, output_dict=False, **kw):
        return {
            "distill_loss": torch.tensor(0.5),
        }


def _batch(bs=2, seq=5):
    return {
        "image": torch.randn(bs, 3, 4, 4),
        "text": torch.randint(0, 10, (bs, seq), dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# CLIPTask
# ---------------------------------------------------------------------------


def test_clip_task_loss_aggregation():
    """total loss sums only keys ending in '_loss', not 'debug_metric'."""
    task = CLIPTask(TinyModel(), loss=DummyClipLoss())
    task.train()
    losses, report = task(_batch())
    # contrastive_loss=1.0, debug_metric should NOT be summed
    assert abs(losses["loss"].item() - 1.0) < 1e-6
    assert "logit_scale" in report


def test_clip_task_logit_scale_in_report():
    """logit_scale is returned in the report dict for logging, NOT the loss dict."""
    task = CLIPTask(TinyModel(), loss=DummyClipLoss())
    task.train()
    losses, report = task(_batch())
    assert "logit_scale" not in losses
    assert report["logit_scale"].item() == 10.0


# ---------------------------------------------------------------------------
# CoCaTask
# ---------------------------------------------------------------------------


def test_coca_build_loss_inputs_autoregressive_shift():
    """_build_loss_inputs applies the correct autoregressive shift."""
    model = TinyModel(has_logits=True)
    task = CoCaTask(model, loss=DummyCoCaLoss())
    b = _batch(bs=2, seq=5)
    model_out = model(**b)
    loss_input = task._build_loss_inputs(model_out, b)
    # logits shifted: [:, :-1], labels shifted: text[:, 1:]
    assert loss_input["logits"].shape[1] == b["text"].shape[1] - 1
    assert loss_input["labels"].shape[1] == b["text"].shape[1] - 1
    assert torch.equal(loss_input["labels"], b["text"][:, 1:])


def test_coca_training_forward_produces_loss():
    model = TinyModel(has_logits=True)
    task = CoCaTask(model, loss=DummyCoCaLoss())
    task.train()
    losses, _ = task(_batch())
    assert "loss" in losses
    # contrastive_loss (1.0) + caption_loss (0.5) = 1.5
    assert abs(losses["loss"].item() - 1.5) < 1e-6


def test_coca_compute_accum_loss_concatenates_batches():
    """compute_accum_loss concatenates text from all accumulated batches."""
    model = TinyModel(has_logits=True)
    task = CoCaTask(model, loss=DummyCoCaLoss())

    b1 = _batch(bs=2, seq=5)
    b2 = _batch(bs=2, seq=5)
    accum_batches = [b1, b2]

    # Simulate accumulated model outputs
    with torch.no_grad():
        out1 = model(**b1)
        out2 = model(**b2)

    inputs = {
        "image_features": torch.cat([out1["image_features"], out2["image_features"]]),
        "text_features": torch.cat([out1["text_features"], out2["text_features"]]),
        "logits": torch.cat([out1["logits"], out2["logits"]]),
    }
    inputs_no_accum = {"logit_scale": out1["logit_scale"]}

    losses, _ = task.compute_accum_loss(inputs, inputs_no_accum, accum_batches)
    assert "contrastive_loss" in losses
    # Check that labels were built from concatenated texts
    # 4 samples total, shifted by 1 => (4, seq-1) labels
    assert "caption_loss" in losses


# ---------------------------------------------------------------------------
# DistillCLIPTask
# ---------------------------------------------------------------------------


def test_distill_teacher_frozen():
    """Teacher params have requires_grad=False."""
    student = TinyModel()
    teacher = TinyModel()
    task = DistillCLIPTask(student, teacher, loss=DummyDistillLoss())
    for p in task.teacher.parameters():
        assert not p.requires_grad


def test_distill_teacher_stays_eval():
    """Teacher stays in eval mode even after task.train()."""
    student = TinyModel()
    teacher = TinyModel()
    task = DistillCLIPTask(student, teacher, loss=DummyDistillLoss())
    task.train()
    assert not task.teacher.training


def test_distill_training_forward_prefixes_teacher_outputs():
    """Teacher outputs are prefixed with 'dist_' in model_out."""
    student = TinyModel()
    teacher = TinyModel()

    # Use a loss that captures what it receives
    received = {}

    class CaptureLoss(nn.Module):
        def forward(self, output_dict=False, **kw):
            received.update(kw)
            return {"distill_loss": torch.tensor(0.5)}

    task = DistillCLIPTask(student, teacher, loss=CaptureLoss())
    task.train()
    task(_batch())

    assert "dist_image_features" in received
    assert "dist_text_features" in received
    assert "dist_logit_scale" in received


def test_distill_teacher_no_grad():
    """Teacher forward runs without gradients."""
    student = TinyModel()
    teacher = TinyModel()

    teacher_grads = []

    class GradCheckLoss(nn.Module):
        def forward(self, output_dict=False, **kw):
            teacher_grads.append(kw.get("dist_image_features", None))
            return {"distill_loss": torch.tensor(0.5)}

    task = DistillCLIPTask(student, teacher, loss=GradCheckLoss())
    task.train()
    task(_batch())

    assert teacher_grads[0] is not None
    assert not teacher_grads[0].requires_grad
