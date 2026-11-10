from types import SimpleNamespace

import torch

from open_clip import loss as loss_module


def test_all_gather_with_grad_uses_functional_collective(monkeypatch):
    calls = []

    def fake_all_gather_tensor_autograd(tensor, gather_dim, group):
        calls.append((tensor, gather_dim, group))
        return torch.cat([tensor, tensor + 1], dim=0)

    monkeypatch.setattr(
        loss_module,
        "dist_fcol",
        SimpleNamespace(all_gather_tensor_autograd=fake_all_gather_tensor_autograd),
    )

    x = torch.randn(2, 3).t()
    out = loss_module._all_gather_with_grad(x)

    assert out.shape == (6, 2)
    assert calls[0][0].is_contiguous()
    assert calls[0][1] == 0


def test_all_gather_with_grad_falls_back_to_distributed_nn(monkeypatch):
    monkeypatch.setattr(loss_module, "dist_fcol", None)
    monkeypatch.setattr(
        torch.distributed.nn,
        "all_gather",
        lambda tensor: (tensor, tensor + 1),
    )

    x = torch.randn(2, 3)
    out = loss_module._all_gather_with_grad(x)

    assert torch.equal(out, torch.cat([x, x + 1], dim=0))


def test_distill_clip_loss_with_logit_bias():
    loss_fn = loss_module.DistillClipLoss()
    kwargs = {
        "image_features": torch.randn(4, 8),
        "text_features": torch.randn(4, 8),
        "logit_scale": torch.tensor(1.0),
        "logit_bias": torch.tensor(0.5),
        "dist_image_features": torch.randn(4, 8),
        "dist_text_features": torch.randn(4, 8),
        "dist_logit_scale": torch.tensor(1.0),
        "dist_logit_bias": torch.tensor(0.2),
    }
    losses = loss_fn(**kwargs, output_dict=True)
    assert "contrastive_loss" in losses
    assert "distill_loss" in losses
    assert not torch.isnan(losses["contrastive_loss"])
    assert not torch.isnan(losses["distill_loss"])
