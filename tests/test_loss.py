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
