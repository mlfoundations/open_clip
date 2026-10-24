from types import SimpleNamespace

import pytest
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


@pytest.mark.parametrize(("world_size", "local_loss"), [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("train_features", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clip_logit_bias_applied_once(monkeypatch, world_size, local_loss, train_features, dtype):
    image = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, requires_grad=train_features)
    text = torch.tensor([[2.0, 1.0], [4.0, 3.0]], dtype=dtype, requires_grad=train_features)
    bias = torch.tensor(-10.0, dtype=dtype, requires_grad=True)
    scale = torch.tensor(2.0, dtype=dtype)

    def gathered_features(image_features, text_features, **kwargs):
        return torch.cat([image_features, image_features + 1]), torch.cat([text_features, text_features - 1])

    monkeypatch.setattr(loss_module, 'gather_features', gathered_features)
    if world_size > 1:
        all_image, all_text = gathered_features(image, text)
        image_source = image if local_loss else all_image
        text_source = text if local_loss else all_text
        expected_image = scale * image_source @ all_text.T + bias
        expected_text = scale * text_source @ all_image.T + bias
    else:
        expected_image = scale * image @ text.T + bias
        expected_text = scale * text @ image.T + bias

    actual_image, actual_text = loss_module.ClipLoss(world_size=world_size, local_loss=local_loss).get_logits(
        image, text, scale, bias)
    torch.testing.assert_close(actual_image, expected_image)
    torch.testing.assert_close(actual_text, expected_text)
    inputs = (bias, image, text) if train_features else (bias,)
    actual_grads = torch.autograd.grad(actual_image.sum() + actual_text.sum(), inputs)
    expected_grads = torch.autograd.grad(expected_image.sum() + expected_text.sum(), inputs)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected)
