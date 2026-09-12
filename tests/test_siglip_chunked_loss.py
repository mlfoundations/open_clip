"""Numerical and training-memory tests for SigLipLoss with chunk_size > 0.

Verifies the chunked path matches the reference (chunk_size=0)
across multiple chunk sizes and both ground-truth modes (negative_only on/off),
in the forward pass and the gradient.
"""
import pytest
import torch
import torch.nn.functional as F

import open_clip.loss as loss_module
from open_clip.loss import SigLipLoss


def _make_inputs(B=64, D=32, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(B, D, generator=g, dtype=dtype)
    img = img / img.norm(dim=-1, keepdim=True)
    txt = torch.randn(B, D, generator=g, dtype=dtype)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    logit_scale = torch.tensor(10.0, dtype=dtype)
    logit_bias = torch.tensor(-10.0, dtype=dtype)
    return img, txt, logit_scale, logit_bias


@pytest.mark.parametrize("chunk_size", [64, 32, 8, 1])
@pytest.mark.parametrize("negative_only", [False, True])
def test_chunked_matches_reference_forward(chunk_size, negative_only):
    img, txt, ls, lb = _make_inputs(B=64)
    ref = SigLipLoss()._loss(img, txt, ls, lb, negative_only=negative_only)
    chunked = SigLipLoss(chunk_size=chunk_size)._loss(img, txt, ls, lb, negative_only=negative_only)
    assert torch.allclose(ref, chunked, atol=1e-5), \
        f"chunked (cs={chunk_size}, neg_only={negative_only}) differs: ref={ref.item()} chunk={chunked.item()}"


@pytest.mark.parametrize("chunk_size", [32, 8])
def test_chunked_matches_reference_backward(chunk_size):
    img, txt, ls, lb = _make_inputs(B=64)
    img1 = img.clone().requires_grad_(True)
    img2 = img.clone().requires_grad_(True)
    SigLipLoss()._loss(img1, txt, ls, lb).backward()
    SigLipLoss(chunk_size=chunk_size)._loss(img2, txt, ls, lb).backward()
    assert torch.allclose(img1.grad, img2.grad, atol=1e-5)


def test_chunked_handles_uneven_chunks():
    """Last chunk may be smaller than chunk_size; should not error."""
    img, txt, ls, lb = _make_inputs(B=70)  # 70 = 8*8 + 6, so last chunk has 6 rows
    ref = SigLipLoss()._loss(img, txt, ls, lb)
    chunked = SigLipLoss(chunk_size=8)._loss(img, txt, ls, lb)
    assert torch.allclose(ref, chunked, atol=1e-5)


def test_chunk_size_zero_skips_chunked_path():
    """Default behavior (chunk_size=0) must equal the original implementation."""
    img, txt, ls, lb = _make_inputs(B=32)
    ref = SigLipLoss()._loss(img, txt, ls, lb)
    explicit = SigLipLoss(chunk_size=0)._loss(img, txt, ls, lb)
    assert torch.equal(ref, explicit)


@pytest.mark.parametrize("batch_size,text_size", [(19, 19), (19, 11), (11, 19)])
@pytest.mark.parametrize("chunk_size", [1, 7, 32])
@pytest.mark.parametrize("negative_only", [False, True])
@pytest.mark.parametrize("trainable", [(True, True, True, True), (False, True, True, True),
                                     (True, False, True, True), (False, False, True, True)])
def test_chunked_all_gradients(batch_size, text_size, chunk_size, negative_only, trainable):
    """Chunk offsets and frozen towers preserve all feature and scalar gradients."""
    img, txt, ls, lb = _make_inputs(B=max(batch_size, text_size))
    inputs = (img[:batch_size], txt[:text_size], ls, lb)
    reference_inputs = [value.clone().requires_grad_(enabled) for value, enabled in zip(inputs, trainable)]
    chunked_inputs = [value.clone().requires_grad_(enabled) for value, enabled in zip(inputs, trainable)]
    ref_img, ref_txt, ref_scale, ref_bias = reference_inputs
    logits = (ref_scale * ref_img @ ref_txt.T + ref_bias).float()
    labels = -torch.ones_like(logits)
    if not negative_only:
        labels.diagonal().fill_(1)
    reference = -F.logsigmoid(labels * logits).sum() / batch_size
    actual = SigLipLoss(chunk_size=chunk_size)._loss(*chunked_inputs, negative_only=negative_only)
    ref_grads = torch.autograd.grad(reference, [value for value in reference_inputs if value.requires_grad])
    actual_grads = torch.autograd.grad(actual, [value for value in chunked_inputs if value.requires_grad])
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    for actual_grad, reference_grad in zip(actual_grads, ref_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_chunked_autocast_and_optional_bias(with_bias, autocast_enabled):
    """Recomputation preserves autocast and allows an omitted logit bias."""
    inputs = _make_inputs(B=35)
    inputs = inputs if with_bias else inputs[:3]
    reference_inputs = [value.clone().requires_grad_() for value in inputs]
    chunked_inputs = [value.clone().requires_grad_() for value in inputs]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        reference = SigLipLoss()._loss(*reference_inputs)
        actual = SigLipLoss(chunk_size=8)._loss(*chunked_inputs)
    ref_grads = torch.autograd.grad(reference, reference_inputs)
    actual_grads = torch.autograd.grad(actual, chunked_inputs)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    for actual_grad, reference_grad in zip(actual_grads, ref_grads):
        # Chunked bf16 matmuls accumulate feature and scalar gradients separately.
        torch.testing.assert_close(actual_grad, reference_grad, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_chunked_no_grad(requires_grad):
    """Evaluation keeps the same scalar and dictionary outputs without a graph."""
    inputs = [value.requires_grad_(requires_grad) for value in _make_inputs(B=19)]
    with torch.no_grad():
        actual = SigLipLoss(chunk_size=7)(*inputs, output_dict=True)["contrastive_loss"]
        reference = SigLipLoss()(*inputs)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("trainable", [(True, True, True, True), (False, False, True, True),
                                     (False, False, True, False), (False, False, False, True)])
def test_chunked_saved_activation_memory(trainable):
    """Backward must not retain every chunk's pairwise activation matrix."""
    batch_size, feature_dim, chunk_size = 256, 16, 16
    inputs = [value.requires_grad_(enabled)
              for value, enabled in zip(_make_inputs(B=batch_size, D=feature_dim), trainable)]
    input_storage = {value.untyped_storage().data_ptr() for value in inputs}
    saved_storage = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in input_storage:
            saved_storage[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = SigLipLoss(chunk_size=chunk_size)._loss(*inputs)

    # Count unique storage, not tensor numel: feature views share the input storage.
    # Allow feature-sized intermediates and two chunks, but not the full B x B matrix.
    memory_budget = 2 * (batch_size * feature_dim + chunk_size * batch_size) * inputs[0].element_size()
    saved_bytes = sum(saved_storage.values())
    assert saved_bytes <= memory_budget, f"saved {saved_bytes} activation bytes; budget is {memory_budget}"
    gradients = torch.autograd.grad(loss, [value for value in inputs if value.requires_grad])
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("dist_impl", ["gather", "reduce", "shift", "bidir"])
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_chunked_distributed_blocks(monkeypatch, dist_impl, world_size):
    """Remote negative blocks keep all gradients without replaying communication.

    Supply differentiable remote features directly so this test does not require
    a distributed backend. Real collectives and their backward are not simulated.
    """
    image, text, scale, bias = _make_inputs(B=7, D=5)
    texts = [text + 0.1 * rank for rank in range(world_size)]

    def run(chunk_size):
        inputs = [value.clone().requires_grad_() for value in (image, scale, bias, *texts)]
        img, ls, lb, *all_text = inputs
        calls = []
        remote = iter(all_text[1:])

        def gather(tensor):
            calls.append("gather")
            return torch.cat(all_text)

        def reduce(tensor, operation):
            index = len(calls)
            calls.append("reduce")
            return all_text[index]

        def exchange(*args):
            calls.append("exchange")
            return next(remote)

        def exchange_bidir(*args):
            calls.append("exchange_bidir")
            return next(remote), next(remote)

        monkeypatch.setattr(loss_module, "_all_gather_with_grad", gather)
        monkeypatch.setattr(torch.distributed.nn, "all_reduce", reduce)
        monkeypatch.setattr(loss_module, "neighbour_exchange_with_grad", exchange)
        monkeypatch.setattr(loss_module, "neighbour_exchange_bidir_with_grad", exchange_bidir)
        loss = SigLipLoss(rank=0, world_size=world_size, dist_impl=dist_impl, chunk_size=chunk_size)(
            img, all_text[0], ls, lb)
        forward_calls = list(calls)
        gradients = torch.autograd.grad(loss, inputs)
        assert calls == forward_calls
        return loss, gradients

    reference, reference_grads = run(0)
    actual, actual_grads = run(3)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-5)
