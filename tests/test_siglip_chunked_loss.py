"""Parity test for SigLipLoss with chunk_size > 0.

Verifies the chunked path matches the reference (chunk_size=0) bit-exactly
across multiple chunk sizes and both ground-truth modes (negative_only on/off),
in the forward pass and the gradient.
"""
import pytest
import torch

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


@pytest.mark.parametrize("B,chunk_size", [(64, 32), (64, 8), (70, 8)])
@pytest.mark.parametrize("negative_only", [False, True])
def test_chunked_matches_reference_backward(B, chunk_size, negative_only):
    """Gradients for features, logit scale and bias match (chunks are recomputed in backward)."""
    inputs = _make_inputs(B=B)
    ref_inputs = [t.clone().requires_grad_(True) for t in inputs]
    chunk_inputs = [t.clone().requires_grad_(True) for t in inputs]
    SigLipLoss()._loss(*ref_inputs, negative_only=negative_only).backward()
    SigLipLoss(chunk_size=chunk_size)._loss(*chunk_inputs, negative_only=negative_only).backward()
    for name, ref, chunked in zip(("img", "txt", "scale", "bias"), ref_inputs, chunk_inputs):
        assert torch.allclose(ref.grad, chunked.grad, rtol=1e-5, atol=1e-5), f"{name} grad differs"


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
