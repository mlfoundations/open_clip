"""Numerical stability checks for the SigLIP pairwise loss."""

import pytest
import torch

from open_clip.loss import SigLipLoss


def _constant_logits_inputs(batch_size=256, dtype=torch.float16):
    """Construct finite, deliberately large pair logits without a large GEMM."""
    image = torch.zeros(batch_size, 4, dtype=dtype)
    text = torch.zeros(batch_size, 4, dtype=dtype)
    scale = torch.ones((), dtype=dtype)
    bias = torch.full((), 10, dtype=dtype)
    return image, text, scale, bias


@pytest.mark.parametrize("chunk_size", [0, 16])
def test_siglip_large_low_precision_reduction_is_finite(chunk_size):
    """The B x B reduction must not overflow when each pair loss is finite."""
    image, text, scale, bias = _constant_logits_inputs()
    loss = SigLipLoss(chunk_size=chunk_size)._loss(image, text, scale, bias)

    reference_image, reference_text, reference_scale, reference_bias = _constant_logits_inputs(
        dtype=torch.float32,
    )
    reference = SigLipLoss()._loss(
        reference_image,
        reference_text,
        reference_scale,
        reference_bias,
    )

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.allclose(loss, reference, rtol=2e-4, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_siglip_chunked_low_precision_gradients_are_finite(dtype):
    """The fp32 loss path should preserve a finite feature gradient under AMP dtypes."""
    generator = torch.Generator().manual_seed(17)
    image = torch.randn(96, 8, generator=generator, dtype=dtype, requires_grad=True)
    text = torch.randn(96, 8, generator=generator, dtype=dtype)
    scale = torch.tensor(10, dtype=dtype)
    bias = torch.tensor(10, dtype=dtype)

    loss = SigLipLoss(chunk_size=8)._loss(image, text, scale, bias)
    loss.backward()

    assert torch.isfinite(loss)
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()

