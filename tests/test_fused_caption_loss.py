"""Fused caption loss (CoCa / MaMMUT) parity with the legacy logits path.

The fused path (model ``forward(labels=...)`` -> ``fused_linear_cross_entropy``) must produce
the same caption loss and the same gradients as the legacy path (materialized ``[B, L, V]``
logits -> ``CoCaLoss`` CE), for every decoder head variant (legacy Parameter heads, modern
Linear heads).
"""
import pytest
import torch
import torch.nn.functional as F

import open_clip
from open_clip.loss import fused_linear_cross_entropy

MODELS = [
    'coca_ViT-B-32',                # classic MultimodalTransformer head (text_projection Parameter)
    'coca2_ViT-B-32',               # coca2, classic text_arch
    'coca2-moderntext_ViT-B-32',    # coca2, modern text_arch (ModernMultimodalTransformer, nn.Linear head)
    'mammut2_ViT-B-32',              # classic MultimodalDecoder head (lm_head Parameter)
    'mammut2-moderntext_ViT-B-32',   # ModernMultimodalDecoder head (nn.Linear, maybe tied)
]


def _make_batch(model, batch_size=3, seed=0):
    torch.manual_seed(seed)
    ctx = model.context_length
    image_size = model.visual.image_size
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]
    image = torch.randn(batch_size, 3, image_size, image_size)
    text = torch.randint(1, 400, (batch_size, ctx))
    # right-padded captions of varying length
    text_valid = torch.zeros(batch_size, ctx, dtype=torch.bool)
    for i, n in enumerate((ctx, ctx * 2 // 3, 5)):
        text_valid[i, :n] = True
    text = text.masked_fill(~text_valid, getattr(model, 'pad_id', 0) or 0)
    labels = text[:, 1:].masked_fill(~text_valid[:, 1:], -100)
    return image, text, text_valid, labels


@pytest.mark.parametrize('model_name', MODELS)
def test_fused_caption_loss_matches_legacy(model_name):
    model = open_clip.create_model(model_name)
    model.train()
    image, text, text_valid, labels = _make_batch(model)

    # legacy: materialized logits, CE with the same shift/mask CoCaTask applies
    out_legacy = model(image=image, text=text, text_valid=text_valid)
    logits = out_legacy['logits'][:, :-1]
    loss_legacy = F.cross_entropy(
        logits.permute(0, 2, 1), labels, ignore_index=-100)

    # fused: labels into forward, model returns the reduced caption loss
    out_fused = model(image=image, text=text, text_valid=text_valid, labels=labels)
    assert 'logits' not in out_fused
    loss_fused = out_fused['caption_loss']

    torch.testing.assert_close(loss_fused, loss_legacy, rtol=1e-5, atol=1e-5)

    # gradient parity on a shared trunk parameter
    trunk_param = next(p for n, p in model.named_parameters() if 'visual' in n and p.dim() > 1)
    model.zero_grad()
    loss_legacy.backward(retain_graph=False)
    g_legacy = trunk_param.grad.clone()
    model.zero_grad()
    out_fused2 = model(image=image, text=text, text_valid=text_valid, labels=labels)
    out_fused2['caption_loss'].backward()
    torch.testing.assert_close(trunk_param.grad, g_legacy, rtol=1e-4, atol=1e-6)


def test_fused_linear_cross_entropy_chunking():
    """Chunked reduction must be exact regardless of chunk size (incl. ignored positions)."""
    torch.manual_seed(0)
    n, d, v = 100, 16, 50
    hidden = torch.randn(n, d, requires_grad=True)
    weight = torch.randn(v, d, requires_grad=True)
    target = torch.randint(0, v, (n,))
    target[::7] = -100
    ref = F.cross_entropy(hidden @ weight.t(), target, ignore_index=-100)
    for chunk in (7, 32, 1000):
        out = fused_linear_cross_entropy(hidden, weight, target, chunk_size=chunk)
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


def test_coca_loss_dual_mode():
    """CoCaLoss accepts either (logits, labels) or a precomputed caption_loss."""
    from open_clip.loss import CoCaLoss
    torch.manual_seed(0)
    loss_fn = CoCaLoss(caption_loss_weight=2.0, clip_loss_weight=1.0, pad_id=None)
    b, l, v, d = 4, 9, 32, 8
    img_f = F.normalize(torch.randn(b, d), dim=-1)
    txt_f = F.normalize(torch.randn(b, d), dim=-1)
    logits = torch.randn(b, l, v)
    labels = torch.randint(0, v, (b, l))
    scale = torch.tensor(10.0)

    legacy = loss_fn(img_f, txt_f, logits=logits, labels=labels, logit_scale=scale, output_dict=True)
    pre = F.cross_entropy(logits.permute(0, 2, 1), labels, ignore_index=-100)
    fused = loss_fn(img_f, txt_f, caption_loss=pre, logit_scale=scale, output_dict=True)
    torch.testing.assert_close(legacy['caption_loss'], fused['caption_loss'])
    torch.testing.assert_close(legacy['contrastive_loss'], fused['contrastive_loss'])
    # legacy positional call order still works
    positional = loss_fn(img_f, txt_f, logits, labels, scale, output_dict=True)
    torch.testing.assert_close(positional['caption_loss'], legacy['caption_loss'])
