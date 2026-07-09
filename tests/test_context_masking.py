"""Cross-attention context masking (``context_valid`` -> ``context_attn_mask``).

Padding the context with extra (invalid) tokens must not change decoder outputs when the
padded positions are masked -- the property NaFlex-style padded token batches rely on.
Also checks ``context_valid=None`` keeps today's dense behavior, for every decoder family.
"""
import pytest
import torch

import open_clip

MAMMUT_MODELS = ['mammut2_ViT-B-32', 'mammut2-moderntext_ViT-B-32']
COCA_MODELS = ['coca2_ViT-B-32', 'coca2-moderntext_ViT-B-32']


def _pad_context(context, n_pad):
    b, n, d = context.shape
    pad = torch.randn(b, n_pad, d)  # garbage the mask must hide
    padded = torch.cat([context, pad], dim=1)
    valid = torch.zeros(b, n + n_pad, dtype=torch.bool)
    valid[:, :n] = True
    return padded, valid


@pytest.mark.parametrize('model_name', MAMMUT_MODELS)
def test_mammut_decoder_context_padding_invariant(model_name):
    torch.manual_seed(0)
    model = open_clip.create_model(model_name).eval()
    dec = model.text
    width = model.map_viz2txt_kv.shape[1]
    text = torch.randint(1, 400, (2, model.context_length))
    context = torch.randn(2, 7, width)
    padded, valid = _pad_context(context, n_pad=5)

    with torch.no_grad():
        ref = dec(text, context=context, mode='caption')
        dense_none = dec(text, context=context, context_valid=None, mode='caption')
        masked = dec(text, context=padded, context_valid=valid, mode='caption')
        unmasked = dec(text, context=padded, mode='caption')

    torch.testing.assert_close(dense_none, ref)                      # None == dense behavior
    torch.testing.assert_close(masked, ref, rtol=1e-4, atol=1e-5)    # padding hidden by mask
    assert not torch.allclose(unmasked, ref, rtol=1e-4, atol=1e-5)   # sanity: pad DOES leak unmasked


@pytest.mark.parametrize('model_name', COCA_MODELS)
def test_coca_decoder_context_padding_invariant(model_name):
    torch.manual_seed(0)
    model = open_clip.create_model(model_name).eval()
    dec = model.text_decoder
    width = dec.lm_head_params[0].shape[1]
    text_embs = torch.randn(2, 9, width)
    context = torch.randn(2, 7, width)
    padded, valid = _pad_context(context, n_pad=5)

    with torch.no_grad():
        ref = dec(context, text_embs)
        dense_none = dec(context, text_embs, context_valid=None)
        masked = dec(padded, text_embs, context_valid=valid)
        unmasked = dec(padded, text_embs)

    torch.testing.assert_close(dense_none, ref)
    torch.testing.assert_close(masked, ref, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(unmasked, ref, rtol=1e-4, atol=1e-5)


def test_context_masking_all_invalid_row_no_nan():
    """A fully-invalid context row (degenerate packed sample) must not NaN the outputs."""
    torch.manual_seed(0)
    model = open_clip.create_model('mammut2_ViT-B-32').eval()
    text = torch.randint(1, 400, (2, model.context_length))
    context = torch.randn(2, 7, model.map_viz2txt_kv.shape[1])
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[1] = False  # sample 1: no valid context at all
    with torch.no_grad():
        out = model.text(text, context=context, context_valid=valid, mode='caption')
    assert torch.isfinite(out).all()


def test_context_attn_mask_from_valid():
    from open_clip.transformer import context_attn_mask_from_valid
    assert context_attn_mask_from_valid(None) is None
    valid = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.int64)
    mask = context_attn_mask_from_valid(valid)
    assert mask.shape == (2, 1, 1, 3) and mask.dtype == torch.bool
    assert mask[0, 0, 0].tolist() == [True, True, False]
