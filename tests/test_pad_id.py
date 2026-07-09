"""pad_id consistency across generative text models.

The chain under test: tokenizer fills -> text tower masks (tower owns pad_id) -> model derives
model.pad_id from its tower -> tasks/eval use it only as the fallback validity derivation when no
text_valid mask is supplied (labels are -100 masked; generation reads it as the fill default).
"""
import types

import pytest
import torch

import open_clip
from open_clip.coca_model import CoCa
from open_clip.factory import _validate_special_tokens


def _coca_cfg(pad_id=None):
    cfg = open_clip.get_model_config('coca_ViT-B-32')
    if pad_id is not None:
        cfg['text_cfg']['pad_id'] = pad_id
    return cfg


def test_coca_pad_id_derived_from_text_tower():
    """CoCa.pad_id follows the tower's masking pad id (text_cfg.pad_id), not a separate ctor value."""
    cfg = _coca_cfg(pad_id=7)
    model = CoCa(
        embed_dim=cfg['embed_dim'],
        multimodal_cfg=cfg['multimodal_cfg'],
        text_cfg=cfg['text_cfg'],
        vision_cfg=cfg['vision_cfg'],
    )
    assert model.text.pad_id == 7
    assert model.pad_id == 7

    args = types.SimpleNamespace(
        model='coca_ViT-B-32', distill=False, siglip=False, local_loss=False,
        gather_with_grad=False, rank=0, world_size=1,
        coca_caption_loss_weight=1.0, coca_contrastive_loss_weight=1.0,
        loss_dist_impl=None, horovod=False,
    )
    task = open_clip.create_task(args, model=model)
    assert task.pad_id == 7  # fallback label masking follows the model pad id
    assert task.loss.pad_id is None  # task path: labels pre-masked to -100
    text = torch.tensor([[3, 5, 7, 7]])
    assert task._caption_labels(text).tolist() == [[5, -100, -100]]
    # standalone loss factory keeps the legacy value-based ignore, keyed to the model pad id
    loss = open_clip.create_loss(args, model=model)
    assert loss.pad_id == 7


def test_coca_pad_id_default_convention():
    """Default (SimpleTokenizer-style) config keeps the historical 0 fill convention."""
    cfg = _coca_cfg()
    model = CoCa(
        embed_dim=cfg['embed_dim'],
        multimodal_cfg=cfg['multimodal_cfg'],
        text_cfg=cfg['text_cfg'],
        vision_cfg=cfg['vision_cfg'],
    )
    assert model.pad_id == 0


class _StubTokenizer:
    def __init__(self, pad_token_id=None):
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id


def test_validate_special_tokens_generative_pad_rules():
    """Generative pad rules: unset pad_id vs a reserved nonzero tokenizer pad warns (fallback-path
    drift; the training pipeline supplies attention masks); explicit nonzero pad_id with a padless
    tokenizer raises (fill and mask would disagree outright)."""
    tok = _StubTokenizer(pad_token_id=1)
    with pytest.warns(UserWarning, match='pad-value fallback'):
        _validate_special_tokens({'pool_type': 'argmax'}, tok, generative=True)
    # explicit and matching: fine
    _validate_special_tokens({'pool_type': 'argmax', 'pad_id': 1}, tok, generative=True)
    # explicit and mismatched: existing rule fires regardless of generative
    with pytest.raises(ValueError, match='does not match'):
        _validate_special_tokens({'pool_type': 'argmax', 'pad_id': 0}, tok, generative=True)
    # explicit nonzero pad but the tokenizer has no reserved pad (fills with 0): config error
    with pytest.raises(ValueError, match='no reserved pad'):
        _validate_special_tokens({'pool_type': 'argmax', 'pad_id': 7}, _StubTokenizer(), generative=True)
    # same drift on a non-generative config: allowed (towers may not consume pad at all)
    _validate_special_tokens({'pool_type': 'argmax', 'pad_id': 7}, _StubTokenizer(), generative=False)
    # non-generative (e.g. SigLIP w/ t5 pad=1, towers don't consume pad): exempt
    _validate_special_tokens({'pool_type': 'argmax'}, tok, generative=False)
    # SimpleTokenizer-style (no reserved pad, unset cfg pad): exempt
    _validate_special_tokens({'pool_type': 'argmax'}, _StubTokenizer(), generative=True)
    # tokenizer whose reserved pad IS 0: unset default agrees, exempt
    _validate_special_tokens({'pool_type': 'argmax'}, _StubTokenizer(pad_token_id=0), generative=True)


def test_get_tokenizer_generative_validation_builtin_configs():
    """Shipped generative configs resolve cleanly under the stricter validation."""
    for name in ('coca_ViT-B-32', 'mammut2_ViT-B-32', 'mammut2-moderntext_ViT-B-32'):
        tokenizer = open_clip.get_tokenizer(name)
        assert tokenizer is not None


# ---- text_valid mask interface ----

def test_simple_tokenizer_output_mask_length_exact():
    """SimpleTokenizer masks are length-derived: mid-caption id-0 tokens ('x!=y' emits id 0)
    stay valid, only fill positions are masked."""
    from open_clip.tokenizer import SimpleTokenizer

    tok = SimpleTokenizer()
    tokens, mask = tok(['x!=y'], output_mask=True)
    assert 0 in tokens[0][mask[0]].tolist()  # genuine id-0 token inside the valid region
    # valid region = sot..eot inclusive, everything after masked
    eot_pos = (tokens[0] == tok.eot_token_id).nonzero()[0, 0].item()
    assert mask[0][:eot_pos + 1].all()
    assert not mask[0][eot_pos + 1:].any()
    # value-derived mask would differ (treats the mid-caption 0 as pad)
    assert not torch.equal(mask[0], tokens[0] != 0)


def test_tiktoken_tokenizer_output_mask():
    from open_clip.tokenizer import TikTokenTokenizer

    tok = TikTokenTokenizer(context_length=16)
    tokens, mask = tok(['hello world'], output_mask=True)
    assert torch.equal(mask, tokens != tok.pad_token_id)  # reserved pad: value-derivation exact


def test_siglip_tokenizer_output_mask_unsupported():
    from open_clip.tokenizer import SigLipTokenizer

    # the guard fires before any tokenizer state is touched; skip the heavy (hub-dependent) __init__
    tok = SigLipTokenizer.__new__(SigLipTokenizer)
    with pytest.raises(NotImplementedError):
        tok(['hi'], output_mask=True)


def _tiny_mammut(**overrides):
    from open_clip.mammut_model import MaMMUT

    mm_cfg = dict(context_length=16, vocab_size=64, width=64, heads=2, layers=2, cross_attn_ratio=2)
    mm_cfg.update(overrides)
    vis_cfg = dict(
        image_size=64, layers=2, width=64, patch_size=32,
        output_tokens=True, pool_type='avg', final_ln_after_pool=True,
    )
    torch.manual_seed(0)
    return MaMMUT(embed_dim=64, multimodal_cfg=mm_cfg, vision_cfg=vis_cfg).eval()


@pytest.mark.parametrize('modern', [False, True], ids=['classic', 'modern'])
def test_mammut_text_valid_semantics(modern):
    overrides = dict(text_arch='modern') if modern else {}
    model = _tiny_mammut(**overrides)
    text = torch.tensor([[1, 5, 0, 7, 2, 0, 0, 0]])  # genuine id-0 at pos 2, fill at 5..7
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        no_mask = model.encode_text(text)
        pad_derived = model.encode_text(text, text_valid=(text != 0))
        true_mask = model.encode_text(text, text_valid=mask)
    # backward compat: absent mask == pad-value fallback
    assert torch.allclose(no_mask, pad_derived)
    # the true mask attends/pools the genuine id-0 token, changing the feature
    assert not torch.allclose(no_mask, true_mask)


def test_mammut_legacy_mode_ignores_text_valid():
    """Legacy openMaMMUT flags never consult validity: supplied masks must be no-ops (bit-parity)."""
    model = _tiny_mammut(pool_type='avg_all', use_pad_mask=False)
    text = torch.tensor([[1, 5, 0, 7, 2, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        assert torch.equal(model.encode_text(text), model.encode_text(text, text_valid=mask))


def test_coca_text_valid_semantics():
    cfg = _coca_cfg()
    cfg['text_cfg'].update(width=64, heads=2, layers=2)
    cfg['vision_cfg'].update(width=64, layers=2, image_size=64, patch_size=32)
    cfg['multimodal_cfg'].update(width=64, heads=2, layers=2)
    torch.manual_seed(0)
    model = CoCa(embed_dim=64, multimodal_cfg=cfg['multimodal_cfg'],
                 text_cfg=cfg['text_cfg'], vision_cfg=cfg['vision_cfg']).eval()
    text = torch.tensor([[1, 5, 0, 7, 2, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        no_mask = model.encode_text(text)
        pad_derived = model.encode_text(text, text_valid=(text != 0))
        true_mask = model.encode_text(text, text_valid=mask)
    assert torch.allclose(no_mask, pad_derived)
    assert not torch.allclose(no_mask, true_mask)


def test_coca_forward_intermediates_text_valid():
    """forward_intermediates honors text_valid like forward (mask changes pooled features)."""
    cfg = _coca_cfg()
    cfg['text_cfg'].update(width=64, heads=2, layers=2)
    cfg['vision_cfg'].update(width=64, layers=2, image_size=64, patch_size=32)
    cfg['multimodal_cfg'].update(width=64, heads=2, layers=2)
    torch.manual_seed(0)
    model = CoCa(embed_dim=64, multimodal_cfg=cfg['multimodal_cfg'],
                 text_cfg=cfg['text_cfg'], vision_cfg=cfg['vision_cfg']).eval()
    text = torch.tensor([[1, 5, 0, 7, 2, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        out_fallback = model.forward_intermediates(text=text)
        out_masked = model.forward_intermediates(text=text, text_valid=mask)
        fwd_masked = model.encode_text(text, text_valid=mask)
    assert not torch.allclose(out_fallback['text_features'], out_masked['text_features'])
    assert torch.allclose(out_masked['text_features'], fwd_masked)


def test_modern_mammut_no_frozen_embedding_row():
    """The modern decoder must not zero-freeze the pad row (SimpleTokenizer id 0 is a real token)."""
    model = _tiny_mammut(text_arch='modern')
    emb = model.text.token_embedding
    assert emb.padding_idx is None
    assert emb.weight[0].abs().sum() > 0
    model.train()
    img = torch.randn(1, 3, 64, 64)
    text = torch.tensor([[1, 0, 5, 2]])
    mask = torch.ones_like(text, dtype=torch.bool)
    out = model(img, text, text_valid=mask)
    out['logits'].sum().backward()
    assert emb.weight.grad[0].abs().sum() > 0  # id-0 embedding receives gradient


def test_tokenize_text_map_sample():
    from open_clip.tokenizer import SimpleTokenizer
    from open_clip_train.data import TokenizeText

    tt = TokenizeText(SimpleTokenizer(), output_mask=True)
    sample = tt.map_sample({'text': b'a dog', 'image': None})
    assert sample['text'].shape == sample['text_valid'].shape
    assert sample['text_valid'].dtype == torch.bool
    # mask off: no extra key, value identical to the plain path
    tt_plain = TokenizeText(SimpleTokenizer())
    sample_plain = tt_plain.map_sample({'text': 'a dog'})
    assert 'text_valid' not in sample_plain


def test_use_pad_mask_ignored_warnings():
    """Config values that towers accept but do not honor must warn rather than silently no-op."""
    import warnings as _warnings
    from open_clip.transformer import TextTransformer

    # causal tower without embed_cls: use_pad_mask is truly dropped -> warn
    with pytest.warns(UserWarning, match='use_pad_mask'):
        TextTransformer(context_length=8, vocab_size=32, width=32, heads=2, layers=1, use_pad_mask=True)
    # bidirectional: honored, no warning
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        TextTransformer(context_length=8, vocab_size=32, width=32, heads=2, layers=1,
                        use_pad_mask=True, no_causal_mask=True, pool_type='first')
    # embed_cls (CoCa): the cls-additive mask includes pad masking regardless -> no warning
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        TextTransformer(context_length=8, vocab_size=32, width=32, heads=2, layers=1,
                        use_pad_mask=True, embed_cls=True)

    # modern MaMMUT decoder ignores use_pad_mask=False (no legacy modern weights) -> warn
    with pytest.warns(UserWarning, match='use_pad_mask'):
        _tiny_mammut(text_arch='modern', use_pad_mask=False)
    # classic decoder honors it (legacy openMaMMUT mode): no warning
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        _tiny_mammut(pool_type='avg_all', use_pad_mask=False)


def test_maybe_compute_generative_loss_respects_pad_id():
    from open_clip_train.train import maybe_compute_generative_loss

    torch.manual_seed(0)
    vocab, pad = 16, 5
    logits = torch.randn(2, 8, vocab)
    texts = torch.randint(0, vocab, (2, 8))
    texts[:, -3:] = pad  # trailing padding
    model_out = {'logits': logits}

    loss_pad_aware = maybe_compute_generative_loss(model_out, texts=texts, pad_id=pad)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].permute(0, 2, 1), texts[:, 1:], ignore_index=pad)
    assert torch.allclose(loss_pad_aware, expected)
    # and it differs from the pad-unaware value (pad labels would otherwise contribute)
    loss_default = maybe_compute_generative_loss(model_out, texts=texts, pad_id=0)
    assert not torch.allclose(loss_pad_aware, loss_default)
    # an explicit attention mask takes precedence over the pad-value fallback
    mask = torch.ones_like(texts, dtype=torch.bool)
    mask[:, -3:] = False
    loss_masked = maybe_compute_generative_loss(model_out, texts=texts, text_valid=mask, pad_id=0)
    assert torch.allclose(loss_masked, loss_pad_aware)  # same positions masked here


@pytest.mark.skipif(
    not pytest.importorskip('transformers', reason='transformers required'),
    reason='transformers required',
)
def test_coca_roberta_pad_id():
    """HF tower: model.pad_id derives from the transformers config pad (roberta=1), and the
    config's explicit pad_id passes tokenizer validation."""
    tokenizer = open_clip.get_tokenizer('coca_roberta-ViT-B-32')
    assert tokenizer.pad_token_id == 1
    model = open_clip.create_model('coca_roberta-ViT-B-32', pretrained_hf=False)
    assert model.text.pad_id == 1
    assert model.pad_id == 1

    args = types.SimpleNamespace(
        model='coca_roberta-ViT-B-32', distill=False, siglip=False, local_loss=False,
        gather_with_grad=False, rank=0, world_size=1,
        coca_caption_loss_weight=1.0, coca_contrastive_loss_weight=1.0,
        loss_dist_impl=None, horovod=False,
    )
    task = open_clip.create_task(args, model=model)
    assert task.pad_id == 1  # fallback label masking follows the roberta pad id
    text = torch.tensor([[0, 5, 2, 1]])  # bos, token, eos, pad
    assert task._caption_labels(text).tolist() == [[5, 2, -100]]
