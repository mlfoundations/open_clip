"""MaMMUT model tests: two-pass behaviour, pooling correctness vs legacy compat,
LAION fork checkpoint conversion, factory integration, and the modern decoder arch."""
import os

import pytest
import torch

import open_clip
from open_clip.factory import _convert_legacy_mammut_cfg, load_checkpoint
from open_clip.mammut_model import MaMMUT
from open_clip.transformer import ModernMultimodalDecoder, MultimodalDecoder

os.environ['CUDA_VISIBLE_DEVICES'] = ''

MULTIMODAL_CFG = dict(
    context_length=16,
    vocab_size=64,
    width=64,
    heads=2,
    layers=4,
    cross_attn_ratio=2,
)
LEGACY_MULTIMODAL_CFG = dict(
    MULTIMODAL_CFG,
    pool_type='avg_all',
    use_pad_mask=False,
)
MODERN_MULTIMODAL_CFG = dict(
    MULTIMODAL_CFG,
    text_arch='modern',
    attn_gated=True,
    value_residual=True,
    qk_norm=True,
)
VISION_CFG = dict(
    image_size=64,
    layers=2,
    width=64,
    patch_size=32,
    output_tokens=True,
    pool_type='avg',
    final_ln_after_pool=True,
)

ARCH_CFGS = [MULTIMODAL_CFG, MODERN_MULTIMODAL_CFG]
ARCH_IDS = ['classic', 'modern']


def _tiny_model(multimodal_cfg=MULTIMODAL_CFG, seed=0):
    torch.manual_seed(seed)
    return MaMMUT(embed_dim=64, multimodal_cfg=multimodal_cfg, vision_cfg=VISION_CFG).eval()


def _tiny_batch(seed=0, batch_size=2, seq_len=16, num_pad=4):
    torch.manual_seed(seed)
    image = torch.randn(batch_size, 3, 64, 64)
    text = torch.randint(3, 64, (batch_size, seq_len))
    if num_pad:
        text[:, -num_pad:] = 0
    return image, text


@pytest.mark.parametrize('mm_cfg', ARCH_CFGS, ids=ARCH_IDS)
def test_mammut_output_contract(mm_cfg):
    """Forward returns the CoCa-style dict with full-length caption logits."""
    model = _tiny_model(mm_cfg)
    image, text = _tiny_batch()
    with torch.no_grad():
        out = model(image, text)
    assert set(out.keys()) == {'image_features', 'text_features', 'logits', 'logit_scale'}
    assert out['image_features'].shape == (2, 64)
    assert out['text_features'].shape == (2, 64)
    assert out['logits'].shape == (2, 16, 64)  # full length, task applies the AR shift


@pytest.mark.parametrize('mm_cfg', ARCH_CFGS, ids=ARCH_IDS)
def test_mammut_contrastive_pass_image_independent(mm_cfg):
    """The contrastive text pass must not see the image (no cross-attention)."""
    model = _tiny_model(mm_cfg)
    image, text = _tiny_batch()
    other_image = torch.randn_like(image)
    with torch.no_grad():
        out1 = model(image, text)
        out2 = model(other_image, text)
        encoded = model.encode_text(text)
    assert torch.allclose(out1['text_features'], out2['text_features'])
    assert torch.allclose(out1['text_features'], encoded)
    # while the caption pass does cross-attend
    assert not torch.allclose(out1['logits'], out2['logits'])


@pytest.mark.parametrize('mm_cfg', ARCH_CFGS, ids=ARCH_IDS)
def test_mammut_decoder_mode_is_explicit(mm_cfg):
    """Pass selection is never inferred from image_embs presence."""
    model = _tiny_model(mm_cfg)
    _, text = _tiny_batch()
    image_kv = torch.randn(2, 5, 64)
    with pytest.raises(AssertionError):
        model.text(text, context=image_kv, mode='contrastive')
    with pytest.raises(AssertionError):
        model.text(text, mode='caption')
    with pytest.raises(ValueError):
        model.text(text, mode='bogus')


@pytest.mark.parametrize('mm_cfg', ARCH_CFGS, ids=ARCH_IDS)
def test_mammut_text_features_pad_invariant(mm_cfg):
    """Default config: pads are excluded from pooling and attention, so trailing
    pad length must not change the text embedding."""
    model = _tiny_model(mm_cfg)
    _, text = _tiny_batch(num_pad=8)
    truncated = text[:, :12]  # same real tokens, 4 fewer trailing pads
    with torch.no_grad():
        full = model.encode_text(text)
        short = model.encode_text(truncated)
    assert torch.allclose(full, short, atol=1e-5)


@pytest.mark.parametrize('mm_cfg', ARCH_CFGS, ids=ARCH_IDS)
def test_mammut_grad_checkpointing(mm_cfg):
    model = _tiny_model(mm_cfg)
    model.set_grad_checkpointing(True)
    model.train()
    image, text = _tiny_batch()
    out = model(image, text)
    (out['logits'].sum() + out['text_features'].sum()).backward()
    assert model.map_viz2txt_kv.grad is not None


@pytest.mark.parametrize('mm_cfg', ARCH_CFGS, ids=ARCH_IDS)
def test_mammut_task_training_forward(mm_cfg):
    """End-to-end loss wiring through CoCaTask (contrastive + caption, AR shift)."""
    from open_clip.task import CoCaTask

    model = _tiny_model(mm_cfg)
    task = CoCaTask(model, caption_loss_weight=1.0, clip_loss_weight=1.0, verbose=False)
    image, text = _tiny_batch()
    losses, report = task.training_forward({'image': image, 'text': text})
    assert set(losses.keys()) >= {'contrastive_loss', 'caption_loss', 'loss'}
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()


@pytest.mark.parametrize('mm_cfg', ARCH_CFGS, ids=ARCH_IDS)
def test_mammut_generate(mm_cfg):
    pytest.importorskip('transformers')
    model = _tiny_model(mm_cfg)
    image, _ = _tiny_batch()
    out = model.generate(
        image,
        generation_type='top_k',
        seq_len=8,
        min_seq_len=3,
        sot_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    assert out.shape[0] == 2
    assert out.shape[1] <= 8
    assert out.dtype == torch.long


def test_mammut_generate_uses_resolved_cfg_special_ids():
    pytest.importorskip('transformers')
    model = _tiny_model(dict(MULTIMODAL_CFG, bos_id=1, eos_id=2))
    image, _ = _tiny_batch()
    out = model.generate(
        image,
        generation_type='top_k',
        seq_len=8,
        min_seq_len=3,
    )
    assert out.shape[0] == 2
    assert out.shape[1] <= 8
    assert torch.equal(out[:, 0], torch.full_like(out[:, 0], 1))


def test_mammut_legacy_pooling_pad_sensitive():
    """Legacy flags (avg_all, no pad mask) must keep reproducing the original
    (pad-contaminated) behaviour for released openMaMMUT weight compat."""
    model = _tiny_model(LEGACY_MULTIMODAL_CFG)
    _, text = _tiny_batch(num_pad=8)
    truncated = text[:, :12]
    with torch.no_grad():
        full = model.encode_text(text)
        short = model.encode_text(truncated)
    assert not torch.allclose(full, short, atol=1e-5)


def test_mammut_legacy_matches_fork_computation():
    """Legacy contrastive pass == the LAION fork's math: bare resblocks (no mask),
    ln_final, unmasked mean over all positions, normalize."""
    model = _tiny_model(LEGACY_MULTIMODAL_CFG)
    _, text = _tiny_batch()
    with torch.no_grad():
        expected = model.text.token_embedding(text) + model.text.positional_embedding[:text.shape[1]]
        for block in model.text.resblocks:
            expected = block(expected)
        expected = torch.nn.functional.normalize(model.text.ln_final(expected).mean(1), dim=-1)
        actual = model.encode_text(text)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_mammut_fork_state_dict_conversion(tmp_path):
    """A LAION fork checkpoint (text.text_projection as vocab head) loads strict
    and reproduces outputs after the lm_head rename."""
    model = _tiny_model(LEGACY_MULTIMODAL_CFG, seed=42)
    image, text = _tiny_batch()
    fork_sd = {
        ('text.text_projection' if k == 'text.lm_head' else k): v
        for k, v in model.state_dict().items()
    }
    checkpoint_path = tmp_path / 'fork_mammut.pt'
    torch.save(fork_sd, checkpoint_path)

    reloaded = _tiny_model(LEGACY_MULTIMODAL_CFG, seed=7)
    load_checkpoint(reloaded, str(checkpoint_path), strict=True)
    with torch.no_grad():
        out1 = model(image, text)
        out2 = reloaded(image, text)
    assert torch.allclose(out1['text_features'], out2['text_features'])
    assert torch.allclose(out1['logits'], out2['logits'])


def test_mammut_new_format_proj_checkpoint_not_misconverted(tmp_path):
    """A new-format classic checkpoint w/ proj_type='linear' holds BOTH text.lm_head and a real
    text.text_projection -- it must not be mistaken for a fork checkpoint (which would clobber
    lm_head with the projection)."""
    torch.manual_seed(42)
    cfg = dict(MULTIMODAL_CFG, proj_type='linear')
    model = MaMMUT(embed_dim=32, multimodal_cfg=cfg, vision_cfg=VISION_CFG).eval()
    assert 'text.text_projection' in model.state_dict() and 'text.lm_head' in model.state_dict()
    checkpoint_path = tmp_path / 'new_format_mammut.pt'
    torch.save(model.state_dict(), checkpoint_path)

    torch.manual_seed(7)
    reloaded = MaMMUT(embed_dim=32, multimodal_cfg=cfg, vision_cfg=VISION_CFG).eval()
    load_checkpoint(reloaded, str(checkpoint_path), strict=True)
    image, text = _tiny_batch()
    with torch.no_grad():
        out1 = model(image, text)
        out2 = reloaded(image, text)
    assert torch.allclose(out1['text_features'], out2['text_features'])
    assert torch.allclose(out1['logits'], out2['logits'])


def test_mammut_create_loss_pad_id():
    """Standalone create_loss() keys the legacy value-based label ignore to the model instance's
    pad id (config-by-name resolution is deliberately avoided); the underlying CE uses -100."""
    import types
    from open_clip import create_loss
    from open_clip.loss import CoCaLoss

    args = types.SimpleNamespace(
        model='mammut_tiny', distill=False, siglip=False, local_loss=False,
        gather_with_grad=False, rank=0, world_size=1,
        coca_caption_loss_weight=1.0, coca_contrastive_loss_weight=1.0,
        horovod=False,
    )
    model = _tiny_model(dict(MULTIMODAL_CFG, pad_id=1))
    loss = create_loss(args, model=model)
    assert isinstance(loss, CoCaLoss)
    assert loss.pad_id == 1  # raw labels equal to the model pad id are ignored
    assert loss.caption_loss.ignore_index == -100
    # without a model instance, falls back to the historical default fill id and name dispatch
    fallback = create_loss(args)
    assert isinstance(fallback, CoCaLoss)
    assert fallback.pad_id == 0


def test_mammut_proj_type_none_requires_matching_dims():
    with pytest.raises(ValueError, match='embed_dim'):
        MaMMUT(embed_dim=32, multimodal_cfg=MULTIMODAL_CFG, vision_cfg=VISION_CFG)


def test_mammut_proj_type_linear_decouples_dims():
    torch.manual_seed(0)
    model = MaMMUT(
        embed_dim=32,
        multimodal_cfg=dict(MULTIMODAL_CFG, proj_type='linear'),
        vision_cfg=dict(VISION_CFG),
    ).eval()
    image, text = _tiny_batch()
    with torch.no_grad():
        out = model(image, text)
    assert out['text_features'].shape == (2, 32)
    assert out['image_features'].shape == (2, 32)
    assert out['logits'].shape == (2, 16, 64)


def test_mammut_cross_attn_interleave():
    model = _tiny_model()
    assert model.text.cross_step == 2
    assert len(model.text.cross_attn) == 2  # after layers 0 and 2 of 4
    dense = MaMMUT(
        embed_dim=64,
        multimodal_cfg=dict(MULTIMODAL_CFG, cross_attn_ratio=1),
        vision_cfg=VISION_CFG,
    )
    assert len(dense.text.cross_attn) == 4


def test_convert_legacy_mammut_cfg():
    fork_cfg = {
        'embed_dim': 768,
        'vision_cfg': {'image_size': 224, 'layers': 24, 'width': 1024, 'patch_size': 14},
        'text_cfg': {
            'context_length': 77, 'vocab_size': 49408, 'width': 768, 'heads': 12, 'layers': 12,
            'output_tokens': True, 'cross_attn_ratio': 2, 'does_full_decoding': True,
        },
        'custom_text': True,
    }
    converted = _convert_legacy_mammut_cfg(fork_cfg)
    assert 'text_cfg' not in converted
    assert 'custom_text' not in converted
    mm = converted['multimodal_cfg']
    assert mm['cross_attn_ratio'] == 2
    assert 'does_full_decoding' not in mm and 'output_tokens' not in mm
    # legacy numerics preserved by default
    assert mm['pool_type'] == 'avg_all'
    assert mm['use_pad_mask'] is False
    assert mm['proj_type'] == 'none'


def test_convert_legacy_mammut_cfg_passthrough():
    clip_cfg = {'embed_dim': 512, 'vision_cfg': {}, 'text_cfg': {'width': 512}}
    assert _convert_legacy_mammut_cfg(clip_cfg) is clip_cfg
    coca_cfg = {'embed_dim': 512, 'vision_cfg': {}, 'text_cfg': {'width': 512}, 'multimodal_cfg': {}}
    assert _convert_legacy_mammut_cfg(coca_cfg) is coca_cfg


def test_convert_legacy_mammut_cfg_rejects_has_mlp():
    cfg = {'embed_dim': 512, 'vision_cfg': {}, 'text_cfg': {'does_full_decoding': True, 'has_mlp': False}}
    with pytest.raises(ValueError, match='has_mlp'):
        _convert_legacy_mammut_cfg(cfg)


def test_get_tokenizer_hub_branch_applies_legacy_mammut_shim(monkeypatch, tmp_path):
    """Config translation happens at ingestion (_get_hf_config), so get_tokenizer's hf-hub branch
    sees the same schema create_model does: a fork-format config (text_cfg + does_full_decoding)
    resolves via multimodal_cfg, and the generative pad validation applies (here: reserved roberta
    pad=1 with no declared pad_id warns)."""
    import json
    from open_clip import factory

    fork_cfg = {
        'model_cfg': {
            'embed_dim': 768,
            'vision_cfg': {'image_size': 224, 'layers': 2, 'width': 768, 'patch_size': 32},
            'text_cfg': {
                'context_length': 77, 'vocab_size': 50265, 'width': 768, 'heads': 12, 'layers': 2,
                'hf_tokenizer_name': 'roberta-base',
                'output_tokens': True, 'cross_attn_ratio': 2, 'does_full_decoding': True,
            },
            'custom_text': True,
        },
    }
    cfg_path = tmp_path / 'open_clip_config.json'
    cfg_path.write_text(json.dumps(fork_cfg))
    # patch the download, not _get_hf_config: the ingestion translation must be exercised
    monkeypatch.setattr(factory, 'download_pretrained_from_hf',
                        lambda model_id, filename=None, cache_dir=None: str(cfg_path))
    with pytest.warns(UserWarning, match='pad-value fallback'):
        tokenizer = open_clip.get_tokenizer('hf-hub:fake/fork-mammut')
    assert tokenizer.pad_token_id == 1


def test_add_model_config_translates_fork_format(tmp_path):
    """The builtin registry door also translates: a fork-format json added via add_model_config
    registers as a MaMMUT config (previously it registered raw and crashed CLIPTextCfg on
    does_full_decoding); an unsupported fork variant warns and is skipped without breaking the scan."""
    import json
    from open_clip.mammut_model import MaMMUT

    fork = {
        'embed_dim': 64,
        'vision_cfg': {'image_size': 64, 'layers': 2, 'width': 64, 'patch_size': 32,
                       'output_tokens': True, 'pool_type': 'avg_all', 'final_ln_after_pool': True},
        'text_cfg': {'context_length': 16, 'vocab_size': 64, 'width': 64, 'heads': 2, 'layers': 2,
                     'output_tokens': True, 'cross_attn_ratio': 2, 'does_full_decoding': True},
        'custom_text': True,
    }
    (tmp_path / 'forkfmt-mammut-tiny.json').write_text(json.dumps(fork))
    bad = dict(fork, text_cfg=dict(fork['text_cfg'], has_mlp=False))
    (tmp_path / 'forkfmt-mammut-bad.json').write_text(json.dumps(bad))

    with pytest.warns(UserWarning, match='has_mlp'):
        open_clip.add_model_config(tmp_path)
    cfg = open_clip.get_model_config('forkfmt-mammut-tiny')
    assert 'multimodal_cfg' in cfg and 'text_cfg' not in cfg
    assert open_clip.get_model_config('forkfmt-mammut-bad') is None
    model = open_clip.create_model('forkfmt-mammut-tiny')
    assert isinstance(model, MaMMUT)


def test_mammut_factory_create():
    model = open_clip.create_model('mammut2_ViT-B-32')
    assert isinstance(model, MaMMUT)
    assert isinstance(model.text, MultimodalDecoder)
    assert model.context_length == 77
    tokenizer = open_clip.get_tokenizer('mammut2_ViT-B-32')
    assert tokenizer.context_length == 77


@pytest.mark.parametrize('mm_cfg', ARCH_CFGS, ids=ARCH_IDS)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
def test_mammut_low_precision(mm_cfg, dtype):
    """Static fp16/bf16 conversion must cover the raw-Parameter projections
    (map_viz2txt_kv, classic lm_head/text_projection)."""
    from open_clip.model import convert_weights_to_lp

    model = _tiny_model(mm_cfg)
    convert_weights_to_lp(model, dtype=dtype)
    assert model.map_viz2txt_kv.dtype == dtype
    image, text = _tiny_batch()
    with torch.no_grad():
        out = model(image.to(dtype), text)
    assert out['logits'].dtype == dtype
    assert torch.isfinite(out['text_features'].float()).all()
    assert torch.isfinite(out['logits'].float()).all()


def test_mammut_cross_attn_ratio_non_divisor():
    """cross_attn_ratio that doesn't divide layers: cross-attn every Nth layer,
    trailing group just shorter (layers=4, ratio=3 -> layers 0 and 3)."""
    classic = _tiny_model(dict(MULTIMODAL_CFG, cross_attn_ratio=3))
    assert classic.text.cross_step == 3
    assert len(classic.text.cross_attn) == 2
    modern = _tiny_model(dict(MODERN_MULTIMODAL_CFG, cross_attn_ratio=3))
    assert [b.xattn is not None for b in modern.text.blocks] == [True, False, False, True]


@pytest.mark.parametrize('mm_cfg', ARCH_CFGS, ids=ARCH_IDS)
def test_mammut_pad_id_propagation(mm_cfg):
    """multimodal_cfg.pad_id flows tower -> model attr -> task fallback masking; the loss itself
    is pad-agnostic (labels arrive -100 masked)."""
    import types
    from open_clip import create_task

    model = _tiny_model(dict(mm_cfg, pad_id=1))
    assert model.pad_id == 1
    assert model.text.pad_id == 1
    args = types.SimpleNamespace(
        model='mammut_tiny', distill=False, siglip=False, local_loss=False,
        gather_with_grad=False, rank=0, world_size=1,
        coca_caption_loss_weight=1.0, coca_contrastive_loss_weight=1.0,
        loss_dist_impl=None,
    )
    task = create_task(args, model=model)
    assert task.pad_id == 1  # fallback label masking follows the model pad id
    assert task.loss.pad_id is None  # task path: labels pre-masked to -100
    assert task.loss.caption_loss.ignore_index == -100
    text = torch.tensor([[3, 5, 1, 1]])
    assert task._caption_labels(text).tolist() == [[5, -100, -100]]


# ---- modern decoder specifics ----

def test_mammut_modern_factory_create():
    model = open_clip.create_model('mammut2-moderntext_ViT-B-32')
    assert isinstance(model, MaMMUT)
    assert isinstance(model.text, ModernMultimodalDecoder)
    tokenizer = open_clip.get_tokenizer('mammut2-moderntext_ViT-B-32')
    assert tokenizer.context_length == 77


def test_mammut_modern_cross_sublayer_placement():
    model = _tiny_model(MODERN_MULTIMODAL_CFG)
    has_xattn = [block.xattn is not None for block in model.text.blocks]
    assert has_xattn == [True, False, True, False]


def test_mammut_modern_tie_lm_head():
    model = _tiny_model(dict(MODERN_MULTIMODAL_CFG, tie_lm_head=True))
    assert model.text.lm_head.weight is model.text.token_embedding.weight
    untied = _tiny_model(MODERN_MULTIMODAL_CFG)
    assert untied.text.lm_head.weight is not untied.text.token_embedding.weight


def test_mammut_modern_caption_pass_is_causal():
    """Changing a suffix token must not affect logits at earlier positions."""
    model = _tiny_model(MODERN_MULTIMODAL_CFG)
    image, text = _tiny_batch(num_pad=0)
    text2 = text.clone()
    text2[:, 9] = (text[:, 9] + 1) % 60 + 3
    with torch.no_grad():
        logits1 = model(image, text)['logits']
        logits2 = model(image, text2)['logits']
    assert torch.allclose(logits1[:, :9], logits2[:, :9], atol=1e-5)
    assert not torch.allclose(logits1[:, 9:], logits2[:, 9:])


def test_mammut_modern_rope_position_sensitive():
    """Guards against the positional encoding being silently dropped (RoPE, no learned pos embed)."""
    model = _tiny_model(MODERN_MULTIMODAL_CFG)
    _, text = _tiny_batch()
    swapped = text.clone()
    swapped[:, [1, 2]] = text[:, [2, 1]]
    with torch.no_grad():
        assert not torch.allclose(model.encode_text(text), model.encode_text(swapped))


def test_mammut_modern_rejects_unsupported_cfg():
    with pytest.raises(ValueError, match='pool_type'):
        _tiny_model(dict(MODERN_MULTIMODAL_CFG, pool_type='avg_all'))
    with pytest.raises(ValueError, match='reg_tokens'):
        _tiny_model(dict(MODERN_MULTIMODAL_CFG, reg_tokens=2))
    with pytest.raises(ValueError, match='eos'):
        _tiny_model(dict(MODERN_MULTIMODAL_CFG, pool_type='eos'))  # eos pooling requires eos_id


def test_mammut_modern_map_pool():
    model = _tiny_model(dict(MODERN_MULTIMODAL_CFG, pool_type='map'))
    _, text = _tiny_batch(num_pad=8)
    with torch.no_grad():
        full = model.encode_text(text)
        short = model.encode_text(text[:, :12])
    assert full.shape == (2, 64)
    assert torch.allclose(full, short, atol=1e-5)  # MAP pool masks pads too
