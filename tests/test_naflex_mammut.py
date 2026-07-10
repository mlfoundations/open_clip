"""MaMMUT2 with a NaFlex (timm NaFlexVit) vision tower.

Covers the TimmModel ``output_tokens`` mode (tensor + NaFlex-dict inputs), the MaMMUT
``image_embs_valid`` threading into decoder cross-attention masking, padding invariance
end-to-end, both caption-loss paths, task/dummy-batch integration, and beam generation
with an expanded validity mask.
"""
import pytest
import torch

import open_clip
from open_clip.coca_model import CoCa, MultimodalCfg
from open_clip.mammut_model import MaMMUT
from open_clip.model import CLIPVisionCfg
from open_clip.task.coca_task import CoCaTask
from open_clip.timm_model import TimmModel
from open_clip_train.naflex_data import NAFLEX_AVAILABLE

pytestmark = pytest.mark.skipif(not NAFLEX_AVAILABLE, reason="timm NaFlex support is not available")

PATCH = 16
PATCH_DIM = PATCH * PATCH * 3  # flattened raw patch
TRUNK_DIM = 64

TINY_VISION_CFG = dict(
    timm_model_name='naflexvit_base_patch16_gap',
    timm_pool='avg',
    timm_proj='linear',
    output_tokens=True,
    image_size=64,
    timm_model_kwargs=dict(embed_dim=TRUNK_DIM, depth=2, num_heads=2),
)
TINY_MM_CFG = dict(
    context_length=24,
    vocab_size=128,
    width=32,  # == embed_dim (default proj_type 'none')
    heads=4,
    layers=2,
    cross_attn_ratio=1,
)


def _tiny_model(seed=0, **vision_overrides):
    torch.manual_seed(seed)
    vision_cfg = dict(TINY_VISION_CFG)
    if vision_overrides:
        vision_cfg['timm_model_kwargs'] = {**vision_cfg['timm_model_kwargs'], **vision_overrides}
    return MaMMUT(embed_dim=32, multimodal_cfg=TINY_MM_CFG, vision_cfg=vision_cfg).eval()


def _patch_batch(batch_size=2, n=16, n_pad=0, seed=0, grid=4):
    """Hand-built NaFlex patch dict: n valid patches on a grid, n_pad trailing padding."""
    torch.manual_seed(seed)
    total = n + n_pad
    patches = torch.randn(batch_size, total, PATCH_DIM)
    coord = torch.zeros(batch_size, total, 2, dtype=torch.long)
    idx = torch.arange(n)
    coord[:, :n, 0] = idx // grid
    coord[:, :n, 1] = idx % grid
    valid = torch.zeros(batch_size, total, dtype=torch.bool)
    valid[:, :n] = True
    return {'patches': patches, 'patch_coord': coord, 'patch_valid': valid}


def _text_batch(batch_size=2, seed=1):
    torch.manual_seed(seed)
    text = torch.randint(3, 100, (batch_size, TINY_MM_CFG['context_length']))
    text[:, -6:] = 0
    return text


# ---------------------------------------------------------------- TimmModel token mode

def test_timm_output_tokens_tensor_parity():
    """Tensor input: pooled output identical with and without token mode; prefix stripped."""
    torch.manual_seed(0)
    tm = TimmModel(
        'naflexvit_base_patch16_gap', embed_dim=32, image_size=64,
        pool='avg', proj='linear', output_tokens=True,
        model_kwargs=dict(embed_dim=TRUNK_DIM, depth=2, num_heads=2),
    ).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = tm(x)
        tm.output_tokens = False
        pooled = tm(x)
    pooled_t, tokens, valid = out['pooled'], out['patch_tokens'], out['patch_valid']
    torch.testing.assert_close(pooled_t, pooled)  # same op sequence -> exact
    n_prefix = tm.trunk.num_prefix_tokens
    assert n_prefix == 4  # naflexvit_gap reg tokens; stripping is load-bearing
    assert tokens.shape == (2, (64 // PATCH) ** 2, TRUNK_DIM)
    assert valid is None


def test_timm_output_tokens_dict_input():
    tm = TimmModel(
        'naflexvit_base_patch16_gap', embed_dim=32, image_size=64,
        pool='avg', proj='linear', output_tokens=True,
        model_kwargs=dict(embed_dim=TRUNK_DIM, depth=2, num_heads=2),
    ).eval()
    batch = _patch_batch(n=16, n_pad=8)
    with torch.no_grad():
        out = tm(batch)
    pooled, tokens, valid = out['pooled'], out['patch_tokens'], out['patch_valid']
    assert pooled.shape == (2, 32)
    assert tokens.shape == (2, 24, TRUNK_DIM)  # prefix stripped: aligned with patch-only valid
    assert valid.shape == (2, 24) and valid.dtype == torch.bool
    torch.testing.assert_close(valid, batch['patch_valid'])


def test_clip_timm_output_tokens_selects_pooled_features():
    model = open_clip.CLIP(
        embed_dim=32,
        vision_cfg=TINY_VISION_CFG,
        text_cfg=dict(context_length=8, vocab_size=64, width=32, heads=4, layers=1),
    ).eval()

    with torch.no_grad():
        features = model.encode_image(torch.randn(2, 3, 64, 64))

    assert features.shape == (2, 32)


def test_coca_rejects_timm_vision_tower_with_clear_error():
    with pytest.raises(ValueError, match='does not support timm vision towers'):
        CoCa(
            embed_dim=32,
            vision_cfg=TINY_VISION_CFG,
            text_cfg=dict(
                context_length=8,
                vocab_size=64,
                width=32,
                heads=4,
                layers=1,
                embed_cls=True,
                output_tokens=True,
            ),
            multimodal_cfg=dict(context_length=8, vocab_size=64, width=32, heads=4, layers=1),
        )


def test_mammut_requires_vision_token_output():
    vision_cfg = dict(TINY_VISION_CFG, output_tokens=False)
    with pytest.raises(ValueError, match='output_tokens=True'):
        MaMMUT(embed_dim=32, multimodal_cfg=TINY_MM_CFG, vision_cfg=vision_cfg)


def test_timm_output_tokens_no_reg_variant():
    """reg_tokens=0 override: num_prefix_tokens=0 path."""
    tm = TimmModel(
        'naflexvit_base_patch16_gap', embed_dim=32, image_size=64,
        pool='avg', proj='linear', output_tokens=True,
        model_kwargs=dict(embed_dim=TRUNK_DIM, depth=2, num_heads=2, reg_tokens=0),
    ).eval()
    assert tm.trunk.num_prefix_tokens == 0
    batch = _patch_batch()
    with torch.no_grad():
        out = tm(batch)
    assert out['patch_tokens'].shape[1] == batch['patches'].shape[1]


def test_timm_output_tokens_plain_vit():
    """Hypothetical exercised: a non-naflex NLC timm ViT works in token mode (dense input)."""
    torch.manual_seed(0)
    tm = TimmModel(
        'vit_tiny_patch16_224', embed_dim=32, image_size=224,
        pool='avg', proj='linear', output_tokens=True,
    ).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = tm(x)
        tm.output_tokens = False
        pooled = tm(x)
    torch.testing.assert_close(out['pooled'], pooled)  # exact parity
    assert out['patch_tokens'].shape == (2, (224 // 16) ** 2, tm.trunk.num_features)
    assert out['patch_valid'] is None


# ---------------------------------------------------------------- MaMMUT integration

def test_mammut_naflex_forward_contract():
    model = _tiny_model()
    image = _patch_batch(n=16, n_pad=8)
    text = _text_batch()
    with torch.no_grad():
        out = model(image=image, text=text)
    assert set(out.keys()) == {'image_features', 'text_features', 'logits', 'logit_scale'}
    assert out['image_features'].shape == (2, 32)
    assert out['logits'].shape == (2, TINY_MM_CFG['context_length'], TINY_MM_CFG['vocab_size'])
    assert torch.isfinite(out['logits']).all()
    # width derivation: map_viz2txt_kv sized from trunk dim, not CLIPVisionCfg.width default (768)
    assert model.map_viz2txt_kv.shape[0] == TRUNK_DIM
    # image-only path surfaces the validity for external caching
    with torch.no_grad():
        img_out = model(image=image)
    assert img_out['image_embs_valid'].shape == (2, 24)


def test_mammut_naflex_padding_invariance():
    """Same real patches at two padding lengths -> same features and caption logits."""
    model = _tiny_model()
    text = _text_batch()
    short = _patch_batch(n=16, n_pad=0)
    long = _patch_batch(n=16, n_pad=16)
    # identical real content
    long['patches'][:, :16] = short['patches']
    with torch.no_grad():
        out_s = model(image=short, text=text)
        out_l = model(image=long, text=text)
    torch.testing.assert_close(out_l['image_features'], out_s['image_features'], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out_l['logits'], out_s['logits'], rtol=1e-4, atol=1e-5)


def test_mammut_naflex_fused_caption_loss_parity():
    model = _tiny_model()
    model.train()
    image = _patch_batch(n=16, n_pad=8)
    text = _text_batch()
    text_valid = text != 0
    labels = text[:, 1:].masked_fill(~text_valid[:, 1:], -100)

    out_legacy = model(image=image, text=text, text_valid=text_valid)
    loss_legacy = torch.nn.functional.cross_entropy(
        out_legacy['logits'][:, :-1].permute(0, 2, 1), labels, ignore_index=-100)
    out_fused = model(image=image, text=text, text_valid=text_valid, labels=labels)
    torch.testing.assert_close(out_fused['caption_loss'], loss_legacy, rtol=1e-5, atol=1e-5)


def test_mammut_naflex_task_training_forward():
    """CoCaTask spreads a naflex batch into the model; finite losses both loss paths."""
    for fused in (False, True):
        model = _tiny_model()
        model.train()
        task = CoCaTask(model, fused_caption_loss=fused)
        batch = {
            'image': _patch_batch(n=16, n_pad=8),
            'text': _text_batch(),
        }
        batch['text_valid'] = batch['text'] != 0
        losses, report = task.training_forward(batch)
        assert torch.isfinite(losses['loss'])
        assert ('caption_loss' in losses) and torch.isfinite(losses['caption_loss'])


def test_mammut_naflex_dummy_batch():
    from open_clip.naflex_config import NaFlexDataConfig
    model = _tiny_model()
    task = CoCaTask(model)
    task.set_naflex_data_config(NaFlexDataConfig.resolve(
        patch_sizes=(PATCH,), seq_lens=(32,), eval_seq_len=32, eval_patch_size=PATCH))
    batch = task.create_dummy_batch(batch_size=2)
    assert isinstance(batch['image'], dict) and 'patches' in batch['image']
    with torch.no_grad():
        out = model(**{k: v for k, v in batch.items() if k in ('image', 'text', 'text_valid')})
    assert torch.isfinite(out['logits']).all()


def test_mammut_naflex_generate_beam():
    """Beam search expands image_embs AND context_valid together (the beam hazard)."""
    pytest.importorskip('transformers')
    model = _tiny_model()
    image = _patch_batch(batch_size=2, n=16, n_pad=8)
    with torch.no_grad():
        out = model.generate(
            image, seq_len=8, max_seq_len=TINY_MM_CFG['context_length'],
            generation_type='beam_search', num_beams=2, num_beam_groups=1, min_seq_len=2,
            sot_token_id=1, eos_token_id=2, pad_token_id=0,
        )
    assert out.shape[0] == 2
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize('name,patch,seq_len', [
    ('mammut2-naflex_ViT-B-16', 16, 256),
    ('mammut2-naflex_ViT-B-32', 32, 64),   # patch_size override on the patch16 timm variant
])
def test_registered_config_builds(name, patch, seq_len):
    """Registered jsons build, derive width 768 from the trunk, expose naflex attrs."""
    model = open_clip.create_model(name)
    assert model.map_viz2txt_kv.shape[0] == 768
    assert model.visual.image_seq_len == seq_len
    assert model.visual.trunk.get_patch_size() == (patch, patch)
    assert model.visual.output_tokens
