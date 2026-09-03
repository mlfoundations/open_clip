"""CoCa2 with a NaFlex (timm NaFlexVit) vision tower.

Covers model-level masked attentional pooling (paper cascade/parallel poolers over trunk tokens
with ``patch_valid`` key masking), the end of masking at the pooler (fixed-count pooled queries ->
no decoder ``context_valid``), padding invariance end-to-end, both caption-loss paths, task
integration, generation, and the registered config.
"""
import pytest
import torch

import open_clip
from open_clip.coca_model import CoCa
from open_clip.task.coca_task import CoCaTask
from open_clip_train.naflex_data import NAFLEX_AVAILABLE

pytestmark = pytest.mark.skipif(not NAFLEX_AVAILABLE, reason="timm NaFlex support is not available")

PATCH = 16
PATCH_DIM = PATCH * PATCH * 3
TRUNK_DIM = 64
EMBED_DIM = 32
CTX = 16

TINY_VISION_CFG = dict(
    timm_model_name='naflexvit_base_patch16_gap',
    timm_pool='avg',
    timm_proj='linear',
    output_tokens=True,
    image_size=64,
    attentional_pool='cascade',
    attn_pooler_queries=8,
    attn_pooler_heads=2,
    timm_model_kwargs=dict(embed_dim=TRUNK_DIM, depth=2, num_heads=2),
)
TINY_TEXT_CFG = dict(
    context_length=CTX, vocab_size=64, width=EMBED_DIM, heads=2, layers=2,
    embed_cls=True, output_tokens=True,
)
TINY_MM_CFG = dict(context_length=CTX, vocab_size=64, width=EMBED_DIM, heads=2, layers=2)
# modern-text variants (ModernTextTransformer tower + ModernMultimodalTransformer decoder), the
# stack the coca2-moderntext-naflex config uses: causal attention, masked mean pooling
TINY_MODERN_TEXT_CFG = dict(
    text_arch='modern', context_length=CTX, vocab_size=64, width=EMBED_DIM, heads=2, layers=2,
    pool_type='mean', attention_mode='causal', pad_id=0, bos_id=1, eos_id=2,
    output_tokens=True,
)
TINY_MODERN_MM_CFG = dict(
    text_arch='modern', context_length=CTX, vocab_size=64, width=EMBED_DIM, heads=2, layers=2,
    pad_id=0, bos_id=1, eos_id=2,
)


def _tiny_model(pool='cascade', seed=0):
    torch.manual_seed(seed)
    vision_cfg = dict(TINY_VISION_CFG, attentional_pool=pool)
    return CoCa(
        embed_dim=EMBED_DIM,
        vision_cfg=vision_cfg,
        text_cfg=dict(TINY_TEXT_CFG),
        multimodal_cfg=dict(TINY_MM_CFG),
    ).eval()


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
    text = torch.randint(3, 60, (batch_size, CTX))
    text[:, -4:] = 0
    return text


@pytest.mark.parametrize('pool', ['cascade', 'parallel'])
def test_coca2_naflex_forward_contract(pool):
    model = _tiny_model(pool=pool)
    assert model.attn_pool is not None and model.attn_pool_contrastive is not None
    batch = _patch_batch(n=16, n_pad=8)
    text = _text_batch()
    with torch.no_grad():
        out = model(image=batch, text=text)
    assert out['image_features'].shape == (2, EMBED_DIM)
    # decoder context = pooled queries (fixed count), not raw patch tokens
    assert out['logits'].shape == (2, CTX, TINY_MM_CFG['vocab_size'])
    for v in out.values():
        if torch.is_tensor(v):
            assert torch.isfinite(v.float()).all()


@pytest.mark.parametrize('pool', ['cascade', 'parallel'])
def test_coca2_naflex_padding_invariance(pool):
    """Same content padded into a longer bucket -> identical features AND caption logits.

    This is the assertion that distinguishes masked pooling from silently attending padding."""
    model = _tiny_model(pool=pool)
    base = _patch_batch(n=16, n_pad=0)
    padded = {
        'patches': torch.cat([base['patches'], torch.zeros(2, 8, PATCH_DIM)], dim=1),
        'patch_coord': torch.cat([base['patch_coord'], torch.zeros(2, 8, 2, dtype=torch.long)], dim=1),
        'patch_valid': torch.cat([base['patch_valid'], torch.zeros(2, 8, dtype=torch.bool)], dim=1),
    }
    text = _text_batch()
    with torch.no_grad():
        out_a = model(image=base, text=text)
        out_b = model(image=padded, text=text)
    torch.testing.assert_close(out_a['image_features'], out_b['image_features'], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out_a['logits'], out_b['logits'], rtol=1e-4, atol=1e-5)


def test_coca2_naflex_dense_tensor_input():
    """Tensor input rides token mode with patch_valid=None (all patches real)."""
    model = _tiny_model()
    with torch.no_grad():
        out = model(image=torch.randn(2, 3, 64, 64), text=_text_batch())
    assert out['image_features'].shape == (2, EMBED_DIM)
    assert torch.isfinite(out['logits'].float()).all()


def test_coca2_naflex_fused_caption_loss_parity():
    """Fused (labels-in-forward) and legacy (logits) paths agree on the caption CE."""
    model = _tiny_model()
    batch = _patch_batch(n=16, n_pad=4)
    text = _text_batch()
    labels = text[:, 1:].clone()
    labels[text[:, 1:] == 0] = -100
    with torch.no_grad():
        fused = model(image=batch, text=text, labels=labels)
        legacy = model(image=batch, text=text)
    ce_legacy = torch.nn.functional.cross_entropy(
        legacy['logits'][:, :-1].reshape(-1, legacy['logits'].shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(fused['caption_loss_ce'], ce_legacy, rtol=1e-4, atol=1e-5)


def test_coca2_naflex_task_training_forward():
    model = _tiny_model().train()
    task = CoCaTask(model, fused_caption_loss=True, verbose=False)
    batch = {'image': _patch_batch(n=16, n_pad=4), 'text': _text_batch()}
    batch['text_valid'] = batch['text'] != 0
    losses, _ = task.training_forward(batch)
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    # masked pooler received grads (it is in the caption + contrastive paths)
    assert model.attn_pool.query.grad is not None
    assert torch.isfinite(model.attn_pool.query.grad).all()
    # every trainable param must receive grad -- DDP's reducer enforces exactly this, and the
    # discarded tower readout (trunk fc_norm + head proj) tripped it before being removed
    orphans = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not orphans, f'params without grad (would break DDP): {orphans}'


def test_coca2_naflex_generate_beam():
    model = _tiny_model()
    image = _patch_batch(n=16, n_pad=8)
    with torch.no_grad():
        out = model.generate(
            image, seq_len=8, max_seq_len=CTX,
            generation_type='beam_search', num_beams=2, num_beam_groups=1, min_seq_len=2,
            sot_token_id=1, eos_token_id=2, pad_token_id=0,
        )
    assert out.shape[0] == 2
    assert torch.isfinite(out.float()).all()


def test_coca2_naflex_modern_text_mean_pool():
    """Modern causal text tower with masked mean pooling, parallel image pooling: forward contract
    plus every-param-receives-grad (the DDP reducer invariant) across both pooling branches."""
    torch.manual_seed(0)
    model = CoCa(
        embed_dim=EMBED_DIM,
        vision_cfg=dict(TINY_VISION_CFG, attentional_pool='parallel'),
        text_cfg=dict(TINY_MODERN_TEXT_CFG),
        multimodal_cfg=dict(TINY_MODERN_MM_CFG),
    ).train()
    assert model.text.pool.pool_type == 'mean'
    out = model(image=_patch_batch(n=16, n_pad=4), text=_text_batch())
    assert out['image_features'].shape == (2, EMBED_DIM)
    assert out['logits'].shape == (2, CTX, TINY_MODERN_MM_CFG['vocab_size'])
    # touch caption + contrastive heads and the logit scale so the orphan check covers everything
    loss = (
        out['logits'].float().mean()
        + ((out['image_features'] * out['text_features']).sum() * out['logit_scale']).float()
    )
    loss.backward()
    orphans = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not orphans, f'params without grad (would break DDP): {orphans}'


def test_coca2_rejects_bidirectional_modern_text():
    """The caption decoder consumes the text tower's token embeddings; a bidirectional tower would
    leak future tokens past the decoder's causal mask, so the ctor must refuse it."""
    with pytest.raises(ValueError, match='bidirectional'):
        CoCa(
            embed_dim=EMBED_DIM,
            vision_cfg=dict(TINY_VISION_CFG),
            text_cfg=dict(TINY_MODERN_TEXT_CFG, attention_mode='bidirectional'),
            multimodal_cfg=dict(TINY_MODERN_MM_CFG),
        )


def test_registered_moderntext_config_builds():
    model = open_clip.create_model('coca2-moderntext-naflex_ViT-B-32')
    assert model.attn_pool_type == 'parallel'
    assert model.attn_pool.query.shape == (128, 512)
    assert model.attn_pool.attn.k_proj_weight.shape[1] == 768  # keys projected from trunk dim
    # parallel: the contrastive pooler also reads (masked) trunk tokens, not pooled queries
    assert model.attn_pool_contrastive.attn.k_proj_weight.shape[1] == 768
    assert model.visual.image_seq_len == 64
    assert model.visual.trunk.get_patch_size() == (32, 32)
    assert model.visual.output_tokens
    assert model.text.pool.pool_type == 'mean'
    assert model.text.cfg.attention_mode == 'causal'
    assert (model.pad_id, model.bos_id, model.eos_id) == (50258, 50259, 50257)
    assert model.context_length == 128
    assert model.text_decoder.lm_head.weight.shape[0] == 50260
