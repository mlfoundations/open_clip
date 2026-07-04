"""coca2 corrected configs: paper-style cascade/parallel attentional pooling, corrected cls mask,
and the modern-text CoCa variant (ModernTextTransformer tower + ModernMultimodalTransformer decoder)."""
import types

import pytest
import torch

import open_clip
from open_clip.coca_model import CoCa
from open_clip.transformer import ModernMultimodalTransformer, ModernTextTransformer

TINY_KW = dict(
    embed_dim=64,
    text_cfg=dict(context_length=16, vocab_size=64, width=64, heads=2, layers=2, embed_cls=True, output_tokens=True),
    vision_cfg=dict(image_size=64, layers=2, width=96, patch_size=32, attn_pooler_queries=8,
                    attn_pooler_heads=2, output_tokens=True),
    multimodal_cfg=dict(context_length=16, vocab_size=64, width=64, heads=2, layers=2),
)


def _tiny_coca(attentional_pool=True, seed=0, **text_over):
    kw = {k: dict(v) if isinstance(v, dict) else v for k, v in TINY_KW.items()}
    kw['vision_cfg']['attentional_pool'] = attentional_pool
    kw['text_cfg'].update(text_over)
    torch.manual_seed(seed)
    return CoCa(**kw)


@pytest.mark.parametrize('mode', ['parallel', 'cascade'])
def test_attentional_pool_paper_modes(mode):
    """parallel/cascade produce [B, embed_dim] contrastive features with width != embed_dim
    (regression: unsqueezed [B, 1, D] features, width-sized ln_post, wrong cascade context_dim)."""
    model = _tiny_coca(attentional_pool=mode).eval()
    assert model.visual.attn_pool_contrastive is not None
    assert model.visual.proj is None  # poolers project themselves; no extra head projection
    image = torch.randn(2, 3, 64, 64)
    text = torch.randint(1, 60, (2, 16))
    with torch.no_grad():
        out = model(image, text)
    assert out['image_features'].shape == (2, 64)
    assert out['logits'].shape == (2, 16, 64)


@pytest.mark.parametrize('mode', ['parallel', 'cascade'])
def test_attentional_pool_paper_modes_grads(mode):
    """Both poolers receive gradients: contrastive loss -> contrastive pooler, caption loss -> generative."""
    model = _tiny_coca(attentional_pool=mode)
    model.train()
    image = torch.randn(2, 3, 64, 64)
    text = torch.randint(1, 60, (2, 16))
    out = model(image, text)
    (out['image_features'].sum() + out['logits'].sum()).backward()
    assert model.visual.attn_pool_contrastive.query.grad is not None
    assert model.visual.attn_pool.query.grad is not None


def test_attentional_pool_legacy_bool_unchanged():
    """Legacy bool mode (released CoCa weights) keeps its structure: single pooler + square proj."""
    model = _tiny_coca(attentional_pool=True).eval()
    assert model.visual.attn_pool_contrastive is None
    assert model.visual.proj is not None
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64), torch.randint(1, 60, (2, 16)))
    assert out['image_features'].shape == (2, 64)


def test_vision_lock_includes_attentional_poolers():
    """layer_groups must place the pooler(s) in the head group: --lock-image has to freeze them
    (regression: cascade/parallel poolers -- and the legacy single pooler -- were ungrouped, so
    locking the tower left them trainable and layer-decay treated them as lr_scale=1.0)."""
    model = _tiny_coca(attentional_pool='cascade')
    model.visual.lock(unlocked_groups=0)
    assert not model.visual.attn_pool.query.requires_grad
    assert not model.visual.attn_pool_contrastive.query.requires_grad
    # unlocking the top (head) group re-enables both poolers
    model.visual.lock(unlocked_groups=1)
    assert model.visual.attn_pool.query.requires_grad
    assert model.visual.attn_pool_contrastive.query.requires_grad
    # legacy single-pooler mode is grouped alongside its proj
    legacy = _tiny_coca(attentional_pool=True)
    legacy.visual.lock(unlocked_groups=0)
    assert not legacy.visual.attn_pool.query.requires_grad
    assert not legacy.visual.proj.requires_grad
    # plain (non-attentional) towers keep their proj-only head group
    from open_clip.transformer import VisionTransformer
    plain = VisionTransformer(image_size=64, patch_size=32, width=64, layers=2, heads=2,
                              mlp_ratio=4.0, output_dim=64)
    assert plain.attn_pool_contrastive is None
    names = [n for n, _ in plain.layer_groups()]
    assert names[-1] == 'proj'


def test_correct_cls_mask_config_plumbing():
    """correct_cls_mask now flows config -> tower (it previously wasn't a CLIPTextCfg field at all)."""
    legacy = _tiny_coca(seed=0)
    fixed = _tiny_coca(seed=0, correct_cls_mask=True)
    assert legacy.text.correct_cls_mask is False
    assert fixed.text.correct_cls_mask is True
    # same weights, different mask construction: padded input diverges
    fixed.load_state_dict(legacy.state_dict())
    text = torch.tensor([[1, 5, 8, 2, 0, 0, 0, 0]])
    with torch.no_grad():
        f_legacy = legacy.eval().encode_text(text)
        f_fixed = fixed.eval().encode_text(text)
    assert not torch.allclose(f_legacy, f_fixed)


def test_coca2_config():
    model = open_clip.create_model('coca2_ViT-B-32')
    assert isinstance(model, CoCa)
    assert model.text.correct_cls_mask is True
    assert model.visual.attn_pool_type == 'cascade'
    assert model.visual.attn_pool_contrastive is not None


def test_coca2_moderntext_config():
    model = open_clip.create_model('coca2-moderntext_ViT-B-32').eval()
    assert isinstance(model, CoCa)
    assert isinstance(model.text, ModernTextTransformer)
    assert isinstance(model.text_decoder, ModernMultimodalTransformer)
    # SimpleTokenizer vocab: id 0 is a real token, the pad row must stay trainable
    assert model.text.token_embedding.padding_idx is None
    assert model.text.token_embedding.weight[0].abs().sum() > 0

    tokenizer = open_clip.get_tokenizer('coca2-moderntext_ViT-B-32')
    text = tokenizer(['a photo of a cat', 'a dog'])
    image = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(image, text)
    assert out['image_features'].shape == (2, 512)
    assert out['text_features'].shape == (2, 512)
    assert out['logits'].shape == (2, text.shape[1], 49408)


def _tiny_modern_coca(seed=0):
    modern = dict(
        text_arch='modern', context_length=16, vocab_size=64, width=64, heads=2, layers=2,
        pool_type='eos', eos_id=2, freeze_pad_embed=False, output_tokens=True,
    )
    kw = {k: dict(v) if isinstance(v, dict) else v for k, v in TINY_KW.items()}
    kw['text_cfg'] = dict(modern)
    kw['multimodal_cfg'] = dict(modern, pool_type='argmax')  # decoder ignores pooling fields
    kw['vision_cfg']['attentional_pool'] = 'cascade'
    torch.manual_seed(seed)
    return CoCa(**kw)


def test_modern_coca_caption_pass_causal():
    """Suffix token changes must not affect earlier caption logits (RoPE + causal SDPA)."""
    model = _tiny_modern_coca().eval()
    image = torch.randn(1, 3, 64, 64)
    text = torch.randint(3, 60, (1, 16))
    text2 = text.clone()
    text2[:, 9] = (text[:, 9] + 1) % 56 + 3
    with torch.no_grad():
        l1 = model(image, text)['logits']
        l2 = model(image, text2)['logits']
    assert torch.allclose(l1[:, :9], l2[:, :9], atol=1e-5)
    assert not torch.allclose(l1[:, 9:], l2[:, 9:])


def test_modern_coca_task_and_grads():
    from open_clip.task import CoCaTask

    model = _tiny_modern_coca()
    task = CoCaTask(model, caption_loss_weight=1.0, clip_loss_weight=1.0, verbose=False)
    image = torch.randn(2, 3, 64, 64)
    text = torch.randint(3, 60, (2, 16))
    mask = torch.ones_like(text, dtype=torch.bool)
    mask[:, -4:] = False
    losses, _ = task.training_forward({'image': image, 'text': text, 'text_valid': mask})
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    assert model.text_decoder.lm_head.weight.grad is not None
    assert model.text.token_embedding.weight.grad is not None
    assert model.visual.attn_pool_contrastive.query.grad is not None


def test_modern_coca_generate():
    pytest.importorskip('transformers')
    model = _tiny_modern_coca().eval()
    out = model.generate(
        torch.randn(2, 3, 64, 64),
        generation_type='top_k', seq_len=8, min_seq_len=3,
        sot_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    assert out.shape[0] == 2 and out.shape[1] <= 8


def test_create_task_dispatch_coca2():
    from open_clip.task import CoCaTask

    model = _tiny_modern_coca()
    args = types.SimpleNamespace(
        model='coca2-moderntext_ViT-B-32', distill=False, siglip=False, local_loss=False,
        gather_with_grad=False, rank=0, world_size=1,
        coca_caption_loss_weight=1.0, coca_contrastive_loss_weight=1.0,
        loss_dist_impl=None, horovod=False,
    )
    task = open_clip.create_task(args, model=model)
    assert isinstance(task, CoCaTask)
