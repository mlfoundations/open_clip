"""coca2 corrected configs: paper-style cascade/parallel attentional pooling, corrected cls mask,
and the modern-text CoCa variant (ModernTextTransformer tower + ModernMultimodalTransformer decoder)."""
import types

import pytest
import torch

import open_clip
from open_clip.coca_model import CoCa, MultimodalCfg
from open_clip.transformer import (
    ModernMultimodalDecoder,
    ModernMultimodalTransformer,
    ModernTextTransformer,
    MultimodalDecoder,
    MultimodalTransformer,
)

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


def test_multimodal_transformer_calls_init_parameters_from_constructor(monkeypatch):
    calls = 0
    original = MultimodalTransformer.init_parameters

    def wrapped(self):
        nonlocal calls
        calls += 1
        original(self)

    monkeypatch.setattr(MultimodalTransformer, 'init_parameters', wrapped)
    MultimodalTransformer(width=32, layers=1, heads=4, context_length=8, output_dim=64)

    assert calls == 1


def test_multimodal_transformer_init_parameters_external_after_to_empty():
    with torch.device('meta'):
        model = MultimodalTransformer(
            width=32,
            layers=2,
            heads=4,
            context_length=8,
            output_dim=64,
            ls_init_value=0.25,
        )
    model.to_empty(device='cpu')
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(torch.nan)
    model.init_parameters()

    for name, param in model.named_parameters():
        assert not param.is_meta, name
        assert torch.isfinite(param).all(), name
    assert torch.allclose(model.ln_final.weight, torch.ones_like(model.ln_final.weight))
    assert torch.allclose(model.ln_final.bias, torch.zeros_like(model.ln_final.bias))
    assert torch.allclose(model.resblocks[0].ln_1.weight, torch.ones_like(model.resblocks[0].ln_1.weight))
    assert torch.allclose(model.resblocks[0].ln_1.bias, torch.zeros_like(model.resblocks[0].ln_1.bias))
    assert torch.allclose(model.cross_attn[0].ln_1_kv.weight, torch.ones_like(model.cross_attn[0].ln_1_kv.weight))
    assert torch.allclose(model.resblocks[0].attn.in_proj_bias, torch.zeros_like(model.resblocks[0].attn.in_proj_bias))
    assert torch.allclose(
        model.resblocks[0].attn.out_proj.bias,
        torch.zeros_like(model.resblocks[0].attn.out_proj.bias),
    )
    assert torch.allclose(model.resblocks[0].mlp.c_fc.bias, torch.zeros_like(model.resblocks[0].mlp.c_fc.bias))
    assert torch.allclose(model.resblocks[0].ls_1.gamma, torch.full_like(model.resblocks[0].ls_1.gamma, 0.25))
    assert model.text_projection.std() > 0
    assert torch.allclose(model.attn_mask, model.build_attention_mask())


def _modern_init_cfg():
    return MultimodalCfg(
        text_arch='modern',
        context_length=8,
        vocab_size=64,
        width=32,
        heads=4,
        layers=2,
        pool_type='map',
        proj_type='linear',
        proj_bias=True,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        qk_norm=True,
        attn_gated=True,
        value_residual=True,
        ls_init_value=0.25,
    )


@pytest.mark.parametrize('factory', [
    pytest.param(
        lambda: MultimodalDecoder(
            context_length=8,
            vocab_size=64,
            width=32,
            heads=4,
            layers=2,
            output_dim=16,
            proj_type='linear',
            ls_init_value=0.25,
        ),
        id='classic-mammut',
    ),
    pytest.param(
        lambda: ModernMultimodalDecoder(_modern_init_cfg(), output_dim=16),
        id='modern-mammut',
    ),
    pytest.param(
        lambda: ModernMultimodalTransformer(_modern_init_cfg()),
        id='modern-coca',
    ),
    pytest.param(
        lambda: ModernTextTransformer(_modern_init_cfg(), output_dim=16),
        id='modern-text',
    ),
])
def test_decoder_init_parameters_covers_all_parameters_after_to_empty(factory):
    with torch.device('meta'):
        model = factory()
    model.to_empty(device='cpu')
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(torch.nan)

    model.init_parameters()

    for name, param in model.named_parameters():
        assert not param.is_meta, name
        assert torch.isfinite(param).all(), name


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
        pool_type='eos', bos_id=1, eos_id=2, freeze_pad_embed=False, output_tokens=True,
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
    )
    assert out.shape[0] == 2 and out.shape[1] <= 8
    assert torch.equal(out[:, 0], torch.full_like(out[:, 0], 1))


def test_coca_generate_with_non_native_text_tower_no_token_embedding():
    pytest.importorskip('transformers')

    class NoTokenEmbeddingText(torch.nn.Module):
        pad_id = 0
        bos_id = 1
        eos_id = 2
        vocab_size = 64

        def forward(self, ids, attention_mask=None):
            token_embs = torch.zeros(ids.shape[0], ids.shape[1], 64, device=ids.device)
            text_latent = torch.zeros(ids.shape[0], 64, device=ids.device)
            return text_latent, token_embs

    model = _tiny_coca(attentional_pool='cascade').eval()
    model.text = NoTokenEmbeddingText()
    model.pad_id = model.text.pad_id
    model.bos_id = model.text.bos_id
    model.eos_id = model.text.eos_id

    out = model.generate(
        torch.randn(2, 3, 64, 64),
        generation_type='top_k',
        seq_len=8,
        min_seq_len=3,
    )
    assert out.shape[0] == 2 and out.shape[1] <= 8
    assert torch.equal(out[:, 0], torch.full_like(out[:, 0], 1))


def test_coca_generate_keeps_provided_prompt_and_trims_common_padding():
    pytest.importorskip('transformers')
    model = _tiny_modern_coca().eval()
    image = torch.randn(2, 3, 64, 64)

    prompt = torch.tensor([[5, 6], [7, 8]])
    out = model.generate(image, text=prompt, generation_type='top_k', seq_len=8, min_seq_len=3)
    assert torch.equal(out[:, :2], prompt)

    padded = torch.tensor([[5, 6, 0, 0], [7, 8, 0, 0]])
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    out = model.generate(
        image,
        text=padded,
        text_valid=valid,
        generation_type='top_k',
        seq_len=8,
        min_seq_len=3,
    )
    assert torch.equal(out[:, :2], prompt)


def test_coca_generate_rejects_variable_length_padded_prompts():
    pytest.importorskip('transformers')
    model = _tiny_modern_coca().eval()
    padded = torch.tensor([[5, 6, 0], [7, 0, 0]])
    valid = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    with pytest.raises(ValueError, match="same valid length"):
        model.generate(
            torch.randn(2, 3, 64, 64),
            text=padded,
            text_valid=valid,
            generation_type='top_k',
            seq_len=8,
            min_seq_len=3,
        )


def test_coca_generate_validates_requested_lengths():
    pytest.importorskip('transformers')
    model = _tiny_modern_coca().eval()
    image = torch.randn(2, 3, 64, 64)
    with pytest.raises(ValueError, match="max_seq_len"):
        model.generate(image, generation_type='top_k', seq_len=8, max_seq_len=4, min_seq_len=3)
    with pytest.raises(ValueError, match="context_length"):
        model.generate(image, generation_type='top_k', seq_len=20, min_seq_len=3)


def test_coca_generate_copies_generation_config_and_fills_special_ids():
    transformers = pytest.importorskip('transformers')
    model = _tiny_modern_coca().eval()
    config = transformers.GenerationConfig(
        max_length=8,
        min_length=3,
        do_sample=True,
        top_k=1,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        use_cache=True,
    )
    out = model.generate(torch.randn(2, 3, 64, 64), generation_config=config)
    assert out.shape[0] == 2 and out.shape[1] <= 8
    assert config.bos_token_id is None
    assert config.eos_token_id is None
    assert config.pad_token_id is None
    assert config.use_cache is True


def test_coca_generate_honors_max_new_tokens_from_generation_config():
    transformers = pytest.importorskip('transformers')
    model = _tiny_modern_coca().eval()
    prompt = torch.tensor([[5, 6], [7, 8]])
    config = transformers.GenerationConfig(
        max_new_tokens=2,
        do_sample=True,
        top_k=1,
    )

    out = model.generate(
        torch.randn(2, 3, 64, 64),
        text=prompt,
        seq_len=3,
        min_seq_len=5,
        fixed_output_length=True,
        generation_config=config,
    )

    assert out.shape == (2, 4)
    assert torch.equal(out[:, :2], prompt)


def test_coca_generate_validates_effective_max_new_tokens_length():
    transformers = pytest.importorskip('transformers')
    model = _tiny_modern_coca().eval()
    config = transformers.GenerationConfig(max_new_tokens=15, do_sample=True, top_k=1)

    with pytest.raises(ValueError, match='context_length'):
        model.generate(
            torch.randn(2, 3, 64, 64),
            text=torch.tensor([[5, 6], [7, 8]]),
            generation_config=config,
        )


def test_coca_generate_custom_max_length_is_independent_of_legacy_seq_len():
    transformers = pytest.importorskip('transformers')
    model = _tiny_modern_coca().eval()
    prompt = torch.tensor([[3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14]])
    config = transformers.GenerationConfig(max_length=8, do_sample=True, top_k=1)

    out = model.generate(
        torch.randn(2, 3, 64, 64),
        text=prompt,
        seq_len=4,
        min_seq_len=5,
        generation_config=config,
    )

    assert out.shape[1] <= 8
    assert torch.equal(out[:, :6], prompt)


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
