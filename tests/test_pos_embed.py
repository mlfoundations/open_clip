"""Tests for open_clip.pos_embed and model position-embedding integrations (CPU)."""
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

import open_clip
from open_clip.audio.naflex_audio import AudioNaFlexCfg, MelPatchEmbed
from open_clip.naflex_genlap_model import NaFlexGenLap
from open_clip.naflex_genlip_model import GenLipPatchEmbed, NaFlexGenLip, NaFlexGenLipVisionCfg
from open_clip.pos_embed import factorized_pos_embed
from open_clip_train.optim import make_wd_exclude


def _patch_batch(shapes, patch_dim):
    n = max(h * w for h, w in shapes) + 2
    coord = torch.full((len(shapes), n, 2), 99, dtype=torch.long)
    valid = torch.zeros(len(shapes), n, dtype=torch.bool)
    for i, (h, w) in enumerate(shapes):
        coord[i, :h * w] = torch.cartesian_prod(torch.arange(h), torch.arange(w))
        valid[i, :h * w] = True
    return {'patches': torch.randn(len(shapes), n, patch_dim), 'patch_coord': coord, 'patch_valid': valid}


def _model(kind, pos_embed='factorized', ar_preserving=False):
    name = 'naflexgenlip_test' if kind == 'image' else f'naflexgenlap_test_{kind}'
    cfg = open_clip.get_model_config(name)
    cfg['text_cfg'].update(vocab_size=32, pad_id=0, bos_id=1, eos_id=2)
    embed_key = 'vision_cfg' if kind == 'image' else 'audio_naflex_cfg'
    cfg[embed_key].update(
        pos_embed=pos_embed, pos_embed_size=8, pos_embed_ar_preserving=ar_preserving,
        input_norm=True, pre_norm=True,
    )
    trunk_key = 'genlip_cfg' if kind == 'image' else 'genlap_cfg'
    cfg[trunk_key].update(prefix_layers=1, prefix_ffn_size=32)
    model = (NaFlexGenLip if kind == 'image' else NaFlexGenLap)(**cfg).eval()
    # Exercise active in-fusion branches as well as the shared trunk.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith('_delta') or '.prefix_mlp.fc2.' in name:
                param.normal_(std=0.02)
    return model


def _embed(model):
    return model.patch_embed if isinstance(model, NaFlexGenLip) else model.audio_embed


def _encode(model, batch):
    return model.encode_image(batch) if isinstance(model, NaFlexGenLip) else model.encode_audio(batch)


@pytest.mark.parametrize('ar_preserving', [False, True])
def test_factorized_resampling_matches_interpolate(ar_preserving):
    torch.manual_seed(0)
    tables = [torch.randn(8, 16, requires_grad=True) for _ in range(2)]
    refs = [table.detach().clone().requires_grad_() for table in tables]
    shapes = [(5, 3), (2, 7), (11, 13), (1, 9)]  # downsample, upsample, and audio strips
    batch = _patch_batch(shapes, 1)
    out = factorized_pos_embed(*tables, batch['patch_coord'], batch['patch_valid'], ar_preserving)
    expected = []
    for h, w in shapes:
        ly, lx = (max(h, w),) * 2 if ar_preserving else (h, w)
        y, x = [F.interpolate(t.T[None], size=n, mode='linear', align_corners=False)[0].T
                for t, n in zip(refs, (ly, lx))]
        expected.append((y[:h, None] + x[None, :w]).reshape(h * w, -1))
    expected = torch.cat(expected)
    actual = out[batch['patch_valid']]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for table, ref in zip(tables, refs):
        torch.testing.assert_close(table.grad, ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('ar_preserving', [False, True])
def test_factorized_resampling_1d_omits_y(ar_preserving):
    torch.manual_seed(0)
    table_x = torch.randn(8, 16, requires_grad=True)
    ref_x = table_x.detach().clone().requires_grad_()
    shapes = [(1, 5), (1, 11), (1, 1)]
    batch = _patch_batch(shapes, 1)
    out = factorized_pos_embed(None, table_x, batch['patch_coord'], batch['patch_valid'], ar_preserving)
    expected = torch.cat([
        F.interpolate(ref_x.T[None], size=w, mode='linear', align_corners=False)[0].T
        for _, w in shapes
    ])
    actual = out[batch['patch_valid']]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(table_x.grad, ref_x.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('kind', ['image', '1d', '2d'])
@pytest.mark.parametrize('pos_embed', ['none', 'factorized'])
def test_patch_embed_standalone_reset_only_direct_params(kind, pos_embed):
    torch.manual_seed(4)
    kwargs = dict(pos_embed=pos_embed, pos_embed_size=8, input_norm=True, pre_norm=True)
    if kind == 'image':
        embed = GenLipPatchEmbed(NaFlexGenLipVisionCfg(patch_size=2, **kwargs), width=64)
    else:
        cfg = AudioNaFlexCfg(n_mels=4, patch_freq=4 if kind == '1d' else 2, patch_time=2, **kwargs)
        embed = MelPatchEmbed(cfg, width=64)
    for param in embed.pos_embed_params():
        assert torch.isfinite(param).all()
        assert 0.017 < param.std() < 0.023
    batch = _patch_batch([(1, 5), (1, 3)], embed.proj.in_features)
    assert torch.isfinite(embed(**batch)).all()

    # Reset only the owning module's parameters, preserving even modified child weights and norms.
    with torch.no_grad():
        children = {name: param for name, param in embed.named_parameters() if '.' in name}
        for param in children.values():
            param.normal_()
        before = {name: param.clone() for name, param in children.items()}
        for param in embed.pos_embed_params():
            param.fill_(float('nan'))
    embed.reset_parameters()
    for name, param in children.items():
        torch.testing.assert_close(param, before[name], atol=0, rtol=0)
    for param in embed.pos_embed_params():
        assert torch.isfinite(param).all()
        assert 0.017 < param.std() < 0.023


@pytest.mark.parametrize('kind', ['image', '1d', '2d'])
def test_factorized_checkpoint_compatibility_and_init(kind):
    torch.manual_seed(1)
    plain = _model(kind, pos_embed='none')
    assert not any('pos_embed' in name for name in plain.state_dict())
    assert plain.no_weight_decay() == set()
    model = _model(kind)
    embed = _embed(model)
    expected_tables = 1 if kind == '1d' else 2
    assert (embed.pos_embed_y is None) == (kind == '1d')
    assert embed.pos_embed_x is not None
    assert len(embed.pos_embed_params()) == expected_tables
    for param in embed.pos_embed_params():
        assert param.shape == (8, 64)
        assert 0.017 < param.std() < 0.023
    model.init_weights(std=0.1)
    for param in embed.pos_embed_params():
        assert 0.085 < param.std() < 0.115

    missing, unexpected = model.load_state_dict(plain.state_dict(), strict=False)
    assert missing and all('pos_embed_' in name for name in missing)
    assert not unexpected
    with torch.no_grad():
        for param in embed.pos_embed_params():
            param.zero_()
    batch = _patch_batch([(2, 5), (1, 3)], embed.proj.in_features)
    # Zero tables reproduce existing checkpoints exactly, including the optional pre-norm.
    torch.testing.assert_close(_encode(model, batch), _encode(plain, batch), atol=0, rtol=0)
    torch.testing.assert_close(_embed(plain)(batch['patches']), embed(**batch), atol=0, rtol=0)
    with pytest.raises(ValueError, match='patch_coord'):
        embed(batch['patches'])
    exclude = make_wd_exclude(model)
    table_names = {n for n, _ in model.named_parameters() if 'pos_embed_' in n}
    assert model.no_weight_decay() == table_names
    assert len(table_names) == expected_tables
    for name, param in model.named_parameters():
        if name in table_names:
            assert exclude(name, param)
        elif name.endswith('embed.proj.weight'):
            assert not exclude(name, param)


@pytest.mark.parametrize('kind', ['image', '1d', '2d'])
@pytest.mark.parametrize('ar_preserving', [False, True])
def test_factorized_encoder_and_caption_paths(kind, ar_preserving):
    torch.manual_seed(2)
    model = _model(kind, ar_preserving=ar_preserving)
    embed = _embed(model)
    shapes = [(1, 9), (1, 5)] if kind == '1d' else [(2, 9), (2, 5)]
    batch = _patch_batch(shapes, embed.proj.in_features)
    text = torch.tensor([[1, 3, 4, 2], [1, 5, 2, 0]])
    # Padded batch and individual unpadded encodes must infer the same per-sample position scale.
    features = _encode(model, batch)
    for i, (h, w) in enumerate(shapes):
        single = {k: v[i:i + 1, :h * w] for k, v in batch.items()}
        torch.testing.assert_close(features[i:i + 1], _encode(model, single))

    # Full logits, block-layout loss, and both packed in-fusion routes all backprop to both axis tables.
    logits = model(batch, text)['logits']
    assert torch.isfinite(logits).all()
    packed = []
    for pack_prefix, route in [(False, 'mask'), (True, 'mask'), (True, 'gather')]:
        model.pack_prefix = pack_prefix
        model.prefix_routing = route
        model.zero_grad(set_to_none=True)
        loss = model(batch, text, compute_loss=True)['caption_loss_ce']
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad.clone() for p in embed.pos_embed_params()]
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
        if pack_prefix:
            packed.append((loss.detach(), grads))
        else:
            ni = batch['patches'].shape[1]
            expected = F.cross_entropy(logits[:, ni - 1:-1].flatten(0, 1), text.flatten(), ignore_index=0)
            torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(packed[0][0], packed[1][0])
    for masked, gathered in zip(packed[0][1], packed[1][1]):
        torch.testing.assert_close(masked, gathered, atol=1e-6, rtol=1e-4)
    with torch.no_grad():
        for param in embed.pos_embed_params():
            param.zero_()
    assert not torch.allclose(features, _encode(model, batch))
    assert not torch.allclose(logits, model(batch, text)['logits'])


@pytest.mark.parametrize('kind', ['image', '1d', '2d'])
def test_factorized_patch_embed_compile_and_autocast(kind):
    torch.manual_seed(3)
    embed = _embed(_model(kind))
    compiled = torch.compile(embed, backend='aot_eager', fullgraph=True, dynamic=True)
    shape_pairs = [[(1, 7), (1, 4)], [(1, 9), (1, 5)]] if kind == '1d' else [[(2, 7), (3, 4)], [(3, 9), (2, 5)]]
    for shapes in shape_pairs:
        batch = _patch_batch(shapes, embed.proj.in_features)
        for dtype in (torch.float32, torch.bfloat16):
            embed.zero_grad(set_to_none=True)
            with torch.autocast('cpu', dtype=dtype, enabled=dtype == torch.bfloat16):
                expected = embed(**batch)
                out = compiled(**batch)
            assert out.dtype == expected.dtype == dtype
            torch.testing.assert_close(out, expected)
            out.square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in embed.pos_embed_params())


def test_factorized_attentive_probe(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from genlip_attentive_probe import extract_patch_features

    model = _model('image')
    batch = _patch_batch([(2, 7), (2, 4)], model.patch_embed.proj.in_features)
    features, valid = extract_patch_features(model.visual, batch, 'cpu', nullcontext)
    pooled = (features * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
    torch.testing.assert_close(model.visual.proj(pooled), model.encode_image(batch))
