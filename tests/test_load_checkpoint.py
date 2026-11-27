import pytest
import torch

import open_clip
from open_clip import CLIP, load_checkpoint


def _tiny_clip(init_logit_bias=None):
    return CLIP(
        embed_dim=16,
        vision_cfg=dict(image_size=32, patch_size=8, width=32, layers=1, head_width=16),
        text_cfg=dict(context_length=8, vocab_size=64, width=32, heads=2, layers=1),
        init_logit_bias=init_logit_bias,
    )


@pytest.mark.parametrize('model_name', ['naflexgenlip_test', 'naflexgenlap_test_1d'])
def test_create_model_loads_generative_checkpoint(tmp_path, model_name):
    model = open_clip.create_model(model_name)
    ckpt = tmp_path / f'{model_name}.pt'
    torch.save(model.state_dict(), ckpt)

    reloaded = open_clip.create_model(model_name, pretrained=str(ckpt))
    for k, v in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[k], v), k


def test_load_siglip_checkpoint_into_clip_non_strict(tmp_path):
    torch.manual_seed(0)
    siglip = _tiny_clip(init_logit_bias=-10.)
    ckpt = tmp_path / 'siglip.pt'
    torch.save(siglip.state_dict(), ckpt)

    torch.manual_seed(1)
    clip = _tiny_clip()
    assert not torch.equal(clip.visual.proj, siglip.visual.proj)
    incompatible = load_checkpoint(clip, str(ckpt), strict=False)
    assert incompatible.unexpected_keys == ['logit_bias']
    assert incompatible.missing_keys == []
    assert torch.equal(clip.visual.proj, siglip.visual.proj)


def test_load_checkpoint_without_logit_scale_into_siglip_non_strict(tmp_path):
    torch.manual_seed(0)
    sd = {k: v for k, v in _tiny_clip().state_dict().items() if k not in ('logit_scale', 'logit_bias')}
    ckpt = tmp_path / 'partial.pt'
    torch.save(sd, ckpt)

    siglip = _tiny_clip(init_logit_bias=-10.)
    incompatible = load_checkpoint(siglip, str(ckpt), strict=False)
    assert set(incompatible.missing_keys) == {'logit_scale', 'logit_bias'}
    assert incompatible.unexpected_keys == []
    assert siglip.logit_bias.item() == -10.


def test_load_clip_checkpoint_into_audio_only_model_non_strict(tmp_path):
    ckpt = tmp_path / 'clip.pt'
    torch.save(_tiny_clip().state_dict(), ckpt)

    genlap = open_clip.create_model('naflexgenlap_test_1d')
    incompatible = load_checkpoint(genlap, str(ckpt), strict=False)
    assert 'visual.positional_embedding' in incompatible.unexpected_keys


def test_load_checkpoint_preserves_persistent_position_ids(tmp_path):
    def _make():
        m = torch.nn.Module()
        m.text = torch.nn.Module()
        m.text.transformer = torch.nn.Module()
        m.text.transformer.embeddings = torch.nn.Module()
        m.text.transformer.embeddings.register_buffer('position_ids', torch.arange(8).unsqueeze(0), persistent=True)
        return m

    src, dst = _make(), _make()
    src.text.transformer.embeddings.position_ids.fill_(5)
    ckpt = tmp_path / 'pos_ids.pt'
    torch.save(src.state_dict(), ckpt)

    incompatible = load_checkpoint(dst, str(ckpt), strict=True)
    assert incompatible.missing_keys == [] and incompatible.unexpected_keys == []
    assert torch.equal(dst.text.transformer.embeddings.position_ids, src.text.transformer.embeddings.position_ids)
