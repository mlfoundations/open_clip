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


def test_create_model_loads_genlip_checkpoint(tmp_path):
    """Generative models have no logit_scale / logit_bias attributes; loading their checkpoint must not crash."""
    model = open_clip.create_model('naflexgenlip_test')
    checkpoint_path = tmp_path / 'genlip.pt'
    torch.save(model.state_dict(), checkpoint_path)

    reloaded = open_clip.create_model('naflexgenlip_test', pretrained=str(checkpoint_path))
    for key, value in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], value), key


def test_load_siglip_checkpoint_into_clip_non_strict(tmp_path):
    """A checkpoint with logit_bias loads non-strictly into a model whose logit_bias is None."""
    torch.manual_seed(0)
    siglip = _tiny_clip(init_logit_bias=-10.)
    checkpoint_path = tmp_path / 'siglip.pt'
    torch.save(siglip.state_dict(), checkpoint_path)

    torch.manual_seed(1)
    clip = _tiny_clip()
    assert clip.logit_bias is None
    incompatible = load_checkpoint(clip, str(checkpoint_path), strict=False)
    assert incompatible.unexpected_keys == ['logit_bias']
    assert torch.equal(clip.logit_scale, siglip.logit_scale)
    assert torch.equal(clip.visual.proj, siglip.visual.proj)
