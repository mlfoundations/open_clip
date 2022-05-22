import torch
from open_clip.hf_model import _POOLERS
from transformers.modeling_outputs import BaseModelOutput
#test poolers
def test_poolers():
    bs, sl, d = 2, 10, 5
    h = torch.arange(sl).repeat(bs).reshape(bs, sl)[..., None] * torch.linspace(0.2, 1., d)
    mask = torch.ones(bs, sl, dtype=torch.long)
    mask[:2, 6:] = 0
    x = BaseModelOutput(h)
    for name, cls in _POOLERS.items():
        pooler = cls()
        res = pooler(x, mask)
        assert res.shape == (bs, d), f"{name} returned wrong shape"
