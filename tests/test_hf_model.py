import pytest

import torch
from open_clip.hf_model import _POOLERS, HFTextEncoder, _auto_model_kwargs
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutput

# test poolers
def test_poolers():
    bs, sl, d = 2, 10, 5
    h = torch.arange(sl).repeat(bs).reshape(bs, sl)[..., None] * torch.linspace(0.2, 1., d)
    mask = torch.ones(bs, sl, dtype=torch.bool)
    mask[:2, 6:] = False
    x = BaseModelOutput(h)
    for name, cls in _POOLERS.items():
        pooler = cls()
        res = pooler(x, mask)
        assert res.shape == (bs, d), f"{name} returned wrong shape"


def test_max_pooler_ignores_padding():
    h = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[True, True, False]])
    x = BaseModelOutput(h)

    assert _POOLERS["max_pooler"]()(x, mask).item() == 3.0


def test_modernbert_hf_text_encoder_supported():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.ModernBertConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    text = torch.tensor([[1, 8, 2, 0, 0], [1, 9, 10, 2, 0]], dtype=torch.long)

    for pooler_type in ("mean_pooler", "cls_pooler", "cls_last_hidden_state_pooler", "max_pooler"):
        model = HFTextEncoder("modernbert", 16, config=cfg, pooler_type=pooler_type, proj_type="linear").eval()
        with torch.no_grad():
            out = model(text)

        assert out.shape == (2, 16)
        assert torch.isfinite(out).all()
        names = [n for n, _ in model.layer_groups()]
        assert names == ["embeddings", "layer.0", "layer.1", "proj"]


def test_modernbert_does_not_request_hf_pooling_layer():
    transformers = pytest.importorskip("transformers")

    assert _auto_model_kwargs(transformers.ModernBertConfig(), "cls_pooler") == {}
    assert _auto_model_kwargs(transformers.BertConfig(), "cls_pooler") == {"add_pooling_layer": True}
    assert _auto_model_kwargs(transformers.BertConfig(), "mean_pooler") == {"add_pooling_layer": False}


def test_hf_text_encoder_ignores_config_pinned_dtype():
    """HF repos may pin a low-precision dtype in their config (gte-modernbert pins float16); the tower must
    still be built fp32 so it composes with the fp32 projection and open_clip's own precision handling."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.ModernBertConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    if hasattr(cfg, "dtype"):  # transformers >= 5 renamed torch_dtype -> dtype
        cfg.dtype = torch.float16
    else:
        cfg.torch_dtype = torch.float16

    model = HFTextEncoder("modernbert", 16, config=cfg, pooler_type="mean_pooler", proj_type="linear").eval()
    assert all(p.dtype == torch.float32 for p in model.transformer.parameters())
    with torch.no_grad():
        out = model(torch.tensor([[1, 8, 2, 0]], dtype=torch.long))
    assert out.dtype == torch.float32


# test HFTextEncoder
@pytest.mark.parametrize("model_id", ["arampacha/roberta-tiny", "roberta-base", "xlm-roberta-base", "google/mt5-base"])
def test_pretrained_text_encoder(model_id):
    bs, sl, d = 2, 10, 64
    cfg = AutoConfig.from_pretrained(model_id)
    model = HFTextEncoder(model_id, d, proj_type='linear')
    x = torch.randint(0, cfg.vocab_size, (bs, sl))
    with torch.no_grad():
        emb = model(x)

    assert emb.shape == (bs, d)
