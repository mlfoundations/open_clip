"""Mixed-precision packing shared by GenLIP and GenLAP."""
import pytest
import torch
import torch.nn.functional as F

from open_clip import get_model_config
from open_clip.naflex_genlip_model import NaFlexGenLip, pack_prefix_sequence
from open_clip.naflex_genlap_model import NaFlexGenLap


@pytest.mark.parametrize("prefix_dtype", [torch.bfloat16, torch.float16])
def test_pack_prefix_matches_cat_dtype_and_gradients(prefix_dtype):
    torch.manual_seed(0)
    prefix = torch.randn(2, 3, 4, dtype=prefix_dtype, requires_grad=True)
    text = torch.randn(2, 2, 4, requires_grad=True)
    prefix_valid = torch.tensor([[True, True, True], [True, False, False]])
    text_valid = torch.tensor([[True, False], [True, True]])
    pos = torch.zeros(3, 2, 5, dtype=torch.long)

    packed = pack_prefix_sequence(prefix, prefix_valid, pos, text, text_valid)[0]
    expected = torch.stack([
        torch.cat([prefix[0], text[0, :1]]),
        F.pad(torch.cat([prefix[1, :1], text[1]]), (0, 0, 0, 1)),
    ])
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    probe = torch.randn_like(expected)
    actual_grads = torch.autograd.grad((packed * probe).sum(), (prefix, text))
    expected_grads = torch.autograd.grad((expected * probe).sum(), (prefix, text))
    torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["genlip", "genlap"])
@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.bfloat16),
    pytest.param("cuda", torch.float16, marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required")),
    pytest.param("cuda", torch.bfloat16, marks=pytest.mark.skipif(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="CUDA bf16 required")),
])
def test_packed_caption_loss_autocast_identity_projection(family, device, dtype):
    torch.manual_seed(0)
    is_audio = family == "genlap"
    cfg = get_model_config("naflexgenlap_test_1d" if is_audio else "naflexgenlip_test")
    trunk_cfg = cfg[f"{family}_cfg"]
    # Production configs have equal widths; the usual tiny configs have a Linear projection,
    # which autocast converts to low precision and therefore hides this packing regression.
    trunk_cfg.update(text_embed_dim=trunk_cfg["width"], pack_prefix=True)
    cfg["text_cfg"].update(vocab_size=64, context_length=4, pad_id=0, bos_id=1, eos_id=2)
    model = (NaFlexGenLap if is_audio else NaFlexGenLip)(**cfg).to(device).eval()
    assert isinstance(model.in_proj, torch.nn.Identity)
    prefix_embed = model.audio_embed if is_audio else model.patch_embed
    patch_dim = model.audio_cfg.patch_dim if is_audio else 16 * 16 * 3
    key = "audio" if is_audio else "image"
    media = {
        "patches": torch.randn(2, 3, patch_dim, device=device),
        "patch_coord": torch.tensor([[0, 0], [0, 1], [0, 2]], device=device).repeat(2, 1, 1),
        "patch_valid": torch.tensor([[True, True, True], [True, False, False]], device=device),
    }
    text = torch.tensor([[1, 3, 4, 2], [1, 2, 0, 0]], device=device)
    batch = {key: media, "text": text, "text_valid": text != 0}
    with torch.autocast(device, dtype=dtype):
        loss = model(**batch, compute_loss=True)["caption_loss_ce"]

    # Removing each row's padding supplies an independent unpacked reference with the same
    # first-token predictor. Its torch.cat also defines the intended stream dtype promotion.
    model.pack_prefix = False
    reference = 0.
    with torch.no_grad(), torch.autocast(device, dtype=dtype):
        for row, (n_prefix, n_text) in enumerate(((3, 4), (1, 2))):
            single = {key: {k: v[row:row + 1, :n_prefix] for k, v in media.items()},
                      "text": text[row:row + 1, :n_text]}
            reference += model(**single, compute_loss=True)["caption_loss_ce"] * n_text / 6
    torch.testing.assert_close(loss, reference, rtol=2e-3, atol=2e-3)
    loss.backward()
    for param in (prefix_embed.proj.weight, model.text_embed.weight, model.lm_head.weight):
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.count_nonzero() > 0
