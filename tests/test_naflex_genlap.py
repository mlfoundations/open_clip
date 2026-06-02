"""Tests for the NaFlex GenLAP generative audio-language model (1-D and 2-D rope configs)."""
from types import SimpleNamespace

import pytest
import torch

import open_clip
from open_clip import create_task, get_tokenizer
from open_clip.audio.naflex_audio import mel_to_patches
from open_clip.naflex_genlap_model import NaFlexGenLap, build_audio_position_ids

CONFIG_1D = "naflexgenlap_test_1d"  # full-height strips -> 1-D time rope
CONFIG_2D = "naflexgenlap_test_2d"  # multi-row -> 2-D axial (freq, time) MRoPE


def _mel(t=40, n_mels=64):
    return torch.randn(t, n_mels)


def _audio_batch(model, b=3, t=40):
    """Build a padded NaFlex audio batch from per-sample mel patchify (last sample shorter)."""
    cfg = model.audio_cfg
    dicts = [mel_to_patches(_mel(t=t), cfg.patch_freq, cfg.patch_time, cfg.in_chans) for _ in range(b)]
    dicts[-1] = mel_to_patches(_mel(t=t - 8), cfg.patch_freq, cfg.patch_time, cfg.in_chans)  # shorter
    n = max(d["patches"].shape[0] for d in dicts)
    pd = dicts[0]["patches"].shape[1]
    patches = torch.zeros(b, n, pd)
    coord = torch.zeros(b, n, 2, dtype=torch.long)
    valid = torch.zeros(b, n, dtype=torch.bool)
    for i, d in enumerate(dicts):
        ni = d["patches"].shape[0]
        patches[i, :ni] = d["patches"]
        coord[i, :ni] = d["patch_coord"]
        valid[i, :ni] = True
    return {"patches": patches, "patch_coord": coord, "patch_valid": valid}


def _text(pad_id, b=3, lt=9):
    text = torch.randint(0, 100277, (b, lt))
    text[0, lt - 3:] = pad_id  # padded tail on one sample
    return text, text != pad_id


@pytest.mark.parametrize("config", [CONFIG_1D, CONFIG_2D])
def test_genlap_forward_logits_and_loss(config):
    model = open_clip.create_model(config).eval()
    audio = _audio_batch(model)
    text, text_valid = _text(model.pad_id)

    out = model(audio=audio, text=text, text_valid=text_valid)
    ni = audio["patches"].shape[1]
    seq = ni + text.shape[1]
    assert out["logits"].shape == (3, seq, model.text_cfg.vocab_size)
    assert out["audio_seq_len"] == ni
    assert torch.isfinite(out["logits"]).all()

    loss = model(audio=audio, text=text, text_valid=text_valid, compute_loss=True)["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_genlap_1d_vs_2d_rope_mode():
    model_1d = open_clip.create_model(CONFIG_1D)
    model_2d = open_clip.create_model(CONFIG_2D)
    assert model_1d.rope_1d is True       # full-height strips (patch_freq == n_mels)
    assert model_2d.rope_1d is False      # multi-row (patch_freq < n_mels)
    assert model_1d.audio_cfg.is_1d_time and not model_2d.audio_cfg.is_1d_time


def test_build_audio_position_ids_1d_broadcasts_time():
    # 1-D: every axis carries the time index (full-capacity 1-D rope)
    coord = torch.stack([torch.zeros(5), torch.arange(5)], dim=-1).long().unsqueeze(0)  # (1, 5, 2)
    valid = torch.ones(1, 5, dtype=torch.bool)
    pos = build_audio_position_ids(coord, valid, rope_1d=True)
    assert torch.equal(pos[0, 0], torch.arange(5))
    assert torch.equal(pos[1, 0], torch.arange(5))
    assert torch.equal(pos[2, 0], torch.arange(5))  # all three axes == time


def test_build_audio_position_ids_2d_axial_with_text():
    # 2-D: t=0, h=freq, w=time; text continues past max(freq, time)
    grid = mel_to_patches(_mel(t=40), patch_freq=32, patch_time=4)  # F=2, Tt=10 -> N=20
    pc, pv = grid["patch_coord"].unsqueeze(0), grid["patch_valid"].unsqueeze(0)
    text_valid = torch.ones(1, 5, dtype=torch.bool)
    pos = build_audio_position_ids(pc, pv, text_valid, rope_1d=False)
    assert pos.shape == (3, 1, 25)
    assert pos[0, 0, :20].max().item() == 0   # t-axis inert
    assert pos[1, 0, :20].max().item() == 1   # freq -> 2 rows
    assert pos[2, 0, :20].max().item() == 9   # time -> 10 cols
    assert pos[:, 0, 20].unique().tolist() == [10]  # text starts at max(1, 9) + 1


@pytest.mark.parametrize("config", [CONFIG_1D, CONFIG_2D])
def test_genlap_encode_audio(config):
    model = open_clip.create_model(config).eval()
    audio = _audio_batch(model)
    feats = model.encode_audio(audio)
    assert feats.shape == (3, model.embed_dim)
    assert torch.isfinite(feats).all()
    normed = model.encode_audio(audio, normalize=True)
    assert torch.allclose(normed.norm(dim=-1), torch.ones(3), atol=1e-5)


@pytest.mark.parametrize("name", [
    "naflexgenlip_so150m2", "naflexgenlap_so150m2", "naflexgenlap_betwixt",
])
def test_new_genlp_configs_valid_mrope(name):
    """SBB-shaped GenLIP/GenLAP trial configs: head_dim divides, and MRoPE 2*sum(section)==head_dim."""
    cfg = open_clip.get_model_config(name)
    trunk = cfg.get("genlip_cfg") or cfg.get("genlap_cfg")
    assert trunk["width"] % trunk["num_heads"] == 0, f"{name}: width not divisible by num_heads"
    head_dim = trunk["width"] // trunk["num_heads"]
    assert 2 * sum(trunk["mrope_section"]) == head_dim, f"{name}: 2*sum(mrope_section) != head_dim ({head_dim})"


def test_genlap_tokenizer_is_tiktoken():
    tok = get_tokenizer(CONFIG_1D)
    assert type(tok).__name__ == "TikTokenTokenizer"
    assert tok.pad_token_id == 100278


def test_create_task_dispatches_genlap_and_trains():
    model = open_clip.create_model(CONFIG_1D)
    args = SimpleNamespace(
        model=CONFIG_1D, distill=False, siglip=False, rank=0, world_size=0,
        coca_caption_loss_weight=2.0, coca_contrastive_loss_weight=1.0,
        local_loss=False, gather_with_grad=False, horovod=False, loss_dist_impl=None,
    )
    task = create_task(args, model=model)
    assert type(task).__name__ == "GenLapTask"
    assert task.data_keys == ("audio", "text")
    assert task.fused_loss is True and not hasattr(task, "loss")

    audio = _audio_batch(model)
    text, text_valid = _text(model.pad_id)
    task.train()
    losses = task({"audio": audio, "text": text, "text_valid": text_valid})
    assert "caption_loss" in losses and "loss" in losses and torch.isfinite(losses["loss"])
