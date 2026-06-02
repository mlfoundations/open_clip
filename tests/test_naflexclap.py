"""Tests for NaFlexClap — contrastive CLAP with a NaFlex spectrogram-ViT audio encoder."""
from types import SimpleNamespace

import pytest
import torch

import open_clip
from open_clip import create_task, get_tokenizer
from open_clip.audio.naflex_audio import AudioNaFlexCfg, AudioNaFlexTransformFactory, mel_to_patches
from open_clip.audio.naflex_tower import NaFlexAudioEncoder

CONFIG = "naflexclap_test"


def _audio_batch(b=2, frames=(40, 24), n_mels=64, pf=64, pt=4):
    """Padded NaFlex audio batch from per-sample mel patchify (variable frames -> variable valid patches)."""
    dicts = [mel_to_patches(torch.randn(f, n_mels), patch_freq=pf, patch_time=pt) for f in frames[:b]]
    n = max(d["patches"].shape[0] for d in dicts)
    pd = dicts[0]["patches"].shape[1]
    audio = {
        "patches": torch.zeros(b, n, pd),
        "patch_coord": torch.zeros(b, n, 2, dtype=torch.long),
        "patch_valid": torch.zeros(b, n, dtype=torch.bool),
    }
    for i, d in enumerate(dicts):
        ni = d["patches"].shape[0]
        audio["patches"][i, :ni] = d["patches"]
        audio["patch_coord"][i, :ni] = d["patch_coord"]
        audio["patch_valid"][i, :ni] = True
    return audio


def test_naflexclap_model_forward_and_encode():
    model = open_clip.create_model(CONFIG, output_dict=True)
    assert type(model).__name__ == "CLAP"
    assert isinstance(model.audio.encoder, NaFlexAudioEncoder)
    # vanilla encoder: axial RoPE, not gated (gating is opt-in only)
    assert model.audio.cfg.rope_type == "axial"
    assert model.audio.cfg.naflexvit_cfg.get("attn_gated", False) is False

    dim = model.embed_dim  # joint embed dim (audio/text both project here)
    assert dim == model.audio.proj[-1].out_features
    audio = _audio_batch(b=2, frames=(40, 24))  # variable-length clips
    text = torch.randint(0, model.vocab_size, (2, model.context_length))
    out = model(audio=audio, text=text)
    assert set(out) >= {"audio_features", "text_features", "logit_scale"}
    af, tf = out["audio_features"], out["text_features"]
    assert af.shape == (2, dim) and tf.shape == (2, dim)
    assert torch.isfinite(af).all() and torch.isfinite(tf).all()
    assert torch.allclose(af.norm(dim=-1), torch.ones(2), atol=1e-5)  # L2-normalized
    assert model.encode_audio(audio).shape == (2, dim)
    assert model.encode_text(text).shape == (2, dim)


@pytest.mark.parametrize("pool", ["map", "avg"])
def test_naflexclap_pool_masks_padding(pool):
    """Both attentive ('map') and masked-mean ('avg') pooling ignore padded patches (variable-length safe).

    Corrupt the invalid/padded positions with garbage; valid-token attention + pooling must mask them, so the
    pooled embedding is unchanged.
    """
    from open_clip.audio.config import CLIPAudioCfg
    from open_clip.audio.naflex_tower import NaFlexAudioEncoder

    cfg = CLIPAudioCfg(
        model_type="naflexvit", patch_freq=64, patch_time=4, in_chans=1, rope_type="axial",
        naflexvit_cfg={"embed_dim": 96, "depth": 2, "num_heads": 6, "global_pool": pool},
    )
    enc = NaFlexAudioEncoder(cfg).eval()
    real = mel_to_patches(torch.randn(40, 64), patch_freq=64, patch_time=4)  # 10 valid patches (Tt=10, F=1)
    n_valid, n_total = real["patches"].shape[0], 16
    patches = torch.zeros(1, n_total, cfg.in_chans * 64 * 4)
    coord = torch.zeros(1, n_total, 2, dtype=torch.long)
    valid = torch.zeros(1, n_total, dtype=torch.bool)
    patches[0, :n_valid] = real["patches"]
    coord[0, :n_valid] = real["patch_coord"]
    valid[0, :n_valid] = True

    with torch.no_grad():
        base = enc({"patches": patches, "patch_coord": coord, "patch_valid": valid})["embedding"]
        corrupted = patches.clone()
        corrupted[0, n_valid:] = 999.0  # garbage in the padded (invalid) slots
        leaked = enc({"patches": corrupted, "patch_coord": coord, "patch_valid": valid})["embedding"]
    assert torch.allclose(base, leaked, atol=1e-4), f"pool={pool} leaked padding into the pooled embedding"


def test_naflexclap_create_task_contrastive():
    model = open_clip.create_model(CONFIG, output_dict=True)
    args = SimpleNamespace(
        model=CONFIG, distill=False, siglip=False, rank=0, world_size=0,
        local_loss=False, gather_with_grad=False, coca_caption_loss_weight=2.0,
        coca_contrastive_loss_weight=1.0, loss_dist_impl=None,
    )
    task = create_task(args, model=model)
    assert type(task).__name__ == "CLAPTask"
    assert task.data_keys == ("audio", "text")

    batch = {"audio": _audio_batch(b=2), "text": torch.randint(0, model.vocab_size, (2, model.context_length))}
    task.train()
    losses = task(batch)
    assert "contrastive_loss" in losses and "loss" in losses and torch.isfinite(losses["loss"])


def test_naflexclap_dummy_batch_is_patch_dict():
    model = open_clip.create_model(CONFIG, output_dict=True)
    task = create_task(SimpleNamespace(model=CONFIG, distill=False, siglip=False, rank=0, world_size=0,
                                       local_loss=False, gather_with_grad=False), model=model)
    dummy = task.create_dummy_batch(batch_size=2)
    assert set(dummy["audio"]) == {"patches", "patch_coord", "patch_valid"}  # NOT waveform/longer
    assert dummy["text"].shape == (2, model.context_length)


def test_naflexclap_naflex_audio_batch_via_scheduler():
    """Contrastive audio NaFlex batch (pad_id=None -> fixed-length text) feeds CLAP to a finite loss."""
    pytest.importorskip("torchaudio")
    from open_clip_train.naflex_data import NaFlexBatchScheduler

    model = open_clip.create_model(CONFIG, output_dict=True)
    naflex_cfg = AudioNaFlexCfg.from_clip_audio_cfg(model.audio.cfg)
    tok = get_tokenizer(CONFIG)
    sched = NaFlexBatchScheduler(
        train_num_samples=100,
        patch_size=(naflex_cfg.patch_freq, naflex_cfg.patch_time),
        seq_lens=(256,),
        max_tokens_per_batch=4096,
        transform_factory=AudioNaFlexTransformFactory(naflex_cfg),
        shuffle=False,
        image_key="audio",
        pad_id=None,             # contrastive: fixed-length text via default collate
        per_row_text_tokens=0,
    )
    sr = naflex_cfg.sample_rate
    samples = [
        {"audio": (torch.randn(1, sr * 2), sr), "text": tok("a dog barking")[0]},
        {"audio": (torch.randn(1, sr * 1), sr), "text": tok("ocean waves")[0]},
    ]
    batch = sched.collate_batch(samples, seq_len=256, patch_idx=0)
    assert set(batch["audio"]) >= {"patches", "patch_coord", "patch_valid"}
    assert batch["text"].shape == (2, model.context_length)   # fixed-length, no text_valid
    assert "text_valid" not in batch
    out = model(audio=batch["audio"], text=batch["text"])
    assert torch.isfinite(out["audio_features"]).all() and torch.isfinite(out["text_features"]).all()


@pytest.mark.parametrize(
    "name,embed_dim,depth,num_heads,reg_tokens",
    [("naflexclap_little", 320, 14, 5, 1), ("naflexclap_mediumd", 512, 20, 8, 4)],
)
def test_naflexclap_sbb_shapes(name, embed_dim, depth, num_heads, reg_tokens):
    """SBB narrower-deeper audio-ViT shapes (little / mediumd) build, forward, and carry reg tokens."""
    model = open_clip.create_model(name, output_dict=True).eval()
    vit = model.audio.encoder.vit
    assert model.audio.encoder.embed_dim == embed_dim
    assert len(vit.blocks) == depth
    assert vit.num_prefix_tokens == reg_tokens   # register tokens present (class_token off)

    # 16x16 patches on 64 mels -> 4 freq rows x time; a real 2-D grid for axial rope
    audio = _audio_batch(b=2, frames=(64, 48), n_mels=64, pf=16, pt=16)
    assert audio["patch_coord"][..., 0].max().item() == 64 // 16 - 1  # 4 freq rows (0..3)
    text = torch.randint(0, model.vocab_size, (2, model.context_length))
    out = model(audio=audio, text=text)
    af = out["audio_features"]
    assert af.shape == (2, model.embed_dim) and torch.isfinite(af).all()
    assert torch.allclose(af.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_naflexclap_preprocess_and_params():
    model, ptrain, pval = open_clip.create_model_and_transforms(CONFIG)
    assert isinstance(ptrain, AudioNaFlexTransformFactory)
    assert isinstance(pval, AudioNaFlexTransformFactory)

    from open_clip_train.params import parse_args
    args = parse_args(["--model", CONFIG, "--train-num-samples", "100"])
    assert args.naflexclap is True
    assert args.use_naflex is True
    assert args.force_naflex_vision is False
