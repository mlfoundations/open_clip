"""Tests for the shared NaFlex audio front-end (mel patchify + embed + position-id reuse)."""
import pytest
import torch

from open_clip.audio.naflex_audio import AudioNaFlexCfg, MelPatchEmbed, mel_to_patches
from open_clip.naflex_genlip_model import build_image_position_ids, build_mrope_position_ids
from open_clip_train.naflex_data import NaFlexBatchScheduler


def _mel(t=100, n_mels=64):
    # deterministic ramp so patch content is checkable
    return torch.arange(t * n_mels, dtype=torch.float32).reshape(t, n_mels)


def test_mel_to_patches_full_height_strips_are_1d_time():
    """patch_freq == n_mels -> F == 1 -> a 1-D time-token sequence."""
    mel = _mel(t=100, n_mels=64)
    out = mel_to_patches(mel, patch_freq=64, patch_time=4)
    # T=100 -> Tt = 100 // 4 = 25; F = 64 // 64 = 1 -> N = 25
    assert out["patches"].shape == (25, 1 * 64 * 4)
    assert out["patch_coord"].shape == (25, 2)
    assert out["patch_valid"].dtype == torch.bool and out["patch_valid"].all()
    # freq axis is degenerate (all 0); time runs 0..24 -> effectively 1-D
    assert out["patch_coord"][:, 0].max().item() == 0
    assert out["patch_coord"][:, 1].tolist() == list(range(25))
    # canonical per-patch layout (p_f, p_t): patch ti spans mel time-rows [ti*4 : ti*4+4] across all 64 bins,
    # freq-major (time on the column axis) -> the transpose of the time-major mel block.
    assert torch.equal(out["patches"][3].reshape(64, 4), mel[12:16].t())


def test_mel_to_patches_multi_row_is_2d_grid():
    """patch_freq < n_mels -> F > 1 -> a 2-D (freq, time) grid, freq-outer/time-inner ordering."""
    mel = _mel(t=100, n_mels=64)
    out = mel_to_patches(mel, patch_freq=32, patch_time=4)
    # F = 64 // 32 = 2, Tt = 25 -> N = 50
    assert out["patches"].shape == (50, 1 * 32 * 4)
    coord = out["patch_coord"]
    assert coord[:, 0].max().item() == 1  # two freq rows
    # freq-outer, time-inner: first 25 rows freq=0/time 0..24, next 25 freq=1/time 0..24
    assert coord[:25, 0].tolist() == [0] * 25
    assert coord[25:, 0].tolist() == [1] * 25
    assert coord[:25, 1].tolist() == list(range(25))


def test_mel_to_patches_pads_time_remainder_not_truncated():
    """Audio pads the time axis UP to a whole patch (no dropped tail) -- unlike the image crop."""
    m = _mel(t=103, n_mels=64)
    out = mel_to_patches(m, patch_freq=64, patch_time=4)
    assert out["patches"].shape[0] == 26       # ceil(103/4) == 26 (was 25 truncated)
    assert out["patch_valid"].all()            # padded frames are patch-internal -> patch stays valid
    assert torch.equal(out["patches"][-1].reshape(64, 4)[:, :3], m[100:103].t())  # real frames front-contiguous (time = col)
    # n_mels not divisible by patch_freq is a real config error and still raises
    with pytest.raises(ValueError):
        mel_to_patches(_mel(t=100, n_mels=64), patch_freq=24, patch_time=4)


def test_mel_to_patches_subpatch_clip_yields_one_valid_patch():
    """A clip shorter than one time patch is padded up to a single valid patch (no longer raises)."""
    m = _mel(t=2, n_mels=64)
    out = mel_to_patches(m, patch_freq=64, patch_time=4)     # ceil(2/4) == 1
    assert out["patches"].shape == (1, 1 * 64 * 4) and out["patch_valid"].all()
    patch = out["patches"][0].reshape(64, 4)                  # canonical (p_f, p_t): time on the column axis
    assert torch.equal(patch[:, :2], m.t())                   # real frames front-contiguous
    assert torch.equal(patch[:, 2:], m.new_full((64, 2), m.amin().item()))  # floor pad (not 0 in dB-mel space)


def test_mel_to_patches_pad_modes():
    """floor = per-clip amin; silence = absolute -100 dB floor; repeat = last real frame (all distinct here)."""
    from open_clip.audio.naflex_audio import _MEL_SILENCE_DB

    m = _mel(t=3, n_mels=64)                                  # ceil(3/4) == 1 -> pad 1 frame; ramp amin == 0.0
    fl = mel_to_patches(m, patch_freq=64, patch_time=4, pad_mode="floor")["patches"][0].reshape(64, 4)
    si = mel_to_patches(m, patch_freq=64, patch_time=4, pad_mode="silence")["patches"][0].reshape(64, 4)
    rp = mel_to_patches(m, patch_freq=64, patch_time=4, pad_mode="repeat")["patches"][0].reshape(64, 4)
    for out in (fl, si, rp):                                  # canonical (p_f, p_t): time on the column axis
        assert torch.equal(out[:, :3], m.t())                 # real frames front-contiguous in every mode
    assert torch.equal(fl[:, 3], m.new_full((64,), m.amin().item()))  # floor == per-clip amin (== 0.0 for the ramp)
    assert torch.equal(si[:, 3], m.new_full((64,), _MEL_SILENCE_DB))  # silence == absolute -100, NOT the clip amin
    assert si[:, 3].min().item() == -100.0 and fl[:, 3].min().item() == 0.0  # floor and silence genuinely differ
    assert torch.equal(rp[:, 3], m[2])                            # repeat == last real frame
    with pytest.raises(ValueError):
        mel_to_patches(m, patch_freq=64, patch_time=4, pad_mode="reflect")  # unsupported mode


def test_mel_patch_embed_forward():
    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=32, patch_time=4, input_norm=True)
    embed = MelPatchEmbed(cfg, width=128)
    out = mel_to_patches(_mel(), cfg.patch_freq, cfg.patch_time, cfg.in_chans)
    batched = out["patches"].unsqueeze(0)  # (1, N, patch_dim)
    assert embed(batched).shape == (1, out["patches"].shape[0], 128)
    assert embed.proj.in_features == cfg.patch_dim == 32 * 4


def test_audio_patch_dicts_collate_via_existing_naflex_collator():
    """Per-sample audio dicts are drop-in for the image NaFlex collator (variable N -> padded + valid mask)."""
    short = mel_to_patches(_mel(t=40), patch_freq=64, patch_time=4)   # Tt=10 -> N=10
    long = mel_to_patches(_mel(t=100), patch_freq=64, patch_time=4)   # Tt=25 -> N=25
    batch = NaFlexBatchScheduler._collate_images([short, long], max_seq_len=32)
    assert batch["patches"].shape == (2, 32, 64 * 4)
    assert batch["patch_coord"].shape == (2, 32, 2)
    assert batch["patch_valid"][0].sum().item() == 10  # short sample
    assert batch["patch_valid"][1].sum().item() == 25  # long sample
    assert not batch["patch_valid"][0, 10:].any()       # padding marked invalid


def test_audio_coords_feed_existing_position_builders():
    """patch_coord = (freq, time) plugs straight into the GenLIP image / MRoPE position builders."""
    # 2-D grid case
    grid = mel_to_patches(_mel(t=40), patch_freq=32, patch_time=4)   # F=2, Tt=10 -> N=20
    pc = grid["patch_coord"].unsqueeze(0)                            # (1, N, 2)
    pv = grid["patch_valid"].unsqueeze(0)
    pos = build_image_position_ids(pc, pv)                          # (3, 1, N)
    assert pos.shape == (3, 1, 20)
    assert torch.equal(pos[0], torch.zeros_like(pos[0]))            # t-axis inert for spectrograms
    assert pos[1].max().item() == 1                                 # freq axis -> 2 rows
    assert pos[2].max().item() == 9                                 # time axis -> 10 cols

    # 1-D strip case: freq axis collapses to 0 -> position ids are purely time
    strip = mel_to_patches(_mel(t=40), patch_freq=64, patch_time=4)  # F=1, Tt=10 -> N=10
    pos1d = build_image_position_ids(strip["patch_coord"].unsqueeze(0), strip["patch_valid"].unsqueeze(0))
    assert pos1d[1].max().item() == 0                                # freq inert -> 1-D
    assert pos1d[2].tolist()[0] == list(range(10))

    # generative case: appended text positions start after max(freq, time)
    text_valid = torch.ones(1, 5, dtype=torch.bool)
    mpos = build_mrope_position_ids(pc, pv, text_valid)             # (3, 1, N + 5)
    assert mpos.shape == (3, 1, 25)
    text_start = max(pos[1].max().item(), pos[2].max().item()) + 1  # max(1, 9) + 1 = 10
    assert mpos[:, 0, 20].unique().tolist() == [text_start]


def test_audio_patchify_transform_variable_duration():
    """Full waveform->mel->patches path: longer audio yields more time tokens (no fusion/truncation)."""
    pytest.importorskip("torchaudio")
    from open_clip.audio.naflex_audio import AudioNaFlexPatchify

    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=64, patch_time=4)  # self-contained: mel + geometry
    patchify = AudioNaFlexPatchify(cfg)

    sr = cfg.sample_rate
    short = patchify((torch.randn(1, sr * 1), sr))   # ~1 s
    long = patchify((torch.randn(1, sr * 3), sr))    # ~3 s
    # full-height strips -> 1 token per p_t frames; ~3x duration -> ~3x time tokens
    assert long["patches"].shape[0] > short["patches"].shape[0]
    assert short["patches"].shape[1] == cfg.patch_dim == 64 * 4
    assert short["patch_coord"][:, 0].max().item() == 0  # 1-D time
    # resampling path: a non-native sample rate still produces valid patches
    resampled = patchify((torch.randn(1, 16000 * 1), 16000))
    assert resampled["patches"].shape[1] == cfg.patch_dim
