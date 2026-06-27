"""Tests for the GenLAP audio NaFlex data pipeline (transform factory + shared NaFlex batching + wiring)."""
from types import SimpleNamespace

import pytest
import torch

import open_clip
from open_clip import get_tokenizer
from open_clip.audio.naflex_audio import AudioNaFlexCfg, AudioNaFlexPatchify, AudioNaFlexTransformFactory

CONFIG_1D = "naflexgenlap_test_1d"
CONFIG_2D = "naflexgenlap_test_2d"


def _tokens(tok, text):
    return tok(text, pad=False)[0]


def test_audio_naflex_batch_via_scheduler_feeds_model():
    """End-to-end: AudioNaFlexTransformFactory + the modality-agnostic scheduler (primary_key='audio') produce
    a `{audio, text, text_valid}` batch the GenLAP model consumes."""
    pytest.importorskip("torchaudio")
    from open_clip_train.naflex_data import NaFlexBatchScheduler

    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=64, patch_time=4)  # 1-D full-height strips
    tok = get_tokenizer(CONFIG_1D)
    sched = NaFlexBatchScheduler(
        train_num_samples=100,
        patch_size=(cfg.patch_freq, cfg.patch_time),  # cosmetic (audio transform returns the dict directly); (pf, pt) order
        seq_lens=(256,),
        max_tokens_per_batch=4096,
        transform_factory=AudioNaFlexTransformFactory(cfg),
        shuffle=False,
        primary_key="audio",
        pad_id=tok.pad_token_id,
        per_row_text_tokens=64,
    )
    sr = cfg.sample_rate
    samples = [
        {"audio": (torch.randn(1, sr * 2), sr), "text": _tokens(tok, "a dog barking in the distance")},
        {"audio": (torch.randn(1, sr * 1), sr), "text": _tokens(tok, "short clip")},
    ]
    batch = sched.collate_batch(samples, seq_len=256, patch_idx=0)

    audio = batch["audio"]
    assert set(("patches", "patch_coord", "patch_valid")).issubset(audio)
    assert audio["patches"].shape[0] == 2 and audio["patches"].shape[2] == cfg.patch_dim == 64 * 4
    assert "text" in batch and "text_valid" in batch
    assert audio["patch_valid"][0].sum() > audio["patch_valid"][1].sum()  # 2 s clip > 1 s clip

    model = open_clip.create_model(CONFIG_1D).eval()
    loss = model(audio=audio, text=batch["text"], text_valid=batch["text_valid"], compute_loss=True)["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_audio_naflex_2d_token_cap_preserves_freq_rows():
    """2-D cap must crop whole time columns (max_time = max_tokens // freq_tokens), keeping every freq row."""
    pytest.importorskip("torchaudio")
    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=32, patch_time=4)  # F = 2
    patchify = AudioNaFlexPatchify(cfg, max_audio_tokens=64)      # max_time = 64 // 2 = 32 columns
    sr = cfg.sample_rate
    out = patchify((torch.randn(1, sr * 10), sr))  # long clip would overflow without the cap
    n = out["patches"].shape[0]
    assert n <= 64
    assert n % cfg.freq_tokens == 0                      # whole time columns only
    assert out["patch_coord"][:, 0].max().item() == 1    # BOTH freq rows present (no dropped row)


def test_caption_length_bucketer_sorts_by_caption():
    """LengthBucketer([CaptionLength]) sorts by caption length (the GenLIP / fixed-modality case)."""
    from open_clip_train.naflex_data import CaptionLength, LengthBucketer

    caps = [5, 9, 2, 3]
    samples = [{"text": torch.zeros(c, dtype=torch.long)} for c in caps]
    bucketer = LengthBucketer(length_fns=[CaptionLength("text")], pool=10, chunk=10, seed=0, epoch=0)
    assert bucketer._length(samples[0]) == 5

    out = list(bucketer(iter(samples)))
    assert len(out) == len(samples)  # permutation, nothing dropped/duplicated
    assert [s["text"].shape[0] for s in out] == sorted(caps)  # sorted by caption length


def test_length_bucketer_prefetch_matches_sync():
    """The threaded prefetch path (prefetch_pools>0) yields the IDENTICAL sample sequence as the synchronous
    path -- it only overlaps disk/decode, so ordering and contents must not change."""
    from open_clip_train.naflex_data import CaptionLength, LengthBucketer

    samples = [{"text": torch.zeros(i % 11 + 1, dtype=torch.long), "id": i} for i in range(97)]

    def order(prefetch_pools):
        bucketer = LengthBucketer(
            length_fns=[CaptionLength("text")], pool=16, chunk=4, seed=0, epoch=0,
            prefetch_pools=prefetch_pools,
        )
        return [s["id"] for s in bucketer(iter(samples))]

    sync = order(0)
    assert sorted(sync) == list(range(97))  # permutation, nothing dropped/duplicated
    assert order(1) == sync                 # 1 pool ahead -> identical
    assert order(2) == sync                 # deeper buffer -> still identical


def test_length_bucketer_prefetch_propagates_upstream_error():
    """An exception from the upstream iterator surfaces on the consumer thread, after buffered output drains."""
    from open_clip_train.naflex_data import CaptionLength, LengthBucketer

    class Boom(Exception):
        pass

    def src():
        for i in range(4):  # exactly one pool (pool=4) before the error
            yield {"text": torch.zeros(i + 1, dtype=torch.long), "id": i}
        raise Boom("upstream failed")

    bucketer = LengthBucketer(
        length_fns=[CaptionLength("text")], pool=4, chunk=2, seed=0, epoch=0, prefetch_pools=1,
    )
    collected = []
    with pytest.raises(Boom):
        for sample in bucketer(src()):
            collected.append(sample)
    assert len(collected) == 4  # the buffered pool still drained before the error propagated


def test_length_bucketer_prefetch_early_close_winds_down_thread():
    """Closing the generator after a partial read must not deadlock, and the daemon producer thread winds down
    (no leak across epochs under persistent_workers)."""
    import threading
    import time
    from open_clip_train.naflex_data import CaptionLength, LengthBucketer

    def src():  # infinite source
        i = 0
        while True:
            i += 1
            yield {"text": torch.zeros(i % 7 + 1, dtype=torch.long)}

    baseline = threading.active_count()
    gen = LengthBucketer(
        length_fns=[CaptionLength("text")], pool=8, chunk=2, seed=0, epoch=0, prefetch_pools=1,
    )(src())
    pulled = [next(gen) for _ in range(5)]
    assert len(pulled) == 5
    gen.close()  # must return (no deadlock)

    deadline = time.monotonic() + 3.0  # producer exits within `poll` (1s); allow margin
    while threading.active_count() > baseline and time.monotonic() < deadline:
        time.sleep(0.05)
    assert threading.active_count() <= baseline  # prefetch thread is gone


def test_prefetch_delivers_sentinel_to_slow_consumer():
    """End-of-stream must not hang: a slow (decode-bound) consumer leaves the queue full when the producer
    finishes, so the producer has to RETRY the done-sentinel until delivered -- dropping it on the first Full
    leaves the consumer blocked on its final get() forever."""
    import threading
    import time
    from open_clip_train.naflex_data import _prefetch

    def make_pools(stop):
        for pool in ([1, 2], [3, 4], [5, 6]):
            yield pool

    out = []
    finished = threading.Event()

    def consume():
        # small poll + a consumer slower than poll => when the producer reaches its finally the queue is full,
        # so the sentinel put hits Full and must be retried (not dropped).
        for s in _prefetch(make_pools, maxsize=1, poll=0.02):
            time.sleep(0.05)
            out.append(s)
        finished.set()

    threading.Thread(target=consume, daemon=True).start()
    assert finished.wait(timeout=10), "prefetch hung at end-of-stream (done sentinel dropped on a full queue)"
    assert out == [1, 2, 3, 4, 5, 6]


def test_generative_audio_bucketer_sorts_by_audio_plus_caption():
    """Generative GenLAP keys on the SUM audio_tokens + caption (LengthBucketer sorts by sum(length_fns))."""
    from open_clip_train.naflex_data import AudioTokenLength, CaptionLength, LengthBucketer

    sr = 16000
    durs = [3, 1, 2, 1]
    caps = [5, 9, 2, 4]
    samples = [
        {"audio": (torch.randn(1, sr * d), sr), "text": torch.zeros(c, dtype=torch.long)}
        for d, c in zip(durs, caps)
    ]
    # freq_tokens=1, patch_time=1, hop_size=sr -> audio_tokens = (samples // sr) + 1 = duration_sec + 1
    length_fns = [
        AudioTokenLength(freq_tokens=1, patch_time=1, hop_size=sr, sample_rate=sr),
        CaptionLength("text"),
    ]
    bucketer = LengthBucketer(length_fns=length_fns, pool=10, chunk=10, seed=0, epoch=0)
    assert bucketer._length(samples[0]) == (3 + 1) + 5  # audio_tokens (dur+1) + caption

    sums = [(d + 1) + c for d, c in zip(durs, caps)]  # [9, 11, 5, 6]
    out = list(bucketer(iter(samples)))
    assert len(out) == len(samples)
    assert [bucketer._length(s) for s in out] == sorted(sums)  # sorted by the token sum


def test_audio_token_length_clamps_long_clips():
    """AudioTokenLength clamps its estimate to max_audio_tokens, so overflowing clips sort by caption rather
    than an outlier duration (the transform truncates them to that cap anyway)."""
    from open_clip_train.naflex_data import AudioTokenLength, CaptionLength, LengthBucketer

    sr = 16000
    length_fns = [
        AudioTokenLength(freq_tokens=1, patch_time=1, hop_size=sr, sample_rate=sr, max_audio_tokens=5),
        CaptionLength("text"),
    ]
    bucketer = LengthBucketer(length_fns=length_fns, pool=10, chunk=10, seed=0, epoch=0)
    long_short_cap = {"audio": (torch.randn(1, sr * 100), sr), "text": torch.zeros(2, dtype=torch.long)}
    long_long_cap = {"audio": (torch.randn(1, sr * 50), sr), "text": torch.zeros(4, dtype=torch.long)}
    # both audio estimates (101, 51) clamp to 5 -> keys are 5+2 and 5+4, NOT 103 vs 55
    assert bucketer._length(long_short_cap) == 5 + 2
    assert bucketer._length(long_long_cap) == 5 + 4


def test_audio_transform_factory_carries_pack_prefix():
    """Concern 1: pack_prefix rides on the transform factory (resolved from the model in _build_preprocess), so
    the data path reads it off the transform it already gets -- robust to local-dir:/hf-hub:/override configs."""
    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=64, patch_time=4)
    assert AudioNaFlexTransformFactory(cfg).pack_prefix is False                  # default off
    assert AudioNaFlexTransformFactory(cfg, pack_prefix=True).pack_prefix is True

    from open_clip.factory import _build_preprocess
    model = open_clip.create_model(CONFIG_1D)
    model.pack_prefix = True  # a packed GenLAP, regardless of how its config was sourced
    train, val = _build_preprocess(model)
    assert train.pack_prefix is True and val.pack_prefix is True


def test_audio_token_length_estimate_matches_actual_patch_count():
    """The AudioTokenLength estimate must equal the patch count the transform actually emits (ceil), for short
    AND remainder clips -- otherwise audio bucketing mis-sizes the per-batch max."""
    pytest.importorskip("torchaudio")
    from open_clip_train.naflex_data import AudioTokenLength

    # patch_time=2 (not 4) so a dropped window_size would actually diverge on the sub-window clip below; and pass
    # window_size like production does, so this mirrors the real bucketer wiring (not an accidental match).
    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=32, patch_time=2)  # F = freq_tokens = 2 (2-D grid)
    patchify = AudioNaFlexPatchify(cfg)
    audio_len = AudioTokenLength(
        audio_key="audio", freq_tokens=cfg.freq_tokens, patch_time=cfg.patch_time,
        hop_size=cfg.hop_size, window_size=cfg.window_size, sample_rate=cfg.sample_rate,
    )
    sr = cfg.sample_rate
    for n_samples in (200, sr // 10, sr, sr * 3 + 137):  # sub-patch, short, 1s, 3s + remainder
        wav = torch.randn(1, n_samples)
        actual = patchify((wav, sr))["patches"].shape[0]
        est = audio_len({"audio": (wav, sr)})
        assert est == actual, f"n_samples={n_samples}: estimate {est} != actual {actual}"


def test_audio_naflex_seq_len_below_freq_tokens_raises():
    """A seq-len bucket smaller than freq_tokens fails fast (else collate truncation drops freq rows)."""
    cfg = AudioNaFlexCfg(n_mels=64, patch_freq=16, patch_time=16)  # freq_tokens = 4
    factory = AudioNaFlexTransformFactory(cfg)
    with pytest.raises(ValueError, match="freq_tokens"):
        factory(max_seq_len=3, patch_size=None)  # 3 < 4 -> raises in AudioNaFlexPatchify.__init__


def test_clip_audio_cfg_propagates_patch_pad_mode():
    """NaFlexClap selects patch_pad_mode via CLIPAudioCfg; from_clip_audio_cfg must carry it to the transform cfg."""
    from open_clip.audio.config import CLIPAudioCfg

    assert AudioNaFlexCfg.from_clip_audio_cfg(CLIPAudioCfg()).patch_pad_mode == "floor"  # default
    cfg = AudioNaFlexCfg.from_clip_audio_cfg(CLIPAudioCfg(patch_pad_mode="silence"))
    assert cfg.patch_pad_mode == "silence"
    assert AudioNaFlexTransformFactory(cfg).cfg.patch_pad_mode == "silence"  # reaches the actual patchify transform


def test_synthetic_audio_rejects_naflex_transform():
    """synthetic-audio is NOT supported for NaFlex audio models (GenLAP/NaFlexClap) -> fail loudly, not with a
    confusing 'AudioNaFlexPatchify object is not subscriptable' collate error."""
    from open_clip_train.audio_data import get_synthetic_audio_dataset

    factory = AudioNaFlexTransformFactory(AudioNaFlexCfg(n_mels=64, patch_freq=64, patch_time=4))
    args = SimpleNamespace(model=CONFIG_1D, train_num_samples=4, batch_size=2, workers=0, distributed=False)
    with pytest.raises(NotImplementedError, match="NaFlex"):
        get_synthetic_audio_dataset(args, factory, is_train=True)


def test_create_model_and_transforms_returns_audio_naflex_factory():
    """GenLAP must not crash create_model_and_transforms (no .audio/.visual) and gets the audio factory."""
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(CONFIG_1D)
    assert type(model).__name__ == "NaFlexGenLap"
    assert isinstance(preprocess_train, AudioNaFlexTransformFactory)
    assert isinstance(preprocess_val, AudioNaFlexTransformFactory)


def test_params_genlap_enables_naflex():
    from open_clip_train.params import parse_args

    args = parse_args(["--model", CONFIG_1D, "--train-num-samples", "100"])
    assert args.genlap is True
    assert args.use_naflex is True
    assert args.force_naflex_vision is False


def test_contrastive_audio_bucketer_reorders_by_audio_tokens():
    """NaFlexClap (constant text) must bucket on audio_tokens, not the constant caption: the audio-keyed
    LengthBucketer reorders by duration, a caption-keyed one cannot (the bug we fixed)."""
    from open_clip_train.naflex_data import AudioTokenLength, CaptionLength, LengthBucketer

    sr = 48000
    audio_len = AudioTokenLength(
        audio_key="audio", freq_tokens=8, patch_time=16, hop_size=480, window_size=1024, sample_rate=sr,
    )
    durs = [1.0, 9.0, 2.0, 8.0, 1.5, 9.5, 3.0, 7.0]
    samples = [{"audio": (torch.randn(1, int(d * sr)), sr), "text": torch.zeros(77, dtype=torch.long)} for d in durs]

    # pool=chunk=len -> a single sorted chunk (no shuffle), so output order == sorted-by-_length order.
    audio_b = LengthBucketer(length_fns=[audio_len], pool=len(samples), chunk=len(samples), seed=0, epoch=0)
    out = [round(s["audio"][0].shape[-1] / sr, 1) for s in audio_b(list(samples))]
    assert out == sorted(durs), out  # audio-keyed -> grouped by duration

    cap_b = LengthBucketer(length_fns=[CaptionLength("text")], pool=len(samples), chunk=len(samples), seed=0, epoch=0)
    assert {cap_b._length(s) for s in samples} == {77}  # constant caption -> no signal
    assert [round(s["audio"][0].shape[-1] / sr, 1) for s in cap_b(list(samples))] == durs  # stable -> no reorder


def test_audio_collate_modulus_batch_max_and_clamp():
    """Audio collate pads to min(ceil(batch_max/M)*M, F*(S//F)): image mode -> seq_len; M=None -> batch_max;
    M set -> modulus, clamped at the per-batch cap; always >= batch_max (holds all tokens)."""
    from open_clip_train.naflex_data import NaFlexBatchScheduler

    F = 8

    def pd(n, d=8):
        return {
            "patches": torch.ones(n, d),
            "patch_coord": torch.zeros(n, 2, dtype=torch.long),
            "patch_valid": torch.ones(n, dtype=torch.bool),
        }

    dicts = [pd(104), pd(456), pd(56), pd(400)]  # batch_max = 456, cap = F*(504//F) = 504
    collate = NaFlexBatchScheduler._collate_patch_dicts
    assert collate(dicts, 504, None, None)["seq_len"] == 504          # image: pad to seq_len
    assert collate(dicts, 504, None, F)["seq_len"] == 456             # audio M=None: batch_max
    assert collate(dicts, 504, 64, F)["seq_len"] == 504               # ceil(456/64)*64=512 -> clamp to cap 504
    assert collate([pd(496)], 504, 64, F)["seq_len"] == 504           # ceil(496/64)*64=512 -> clamp to 504
    tight = [pd(400), pd(408), pd(392)]
    assert collate(tight, 504, 64, F)["seq_len"] == 448               # ceil(408/64)*64=448, no clamp

    out = collate(dicts, 504, 64, F)  # never drops a valid token
    assert int(out["patch_valid"].sum()) == sum(d["patches"].shape[0] for d in dicts)
