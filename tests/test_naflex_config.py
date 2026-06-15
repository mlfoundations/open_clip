import pytest

import open_clip
from open_clip.naflex_config import NaFlexDataConfig


def test_naflex_data_config_resolves_train_and_eval_values():
    config = NaFlexDataConfig.resolve(
        patch_sizes=[16, 32],
        patch_size_probs=[1, 3],
        seq_lens=[4, 8],
        train_num_image_tokens=128,
        max_tokens_per_batch=64,
        batch_divisor=2,
    )

    assert config.train_patch_sizes == ((16, 16), (32, 32))
    assert config.train_patch_size_probs == (0.25, 0.75)
    assert config.train_seq_lens == (4, 8)
    assert config.train_num_image_tokens == 128
    assert config.max_tokens_per_batch == 64
    assert config.batch_divisor == 2
    assert config.variable_patch_size
    assert config.eval_config == ((16, 16), 8)


def test_naflex_data_config_accepts_explicit_eval_seq_len():
    config = NaFlexDataConfig.resolve(
        patch_sizes=[16],
        seq_lens=[128, 1024],
        eval_seq_len=576,
    )

    assert config.train_seq_lens == (128, 1024)
    assert config.eval_config == ((16, 16), 576)


def test_siglip2_naflex_configs_default_to_384_dense_and_576_tokens():
    for model_name in ("ViT-B-16-SigLIP2-naflex", "ViT-SO400M-16-SigLIP2-naflex"):
        vision_cfg = open_clip.get_model_config(model_name)["vision_cfg"]
        assert vision_cfg["image_size"] == 384
        assert vision_cfg["image_seq_len"] == 576


def test_naflex_data_config_rejects_negative_patch_size_probs():
    with pytest.raises(ValueError, match="non-negative"):
        NaFlexDataConfig.resolve(
            patch_sizes=[16, 32],
            patch_size_probs=[-1, 2],
            seq_lens=[4],
        )


def test_naflex_data_config_normalizes_seq_len_probs():
    config = NaFlexDataConfig.resolve(seq_lens=[384, 512, 1024], seq_len_probs=[5, 4, 1])
    assert config.train_seq_lens == (384, 512, 1024)
    assert config.train_seq_len_probs == (0.5, 0.4, 0.1)  # normalized, aligned to seq_lens order
    # Unset -> uniform (None), preserving legacy randint sampling.
    assert NaFlexDataConfig.resolve(seq_lens=[384, 512]).train_seq_len_probs is None


def test_naflex_data_config_rejects_mismatched_seq_len_probs():
    with pytest.raises(ValueError, match="match seq-lens length"):
        NaFlexDataConfig.resolve(seq_lens=[384, 512, 1024], seq_len_probs=[0.5, 0.5])


def test_naflex_scheduler_samples_seq_lens_by_weight_deterministically():
    """Scheduler honors seq_len_choice_probs, aligns them after the canonical sort, and is seed-deterministic
    (so the per-batch seq-len schedule stays identical across DDP ranks)."""
    import torch
    from collections import Counter

    from open_clip_train.naflex_data import NaFlexBatchScheduler

    def make(probs):
        return NaFlexBatchScheduler(
            train_num_samples=64, patch_size=16, seq_lens=[1024, 384, 512], seq_len_choice_probs=probs,
            transform_factory=lambda **kw: (lambda x: x), max_tokens_per_batch=4096, batch_divisor=1,
        )

    sched = make([0.1, 0.5, 0.4])  # given in [1024, 384, 512] order -> realigned to sorted seq_lens
    assert sched.seq_lens == [384, 512, 1024]
    assert [round(p, 3) for p in sched.seq_len_probs] == [0.5, 0.4, 0.1]

    gen = torch.Generator().manual_seed(0)
    counts = Counter(sched._next_seq_len(gen) for _ in range(20000))
    frac = {k: counts[k] / sum(counts.values()) for k in counts}
    assert abs(frac[384] - 0.5) < 0.03 and abs(frac[512] - 0.4) < 0.03 and abs(frac[1024] - 0.1) < 0.03

    # Same seed across independent instances -> identical sequence (DDP rank-sync invariant).
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert [make([0.1, 0.5, 0.4])._next_seq_len(g1) for _ in range(40)] == \
           [make([0.1, 0.5, 0.4])._next_seq_len(g2) for _ in range(40)]

    # Unset -> uniform.
    assert make(None).seq_len_probs is None
