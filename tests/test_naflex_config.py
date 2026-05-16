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
