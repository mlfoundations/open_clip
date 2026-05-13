import types

import pytest
import torch
import torch.nn as nn

from open_clip import factory
from open_clip_train.naflex_data import (
    NAFLEX_AVAILABLE,
    create_naflex_data_config_from_args,
    get_naflex_model_patch_size,
)
from open_clip_train.params import parse_args


class _DummyTimmTrunk(nn.Module):
    num_features = 4
    default_cfg = {}

    def forward(self, x):
        return torch.zeros(x.shape[0], self.num_features, device=x.device, dtype=x.dtype)


def _tiny_timm_clip_config(timm_model_name):
    return {
        "embed_dim": 4,
        "vision_cfg": {
            "image_size": 224,
            "timm_model_name": timm_model_name,
            "timm_model_pretrained": False,
            "timm_pool": "avg",
            "timm_proj": "none",
        },
        "text_cfg": {
            "context_length": 4,
            "vocab_size": 16,
            "width": 4,
            "heads": 1,
            "layers": 1,
        },
    }


def test_force_naflex_vision_passes_use_naflex_to_timm(monkeypatch):
    captured = {}

    def _create_model(_model_name, **kwargs):
        captured.update(kwargs)
        return _DummyTimmTrunk()

    monkeypatch.setitem(factory._MODEL_CONFIGS, "test-eva-naflex", _tiny_timm_clip_config("eva02_test"))
    monkeypatch.setattr("open_clip.timm_model.timm.create_model", _create_model)

    factory.create_model(
        "test-eva-naflex",
        load_weights=False,
        force_naflex_vision=True,
    )

    assert captured["use_naflex"] is True


def test_force_naflex_vision_rejects_non_timm_model():
    with pytest.raises(RuntimeError, match="requires a timm vision tower"):
        factory.create_model(
            "RN50",
            load_weights=False,
            force_naflex_vision=True,
        )


def test_parse_use_naflex_enables_timm_naflex_aug_cfg():
    args = parse_args(["--use-naflex"])

    assert args.force_naflex_vision is True
    assert args.aug_cfg["use_timm"] is True
    assert args.aug_cfg["naflex"] is True


def test_parse_force_naflex_vision_does_not_enable_naflex_data_pipeline():
    args = parse_args(["--force-naflex-vision"])

    assert args.force_naflex_vision is True
    assert args.use_naflex is False
    assert "naflex" not in args.aug_cfg
    assert "use_timm" not in args.aug_cfg


def test_naflex_data_config_defaults_to_model_patch_size():
    model = types.SimpleNamespace(
        visual=types.SimpleNamespace(
            trunk=types.SimpleNamespace(get_patch_size=lambda: (14, 14)),
        ),
    )
    args = types.SimpleNamespace(
        naflex_patch_sizes=None,
        naflex_patch_size_probs=None,
        naflex_seq_lens=[4],
    )

    config = create_naflex_data_config_from_args(
        args,
        default_patch_size=get_naflex_model_patch_size(model),
    )

    assert config.train_patch_sizes == ((14, 14),)
    assert config.eval_patch_size == (14, 14)


@pytest.mark.skipif(not NAFLEX_AVAILABLE, reason="timm NaFlex data support is not available")
def test_convert_naflex_timm_state_dict_maps_patch_embed_weight():
    from timm.models.naflexvit import NaFlexVit, NaFlexVitCfg

    trunk = NaFlexVit(
        cfg=NaFlexVitCfg(
            patch_size=16,
            embed_dim=4,
            depth=1,
            num_heads=1,
            class_token=True,
            global_pool="token",
        ),
        img_size=32,
        num_classes=4,
    )
    model = types.SimpleNamespace(visual=types.SimpleNamespace(trunk=trunk))
    state_dict = {
        "visual.trunk.patch_embed.proj.weight": torch.randn(4, 3, 16, 16),
        "text.token_embedding.weight": torch.randn(16, 4),
    }

    converted = factory._convert_naflex_timm_state_dict(model, state_dict)

    assert "visual.trunk.patch_embed.proj.weight" not in converted
    assert converted["visual.trunk.embeds.proj.weight"].shape == (4, 16 * 16 * 3)
    assert "text.token_embedding.weight" in converted
