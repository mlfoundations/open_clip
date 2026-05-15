import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from open_clip import factory
from open_clip import naflex_convert
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


def _tiny_native_vit_clip_config():
    return {
        "embed_dim": 4,
        "vision_cfg": {
            "image_size": 32,
            "patch_size": 16,
            "width": 4,
            "layers": 1,
            "head_width": 4,
            "mlp_ratio": 2.0,
        },
        "text_cfg": {
            "context_length": 4,
            "vocab_size": 16,
            "width": 4,
            "heads": 1,
            "layers": 1,
        },
    }


def _tiny_native_vit_quickgelu_clip_config():
    config = _tiny_native_vit_clip_config()
    config["quick_gelu"] = True
    return config


def _tiny_naflex_siglip2_clip_config():
    config = _tiny_timm_clip_config("naflexvit_base_patch16_siglip")
    config.update({
        "custom_text": True,
        "init_logit_bias": -10,
    })
    config["vision_cfg"].update({
        "image_size": 32,
        "timm_pool": "map",
        "timm_model_kwargs": {
            "patch_size": 16,
            "embed_dim": 4,
            "depth": 1,
            "num_heads": 1,
            "mlp_ratio": 2.0,
            "pos_embed_grid_size": (2, 2),
        },
    })
    config["text_cfg"].update({
        "no_causal_mask": True,
        "proj_bias": True,
        "pool_type": "last",
        "mlp_ratio": 2.0,
    })
    return config


def _write_tiny_big_vision_npz(path):
    rng = np.random.default_rng(0)
    width = 4
    mlp_width = 8
    vocab_size = 16
    context_length = 4
    patch_dim = 16 * 16 * 3

    def rand(shape):
        return (0.02 * rng.standard_normal(shape)).astype(np.float32)

    def add_encoder_block(weights, prefix, layers=1):
        weights[f"{prefix}LayerNorm_0/scale"] = np.ones((layers, width), dtype=np.float32)
        weights[f"{prefix}LayerNorm_0/bias"] = np.zeros((layers, width), dtype=np.float32)
        weights[f"{prefix}LayerNorm_1/scale"] = np.ones((layers, width), dtype=np.float32)
        weights[f"{prefix}LayerNorm_1/bias"] = np.zeros((layers, width), dtype=np.float32)
        weights[f"{prefix}MultiHeadDotProductAttention_0/query/kernel"] = rand((layers, width, 1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/key/kernel"] = rand((layers, width, 1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/value/kernel"] = rand((layers, width, 1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/query/bias"] = rand((layers, 1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/key/bias"] = rand((layers, 1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/value/bias"] = rand((layers, 1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/out/kernel"] = rand((layers, 1, width, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/out/bias"] = rand((layers, width))
        weights[f"{prefix}MlpBlock_0/Dense_0/kernel"] = rand((layers, width, mlp_width))
        weights[f"{prefix}MlpBlock_0/Dense_0/bias"] = rand((layers, mlp_width))
        weights[f"{prefix}MlpBlock_0/Dense_1/kernel"] = rand((layers, mlp_width, width))
        weights[f"{prefix}MlpBlock_0/Dense_1/bias"] = rand((layers, width))

    def add_map_head(weights, prefix):
        weights[f"{prefix}probe"] = rand((1, 1, width))
        weights[f"{prefix}LayerNorm_0/scale"] = np.ones((width,), dtype=np.float32)
        weights[f"{prefix}LayerNorm_0/bias"] = np.zeros((width,), dtype=np.float32)
        weights[f"{prefix}MultiHeadDotProductAttention_0/query/kernel"] = rand((width, 1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/key/kernel"] = rand((width, 1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/value/kernel"] = rand((width, 1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/query/bias"] = rand((1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/key/bias"] = rand((1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/value/bias"] = rand((1, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/out/kernel"] = rand((1, width, width))
        weights[f"{prefix}MultiHeadDotProductAttention_0/out/bias"] = rand((width,))
        weights[f"{prefix}MlpBlock_0/Dense_0/kernel"] = rand((width, mlp_width))
        weights[f"{prefix}MlpBlock_0/Dense_0/bias"] = rand((mlp_width,))
        weights[f"{prefix}MlpBlock_0/Dense_1/kernel"] = rand((mlp_width, width))
        weights[f"{prefix}MlpBlock_0/Dense_1/bias"] = rand((width,))

    weights = {
        "params/b": np.array([-1.0], dtype=np.float32),
        "params/t": np.array([2.0], dtype=np.float32),
        "params/img/embedding/kernel": rand((patch_dim, width)),
        "params/img/embedding/bias": rand((width,)),
        "params/img/pos_embedding": rand((2, 2, width)),
        "params/img/Transformer/encoder_norm/scale": np.ones((width,), dtype=np.float32),
        "params/img/Transformer/encoder_norm/bias": np.zeros((width,), dtype=np.float32),
        "params/txt/Embed_0/embedding": rand((vocab_size, width)),
        "params/txt/pos_embedding": rand((1, context_length, width)),
        "params/txt/Encoder_0/encoder_norm/scale": np.ones((width,), dtype=np.float32),
        "params/txt/Encoder_0/encoder_norm/bias": np.zeros((width,), dtype=np.float32),
        "params/txt/head/kernel": rand((width, width)),
        "params/txt/head/bias": rand((width,)),
    }
    add_encoder_block(weights, "params/img/Transformer/encoderblock/")
    add_encoder_block(weights, "params/txt/Encoder_0/encoderblock/")
    add_map_head(weights, "params/img/MAPHead_0/")
    np.savez(path, **weights)


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


@pytest.mark.skipif(not NAFLEX_AVAILABLE, reason="timm NaFlex data support is not available")
def test_force_naflex_vision_converts_native_vit_config():
    config = _tiny_native_vit_clip_config()
    naflex_convert.apply_naflex_vision_config(config)

    vision_cfg = config["vision_cfg"]
    assert vision_cfg["timm_model_name"] == "vit_base_patch16_clip_224"
    assert vision_cfg["timm_pool"] == "token"
    assert vision_cfg["timm_proj"] == "linear"
    assert vision_cfg["timm_model_kwargs"]["use_naflex"] is True
    assert vision_cfg["timm_model_kwargs"]["patch_size"] == 16
    assert vision_cfg["timm_model_kwargs"]["embed_dim"] == 4
    assert vision_cfg["timm_model_kwargs"]["depth"] == 1
    assert vision_cfg["timm_model_kwargs"]["num_heads"] == 1
    assert vision_cfg["timm_model_kwargs"]["pos_embed_grid_size"] == (2, 2)


def test_force_naflex_vision_rejects_non_vit_model():
    with pytest.raises(RuntimeError, match="standard native OpenCLIP/OpenAI ViT"):
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


def test_builtin_naflex_siglip2_configs_select_naflex_towers():
    expected_towers = {
        "ViT-B-16-SigLIP2-naflex": "naflexvit_base_patch16_siglip",
        "ViT-SO400M-16-SigLIP2-naflex": "naflexvit_so400m_patch16_siglip",
    }

    for model_name, timm_model_name in expected_towers.items():
        config = factory.get_model_config(model_name)

        assert config["vision_cfg"]["timm_model_name"] == timm_model_name
        assert config["vision_cfg"]["timm_pool"] == "map"
        assert config["vision_cfg"]["timm_proj"] == "none"


@pytest.mark.skipif(not NAFLEX_AVAILABLE, reason="timm NaFlex data support is not available")
def test_big_vision_npz_loads_naflex_siglip2_tower(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "tiny_siglip2_naflex.npz"
    _write_tiny_big_vision_npz(checkpoint_path)
    monkeypatch.setitem(factory._MODEL_CONFIGS, "test-naflex-siglip2", _tiny_naflex_siglip2_clip_config())

    model = factory.create_model(
        "test-naflex-siglip2",
        pretrained=str(checkpoint_path),
        output_dict=True,
    )
    model.eval()

    trunk = model.visual.trunk
    checkpoint = np.load(checkpoint_path)
    expected_patch_weight = torch.from_numpy(checkpoint["params/img/embedding/kernel"].T)
    assert trunk.__class__.__name__ == "NaFlexVit"
    assert torch.equal(trunk.embeds.proj.weight, expected_patch_weight)
    assert trunk.embeds.pos_embed.shape == (1, 2, 2, 4)
    assert model.logit_scale.item() == pytest.approx(2.0)
    assert model.logit_bias.item() == pytest.approx(-1.0)

    image = torch.randn(1, 3, 32, 32)
    text = torch.randint(0, model.vocab_size, (1, model.context_length))
    with torch.inference_mode():
        output = model(image=image, text=text)

    assert output["image_features"].shape == (1, 4)
    assert output["text_features"].shape == (1, 4)
    assert torch.isfinite(output["image_features"]).all()
    assert torch.isfinite(output["text_features"]).all()


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

    converted = naflex_convert.convert_naflex_state_dict(model, state_dict)

    assert "visual.trunk.patch_embed.proj.weight" not in converted
    assert converted["visual.trunk.embeds.proj.weight"].shape == (4, 16 * 16 * 3)
    assert "text.token_embedding.weight" in converted


@pytest.mark.skipif(not NAFLEX_AVAILABLE, reason="timm NaFlex data support is not available")
def test_convert_naflex_native_vit_state_dict_folds_class_pos_embed():
    state_dict = {
        "visual.class_embedding": torch.ones(4),
        "visual.positional_embedding": torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4),
        "visual.conv1.weight": torch.randn(4, 3, 16, 16),
        "visual.ln_pre.weight": torch.ones(4),
        "visual.ln_pre.bias": torch.zeros(4),
        "visual.ln_post.weight": torch.ones(4),
        "visual.ln_post.bias": torch.zeros(4),
        "visual.proj": torch.randn(4, 4),
        "visual.transformer.resblocks.0.ln_1.weight": torch.ones(4),
        "visual.transformer.resblocks.0.ln_1.bias": torch.zeros(4),
        "visual.transformer.resblocks.0.attn.in_proj_weight": torch.randn(12, 4),
        "visual.transformer.resblocks.0.attn.in_proj_bias": torch.randn(12),
        "visual.transformer.resblocks.0.attn.out_proj.weight": torch.randn(4, 4),
        "visual.transformer.resblocks.0.attn.out_proj.bias": torch.randn(4),
        "visual.transformer.resblocks.0.ln_2.weight": torch.ones(4),
        "visual.transformer.resblocks.0.ln_2.bias": torch.zeros(4),
        "visual.transformer.resblocks.0.mlp.c_fc.weight": torch.randn(8, 4),
        "visual.transformer.resblocks.0.mlp.c_fc.bias": torch.randn(8),
        "visual.transformer.resblocks.0.mlp.c_proj.weight": torch.randn(4, 8),
        "visual.transformer.resblocks.0.mlp.c_proj.bias": torch.randn(4),
        "text.token_embedding.weight": torch.randn(16, 4),
    }

    converted = naflex_convert._convert_naflex_native_vit_state_dict(state_dict)

    expected_cls = state_dict["visual.class_embedding"] + state_dict["visual.positional_embedding"][0]
    assert "visual.class_embedding" not in converted
    assert "visual.positional_embedding" not in converted
    assert torch.equal(converted["visual.trunk.embeds.cls_token"], expected_cls.reshape(1, 1, 4))
    assert converted["visual.trunk.embeds.pos_embed"].shape == (1, 2, 2, 4)
    assert converted["visual.trunk.embeds.proj.weight"].shape == (4, 16 * 16 * 3)
    assert converted["visual.trunk.blocks.0.attn.qkv.weight"].shape == (12, 4)
    assert converted["visual.trunk.blocks.0.mlp.fc1.weight"].shape == (8, 4)
    assert "text.token_embedding.weight" in converted


@pytest.mark.parametrize("config_fn", [_tiny_native_vit_clip_config, _tiny_native_vit_quickgelu_clip_config])
@pytest.mark.skipif(not NAFLEX_AVAILABLE, reason="timm NaFlex data support is not available")
def test_force_naflex_native_vit_dense_output_matches_native(monkeypatch, config_fn):
    monkeypatch.setitem(factory._MODEL_CONFIGS, "test-native-vit-naflex", config_fn())

    native = factory.create_model("test-native-vit-naflex", load_weights=False)
    converted = factory.create_model(
        "test-native-vit-naflex",
        load_weights=False,
        force_naflex_vision=True,
    )
    converted.load_state_dict(naflex_convert.convert_naflex_state_dict(converted, native.state_dict()), strict=True)
    native.eval()
    converted.eval()

    image = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        native_features = native.encode_image(image)
        converted_features = converted.encode_image(image)

    torch.testing.assert_close(converted_features, native_features, rtol=1e-5, atol=1e-5)
