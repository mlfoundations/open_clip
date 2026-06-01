import torch
import pytest
from torch import nn

import open_clip
from open_clip.audio.convert import convert_hf_clap_state_dict
from open_clip.hf_model import HFTextEncoder


def test_convert_hf_clap_state_dict_maps_audio_qkv_and_projection():
    state_dict = {
        "logit_scale_a": torch.tensor(2.0),
        "audio_model.audio_encoder.layers.0.blocks.0.attention.self.query.weight": torch.ones(2, 3),
        "audio_model.audio_encoder.layers.0.blocks.0.attention.self.key.weight": torch.ones(2, 3) * 2,
        "audio_model.audio_encoder.layers.0.blocks.0.attention.self.value.weight": torch.ones(2, 3) * 3,
        "audio_model.audio_encoder.layers.0.blocks.0.attention.self.query.bias": torch.ones(2),
        "audio_model.audio_encoder.layers.0.blocks.0.attention.self.key.bias": torch.ones(2) * 2,
        "audio_model.audio_encoder.layers.0.blocks.0.attention.self.value.bias": torch.ones(2) * 3,
        "audio_model.audio_encoder.layers.0.blocks.0.layernorm_before.weight": torch.ones(3),
        "audio_model.audio_encoder.layers.0.blocks.0.attention.output.dense.weight": torch.ones(3, 3),
        "audio_model.audio_encoder.layers.0.downsample.reduction.weight": torch.ones(6, 12),
        "audio_model.audio_encoder.batch_norm.weight": torch.ones(64),
        "audio_projection.linear1.weight": torch.ones(4, 6),
        "audio_projection.linear2.bias": torch.ones(4),
    }

    converted = convert_hf_clap_state_dict(state_dict)

    assert torch.equal(converted["logit_scale"], torch.tensor(2.0))
    assert converted["audio.encoder.layers.0.blocks.0.attn.qkv.weight"].shape == (6, 3)
    assert torch.equal(
        converted["audio.encoder.layers.0.blocks.0.attn.qkv.bias"],
        torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
    )
    assert "audio.encoder.layers.0.blocks.0.norm1.weight" in converted
    assert "audio.encoder.layers.0.blocks.0.attn.proj.weight" in converted
    assert "audio.encoder.layers.0.downsample.reduction.weight" in converted
    assert "audio.encoder.bn0.weight" in converted
    assert "audio.proj.0.weight" in converted
    assert "audio.proj.2.bias" in converted


def test_convert_hf_clap_state_dict_maps_text_tower_and_projection():
    state_dict = {
        "text_model.embeddings.word_embeddings.weight": torch.ones(8, 4),
        "text_model.embeddings.position_ids": torch.arange(4),
        "text_model.encoder.layer.0.attention.self.query.weight": torch.ones(4, 4),
        "text_projection.linear1.weight": torch.ones(3, 4),
        "text_projection.linear1.bias": torch.ones(3),
        "text_projection.linear2.weight": torch.ones(3, 3),
    }

    converted = convert_hf_clap_state_dict(state_dict)

    assert "text.transformer.embeddings.word_embeddings.weight" in converted
    assert "text.transformer.embeddings.position_ids" not in converted
    assert "text.transformer.encoder.layer.0.attention.self.query.weight" in converted
    assert "text.proj.0.weight" in converted
    assert "text.proj.0.bias" in converted
    assert "text.proj.2.weight" in converted


def test_hf_text_encoder_clap_mlp_projection_matches_transformers_clap_shape():
    transformers = pytest.importorskip("transformers")
    config = transformers.RobertaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
    )

    encoder = HFTextEncoder(
        "unused",
        output_dim=4,
        config=config,
        pooler_type="cls_pooler",
        proj_type="clap_mlp",
    )

    assert encoder.proj[0].weight.shape == (4, 8)
    assert encoder.proj[0].bias.shape == (4,)
    assert isinstance(encoder.proj[1], torch.nn.ReLU)
    assert encoder.proj[2].weight.shape == (4, 4)
    assert encoder.proj[2].bias.shape == (4,)


def test_random_transformers_clap_state_dict_loads_into_native(monkeypatch):
    transformers = pytest.importorskip("transformers")
    import importlib.machinery
    import sys
    import types

    for name in ("torchlibrosa", "torchlibrosa.stft", "torchlibrosa.augmentation"):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        monkeypatch.setitem(sys.modules, name, module)

    class DummyAudioModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    sys.modules["torchlibrosa.stft"].Spectrogram = DummyAudioModule
    sys.modules["torchlibrosa.stft"].LogmelFilterBank = DummyAudioModule
    sys.modules["torchlibrosa.augmentation"].SpecAugmentation = DummyAudioModule

    # htsat binds these names at import time (`from torchlibrosa.stft import ...`), so the sys.modules swap
    # above only shadows a *fresh* import. If an earlier test in the same worker already imported htsat with
    # the real torchlibrosa, rebind the names on the module itself so the dummies take effect regardless of
    # import order (avoids leaking real Spectrogram/LogmelFilterBank buffers into the model). Auto-reverted.
    import open_clip.audio.htsat as htsat
    monkeypatch.setattr(htsat, "Spectrogram", DummyAudioModule)
    monkeypatch.setattr(htsat, "LogmelFilterBank", DummyAudioModule)
    monkeypatch.setattr(htsat, "SpecAugmentation", DummyAudioModule)

    hf_model = transformers.ClapModel(transformers.ClapConfig())
    converted = convert_hf_clap_state_dict(hf_model.state_dict())
    native = open_clip.create_model(
        "CLAP-HTSAT-tiny-Roberta-base",
        pretrained=None,
        load_weights=False,
        pretrained_text=False,
    )
    incompatible = native.load_state_dict(converted, strict=False)

    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        "audio.encoder.layers.0.blocks.1.attn_mask",
        "audio.encoder.layers.1.blocks.1.attn_mask",
        "audio.encoder.layers.2.blocks.1.attn_mask",
        "audio.encoder.layers.2.blocks.3.attn_mask",
        "audio.encoder.layers.2.blocks.5.attn_mask",
        "audio.encoder.tscam_conv.weight",
        "audio.encoder.tscam_conv.bias",
        "audio.encoder.head.weight",
        "audio.encoder.head.bias",
    }
