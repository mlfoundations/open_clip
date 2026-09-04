"""Loader text layout is resolved from traits and overrides, independently of trainer initialization."""
import io
import tarfile
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

from open_clip.model_traits import CLAP_TRAITS, CLIP_TRAITS, InputMode
from open_clip.naflex_config import NaFlexDataConfig
from open_clip_train import audio_data, data
from open_clip_train.legacy_data import get_data_legacy
from open_clip_train.naflex_data import NAFLEX_AVAILABLE
from open_clip_train.params import parse_args
from util_test import VariableTokenizer


class _AudioTransform:
    cfg = {"sample_rate": 8, "clip_samples": 8}

    def __call__(self, audio):
        waveform, _ = audio
        return {"waveform": waveform.squeeze(0), "longer": False}


@pytest.fixture
def loader_inputs(tmp_path, monkeypatch):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(image_path)
    captions = ["Dummy caption", "x"] * 10
    csv_path = tmp_path / "data.tsv"
    pd.DataFrame({"filepath": [str(image_path)] * len(captions), "title": captions}).to_csv(
        csv_path, sep="\t", index=False,
    )
    tar_path = tmp_path / "data.tar"
    with tarfile.open(tar_path, "w") as archive:
        for index, caption in enumerate(captions):
            for ext, payload in (("png", image_path.read_bytes()), ("wav", b"stub"), ("txt", caption.encode())):
                member = tarfile.TarInfo(f"{index:04d}.{ext}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    # Only audio decoding is stubbed; tokenization, pipeline assembly, scheduling, and collation are real.
    monkeypatch.setattr(audio_data, "_decode_audio_bytes", lambda _: (torch.zeros(1, 8), 8))
    return csv_path, tar_path, captions


def _loader_setup(kind, loader_inputs):
    csv_path, tar_path, captions = loader_inputs
    args = parse_args(["--model", "renamed-model", "--batch-size", "2", "--workers", "0"])
    args.rank, args.world_size, args.distributed = 0, 1, False
    args.train_data = args.val_data = str(csv_path if kind == "csv" else tar_path)
    args.train_num_samples = args.val_num_samples = len(captions)
    args.text_attention_mask = False
    args.audio_ext = "wav"
    if "audio" in kind:
        transform, traits = _AudioTransform(), CLAP_TRAITS
    else:
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        traits = CLIP_TRAITS
    return args, transform, traits


@pytest.mark.parametrize("kind", ["csv", "webdataset", "webdataset-audio", "synthetic", "synthetic-audio"])
@pytest.mark.parametrize("layout", ["fixed", "trait", "override", "budget"])
def test_loaders_resolve_text_layout_without_trainer(kind, layout, loader_inputs):
    args, transform, traits = _loader_setup(kind, loader_inputs)
    args.variable_text = layout == "override"
    traits = replace(traits, variable_text=layout == "trait", naflex_text_in_token_budget=layout == "budget")
    tokenizer = VariableTokenizer()
    builder = data.get_dataset_fn(args.val_data, kind)
    info = builder(args, transform, is_train=False, tokenizer=tokenizer, model_traits=traits)
    batch = next(iter(info.dataloader))

    captions = ["Dummy caption"] * 2 if kind.startswith("synthetic") else loader_inputs[2][:2]
    width = tokenizer.context_length if layout == "fixed" else max(len(c) + 1 for c in captions)
    expected = tokenizer(captions)[:, :width]
    assert torch.equal(batch["text"], expected)
    if layout == "fixed":
        assert "text_valid" not in batch
    else:
        assert torch.equal(batch["text_valid"], expected != tokenizer.pad_token_id)
    # Builders must not depend on, or perform, apply_model_traits' mutation of args.
    assert args.variable_text is (layout == "override")


@pytest.mark.parametrize("kind", ["synthetic", "synthetic-audio"])
@pytest.mark.parametrize("variable_text", [False, True])
def test_legacy_synthetic_loaders_remain_args_only(kind, variable_text, loader_inputs):
    args, transform, _ = _loader_setup(kind, loader_inputs)
    args.train_data = args.val_data = None
    args.dataset_type = kind
    args.variable_text = variable_text
    tokenizer = VariableTokenizer()
    loaders = get_data_legacy(args, (transform, transform), tokenizer=tokenizer)
    batch = next(iter(loaders["train"].dataloader))
    width = len("Dummy caption") + 1 if variable_text else tokenizer.context_length
    assert batch["text"].shape == (args.batch_size, width)
    assert ("text_valid" in batch) is variable_text


@pytest.mark.parametrize("kind", ["synthetic", "synthetic-audio"])
def test_synthetic_loaders_require_traits(kind, loader_inputs):
    args, transform, _ = _loader_setup(kind, loader_inputs)
    with pytest.raises(ValueError, match="require model_traits"):
        data.get_dataset_fn(None, kind)(args, transform, is_train=False, tokenizer=VariableTokenizer())


def test_variable_padding_does_not_imply_caption_budget():
    args = SimpleNamespace(variable_text=False)
    assert data.resolve_text_layout(args, replace(CLIP_TRAITS, variable_text=True)) == (False, True)
    assert data.resolve_text_layout(args, replace(CLIP_TRAITS, naflex_text_in_token_budget=True)) == (True, True)


class _NaFlexTransformFactory:
    is_naflex_transform_factory = True

    def __init__(self, audio=False):
        self.audio = audio

    def __call__(self, max_seq_len, patch_size):
        if not self.audio:
            return transforms.ToTensor()
        return lambda _: {
            "patches": torch.zeros(max_seq_len, 16),
            "patch_coord": torch.zeros(max_seq_len, 2, dtype=torch.long),
            "patch_valid": torch.ones(max_seq_len, dtype=torch.bool),
        }


@pytest.mark.skipif(not NAFLEX_AVAILABLE, reason="timm NaFlex data support is not available")
@pytest.mark.parametrize("kind", ["csv", "webdataset", "webdataset-audio"])
@pytest.mark.parametrize("text_in_budget", [False, True])
def test_only_budget_trait_charges_captions_to_naflex_rows(kind, text_in_budget, loader_inputs):
    args, _, traits = _loader_setup(kind, loader_inputs)
    is_audio = kind == "webdataset-audio"
    traits = replace(
        traits, variable_text=True, naflex_text_in_token_budget=text_in_budget,
        **({"audio_input": InputMode.NAFLEX} if is_audio else {"image_input": InputMode.NAFLEX}),
    )
    config = NaFlexDataConfig.resolve(
        patch_sizes=[16], seq_lens=[4], max_tokens_per_batch=40, batch_divisor=1,
    )
    tokenizer = VariableTokenizer()
    builder = data.get_dataset_fn(args.train_data, kind)
    info = builder(
        args, _NaFlexTransformFactory(audio=is_audio), is_train=True, tokenizer=tokenizer,
        model_traits=traits, naflex_data_config=config,
    )
    batch = next(iter(info.dataloader))
    # 40 / 4 modality tokens = 10 rows, or 40 / (4 + 16 caption tokens) = 2 rows.
    rows = 2 if text_in_budget else 10
    assert batch["text"].shape[0] == rows
    assert batch["audio" if is_audio else "image"]["patches"].shape[:2] == (rows, 4)
    assert torch.equal(batch["text_valid"], batch["text"] != tokenizer.pad_token_id)
