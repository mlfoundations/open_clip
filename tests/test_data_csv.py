"""Unit tests for CsvDataset and get_csv_dataset."""
import types

import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

import open_clip
from open_clip.naflex_config import NaFlexDataConfig
from open_clip_train.data import CsvDataset, get_csv_dataset
from open_clip_train.naflex_data import NAFLEX_AVAILABLE


def _make_csv(tmp_path, n=3, image_size=(4, 4)):
    """Create a tiny TSV dataset with n image/caption pairs."""
    rows = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", image_size, color=(i, i, i)).save(p)
        rows.append({"filepath": str(p), "caption": f"cap_{i}"})
    csv_path = tmp_path / "data.tsv"
    pd.DataFrame(rows).to_csv(csv_path, sep="\t", index=False)
    return csv_path


# ---------------------------------------------------------------------------
# CsvDataset internals
# ---------------------------------------------------------------------------


def test_csvdataset_stores_pandas_series(tmp_path):
    """images and captions are pandas Series, not Python lists."""
    csv_path = _make_csv(tmp_path)
    tok = lambda xs: [x.upper() for x in xs]
    ds = CsvDataset(csv_path, transforms.ToTensor(), "filepath", "caption", sep="\t", tokenizer=tok)
    assert isinstance(ds.images, pd.Series)
    assert isinstance(ds.captions, pd.Series)


def test_csvdataset_len(tmp_path):
    csv_path = _make_csv(tmp_path, n=5)
    tok = lambda xs: xs
    ds = CsvDataset(csv_path, transforms.ToTensor(), "filepath", "caption", sep="\t", tokenizer=tok)
    assert len(ds) == 5


def test_csvdataset_getitem_returns_dict(tmp_path):
    csv_path = _make_csv(tmp_path)
    tok = lambda xs: [f"T::{x}" for x in xs]
    ds = CsvDataset(csv_path, transforms.ToTensor(), "filepath", "caption", sep="\t", tokenizer=tok)
    sample = ds[0]
    assert isinstance(sample, dict)
    assert set(sample.keys()) == {"image", "text"}
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["text"] == "T::cap_0"


def test_csvdataset_getitem_stringifies_numeric_caption(tmp_path):
    """Numeric captions are converted to strings before tokenizing."""
    rows = []
    for i in range(2):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (4, 4)).save(p)
        rows.append({"filepath": str(p), "caption": i})  # integer caption
    csv_path = tmp_path / "numeric.tsv"
    pd.DataFrame(rows).to_csv(csv_path, sep="\t", index=False)

    received = []
    tok = lambda xs: [received.append(xs[0]) or xs[0]]
    ds = CsvDataset(csv_path, transforms.ToTensor(), "filepath", "caption", sep="\t", tokenizer=tok)
    ds[0]
    # The tokenizer should receive the stringified caption
    assert received[0] == "0"


# ---------------------------------------------------------------------------
# get_csv_dataset
# ---------------------------------------------------------------------------


def test_get_csv_dataset_train_drop_last(tmp_path):
    """Training dataloader uses drop_last=True."""
    csv_path = _make_csv(tmp_path, n=3)
    args = types.SimpleNamespace(
        train_data=str(csv_path),
        val_data=str(csv_path),
        csv_img_key="filepath",
        csv_caption_key="caption",
        csv_separator="\t",
        distributed=False,
        batch_size=2,
        workers=0,
    )
    tok = lambda xs: xs
    train_info = get_csv_dataset(args, transforms.ToTensor(), is_train=True, tokenizer=tok)
    # 3 samples, batch_size=2, drop_last=True => 1 batch
    assert train_info.dataloader.num_batches == 1


def test_get_csv_dataset_val_keeps_partial(tmp_path):
    """Val dataloader uses drop_last=False."""
    csv_path = _make_csv(tmp_path, n=3)
    args = types.SimpleNamespace(
        train_data=str(csv_path),
        val_data=str(csv_path),
        csv_img_key="filepath",
        csv_caption_key="caption",
        csv_separator="\t",
        distributed=False,
        batch_size=2,
        workers=0,
    )
    tok = lambda xs: xs
    val_info = get_csv_dataset(args, transforms.ToTensor(), is_train=False, tokenizer=tok)
    # 3 samples, batch_size=2, drop_last=False => 2 batches
    assert val_info.dataloader.num_batches == 2


def test_get_csv_dataset_batches_are_dicts(tmp_path):
    """Batches from the dataloader should be dicts with 'image' and 'text'."""
    csv_path = _make_csv(tmp_path, n=4)
    args = types.SimpleNamespace(
        train_data=str(csv_path),
        val_data=str(csv_path),
        csv_img_key="filepath",
        csv_caption_key="caption",
        csv_separator="\t",
        distributed=False,
        batch_size=2,
        workers=0,
    )
    tok = lambda xs: [torch.tensor([ord(c) for c in x[:5]]) for x in xs]
    info = get_csv_dataset(args, transforms.ToTensor(), is_train=False, tokenizer=tok)
    batch = next(iter(info.dataloader))
    assert isinstance(batch, dict)
    assert "image" in batch and "text" in batch
    assert batch["image"].shape[0] == 2


@pytest.mark.skipif(not NAFLEX_AVAILABLE, reason="timm NaFlex data support is not available")
def test_get_csv_dataset_naflex_train_outputs_patched_dict_batches(tmp_path):
    csv_path = _make_csv(tmp_path, n=4, image_size=(32, 32))
    args = types.SimpleNamespace(
        train_data=str(csv_path),
        val_data=str(csv_path),
        csv_img_key="filepath",
        csv_caption_key="caption",
        csv_separator="\t",
        distributed=False,
        rank=0,
        world_size=1,
        seed=0,
        batch_size=2,
        workers=0,
    )
    transform_factory = lambda max_seq_len, patch_size: transforms.ToTensor()
    transform_factory.is_naflex_transform_factory = True
    tokenizer = lambda xs: [torch.tensor([len(x)]) for x in xs]
    config = NaFlexDataConfig.resolve(
        patch_sizes=[16],
        seq_lens=[4],
        max_tokens_per_batch=8,
        batch_divisor=1,
    )

    info = get_csv_dataset(
        args,
        transform_factory,
        is_train=True,
        tokenizer=tokenizer,
        naflex_data_config=config,
    )
    batch = next(iter(info.dataloader))

    assert info.dataloader.num_batches == 2
    assert set(batch.keys()) == {"image", "text"}
    assert batch["image"]["patches"].shape == (2, 4, 16 * 16 * 3)
    assert batch["image"]["patch_coord"].shape == (2, 4, 2)
    assert batch["image"]["patch_valid"].all()
    assert batch["text"].shape == (2, 1)


@pytest.mark.skipif(not NAFLEX_AVAILABLE, reason="timm NaFlex data support is not available")
def test_get_csv_dataset_naflex_val_outputs_patched_dict_batches(tmp_path):
    csv_path = _make_csv(tmp_path, n=2, image_size=(32, 32))
    args = types.SimpleNamespace(
        train_data=str(csv_path),
        val_data=str(csv_path),
        csv_img_key="filepath",
        csv_caption_key="caption",
        csv_separator="\t",
        distributed=False,
        batch_size=2,
        workers=0,
    )
    _, _, preprocess_val = open_clip.create_model_and_transforms(
        "naflex_ViT-B-16",
        pretrained=None,
        device="cpu",
        aug_cfg={"use_timm": True, "naflex": True},
    )
    tokenizer = lambda xs: [torch.tensor([len(x)]) for x in xs]
    config = NaFlexDataConfig.resolve(patch_sizes=[16], seq_lens=[4])

    info = get_csv_dataset(
        args,
        preprocess_val,
        is_train=False,
        tokenizer=tokenizer,
        naflex_data_config=config,
    )
    batch = next(iter(info.dataloader))

    assert set(batch.keys()) == {"image", "text"}
    assert batch["image"]["patches"].shape == (2, 4, 16 * 16 * 3)
    assert batch["image"]["patch_coord"].shape == (2, 4, 2)
    assert batch["image"]["patch_valid"].all()
    assert batch["text"].shape == (2, 1)
