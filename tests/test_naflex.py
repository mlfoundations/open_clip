import io
import tarfile
import types

import pytest
import torch
from PIL import Image
from torchvision import transforms

from open_clip.transform import image_transform
from open_clip_train.data import get_wds_dataset
from open_clip_train.naflex_data import NAFLEX_AVAILABLE, NaFlexBatcher
from open_clip_train.params import parse_args


pytestmark = pytest.mark.skipif(not NAFLEX_AVAILABLE, reason="timm NaFlex data support is not available")


def _transform_factory(max_seq_len, patch_size):
    return transforms.ToTensor()


_transform_factory.is_naflex_transform_factory = True


def _samples(num_samples=4):
    return [
        {
            "image": Image.new("RGB", (32, 32), color=(idx, idx, idx)),
            "text": torch.tensor([idx], dtype=torch.long),
        }
        for idx in range(num_samples)
    ]


def test_naflex_batcher_returns_dict_batches():
    batcher = NaFlexBatcher(
        train_num_samples=4,
        patch_size=16,
        seq_lens=(4,),
        max_tokens_per_batch=8,
        transform_factory=_transform_factory,
        batch_divisor=1,
        shuffle=False,
    )

    batches = list(batcher.run(_samples()))

    assert len(batches) == 2
    batch = batches[0]
    assert set(batch.keys()) == {"image", "text"}
    assert set(batch["image"].keys()) == {"patches", "patch_coord", "patch_valid", "seq_len"}
    assert batch["image"]["patches"].shape == (2, 4, 16 * 16 * 3)
    assert batch["image"]["patch_coord"].shape == (2, 4, 2)
    assert batch["image"]["patch_valid"].all()
    assert batch["text"].shape == (2, 1)


def test_naflex_batcher_token_schedule_handles_distributed_padding():
    batcher = NaFlexBatcher(
        train_num_tokens=17,
        patch_size=16,
        seq_lens=(4,),
        max_tokens_per_batch=8,
        transform_factory=_transform_factory,
        batch_divisor=1,
        distributed=True,
        rank=0,
        world_size=2,
        shuffle=False,
    )

    assert batcher.num_batches == 2
    assert batcher.num_samples == 6


def test_image_transform_naflex_returns_timm_transform_factory():
    factory = image_transform(
        32,
        is_train=True,
        aug_cfg={"use_timm": True, "naflex": True, "scale": (1.0, 1.0), "ratio": (1.0, 1.0)},
    )

    transform = factory(max_seq_len=4, patch_size=16)
    image = transform(Image.new("RGB", (32, 32)))

    assert getattr(factory, "is_naflex_transform_factory")
    assert isinstance(image, torch.Tensor)
    assert image.shape[-2:] == (32, 32)


def test_image_transform_timm_non_naflex_remains_image_callable():
    transform = image_transform(
        32,
        is_train=True,
        aug_cfg={"use_timm": True, "scale": (1.0, 1.0), "ratio": (1.0, 1.0)},
    )

    image = transform(Image.new("RGB", (32, 32)))

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)


def test_parse_naflex_args():
    args = parse_args([
        "--use-naflex",
        "--naflex-num-train-image-tokens",
        "1024",
        "--naflex-patch-sizes",
        "16",
        "32",
        "--naflex-patch-size-probs",
        "0.25",
        "0.75",
        "--naflex-seq-lens",
        "128",
        "256",
        "--naflex-max-image-tokens-per-batch",
        "4096",
        "--naflex-batch-divisor",
        "4",
    ])

    assert args.use_naflex
    assert args.naflex_num_train_image_tokens == 1024
    assert args.naflex_patch_sizes == [16, 32]
    assert args.naflex_patch_size_probs == [0.25, 0.75]
    assert args.naflex_seq_lens == [128, 256]
    assert args.naflex_max_image_tokens_per_batch == 4096
    assert args.naflex_batch_divisor == 4


def test_get_wds_dataset_naflex_keeps_dictionary_contract(tmp_path):
    tar_path = tmp_path / "samples.tar"
    with tarfile.open(tar_path, "w") as tar:
        for idx in range(4):
            image = Image.new("RGB", (32, 32), color=(idx, idx, idx))
            image_file = io.BytesIO()
            image.save(image_file, format="PNG")
            image_file.seek(0)
            image_info = tarfile.TarInfo(f"{idx}.png")
            image_info.size = len(image_file.getbuffer())
            tar.addfile(image_info, image_file)

            text_file = io.BytesIO(str(idx).encode("utf-8"))
            text_info = tarfile.TarInfo(f"{idx}.txt")
            text_info.size = len(text_file.getbuffer())
            tar.addfile(text_info, text_file)

    args = types.SimpleNamespace(
        train_data=str(tar_path),
        val_data=None,
        dataset_resampled=False,
        train_data_upsampling_factors=None,
        train_num_samples=4,
        val_num_samples=None,
        use_naflex=True,
        naflex_num_train_image_tokens=None,
        naflex_patch_sizes=[16],
        naflex_patch_size_probs=None,
        naflex_seq_lens=[4],
        naflex_max_image_tokens_per_batch=8,
        naflex_batch_divisor=1,
        seed=0,
        workers=0,
        batch_size=2,
        distributed=False,
        rank=0,
        world_size=1,
    )
    tokenizer = lambda text: [torch.tensor([int(text.strip())], dtype=torch.long)]

    info = get_wds_dataset(args, _transform_factory, is_train=True, tokenizer=tokenizer)
    batch = next(iter(info.dataloader))

    assert set(batch.keys()) == {"image", "text"}
    assert batch["image"]["patches"].shape == (2, 4, 16 * 16 * 3)
    assert batch["text"].shape == (2, 1)
