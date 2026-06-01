import io
import tarfile
import types

import pytest
import torch
from PIL import Image
from torchvision import transforms

import open_clip
from open_clip.transform import image_transform
from open_clip_train.data import get_imagenet, get_wds_dataset
from open_clip_train.naflex_data import (
    NAFLEX_AVAILABLE,
    NaFlexBatcher,
    collate_naflex_tuples,
    create_naflex_data_config_from_args,
)
from open_clip_train.params import parse_args
from open_clip_train.train import get_naflex_loss_scale


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


def _write_tar(path, num_samples=4):
    with tarfile.open(path, "w") as tar:
        for idx in range(num_samples):
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


def test_naflex_batcher_pads_schedule_to_worker_count():
    batcher = NaFlexBatcher(
        train_num_samples=1,
        patch_size=16,
        seq_lens=(4,),
        max_tokens_per_batch=8,
        transform_factory=_transform_factory,
        batch_divisor=1,
        shuffle=False,
    )

    assert batcher.num_batches == 1
    assert batcher.num_batches_for_workers(4) == 4
    assert batcher.num_samples_for_workers(4) == 4


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
        "--naflex-max-tokens-per-batch",
        "4096",
        "--naflex-batch-divisor",
        "4",
        "--naflex-loss-scale",
        "sqrt",
    ])

    assert args.use_naflex
    assert args.naflex_num_train_image_tokens == 1024
    assert args.naflex_patch_sizes == [16, 32]
    assert args.naflex_patch_size_probs == [0.25, 0.75]
    assert args.naflex_seq_lens == [128, 256]
    assert args.naflex_max_tokens_per_batch == 4096
    assert args.naflex_batch_divisor == 4
    assert args.naflex_loss_scale == "sqrt"


def test_naflex_loss_scale_defaults_to_none():
    args = parse_args([])

    assert args.naflex_loss_scale == "none"


def test_naflex_loss_scale_uses_actual_batch_size():
    batch = {"image": {"patches": torch.zeros(8, 4, 3)}, "text": torch.zeros(8, 1)}
    task = types.SimpleNamespace(batch_size=lambda batch: batch["image"]["patches"].shape[0])

    assert get_naflex_loss_scale(batch, types.SimpleNamespace(naflex_loss_scale="none", batch_size=4), task) == 1.0
    assert get_naflex_loss_scale(batch, types.SimpleNamespace(naflex_loss_scale="linear", batch_size=4), task) == 2.0
    assert get_naflex_loss_scale(batch, types.SimpleNamespace(naflex_loss_scale="sqrt", batch_size=2), task) == 2.0


def test_naflex_loss_scale_ignores_dense_batches():
    batch = {"image": torch.zeros(8, 3, 32, 32), "text": torch.zeros(8, 1)}
    task = types.SimpleNamespace(batch_size=lambda batch: len(batch["image"]))

    assert get_naflex_loss_scale(batch, types.SimpleNamespace(naflex_loss_scale="linear", batch_size=4), task) == 1.0


def test_naflex_eval_config_rejects_non_positive_values():
    args = types.SimpleNamespace(naflex_patch_sizes=[16], naflex_seq_lens=[0, 4])
    with pytest.raises(ValueError, match="sequence lengths"):
        create_naflex_data_config_from_args(args)

    args = types.SimpleNamespace(naflex_patch_sizes=[0], naflex_seq_lens=[4])
    with pytest.raises(ValueError, match="patch size"):
        create_naflex_data_config_from_args(args)


def test_naflex_data_config_uses_model_eval_seq_len_default():
    args = types.SimpleNamespace(naflex_patch_sizes=[16], naflex_seq_lens=None)
    config = create_naflex_data_config_from_args(args, default_eval_seq_len=576)

    assert config.train_seq_lens == (128, 256, 576, 784, 1024)
    assert config.eval_seq_len == 576


def test_naflex_seq_lens_override_model_eval_seq_len_default():
    args = types.SimpleNamespace(naflex_patch_sizes=[16], naflex_seq_lens=[64])
    config = create_naflex_data_config_from_args(args, default_eval_seq_len=576)

    assert config.train_seq_lens == (64,)
    assert config.eval_seq_len == 64


def test_get_wds_dataset_naflex_keeps_dictionary_contract(tmp_path):
    tar_path = tmp_path / "samples.tar"
    _write_tar(tar_path)

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
        naflex_max_tokens_per_batch=8,
        naflex_batch_divisor=1,
        seed=0,
        workers=0,
        batch_size=2,
        distributed=False,
        rank=0,
        world_size=1,
    )
    tokenizer = lambda text: [torch.tensor([int(text.strip())], dtype=torch.long)]

    info = get_wds_dataset(
        args,
        _transform_factory,
        is_train=True,
        tokenizer=tokenizer,
        naflex_data_config=create_naflex_data_config_from_args(args),
    )
    batch = next(iter(info.dataloader))

    assert set(batch.keys()) == {"image", "text"}
    assert batch["image"]["patches"].shape == (2, 4, 16 * 16 * 3)
    assert batch["text"].shape == (2, 1)


def test_get_wds_dataset_naflex_rolls_over_non_resampled_input(tmp_path):
    tar_path = tmp_path / "samples.tar"
    _write_tar(tar_path, num_samples=2)

    args = types.SimpleNamespace(
        train_data=str(tar_path),
        val_data=None,
        dataset_resampled=False,
        train_data_upsampling_factors=None,
        train_num_samples=6,
        val_num_samples=None,
        use_naflex=True,
        naflex_num_train_image_tokens=None,
        naflex_patch_sizes=[16],
        naflex_patch_size_probs=None,
        naflex_seq_lens=[4],
        naflex_max_tokens_per_batch=8,
        naflex_batch_divisor=1,
        seed=0,
        workers=0,
        batch_size=2,
        distributed=False,
        rank=0,
        world_size=1,
    )
    tokenizer = lambda text: [torch.tensor([int(text.strip())], dtype=torch.long)]

    info = get_wds_dataset(
        args,
        _transform_factory,
        is_train=True,
        tokenizer=tokenizer,
        naflex_data_config=create_naflex_data_config_from_args(args),
    )
    batches = list(info.dataloader)

    assert len(batches) == 3
    assert sum(batch["image"]["patches"].shape[0] for batch in batches) == 6


def test_create_model_and_transforms_returns_naflex_eval_factory():
    _, _, preprocess_val = open_clip.create_model_and_transforms(
        "naflex_ViT-B-16",
        pretrained=None,
        device="cpu",
        aug_cfg={"use_timm": True, "naflex": True},
    )

    transform = preprocess_val(max_seq_len=4, patch_size=16)
    image = transform(Image.new("RGB", (32, 32)))

    assert getattr(preprocess_val, "is_naflex_eval_transform_factory")
    assert set(image.keys()) == {"patches", "patch_coord", "patch_valid"}
    assert image["patches"].shape == (4, 16 * 16 * 3)
    assert image["patch_coord"].shape == (4, 2)
    assert image["patch_valid"].all()


def test_get_wds_dataset_naflex_eval_outputs_patched_image_dict(tmp_path):
    tar_path = tmp_path / "samples.tar"
    _write_tar(tar_path)
    _, _, preprocess_val = open_clip.create_model_and_transforms(
        "naflex_ViT-B-16",
        pretrained=None,
        device="cpu",
        aug_cfg={"use_timm": True, "naflex": True},
    )
    args = types.SimpleNamespace(
        train_data=None,
        val_data=str(tar_path),
        dataset_resampled=False,
        train_data_upsampling_factors=None,
        train_num_samples=None,
        val_num_samples=4,
        use_naflex=True,
        naflex_patch_sizes=[16],
        naflex_patch_size_probs=None,
        naflex_seq_lens=[4],
        naflex_max_tokens_per_batch=8,
        naflex_batch_divisor=1,
        seed=0,
        workers=0,
        batch_size=2,
        distributed=False,
        rank=0,
        world_size=1,
    )
    tokenizer = lambda text: [torch.tensor([int(text.strip())], dtype=torch.long)]

    info = get_wds_dataset(
        args,
        preprocess_val,
        is_train=False,
        tokenizer=tokenizer,
        naflex_data_config=create_naflex_data_config_from_args(args),
    )
    batch = next(iter(info.dataloader))

    assert set(batch.keys()) == {"image", "text"}
    assert batch["image"]["patches"].shape == (2, 4, 16 * 16 * 3)
    assert batch["image"]["patch_coord"].shape == (2, 4, 2)
    assert batch["image"]["patch_valid"].all()
    assert batch["text"].shape == (2, 1)


def test_get_imagenet_naflex_eval_outputs_patched_image_dict(tmp_path):
    class_dir = tmp_path / "imagenet" / "class0"
    class_dir.mkdir(parents=True)
    for idx in range(2):
        Image.new("RGB", (32, 32), color=(idx, idx, idx)).save(class_dir / f"{idx}.png")

    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "naflex_ViT-B-16",
        pretrained=None,
        device="cpu",
        aug_cfg={"use_timm": True, "naflex": True},
    )
    args = types.SimpleNamespace(
        imagenet_train=None,
        imagenet_val=str(tmp_path / "imagenet"),
        imagenet_v2=None,
        use_naflex=True,
        naflex_patch_sizes=[16],
        naflex_seq_lens=[4],
        batch_size=2,
        workers=0,
    )

    info = get_imagenet(
        args,
        (preprocess_train, preprocess_val),
        "val",
        naflex_data_config=create_naflex_data_config_from_args(args),
    )
    images, targets = next(iter(info.dataloader))

    assert images["patches"].shape == (2, 4, 16 * 16 * 3)
    assert images["patch_coord"].shape == (2, 4, 2)
    assert images["patch_valid"].all()
    assert targets.tolist() == [0, 0]


def test_get_imagenet_naflex_train_raises_clear_error():
    args = types.SimpleNamespace(use_naflex=True)

    with pytest.raises(ValueError, match="--imagenet-train"):
        get_imagenet(
            args,
            (_transform_factory, _transform_factory),
            "train",
            naflex_data_config=create_naflex_data_config_from_args(args),
        )


def test_naflex_model_forward_accepts_eval_transform_image_dict():
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "naflex_ViT-B-16",
        pretrained=None,
        device="cpu",
        aug_cfg={"use_timm": True, "naflex": True},
    )
    transform = preprocess_val(max_seq_len=4, patch_size=16)
    images, _ = collate_naflex_tuples(
        [(transform(Image.new("RGB", (32, 32))), torch.tensor(0)) for _ in range(2)],
        max_seq_len=4,
    )
    text = torch.zeros(2, 77, dtype=torch.long)

    with torch.inference_mode():
        image_features, text_features, _ = model(images, text)

    assert image_features.shape == (2, 512)
    assert text_features.shape == (2, 512)


def test_naflex_model_forward_accepts_batcher_image_dict():
    samples = [
        {
            "image": Image.new("RGB", (32, 32), color=(idx, idx, idx)),
            "text": torch.zeros(77, dtype=torch.long),
        }
        for idx in range(2)
    ]
    batcher = NaFlexBatcher(
        train_num_samples=2,
        patch_size=16,
        seq_lens=(4,),
        max_tokens_per_batch=8,
        transform_factory=_transform_factory,
        batch_divisor=1,
        shuffle=False,
    )
    batch = next(iter(batcher.run(samples)))
    model = open_clip.create_model("naflex_ViT-B-16", pretrained=None, device="cpu")

    with torch.inference_mode():
        image_features, text_features, _ = model(batch["image"], batch["text"])

    assert image_features.shape == (2, 512)
    assert text_features.shape == (2, 512)


def test_naflex_model_forward_accepts_dense_image_tensor():
    model = open_clip.create_model("naflex_ViT-B-16", pretrained=None, device="cpu")
    image = torch.randn(2, 3, 32, 32)
    text = torch.zeros(2, 77, dtype=torch.long)

    with torch.inference_mode():
        image_features, text_features, _ = model(image, text)

    assert image_features.shape == (2, 512)
    assert text_features.shape == (2, 512)
