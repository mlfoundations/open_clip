import re

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode

import open_clip
from open_clip.transform import image_transform


def test_image_transform_scalar_zero_mean_not_replaced_by_default():
    # Scalar mean/std broadcast to all channels; 0 is a valid value, not "unset".
    transform = image_transform(32, is_train=False, mean=0.0, std=1.0)
    image = transform(Image.new("RGB", (32, 32), (255, 0, 0)))
    assert torch.equal(image[0], torch.ones(32, 32))


def test_image_transform_train_uses_requested_interpolation():
    transform = image_transform(32, is_train=True, interpolation="bilinear")
    assert transform.transforms[0].interpolation == InterpolationMode.BILINEAR


def test_builtin_config_image_size_matches_resolution_suffix():
    # Eval preprocessing is sized from vision_cfg.image_size, so it must match a resolution suffix in the
    # config or timm model name (e.g. ViTamin-XL-384 / vitamin_xlarge_384), the size the weights expect.
    for name in open_clip.list_models():
        vision_cfg = open_clip.get_model_config(name).get("vision_cfg") or {}
        size = vision_cfg.get("image_size")
        size = size[0] if isinstance(size, (list, tuple)) else size
        for source in (name.removesuffix("-quickgelu"), vision_cfg.get("timm_model_name", "")):
            match = re.search(r"[-_](\d{3})$", source)
            if match:
                assert size == int(match.group(1)), name
