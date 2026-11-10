import pytest
import torch
from PIL import Image
from torchvision.transforms import Normalize

from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip.transform import image_transform


def _normalize(transform):
    return next(t for t in transform.transforms if isinstance(t, Normalize))


@pytest.mark.parametrize("is_train", [False, True])
def test_scalar_zero_mean_is_not_replaced_by_default(is_train):
    # a scalar is broadcast to all channels; 0 is a valid value and must not mean "unset"
    transform = image_transform(32, is_train=is_train, mean=0.0, std=1.0)
    normalize = _normalize(transform)
    assert tuple(normalize.mean) == (0.0, 0.0, 0.0)
    assert tuple(normalize.std) == (1.0, 1.0, 1.0)

    out = transform(Image.new("RGB", (32, 32), (255, 0, 0)))
    if not is_train:
        assert torch.allclose(out[0], torch.ones(32, 32))


def test_unset_mean_std_use_openai_defaults():
    normalize = _normalize(image_transform(32, is_train=False))
    assert tuple(normalize.mean) == pytest.approx(OPENAI_DATASET_MEAN)
    assert tuple(normalize.std) == pytest.approx(OPENAI_DATASET_STD)
