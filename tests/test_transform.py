import pytest
import torch
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Normalize, RandomResizedCrop, ToTensor

from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip.transform import image_transform


@pytest.mark.parametrize("interpolation, expected", [
    ("bicubic", InterpolationMode.BICUBIC),
    ("bilinear", InterpolationMode.BILINEAR),
])
def test_train_transform_uses_requested_interpolation(interpolation, expected):
    transform = image_transform(224, is_train=True, interpolation=interpolation)
    reference = Compose([
        RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=expected),
        ToTensor(),
        Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
    ])
    # non-square random image so the crop is a real resize, not an identity
    img = Image.fromarray((torch.rand(97, 131, 3, generator=torch.Generator().manual_seed(0)) * 255).byte().numpy())

    torch.manual_seed(0)
    out = transform(img)
    torch.manual_seed(0)
    torch.testing.assert_close(out, reference(img), rtol=0, atol=0)
