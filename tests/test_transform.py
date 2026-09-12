import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from open_clip.transform import image_transform


@pytest.mark.parametrize('image_size,resized_size', [
    ((8, 12), (8, 16)),
    ((12, 8), (12, 24)),
    ((8, 8), (8, 16)),
])
@pytest.mark.parametrize('interpolation', ['bilinear', 'bicubic'])
def test_shortest_resize_respects_interpolation(image_size, resized_size, interpolation):
    """Square and rectangular evaluation transforms use the requested interpolation."""
    checkerboard = (np.indices((10, 20)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = Image.fromarray(np.repeat(checkerboard[..., None], 3, axis=2))
    transform = image_transform(
        image_size,
        is_train=False,
        resize_mode='shortest',
        interpolation=interpolation,
        mean=(0., 0., 0.),
        std=(1., 1., 1.),
    )

    resized = F.resize(image, resized_size, interpolation=InterpolationMode(interpolation))
    expected = F.to_tensor(F.center_crop(resized, image_size))

    torch.testing.assert_close(transform(image), expected)
