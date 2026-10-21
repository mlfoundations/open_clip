import pytest
import torch
from PIL import Image

from open_clip.transform import ResizeKeepRatio, image_transform


@pytest.mark.parametrize(
    ("input_size", "target_size", "resized_size"),
    [
        ((1000, 1), 16, (16, 1)),
        ((1, 1000), 16, (1, 16)),
        ((10000, 4), (8, 12), (12, 1)),
        ((4, 10000), (8, 12), (1, 8)),
        ((16, 12), (8, 12), (11, 8)),
        ((12, 16), (8, 12), (6, 8)),
        ((16, 16), 16, (16, 16)),
        ((32, 16), 16, (16, 8)),
    ],
)
def test_longest_resize_retains_a_pixel_for_thin_images(input_size, target_size, resized_size):
    """A nonempty image keeps a positive short edge before longest-edge padding."""
    image = Image.new('RGB', input_size, color=(255, 128, 64))
    resized = ResizeKeepRatio(target_size, longest=1)(image)
    assert resized.size == resized_size

    transform = image_transform(
        target_size, is_train=False, resize_mode='longest',
        mean=(0, 0, 0), std=(1, 1, 1),
    )
    actual = transform(image)
    height, width = (target_size, target_size) if isinstance(target_size, int) else target_size
    expected = torch.zeros((3, height, width))
    resized_width, resized_height = resized_size
    top, left = (height - resized_height) // 2, (width - resized_width) // 2
    expected[:, top:top + resized_height, left:left + resized_width] = torch.tensor(
        [1.0, 128 / 255, 64 / 255]).view(3, 1, 1)
    torch.testing.assert_close(actual, expected)
