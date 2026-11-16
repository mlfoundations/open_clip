import torch
from PIL import Image

from open_clip.transform import image_transform


def test_image_transform_scalar_zero_mean_not_replaced_by_default():
    # Scalar mean/std broadcast to all channels; 0 is a valid value, not "unset".
    transform = image_transform(32, is_train=False, mean=0.0, std=1.0)
    image = transform(Image.new("RGB", (32, 32), (255, 0, 0)))
    assert torch.equal(image[0], torch.ones(32, 32))
