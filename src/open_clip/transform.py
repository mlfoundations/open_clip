from typing import Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

from PIL import Image

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


class BoundingBoxBlurrer(nn.Module):
    
    def __init__(self, blur_field) -> None:
        self.blur_field = blur_field

    def forward(self, item):
        if self.blur_field is None:
            return item

        img, data = item
        bbox_list = data.get(self.blur_field, [])

        # Skip if there are no boxes to blur.
        if len(bbox_list) == 0:
            return img

        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size

        mask = torch.zeros((height, width), dtype=torch.float32)

        # Incorporate max diagonal from ImageNet code.
        max_diagonal = 0

        for bbox in bbox_list:
            adjusted_bbox = [
                int(bbox[0] * width),
                int(bbox[1] * height),
                int(bbox[2] * width),
                int(bbox[3] * height),
            ]

            diagonal = max(adjusted_bbox[2] - adjusted_bbox[0], adjusted_bbox[3] - adjusted_bbox[1])
            max_diagonal = max(max_diagonal, diagonal)

            # Adjusting bbox as in:
            # https://github.com/princetonvisualai/imagenet-face-obfuscation
            adjusted_bbox[0] = int(adjusted_bbox[0] - 0.1 * diagonal)
            adjusted_bbox[1] = int(adjusted_bbox[1] - 0.1 * diagonal)
            adjusted_bbox[2] = int(adjusted_bbox[2] + 0.1 * diagonal)
            adjusted_bbox[3] = int(adjusted_bbox[3] + 0.1 * diagonal)

            # Clipping for indexing.
            adjusted_bbox[0] = np.clip(adjusted_bbox[0], 0, width - 1)
            adjusted_bbox[1] = np.clip(adjusted_bbox[1], 0, height - 1)
            adjusted_bbox[2] = np.clip(adjusted_bbox[2], 0, width - 1)
            adjusted_bbox[3] = np.clip(adjusted_bbox[3], 0, height - 1)

            mask[adjusted_bbox[1] : adjusted_bbox[3], adjusted_bbox[0] : adjusted_bbox[2], ...] = 1.0

        sigma = 0.1 * max_diagonal
        ksize = int(2 * np.ceil(4 * sigma)) + 1
        blurred_img = F.gaussian_blur(img, kernel_size=ksize, sigma=sigma)
        blurred_mask = F.gaussian_blur(mask, kernel_size=ksize, sigma=sigma)

        if isinstance(img, torch.Tensor):
            result = img.float() * (1 - blurred_mask) + blurred_img.float() * blurred_mask
            if img.dtype == torch.uint8:
                result = result.type(torch.uint8)
        else:
            blurred_mask = F.to_pil_image(blurred_mask)
            result = Image.composite(blurred_img, img, blurred_mask)
        return result


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
        blur_field: Optional[str] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        return Compose([
            BoundingBoxBlurrer(blur_field=blur_field),
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        if resize_longest_max:
            transforms = [
                BoundingBoxBlurrer(blur_field=blur_field),
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                BoundingBoxBlurrer(blur_field=blur_field),
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)
