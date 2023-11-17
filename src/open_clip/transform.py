import numbers
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop, ColorJitter, Grayscale

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .utils import to_2tuple


@dataclass
class PreprocessCfg:
    size: Union[int, Tuple[int, int]] = 224
    mode: str = 'RGB'
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN
    std: Tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: str = 'bicubic'
    resize_mode: str = 'shortest'
    fill_color: int = 0

    def __post_init__(self):
        assert self.mode in ('RGB',)

    @property
    def num_channels(self):
        return 3

    @property
    def input_size(self):
        return (self.num_channels,) + to_2tuple(self.size)

_PREPROCESS_KEYS = set(asdict(PreprocessCfg()).keys())


def merge_preprocess_dict(
        base: Union[PreprocessCfg, Dict],
        overlay: Dict,
):
    """ Merge overlay key-value pairs on top of base preprocess cfg or dict.
    Input dicts are filtered based on PreprocessCfg fields.
    """
    if isinstance(base, PreprocessCfg):
        base_clean = asdict(base)
    else:
        base_clean = {k: v for k, v in base.items() if k in _PREPROCESS_KEYS}
    if overlay:
        overlay_clean = {k: v for k, v in overlay.items() if k in _PREPROCESS_KEYS and v is not None}
        base_clean.update(overlay_clean)
    return base_clean


def merge_preprocess_kwargs(base: PreprocessCfg, **kwargs):
    return merge_preprocess_dict(base, kwargs)


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float], Tuple[float, float, float, float]]] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False

    # params for simclr_jitter_gray
    color_jitter_prob: float = None
    gray_scale_prob: float = None


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class ResizeKeepRatio:
    """ Resize and Keep Ratio

    Copy & paste from `timm`
    """

    def __init__(
            self,
            size,
            longest=0.,
            interpolation=InterpolationMode.BICUBIC,
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = interpolation
        self.longest = float(longest)  # [0, 1] where 0 == shortest edge, 1 == longest
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
            img,
            target_size,
            longest,
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        """Get parameters
        """
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (1. - longest)
        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1., 1.)
        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            aspect_factor = random.uniform(random_aspect_range[0], random_aspect_range[1])
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)
        size = [round(x * f / ratio) for x, f in zip(source_size, ratio_factor)]
        return size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Resized, padded to at least target size, possibly cropped to exactly target size
        """
        size = self.get_params(
            img, self.size, self.longest,
            self.random_scale_prob, self.random_scale_range,
            self.random_aspect_prob, self.random_aspect_range
        )
        img = F.resize(img, size, self.interpolation)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += f', interpolation={self.interpolation})'
        format_string += f', longest={self.longest:.3f})'
        return format_string


def center_crop_or_pad(img: torch.Tensor, output_size: List[int], fill=0) -> torch.Tensor:
    """Center crops and/or pads the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = F.get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = F.pad(img, padding_ltrb, fill=fill)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)


class CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size, fill=0):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return center_crop_or_pad(img, self.size, fill=self.fill)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


def _convert_to_rgb(image):
    return image.convert('RGB')


class color_jitter(object):
    """
    Apply Color Jitter to the PIL image with a specified probability.
    """
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0., p=0.8):
        assert 0. <= p <= 1.
        self.p = p
        self.transf = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


class gray_scale(object):
    """
    Apply Gray Scale to the PIL image with a specified probability.
    """
    def __init__(self, p=0.2):
        assert 0. <= p <= 1.
        self.p = p
        self.transf = Grayscale(num_output_channels=3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


def image_transform(
        image_size: Union[int, Tuple[int, int]],
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_mode: Optional[str] = None,
        interpolation: Optional[str] = None,
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    interpolation = interpolation or 'bicubic'
    assert interpolation in ['bicubic', 'bilinear', 'random']
    # NOTE random is ignored for interpolation_mode, so defaults to BICUBIC for inference if set
    interpolation_mode = InterpolationMode.BILINEAR if interpolation == 'bilinear' else InterpolationMode.BICUBIC

    resize_mode = resize_mode or 'shortest'
    assert resize_mode in ('shortest', 'longest', 'squash')

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()

    normalize = Normalize(mean=mean, std=std)

    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        if use_timm:
            from timm.data import create_transform  # timm can still be optional
            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)

            aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
            # drop extra non-timm items
            aug_cfg_dict.pop('color_jitter_prob', None)
            aug_cfg_dict.pop('gray_scale_prob', None)

            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.,
                mean=mean,
                std=std,
                re_mode='pixel',
                interpolation=interpolation,
                **aug_cfg_dict,
            )
        else:
            train_transform = [
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                _convert_to_rgb,
            ]
            if aug_cfg.color_jitter_prob:
                assert aug_cfg.color_jitter is not None and len(aug_cfg.color_jitter) == 4
                train_transform.extend([
                    color_jitter(*aug_cfg.color_jitter, p=aug_cfg.color_jitter_prob)
                ])
            if aug_cfg.gray_scale_prob:
                train_transform.extend([
                    gray_scale(aug_cfg.gray_scale_prob)
                ])
            train_transform.extend([
                ToTensor(),
                normalize,
            ])
            train_transform = Compose(train_transform)
            if aug_cfg_dict:
                warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
        return train_transform
    else:
        if resize_mode == 'longest':
            transforms = [
                ResizeKeepRatio(image_size, interpolation=interpolation_mode, longest=1),
                CenterCropOrPad(image_size, fill=fill_color)
            ]
        elif resize_mode == 'squash':
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            transforms = [
                Resize(image_size, interpolation=interpolation_mode),
            ]
        else:
            assert resize_mode == 'shortest'
            if not isinstance(image_size, (tuple, list)):
                image_size = (image_size, image_size)
            if image_size[0] == image_size[1]:
                # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
                transforms = [
                    Resize(image_size[0], interpolation=interpolation_mode)
                ]
            else:
                # resize shortest edge to matching target dim for non-square target
                transforms = [ResizeKeepRatio(image_size)]
            transforms += [CenterCrop(image_size)]

        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)


def image_transform_v2(
        cfg: PreprocessCfg,
        is_train: bool,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    return image_transform(
        image_size=cfg.size,
        is_train=is_train,
        mean=cfg.mean,
        std=cfg.std,
        interpolation=cfg.interpolation,
        resize_mode=cfg.resize_mode,
        fill_color=cfg.fill_color,
        aug_cfg=aug_cfg,
    )
