"""
Adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/flickr.py
Thanks to the authors of torchvision
"""
import glob
import os
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset


class Flickr(VisionDataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        data = defaultdict(list)
        with open(ann_file) as fd:
            fd.readline()
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    img, caption = line.strip().split('.jpg,')
                    img = img + '.jpg'
                    data[img].append(caption)
        self.data = list(data.items())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img, captions = self.data[index]

        # Image
        img = Image.open(os.path.join(self.root, img)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = captions
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
