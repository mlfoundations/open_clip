"""
Code adapted from https://github.com/mlfoundations/wise-ft/blob/master/src/datasets/objectnet.py
Thanks to the authors of wise-ft
"""

import json
import os
from pathlib import Path

import numpy as np
import PIL
import torch
from torchvision import datasets
from torchvision.transforms import Compose


def get_metadata(folder):
    metadata = Path(folder)

    with open(metadata / 'folder_to_objectnet_label.json', 'r') as f:
        folder_map = json.load(f)
        folder_map = {v: k for k, v in folder_map.items()}
    with open(metadata / 'objectnet_to_imagenet_1k.json', 'r') as f:
        objectnet_map = json.load(f)

    with open(metadata / 'pytorch_to_imagenet_2012_id.json', 'r') as f:
        pytorch_map = json.load(f)
        pytorch_map = {v: k for k, v in pytorch_map.items()}

    with open(metadata / 'imagenet_to_label_2012_v2', 'r') as f:
        imagenet_map = {v.strip(): str(pytorch_map[i]) for i, v in enumerate(f)}

    folder_to_ids, class_sublist = {}, []
    classnames = []
    for objectnet_name, imagenet_names in objectnet_map.items():
        imagenet_names = imagenet_names.split('; ')
        imagenet_ids = [
            int(imagenet_map[imagenet_name]) for imagenet_name in imagenet_names
        ]
        class_sublist.extend(imagenet_ids)
        folder_to_ids[folder_map[objectnet_name]] = imagenet_ids

    class_sublist = sorted(class_sublist)
    class_sublist_mask = [(i in class_sublist) for i in range(1000)]
    classname_map = {v: k for k, v in folder_map.items()}
    return class_sublist, class_sublist_mask, folder_to_ids, classname_map


class ObjectNetDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        (
            self._class_sublist,
            self.class_sublist_mask,
            self.folders_to_ids,
            self.classname_map,
        ) = get_metadata(root)
        subdir = os.path.join(root, 'objectnet-1.0', 'images')
        label_map = {
            name: idx
            for idx, name in enumerate(sorted(list(self.folders_to_ids.keys())))
        }
        self.label_map = label_map
        super().__init__(subdir, transform=transform)
        self.samples = [
            d
            for d in self.samples
            if os.path.basename(os.path.dirname(d[0])) in self.label_map
        ]
        self.imgs = self.samples
        self.classes = sorted(list(self.folders_to_ids.keys()))
        self.classes = [self.classname_map[c].lower() for c in self.classes]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        label = os.path.basename(os.path.dirname(path))
        return sample, self.label_map[label]
