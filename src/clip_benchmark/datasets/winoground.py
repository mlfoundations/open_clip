import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class WinoGround(Dataset):
    def __init__(self, root='.', transform=None):
        from datasets import load_dataset

        self.ds = load_dataset('facebook/winoground', cache_dir=root)['test']
        self.transform = transform

    def __getitem__(self, idx):
        data = self.ds[idx]
        img0 = data['image_0']
        img1 = data['image_1']
        cap0 = data['caption_0']
        cap1 = data['caption_1']
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            imgs = torch.stack([img0, img1])
        else:
            imgs = [img0, img1]
        caps = [cap0, cap1]
        return imgs, caps

    def __len__(self):
        return len(self.ds)
