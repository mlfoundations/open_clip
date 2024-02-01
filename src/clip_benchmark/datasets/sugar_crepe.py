import json
import os

from PIL import Image
from torch.utils.data import Dataset


class SugarCrepe(Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.root = root
        self.ann = json.load(open(ann_file))
        self.transform = transform

    def __getitem__(self, idx):
        data = self.ann[str(idx)]
        img = Image.open(os.path.join(self.root, data['filename']))
        if self.transform is not None:
            img = self.transform(img)
        caption = data['caption']
        negative_caption = data['negative_caption']
        return img, [caption, negative_caption]

    def __len__(self):
        return len(self.ann)
