"""
Code from https://github.com/mlfoundations/wise-ft/blob/master/src/datasets/imagenetv2.py
Thanks to the authors of wise-ft
"""
import pathlib
import shutil
import tarfile

import requests
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

URLS = {
    'matched-frequency': 'https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz',
    'threshold-0.7': 'https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-threshold0.7.tar.gz',
    'top-images': 'https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-top-images.tar.gz',
    'val': 'https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenet_validation.tar.gz',
}

FNAMES = {
    'matched-frequency': 'imagenetv2-matched-frequency-format-val',
    'threshold-0.7': 'imagenetv2-threshold0.7-format-val',
    'top-images': 'imagenetv2-top-images-format-val',
    'val': 'imagenet_validation',
}


V2_DATASET_SIZE = 10000
VAL_DATASET_SIZE = 50000


class ImageNetValDataset(Dataset):
    def __init__(self, transform=None, location='.'):
        self.dataset_root = pathlib.Path(f'{location}/imagenet_validation/')
        self.tar_root = pathlib.Path(f'{location}/imagenet_validation.tar.gz')
        self.fnames = list(self.dataset_root.glob('**/*.JPEG'))
        self.transform = transform
        if not self.dataset_root.exists() or len(self.fnames) != VAL_DATASET_SIZE:
            if not self.tar_root.exists():
                print(f'Dataset imagenet-val not found on disk, downloading....')
                response = requests.get(URLS['val'], stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(
                    total=total_size_in_bytes, unit='iB', unit_scale=True
                )
                with open(self.tar_root, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    assert False, f'Downloading from {URLS[variant]} failed'
            print('Extracting....')
            tarfile.open(self.tar_root).extractall(f'{location}')
            shutil.move(f"{location}/{FNAMES['val']}", self.dataset_root)

        self.dataset = ImageFolder(self.dataset_root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img, label = self.dataset[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class ImageNetV2Dataset(Dataset):
    def __init__(self, variant='matched-frequency', transform=None, location='.'):
        self.dataset_root = pathlib.Path(f'{location}/ImageNetV2-{variant}/')
        self.tar_root = pathlib.Path(f'{location}/ImageNetV2-{variant}.tar.gz')
        self.fnames = list(self.dataset_root.glob('**/*.jpeg'))
        self.transform = transform
        assert variant in URLS, f'unknown V2 Variant: {variant}'
        if not self.dataset_root.exists() or len(self.fnames) != V2_DATASET_SIZE:
            if not self.tar_root.exists():
                print(f'Dataset {variant} not found on disk, downloading....')
                response = requests.get(URLS[variant], stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(
                    total=total_size_in_bytes, unit='iB', unit_scale=True
                )
                with open(self.tar_root, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    assert False, f'Downloading from {URLS[variant]} failed'
            print('Extracting....')
            tarfile.open(self.tar_root).extractall(f'{location}')
            shutil.move(f'{location}/{FNAMES[variant]}', self.dataset_root)
            self.fnames = list(self.dataset_root.glob('**/*.jpeg'))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
