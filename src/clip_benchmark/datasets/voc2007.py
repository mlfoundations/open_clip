# Code from https://github.com/SsnL/dataset-distillation/blob/master/datasets/pascal_voc.py , thanks to the authors
"""Dataset setting and data loader for PASCAL VOC 2007 as a classification task.

Modified from
https://github.com/Cadene/pretrained-models.pytorch/blob/56aa8c921819d14fb36d7248ab71e191b37cb146/pretrainedmodels/datasets/voc.py
"""

import os
import os.path
import tarfile
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import torch
import torch.utils.data as data
import torchvision
from PIL import Image

object_categories = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

category_to_idx = {c: i for i, c in enumerate(object_categories)}

urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}


def download_url(url, path):
    root, filename = os.path.split(path)
    torchvision.datasets.utils.download_url(url, root=root, filename=filename, md5=None)


def download_voc2007(root):
    path_devkit = os.path.join(root, 'VOCdevkit')
    path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
    tmpdir = os.path.join(root, 'tmp')

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(path_devkit):
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        parts = urlparse(urls['devkit'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            download_url(urls['devkit'], cached_file)

        # extract file
        print(
            '[dataset] Extracting tar file {file} to {path}'.format(
                file=cached_file, path=root
            )
        )
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, 'r')
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # train/val images/annotations
    if not os.path.exists(path_images):
        # download train/val images/annotations
        parts = urlparse(urls['trainval_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            download_url(urls['trainval_2007'], cached_file)

        # extract file
        print(
            '[dataset] Extracting tar file {file} to {path}'.format(
                file=cached_file, path=root
            )
        )
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, 'r')
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test annotations
    test_anno = os.path.join(path_devkit, 'VOC2007/ImageSets/Main/aeroplane_test.txt')
    if not os.path.exists(test_anno):
        # download test annotations
        parts = urlparse(urls['test_images_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            download_url(urls['test_images_2007'], cached_file)

        # extract file
        print(
            '[dataset] Extracting tar file {file} to {path}'.format(
                file=cached_file, path=root
            )
        )
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, 'r')
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test images
    test_image = os.path.join(path_devkit, 'VOC2007/JPEGImages/000001.jpg')
    if not os.path.exists(test_image):
        # download test images
        parts = urlparse(urls['test_anno_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            download_url(urls['test_anno_2007'], cached_file)

        # extract file
        print(
            '[dataset] Extracting tar file {file} to {path}'.format(
                file=cached_file, path=root
            )
        )
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, 'r')
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')


def read_split(root, dataset, split):
    base_path = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    filename = os.path.join(base_path, object_categories[0] + '_' + split + '.txt')

    with open(filename, 'r') as f:
        paths = []
        for line in f.readlines():
            line = line.strip().split()
            if len(line) > 0:
                assert len(line) == 2
                paths.append(line[0])

        return tuple(paths)


def read_bndbox(root, dataset, paths):
    xml_base = os.path.join(root, 'VOCdevkit', dataset, 'Annotations')
    instances = []
    for path in paths:
        xml = ET.parse(os.path.join(xml_base, path + '.xml'))
        for obj in xml.findall('object'):
            c = obj[0]
            assert c.tag == 'name', c.tag
            c = category_to_idx[c.text]
            bndbox = obj.find('bndbox')
            xmin = int(bndbox[0].text)  # left
            ymin = int(bndbox[1].text)  # top
            xmax = int(bndbox[2].text)  # right
            ymax = int(bndbox[3].text)  # bottom
            instances.append((path, (xmin, ymin, xmax, ymax), c))
    return instances


class PASCALVoc2007(data.Dataset):
    """
    Multi-label classification problem for voc2007
    labels are of one hot of shape (C,), denoting the presence/absence
    of each class in each image, where C is the number of classes.
    """

    def __init__(
        self, root, set, transform=None, download=False, target_transform=None
    ):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.transform = transform
        self.target_transform = target_transform

        # download dataset
        if download:
            download_voc2007(self.root)

        paths = read_split(self.root, 'VOC2007', set)
        bndboxes = read_bndbox(self.root, 'VOC2007', paths)
        labels = torch.zeros(len(paths), len(object_categories))
        path_index = {}
        for i, p in enumerate(paths):
            path_index[p] = i
        for path, bbox, c in bndboxes:
            labels[path_index[path], c] = 1
        self.labels = labels
        self.classes = object_categories
        self.paths = paths

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        target = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.paths)


class PASCALVoc2007Cropped(data.Dataset):
    """
    voc2007 is originally object detection and multi-label.
    In this version, we just convert it to single-label per image classification
    problem by looping over bounding boxes in the dataset and cropping the relevant
    object.
    """

    def __init__(
        self, root, set, transform=None, download=False, target_transform=None
    ):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.transform = transform
        self.target_transform = target_transform

        # download dataset
        if download:
            download_voc2007(self.root)

        paths = read_split(self.root, 'VOC2007', set)
        self.bndboxes = read_bndbox(self.root, 'VOC2007', paths)
        self.classes = object_categories

        print(
            '[dataset] VOC 2007 classification set=%s number of classes=%d  number of bndboxes=%d'
            % (set, len(self.classes), len(self.bndboxes))
        )

    def __getitem__(self, index):
        path, crop, target = self.bndboxes[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        img = img.crop(crop)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.bndboxes)
