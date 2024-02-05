import codecs
import json
import os
from subprocess import call

from PIL import Image
from torchvision.datasets import VisionDataset

SUPPORTED_LANGUAGES = [
    'ar',
    'bn',
    'cs',
    'da',
    'de',
    'el',
    'en',
    'es',
    'fa',
    'fi',
    'fil',
    'fr',
    'he',
    'hi',
    'hr',
    'hu',
    'id',
    'it',
    'ja',
    'ko',
    'mi',
    'nl',
    'no',
    'pl',
    'pt',
    'quz',
    'ro',
    'ru',
    'sv',
    'sw',
    'te',
    'th',
    'tr',
    'uk',
    'vi',
    'zh',
]

CAPTIONS_DOWNLOAD_URL = 'https://google.github.io/crossmodal-3600/web-data/captions.zip'
IMAGES_DOWNLOAD_URL = (
    'https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz'
)
OUTPUT_FILENAME_TEMPLATE = 'crossmodal3600_captions-{}.json'


class Crossmodal3600(VisionDataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        with codecs.open(ann_file, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        self.data = [
            (img_path, txt)
            for img_path, txt in zip(data['image_paths'], data['annotations'])
        ]

    def __getitem__(self, index):
        img, captions = self.data[index]

        # Image
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = [
            captions,
        ]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def _download_captions(out_path):
    os.makedirs(out_path, exist_ok=True)
    print('Downloading captions')
    call(f'wget {CAPTIONS_DOWNLOAD_URL} -O captions.zip', shell=True)
    call(f'unzip captions.zip -d {out_path}', shell=True)
    call('rm captions.zip', shell=True)


def _download_images(out_path):
    os.makedirs(out_path, exist_ok=True)
    print('Downloading images')
    call(f'wget {IMAGES_DOWNLOAD_URL} -O images.tgz', shell=True)
    call(f'tar -xzf images.tgz -C {out_path}', shell=True)
    call('rm images.tgz', shell=True)


def create_annotation_file(root, lang_code):
    if lang_code not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f'Language code {lang_code} not supported. Supported languages are {SUPPORTED_LANGUAGES}'
        )
    data_dir = os.path.join(root, 'xm3600')
    images_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(images_dir):
        _download_images(images_dir)
    captions_path = os.path.join(data_dir, 'captions.jsonl')
    if not os.path.exists(captions_path):
        _download_captions(data_dir)
    with open(captions_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]

    number_of_missing_images = 0
    valid_images, valid_annotations, valid_indicies = [], [], []
    for i, data_item in enumerate(data):
        image_id = data_item['image/key']
        image_name = f'{image_id}.jpg'
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print('Missing image file', image_name)
            number_of_missing_images += 1
            continue
        captions = data_item[lang_code]['caption']
        txt = captions[0]

        valid_images.append(image_path)
        valid_annotations.append(txt)
        valid_indicies.append(i)

    if number_of_missing_images > 0:
        print(f'*** WARNING *** missing {number_of_missing_images} files.')

    with codecs.open(
        os.path.join(root, OUTPUT_FILENAME_TEMPLATE.format(lang_code)),
        'w',
        encoding='utf-8',
    ) as fp:
        json.dump(
            {
                'image_paths': valid_images,
                'annotations': valid_annotations,
                'indicies': valid_indicies,
            },
            fp,
            ensure_ascii=False,
        )
