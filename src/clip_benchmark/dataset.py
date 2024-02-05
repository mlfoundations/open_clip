import json
import os
import sys
import warnings
from subprocess import call
from typing import Any, Optional

import torch
from torch.utils.data import default_collate
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    DTD,
    GTSRB,
    MNIST,
    PCAM,
    STL10,
    SUN397,
    CocoCaptions,
    Country211,
    EuroSAT,
    FGVCAircraft,
    Flowers102,
    Food101,
    ImageFolder,
    ImageNet,
    OxfordIIITPet,
    RenderedSST2,
    StanfordCars,
)

from clip_benchmark.datasets import (
    babel_imagenet,
    caltech101,
    flickr,
    imagenetv2,
    objectnet,
    sugar_crepe,
    voc2007,
    winoground,
)


def value_from_first_key_found(dic, keys):
    for k in keys:
        if k in dic:
            return dic[k]


class Dummy:
    def __init__(self):
        self.classes = ['blank image', 'noisy image']

    def __getitem__(self, i):
        return torch.zeros(3, 224, 224), 0

    def __len__(self):
        return 1


def get_dataset_default_task(dataset):
    if dataset in (
        'flickr30k',
        'flickr8k',
        'mscoco_captions',
        'multilingual_mscoco_captions',
        'flickr30k-200',
        'crossmodal3600',
        'xtd200',
    ):
        return 'zeroshot_retrieval'
    elif dataset.startswith('sugar_crepe') or dataset == 'winoground':
        return 'image_caption_selection'
    else:
        return 'zeroshot_classification'


def get_dataset_collate_fn(dataset_name):
    if dataset_name in (
        'mscoco_captions',
        'multilingual_mscoco_captions',
        'flickr30k',
        'flickr8k',
        'flickr30k-200',
        'crossmodal3600',
        'xtd200',
        'winoground',
    ) or dataset_name.startswith('sugar_crepe'):
        return image_captions_collate_fn
    else:
        return default_collate


def has_gdown():
    return call('which gdown', shell=True) == 0


def has_kaggle():
    return call('which kaggle', shell=True) == 0


def _extract_task(dataset_name):
    prefix, *task_name_list = dataset_name.split('_')
    task = '_'.join(task_name_list)
    return task


def image_captions_collate_fn(batch):
    transposed = list(zip(*batch))
    imgs = default_collate(transposed[0])
    texts = transposed[1]
    return imgs, texts


def get_dataset_collection_from_file(path):
    return [line.strip() for line in open(path).readlines()]


dataset_collection = {
    'vtab': [
        'vtab/caltech101',
        'vtab/cifar100',
        'vtab/clevr_count_all',
        'vtab/clevr_closest_object_distance',
        'vtab/diabetic_retinopathy',
        'vtab/dmlab',
        'vtab/dsprites_label_orientation',
        'vtab/dsprites_label_x_position',
        'vtab/dtd',
        'vtab/eurosat',
        'vtab/kitti_closest_vehicle_distance',
        'vtab/flowers',
        'vtab/pets',
        'vtab/pcam',
        'vtab/resisc45',
        'vtab/smallnorb_label_azimuth',
        'vtab/smallnorb_label_elevation',
        'sun397',
        'vtab/svhn',
    ],
    'vtab+': [
        'imagenet1k',
        'imagenetv2',
        'imagenet_sketch',
        'imagenet-a',
        'imagenet-r',
        'objectnet',
        'fer2013',
        'voc2007',
        'voc2007_multilabel',
        'sun397',
        'cars',
        'fgvc_aircraft',
        'mnist',
        'stl10',
        'gtsrb',
        'country211',
        'renderedsst2',
        'vtab/caltech101',
        'vtab/cifar10',
        'vtab/cifar100',
        'vtab/clevr_count_all',
        'vtab/clevr_closest_object_distance',
        'vtab/diabetic_retinopathy',
        'vtab/dmlab',
        'vtab/dsprites_label_orientation',
        'vtab/dsprites_label_x_position',
        'vtab/dtd',
        'vtab/eurosat',
        'vtab/kitti_closest_vehicle_distance',
        'vtab/flowers',
        'vtab/pets',
        'vtab/pcam',
        'vtab/resisc45',
        'vtab/smallnorb_label_azimuth',
        'vtab/smallnorb_label_elevation',
        'vtab/svhn',
    ],
    'retrieval': [
        'mscoco_captions',
        'flickr8k',
        'flickr30k',
    ],
    'imagenet_robustness': [
        'imagenetv2',
        'imagenet_sketch',
        'imagenet-a',
        'imagenet-r',
        'objectnet',
    ],
    'sugar_crepe': [
        'sugar_crepe/add_att',
        'sugar_crepe/add_obj',
        'sugar_crepe/replace_att',
        'sugar_crepe/replace_obj',
        'sugar_crepe/replace_rel',
        'sugar_crepe/swap_att',
        'sugar_crepe/swap_obj',
    ],
}


def build_dataset(
    dataset_name: str,
    root: str = 'root',
    transform: Any = None,
    split: str = 'test',
    download: bool = True,
    annotation_file: Optional[str] = None,
    language: str = 'en',
    task: str = 'zeroshot_classification',
    wds_cache_dir: Optional[str] = None,
    custom_classname_file: Optional[str] = None,
    custom_template_file: Optional[str] = None,
    **kwargs,
):
    """
    Main function to use in order to build a dataset instance,

    dataset_name: str
        name of the dataset
    root: str
        root folder where the dataset is downloaded and stored. can be shared among
        datasets.
    transform: torchvision transform applied to images
    split: str
        split to use, depending on the dataset can have different options.
        In general, `train` and `test` are available.
        For specific splits, please look at the corresponding dataset.
    annotation_file: str or None
        only for datasets with captions (used for retrieval) such as COCO
        and Flickr.
    custom_classname_file: str or None
        Custom classname file where keys are dataset names and values are list of
        classnames.
    custom_template_file: str or None
        Custom template file where keys are dataset names and values are list of
        prompts, or dicts where keys are classnames and values are class-specific
        prompts.
    """

    use_classnames_and_templates = task in ('zeroshot_classification', 'linear_probe')

    if use_classnames_and_templates:  # Only load templates and classnames if we have to
        current_folder = os.path.dirname(__file__)

        # Load <LANG>_classnames.json (packaged with CLIP benchmark that are used by
        # default)
        default_classname_file = os.path.join(
            current_folder, language + '_classnames.json'
        )
        if os.path.exists(default_classname_file):
            with open(default_classname_file, 'r') as f:
                default_classnames = json.load(f)
        else:
            default_classnames = None

        # Load <LANG>_zeroshot_classification_templates.json  (packaged with CLIP
        # benchmark that are used by default)
        default_template_file = os.path.join(
            current_folder, language + '_zeroshot_classification_templates.json'
        )
        if os.path.exists(default_template_file):
            with open(default_template_file, 'r') as f:
                default_templates = json.load(f)
        else:
            default_templates = None

        # Load custom classnames file if --custom_classname_file is specified
        if custom_classname_file:
            if not os.path.exists(custom_classname_file):
                custom_classname_file = os.path.join(
                    current_folder, custom_classname_file
                )
            assert os.path.exists(
                custom_classname_file
            ), f"Custom classname file '{custom_classname_file}' does not exist"
            with open(custom_classname_file, 'r') as f:
                custom_classnames = json.load(f)
        else:
            custom_classnames = None

        # Load custom template file if --custom_template_file is specified
        if custom_template_file:
            if not os.path.exists(custom_template_file):
                # look at current_folder
                custom_template_file = os.path.join(
                    current_folder, custom_template_file
                )
            assert os.path.exists(
                custom_template_file
            ), f"Custom template file '{custom_template_file}' does not exist"
            with open(custom_template_file, 'r') as f:
                custom_templates = json.load(f)
        else:
            custom_templates = None

    def download_imagenet(r):
        os.makedirs(r, exist_ok=True)
        call(
            (
                f'wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_'
                f'devkit_t12.tar.gz --output-document={r}/ILSVRC2012_devkit_t12.tar.gz'
            ),
            shell=True,
        )
        call(
            (
                f'wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_'
                f'img_val.tar --output-document={r}/ILSVRC2012_img_val.tar'
            ),
            shell=True,
        )

    train = split == 'train'
    if dataset_name == 'cifar10':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        ds = CIFAR10(
            root=root, train=train, transform=transform, download=download, **kwargs
        )

    elif dataset_name == 'cifar100':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        ds = CIFAR100(
            root=root, train=train, transform=transform, download=download, **kwargs
        )

    elif dataset_name == 'imagenet1k':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        if not os.path.exists(root):
            download_imagenet(root)
        ds = ImageNet(
            root=root, split='train' if train else 'val', transform=transform, **kwargs
        )
        ds.classes = default_classnames['imagenet1k']

    elif dataset_name == 'imagenet-w':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'

        from imagenet_w import AddWatermark
        from torchvision.transforms import CenterCrop, Normalize

        if not os.path.exists(root):
            download_imagenet(root)
        index_normalize = None
        crop_size = None
        for i, t in enumerate(transform.transforms):
            if isinstance(t, Normalize):
                index_normalize = i
            elif isinstance(t, CenterCrop):
                crop_size = min(t.size)
        assert crop_size is not None, 'CenterCrop not found in transform'
        assert index_normalize is not None, 'Normalize not found in transform'
        transform.transforms.insert(index_normalize, AddWatermark(crop_size))
        ds = ImageNet(
            root=root, split='train' if train else 'val', transform=transform, **kwargs
        )
        ds.classes = custom_classnames['imagenet1k']

    elif dataset_name == 'babel_imagenet':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        # babel ImageNet from https://github.com/gregor-ge/Babel-ImageNet
        if not os.path.exists(root):
            download_imagenet(root)
        classnames = json.load(
            open(os.path.join(current_folder, 'babel_imagenet.json'))
        )
        assert (
            language.upper() in classnames
        ), f"Language '{language}' not supported for Babel-ImageNet"
        classnames = classnames[language.upper()]
        templates = json.load(
            open(os.path.join(current_folder, 'nllb_dist13b_prompts.json'))
        )
        templates = templates[language.upper()]
        templates = [t.replace('{}', '{c}') for t in templates]
        idxs, classnames = classnames
        ds = babel_imagenet.BabelImageNet(
            root=root,
            idxs=idxs,
            split='train' if train else 'val',
            transform=transform,
            **kwargs,
        )
        ds.classes = classnames
        ds.templates = templates

    elif dataset_name == 'imagenet1k-unverified':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        split = 'train' if train else 'val'
        ds = ImageFolder(root=os.path.join(root, split), transform=transform, **kwargs)
        # use classnames from OpenAI
        ds.classes = default_classnames['imagenet1k']

    elif dataset_name == 'imagenetv2':
        assert split == 'test', f'Only `test` split available for {dataset_name}'
        os.makedirs(root, exist_ok=True)
        ds = imagenetv2.ImageNetV2Dataset(
            variant='matched-frequency', transform=transform, location=root
        )
        ds.classes = default_classnames['imagenet1k']

    elif dataset_name == 'imagenet_sketch':
        assert split == 'test', f'Only `test` split available for {dataset_name}'
        # Downloadable from
        # https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA
        if not os.path.exists(root):
            # Automatic download
            print('Downloading imagenet_sketch...')
            if not has_gdown():
                print(
                    'GDown is needed to download the dataset. Please install it '
                    'via `pip install gdown`'
                )
                sys.exit(1)
            # Download ImageNet-Sketch.zip
            call('gdown --id 1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA', shell=True)
            assert os.path.exists('ImageNet-Sketch.zip')
            # Unzip and move to `root`
            call('unzip ImageNet-Sketch.zip', shell=True)
            call(f'mv sketch {root}', shell=True)
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = default_classnames['imagenet1k']

    elif dataset_name == 'imagenet-a':
        assert split == 'test', f'Only `test` split available for {dataset_name}'
        # Downloadable from https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
        if not os.path.exists(root):
            print('Downloading imagenet-a...')
            call(
                'wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar',
                shell=True,
            )
            # Untar and move to `root`
            call('tar xvf imagenet-a.tar', shell=True)
            call(f'mv imagenet-a {root}', shell=True)
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = default_classnames['imagenet1k']

        imagenet_a_wnids = [
            'n01498041',
            'n01531178',
            'n01534433',
            'n01558993',
            'n01580077',
            'n01614925',
            'n01616318',
            'n01631663',
            'n01641577',
            'n01669191',
            'n01677366',
            'n01687978',
            'n01694178',
            'n01698640',
            'n01735189',
            'n01770081',
            'n01770393',
            'n01774750',
            'n01784675',
            'n01819313',
            'n01820546',
            'n01833805',
            'n01843383',
            'n01847000',
            'n01855672',
            'n01882714',
            'n01910747',
            'n01914609',
            'n01924916',
            'n01944390',
            'n01985128',
            'n01986214',
            'n02007558',
            'n02009912',
            'n02037110',
            'n02051845',
            'n02077923',
            'n02085620',
            'n02099601',
            'n02106550',
            'n02106662',
            'n02110958',
            'n02119022',
            'n02123394',
            'n02127052',
            'n02129165',
            'n02133161',
            'n02137549',
            'n02165456',
            'n02174001',
            'n02177972',
            'n02190166',
            'n02206856',
            'n02219486',
            'n02226429',
            'n02231487',
            'n02233338',
            'n02236044',
            'n02259212',
            'n02268443',
            'n02279972',
            'n02280649',
            'n02281787',
            'n02317335',
            'n02325366',
            'n02346627',
            'n02356798',
            'n02361337',
            'n02410509',
            'n02445715',
            'n02454379',
            'n02486410',
            'n02492035',
            'n02504458',
            'n02655020',
            'n02669723',
            'n02672831',
            'n02676566',
            'n02690373',
            'n02701002',
            'n02730930',
            'n02777292',
            'n02782093',
            'n02787622',
            'n02793495',
            'n02797295',
            'n02802426',
            'n02814860',
            'n02815834',
            'n02837789',
            'n02879718',
            'n02883205',
            'n02895154',
            'n02906734',
            'n02948072',
            'n02951358',
            'n02980441',
            'n02992211',
            'n02999410',
            'n03014705',
            'n03026506',
            'n03124043',
            'n03125729',
            'n03187595',
            'n03196217',
            'n03223299',
            'n03250847',
            'n03255030',
            'n03291819',
            'n03325584',
            'n03355925',
            'n03384352',
            'n03388043',
            'n03417042',
            'n03443371',
            'n03444034',
            'n03445924',
            'n03452741',
            'n03483316',
            'n03584829',
            'n03590841',
            'n03594945',
            'n03617480',
            'n03666591',
            'n03670208',
            'n03717622',
            'n03720891',
            'n03721384',
            'n03724870',
            'n03775071',
            'n03788195',
            'n03804744',
            'n03837869',
            'n03840681',
            'n03854065',
            'n03888257',
            'n03891332',
            'n03935335',
            'n03982430',
            'n04019541',
            'n04033901',
            'n04039381',
            'n04067472',
            'n04086273',
            'n04099969',
            'n04118538',
            'n04131690',
            'n04133789',
            'n04141076',
            'n04146614',
            'n04147183',
            'n04179913',
            'n04208210',
            'n04235860',
            'n04252077',
            'n04252225',
            'n04254120',
            'n04270147',
            'n04275548',
            'n04310018',
            'n04317175',
            'n04344873',
            'n04347754',
            'n04355338',
            'n04366367',
            'n04376876',
            'n04389033',
            'n04399382',
            'n04442312',
            'n04456115',
            'n04482393',
            'n04507155',
            'n04509417',
            'n04532670',
            'n04540053',
            'n04554684',
            'n04562935',
            'n04591713',
            'n04606251',
            'n07583066',
            'n07695742',
            'n07697313',
            'n07697537',
            'n07714990',
            'n07718472',
            'n07720875',
            'n07734744',
            'n07749582',
            'n07753592',
            'n07760859',
            'n07768694',
            'n07831146',
            'n09229709',
            'n09246464',
            'n09472597',
            'n09835506',
            'n11879895',
            'n12057211',
            'n12144580',
            'n12267677',
        ]
        imagenet_a_mask = [
            wnid in set(imagenet_a_wnids) for wnid in all_imagenet_wordnet_ids
        ]
        ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_a_mask) if mask]

    elif dataset_name == 'imagenet-r':
        assert split == 'test', f'Only `test` split available for {dataset_name}'
        # downloadable from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
        if not os.path.exists(root):
            print('Downloading imagenet-r...')
            call(
                'wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar',
                shell=True,
            )
            # Untar and move to `root`
            call('tar xvf imagenet-r.tar', shell=True)
            call(f'mv imagenet-r {root}', shell=True)
        imagenet_r_wnids = {
            'n01443537',
            'n01484850',
            'n01494475',
            'n01498041',
            'n01514859',
            'n01518878',
            'n01531178',
            'n01534433',
            'n01614925',
            'n01616318',
            'n01630670',
            'n01632777',
            'n01644373',
            'n01677366',
            'n01694178',
            'n01748264',
            'n01770393',
            'n01774750',
            'n01784675',
            'n01806143',
            'n01820546',
            'n01833805',
            'n01843383',
            'n01847000',
            'n01855672',
            'n01860187',
            'n01882714',
            'n01910747',
            'n01944390',
            'n01983481',
            'n01986214',
            'n02007558',
            'n02009912',
            'n02051845',
            'n02056570',
            'n02066245',
            'n02071294',
            'n02077923',
            'n02085620',
            'n02086240',
            'n02088094',
            'n02088238',
            'n02088364',
            'n02088466',
            'n02091032',
            'n02091134',
            'n02092339',
            'n02094433',
            'n02096585',
            'n02097298',
            'n02098286',
            'n02099601',
            'n02099712',
            'n02102318',
            'n02106030',
            'n02106166',
            'n02106550',
            'n02106662',
            'n02108089',
            'n02108915',
            'n02109525',
            'n02110185',
            'n02110341',
            'n02110958',
            'n02112018',
            'n02112137',
            'n02113023',
            'n02113624',
            'n02113799',
            'n02114367',
            'n02117135',
            'n02119022',
            'n02123045',
            'n02128385',
            'n02128757',
            'n02129165',
            'n02129604',
            'n02130308',
            'n02134084',
            'n02138441',
            'n02165456',
            'n02190166',
            'n02206856',
            'n02219486',
            'n02226429',
            'n02233338',
            'n02236044',
            'n02268443',
            'n02279972',
            'n02317335',
            'n02325366',
            'n02346627',
            'n02356798',
            'n02363005',
            'n02364673',
            'n02391049',
            'n02395406',
            'n02398521',
            'n02410509',
            'n02423022',
            'n02437616',
            'n02445715',
            'n02447366',
            'n02480495',
            'n02480855',
            'n02481823',
            'n02483362',
            'n02486410',
            'n02510455',
            'n02526121',
            'n02607072',
            'n02655020',
            'n02672831',
            'n02701002',
            'n02749479',
            'n02769748',
            'n02793495',
            'n02797295',
            'n02802426',
            'n02808440',
            'n02814860',
            'n02823750',
            'n02841315',
            'n02843684',
            'n02883205',
            'n02906734',
            'n02909870',
            'n02939185',
            'n02948072',
            'n02950826',
            'n02951358',
            'n02966193',
            'n02980441',
            'n02992529',
            'n03124170',
            'n03272010',
            'n03345487',
            'n03372029',
            'n03424325',
            'n03452741',
            'n03467068',
            'n03481172',
            'n03494278',
            'n03495258',
            'n03498962',
            'n03594945',
            'n03602883',
            'n03630383',
            'n03649909',
            'n03676483',
            'n03710193',
            'n03773504',
            'n03775071',
            'n03888257',
            'n03930630',
            'n03947888',
            'n04086273',
            'n04118538',
            'n04133789',
            'n04141076',
            'n04146614',
            'n04147183',
            'n04192698',
            'n04254680',
            'n04266014',
            'n04275548',
            'n04310018',
            'n04325704',
            'n04347754',
            'n04389033',
            'n04409515',
            'n04465501',
            'n04487394',
            'n04522168',
            'n04536866',
            'n04552348',
            'n04591713',
            'n07614500',
            'n07693725',
            'n07695742',
            'n07697313',
            'n07697537',
            'n07714571',
            'n07714990',
            'n07718472',
            'n07720875',
            'n07734744',
            'n07742313',
            'n07745940',
            'n07749582',
            'n07753275',
            'n07753592',
            'n07768694',
            'n07873807',
            'n07880968',
            'n07920052',
            'n09472597',
            'n09835506',
            'n10565667',
            'n12267677',
        }
        imagenet_r_mask = [
            wnid in imagenet_r_wnids for wnid in all_imagenet_wordnet_ids
        ]
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = default_classnames['imagenet1k']
        ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_r_mask) if mask]

    elif dataset_name == 'imagenet-o':
        assert split == 'test', f'Only `test` split available for {dataset_name}'
        # downloadable from https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar
        if not os.path.exists(root):
            print('Downloading imagenet-o...')
            call(
                'wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar',
                shell=True,
            )
            # Untar and move to `root`
            call('tar xvf imagenet-o.tar', shell=True)
            call(f'mv imagenet-o {root}', shell=True)
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = default_classnames['imagenet1k']
        imagenet_o_wnids = [
            'n01443537',
            'n01704323',
            'n01770081',
            'n01784675',
            'n01819313',
            'n01820546',
            'n01910747',
            'n01917289',
            'n01968897',
            'n02074367',
            'n02317335',
            'n02319095',
            'n02395406',
            'n02454379',
            'n02606052',
            'n02655020',
            'n02666196',
            'n02672831',
            'n02730930',
            'n02777292',
            'n02783161',
            'n02786058',
            'n02787622',
            'n02791270',
            'n02808304',
            'n02817516',
            'n02841315',
            'n02865351',
            'n02877765',
            'n02892767',
            'n02906734',
            'n02910353',
            'n02916936',
            'n02948072',
            'n02965783',
            'n03000134',
            'n03000684',
            'n03017168',
            'n03026506',
            'n03032252',
            'n03075370',
            'n03109150',
            'n03126707',
            'n03134739',
            'n03160309',
            'n03196217',
            'n03207743',
            'n03218198',
            'n03223299',
            'n03240683',
            'n03271574',
            'n03291819',
            'n03297495',
            'n03314780',
            'n03325584',
            'n03344393',
            'n03347037',
            'n03372029',
            'n03376595',
            'n03388043',
            'n03388183',
            'n03400231',
            'n03445777',
            'n03457902',
            'n03467068',
            'n03482405',
            'n03483316',
            'n03494278',
            'n03530642',
            'n03544143',
            'n03584829',
            'n03590841',
            'n03598930',
            'n03602883',
            'n03649909',
            'n03661043',
            'n03666591',
            'n03676483',
            'n03692522',
            'n03706229',
            'n03717622',
            'n03720891',
            'n03721384',
            'n03724870',
            'n03729826',
            'n03733131',
            'n03733281',
            'n03742115',
            'n03786901',
            'n03788365',
            'n03794056',
            'n03804744',
            'n03814639',
            'n03814906',
            'n03825788',
            'n03840681',
            'n03843555',
            'n03854065',
            'n03857828',
            'n03868863',
            'n03874293',
            'n03884397',
            'n03891251',
            'n03908714',
            'n03920288',
            'n03929660',
            'n03930313',
            'n03937543',
            'n03942813',
            'n03944341',
            'n03961711',
            'n03970156',
            'n03982430',
            'n03991062',
            'n03995372',
            'n03998194',
            'n04005630',
            'n04023962',
            'n04033901',
            'n04040759',
            'n04067472',
            'n04074963',
            'n04116512',
            'n04118776',
            'n04125021',
            'n04127249',
            'n04131690',
            'n04141975',
            'n04153751',
            'n04154565',
            'n04201297',
            'n04204347',
            'n04209133',
            'n04209239',
            'n04228054',
            'n04235860',
            'n04243546',
            'n04252077',
            'n04254120',
            'n04258138',
            'n04265275',
            'n04270147',
            'n04275548',
            'n04330267',
            'n04332243',
            'n04336792',
            'n04347754',
            'n04371430',
            'n04371774',
            'n04372370',
            'n04376876',
            'n04409515',
            'n04417672',
            'n04418357',
            'n04423845',
            'n04429376',
            'n04435653',
            'n04442312',
            'n04482393',
            'n04501370',
            'n04507155',
            'n04525305',
            'n04542943',
            'n04554684',
            'n04557648',
            'n04562935',
            'n04579432',
            'n04591157',
            'n04597913',
            'n04599235',
            'n06785654',
            'n06874185',
            'n07615774',
            'n07693725',
            'n07695742',
            'n07697537',
            'n07711569',
            'n07714990',
            'n07715103',
            'n07716358',
            'n07717410',
            'n07718472',
            'n07720875',
            'n07742313',
            'n07745940',
            'n07747607',
            'n07749582',
            'n07753275',
            'n07753592',
            'n07754684',
            'n07768694',
            'n07836838',
            'n07871810',
            'n07873807',
            'n07880968',
            'n09229709',
            'n09472597',
            'n12144580',
            'n12267677',
            'n13052670',
        ]
        imagenet_o_mask = [
            wnid in set(imagenet_o_wnids) for wnid in all_imagenet_wordnet_ids
        ]
        ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_o_mask) if mask]

    elif dataset_name == 'objectnet':
        assert split == 'test', f'Only `test` split available for {dataset_name}'
        # downloadable from
        # https://objectnet.dev/downloads/objectnet-1.0.zip
        # or https://www.dropbox.com/s/raw/cxeztdtm16nzvuw/objectnet-1.0.zip
        if not os.path.exists(root):
            print('Downloading objectnet...')
            call('wget https://objectnet.dev/downloads/objectnet-1.0.zip', shell=True)
            # Untar and move to `root`
            call(
                (
                    'UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE '
                    'unzip -P objectnetisatestset objectnet-1.0.zip'
                ),
                shell=True,
            )
            os.makedirs(root)
            call(f'mv objectnet-1.0 {root}', shell=True)
            call(f'cp {root}/objectnet-1.0/mappings/* {root}', shell=True)
        ds = objectnet.ObjectNetDataset(root=root, transform=transform)

    elif dataset_name == 'voc2007':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        ds = voc2007.PASCALVoc2007Cropped(
            root=root, set=split, transform=transform, download=download, **kwargs
        )

    elif dataset_name == 'voc2007_multilabel':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        ds = voc2007.PASCALVoc2007(
            root=root, set=split, transform=transform, download=download, **kwargs
        )

    elif dataset_name.startswith('sugar_crepe'):
        # https://github.com/RAIVNLab/sugar-crepe/tree/main
        _, task = dataset_name.split('/')
        assert task in (
            'add_att',
            'add_obj',
            'replace_att',
            'replace_obj',
            'replace_rel',
            'swap_att',
            'swap_obj',
        ), f'Unknown task {task} for {dataset_name}'
        assert split == 'test', f'Only `test` split available for {dataset_name}'
        archive_name = 'val2017.zip'
        root_split = os.path.join(root, archive_name.replace('.zip', ''))
        if not os.path.exists(root_split):
            print(f'Downloading coco captions {archive_name}...')
            if not os.path.exists(os.path.join(root, archive_name)):
                call(
                    (
                        f'wget http://images.cocodataset.org/zips/{archive_name} '
                        f'--output-document={root}/{archive_name}'
                    ),
                    shell=True,
                )
            call(f'unzip {root}/{archive_name} -d {root}', shell=True)
        ann = f'{root}/{task}.json'
        if not os.path.exists(ann):
            url = (
                f'https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/'
                f'data/{task}.json'
            )
            call(f'wget {url} --output-document={ann}', shell=True)
        ds = sugar_crepe.SugarCrepe(
            root=os.path.join(root, 'val2017'),
            ann_file=ann,
            transform=transform,
        )

    elif dataset_name == 'winoground':
        ds = winoground.WinoGround(root=root, transform=transform)

    elif dataset_name == 'mscoco_captions':
        # https://github.com/mehdidc/retrieval_annotations/releases/tag/
        # 1.0.0(annotations)
        if split == 'train':
            archive_name = 'train2014.zip'
        elif split in ('val', 'test'):
            archive_name = 'val2014.zip'
        else:
            raise ValueError(
                f'split should be `train` or `val` or `test` for `{dataset_name}`'
            )
        root_split = os.path.join(root, archive_name.replace('.zip', ''))
        if not os.path.exists(root_split):
            print(f'Downloading mscoco_captions {archive_name}...')
            if not os.path.exists(os.path.join(root, archive_name)):
                call(
                    (
                        f'wget http://images.cocodataset.org/zips/{archive_name} '
                        f'--output-document={root}/{archive_name}'
                    ),
                    shell=True,
                )
            call(f'unzip {root}/{archive_name} -d {root}', shell=True)
        if not annotation_file:
            annotation_file = f'{root}/coco_{split}_karpathy.json'
        if not os.path.exists(annotation_file):
            call(
                (
                    f'wget https://github.com/mehdidc/retrieval_annotations/releases/'
                    f'download/1.0.0/coco_{split}_karpathy.json '
                    f'--output-document={annotation_file}'
                ),
                shell=True,
            )
        ds = CocoCaptions(
            root=root_split, annFile=annotation_file, transform=transform, **kwargs
        )

    elif dataset_name == 'multilingual_mscoco_captions':
        from clip_benchmark.datasets import multilingual_mscoco

        if language not in multilingual_mscoco.SUPPORTED_LANGUAGES:
            raise ValueError('Unsupported language for multilingual_ms_coco:', language)
        annotation_file = os.path.join(
            root, multilingual_mscoco.OUTPUT_FILENAME_TEMPLATE.format(language)
        )
        if not os.path.exists(annotation_file):
            multilingual_mscoco.create_annotation_file(root, language)
        ds = multilingual_mscoco.Multilingual_MSCOCO(
            root=root, ann_file=annotation_file, transform=transform, **kwargs
        )

    elif dataset_name == 'crossmodal3600':
        from clip_benchmark.datasets import crossmodal3600

        if language not in crossmodal3600.SUPPORTED_LANGUAGES:
            raise ValueError('Unsupported language for Crossmodal-3600:', language)
        annotation_file = os.path.join(
            root, crossmodal3600.OUTPUT_FILENAME_TEMPLATE.format(language)
        )
        if not os.path.exists(annotation_file):
            crossmodal3600.create_annotation_file(root, language)
        ds = crossmodal3600.Crossmodal3600(
            root=root, ann_file=annotation_file, transform=transform, **kwargs
        )

    elif dataset_name == 'xtd200':
        from clip_benchmark.datasets import xtd200

        if language not in xtd200.SUPPORTED_LANGUAGES:
            raise ValueError('Unsupported language for xtd200:', language)
        annotation_file = os.path.join(
            root, xtd200.OUTPUT_FILENAME_TEMPLATE.format(language)
        )
        if not os.path.exists(annotation_file):
            xtd200.create_annotation_file(root, language)
        ds = xtd200.XTD200(
            root=root, ann_file=annotation_file, transform=transform, **kwargs
        )

    elif dataset_name == 'flickr30k-200':
        from clip_benchmark.datasets import flickr30k_200

        if language not in flickr30k_200.SUPPORTED_LANGUAGES:
            raise ValueError('Unsupported language for flickr30k-200:', language)
        annotation_file = os.path.join(
            root, flickr30k_200.OUTPUT_FILENAME_TEMPLATE.format(language)
        )
        if not os.path.exists(annotation_file):
            flickr30k_200.create_annotation_file(root, language)

        ds = flickr30k_200.Flickr30k_200(
            root=root, ann_file=annotation_file, transform=transform, **kwargs
        )

    elif dataset_name == 'flickr30k':
        # downloadable from https://www.kaggle.com/datasets/adityajn105/flickr30k
        # https://github.com/mehdidc/retrieval_annotations/releases/tag/1.0.0
        # (annotations)
        # `kaggle datasets download -d adityajn105/flickr30k`
        assert split in (
            'train',
            'val',
            'test',
        ), f'Only `train` and `val` and `test` split available for {dataset_name}'
        if not os.path.exists(root):
            # Automatic download
            print('Downloading flickr30k...')
            if not has_kaggle():
                print(
                    'Kaggle is needed to download the dataset. Please install it via '
                    '`pip install kaggle`'
                )
                sys.exit(1)
            call(
                'kaggle datasets download -d hsankesara/flickr-image-dataset',
                shell=True,
            )
            call(f'unzip flickr-image-dataset.zip', shell=True)
            call(
                (
                    f'mv flickr30k_images/flickr30k_images {root} '
                    f'&& rm -rf flickr30k_images'
                ),
                shell=True,
            )
        if not annotation_file:
            if language == 'en':
                annotation_file = f'{root}/flickr30k_{split}_karpathy.txt'
            elif language == 'zh':
                annotation_file = f'{root}/flickr30k_{split}_zh.txt'
            else:
                raise ValueError(
                    f'Unsupported language {language} for `{dataset_name}`'
                )
        if not os.path.exists(annotation_file):
            # Download Flickr30K Karpathy test set
            if language == 'en':
                call(
                    (
                        f'wget https://github.com/mehdidc/retrieval_annotations/'
                        f'releases/download/1.0.0/flickr30k_{split}_karpathy.txt '
                        f'--output-document={annotation_file}'
                    ),
                    shell=True,
                )
            elif language == 'zh':
                call(
                    (
                        f'wget https://github.com/mehdidc/retrieval_annotations/'
                        f'releases/download/1.0.0/flickr30k_{split}_zh.txt '
                        f'--output-document={annotation_file}'
                    ),
                    shell=True,
                )
            else:
                raise ValueError(
                    f'Unsupported language {language} for `{dataset_name}`'
                )
        ds = flickr.Flickr(
            root=root, ann_file=annotation_file, transform=transform, **kwargs
        )

    elif dataset_name == 'flickr8k':
        assert split in (
            'train',
            'val',
            'test',
        ), f'Only `train` and `val` and `test` split available for {dataset_name}'
        # downloadable from https://www.kaggle.com/datasets/adityajn105/flickr8k
        # `kaggle datasets download -d adityajn105/flickr8k`
        # https://github.com/mehdidc/retrieval_annotations/releases/tag/1.0.0
        # (annotations)
        if not os.path.exists(root):
            # Automatic download
            print('Downloading flickr8k...')
            if not has_kaggle():
                print(
                    'Kaggle is needed to download the dataset. Please install it '
                    'via `pip install kaggle`'
                )
                sys.exit(1)
            call('kaggle datasets download -d adityajn105/flickr8k', shell=True)
            call(f'unzip flickr8k.zip', shell=True)
            call(f'mv Images {root}', shell=True)
            call(f'mv captions.txt {root}', shell=True)
        if not annotation_file:
            if language == 'en':
                annotation_file = f'{root}/flickr8k_{split}_karpathy.txt'
            elif language == 'zh':
                annotation_file = f'{root}/flickr8k_{split}_zh.txt'
            else:
                raise ValueError(
                    f'Unsupported language {language} for `{dataset_name}`'
                )
        if not os.path.exists(annotation_file):
            # Download Flickr8K Karpathy test set
            if language == 'en':
                call(
                    (
                        f'wget https://github.com/mehdidc/retrieval_annotations/'
                        f'releases/download/1.0.0/flickr8k_{split}_karpathy.txt '
                        f'--output-document={annotation_file}'
                    ),
                    shell=True,
                )
            elif language == 'zh':
                call(
                    (
                        f'wget https://github.com/mehdidc/retrieval_annotations/'
                        f'releases/download/1.0.0/flickr8k_{split}_zh.txt '
                        f'--output-document={annotation_file}'
                    ),
                    shell=True,
                )
            else:
                raise ValueError(
                    f'Unsupported language {language} for `{dataset_name}`'
                )
        ds = flickr.Flickr(
            root=root, ann_file=annotation_file, transform=transform, **kwargs
        )

    elif dataset_name == 'food101':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        ds = Food101(
            root=root, split=split, transform=transform, download=download, **kwargs
        )
        # we use the default class names, we just  replace "_" by spaces
        # to delimit words
        ds.classes = [cl.replace('_', ' ') for cl in ds.classes]

    elif dataset_name == 'sun397':
        warnings.warn(
            f'split argument ignored for `{dataset_name}`, '
            f'there are no pre-defined train/test splits for this dataset'
        )
        # we use the default class names, we just  replace "_" and "/" by spaces
        # to delimit words
        ds = SUN397(root=root, transform=transform, download=download, **kwargs)
        ds.classes = [cl.replace('_', ' ').replace('/', ' ') for cl in ds.classes]

    elif dataset_name == 'cars':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        ds = StanfordCars(
            root=root, split=split, transform=transform, download=download, **kwargs
        )

    elif dataset_name == 'fgvc_aircraft':
        assert split in ('train', 'val', 'trainval', 'test'), (
            f'Only `train` and `val` and `trainval` and `test` split available for '
            f'{dataset_name}'
        )
        ds = FGVCAircraft(
            root=root,
            annotation_level='variant',
            split=split,
            transform=transform,
            download=download,
            **kwargs,
        )

    elif dataset_name == 'dtd':
        assert split in (
            'train',
            'val',
            'test',
        ), f'Only `train` and `val` and `test` split available for {dataset_name}'
        ds = DTD(
            root=root, split=split, transform=transform, download=download, **kwargs
        )

    elif dataset_name == 'pets':
        assert split in (
            'trainval',
            'test',
        ), f'Only `trainval` and `test` split available for {dataset_name}'
        ds = OxfordIIITPet(
            root=root,
            split=split,
            target_types='category',
            transform=transform,
            download=download,
            **kwargs,
        )

    elif dataset_name == 'caltech101':
        warnings.warn(
            f'split argument ignored for `{dataset_name}`, '
            f'there are no pre-defined train/test splits for this dataset'
        )
        # broken download link (can't download google drive), fixed by this PR
        # https://github.com/pytorch/vision/pull/5645
        # also available in "vtab/caltech101" using VTAB splits, we advice to use
        # VTAB version rather than this one since in this one (torchvision) there are
        # no pre-defined test splits
        ds = caltech101.Caltech101(
            root=root,
            target_type='category',
            transform=transform,
            download=download,
            **kwargs,
        )
        ds.classes = default_classnames['caltech101']

    elif dataset_name == 'flowers':
        assert split in (
            'train',
            'val',
            'test',
        ), f'Only `train` and `val` and `test` split available for {dataset_name}'
        ds = Flowers102(
            root=root, split=split, transform=transform, download=download, **kwargs
        )
        # class indices started by 1 until it was fixed in  a  PR (#TODO link of the PR)
        # if older torchvision version, fix it using a target transform that decrements
        # label index
        # TODO figure out minimal torchvision version needed instead of decrementing
        if ds[0][1] == 1:
            ds.target_transform = lambda y: y - 1
        ds.classes = default_classnames['flowers']

    elif dataset_name == 'mnist':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        ds = MNIST(
            root=root, train=train, transform=transform, download=download, **kwargs
        )
        ds.classes = default_classnames['mnist']

    elif dataset_name == 'stl10':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        ds = STL10(
            root=root, split=split, transform=transform, download=download, **kwargs
        )

    elif dataset_name == 'eurosat':
        warnings.warn(
            f'split argument ignored for `{dataset_name}`, '
            f'there are no pre-defined train/test splits for this dataset'
        )
        ds = EuroSAT(root=root, transform=transform, download=download, **kwargs)
        ds.classes = default_classnames['eurosat']

    elif dataset_name == 'gtsrb':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        ds = GTSRB(
            root=root, split=split, transform=transform, download=download, **kwargs
        )
        ds.classes = default_classnames['gtsrb']

    elif dataset_name == 'country211':
        assert split in (
            'train',
            'valid',
            'test',
        ), f'Only `train` and `valid` and `test` split available for {dataset_name}'
        ds = Country211(
            root=root, split=split, transform=transform, download=download, **kwargs
        )
        ds.classes = default_classnames['country211']

    elif dataset_name == 'pcam':
        assert split in (
            'train',
            'val',
            'test',
        ), f'Only `train` and `val` and `test` split available for {dataset_name}'
        # Dead link. Fixed by this PR on torchvision
        # https://github.com/pytorch/vision/pull/5645
        # TODO figure out minimal torchvision version needed
        ds = PCAM(
            root=root, split=split, transform=transform, download=download, **kwargs
        )
        ds.classes = default_classnames['pcam']

    elif dataset_name == 'renderedsst2':
        assert split in (
            'train',
            'val',
            'test',
        ), f'Only `train` and `val` and `test` split available for {dataset_name}'
        ds = RenderedSST2(
            root=root, split=split, transform=transform, download=download, **kwargs
        )

    elif dataset_name == 'fer2013':
        assert split in (
            'train',
            'test',
        ), f'Only `train` and `test` split available for {dataset_name}'
        # Downloadable from  https://www.kaggle.com/datasets/msambare/fer2013
        # `kaggle datasets download -d msambare/fer2013`
        if not os.path.exists(root):
            # Automatic download
            print('Downloading fer2013...')
            if not has_kaggle():
                print(
                    'Kaggle is needed to download the dataset. Please install it '
                    'via `pip install kaggle`'
                )
                sys.exit(1)
            call('kaggle datasets download -d msambare/fer2013', shell=True)
            call(f'unzip fer2013.zip -d {root}', shell=True)
        root = os.path.join(root, 'train' if train else 'test')
        ds = ImageFolder(root=root, transform=transform)
        ds.classes = default_classnames['fer2013']

    elif dataset_name.startswith('tfds/'):
        # TFDS datasets support using `timm` and `tensorflow_datasets`
        prefix, *name_list = dataset_name.split('/')
        name = '/'.join(name_list)
        ds = build_tfds_dataset(
            name, download=download, split=split, data_dir=root, transform=transform
        )

    elif dataset_name.startswith('vtab/'):
        # VTAB datasets support using `tensorflow_datasets` and `task_adaptation`
        prefix, *name_list = dataset_name.split('/')
        name = '/'.join(name_list)
        ds = build_vtab_dataset(
            name,
            split=split,
            data_dir=root,
            transform=transform,
            classnames=default_classnames,
        )

    elif dataset_name.startswith('wds/'):
        # WebDataset support using `webdataset` library
        ds = build_wds_dataset(
            transform=transform,
            split=split,
            data_dir=root,
            cache_dir=wds_cache_dir,
        )
        # WDS specify classnames and templates on its own.
    elif dataset_name == 'dummy':
        ds = Dummy()
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}.')

    default_dataset_for_templates = 'imagenet1k'
    if (
        dataset_name.startswith('tfds/')
        or dataset_name.startswith('vtab/')
        or dataset_name.startswith('wds/')
    ):
        prefix, *rest = dataset_name.split('/')
        short_name = '/'.join(rest)
        # if it's a vtab/tfds/wds/ dataset, we look for e.g. vtab/<name>
        # as well as <name> in the custom template file/classname file,
        # whichever is found.
        keys_to_lookup = [dataset_name, short_name]
    else:
        keys_to_lookup = [dataset_name]

    if use_classnames_and_templates:
        # Specify templates for the dataset (if needed)
        if custom_templates:
            # We override with custom templates ONLY if they are provided,
            # which is the case when `custom_templates` is loaded.
            ds.templates = value_from_first_key_found(
                custom_templates, keys=keys_to_lookup + [default_dataset_for_templates]
            )
            assert (
                ds.templates is not None
            ), f'Templates not specified for {dataset_name}'
        elif not hasattr(ds, 'templates'):
            # No templates specified by the dataset itself,
            # so we use  templates are packaged with CLIP benchmark
            # (loaded from <LANG>_zeroshot_classification_templates.json).
            ds.templates = value_from_first_key_found(
                default_templates, keys=keys_to_lookup + [default_dataset_for_templates]
            )
            assert (
                ds.templates is not None
            ), f'Templates not specified for {dataset_name}'
        else:
            # dataset has templates already (e.g., WDS case), so we keep it as is.
            pass

        # We override with custom classnames ONLY if they are provided.
        if custom_classnames:
            ds.classes = value_from_first_key_found(
                custom_classnames, keys=keys_to_lookup
            )

        assert ds.classes is not None, f'Classes not specified for {dataset_name}'
        assert ds.templates is not None, f'Templates not specified for {dataset_name}'

    return ds


def build_vtab_dataset(
    dataset_name, transform, split='test', data_dir='root', classnames=None
):
    # Using VTAB splits instead of default TFDS splits
    from clip_benchmark.datasets.tfds import (
        VTABIterableDataset,
        disable_gpus_on_tensorflow,
        download_tfds_dataset,
    )

    classnames = classnames or {}

    # avoid Tensorflow owning GPUs to not clash with PyTorch
    disable_gpus_on_tensorflow()

    # by default we take classes from TFDS (default behavior if `classes` stays None),
    # except for the datasets that will override `classes` (e.g., clevr_*)
    classes = None
    if dataset_name == 'caltech101':
        from task_adaptation.data.caltech import Caltech101

        tfds_dataset = Caltech101(data_dir=data_dir)
        classes = classnames['caltech101_vtab']

    elif dataset_name == 'cars':
        from task_adaptation.data.cars import CarsData

        tfds_dataset = CarsData(data_dir=data_dir)

    elif dataset_name in ('cifar10', 'cifar100'):
        from task_adaptation.data.cifar import CifarData

        tfds_dataset = CifarData(
            data_dir=data_dir, num_classes=10 if dataset_name == 'cifar10' else 100
        )

    elif dataset_name.startswith('clevr_'):
        from task_adaptation.data.clevr import CLEVRData

        task = _extract_task(dataset_name)
        assert task in ('count_all', 'closest_object_distance')
        tfds_dataset = CLEVRData(task=task, data_dir=data_dir)
        if task == 'count_all':
            classes = classnames['clevr_count_all']
        elif task == 'closest_object_distance':
            classes = classnames['clevr_closest_object_distance']
        else:
            raise ValueError(f'non supported: {task}')

    elif dataset_name == 'cub':
        from task_adaptation.data.cub import CUB2011Data

        tfds_dataset = CUB2011Data(data_dir=data_dir)

    elif dataset_name == 'diabetic_retinopathy':
        # Needs manual download from Kaggle
        # 1) `kaggle competitions download -c diabetic-retinopathy-detection` on
        #       $ROOT/downloads/manual
        # 2) extract archives  on $ROOT/downloads/manual
        if not os.path.exists(data_dir):
            # Automatic download
            print('Downloading diabetic_retinopathy...')
            if not has_kaggle():
                print(
                    'Kaggle is needed to download the dataset. Please install it via '
                    '`pip install kaggle`'
                )
                sys.exit(1)
            os.makedirs(os.path.join(data_dir, 'downloads', 'manual'))
            call(
                (
                    f'kaggle competitions download -c diabetic-retinopathy-detection '
                    f'-p {data_dir}/downloads/manual'
                ),
                shell=True,
            )
            call(
                (
                    f'cd {data_dir}/downloads/manual;unzip '
                    f'diabetic-retinopathy-detection.zip;cat train.zip*>train.zip;'
                    f'cat test.zip*>test.zip;unzip train.zip; '
                    f'unzip test.zip;unzip sample.zip;unzip trainLabels.csv.zip'
                ),
                shell=True,
            )
        from task_adaptation.data.diabetic_retinopathy import RetinopathyData

        tfds_dataset = RetinopathyData(config='btgraham-300', data_dir=data_dir)
        classes = classnames['diabetic_retinopathy']

    elif dataset_name == 'dmlab':
        from task_adaptation.data.dmlab import DmlabData

        download_tfds_dataset(
            'dmlab', data_dir=data_dir
        )  # it's not called in the original VTAB code, so we do it explictly
        tfds_dataset = DmlabData(data_dir=data_dir)
        classes = classnames['dmlab']

    elif dataset_name.startswith('dsprites_'):
        from task_adaptation.data.dsprites import DSpritesData

        task = _extract_task(dataset_name)
        assert task in (
            'label_shape',
            'label_scale',
            'label_orientation',
            'label_x_position',
            'label_y_position',
        )
        tfds_dataset = DSpritesData(task, data_dir=data_dir)
        classes = tfds_dataset._dataset_builder.info.features[task].names

    elif dataset_name == 'dtd':
        from task_adaptation.data.dtd import DTDData

        tfds_dataset = DTDData(data_dir=data_dir)

    elif dataset_name == 'eurosat':
        from task_adaptation.data.eurosat import EurosatData

        tfds_dataset = EurosatData(subset='rgb', data_key='image', data_dir=data_dir)
        classes = classnames['eurosat']

    elif dataset_name == 'food101':
        from task_adaptation.data.food101 import Food101Data

        tfds_dataset = Food101Data(data_dir=data_dir)

    elif dataset_name == 'inaturalist':
        from task_adaptation.data.inaturalist import INaturalistData

        tfds_dataset = INaturalistData(data_dir=data_dir, year=2017)

    elif dataset_name.startswith('kitti_'):
        from clip_benchmark.datasets.kitti import KittiData

        task = _extract_task(dataset_name)
        assert task in (
            'count_all',
            'count_left',
            'count_far',
            'count_near',
            'closest_object_distance',
            'closest_object_x_location',
            'count_vehicles',
            'closest_vehicle_distance',
        )
        tfds_dataset = KittiData(task=task, data_dir=data_dir)
        if task == 'closest_vehicle_distance':
            classes = classnames['kitti_closest_vehicle_distance']
        else:
            raise ValueError(f'Unsupported task: {task}')

    elif dataset_name == 'flowers':
        from task_adaptation.data.oxford_flowers102 import OxfordFlowers102Data

        tfds_dataset = OxfordFlowers102Data(data_dir=data_dir)

    elif dataset_name == 'pets':
        from task_adaptation.data.oxford_iiit_pet import OxfordIIITPetData

        tfds_dataset = OxfordIIITPetData(data_dir=data_dir)
        classes = classnames['pets']

    elif dataset_name == 'pcam':
        from task_adaptation.data.patch_camelyon import PatchCamelyonData

        tfds_dataset = PatchCamelyonData(data_dir=data_dir)
        classes = classnames['pcam']

    elif dataset_name == 'resisc45':
        # Needs download from OneDrive: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
        # The archive needs to to be put at <DATASET_ROOT>/downloads/manual
        # then extracted
        if not os.path.exists(data_dir):
            os.makedirs(os.path.join(data_dir, 'downloads', 'manual'))
            call(
                (
                    f"wget 'https://onedrive.live.com/download?"
                    f"resid=5C5E061130630A68!107&authkey=!AHHNaHIlzp_IXjs' "
                    f'--output-document={data_dir}/downloads/manual/resisc45.rar'
                ),
                shell=True,
            )
            call(f'cd {data_dir}/downloads/manual;unrar x resisc45.rar', shell=True)
        from task_adaptation.data.resisc45 import Resisc45Data

        tfds_dataset = Resisc45Data(data_dir=data_dir)

    elif dataset_name.startswith('smallnorb_'):
        from task_adaptation.data.smallnorb import SmallNORBData

        task = _extract_task(dataset_name)
        assert task in (
            'label_category',
            'label_elevation',
            'label_azimuth',
            'label_lighting',
        )
        tfds_dataset = SmallNORBData(predicted_attribute=task, data_dir=data_dir)
        classes = tfds_dataset._dataset_builder.info.features[task].names

    elif dataset_name == 'sun397':
        from task_adaptation.data.sun397 import Sun397Data

        # FIXME There is a problem in `sun397`, when TFDS tries download it
        # there is an image that cannot be decoded. For the time being
        # we will use torchvision's SUN397 instead.
        tfds_dataset = Sun397Data(config='tfds', data_dir=data_dir)

    elif dataset_name == 'svhn':
        from task_adaptation.data.svhn import SvhnData

        tfds_dataset = SvhnData(data_dir=data_dir)
        classes = classnames['svhn']
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    ds = VTABIterableDataset(
        tfds_dataset,
        input_name='image',
        label_name='label',
        transform=transform,
        target_transform=int,
        split=split,
        classes=classes,
    )
    return ds


def build_tfds_dataset(
    name, transform, download=True, split='test', data_dir='root', classes=None
):
    from clip_benchmark.datasets.tfds import disable_gpus_on_tensorflow

    disable_gpus_on_tensorflow()
    import tensorflow_datasets as tfds
    import timm

    builder = tfds.builder(name, data_dir=data_dir)
    if download:
        builder.download_and_prepare()
    splits = list(builder.info.splits.keys())
    assert split in splits, (split, splits)
    ds = timm.data.create_dataset(
        f'tfds/{name}', data_dir, split=split, transform=transform, target_transform=int
    )
    ds.classes = builder.info.features['label'].names if classes is None else classes
    return ds


def build_wds_dataset(transform, split='test', data_dir='root', cache_dir=None):
    """
    Load a dataset in WebDataset format. Either local paths or HTTP URLs can be
    specified. Expected file structure is:
    ```
    data_dir/
        train/
            nshards.txt
            0.tar
            1.tar
            ...
        test/
            nshards.txt
            0.tar
            1.tar
            ...
        classnames.txt
        zeroshot_classification_templates.txt
        dataset_type.txt
    ```
    Classnames and templates are required for zeroshot classification, while dataset
    type (equal to "retrieval") is required for zeroshot retrieval datasets.

    You can use the `clip_benchmark_export_wds` or corresponding API
    (`clip_benchmark.webdataset_builder.convert_dataset`) to convert datasets to this
    format.

    Set `cache_dir` to a path to cache the dataset, otherwise, no caching will occur.
    """
    import webdataset as wds

    def read_txt(fname):
        if '://' in fname:
            stream = os.popen("curl -L -s --fail '%s'" % fname, 'r')
            value = stream.read()
            if stream.close():
                raise FileNotFoundError('Failed to retreive data')
        else:
            with open(fname, 'r') as file:
                value = file.read()
        return value

    # Special handling for Huggingface datasets
    # Git LFS files have a different file path to access the raw data than other files
    if data_dir.startswith('https://huggingface.co/datasets'):
        # Format: https://huggingface.co/datasets/<USERNAME>/<REPO>/tree/<BRANCH>
        *split_url_head, _, url_path = data_dir.split('/', 7)
        url_head = '/'.join(split_url_head)
        metadata_dir = '/'.join([url_head, 'raw', url_path])
        tardata_dir = '/'.join([url_head, 'resolve', url_path])
    else:
        metadata_dir = tardata_dir = data_dir

    # Get number of shards
    nshards_fname = os.path.join(metadata_dir, split, 'nshards.txt')
    nshards = int(
        read_txt(nshards_fname)
    )  # Do not catch FileNotFound, nshards.txt should be mandatory
    # Get dataset type (classification or retrieval)

    type_fname = os.path.join(metadata_dir, 'dataset_type.txt')
    try:
        dataset_type = read_txt(type_fname).strip().lower()
    except FileNotFoundError:
        # print("WARNING: dataset_type.txt not found, assuming type=classification")
        dataset_type = 'classification'
    #
    filepattern = os.path.join(tardata_dir, split, '{0..%d}.tar' % (nshards - 1))

    # Load webdataset (support WEBP, PNG, and JPG for now)
    if not cache_dir or not isinstance(cache_dir, str):
        cache_dir = None
    dataset = wds.WebDataset(
        filepattern, cache_dir=cache_dir, nodesplitter=lambda src: src
    ).decode(
        wds.autodecode.ImageHandler('pil', extensions=['webp', 'png', 'jpg', 'jpeg'])
    )
    # Load based on classification or retrieval task
    if dataset_type == 'retrieval':
        dataset = dataset.to_tuple(['webp', 'png', 'jpg', 'jpeg'], 'txt').map_tuple(
            transform, str.splitlines
        )
        dataset.classes = dataset.templates = None
    else:
        label_type = (
            'npy' if dataset_type == 'multilabel' else 'cls'
        )  # Special case for multilabel
        dataset = dataset.to_tuple(
            ['webp', 'png', 'jpg', 'jpeg'], label_type
        ).map_tuple(transform, None)
        # Get class names if present
        classnames_fname = os.path.join(metadata_dir, 'classnames.txt')
        try:
            dataset.classes = [
                line.strip() for line in read_txt(classnames_fname).splitlines()
            ]
        except FileNotFoundError:
            print('WARNING: classnames.txt not found')
            dataset.classes = None
        # Get zeroshot classification templates if present
        templates_fname = os.path.join(
            metadata_dir, 'zeroshot_classification_templates.txt'
        )
        try:
            dataset.templates = [
                line.strip() for line in read_txt(templates_fname).splitlines()
            ]
        except FileNotFoundError:
            print('WARNING: zeroshot_classification_templates.txt not found')
            dataset.templates = None

    return dataset


# use by imagenet robustness datasets
all_imagenet_wordnet_ids = [
    'n01440764',
    'n01443537',
    'n01484850',
    'n01491361',
    'n01494475',
    'n01496331',
    'n01498041',
    'n01514668',
    'n01514859',
    'n01518878',
    'n01530575',
    'n01531178',
    'n01532829',
    'n01534433',
    'n01537544',
    'n01558993',
    'n01560419',
    'n01580077',
    'n01582220',
    'n01592084',
    'n01601694',
    'n01608432',
    'n01614925',
    'n01616318',
    'n01622779',
    'n01629819',
    'n01630670',
    'n01631663',
    'n01632458',
    'n01632777',
    'n01641577',
    'n01644373',
    'n01644900',
    'n01664065',
    'n01665541',
    'n01667114',
    'n01667778',
    'n01669191',
    'n01675722',
    'n01677366',
    'n01682714',
    'n01685808',
    'n01687978',
    'n01688243',
    'n01689811',
    'n01692333',
    'n01693334',
    'n01694178',
    'n01695060',
    'n01697457',
    'n01698640',
    'n01704323',
    'n01728572',
    'n01728920',
    'n01729322',
    'n01729977',
    'n01734418',
    'n01735189',
    'n01737021',
    'n01739381',
    'n01740131',
    'n01742172',
    'n01744401',
    'n01748264',
    'n01749939',
    'n01751748',
    'n01753488',
    'n01755581',
    'n01756291',
    'n01768244',
    'n01770081',
    'n01770393',
    'n01773157',
    'n01773549',
    'n01773797',
    'n01774384',
    'n01774750',
    'n01775062',
    'n01776313',
    'n01784675',
    'n01795545',
    'n01796340',
    'n01797886',
    'n01798484',
    'n01806143',
    'n01806567',
    'n01807496',
    'n01817953',
    'n01818515',
    'n01819313',
    'n01820546',
    'n01824575',
    'n01828970',
    'n01829413',
    'n01833805',
    'n01843065',
    'n01843383',
    'n01847000',
    'n01855032',
    'n01855672',
    'n01860187',
    'n01871265',
    'n01872401',
    'n01873310',
    'n01877812',
    'n01882714',
    'n01883070',
    'n01910747',
    'n01914609',
    'n01917289',
    'n01924916',
    'n01930112',
    'n01943899',
    'n01944390',
    'n01945685',
    'n01950731',
    'n01955084',
    'n01968897',
    'n01978287',
    'n01978455',
    'n01980166',
    'n01981276',
    'n01983481',
    'n01984695',
    'n01985128',
    'n01986214',
    'n01990800',
    'n02002556',
    'n02002724',
    'n02006656',
    'n02007558',
    'n02009229',
    'n02009912',
    'n02011460',
    'n02012849',
    'n02013706',
    'n02017213',
    'n02018207',
    'n02018795',
    'n02025239',
    'n02027492',
    'n02028035',
    'n02033041',
    'n02037110',
    'n02051845',
    'n02056570',
    'n02058221',
    'n02066245',
    'n02071294',
    'n02074367',
    'n02077923',
    'n02085620',
    'n02085782',
    'n02085936',
    'n02086079',
    'n02086240',
    'n02086646',
    'n02086910',
    'n02087046',
    'n02087394',
    'n02088094',
    'n02088238',
    'n02088364',
    'n02088466',
    'n02088632',
    'n02089078',
    'n02089867',
    'n02089973',
    'n02090379',
    'n02090622',
    'n02090721',
    'n02091032',
    'n02091134',
    'n02091244',
    'n02091467',
    'n02091635',
    'n02091831',
    'n02092002',
    'n02092339',
    'n02093256',
    'n02093428',
    'n02093647',
    'n02093754',
    'n02093859',
    'n02093991',
    'n02094114',
    'n02094258',
    'n02094433',
    'n02095314',
    'n02095570',
    'n02095889',
    'n02096051',
    'n02096177',
    'n02096294',
    'n02096437',
    'n02096585',
    'n02097047',
    'n02097130',
    'n02097209',
    'n02097298',
    'n02097474',
    'n02097658',
    'n02098105',
    'n02098286',
    'n02098413',
    'n02099267',
    'n02099429',
    'n02099601',
    'n02099712',
    'n02099849',
    'n02100236',
    'n02100583',
    'n02100735',
    'n02100877',
    'n02101006',
    'n02101388',
    'n02101556',
    'n02102040',
    'n02102177',
    'n02102318',
    'n02102480',
    'n02102973',
    'n02104029',
    'n02104365',
    'n02105056',
    'n02105162',
    'n02105251',
    'n02105412',
    'n02105505',
    'n02105641',
    'n02105855',
    'n02106030',
    'n02106166',
    'n02106382',
    'n02106550',
    'n02106662',
    'n02107142',
    'n02107312',
    'n02107574',
    'n02107683',
    'n02107908',
    'n02108000',
    'n02108089',
    'n02108422',
    'n02108551',
    'n02108915',
    'n02109047',
    'n02109525',
    'n02109961',
    'n02110063',
    'n02110185',
    'n02110341',
    'n02110627',
    'n02110806',
    'n02110958',
    'n02111129',
    'n02111277',
    'n02111500',
    'n02111889',
    'n02112018',
    'n02112137',
    'n02112350',
    'n02112706',
    'n02113023',
    'n02113186',
    'n02113624',
    'n02113712',
    'n02113799',
    'n02113978',
    'n02114367',
    'n02114548',
    'n02114712',
    'n02114855',
    'n02115641',
    'n02115913',
    'n02116738',
    'n02117135',
    'n02119022',
    'n02119789',
    'n02120079',
    'n02120505',
    'n02123045',
    'n02123159',
    'n02123394',
    'n02123597',
    'n02124075',
    'n02125311',
    'n02127052',
    'n02128385',
    'n02128757',
    'n02128925',
    'n02129165',
    'n02129604',
    'n02130308',
    'n02132136',
    'n02133161',
    'n02134084',
    'n02134418',
    'n02137549',
    'n02138441',
    'n02165105',
    'n02165456',
    'n02167151',
    'n02168699',
    'n02169497',
    'n02172182',
    'n02174001',
    'n02177972',
    'n02190166',
    'n02206856',
    'n02219486',
    'n02226429',
    'n02229544',
    'n02231487',
    'n02233338',
    'n02236044',
    'n02256656',
    'n02259212',
    'n02264363',
    'n02268443',
    'n02268853',
    'n02276258',
    'n02277742',
    'n02279972',
    'n02280649',
    'n02281406',
    'n02281787',
    'n02317335',
    'n02319095',
    'n02321529',
    'n02325366',
    'n02326432',
    'n02328150',
    'n02342885',
    'n02346627',
    'n02356798',
    'n02361337',
    'n02363005',
    'n02364673',
    'n02389026',
    'n02391049',
    'n02395406',
    'n02396427',
    'n02397096',
    'n02398521',
    'n02403003',
    'n02408429',
    'n02410509',
    'n02412080',
    'n02415577',
    'n02417914',
    'n02422106',
    'n02422699',
    'n02423022',
    'n02437312',
    'n02437616',
    'n02441942',
    'n02442845',
    'n02443114',
    'n02443484',
    'n02444819',
    'n02445715',
    'n02447366',
    'n02454379',
    'n02457408',
    'n02480495',
    'n02480855',
    'n02481823',
    'n02483362',
    'n02483708',
    'n02484975',
    'n02486261',
    'n02486410',
    'n02487347',
    'n02488291',
    'n02488702',
    'n02489166',
    'n02490219',
    'n02492035',
    'n02492660',
    'n02493509',
    'n02493793',
    'n02494079',
    'n02497673',
    'n02500267',
    'n02504013',
    'n02504458',
    'n02509815',
    'n02510455',
    'n02514041',
    'n02526121',
    'n02536864',
    'n02606052',
    'n02607072',
    'n02640242',
    'n02641379',
    'n02643566',
    'n02655020',
    'n02666196',
    'n02667093',
    'n02669723',
    'n02672831',
    'n02676566',
    'n02687172',
    'n02690373',
    'n02692877',
    'n02699494',
    'n02701002',
    'n02704792',
    'n02708093',
    'n02727426',
    'n02730930',
    'n02747177',
    'n02749479',
    'n02769748',
    'n02776631',
    'n02777292',
    'n02782093',
    'n02783161',
    'n02786058',
    'n02787622',
    'n02788148',
    'n02790996',
    'n02791124',
    'n02791270',
    'n02793495',
    'n02794156',
    'n02795169',
    'n02797295',
    'n02799071',
    'n02802426',
    'n02804414',
    'n02804610',
    'n02807133',
    'n02808304',
    'n02808440',
    'n02814533',
    'n02814860',
    'n02815834',
    'n02817516',
    'n02823428',
    'n02823750',
    'n02825657',
    'n02834397',
    'n02835271',
    'n02837789',
    'n02840245',
    'n02841315',
    'n02843684',
    'n02859443',
    'n02860847',
    'n02865351',
    'n02869837',
    'n02870880',
    'n02871525',
    'n02877765',
    'n02879718',
    'n02883205',
    'n02892201',
    'n02892767',
    'n02894605',
    'n02895154',
    'n02906734',
    'n02909870',
    'n02910353',
    'n02916936',
    'n02917067',
    'n02927161',
    'n02930766',
    'n02939185',
    'n02948072',
    'n02950826',
    'n02951358',
    'n02951585',
    'n02963159',
    'n02965783',
    'n02966193',
    'n02966687',
    'n02971356',
    'n02974003',
    'n02977058',
    'n02978881',
    'n02979186',
    'n02980441',
    'n02981792',
    'n02988304',
    'n02992211',
    'n02992529',
    'n02999410',
    'n03000134',
    'n03000247',
    'n03000684',
    'n03014705',
    'n03016953',
    'n03017168',
    'n03018349',
    'n03026506',
    'n03028079',
    'n03032252',
    'n03041632',
    'n03042490',
    'n03045698',
    'n03047690',
    'n03062245',
    'n03063599',
    'n03063689',
    'n03065424',
    'n03075370',
    'n03085013',
    'n03089624',
    'n03095699',
    'n03100240',
    'n03109150',
    'n03110669',
    'n03124043',
    'n03124170',
    'n03125729',
    'n03126707',
    'n03127747',
    'n03127925',
    'n03131574',
    'n03133878',
    'n03134739',
    'n03141823',
    'n03146219',
    'n03160309',
    'n03179701',
    'n03180011',
    'n03187595',
    'n03188531',
    'n03196217',
    'n03197337',
    'n03201208',
    'n03207743',
    'n03207941',
    'n03208938',
    'n03216828',
    'n03218198',
    'n03220513',
    'n03223299',
    'n03240683',
    'n03249569',
    'n03250847',
    'n03255030',
    'n03259280',
    'n03271574',
    'n03272010',
    'n03272562',
    'n03290653',
    'n03291819',
    'n03297495',
    'n03314780',
    'n03325584',
    'n03337140',
    'n03344393',
    'n03345487',
    'n03347037',
    'n03355925',
    'n03372029',
    'n03376595',
    'n03379051',
    'n03384352',
    'n03388043',
    'n03388183',
    'n03388549',
    'n03393912',
    'n03394916',
    'n03400231',
    'n03404251',
    'n03417042',
    'n03424325',
    'n03425413',
    'n03443371',
    'n03444034',
    'n03445777',
    'n03445924',
    'n03447447',
    'n03447721',
    'n03450230',
    'n03452741',
    'n03457902',
    'n03459775',
    'n03461385',
    'n03467068',
    'n03476684',
    'n03476991',
    'n03478589',
    'n03481172',
    'n03482405',
    'n03483316',
    'n03485407',
    'n03485794',
    'n03492542',
    'n03494278',
    'n03495258',
    'n03496892',
    'n03498962',
    'n03527444',
    'n03529860',
    'n03530642',
    'n03532672',
    'n03534580',
    'n03535780',
    'n03538406',
    'n03544143',
    'n03584254',
    'n03584829',
    'n03590841',
    'n03594734',
    'n03594945',
    'n03595614',
    'n03598930',
    'n03599486',
    'n03602883',
    'n03617480',
    'n03623198',
    'n03627232',
    'n03630383',
    'n03633091',
    'n03637318',
    'n03642806',
    'n03649909',
    'n03657121',
    'n03658185',
    'n03661043',
    'n03662601',
    'n03666591',
    'n03670208',
    'n03673027',
    'n03676483',
    'n03680355',
    'n03690938',
    'n03691459',
    'n03692522',
    'n03697007',
    'n03706229',
    'n03709823',
    'n03710193',
    'n03710637',
    'n03710721',
    'n03717622',
    'n03720891',
    'n03721384',
    'n03724870',
    'n03729826',
    'n03733131',
    'n03733281',
    'n03733805',
    'n03742115',
    'n03743016',
    'n03759954',
    'n03761084',
    'n03763968',
    'n03764736',
    'n03769881',
    'n03770439',
    'n03770679',
    'n03773504',
    'n03775071',
    'n03775546',
    'n03776460',
    'n03777568',
    'n03777754',
    'n03781244',
    'n03782006',
    'n03785016',
    'n03786901',
    'n03787032',
    'n03788195',
    'n03788365',
    'n03791053',
    'n03792782',
    'n03792972',
    'n03793489',
    'n03794056',
    'n03796401',
    'n03803284',
    'n03804744',
    'n03814639',
    'n03814906',
    'n03825788',
    'n03832673',
    'n03837869',
    'n03838899',
    'n03840681',
    'n03841143',
    'n03843555',
    'n03854065',
    'n03857828',
    'n03866082',
    'n03868242',
    'n03868863',
    'n03871628',
    'n03873416',
    'n03874293',
    'n03874599',
    'n03876231',
    'n03877472',
    'n03877845',
    'n03884397',
    'n03887697',
    'n03888257',
    'n03888605',
    'n03891251',
    'n03891332',
    'n03895866',
    'n03899768',
    'n03902125',
    'n03903868',
    'n03908618',
    'n03908714',
    'n03916031',
    'n03920288',
    'n03924679',
    'n03929660',
    'n03929855',
    'n03930313',
    'n03930630',
    'n03933933',
    'n03935335',
    'n03937543',
    'n03938244',
    'n03942813',
    'n03944341',
    'n03947888',
    'n03950228',
    'n03954731',
    'n03956157',
    'n03958227',
    'n03961711',
    'n03967562',
    'n03970156',
    'n03976467',
    'n03976657',
    'n03977966',
    'n03980874',
    'n03982430',
    'n03983396',
    'n03991062',
    'n03992509',
    'n03995372',
    'n03998194',
    'n04004767',
    'n04005630',
    'n04008634',
    'n04009552',
    'n04019541',
    'n04023962',
    'n04026417',
    'n04033901',
    'n04033995',
    'n04037443',
    'n04039381',
    'n04040759',
    'n04041544',
    'n04044716',
    'n04049303',
    'n04065272',
    'n04067472',
    'n04069434',
    'n04070727',
    'n04074963',
    'n04081281',
    'n04086273',
    'n04090263',
    'n04099969',
    'n04111531',
    'n04116512',
    'n04118538',
    'n04118776',
    'n04120489',
    'n04125021',
    'n04127249',
    'n04131690',
    'n04133789',
    'n04136333',
    'n04141076',
    'n04141327',
    'n04141975',
    'n04146614',
    'n04147183',
    'n04149813',
    'n04152593',
    'n04153751',
    'n04154565',
    'n04162706',
    'n04179913',
    'n04192698',
    'n04200800',
    'n04201297',
    'n04204238',
    'n04204347',
    'n04208210',
    'n04209133',
    'n04209239',
    'n04228054',
    'n04229816',
    'n04235860',
    'n04238763',
    'n04239074',
    'n04243546',
    'n04251144',
    'n04252077',
    'n04252225',
    'n04254120',
    'n04254680',
    'n04254777',
    'n04258138',
    'n04259630',
    'n04263257',
    'n04264628',
    'n04265275',
    'n04266014',
    'n04270147',
    'n04273569',
    'n04275548',
    'n04277352',
    'n04285008',
    'n04286575',
    'n04296562',
    'n04310018',
    'n04311004',
    'n04311174',
    'n04317175',
    'n04325704',
    'n04326547',
    'n04328186',
    'n04330267',
    'n04332243',
    'n04335435',
    'n04336792',
    'n04344873',
    'n04346328',
    'n04347754',
    'n04350905',
    'n04355338',
    'n04355933',
    'n04356056',
    'n04357314',
    'n04366367',
    'n04367480',
    'n04370456',
    'n04371430',
    'n04371774',
    'n04372370',
    'n04376876',
    'n04380533',
    'n04389033',
    'n04392985',
    'n04398044',
    'n04399382',
    'n04404412',
    'n04409515',
    'n04417672',
    'n04418357',
    'n04423845',
    'n04428191',
    'n04429376',
    'n04435653',
    'n04442312',
    'n04443257',
    'n04447861',
    'n04456115',
    'n04458633',
    'n04461696',
    'n04462240',
    'n04465501',
    'n04467665',
    'n04476259',
    'n04479046',
    'n04482393',
    'n04483307',
    'n04485082',
    'n04486054',
    'n04487081',
    'n04487394',
    'n04493381',
    'n04501370',
    'n04505470',
    'n04507155',
    'n04509417',
    'n04515003',
    'n04517823',
    'n04522168',
    'n04523525',
    'n04525038',
    'n04525305',
    'n04532106',
    'n04532670',
    'n04536866',
    'n04540053',
    'n04542943',
    'n04548280',
    'n04548362',
    'n04550184',
    'n04552348',
    'n04553703',
    'n04554684',
    'n04557648',
    'n04560804',
    'n04562935',
    'n04579145',
    'n04579432',
    'n04584207',
    'n04589890',
    'n04590129',
    'n04591157',
    'n04591713',
    'n04592741',
    'n04596742',
    'n04597913',
    'n04599235',
    'n04604644',
    'n04606251',
    'n04612504',
    'n04613696',
    'n06359193',
    'n06596364',
    'n06785654',
    'n06794110',
    'n06874185',
    'n07248320',
    'n07565083',
    'n07579787',
    'n07583066',
    'n07584110',
    'n07590611',
    'n07613480',
    'n07614500',
    'n07615774',
    'n07684084',
    'n07693725',
    'n07695742',
    'n07697313',
    'n07697537',
    'n07711569',
    'n07714571',
    'n07714990',
    'n07715103',
    'n07716358',
    'n07716906',
    'n07717410',
    'n07717556',
    'n07718472',
    'n07718747',
    'n07720875',
    'n07730033',
    'n07734744',
    'n07742313',
    'n07745940',
    'n07747607',
    'n07749582',
    'n07753113',
    'n07753275',
    'n07753592',
    'n07754684',
    'n07760859',
    'n07768694',
    'n07802026',
    'n07831146',
    'n07836838',
    'n07860988',
    'n07871810',
    'n07873807',
    'n07875152',
    'n07880968',
    'n07892512',
    'n07920052',
    'n07930864',
    'n07932039',
    'n09193705',
    'n09229709',
    'n09246464',
    'n09256479',
    'n09288635',
    'n09332890',
    'n09399592',
    'n09421951',
    'n09428293',
    'n09468604',
    'n09472597',
    'n09835506',
    'n10148035',
    'n10565667',
    'n11879895',
    'n11939491',
    'n12057211',
    'n12144580',
    'n12267677',
    'n12620546',
    'n12768682',
    'n12985857',
    'n12998815',
    'n13037406',
    'n13040303',
    'n13044778',
    'n13052670',
    'n13054560',
    'n13133613',
    'n15075141',
]
