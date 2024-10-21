import hashlib
import os
import urllib
import warnings
from functools import partial
from typing import Dict, Iterable, Optional, Union

from tqdm import tqdm


try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False


from .constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INCEPTION_MEAN,
    INCEPTION_STD,
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    HF_WEIGHTS_NAME,
    HF_SAFE_WEIGHTS_NAME,
)
from .version import __version__

try:
    from huggingface_hub import hf_hub_download
    hf_hub_download = partial(hf_hub_download, library_name="open_clip", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False


def _pcfg(url='', hf_hub='', **kwargs):
    # OpenAI / OpenCLIP defaults
    return {
        'url': url,
        'hf_hub': hf_hub,
        'mean': OPENAI_DATASET_MEAN,
        'std': OPENAI_DATASET_STD,
        'interpolation': 'bicubic',
        'resize_mode': 'shortest',
        **kwargs,
    }


def _slpcfg(url='', hf_hub='', **kwargs):
    # SiGLIP defaults
    return {
        'url': url,
        'hf_hub': hf_hub,
        'mean': INCEPTION_MEAN,
        'std': INCEPTION_STD,
        'interpolation': 'bicubic',
        'resize_mode': 'squash',
        **kwargs,
    }


def _apcfg(url='', hf_hub='', **kwargs):
    # CLIPA defaults
    return {
        'url': url,
        'hf_hub': hf_hub,
        'mean': IMAGENET_MEAN,
        'std': IMAGENET_STD,
        'interpolation': 'bilinear',
        'resize_mode': 'squash',
        **kwargs,
    }


def _mccfg(url='', hf_hub='', **kwargs):
    # MobileCLIP
    return {
        'url': url,
        'hf_hub': hf_hub,
        'mean': (0., 0., 0.),
        'std': (1., 1., 1.),
        'interpolation': 'bilinear',
        'resize_mode': 'shortest',
        **kwargs,
    }



_RN50 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt"),
    cc12m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt"),
)

_RN50_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt"),
    cc12m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt"),
)

_RN101 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt"),
)

_RN101_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt"),
)

_RN50x4 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt"),
)

_RN50x16 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt"),
)

_RN50x64 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt"),
)

_VITB32 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt"),
    laion2b_e16=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth"),
    laion2b_s34b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-laion2B-s34B-b79K/'),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K/'),
    # DataComp-M models
    datacomp_m_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K/'),
    commonpool_m_clip_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.clip-s128M-b4K/'),
    commonpool_m_laion_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.laion-s128M-b4K/'),
    commonpool_m_image_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.image-s128M-b4K/'),
    commonpool_m_text_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.text-s128M-b4K/'),
    commonpool_m_basic_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K/'),
    commonpool_m_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K/'),
    # DataComp-S models
    datacomp_s_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K/'),
    commonpool_s_clip_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.clip-s13M-b4K/'),
    commonpool_s_laion_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.laion-s13M-b4K/'),
    commonpool_s_image_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.image-s13M-b4K/'),
    commonpool_s_text_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K/'),
    commonpool_s_basic_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.basic-s13M-b4K/'),
    commonpool_s_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K/'),
)

_VITB32_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt"),
    metaclip_400m=_pcfg(
        "https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_400m.pt"),
    metaclip_fullcc=_pcfg(
        "https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_fullcc2.5b.pt"),
)

_VITB32_256 = dict(
    datacomp_s34b_b86k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K/'),
)

_VITB16 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e31-00efa78f.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e32-55e67d44.pt"),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-laion2B-s34B-b88K/'),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/'),
    # DataComp-L models
    datacomp_l_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K/'),
    commonpool_l_clip_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K/'),
    commonpool_l_laion_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K/'),
    commonpool_l_image_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K/'),
    commonpool_l_text_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.text-s1B-b8K/'),
    commonpool_l_basic_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K/'),
    commonpool_l_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K/'),
    # DFN
    dfn2b=_pcfg(hf_hub='apple/DFN2B-CLIP-ViT-B-16/')
)

_VITB16_quickgelu = dict(
    metaclip_400m=_pcfg(
        "https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt"),
    metaclip_fullcc=_pcfg(
        "https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_fullcc2.5b.pt"),
)

_VITB16_PLUS_240 = dict(
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e32-699c4b84.pt"),
)

_VITL14 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e31-69988bb6.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e32-3d133497.pt"),
    laion2b_s32b_b82k=_pcfg(
        hf_hub='laion/CLIP-ViT-L-14-laion2B-s32B-b82K/',
        mean=INCEPTION_MEAN, std=INCEPTION_STD),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/'),
    commonpool_xl_clip_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K/'),
    commonpool_xl_laion_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K/'),
    commonpool_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K/'),
)

_VITL14_quickgelu = dict(
    metaclip_400m=_pcfg(
        "https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_400m.pt"),
    metaclip_fullcc=_pcfg(
        "https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt"),
    dfn2b=_pcfg(hf_hub='apple/DFN2B-CLIP-ViT-L-14/'),
)

_VITL14_336 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"),
)

_VITH14 = dict(
    laion2b_s32b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-H-14-laion2B-s32B-b79K/'),
)

_VITH14_quickgelu = dict(
    metaclip_fullcc=_pcfg(
        "https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_fullcc2.5b.pt"),
    dfn5b=_pcfg(
        hf_hub='apple/DFN5B-CLIP-ViT-H-14/',
        interpolation="bicubic",
        resize_mode="squash"
    ),
)

_VITH14_378_quickgelu = dict(
    dfn5b=_pcfg(
        hf_hub='apple/DFN5B-CLIP-ViT-H-14-378/',
        interpolation="bicubic",
        resize_mode="squash"
    ),
)

_VITg14 = dict(
    laion2b_s12b_b42k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s12B-b42K/'),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s34B-b88K/'),
)

_VITbigG14 = dict(
    laion2b_s39b_b160k=_pcfg(hf_hub='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/'),
)

_VITbigG14_quickgelu = dict(
    metaclip_fullcc=_pcfg(url='https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt'),
)

_robertaViTB32 = dict(
    laion2b_s12b_b32k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k/'),
)

_xlmRobertaBaseViTB32 = dict(
    laion5b_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k/'),
)

_xlmRobertaLargeFrozenViTH14 = dict(
    frozen_laion5b_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/'),
)

_convnext_base = dict(
    laion400m_s13b_b51k=_pcfg(hf_hub='laion/CLIP-convnext_base-laion400M-s13B-b51K/'),
)

_convnext_base_w = dict(
    laion2b_s13b_b82k=_pcfg(hf_hub='laion/CLIP-convnext_base_w-laion2B-s13B-b82K/'),
    laion2b_s13b_b82k_augreg=_pcfg(hf_hub='laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg/'),
    laion_aesthetic_s13b_b82k=_pcfg(hf_hub='laion/CLIP-convnext_base_w-laion_aesthetic-s13B-b82K/'),
)

_convnext_base_w_320 = dict(
    laion_aesthetic_s13b_b82k=_pcfg(hf_hub='laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K/'),
    laion_aesthetic_s13b_b82k_augreg=_pcfg(hf_hub='laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg/'),
)

_convnext_large_d = dict(
    laion2b_s26b_b102k_augreg=_pcfg(hf_hub='laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg/'),
)

_convnext_large_d_320 = dict(
    laion2b_s29b_b131k_ft=_pcfg(hf_hub='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft/'),
    laion2b_s29b_b131k_ft_soup=_pcfg(hf_hub='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/'),
)

_convnext_xxlarge = dict(
    laion2b_s34b_b82k_augreg=_pcfg(hf_hub='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg/'),
    laion2b_s34b_b82k_augreg_rewind=_pcfg(hf_hub='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-rewind/'),
    laion2b_s34b_b82k_augreg_soup=_pcfg(hf_hub='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/'),
)

_coca_VITB32 = dict(
    laion2b_s13b_b90k=_pcfg(hf_hub='laion/CoCa-ViT-B-32-laion2B-s13B-b90k/'),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(hf_hub='laion/mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k/')
)

_coca_VITL14 = dict(
    laion2b_s13b_b90k=_pcfg(hf_hub='laion/CoCa-ViT-L-14-laion2B-s13B-b90k/'),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(hf_hub='laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/')
)


_PRETRAINED = {
    "RN50": _RN50,
    "RN50-quickgelu": _RN50_quickgelu,
    "RN101": _RN101,
    "RN101-quickgelu": _RN101_quickgelu,
    "RN50x4": _RN50x4,
    "RN50x16": _RN50x16,
    "RN50x64": _RN50x64,

    "ViT-B-32": _VITB32,
    "ViT-B-32-256": _VITB32_256,
    "ViT-B-32-quickgelu": _VITB32_quickgelu,
    "ViT-B-16": _VITB16,
    "ViT-B-16-quickgelu": _VITB16_quickgelu,
    "ViT-B-16-plus-240": _VITB16_PLUS_240,
    "ViT-L-14": _VITL14,
    "ViT-L-14-quickgelu": _VITL14_quickgelu,
    "ViT-L-14-336": _VITL14_336,
    "ViT-H-14": _VITH14,
    "ViT-H-14-quickgelu": _VITH14_quickgelu,
    "ViT-H-14-378-quickgelu": _VITH14_378_quickgelu,
    "ViT-g-14": _VITg14,
    "ViT-bigG-14": _VITbigG14,
    "ViT-bigG-14-quickgelu": _VITbigG14_quickgelu,

    "roberta-ViT-B-32": _robertaViTB32,
    "xlm-roberta-base-ViT-B-32": _xlmRobertaBaseViTB32,
    "xlm-roberta-large-ViT-H-14": _xlmRobertaLargeFrozenViTH14,

    "convnext_base": _convnext_base,
    "convnext_base_w": _convnext_base_w,
    "convnext_base_w_320": _convnext_base_w_320,
    "convnext_large_d": _convnext_large_d,
    "convnext_large_d_320": _convnext_large_d_320,
    "convnext_xxlarge": _convnext_xxlarge,

    "coca_ViT-B-32": _coca_VITB32,
    "coca_ViT-L-14": _coca_VITL14,

    "EVA01-g-14": dict(
        # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_psz14_s11B.pt
        laion400m_s11b_b41k=_pcfg(hf_hub='timm/eva_giant_patch14_clip_224.laion400m_s11b_b41k/'),
    ),
    "EVA01-g-14-plus": dict(
        # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_plus_psz14_s11B.pt
        merged2b_s11b_b114k=_pcfg(hf_hub='timm/eva_giant_patch14_plus_clip_224.merged2b_s11b_b114k/'),
    ),
    "EVA02-B-16": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_B_psz16_s8B.pt
        merged2b_s8b_b131k=_pcfg(hf_hub='timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k/'),
    ),
    "EVA02-L-14": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_s4B.pt
        merged2b_s4b_b131k=_pcfg(hf_hub='timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k/'),
    ),
    "EVA02-L-14-336": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt
        merged2b_s6b_b61k=_pcfg(hf_hub='timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k/'),
    ),
    "EVA02-E-14": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_s4B.pt
        laion2b_s4b_b115k=_pcfg(hf_hub='timm/eva02_enormous_patch14_clip_224.laion2b_s4b_b115k/'),
    ),
    "EVA02-E-14-plus": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt
        laion2b_s9b_b144k=_pcfg(hf_hub='timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k/'),
    ),

    "ViT-B-16-SigLIP": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP/'),
    ),
    "ViT-B-16-SigLIP-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP-256/'),
    ),
    "ViT-B-16-SigLIP-i18n-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP-i18n-256/'),
    ),
    "ViT-B-16-SigLIP-384": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP-384/'),
    ),
    "ViT-B-16-SigLIP-512": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP-512/'),
    ),
    "ViT-L-16-SigLIP-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-L-16-SigLIP-256/'),
    ),
    "ViT-L-16-SigLIP-384": dict(
        webli=_slpcfg(hf_hub='timm/ViT-L-16-SigLIP-384/'),
    ),
    "ViT-SO400M-14-SigLIP": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-14-SigLIP/'),
    ),
    "ViT-SO400M-16-SigLIP-i18n-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-16-SigLIP-i18n-256/'),
    ),
    "ViT-SO400M-14-SigLIP-378": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-14-SigLIP-384/'),  # NOTE using 384 weights, but diff img_size used
    ),
    "ViT-SO400M-14-SigLIP-384": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-14-SigLIP-384/'),
    ),

    "ViT-L-14-CLIPA": dict(
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B/'),
    ),
    "ViT-L-14-CLIPA-336": dict(
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B/'),
    ),
    "ViT-H-14-CLIPA": dict(
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-H-14-CLIPA-datacomp1B/'),
    ),
    "ViT-H-14-CLIPA-336": dict(
        laion2b=_apcfg(hf_hub='UCSC-VLAA/ViT-H-14-CLIPA-336-laion2B/'),
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-H-14-CLIPA-336-datacomp1B/'),
    ),
    "ViT-bigG-14-CLIPA": dict(
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-bigG-14-CLIPA-datacomp1B/'),
    ),
    "ViT-bigG-14-CLIPA-336": dict(
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-bigG-14-CLIPA-336-datacomp1B/'),
    ),

    "nllb-clip-base": dict(
        v1=_pcfg(hf_hub='visheratin/nllb-clip-base-oc/'),
    ),
    "nllb-clip-large": dict(
        v1=_pcfg(hf_hub='visheratin/nllb-clip-large-oc/'),
    ),

    "nllb-clip-base-siglip": dict(
        v1=_slpcfg(hf_hub='visheratin/nllb-clip-base-siglip/'),
        mrl=_slpcfg(hf_hub='visheratin/nllb-siglip-mrl-base/'),
    ),
    "nllb-clip-large-siglip": dict(
        v1=_slpcfg(hf_hub='visheratin/nllb-clip-large-siglip/'),
        mrl=_slpcfg(hf_hub='visheratin/nllb-siglip-mrl-large/'),
    ),

    "MobileCLIP-S1": dict(
        datacompdr=_mccfg(hf_hub='apple/MobileCLIP-S1-OpenCLIP/')),
    "MobileCLIP-S2": dict(
        datacompdr=_mccfg(hf_hub='apple/MobileCLIP-S2-OpenCLIP/')),
    "MobileCLIP-B": dict(
        datacompdr=_mccfg(hf_hub='apple/MobileCLIP-B-OpenCLIP/'),
        datacompdr_lt=_mccfg(hf_hub='apple/MobileCLIP-B-LT-OpenCLIP/'),
    ),

    "ViTamin-S": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-S/pytorch_model.bin'),
    ),
    "ViTamin-S-LTT": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-S-LTT/pytorch_model.bin'),
    ),
    "ViTamin-B": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-B/pytorch_model.bin'),
    ),
    "ViTamin-B-LTT": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-B-LTT/pytorch_model.bin'),
    ),
    "ViTamin-L": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L-224px/pytorch_model.bin'),
    ),
    "ViTamin-L-256": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L-256px/pytorch_model.bin'),
    ),
    "ViTamin-L-336": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L-336px/pytorch_model.bin'),
    ),
    "ViTamin-L-384": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L-384px/pytorch_model.bin'),
    ),
    "ViTamin-L2": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L2-224px/pytorch_model.bin'),
    ),
    "ViTamin-L2-256": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L2-256px/pytorch_model.bin'),
    ),
    "ViTamin-L2-336": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L2-336px/pytorch_model.bin'),
    ),
    "ViTamin-L2-384": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L2-384px/pytorch_model.bin'),
    ),
    "ViTamin-XL-256": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-XL-256px/pytorch_model.bin'),
    ),
    "ViTamin-XL-336": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-XL-336px/pytorch_model.bin'),
    ),
    "ViTamin-XL-384": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-XL-384px/pytorch_model.bin'),
    ),
}


def _clean_tag(tag: str):
    # normalize pretrained tags
    return tag.lower().replace('-', '_')


def list_pretrained(as_str: bool = False):
    """ returns list of pretrained models
    Returns a tuple (model_name, pretrain_tag) by default or 'name:tag' if as_str == True
    """
    return [':'.join([k, t]) if as_str else (k, t) for k in _PRETRAINED.keys() for t in _PRETRAINED[k].keys()]


def list_pretrained_models_by_tag(tag: str):
    """ return all models having the specified pretrain tag """
    models = []
    tag = _clean_tag(tag)
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)
    return models


def list_pretrained_tags_by_model(model: str):
    """ return all pretrain tags for the specified model architecture """
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags


def is_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return False
    return _clean_tag(tag) in _PRETRAINED[model]


def get_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return {}
    model_pretrained = _PRETRAINED[model]
    return model_pretrained.get(_clean_tag(tag), {})


def get_pretrained_url(model: str, tag: str):
    cfg = get_pretrained_cfg(model, _clean_tag(tag))
    return cfg.get('url', '')


def download_pretrained_from_url(
        url: str,
        cache_dir: Union[str, None] = None,
):
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    elif 'mlfoundations' in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ''

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return _has_hf_hub


def _get_safe_alternatives(filename: str) -> Iterable[str]:
    """Returns potential safetensors alternatives for a given filename.

    Use case:
        When downloading a model from the Huggingface Hub, we first look if a .safetensors file exists and if yes, we use it.
    """
    if filename == HF_WEIGHTS_NAME:
        yield HF_SAFE_WEIGHTS_NAME

    if filename not in (HF_WEIGHTS_NAME,) and filename.endswith(".bin") or filename.endswith(".pth"):
        yield filename[:-4] + ".safetensors"


def download_pretrained_from_hf(
        model_id: str,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
):
    has_hf_hub(True)

    filename = filename or HF_WEIGHTS_NAME

    # Look for .safetensors alternatives and load from it if it exists
    if _has_safetensors:
        for safe_filename in _get_safe_alternatives(filename):
            try:
                cached_file = hf_hub_download(
                    repo_id=model_id, filename=safe_filename, revision=revision, cache_dir=cache_dir)
                return cached_file
            except Exception:
                pass

    try:
        # Attempt to download the file
        cached_file = hf_hub_download(
            repo_id=model_id, filename=filename, revision=revision, cache_dir=cache_dir)
        return cached_file  # Return the path to the downloaded file if successful
    except Exception as e:
        raise FileNotFoundError(f"Failed to download any files for {model_id}. Last error: {e}")


def download_pretrained(
        cfg: Dict,
        force_hf_hub: bool = False,
        cache_dir: Optional[str] = None,
):
    target = ''
    if not cfg:
        return target

    download_url = cfg.get('url', '')
    download_hf_hub = cfg.get('hf_hub', '')
    if download_hf_hub and force_hf_hub:
        # use HF hub even if url exists
        download_url = ''

    if download_url:
        target = download_pretrained_from_url(download_url, cache_dir=cache_dir)
    elif download_hf_hub:
        has_hf_hub(True)
        # we assume the hf_hub entries in pretrained config combine model_id + filename in
        # 'org/model_name/filename.pt' form. To specify just the model id w/o filename and
        # use 'open_clip_pytorch_model.bin' default, there must be a trailing slash 'org/model_name/'.
        model_id, filename = os.path.split(download_hf_hub)
        if filename:
            target = download_pretrained_from_hf(model_id, filename=filename, cache_dir=cache_dir)
        else:
            target = download_pretrained_from_hf(model_id, cache_dir=cache_dir)

    return target
