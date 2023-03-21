import hashlib
import os
import urllib
import warnings
from functools import partial
from typing import Dict, Union

from tqdm import tqdm

from .version import __version__

try:
    from huggingface_hub import hf_hub_download
    hf_hub_download = partial(hf_hub_download, library_name="open_clip", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False


def _pcfg(url='', hf_hub='', mean=None, std=None):
    return dict(
        url=url,
        hf_hub=hf_hub,
        mean=mean,
        std=std,
    )


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
    laion2b_s34b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-laion2B-s34B-b79K/')
)

_VITB32_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt"),
)

_VITB16 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e31-00efa78f.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e32-55e67d44.pt"),
    # laion400m_32k=_pcfg(
    #     url="",
    #     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    # laion400m_64k=_pcfg(
    #     url="",
    #     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-laion2B-s34B-b88K/'),
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
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
)

_VITL14_336 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"),
)

_VITH14 = dict(
    laion2b_s32b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-H-14-laion2B-s32B-b79K/'),
)

_VITg14 = dict(
    laion2b_s12b_b42k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s12B-b42K/'),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s34B-b88K/'),
)

_VITbigG14 = dict(
    laion2b_s39b_b160k=_pcfg(hf_hub='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/'),
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
    "ViT-B-32-quickgelu": _VITB32_quickgelu,
    "ViT-B-16": _VITB16,
    "ViT-B-16-plus-240": _VITB16_PLUS_240,
    "ViT-L-14": _VITL14,
    "ViT-L-14-336": _VITL14_336,
    "ViT-H-14": _VITH14,
    "ViT-g-14": _VITg14,
    "ViT-bigG-14": _VITbigG14,
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


def download_pretrained_from_hf(
        model_id: str,
        filename: str = 'open_clip_pytorch_model.bin',
        revision=None,
        cache_dir: Union[str, None] = None,
):
    has_hf_hub(True)
    cached_file = hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir)
    return cached_file


def download_pretrained(
        cfg: Dict,
        force_hf_hub: bool = False,
        cache_dir: Union[str, None] = None,
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
