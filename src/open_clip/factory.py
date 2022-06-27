import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from .model import CLIP, convert_weights_to_fp16
from .openai import load_openai_model
from .pretrained import get_pretrained_url, download_pretrained
from .transform import image_transform

try:
    from coca_pytorch.coca_pytorch import CoCa
    from vit_pytorch import ViT
    from vit_pytorch.extractor import Extractor
except:
    logging.debug("coca and vit from lucidrains are not installed")

try:
    import timm
except ImportError:
    logging.debug("timm is not installed")

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def create_model(
        model_name: str,
        pretrained: str = '',
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        jit: bool = False,
        force_quick_gelu: bool = False,
        pretrained_image: bool = False,
):
    if model_name == "coca":
        # enc = timm.create_model('lambda_resnet26rpt_256', pretrained=True)

        # enc.head = torch.nn.Sequential(
        #     View((-1, 64, 2048)),
        # )
        enc = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048
        )
        enc = Extractor(enc, return_embeddings_only = True)
        # import CoCa and instantiate it
        model = CoCa(
            dim = 512,                     # model dimension
            img_encoder = enc,             # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
            image_dim = 1024,              # image embedding dimension, if not the same as model dimensions
            num_tokens = 49408,            # number of text tokens
            unimodal_depth = 6,            # depth of the unimodal transformer
            multimodal_depth = 6,          # depth of the multimodal transformer
            dim_head = 64,                 # dimension per attention head
            heads = 8,                     # number of attention heads
            caption_loss_weight = 1.,      # weight on the autoregressive caption loss
            contrastive_loss_weight = 1.,  # weight on the contrastive loss between image and text CLS embeddings
        )

        return model
    
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names

    if pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(model_name, device=device, jit=jit)
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()
    else:
        if model_name in _MODEL_CONFIGS:
            logging.info(f'Loading {model_name} model config.')
            model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if pretrained_image:
            if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        model = CLIP(**model_cfg)
        
        if pretrained:
            checkpoint_path = ''
            url = get_pretrained_url(model_name, pretrained)
            if url:
                checkpoint_path = download_pretrained(url)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                model.load_state_dict(load_state_dict(checkpoint_path))
            else:
                logging.warning(f'Pretrained weights ({pretrained}) not found for model {model_name}.')
                raise RuntimeError(f'Pretrained weights ({pretrained}) not found for model {model_name}.')

        model.to(device=device)
        if precision == "fp16":
            assert device.type != 'cpu'
            convert_weights_to_fp16(model)

        if jit:
            model = torch.jit.script(model)

    return model
    
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    if pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(model_name, device=device, jit=jit)
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()
        model = CLIP(**model_cfg)
    else:
        if model_name in _MODEL_CONFIGS:
            logging.info(f'Loading {model_name} model config.')
            model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if pretrained_image:
            if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        model = CLIP(**model_cfg)
        
    if pretrained:
        checkpoint_path = ''
        url = get_pretrained_url(model_name, pretrained)
        if url:
            checkpoint_path = download_pretrained(url)
        elif os.path.exists(pretrained):
            checkpoint_path = pretrained

        if checkpoint_path:
            logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
            model.load_state_dict(load_state_dict(checkpoint_path))
        else:
            logging.warning(f'Pretrained weights ({pretrained}) not found for model {model_name}.')
            raise RuntimeError(f'Pretrained weights ({pretrained}) not found for model {model_name}.')

    model.to(device=device)
    if precision == "fp16":
        assert device.type != 'cpu'
        convert_weights_to_fp16(model)

    if jit:
        model = torch.jit.script(model)

    return model

def create_model_and_transforms(
        model_name: str,
        pretrained: str = '',
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        jit: bool = False,
        force_quick_gelu: bool = False,
        pretrained_image: bool = False,
):
    model = create_model(
    model_name, pretrained, precision, device, jit,
    force_quick_gelu=force_quick_gelu,
    pretrained_image=pretrained_image
    )
    #FIXME hardcoded size
    if model_name == "coca":
        preprocess_train = image_transform(256, is_train=True)
        preprocess_val = image_transform(256, is_train=False)
    else:
        preprocess_train = image_transform(model.visual.image_size, is_train=True)
        preprocess_val = image_transform(model.visual.image_size, is_train=False)
    return model, preprocess_train, preprocess_val

def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()
