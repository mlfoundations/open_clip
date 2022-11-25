import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype
from .openai import load_openai_model
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained, list_pretrained_tags_by_model
from .transform import image_transform
from .tokenizer import HFTokenizer, tokenize
from .eva_vit_model import resize_rel_pos_embed

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


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


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(model_name):
    config = get_model_config(model_name)
    tokenizer = HFTokenizer(config['text_cfg']['hf_tokenizer_name']) if 'hf_tokenizer_name' in config['text_cfg'] else tokenize
    return tokenizer


def load_state_dict(checkpoint_path: str, map_location: str='cpu', model_key: str='model|module|state_dict', is_openai: bool=False, skip_list: list=[]):
    if is_openai:
        model = torch.jit.load(checkpoint_path, map_location="cpu").eval()
        state_dict = model.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        for mk in model_key.split('|'):
            if isinstance(checkpoint, dict) and mk in checkpoint:
                state_dict = checkpoint[mk]
                break
            else:
                state_dict = checkpoint
        if next(iter(state_dict.items()))[0].startswith('module'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    for k in skip_list:
        if k in list(state_dict.keys()):
            logging.info(f"Removing key {k} from pretrained checkpoint")
            del state_dict[k]
    return state_dict


def load_checkpoint(model, checkpoint_path, model_key="model|module|state_dict", strict=True):
    state_dict = load_state_dict(checkpoint_path, model_key=model_key, is_openai=False)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    logging.info(f"incompatible_keys.missing_keys: {incompatible_keys.missing_keys}")
    return incompatible_keys


def load_clip_visual_state_dict(checkpoint_path: str, map_location: str='cpu', is_openai: bool=False, skip_list: list=[]):
    state_dict = load_state_dict(checkpoint_path, map_location=map_location, is_openai=is_openai, skip_list=skip_list)

    for k in list(state_dict.keys()):
        if not k.startswith('visual.'):
            del state_dict[k]
    for k in list(state_dict.keys()):
        if k.startswith('visual.'):
            new_k = k[7:]
            state_dict[new_k] = state_dict[k]
            del state_dict[k]
    return state_dict

def load_clip_text_state_dict(checkpoint_path: str, map_location: str='cpu', is_openai: bool=False, skip_list: list=[]):
    state_dict = load_state_dict(checkpoint_path, map_location=map_location, is_openai=is_openai, skip_list=skip_list)

    for k in list(state_dict.keys()):
        if k.startswith('visual.'):
            del state_dict[k]
    return state_dict

def load_pretrained_checkpoint(
        model,
        visual_checkpoint_path,
        text_checkpoint_path,
        strict=True,
        visual_source="other",
        text_source="open_clip",
        model_key="model|module|state_dict",
        skip_list=[]):
    assert visual_source in ["other", "open_clip", "clip"], print(f"visual_source should be one of [other, open_clip, clip], but get {visual_source}.")
    assert text_source in ["other", "open_clip", "clip"], print(f"text_source should be one of [other, open_clip, clip], but get {text_source}.")

    logging.info(f"num of model state_dict keys: {len(model.state_dict().keys())}")
    visual_incompatible_keys, text_incompatible_keys = None, None
    if visual_checkpoint_path:
        if visual_source == "open_clip":
            visual_state_dict = load_clip_visual_state_dict(visual_checkpoint_path, is_openai=False, skip_list=skip_list)
        elif visual_source == "clip":
            visual_state_dict = load_clip_visual_state_dict(visual_checkpoint_path, is_openai=True, skip_list=skip_list)
        else:
            visual_state_dict = load_state_dict(visual_checkpoint_path, model_key=model_key, is_openai=False, skip_list=skip_list)
    
        # resize_pos_embed for CLIP and open CLIP
        if 'positional_embedding' in visual_state_dict:
            resize_pos_embed(visual_state_dict, model)
        # specified to eva_vit_model
        elif 'pos_embed' in visual_state_dict:
            resize_rel_pos_embed(visual_state_dict, model)
        visual_incompatible_keys = model.visual.load_state_dict(visual_state_dict, strict=strict)
        logging.info(f"num of loaded visual_state_dict keys: {len(visual_state_dict.keys())}")
        logging.info(f"visual_incompatible_keys.missing_keys: {visual_incompatible_keys.missing_keys}")

    if text_checkpoint_path:
        if text_source == "open_clip":
            text_state_dict = load_clip_text_state_dict(text_checkpoint_path, is_openai=False, skip_list=skip_list)
        elif text_source == "clip":
            text_state_dict = load_clip_text_state_dict(text_checkpoint_path, is_openai=True, skip_list=skip_list)
        else:
            text_state_dict = load_state_dict(visual_checkpoint_path, model_key=model_key, is_openai=False, skip_list=skip_list)

        text_incompatible_keys = model.text.load_state_dict(text_state_dict, strict=strict)
        
        logging.info(f"num of loaded text_state_dict keys: {len(text_state_dict.keys())}")
        logging.info(f"text_incompatible_keys.missing_keys: {text_incompatible_keys.missing_keys}")

    return visual_incompatible_keys, text_incompatible_keys


def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        pretrained_image: str = '',
        pretrained_text: str = '',
        pretrained_hf: bool = True,
        pretrained_visual_source: str = 'other',
        pretrained_text_source: str = 'clip',
        cache_dir: Optional[str] = None,
        skip_list: list = []
):
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            jit=jit,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f'Loaded {model_name} model config.')
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        cast_dtype = get_cast_dtype(precision)
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text or ('hf_model_name' in model_cfg.get('text_cfg', {}))

        if custom_text:
            if 'hf_model_name' in model_cfg.get('text_cfg', {}):
                model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf
            model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        pretrained_cfg = {}
        if pretrained:
            checkpoint_path = ''
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    f'Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                logging.warning(error_str)
                raise RuntimeError(error_str)
        else:
            visual_checkpoint_path = ''
            text_checkpoint_path = ''
            if pretrained_image:
                if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
                    # pretrained weight loading for timm models set via vision_cfg
                    model_cfg['vision_cfg']['timm_model_pretrained'] = True
                elif os.path.exists(pretrained_image):
                    visual_checkpoint_path = pretrained_image
                else:
                    logging.warning(f'Pretrained weights ({visual_checkpoint_path}) not found for model {model_name}.visual.')
                    raise RuntimeError(f'Pretrained weights ({visual_checkpoint_path}) not found for model {model_name}.visual.')

            if pretrained_text:
                if os.path.exists(pretrained_text):
                    text_checkpoint_path = pretrained_text
                else:
                    logging.warning(f'Pretrained weights ({text_checkpoint_path}) not found for model {model_name}.text.')
                    raise RuntimeError(f'Pretrained weights ({text_checkpoint_path}) not found for model {model_name}.text.')
            
            if visual_checkpoint_path:
                logging.info(f'Loading pretrained {model_name}.visual weights ({visual_checkpoint_path}).')
            if text_checkpoint_path:
                logging.info(f'Loading pretrained {model_name}.text weights ({text_checkpoint_path}).')

            if visual_checkpoint_path or text_checkpoint_path:
                load_pretrained_checkpoint(
                    model,
                    visual_checkpoint_path,
                    text_checkpoint_path,
                    strict=False,
                    visual_source=pretrained_visual_source,
                    text_source=pretrained_text_source,
                    skip_list=skip_list
                )

        model.to(device=device)
        if precision in ("fp16", "bf16"):
            convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)

        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
        model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

        if jit:
            model = torch.jit.script(model)

    return model


def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        pretrained_image: str = '',
        pretrained_text: str = '',
        pretrained_hf: bool = True,
        pretrained_visual_source: str = 'other',
        pretrained_text_source: str = 'clip',
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
        skip_list: list = []
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        pretrained_image=pretrained_image,
        pretrained_text=pretrained_text,
        pretrained_hf=pretrained_hf,
        pretrained_visual_source=pretrained_visual_source,
        pretrained_text_source=pretrained_text_source,
        cache_dir=cache_dir,
        skip_list=skip_list
    )

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess_train = image_transform(
        model.visual.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std
    )

    return model, preprocess_train, preprocess_val


def create_model_from_pretrained(
        model_name: str,
        pretrained: str,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        return_transform: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
):
    if not is_pretrained_cfg(model_name, pretrained) and not os.path.exists(pretrained):
        raise RuntimeError(
            f'{pretrained} is not a valid pretrained cfg or checkpoint for {model_name}.'
            f' Use open_clip.list_pretrained() to find one.')

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        cache_dir=cache_dir,
    )

    if not return_transform:
        return model

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std
    )

    return model, preprocess