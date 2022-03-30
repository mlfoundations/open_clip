import json
import logging
import os
from pathlib import Path

import torch

from .openai_clip import load_openai
from .model import CLIP, convert_weights_to_fp16
from .pretrained import get_pretrained_url, download_pretrained
from .transform import image_transform


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def create_model_and_transforms(
        model_name: str,
        pretrained: str,
        precision: str,
        device: torch.device,
        force_quick_gelu: bool = False,
):
    pretrained = pretrained.lower()
    if pretrained == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model, preprocess_train, preprocess_val = load_openai(model_name, device=device, jit=False)
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()
    else:
        model_config_file = Path(__file__).parent / f"model_configs/{model_name}.json"
        logging.info(f'Loading model config from {model_config_file}.')
        assert os.path.exists(model_config_file)
        with open(model_config_file, 'r') as f:
            model_cfg = json.load(f)
        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True
        
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

        preprocess_train = image_transform(model.visual.image_size, is_train=True)
        preprocess_val = image_transform(model.visual.image_size, is_train=False)

        model.to(device=device)
        if precision == "fp16":
            convert_weights_to_fp16(model)

    return model, preprocess_train, preprocess_val
