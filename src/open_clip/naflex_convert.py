from copy import deepcopy
from typing import Any, Dict, Tuple

import torch
from torch import nn


_NAFLEX_TIMM_CONVERT_MODULES = ('eva', 'vision_transformer')
_NAFLEX_NATIVE_TIMM_MODEL_NAME = 'vit_base_patch16_clip_224'


def _to_2tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return value, value


def _is_naflex_timm_model_name(model_name: str) -> bool:
    return model_name.startswith('naflexvit')


def _is_timm_model_in_modules(model_name: str, module_names: Tuple[str, ...]) -> bool:
    try:
        from timm.models import is_model_in_modules
    except ImportError:
        return False
    return is_model_in_modules(model_name, module_names)


def _can_convert_timm_model_to_naflex(model_name: str) -> bool:
    return (
        _is_naflex_timm_model_name(model_name) or
        _is_timm_model_in_modules(model_name, _NAFLEX_TIMM_CONVERT_MODULES)
    )


def _is_standard_native_vit_cfg(vision_cfg: Dict[str, Any]) -> bool:
    if vision_cfg.get('timm_model_name') or isinstance(vision_cfg.get('layers'), (tuple, list)):
        return False
    if vision_cfg.get('attentional_pool', False):
        return False
    if vision_cfg.get('pool_type', 'tok') != 'tok':
        return False
    if vision_cfg.get('no_ln_pre', False) or vision_cfg.get('final_ln_after_pool', False):
        return False
    if vision_cfg.get('output_tokens', False):
        return False
    if vision_cfg.get('pos_embed_type', 'learnable') != 'learnable':
        return False
    unsupported_custom_block = (
        vision_cfg.get('block_type') not in (None, 'default') or
        vision_cfg.get('qk_norm', False) or
        vision_cfg.get('scaled_cosine_attn', False) or
        vision_cfg.get('scale_heads', False) or
        vision_cfg.get('scale_attn_inner', False) or
        vision_cfg.get('scale_attn', False) or
        vision_cfg.get('scale_fc', False)
    )
    if unsupported_custom_block:
        return False
    if vision_cfg.get('act_kwargs') is not None or vision_cfg.get('norm_kwargs') is not None:
        return False
    return True


def _force_naflex_native_vit_vision(vision_cfg: Dict[str, Any], quick_gelu: bool = False) -> None:
    if not _is_standard_native_vit_cfg(vision_cfg):
        raise RuntimeError(
            "NaFlex vision mode can only convert standard native OpenCLIP/OpenAI ViT towers "
            "or compatible timm EVA/ViT towers."
        )

    image_size = vision_cfg.get('image_size', 224)
    patch_size = vision_cfg.get('patch_size', 16)
    width = vision_cfg.get('width', 768)
    head_width = vision_cfg.get('head_width', 64)
    heads = width // head_width
    if heads <= 0 or width % head_width != 0:
        raise RuntimeError(
            f"NaFlex vision mode cannot convert native ViT with width={width} and head_width={head_width}."
        )

    timm_model_kwargs = deepcopy(vision_cfg.get('timm_model_kwargs') or {})
    timm_model_kwargs.update({
        'img_size': image_size,
        'use_naflex': True,
        'patch_size': patch_size,
        'embed_dim': width,
        'depth': vision_cfg.get('layers', 12),
        'num_heads': heads,
        'mlp_ratio': vision_cfg.get('mlp_ratio', 4.0),
        'class_token': True,
        'reg_tokens': 0,
        'global_pool': 'token',
        'pool_include_prefix': False,
        'pos_embed': 'learned',
        'pos_embed_grid_size': tuple(s // p for s, p in zip(
            _to_2tuple(image_size),
            _to_2tuple(patch_size),
        )),
        'pre_norm': True,
        'final_norm': True,
        'fc_norm': False,
        'embed_proj_type': 'linear',
        'qkv_bias': True,
        'proj_bias': True,
        'act_layer': 'quick_gelu' if quick_gelu else None,
    })
    if vision_cfg.get('ls_init_value', None) is not None:
        timm_model_kwargs['init_values'] = vision_cfg['ls_init_value']
    vision_cfg.update({
        'timm_model_name': _NAFLEX_NATIVE_TIMM_MODEL_NAME,
        'timm_model_pretrained': False,
        'timm_pool': 'token',
        'timm_proj': 'linear',
        'timm_proj_bias': False,
        'timm_model_kwargs': timm_model_kwargs,
    })


def _force_naflex_timm_vision(vision_cfg: Dict[str, Any]) -> None:
    timm_model_name = vision_cfg.get('timm_model_name')
    if not timm_model_name:
        raise RuntimeError("NaFlex vision mode requires a compatible timm or native ViT vision tower.")
    if _is_naflex_timm_model_name(timm_model_name):
        return
    if not _can_convert_timm_model_to_naflex(timm_model_name):
        raise RuntimeError(
            f"NaFlex vision mode cannot convert timm model '{timm_model_name}'. "
            "Use a timm EVA/ViT model or an explicit naflexvit model config."
        )
    timm_model_kwargs = deepcopy(vision_cfg.get('timm_model_kwargs') or {})
    timm_model_kwargs['use_naflex'] = True
    if vision_cfg.get('timm_pool') == 'map' and _is_timm_model_in_modules(timm_model_name, ('eva',)):
        timm_model_kwargs.setdefault('pool_include_prefix', True)
    vision_cfg['timm_model_kwargs'] = timm_model_kwargs


def apply_naflex_vision_config(model_cfg: Dict[str, Any]) -> None:
    vision_cfg = model_cfg['vision_cfg']
    if vision_cfg.get('timm_model_name'):
        _force_naflex_timm_vision(vision_cfg)
    else:
        _force_naflex_native_vit_vision(vision_cfg, quick_gelu=model_cfg.get('quick_gelu', False))


def _reshape_native_pos_embed_for_naflex(pos_embed: torch.Tensor) -> torch.Tensor:
    num_patches = pos_embed.shape[0]
    grid_size = int(num_patches ** 0.5)
    if grid_size * grid_size != num_patches:
        raise RuntimeError(
            f"Cannot convert native OpenCLIP ViT positional embedding with {num_patches} patch tokens."
        )
    return pos_embed.reshape(1, grid_size, grid_size, pos_embed.shape[-1])


def _convert_naflex_native_vit_state_dict(
        state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    pos_embed = state_dict.get('visual.positional_embedding')
    cls_embed = state_dict.get('visual.class_embedding')
    if pos_embed is None or cls_embed is None:
        return state_dict

    converted_state_dict = {
        k: v
        for k, v in state_dict.items()
        if not (
            k.startswith('visual.conv1.') or
            k.startswith('visual.ln_pre.') or
            k.startswith('visual.transformer.resblocks.') or
            k.startswith('visual.ln_post.') or
            k in ('visual.class_embedding', 'visual.positional_embedding', 'visual.proj')
        )
    }

    patch_pos_embed = pos_embed[1:]
    converted_state_dict['visual.trunk.embeds.cls_token'] = (cls_embed + pos_embed[0]).reshape(1, 1, -1)
    converted_state_dict['visual.trunk.embeds.pos_embed'] = _reshape_native_pos_embed_for_naflex(patch_pos_embed)
    converted_state_dict['visual.trunk.embeds.proj.weight'] = state_dict['visual.conv1.weight'].permute(
        0, 2, 3, 1).flatten(1)
    converted_state_dict['visual.trunk.norm_pre.weight'] = state_dict['visual.ln_pre.weight']
    converted_state_dict['visual.trunk.norm_pre.bias'] = state_dict['visual.ln_pre.bias']
    converted_state_dict['visual.trunk.norm.weight'] = state_dict['visual.ln_post.weight']
    converted_state_dict['visual.trunk.norm.bias'] = state_dict['visual.ln_post.bias']
    converted_state_dict['visual.head.proj.weight'] = state_dict['visual.proj'].T

    block_prefix = 'visual.transformer.resblocks.'
    block_name_map = {
        'ln_1.': 'norm1.',
        'attn.in_proj_weight': 'attn.qkv.weight',
        'attn.in_proj_bias': 'attn.qkv.bias',
        'attn.out_proj.': 'attn.proj.',
        'ln_2.': 'norm2.',
        'mlp.c_fc.': 'mlp.fc1.',
        'mlp.c_proj.': 'mlp.fc2.',
        'ls_1.gamma': 'ls1.gamma',
        'ls_2.gamma': 'ls2.gamma',
    }
    for key, value in state_dict.items():
        if not key.startswith(block_prefix):
            continue
        suffix = key[len(block_prefix):]
        block_id, _, block_key = suffix.partition('.')
        if not block_key:
            continue
        for old, new in block_name_map.items():
            if block_key.startswith(old):
                converted_state_dict[f'visual.trunk.blocks.{block_id}.{new}{block_key[len(old):]}'] = value
                break

    return converted_state_dict


def _convert_naflex_timm_state_dict(
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    visual = getattr(model, 'visual', None)
    trunk = getattr(visual, 'trunk', None)
    if trunk is None or trunk.__class__.__name__ != 'NaFlexVit':
        return state_dict

    if 'visual.conv1.weight' in state_dict:
        return _convert_naflex_native_vit_state_dict(state_dict)

    try:
        from timm.models.naflexvit import checkpoint_filter_fn
    except ImportError:
        return state_dict

    prefix = 'visual.trunk.'
    trunk_state_dict = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    if not trunk_state_dict:
        return state_dict

    converted_trunk_state_dict = checkpoint_filter_fn(dict(trunk_state_dict), trunk)
    converted_state_dict = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith(prefix)
    }
    converted_state_dict.update({
        prefix + k: v
        for k, v in converted_trunk_state_dict.items()
    })
    return converted_state_dict


def convert_naflex_state_dict(
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return _convert_naflex_timm_state_dict(model, state_dict)
