""" timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    import timm
    from timm.layers import RotAttentionPool2d
    from timm.layers import AttentionPool2d as AbsAttentionPool2d
    from timm.layers import Mlp, to_2tuple
except ImportError:
    timm = None

from .utils import freeze_batch_norm_2d


class TimmModel(nn.Module):
    """ timm model adapter
    """

    def __init__(
            self,
            model_name: str,
            embed_dim: int,
            image_size: Union[int, Tuple[int, int]] = 224,
            pool: str = 'avg',
            proj: str = 'linear',
            proj_bias: bool = False,
            drop: float = 0.,
            drop_path: Optional[float] = None,
            patch_drop: Optional[float] = None,
            pretrained: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please install the latest timm (`pip install timm`) to use timm based models.")
        self.image_size = to_2tuple(image_size)

        # setup kwargs that may not be common across all models
        timm_kwargs = {}
        if drop_path is not None:
            timm_kwargs['drop_path_rate'] = drop_path
        if patch_drop is not None:
            timm_kwargs['patch_drop_rate'] = patch_drop

        custom_pool = pool in ('abs_attn', 'rot_attn')
        if proj:
            assert proj in ("linear", "mlp", "none")
        extra_proj = proj in ("linear", "mlp")
        if not extra_proj and not custom_pool:
            # use network classifier head as projection if no proj specified and no custom pooling used
            # if projection is explicitly set to "none" will be pass through from network trunk
            proj_dim = 0 if proj == 'none' else embed_dim
            self.trunk = timm.create_model(
                model_name,
                num_classes=proj_dim,
                global_pool=pool,
                pretrained=pretrained,
                **timm_kwargs,
            )
            prev_chs = embed_dim
        else:
            self.trunk = timm.create_model(
                model_name,
                pretrained=pretrained,
                **timm_kwargs,
            )
            feat_size = self.trunk.default_cfg.get('pool_size', None)
            feature_ndim = 1 if not feat_size else 2
            if custom_pool:
                assert feature_ndim == 2
                # if attn pooling used, remove both classifier and default pool
                self.trunk.reset_classifier(0, global_pool='')
            else:
                # reset global pool if pool config set, otherwise leave as network default
                reset_kwargs = dict(global_pool=pool) if pool else {}
                self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        # Add custom pooling to head
        if pool == 'abs_attn':
            head_layers['pool'] = AbsAttentionPool2d(prev_chs, feat_size=feat_size, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == 'rot_attn':
            head_layers['pool'] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if proj == 'linear':
            head_layers['drop'] = nn.Dropout(drop)
            head_layers['proj'] = nn.Linear(prev_chs, embed_dim, bias=proj_bias)
        elif proj == 'mlp':
            head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=(drop, 0), bias=(True, proj_bias))

        self.head = nn.Sequential(head_layers)

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_parameters, group_modules
            except ImportError:
                raise RuntimeError(
                    'Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`')
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning('grad checkpointing not supported for this timm image tower, continuing without...')

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            output_fmt: str = 'NCHW',
            output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize_intermediates: Apply norm layer to all intermediates
            intermediates_only: Only return intermediate features
            output_fmt: Shape of intermediate feature outputs
            output_extra_tokens: Return both prefix and spatial intermediate tokens
        Returns:
        """
        extra_args = {}
        if output_extra_tokens:
            extra_args['return_prefix_tokens'] = True
        trunk_output = self.trunk.forward_intermediates(
                x,
                indices=indices,
                intermediates_only=intermediates_only,
                norm=normalize_intermediates,
                stop_early=stop_early,
                output_fmt=output_fmt,
                **extra_args,
            )

        return_dict = {}
        intermediates = trunk_output if intermediates_only else trunk_output[1]
        if output_extra_tokens and intermediates and isinstance(intermediates[0], tuple):
            intermediates_prefix = [xi[1] for xi in intermediates]
            intermediates = [xi[0] for xi in intermediates]
            return_dict['image_intermediates_prefix'] = intermediates_prefix

        return_dict['image_intermediates'] = intermediates
        if intermediates_only:
            return return_dict

        image_features = self.trunk.forward_head(trunk_output[0])  # run through timm pooling / projection
        image_features = self.head(image_features) # run through adapter pooling / projection
        return_dict['image_features'] = image_features
        return return_dict

    def set_input_size(self, image_size: Union[int, Tuple[int, int]]):
        """Set the input image size for the model after initialization.

        This method attempts to call set_input_size on the underlying timm model
        if it supports dynamic input size adjustment.

        Args:
            image_size: New image size as int (square) or tuple (h, w)
        """
        self.image_size = to_2tuple(image_size)

        # Check if the underlying timm model has set_input_size method
        if hasattr(self.trunk, 'set_input_size'):
            self.trunk.set_input_size(image_size)
        else:
            logging.info(f"timm model {self.trunk.__class__.__name__} does not have set_input_size method. Skipping.")

    def forward(self, x):
        x = self.trunk(x)
        x = self.head(x)
        return x
