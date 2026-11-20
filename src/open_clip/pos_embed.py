# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _resample_rows(table: torch.Tensor, idx: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """Select rows from a linearly resampled ``[L, C]`` table without per-sample tables or host sync.

    ``idx`` is ``[B, N]`` and ``length`` is ``[B, 1]``. Matches ``F.interpolate`` with
    ``mode='linear', align_corners=False``.
    """
    size = table.shape[0]
    src = ((idx.float() + 0.5) * (size / length.float()) - 0.5).clamp(0, size - 1)
    i0 = src.floor().long()
    i1 = (i0 + 1).clamp(max=size - 1)
    weight = (src - i0)[..., None].to(table.dtype)
    return torch.lerp(table[i0], table[i1], weight)


def factorized_pos_embed(
        pos_embed_y: Optional[torch.Tensor],
        pos_embed_x: Optional[torch.Tensor],
        patch_coord: torch.Tensor,
        patch_valid: Optional[torch.Tensor] = None,
        ar_preserving: bool = False,
) -> torch.Tensor:
    """Learned factorized positions for NaFlex patches (as in timm NaFlexVit's factorized pos embed).

    Returns ``[B, N, C]`` at ``(y, x)`` coordinates (or ``(freq, time)`` for audio). Infer each
    sample's grid from valid coordinates, then resize the axis tables to its height / width.
    Either axis table may be ``None`` (e.g. ``pos_embed_y=None`` for 1-D full-height audio strips).
    With ``ar_preserving``, resize both to ``max(height, width)`` and crop the shorter axis.
    Padding coordinates do not affect the grid size; callers still mask padded tokens downstream.
    """
    coord = patch_coord.long()
    if patch_valid is not None:
        coord = coord.masked_fill(~patch_valid.bool()[..., None], 0)
    shape = coord.amax(1) + 1
    if ar_preserving:
        len_y = len_x = shape.amax(1, keepdim=True)
    else:
        len_y, len_x = shape[:, :1], shape[:, 1:]
    out: Optional[torch.Tensor] = None
    if pos_embed_y is not None:
        out = _resample_rows(pos_embed_y, coord[..., 0], len_y)
    if pos_embed_x is not None:
        emb_x = _resample_rows(pos_embed_x, coord[..., 1], len_x)
        out = emb_x if out is None else out + emb_x
    if out is None:
        raise ValueError('at least one of pos_embed_y or pos_embed_x must be provided')
    return out


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
