from collections import OrderedDict
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .utils import to_2tuple, feature_take_indices
from .pos_embed import get_2d_sincos_pos_embed


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(
            self,
            prob: float = 0.5,
            exclude_first_token: bool = True
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = x[:, :1]

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            kdim: Optional[int] = None,
            vdim: Optional[int] = None,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            scaled_cosine: bool = False,
            scale_heads: bool = False,
            inner_norm: bool = False,
            logit_scale_max: float = math.log(1. / 0.01),
            norm_layer: Type[nn.Module] = LayerNormFp32,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            use_sdpa: bool = True,
    ):
        super().__init__()
        assert not (scaled_cosine and qk_norm), "Cannot activate both scaled cosine and QK normalization"
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max
        self.use_sdpa = use_sdpa

        kdim = kdim if kdim is not None else dim
        vdim = vdim if vdim is not None else dim

        if kdim == dim and vdim == dim:
            # Same-dim: combined in_proj_weight (3*dim, dim) — matches nn.MHA and existing checkpoints
            self.in_proj_weight = nn.Parameter(torch.empty((dim * 3, dim)))
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None
        else:
            # Different-dim: separate projections — key names match nn.MHA for checkpoint compat
            self.in_proj_weight = None
            self.q_proj_weight = nn.Parameter(torch.empty((dim, dim)))
            self.k_proj_weight = nn.Parameter(torch.empty((dim, kdim)))
            self.v_proj_weight = nn.Parameter(torch.empty((dim, vdim)))

        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.empty(dim * 3))
        else:
            self.in_proj_bias = None

        # QK normalization (with LN) from https://arxiv.org/abs/2106.04560 and related to other QK Norm ideas
        if qk_norm:
            self.ln_q = norm_layer(self.head_dim)
            self.ln_k = norm_layer(self.head_dim)
        else:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()

        # Scaled cosine attention (from Swin Transformer V2, https://arxiv.org/abs/2111.09883)
        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None

        self.attn_drop = nn.Dropout(attn_drop)

        # Per-head attention logit scaling (from NormFormer, https://arxiv.org/abs/2110.09456)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None

        # Normalization of attention logits, before final projection.
        # Origin likely Sub-LN in (Foundation Transformers, https://arxiv.org/abs/2210.06423)
        if inner_norm:
            self.ln_inner = norm_layer(dim)
        else:
            self.ln_inner = nn.Identity()

        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

        self._reset_parameters()

    def _reset_parameters(self):
        # Match nn.MultiheadAttention init
        if self.in_proj_weight is not None:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
            self,
            x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        N, L, C = x.shape

        if k_x is None and v_x is None:
            # Self-attention fast path: fused QKV projection
            if self.in_proj_weight is not None:
                q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
            else:
                bias_q, bias_k, bias_v = (
                    self.in_proj_bias.chunk(3) if self.in_proj_bias is not None
                    else (None, None, None)
                )
                q = F.linear(x, self.q_proj_weight, bias_q)
                k = F.linear(x, self.k_proj_weight, bias_k)
                v = F.linear(x, self.v_proj_weight, bias_v)
        else:
            # Cross-attention path: separate Q/K/V projections
            if k_x is None:
                k_x = x
            if v_x is None:
                v_x = x

            bias_q, bias_k, bias_v = (
                self.in_proj_bias.chunk(3) if self.in_proj_bias is not None
                else (None, None, None)
            )
            if self.in_proj_weight is not None:
                # Same-dim case: split combined weight into 3 chunks
                w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
            else:
                w_q, w_k, w_v = self.q_proj_weight, self.k_proj_weight, self.v_proj_weight

            q = F.linear(x, w_q, bias_q)
            k = F.linear(k_x, w_k, bias_k)
            v = F.linear(v_x, w_v, bias_v)

        q = q.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if attn_mask is not None and attn_mask.ndim == 3:
            # reshape (N*num_heads, L, L) -> (N, num_heads, L, L)
            attn_mask = attn_mask.reshape(N, self.num_heads, L, L)

        if self.logit_scale is not None:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-1, -2)
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn * logit_scale
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn.masked_fill_(~attn_mask, float("-inf"))
                else:
                    attn = attn + attn_mask.to(dtype=attn.dtype)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        else:
            q = self.ln_q(q)
            k = self.ln_k(k)
            if self.use_sdpa:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                    scale=self.scale,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-1, -2)
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn.masked_fill_(~attn_mask, float("-inf"))
                    else:
                        attn = attn + attn_mask.to(dtype=attn.dtype)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

        # N, num_heads, L, head_dim
        if self.head_scale is not None:
            x = x * self.head_scale
        x = x.transpose(1, 2).reshape(N, -1, C)
        x = self.ln_inner(x)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = Attention(d_model, num_heads=n_head, kdim=context_dim, vdim=context_dim, qkv_bias=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), k_x=x, v_x=x)
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(d_model, num_heads=n_head, qkv_bias=True)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_weight_dtype(self) -> torch.dtype:
        if hasattr(self.mlp.c_fc, 'int8_original_dtype'):
            return self.mlp.c_fc.int8_original_dtype
        return self.mlp.c_fc.weight.dtype

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            qk_norm: bool = False,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn_inner: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            qk_norm=qk_norm,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            inner_norm=scale_attn_inner,
            norm_layer=norm_layer,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),  # from NormFormer / Foundation Transformers
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_weight_dtype(self) -> torch.dtype:
        if hasattr(self.mlp.c_fc, 'int8_original_dtype'):
            return self.mlp.c_fc.int8_original_dtype
        return self.mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomTransformer(nn.Module):
    """ A custom transformer that can use different block types. """
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            block_types: Union[str, List[str]] = 'CustomResidualAttentionBlock',
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        if isinstance(block_types, str):
            block_types = [block_types] * layers
        assert len(block_types) == layers

        def _create_block(bt: str):
            if bt == 'CustomResidualAttentionBlock':
                return CustomResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
            else:
                assert False

        self.resblocks = nn.ModuleList([
            _create_block(bt)
            for bt in block_types
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].get_weight_dtype()

    def forward_intermediates(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
    ):
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)

        intermediates = []
        blocks = self.resblocks if not stop_early else self.resblocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing:
                x = checkpoint(blk, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = blk(x, attn_mask=attn_mask)

            if i in take_indices:
                intermediates.append(x)

        return x, intermediates

    def prune_intermediate_layers(self, indices: Union[int, List[int]] = 1):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[:max_index + 1]  # truncate blocks
        return take_indices

    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        if impl == 'composable' and enable:
            from torch.distributed._composable import checkpoint as composable_checkpoint
            for r in self.resblocks:
                composable_checkpoint(r)
        else:
            self.grad_checkpointing = enable

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing:
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)

        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            block_type: Optional[str] = None,
            qk_norm: bool = False,
            scaled_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn_inner: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        # Auto-select custom block if any custom features are enabled
        if block_type is None:
            if any([qk_norm, scaled_cosine_attn, scale_heads, scale_attn_inner, scale_attn, scale_fc]):
                block_type = 'custom'
            else:
                block_type = 'default'

        if block_type == 'custom':
            self.resblocks = nn.ModuleList([
                CustomResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    scale_cosine_attn=scaled_cosine_attn,
                    scale_heads=scale_heads,
                    scale_attn_inner=scale_attn_inner,
                    scale_attn=scale_attn,
                    scale_fc=scale_fc,
                )
                for _ in range(layers)
            ])
        else:
            self.resblocks = nn.ModuleList([
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(layers)
            ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].get_weight_dtype()

    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        if impl == 'composable' and enable:
            from torch.distributed._composable import checkpoint as composable_checkpoint
            for r in self.resblocks:
                composable_checkpoint(r)
        else:
            self.grad_checkpointing = enable

    def forward_intermediates(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
    ):
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)

        intermediates = []
        blocks = self.resblocks if not stop_early else self.resblocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing:
                x = checkpoint(blk, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = blk(x, attn_mask=attn_mask)

            if i in take_indices:
                intermediates.append(x)

        return x, intermediates

    def prune_intermediate_layers(self, indices: Union[int, List[int]] = 1):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[:max_index + 1]  # truncate blocks
        return take_indices

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing:
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)

        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class VisionTransformer(nn.Module):

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            block_type: Optional[str] = None,
            qk_norm: bool = False,
            scaled_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn_inner: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            block_type=block_type,
            qk_norm=qk_norm,
            scaled_cosine_attn=scaled_cosine_attn,
            scale_heads=scale_heads,
            scale_attn_inner=scale_attn_inner,
            scale_attn=scale_attn,
            scale_fc=scale_fc,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def layer_groups(self, pooler_in_head: bool = True):
        """Ordered, complete partition into named ``(name, [members])`` groups, input -> output, shared by ``lock``
        and layer-wise LR decay. Groups: ``embeddings`` (patch conv + class/pos embeds + ln_pre), ``layer.{i}``
        (transformer blocks; the final block is grouped with ``ln_post``), and ``proj`` (the projection head).
        ``pooler_in_head`` is accepted for a common signature with the text towers but has no effect here.
        """
        embed = [
            m for m in (
                self.conv1,
                getattr(self, "class_embedding", None),
                getattr(self, "positional_embedding", None),
                self.ln_pre,
            )
            if m is not None
        ]
        groups = [("embeddings", embed)]
        resblocks = self.transformer.resblocks
        n = len(resblocks)
        for i, block in enumerate(resblocks):
            members = [block]
            if i == n - 1:
                members.append(self.ln_post)
            groups.append((f"layer.{i}", members))
        if self.proj is not None:
            groups.append(("proj", [self.proj]))
        return groups

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        # Freeze the tower top-down via the shared ``layer_groups`` enumeration: ``unlocked_groups`` counts the top
        # groups (the proj head first) left trainable, so unlocked_groups=0 freezes everything. Each group is set
        # explicitly (frozen -> False, unlocked -> True) so repeated calls with different counts are idempotent
        # (e.g. progressive unfreezing / multi-stage training).
        groups = self.layer_groups()
        n_freeze = len(groups) if not unlocked_groups else len(groups) - unlocked_groups
        for i, (_, members) in enumerate(groups):
            _set_group_requires_grad(members, requires_grad=(i >= n_freeze))

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        self.transformer.set_grad_checkpointing(enable, impl=impl)

    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding', 'class_embedding'}
        return no_wd

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def _embeds(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # patch dropout (if active)
        x = self.patch_dropout(x)

        # apply norm before transformer
        x = self.ln_pre(x)
        return x

    def _pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        return pooled, tokens

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
            intermediates_only: Only return intermediate features
            normalize_intermediates: Apply final norm layer to all intermediates
            output_fmt: Shape of intermediate feature outputs
            output_extra_tokens: Return both extra prefix class tokens
        Returns:

        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'

        # forward pass
        B, _, height, width = x.shape
        x = self._embeds(x)
        x, intermediates = self.transformer.forward_intermediates(
            x,
            indices=indices,
            stop_early=stop_early,
        )

        # process intermediates
        if normalize_intermediates:
            # apply final norm to all intermediates
            intermediates = [self.ln_post(xi) for xi in intermediates]
        num_prefix_tokens = 1  # one class token that's always there (as of now)
        if num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, num_prefix_tokens:] for y in intermediates]
        else:
            prefix_tokens = None
        if reshape:
            # reshape to BCHW output format
            H, W = height // self.patch_size[0], width // self.patch_size[1]
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        output = {'image_intermediates': intermediates}
        if prefix_tokens is not None and output_extra_tokens:
            output['image_intermediates_prefix'] = prefix_tokens

        if intermediates_only:
            return output

        pooled, _ = self._pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        output['image_features'] = pooled

        return output

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices = self.transformer.prune_intermediate_layers(indices)
        if prune_norm:
            self.ln_post = nn.Identity()
        if prune_head:
            self.proj = None
        return take_indices

    def forward(self, x: torch.Tensor):
        x = self._embeds(x)
        x = self.transformer(x)
        pooled, tokens = self._pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled


def text_global_pool(
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        pool_type: str = 'argmax',
        eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    if pool_type == 'first':
        pooled = x[:, 0]
    elif pool_type == 'last':
        pooled = x[:, -1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
    elif pool_type == 'eos':
        # take features from tokenizer specific eos
        assert text is not None
        assert eos_token_id is not None
        idx = (text == eos_token_id).int().argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), idx]
    else:
        pooled = x

    return pooled


class ModernRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.to(dtype) * self.weight.to(dtype))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.w12(x).chunk(2, dim=-1)
        return self.w3(x * F.silu(gate))


class ReLUSquared(nn.Module):
    """Squared-ReLU activation (Primer); simpler than SwiGLU and competitive at small model scale."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class RotaryEmbedding1D(nn.Module):
    def __init__(self, dim: int, temperature: float = 10000.0):
        super().__init__()
        if dim % 2:
            raise ValueError(f"RoPE head dim must be even, got {dim}.")
        inv_freq = 1.0 / (temperature ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return cos/sin concatenated as one ``[seq_len, head_dim]`` table (cos | sin halves).

        Computed once per forward in the parent tower and threaded through every block (timm-style cat layout),
        rather than recomputed per layer.
        """
        pos = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq.to(device=device))
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype=dtype)


def _apply_rope_1d(x: torch.Tensor, rope_embed: torch.Tensor) -> torch.Tensor:
    cos, sin = rope_embed.chunk(2, dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class ModernTextAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int,
            qk_norm: bool = False,
            norm_layer: Optional[Callable[[int], nn.Module]] = None,
            gated: bool = False,
            value_residual: bool = False,
            vr_first: bool = False,
            bias: bool = True,
            gate_bias: bool = True,
    ):
        super().__init__()
        if dim % heads:
            raise ValueError(f"text width {dim} must be divisible by heads {heads}.")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        # qk-norm over head_dim follows the model's norm type (norm_layer), default RMSNorm.
        qk_norm_layer = norm_layer if norm_layer is not None else ModernRMSNorm
        self.q_norm = qk_norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = qk_norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # Gate bias is decoupled from `bias` so the mostly-open init (init_parameters) can survive bias-free attn.
        self.gate = nn.Linear(dim, dim, bias=gate_bias) if gated else None
        # Value residual (ResFormer): mix this layer's V with layer-0's V via a learned scalar,
        # v = lerp(v_first, v, vr_lambda); init 0.5 = equal mix, lambda -> 1 recovers plain attention.
        # Layer 0 only *produces* v_first (vr_first), so it gets no lambda (an unused param breaks DDP).
        self.value_residual = value_residual
        self.vr_lambda = nn.Parameter(torch.full((1,), 0.5)) if (value_residual and not vr_first) else None
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(
            self,
            x: torch.Tensor,
            rope_embed: Optional[torch.Tensor] = None,
            key_bias: Optional[torch.Tensor] = None,
            is_causal: bool = False,
            v_first: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, l, c = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        v_out = None
        if self.value_residual:
            # Layer 0 (v_first None) contributes its raw V; later layers mix and propagate v_first unchanged.
            # vr_lambda stays fp32 under static low-precision conversion (convert_weights_to_lp only converts
            # module weights), so cast at use like reg_tokens / norm weights.
            v_out = v if v_first is None else v_first
            if self.vr_lambda is not None and v_first is not None:
                v = torch.lerp(v_first, v, self.vr_lambda.to(v.dtype))
        q, k = self.q_norm(q), self.k_norm(k)
        if rope_embed is not None:
            q = _apply_rope_1d(q, rope_embed)
            k = _apply_rope_1d(k, rope_embed)
        # Causal mode passes is_causal (no mask tensor); bidirectional passes a [B, 1, 1, L] key-pad bias. SDPA
        # forbids both at once, and the two are mutually exclusive here (see ModernTextTransformer._attn_inputs).
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_bias, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(b, l, c)
        if self.gate is not None:
            out = out * self.gate(x).sigmoid()
        return self.proj(out), v_out


class ModernTextBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int,
            mlp_ratio: float,
            norm_layer: Callable[[int], nn.Module],
            qk_norm: bool = False,
            attn_gated: bool = False,
            mlp_type: str = "swiglu",
            norm_placement: str = "pre",
            ls_init_value: Optional[float] = None,
            value_residual: bool = False,
            vr_first: bool = False,
            attn_bias: bool = True,
            gate_bias: bool = True,
            mlp_bias: bool = True,
    ):
        super().__init__()
        if norm_placement not in ("pre", "sandwich"):
            raise ValueError(f"unknown modern text norm_placement={norm_placement!r}")
        sandwich = norm_placement == "sandwich"
        self.norm1 = norm_layer(dim)
        self.attn = ModernTextAttention(
            dim,
            heads,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            gated=attn_gated,
            value_residual=value_residual,
            vr_first=vr_first,
            bias=attn_bias,
            gate_bias=gate_bias,
        )
        # Sandwich placement (Gemma-2 style): an extra norm on each sublayer *output*, before LayerScale and
        # the residual add; widens the stable-LR range at the cost of two extra norms per block.
        self.norm1_post = norm_layer(dim) if sandwich else nn.Identity()
        self.ls1 = LayerScale(dim, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        if mlp_type == "swiglu":
            self.mlp = SwiGLU(dim, hidden, bias=mlp_bias)
        elif mlp_type in ("mlp", "relu2"):
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(dim, hidden, bias=mlp_bias)),
                ("act", nn.GELU() if mlp_type == "mlp" else ReLUSquared()),
                ("c_proj", nn.Linear(hidden, dim, bias=mlp_bias)),
            ]))
        else:
            raise ValueError(f"unknown modern text mlp_type={mlp_type!r}")
        self.norm2_post = norm_layer(dim) if sandwich else nn.Identity()
        self.ls2 = LayerScale(dim, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_weight_dtype(self) -> torch.dtype:
        return self.attn.qkv.weight.dtype

    def forward(
            self,
            x: torch.Tensor,
            rope_embed: Optional[torch.Tensor] = None,
            key_bias: Optional[torch.Tensor] = None,
            is_causal: bool = False,
            v_first: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, v_first = self.attn(
            self.norm1(x), rope_embed=rope_embed, key_bias=key_bias, is_causal=is_causal, v_first=v_first,
        )
        x = x + self.ls1(self.norm1_post(attn_out))
        x = x + self.ls2(self.norm2_post(self.mlp(self.norm2(x))))
        return x, v_first


class ModernTextPool(nn.Module):
    def __init__(
            self,
            width: int,
            pool_type: str = "eos",
            heads: int = 8,
            norm_layer: Optional[Callable[[int], nn.Module]] = None,
            bias: bool = True,
            qk_norm: bool = False,
    ):
        super().__init__()
        self.pool_type = pool_type
        self.heads = heads
        if pool_type == "map":
            if width % heads:
                raise ValueError(f"text width {width} must be divisible by MAP heads {heads}.")
            self.head_dim = width // heads
            self.query = nn.Parameter(torch.empty(1, 1, width))
            self.q = nn.Linear(width, width, bias=bias)
            self.kv = nn.Linear(width, width * 2, bias=bias)
            # qk-norm over head_dim (bf16 logit stability, as in the blocks): model norm type, default RMSNorm.
            # No kv pre-norm -- ln_final already normalises the pool input.
            qk_norm_layer = norm_layer if norm_layer is not None else ModernRMSNorm
            self.q_norm = qk_norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = qk_norm_layer(self.head_dim) if qk_norm else nn.Identity()
            nn.init.normal_(self.query, std=width ** -0.5)
        elif pool_type in ("eos", "mean"):
            self.query = None
        else:
            # NOTE: no 'last' here on purpose -- in open_clip 'last' means the last *physical* position
            # (SigLIP / CLIPA fixed-length contract), which is padding for this variable-length tower. The
            # masked equivalent (last valid token) is the no-eos-in-row fallback of the 'eos' branch.
            raise ValueError(f"unknown modern text pool_type={pool_type!r}")

    def forward(
            self,
            x: torch.Tensor,
            text: torch.Tensor,
            valid: torch.Tensor,
            eos_id: Optional[int],
    ) -> torch.Tensor:
        if self.pool_type == "mean":
            weights = valid.to(x.dtype)
            return (x * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp(min=1)
        if self.pool_type == "eos":
            # Pool at the first eos_id occurrence; rows without one (eos lost to truncation) fall back to the
            # last valid (non-pad) token, which is where the eos would sit under the right-padded contract.
            eos = text == eos_id
            last_valid = valid.long().sum(dim=1).sub(1).clamp(min=0)
            idx = torch.where(eos.any(dim=1), eos.int().argmax(dim=1), last_valid)
            return x[torch.arange(x.shape[0], device=x.device), idx]

        b, l, c = x.shape
        # The learned query latent stays fp32 under static low-precision conversion; cast at use.
        q = self.q(self.query.to(x.dtype).expand(b, -1, -1)).reshape(b, 1, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(x).reshape(b, l, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn_bias = x.new_zeros((b, 1, 1, l))
        attn_bias.masked_fill_(~valid[:, None, None, :], torch.finfo(x.dtype).min)
        pooled = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)
        return pooled.transpose(1, 2).reshape(b, c)


class ModernTextTransformer(nn.Module):
    """Dedicated variable-length text tower: RoPE, SwiGLU/ReLU^2, masked pooling, and optional gated attention,
    qk-norm, embedding pre-norm, sandwich norms, register tokens, and layer-0 value residuals."""

    def __init__(
            self,
            cfg: Any,
            output_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.variable_text = bool(getattr(cfg, "variable_text", False))
        # Classic 'argmax' pooling (CLIP picks the EOT as the max token id) maps to 'eos' here: same intent,
        # keyed on the configured eos token id. eos_id has no config default and is validated below (and against
        # the tokenizer in get_tokenizer), so a stale/wrong id fails fast instead of pooling silently wrong.
        pool_type = "eos" if cfg.pool_type == "argmax" else cfg.pool_type
        if pool_type == "eos" and cfg.eos_id is None:
            raise ValueError(
                "modern text 'eos' (or 'argmax') pooling requires text_cfg.eos_id "
                "(must match the tokenizer eos/eot token id)."
            )

        if cfg.attention_mode not in ("causal", "bidirectional"):
            raise ValueError(f"unknown modern text attention_mode={cfg.attention_mode!r}")
        if cfg.pos_embed not in ("rope", "none", ""):
            raise ValueError(f"unknown modern text pos_embed={cfg.pos_embed!r}")
        if cfg.width % cfg.heads != 0:
            raise ValueError(f"modern text width ({cfg.width}) must be divisible by heads ({cfg.heads}).")
        if cfg.pos_embed == "rope" and (cfg.width // cfg.heads) % 2 != 0:
            raise ValueError(
                f"modern text RoPE head dim must be even, got width / heads = {cfg.width // cfg.heads}."
            )

        # Public text-tower API used by CustomTextCLIP/CLAP, tokenizer checks, and training setup. Keep the rest
        # in cfg or private fields so config knobs do not become accidental model attributes.
        self.context_length = cfg.context_length
        self.num_pos = cfg.context_length
        self.vocab_size = cfg.vocab_size
        self.width = cfg.width
        self.layers = cfg.layers
        self.output_dim = output_dim
        self.pad_id = cfg.pad_id
        self.eos_id = cfg.eos_id

        # norm_type is tri-state on the shared cfg (None = arch default); the modern tower defaults to RMSNorm.
        norm_type = cfg.norm_type if cfg.norm_type is not None else "rmsnorm"
        if norm_type == "rmsnorm":
            norm_layer = lambda dim: ModernRMSNorm(dim, eps=cfg.norm_eps)
        elif norm_type == "layernorm":
            norm_layer = lambda dim: LayerNorm(dim, eps=cfg.norm_eps)
        else:
            raise ValueError(f"unknown modern text norm_type={norm_type!r}")

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width, padding_idx=cfg.pad_id)
        # Learned register tokens: prepended to the sequence (always valid) and excluded from pooling. In
        # bidirectional mode they are true scratch positions (read and write the whole sequence). In causal
        # mode the prefix can only be *attended to*, not read from text, so they act as learned attention
        # sinks -- still useful for soaking up sink/outlier mass, but a weaker mechanism than bidirectional
        # registers, and largely redundant with a BOS token.
        self.num_reg = max(0, int(getattr(cfg, "reg_tokens", 0) or 0))
        self.reg_tokens = nn.Parameter(torch.empty(1, self.num_reg, cfg.width)) if self.num_reg else None
        # One rope module on the tower; the cos/sin table is computed once per forward and threaded through the
        # blocks (ModernTextAttention applies it), instead of being recomputed inside every layer.
        self.rope = RotaryEmbedding1D(
            cfg.width // cfg.heads,
            temperature=cfg.rope_temperature,
        ) if cfg.pos_embed == "rope" else None
        # Embedding norm before block 0 (timm ViT pre_norm / norm_pre; ModernBERT embeddings.norm), applied
        # after register concat.
        self.norm_pre = norm_layer(cfg.width) if getattr(cfg, "pre_norm", False) else nn.Identity()
        # attention_bias/mlp_bias are tri-state on the shared cfg (None = arch default). This is the modern tower,
        # so None resolves off; bool() collapses None/False -> False, True -> True. attention_bias covers
        # qkv/proj/gate + MAP q/kv, mlp_bias the MLP.
        attention_bias = bool(getattr(cfg, "attention_bias", None))
        mlp_bias = bool(getattr(cfg, "mlp_bias", None))
        # Gate-bias override: None inherits attention_bias; True/False force on/off so the mostly-open gate init
        # stays available with attention_bias off (see init_parameters).
        gate_bias_cfg = getattr(cfg, "gate_bias", None)
        gate_bias = attention_bias if gate_bias_cfg is None else bool(gate_bias_cfg)
        self.blocks = nn.ModuleList([
            ModernTextBlock(
                cfg.width,
                cfg.heads,
                mlp_ratio=cfg.mlp_ratio,
                norm_layer=norm_layer,
                qk_norm=cfg.qk_norm,
                attn_gated=cfg.attn_gated,
                mlp_type=cfg.mlp_type,
                norm_placement=getattr(cfg, "norm_placement", "pre"),
                ls_init_value=cfg.ls_init_value,
                value_residual=getattr(cfg, "value_residual", False),
                vr_first=(i == 0),
                attn_bias=attention_bias,
                gate_bias=gate_bias,
                mlp_bias=mlp_bias,
            )
            for i in range(cfg.layers)
        ])
        self.ln_final = norm_layer(cfg.width)
        self.pool = ModernTextPool(
            cfg.width,
            pool_type=pool_type,
            heads=cfg.heads,
            norm_layer=norm_layer,
            bias=attention_bias,
            qk_norm=cfg.qk_norm,
        )
        self.text_projection = (
            None
            if cfg.proj_type == "none" or not output_dim
            else nn.Linear(cfg.width, output_dim, bias=cfg.proj_bias)
        )
        self.grad_checkpointing = False
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.pad_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.pad_id].zero_()
        if self.reg_tokens is not None:
            # timm cls/reg token convention: near-zero so registers start as blank slates that attention
            # cannot initially key onto, growing only as needed.
            nn.init.normal_(self.reg_tokens, std=1e-6)

        # Block init scheme:
        # - 'pre' norm (default): GPT-2 / CLIP convention -- inputs at `width**-0.5`, residual out-projs
        #   (`proj`, `w3` / `c_proj`) scaled by `(2*layers)**-0.5` to keep residual variance flat with depth.
        # - 'sandwich' norm: flat N(0, 0.02) (OLMo2-style); the post-norms undo depth scaling anyway.
        # - `zero_init_residual`: out-projs zeroed -> identity blocks at init; supersedes the depth factor
        #   (overlaps with `ls_init_value` -- prefer one or the other).
        # Norm weights stay 1.0 (set in constructors); biases zeroed except the attention gate.
        sandwich = getattr(self.cfg, "norm_placement", "pre") == "sandwich"
        zero_residual = bool(getattr(self.cfg, "zero_init_residual", False))
        attn_std = 0.02 if sandwich else self.width ** -0.5
        fc_std = 0.02 if sandwich else (2 * self.width) ** -0.5
        proj_std = 0.02 if sandwich else attn_std * ((2 * self.layers) ** -0.5)
        # SwiGLU correction ('pre' scheme only): the gate product `u * silu(v)` carries ~2.4x less variance at
        # init than the GELU hidden `fc_std` assumes; 1.22x on `w12` (MC-calibrated, width-independent since
        # pre-act var is exactly 1/2 under this scheme) restores parity.
        swiglu_fc_std = fc_std if sandwich else fc_std * 1.22

        def init_residual_out(weight):
            if zero_residual:
                nn.init.zeros_(weight)
            else:
                nn.init.normal_(weight, std=proj_std)

        for block in self.blocks:
            attn = block.attn
            nn.init.normal_(attn.qkv.weight, std=attn_std)
            if attn.qkv.bias is not None:
                nn.init.zeros_(attn.qkv.bias)
            init_residual_out(attn.proj.weight)
            if attn.proj.bias is not None:
                nn.init.zeros_(attn.proj.bias)
            if attn.gate is not None:
                # Mostly-open gate at init: bias 1 -> sigmoid(1) ~= 0.73, so gating starts near-transparent
                # (a half-open 0.5 gate halves attention output magnitude, fighting the residual init scheme).
                # Needs a gate bias to exist: set gate_bias=True to keep this when attention_bias is off, else the
                # bias-free gate falls back to sigmoid(~0) ~= 0.5.
                nn.init.normal_(attn.gate.weight, std=attn_std)
                if attn.gate.bias is not None:
                    nn.init.ones_(attn.gate.bias)
            if attn.vr_lambda is not None:
                nn.init.constant_(attn.vr_lambda, 0.5)  # equal layer-0 / current-layer value mix at init
            mlp = block.mlp
            if isinstance(mlp, SwiGLU):
                nn.init.normal_(mlp.w12.weight, std=swiglu_fc_std)
                if mlp.w12.bias is not None:
                    nn.init.zeros_(mlp.w12.bias)
                init_residual_out(mlp.w3.weight)
                if mlp.w3.bias is not None:
                    nn.init.zeros_(mlp.w3.bias)
            else:  # nn.Sequential MLP (c_fc, act, c_proj)
                nn.init.normal_(mlp.c_fc.weight, std=fc_std)
                if mlp.c_fc.bias is not None:
                    nn.init.zeros_(mlp.c_fc.bias)
                init_residual_out(mlp.c_proj.weight)
                if mlp.c_proj.bias is not None:
                    nn.init.zeros_(mlp.c_proj.bias)

        # MAP attentive-pool projections (the learned ``query`` is already initialized in ModernTextPool.__init__).
        # The pool/head init is scheme-independent: the block-init choice (flat/sandwich, zero-residual) only
        # governs the residual stream, not the readout.
        if getattr(self.pool, "query", None) is not None:
            nn.init.normal_(self.pool.q.weight, std=self.width ** -0.5)
            if self.pool.q.bias is not None:
                nn.init.zeros_(self.pool.q.bias)
            nn.init.normal_(self.pool.kv.weight, std=self.width ** -0.5)
            if self.pool.kv.bias is not None:
                nn.init.zeros_(self.pool.kv.bias)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection.weight, std=self.width ** -0.5)
            if self.text_projection.bias is not None:
                nn.init.zeros_(self.text_projection.bias)

    def _valid_mask(self, text: torch.Tensor) -> torch.Tensor:
        if self.pad_id is None:
            return torch.ones_like(text, dtype=torch.bool)
        valid = text != self.pad_id
        # Guarantee at least one valid position per row so degenerate all-pad rows do not yield NaNs at pooling.
        # Done branchlessly (no data-dependent ``if``) so the tower stays torch.compile(fullgraph=True)-friendly:
        # for rows that already have a valid token this is a no-op (the OR'd term is all-False).
        empty = ~valid.any(dim=1, keepdim=True)
        first = torch.zeros_like(valid)
        first[:, 0] = True
        return valid | (empty & first)

    def _attn_inputs(
            self,
            text: torch.Tensor,
            dtype: torch.dtype,
            num_prefix: int = 0,
    ) -> Tuple[bool, Optional[torch.Tensor], torch.Tensor]:
        """Resolve SDPA attention inputs for this batch as ``(is_causal, key_bias, valid)``.

        The branch is on the static attention mode (no data-dependent control flow, so torch.compile
        sees one graph per mode):

        - ``"causal"``: ``is_causal=True`` with **no** mask tensor. Captions are right-padded (the collators fill
          the tail with ``pad_id``), so under the causal constraint real tokens never attend to pad positions and
          pad-position outputs are discarded at pooling -- masking pad keys is a no-op for the pooled output, and
          dropping the ``[B, 1, L, L]`` mask removes the L^2 materialization (the dominant memory + compile cost).
        - ``"bidirectional"``: a ``[B, 1, 1, L]`` additive key-pad bias (broadcast over the query axis); real
          tokens here *can* see pads, so the mask is required.

        ``num_prefix`` always-valid positions (register tokens) are prepended to the key axis of the bias;
        the returned ``valid`` covers the text positions only (it drives pooling, which excludes registers).
        """
        b, l = text.shape
        valid = self._valid_mask(text)
        if self.cfg.attention_mode == "causal":
            return True, None, valid
        key_valid = valid
        if num_prefix:
            key_valid = torch.cat([valid.new_ones(b, num_prefix), valid], dim=1)
        key_bias = torch.zeros((b, 1, 1, l + num_prefix), device=text.device, dtype=dtype)
        key_bias.masked_fill_(~key_valid[:, None, None, :], torch.finfo(dtype).min)
        return False, key_bias, valid

    def get_cast_dtype(self) -> torch.dtype:
        return self.blocks[0].get_weight_dtype() if self.blocks else self.token_embedding.weight.dtype

    def set_grad_checkpointing(self, enable: bool = True, impl: str = "inline"):
        if impl == "composable" and enable:
            from torch.distributed._composable import checkpoint as composable_checkpoint
            for block in self.blocks:
                composable_checkpoint(block)
        else:
            self.grad_checkpointing = enable

    def layer_groups(self, pooler_in_head: bool = True):
        embed = [self.token_embedding]
        if self.reg_tokens is not None:
            embed.append(self.reg_tokens)
        if not isinstance(self.norm_pre, nn.Identity):
            embed.append(self.norm_pre)
        groups = [("embeddings", embed)]
        n = len(self.blocks)
        for i, block in enumerate(self.blocks):
            members = [block]
            if i == n - 1:
                members.append(self.ln_final)
                if not pooler_in_head:
                    members.append(self.pool)
            groups.append((f"layer.{i}", members))
        head = []
        if pooler_in_head:
            head.append(self.pool)
        if self.text_projection is not None:
            head.append(self.text_projection)
        if head:
            groups.append(("proj", head))
        return groups

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True, pooler_in_head: bool = True):
        assert freeze_layer_norm, 'Unfreezing normalization layers is not supported.'
        _lock_layer_groups(self.layer_groups(pooler_in_head), unlocked_layers)

    def no_weight_decay(self):
        # Learned token-like params follow the positional_embedding / cls_emb convention: excluded from decay
        # (registers and the MAP pool's query latent); the 1-D rule already covers norms, biases, and scalars.
        no_wd = set()
        if self.reg_tokens is not None:
            no_wd.add("reg_tokens")
        if getattr(self.pool, "query", None) is not None:
            no_wd.add("pool.query")
        return no_wd

    def _encode_tokens(
            self,
            text: torch.Tensor,
            take_indices: Optional[List[int]] = None,
            stop_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Run the block stack, returning ``(x, valid, intermediates)`` with ``x`` pre-``ln_final``.

        ``x`` (and collected intermediates) include the ``num_reg`` register prefix when configured; ``valid``
        covers the text positions only. ``take_indices`` collects raw block outputs for
        ``forward_intermediates``; ``stop_index`` truncates the stack (early exit) when only intermediates are
        needed.
        """
        cast_dtype = self.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        if self.reg_tokens is not None:
            x = torch.cat([self.reg_tokens.to(x.dtype).expand(x.shape[0], -1, -1), x], dim=1)
        x = self.norm_pre(x)
        is_causal, key_bias, valid = self._attn_inputs(text, x.dtype, num_prefix=self.num_reg)
        rope_embed = self.rope(x.shape[1], x.device, x.dtype) if self.rope is not None else None
        blocks = self.blocks if stop_index is None else self.blocks[:stop_index + 1]
        intermediates = []
        v_first = None
        for i, block in enumerate(blocks):
            if self.grad_checkpointing:
                x, v_first = checkpoint(block, x, rope_embed, key_bias, is_causal, v_first, use_reentrant=False)
            else:
                x, v_first = block(
                    x, rope_embed=rope_embed, key_bias=key_bias, is_causal=is_causal, v_first=v_first,
                )
            if take_indices is not None and i in take_indices:
                intermediates.append(x)
        return x, valid, intermediates

    def forward_intermediates(
            self,
            text: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            output_fmt: str = 'NLC',
            output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        assert output_fmt == 'NLC', 'ModernTextTransformer only supports NLC text intermediates.'
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        x, valid, intermediates = self._encode_tokens(
            text,
            take_indices=take_indices,
            stop_index=max_index if stop_early else None,
        )
        if normalize_intermediates:
            intermediates = [self.ln_final(t) for t in intermediates]
        # Split the register prefix out of the intermediates: text positions under the usual key, registers
        # (the only "extra" tokens this tower has) under the extra key when requested.
        extra = []
        if self.num_reg:
            extra = [t[:, :self.num_reg] for t in intermediates]
            intermediates = [t[:, self.num_reg:] for t in intermediates]

        output = {"text_intermediates": intermediates}
        if output_extra_tokens:
            output["text_intermediates_extra"] = extra
        if intermediates_only:
            return output

        x = self.ln_final(x)
        pooled = self.pool(x[:, self.num_reg:] if self.num_reg else x, text=text, valid=valid, eos_id=self.eos_id)
        if self.text_projection is not None:
            pooled = self.text_projection(pooled)
        output["text_features"] = pooled
        return output

    def forward(self, text: torch.Tensor):
        x, valid, _ = self._encode_tokens(text)
        x = self.ln_final(x)
        tokens = x[:, self.num_reg:] if self.num_reg else x
        pooled = self.pool(tokens, text=text, valid=valid, eos_id=self.eos_id)
        if self.text_projection is not None:
            pooled = self.text_projection(pooled)
        if self.cfg.output_tokens:
            # NOTE: tokens cover the text positions (registers stripped) but pad positions are NOT masked --
            # consumers of per-token features must mask via `text != pad_id`, like the pooling here does.
            return pooled, tokens
        return pooled


class TextTransformer(nn.Module):

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            output_dim: Optional[int] = 512,
            embed_cls: bool = False,
            no_causal_mask: bool = False,
            use_pad_mask: bool = False,
            correct_cls_mask: bool = False,
            pad_id: int = 0,
            eos_id: int = 2,
            pool_type: str = 'argmax',
            proj_type: str = 'linear',
            proj_bias: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            output_tokens: bool = False,
            block_type: Optional[str] = None,
            qk_norm: bool = False,
            scaled_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn_inner: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'eos', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.pool_type = pool_type
        self.use_pad_mask = use_pad_mask and no_causal_mask  # only use in bi‑dir mode
        self.correct_cls_mask = correct_cls_mask  # use the correct cls mask for CoCa (original is wrong)

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            block_type=block_type,
            qk_norm=qk_norm,
            scaled_cosine_attn=scaled_cosine_attn,
            scale_heads=scale_heads,
            scale_attn_inner=scale_attn_inner,
            scale_attn=scale_attn,
            scale_fc=scale_fc,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None  # bi‑directional
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)

        if proj_type == 'none' or not output_dim:
            self.text_projection = None
        else:
            if proj_bias:
                self.text_projection = nn.Linear(width, output_dim)
            else:
                self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        self.transformer.set_grad_checkpointing(enable, impl=impl)

    def layer_groups(self, pooler_in_head: bool = True):
        """Ordered, complete partition into named ``(name, [members])`` groups, shared by ``lock`` and layer-wise
        LR decay. See :func:`_text_layer_groups`. ``pooler_in_head`` is accepted for a common signature with the HF
        text tower but has no effect here (the native tower has no readout pooler).
        """
        return _text_layer_groups(self, pooler_in_head)

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True, pooler_in_head: bool = True):
        """
        Lock the text transformer layers, optionally leaving some layers unlocked.

        Args:
            unlocked_layers: Number of layers to leave unlocked (from the end).
            freeze_layer_norm: LayerNorm freeze (only for API compatibility, not functional)
            pooler_in_head: pooler placement policy (no effect for the native tower; kept for a common signature).
        """
        assert freeze_layer_norm, 'Unfreezing LayerNorm is not supported. LayerNorm treated like other weights.'
        lock_text_tower(self, unlocked_layers, pooler_in_head)

    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding'}
        if self.cls_emb is not None:
            no_wd.add('cls_emb')
        return no_wd

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _build_additive_mask(
        self,
        text: torch.Tensor,  # [B, L] – original text ids without CLS yet
        seq_len: int,  # L (+1 if CLS added)
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns an additive (-inf) mask of shape [B*heads, seq_len, seq_len] that
        simultaneously masks padding tokens and (optionally) the CLS token.
        """
        valid = text != self.pad_id  # [B, L] (True = keep)

        if self.cls_emb is not None:
            cls_valid = valid.new_ones(valid.size(0), 1) # [B, 1]
            # cls mask pos at end if correct or front for incorrect legacy mode in existing CoCa weights
            valid = torch.cat([valid, cls_valid] if self.correct_cls_mask else [cls_valid, valid], 1)

        # broadcast over query dimension
        key_mask = valid.unsqueeze(1).expand(-1, seq_len, -1)  # [B, Q, K]
        additive = torch.zeros_like(key_mask, dtype=dtype)
        additive.masked_fill_(~key_mask, float("-inf"))
        additive = additive.repeat_interleave(self.heads, 0)  # [B*H, Q, K]
        return additive

    def _embeds(self, text) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cast_dtype = self.transformer.get_cast_dtype()
        B, seq_len = text.shape

        x = self.token_embedding(text).to(cast_dtype)

        # Optional class token (always appended ala CoCa)
        if self.cls_emb is not None:
            x = torch.cat([x, _expand_token(self.cls_emb, x.size(0))], 1)
            seq_len += 1

        attn_mask = self.attn_mask  # Base causal mask (if any)
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        # Class + padding additive mask
        if self.use_pad_mask or self.cls_emb is not None:
            add_mask  = self._build_additive_mask(text, seq_len, x.dtype)
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0) + add_mask
            else:
                attn_mask = add_mask

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        return x, attn_mask

    def forward_intermediates(
            self,
            text: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            output_fmt: str = 'NLC',
            output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            text: Input text ids
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize_intermediates: Apply norm layer to all intermediates
            intermediates_only: Only return intermediate features
            output_fmt: Shape of intermediate feature outputs
            output_extra_tokens: Return both prefix and intermediate tokens
        Returns:

        """
        assert output_fmt in ('NLC',), 'Output format must be NLC.'
        # forward pass
        x, attn_mask = self._embeds(text)
        x, intermediates = self.transformer.forward_intermediates(
            x,
            attn_mask=attn_mask,
            indices=indices,
            stop_early=stop_early,
        )

        # process intermediates
        if normalize_intermediates:
            # apply final norm to all intermediates
            intermediates = [self.ln_final(xi) for xi in intermediates]

        output = {}

        if self.cls_emb is not None:
            seq_intermediates = [xi[:, :-1] for xi in intermediates]  # separate concat'd class token from sequence
            if output_extra_tokens:
                # return suffix class tokens separately
                cls_intermediates = [xi[:, -1:] for xi in intermediates]
                output['text_intermediates_suffix'] = cls_intermediates
            intermediates = seq_intermediates
        output['text_intermediates'] = intermediates

        if intermediates_only:
            return output

        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled = text_global_pool(x, text, pool_type=self.pool_type, eos_token_id=getattr(self, "eos_id", None))

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        output['text_features'] = pooled

        return output

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices = self.transformer.prune_intermediate_layers(indices)
        if prune_norm:
            self.ln_final = nn.Identity()
        if prune_head:
            self.text_projection = None
        return take_indices

    def forward(self, text):
        x, attn_mask = self._embeds(text)

        x = self.transformer(x, attn_mask=attn_mask)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
            tokens = x[:, :-1]
        else:
            x = self.ln_final(x)
            pooled = text_global_pool(x, text, pool_type=self.pool_type, eos_token_id=getattr(self, "eos_id", None))
            tokens = x

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            output_dim: int = 512,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward_intermediates(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
    ):
        assert False, "Not currently implemented for MultimodalTransformer w/ xattn"

    def forward(self, image_embs, text_embs):
        # TODO(kv-cache): Accept past_key_values (list of (K, V) per layer)
        # and cache_position. When past_key_values is not None, only process
        # new positions for self-attention (concatenate past K/V).
        # Cross-attention K/V from image_embs are constant per generation and
        # can be cached once.
        seq_len = text_embs.shape[1]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing:
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(
                    resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len], use_reentrant=False)
                text_embs = checkpoint(
                    cross_attn, text_embs, image_embs, image_embs, None, use_reentrant=False)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        out = self.ln_final(text_embs)
        if self.text_projection is not None:
            out = out @ self.text_projection

        return out

    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        if impl == 'composable' and enable:
            from torch.distributed._composable import checkpoint as composable_checkpoint
            for resblock in self.resblocks:
                composable_checkpoint(resblock)
            for cross_attn in self.cross_attn:
                composable_checkpoint(cross_attn)
        else:
            self.grad_checkpointing = enable


def _text_layer_groups(module: nn.Module, pooler_in_head: bool = True):
    """Ordered, complete partition of a native (CLIP) text tower into named ``(name, [members])`` groups.

    Reads the duck-typed text attributes (``token_embedding``/``positional_embedding``/``cls_emb``,
    ``transformer.resblocks``, ``ln_final``, ``text_projection``), so it works both for a ``TextTransformer`` and
    for a standard CLIP that unpacks those attributes onto itself. Groups, input -> output: ``embeddings``,
    ``layer.{i}`` (the final block is grouped with ``ln_final``), and ``proj`` (the projection head). The native
    tower has no readout pooler, so ``pooler_in_head`` is accepted for a common signature but has no effect.
    """
    embed = [
        m for m in (
            getattr(module, "token_embedding", None),
            getattr(module, "positional_embedding", None),
            getattr(module, "cls_emb", None),
        )
        if m is not None
    ]
    groups = []
    if embed:
        groups.append(("embeddings", embed))
    transformer = getattr(module, "transformer", None)
    resblocks = transformer.resblocks if (transformer is not None and hasattr(transformer, "resblocks")) else []
    ln_final = getattr(module, "ln_final", None)
    n = len(resblocks)
    for i, block in enumerate(resblocks):
        layer_members = [block]
        if (i == n - 1) and (ln_final is not None):
            layer_members.append(ln_final)
        groups.append((f"layer.{i}", layer_members))
    text_projection = getattr(module, "text_projection", None)
    if text_projection is not None:
        groups.append(("proj", [text_projection]))
    return groups


def _set_group_requires_grad(members, requires_grad: bool):
    """Set ``requires_grad`` on a group's members (a mix of ``nn.Module`` and ``nn.Parameter``)."""
    for m in members:
        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad


def _lock_layer_groups(groups, unlocked_layers: int = 0):
    """Freeze layer groups bottom-up, leaving the top ``unlocked_layers`` groups trainable.

    Sets every group explicitly (frozen -> False, unlocked -> True) so repeated calls with different counts are
    idempotent (e.g. progressive unfreezing / multi-stage training). Shared by ``lock_text_tower`` and tower
    ``lock`` methods so the freeze policy cannot drift between towers.
    """
    n_freeze = len(groups) if not unlocked_layers else len(groups) - unlocked_layers
    for i, (_, members) in enumerate(groups):
        _set_group_requires_grad(members, requires_grad=(i >= n_freeze))


def lock_text_tower(
        model: nn.Module,
        unlocked_layers: int = 0,
        pooler_in_head: bool = True,
    ):
    """Freeze a native (CLIP) text tower top-down, leaving the top ``unlocked_layers`` groups trainable.

    Shares the ``_text_layer_groups`` enumeration with layer-wise LR decay, so freezing and decay agree on the
    layer partition. ``unlocked_layers`` counts the top groups (the projection head first) left trainable, so
    ``unlocked_layers=0`` freezes the whole tower. This is offset by one from layer-wise LR decay, which keeps
    the head at lr_scale 1.0 (depth 0).

    Works with both CustomTextCLIP (text components under ``model.text``) and standard CLIP (unpacked attributes).

    Args:
        model: the CLIP model or a ``TextTransformer``.
        unlocked_layers: number of groups (from the output side, head first) to leave trainable; 0 freezes all.
        pooler_in_head: pooler placement policy (no effect for the native tower; kept for a common signature).
    """
    text_module = model.text if hasattr(model, "text") else model
    _lock_layer_groups(_text_layer_groups(text_module, pooler_in_head), unlocked_layers)
