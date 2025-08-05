from collections import OrderedDict
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

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
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

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
            qkv_bias: bool = True,
            qk_norm: bool = False,
            scaled_cosine: bool = False,
            scale_heads: bool = False,
            inner_norm: bool = False,
            logit_scale_max: float = math.log(1. / 0.01),
            norm_layer: Type[nn.Module] = LayerNormFp32,
            attn_drop: float = 0.,
            proj_drop: float = 0.
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
        self.use_fsdpa = hasattr(nn.functional, 'scaled_dot_product_attention')

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
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

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        N, L, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(N, L, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(N, L, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(N, L, self.num_heads, -1).transpose(1, 2)

        if attn_mask is not None:
            if attn_mask.ndim == 3:
                # this module works with (L, L), or (N, num_heads, L, L) masks
                attn_mask = attn_mask.reshape(N, self.num_heads, L, L)
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            else:
                attn_mask = attn_mask.to(dtype=q.dtype)

        if self.logit_scale is not None:
            attn = torch.bmm(
                F.normalize(q, dim=-1),
                F.normalize(k, dim=-1).transpose(-1, -2)
            )
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn * logit_scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            q = self.ln_q(q)
            k = self.ln_k(k)
            if self.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, v)

        # N, num_heads, L, head_dim
        if self.head_scale is not None:
            x = x * self.head_scale
        x = x.transpose(1, 2).reshape(N, L, C)
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
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
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
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
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
        return self.attn(
            q_x, k_x, v_x,
            need_weights=False,
            attn_mask=attn_mask
        )[0]

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
            batch_first: bool = True,
    ):
        super().__init__()
        assert batch_first, 'batch_first must be True for CustomResidualAttentionBlock'

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
            batch_first: bool = True,
            block_types: Union[str, List[str]] = 'CustomResidualAttentionBlock',
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first  # run transformer stack in batch first (N, L, D)
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
                    batch_first=batch_first,
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

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()  # NLD -> LND

        intermediates = []
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.resblocks
        else:
            blocks = self.resblocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = blk(x, attn_mask=attn_mask)

            if i in take_indices:
                intermediates.append(x.transpose(0, 1) if not self.batch_first else x)

        if not self.batch_first:
            x = x.transpose(0, 1)  # LND -> NLD

        return x, intermediates

    def prune_intermediate_layers(self, indices: Union[int, List[int]] = 1):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[:max_index + 1]  # truncate blocks
        return take_indices

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND
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
            batch_first: bool = True,
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
        self.batch_first = batch_first
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
                    batch_first=batch_first,
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
                    batch_first=batch_first,
                )
                for _ in range(layers)
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

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND

        intermediates = []
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.resblocks
        else:
            blocks = self.resblocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = blk(x, attn_mask=attn_mask)

            if i in take_indices:
                intermediates.append(x.transpose(0, 1) if not self.batch_first else x)

        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD

        return x, intermediates

    def prune_intermediate_layers(self, indices: Union[int, List[int]] = 1):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[:max_index + 1]  # truncate blocks
        return take_indices

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

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

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

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

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
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


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

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

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        """
        Lock the text transformer layers, optionally leaving some layers unlocked.

        Args:
            unlocked_layers: Number of layers to leave unlocked (from the end).
            freeze_layer_norm: LayerNorm freeze (only for API compatibility, not functional)
        """
        assert freeze_layer_norm, 'Unfreezing LayerNorm is not supported. LayerNorm treated like other weights.'
        lock_text_tower(self, unlocked_layers)

    @torch.jit.ignore
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

        # Class + padding additive mask
        if self.use_pad_mask or self.cls_emb is not None:
            add_mask  = self._build_additive_mask(text, seq_len, x.dtype)
            if attn_mask is not None:
                # Slice the causal mask to match current sequence length
                attn_mask = attn_mask[:seq_len, :seq_len].unsqueeze(0) + add_mask
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
            output_fmt: str = 'NCHW',
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
            batch_first: bool = True,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=batch_first,
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
                batch_first=batch_first,
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
        seq_len = text_embs.shape[1]
        if not self.batch_first:
            image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
            text_embs = text_embs.permute(1, 0, 2)  # NLD -> LND

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(
                    resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len], use_reentrant=False)
                text_embs = checkpoint(
                    cross_attn, text_embs, image_embs, image_embs, None, use_reentrant=False)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        if not self.batch_first:
            text_embs = text_embs.permute(1, 0, 2)  # LND -> NLD

        out = self.ln_final(text_embs)
        if self.text_projection is not None:
            out = out @ self.text_projection

        return out

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable


def lock_text_tower(
    model: nn.Module,
    unlocked_layers: int = 0,
):
    """
    Lock text tower layers for CLIP models.

    Works with both model architectures:
    - CustomTextCLIP where text components are in self.text
    - Standard CLIP where text components are unpacked as attributes

    Args:
        model: The CLIP model or TextTransformer module
        unlocked_layers: Number of layers to leave unlocked (from the end)
    """
    # Determine where to look for text components
    if hasattr(model, 'text'):
        # CustomTextCLIP or already a TextTransformer with nested structure
        text_module = model.text
    else:
        # Standard CLIP or direct TextTransformer
        text_module = model

    # Collect text components
    text_params = {}
    text_params['token_embedding'] = getattr(text_module, 'token_embedding', None)
    text_params['positional_embedding'] = getattr(text_module, 'positional_embedding', None)
    text_params['cls_emb'] = getattr(text_module, 'cls_emb', None)
    text_params['transformer'] = getattr(text_module, 'transformer', None)
    text_params['ln_final'] = getattr(text_module, 'ln_final', None)
    text_params['text_projection'] = getattr(text_module, 'text_projection', None)

    # Filter out None values
    text_params = {k: v for k, v in text_params.items() if v is not None}

    # Freeze all text parameters first
    for module in text_params.values():
        if isinstance(module, nn.Parameter):
            module.requires_grad = False
        elif isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = False

    if unlocked_layers == 0:
        return

    # Check if we have transformer blocks to work with
    transformer = text_params['transformer']
    if not transformer or not hasattr(transformer, 'resblocks'):
        return

    total_layers = len(transformer.resblocks)
    if total_layers == 0:
        return

    # Build groups for selective unlocking
    groups = []

    # Group 1: Embeddings
    embedding_group = []
    for key in ['token_embedding', 'positional_embedding', 'cls_emb']:
        if key in text_params:
            embedding_group.append(text_params[key])
    if embedding_group:
        groups.append(embedding_group)

    # Group 2-N: Individual transformer blocks (except last)
    if total_layers > 1:
        for block in transformer.resblocks[:-1]:
            groups.append([block])

    # Combine last transformer block + final ln as the penultimate group
    last_block = [transformer.resblocks[-1]]
    if 'ln_final' in text_params:
        last_block.append(text_params['ln_final'])
    groups.append(last_block)

    # The final group is the projection only
    if 'text_projection' in text_params:
        groups.append([text_params['text_projection']])

    # Helper function to unlock parameters
    def _unlock(module):
        if isinstance(module, Sequence):
            for m in module:
                _unlock(m)
        elif isinstance(module, nn.Parameter):
            module.requires_grad = True
        elif isinstance(module, nn.Module):
            for name, param in module.named_parameters():
                param.requires_grad = True

    # Unlock the specified number of layer groups from the end
    num_groups_to_unlock = min(unlocked_layers, len(groups))
    for group in groups[-num_groups_to_unlock:]:
        _unlock(group)
