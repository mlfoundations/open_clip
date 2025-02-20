""" Conversion functions for 3rd part state-dicts and non-torch native checkpoint formats.
"""
from typing import Union

import torch
import numpy as np

from .model import CLIP, CustomTextCLIP
from .transformer import TextTransformer, Transformer


@torch.no_grad()
def load_big_vision_weights(model: CustomTextCLIP, checkpoint_path: str):
    """ Load weights from .npz checkpoints for official Google big_vision image-text models

    Currently, the SigLIP source models are supported and a CustomTextCLIP destination model
    w/ timm image encoder.
    """
    from timm.layers import resample_patch_embed, resample_abs_pos_embed

    def _n2p(w, t=True, idx=None):
        if idx is not None:
            w = w[idx]
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False

    def _convert_timm_img(module, prefix):
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
        if embed_conv_w.shape[-2:] != module.patch_embed.proj.weight.shape[-2:]:
            embed_conv_w = resample_patch_embed(
                embed_conv_w,
                module.patch_embed.proj.weight.shape[-2:],
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        module.patch_embed.proj.weight.copy_(embed_conv_w)
        module.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))

        if module.cls_token is not None:
            module.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))

        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
        if pos_embed_w.shape != module.pos_embed.shape:
            assert False, f'{pos_embed_w.shape}, {module.pos_embed.shape}'
            num_prefix_tokens = 0 if getattr(module, 'no_embed_class', False) else getattr(module, 'num_prefix_tokens', 1)
            pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
                pos_embed_w,
                new_size=module.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        module.pos_embed.copy_(pos_embed_w)

        mha_sub, b_sub, ln1_sub = (0, 0, 1)
        for i, block in enumerate(module.blocks.children()):
            if f'{prefix}Transformer/encoderblock/LayerNorm_0/scale' in w:
                block_prefix = f'{prefix}Transformer/encoderblock/'
                idx = i
            else:
                block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
                idx = None
            mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
            block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale'], idx=idx))
            block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias'], idx=idx))
            block.attn.qkv.weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False, idx=idx).flatten(1).T for n in ('query', 'key', 'value')]))
            block.attn.qkv.bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False, idx=idx).reshape(-1) for n in ('query', 'key', 'value')]))
            block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel'], idx=idx).flatten(1))
            block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias'], idx=idx))
            block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale'], idx=idx))
            block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias'], idx=idx))
            for r in range(2):
                getattr(block.mlp, f'fc{r + 1}').weight.copy_(
                    _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel'], idx=idx))
                getattr(block.mlp, f'fc{r + 1}').bias.copy_(
                    _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias'], idx=idx))

        module.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
        module.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

        if module.attn_pool is not None:
            block_prefix = f'{prefix}MAPHead_0/'
            mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
            module.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
            module.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
            module.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
            module.attn_pool.kv.weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
            module.attn_pool.kv.bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
            module.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
            module.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
            module.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
            module.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
            for r in range(2):
                getattr(module.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
                getattr(module.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    def _convert_openclip_transformer(module: Transformer, prefix):
        for i, block in enumerate(module.resblocks.children()):
            if f'{prefix}encoderblock/LayerNorm_0/scale' in w:
                block_prefix = f'{prefix}encoderblock/'
                idx = i
            else:
                block_prefix = f'{prefix}encoderblock_{i}/'
                idx = None
            mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
            block.ln_1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale'], idx=idx))
            block.ln_1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias'], idx=idx))
            block.attn.in_proj_weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False, idx=idx).flatten(1).T for n in ('query', 'key', 'value')]))
            block.attn.in_proj_bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False, idx=idx).reshape(-1) for n in ('query', 'key', 'value')]))
            block.attn.out_proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel'], idx=idx).flatten(1))
            block.attn.out_proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias'], idx=idx))
            block.ln_2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_1/scale'], idx=idx))
            block.ln_2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_1/bias'], idx=idx))
            block.mlp.c_fc.weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_0/kernel'], idx=idx))
            block.mlp.c_fc.bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_0/bias'], idx=idx))
            block.mlp.c_proj.weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_1/kernel'], idx=idx))
            block.mlp.c_proj.bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_1/bias'], idx=idx))

    def _convert_openclip_txt(module: TextTransformer, prefix):
        module.token_embedding.weight.copy_(_n2p(w[f'{prefix}Embed_0/embedding'], t=False))
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False).squeeze(0)
        module.positional_embedding.copy_(pos_embed_w)
        _convert_openclip_transformer(module.transformer, prefix=prefix + 'Encoder_0/')
        module.ln_final.weight.copy_(_n2p(w[f'{prefix}Encoder_0/encoder_norm/scale']))
        module.ln_final.bias.copy_(_n2p(w[f'{prefix}Encoder_0/encoder_norm/bias']))
        if module.text_projection is not None:
            module.text_projection.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
            module.text_projection.bias.copy_(_n2p(w[f'{prefix}head/bias']))

    root_prefix = 'params/' if 'params/b' in w else ''
    _convert_timm_img(model.visual.trunk, f'{root_prefix}img/')
    _convert_openclip_txt(model.text, f'{root_prefix}txt/')
    model.logit_bias.copy_(_n2p(w[f'{root_prefix}b'])[0])
    model.logit_scale.copy_(_n2p(w[f'{root_prefix}t'])[0])


@torch.no_grad()
def convert_mobile_clip_state_dict(model: CustomTextCLIP, state_dict, fastvit = True):

    def _convert_timm_img(state_dict):
        if fastvit:
            from timm.models.fastvit import checkpoint_filter_fn
        else:
            from timm.models.vision_transformer_hybrid import checkpoint_filter_fn
        timm_state_dict = checkpoint_filter_fn(state_dict, model.visual.trunk)
        timm_state_dict = {'visual.trunk.' + k: v for k, v in timm_state_dict.items()}
        return timm_state_dict

    def _convert_openclip_txt(state_dict, prefix='text_encoder.'):
        text_dict = {}
        for k, v in state_dict.items():
            if not k.startswith(prefix):
                continue
            k = k.replace(prefix, '')
            k = k.replace('projection_layer', 'text_projection')
            k = k.replace('embedding_layer', 'token_embedding')
            if k.startswith('positional_embedding.pos_embed.pos_embed'):
                k = k.replace('positional_embedding.pos_embed.pos_embed', 'positional_embedding')
                v = v.squeeze()
            k = k.replace('final_layer_norm', 'ln_final')
            k = k.replace('pre_norm_mha.0', 'ln_1')
            k = k.replace('pre_norm_mha.1', 'attn')
            k = k.replace('pre_norm_ffn.0', 'ln_2')
            k = k.replace('pre_norm_ffn.1', 'mlp.c_fc')
            k = k.replace('pre_norm_ffn.4', 'mlp.c_proj')
            k = k.replace('qkv_proj.weight', 'in_proj_weight')
            k = k.replace('qkv_proj.bias', 'in_proj_bias')
            k = k.replace('transformer.', 'transformer.resblocks.')
            text_dict['text.' + k] = v
        return text_dict

    image_dict = _convert_timm_img(state_dict)
    text_dict = _convert_openclip_txt(state_dict)
    out_dict = {**image_dict, **text_dict}
    out_dict['logit_scale'] = state_dict['logit_scale']
    return out_dict


def convert_state_dict(model: Union[CustomTextCLIP, CLIP], state_dict):
    if 'image_encoder.model.patch_embed.0.rbr_conv.0.conv.weight' in state_dict:
        # Apple MobileCLIP s1 & s2 state_dicts (s0 and b not currently supported)
        state_dict = convert_mobile_clip_state_dict(model, state_dict)
    if 'image_encoder.model.patch_emb.0.block.conv.weight' in state_dict:
        # convert b model
        state_dict = convert_mobile_clip_state_dict(model, state_dict, fastvit=False)
    return state_dict
