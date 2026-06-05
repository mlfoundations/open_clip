""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re
from typing import Optional

import torch
import torch.nn as nn
from torch import TensorType

try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    transformers = None


    class BaseModelOutput:
        pass


    class PretrainedConfig:
        pass

from .hf_configs import arch_dict


# utils
def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


# TODO: ?last - for gpt-like models
_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        if (self.use_pooler_output and
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        return x.last_hidden_state[:, self.cls_token_position, :]


class HFTextEncoder(nn.Module):
    """HuggingFace model adapter"""

    def __init__(
            self,
            model_name_or_path: str,
            output_dim: int,
            config: PretrainedConfig = None,
            pooler_type: str = None,
            proj_type: str = None,
            pretrained: bool = True,
            output_tokens: bool = False,
            model_config: Optional[dict] = None,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        # TODO: find better way to get this information
        uses_transformer_pooler = (pooler_type == "cls_pooler")

        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            # Apply optional HF config overrides (e.g. {"hidden_dropout_prob": 0.0,
            # "attention_probs_dropout_prob": 0.0}) before construction so they take effect for both the
            # pretrained and from-scratch branches; passing the modified config to from_pretrained still
            # loads the pretrained weights.
            for key, value in (model_config or {}).items():
                setattr(self.config, key, value)
            # TODO: do all model configs have this attribute? PretrainedConfig does so yes??
            if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
                self.transformer = (
                    AutoModel.from_pretrained(model_name_or_path, config=self.config)
                    if pretrained else AutoModel.from_config(self.config)
                )
                self.transformer = self.transformer.encoder
            else:
                self.transformer = (
                    AutoModel.from_pretrained(
                        model_name_or_path, config=self.config, add_pooling_layer=uses_transformer_pooler)
                    if pretrained else
                    AutoModel.from_config(self.config, add_pooling_layer=uses_transformer_pooler)
                )
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)
            # Normalize encoder-decoder models to their encoder, matching the model_name branch above, so the rest
            # of the class (layer_groups / forward) sees encoder attrs (e.g. embed_tokens) regardless of how the
            # tower was constructed.
            if getattr(config, "is_encoder_decoder", False):
                self.transformer = self.transformer.encoder
        if pooler_type is None:  # get default arch pooler
            pooler_type = (arch_dict[self.config.model_type]["pooler"])

        # FIXME downstream users of OpenCLIP models use these attr, need to verify valid across all models
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)

        self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )
        elif proj_type == 'clap_mlp':
            # Matches the HF Transformers CLAP text projection checkpoint layout.
            self.proj = nn.Sequential(
                nn.Linear(d_model, output_dim, bias=True),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim, bias=True),
            )

    def forward(self, x: TensorType):
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[:, torch.arange(seq_len) != self.pooler.cls_token_position, :] 
            if type(self.pooler) == ClsPooler 
            else out.last_hidden_state
        )
        
        if self.output_tokens:
            return projected, tokens
        return projected

    def layer_groups(self, pooler_in_head: bool = True):
        """Ordered, complete partition of this text tower into named ``(name, [members])`` groups, input -> output.

        Shared by ``lock`` (freeze the bottom groups) and layer-wise LR decay (depth -> lr_scale), so the
        model-specific layer enumeration -- and the pooler-placement policy -- live in one place. ``members`` is a
        list of ``nn.Module``/``nn.Parameter``. Base groups: ``embeddings``, ``layer.{i}`` (encoder blocks), and
        ``proj`` (the projection head). A parametrized readout ``pooler`` (e.g. cls_pooler/roberta) is placed by
        ``pooler_in_head``: True folds it into the ``proj`` head group (never frozen / lr_scale 1.0); False gives it
        its own group just below ``proj`` (frozen with the encoder under ``lock`` / decayed under LLRD). Every
        trainable parameter of the tower is covered exactly once.
        """
        cfg_names = arch_dict[self.config.model_type]["config_names"]
        encoder = self.transformer.encoder if hasattr(self.transformer, "encoder") else self.transformer
        embeddings = getattr(self.transformer, cfg_names["token_embeddings_attr"])
        layer_list = getattr(encoder, cfg_names["layer_attr"])
        pooler = getattr(self.transformer, "pooler", None)
        has_pooler = (pooler is not None) and (next(pooler.parameters(), None) is not None)
        has_proj = not isinstance(self.proj, nn.Identity)

        groups = [("embeddings", [embeddings])]
        groups += [(f"layer.{i}", [layer]) for i, layer in enumerate(layer_list)]
        if has_pooler and not pooler_in_head:
            groups.append(("pooler", [pooler]))
        head = ([pooler] if (has_pooler and pooler_in_head) else []) + ([self.proj] if has_proj else [])
        if head:
            groups.append(("proj", head))
        return groups

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True, pooler_in_head: bool = True):
        # Freeze the tower top-down, reusing the same enumeration (and pooler policy) as layer-wise LR decay via
        # ``layer_groups``: ``unlocked_layers`` counts the top groups (the ``proj`` head first) left trainable, so
        # unlocked_layers=0 freezes everything. This is offset by one from decay, which keeps the head at lr_scale
        # 1.0 (depth 0). The pooler rides in the proj head (pooler_in_head=True) or is its own group just below it
        # (False), so the policy governs whether unlocked_layers=1 frees the pooler (True) or only the projection.
        groups = self.layer_groups(pooler_in_head)
        n_freeze = len(groups) if not unlocked_layers else len(groups) - unlocked_layers
        print(f"Locking {n_freeze}/{len(groups)} text groups of hf model (unlocked_layers={unlocked_layers})")
        # Set every group explicitly so repeated calls with different counts are idempotent (multi-stage training):
        # frozen groups -> False (LayerNorm kept trainable unless freeze_layer_norm), unlocked groups -> True.
        for i, (_, members) in enumerate(groups):
            freeze = i < n_freeze
            for module in members:
                for n, p in module.named_parameters():
                    if freeze:
                        p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
                    else:
                        p.requires_grad = True

    # Names of *learned* positional / type embeddings to exclude from weight decay. They are lookup tables, not
    # content weights (the word/token table stays decayed, the usual convention), and being non-1-D they are not
    # caught by the optimizer's dimensional rule. Covers BERT/RoBERTa (position_embeddings, token_type_embeddings),
    # T5/mT5 (relative_attention_bias) and learned-absolute-position seq2seq such as BART (embed_positions).
    # Sinusoidal positions (m2m_100/NLLB, Whisper) are registered as buffers, so the patterns match no parameters
    # there and they are correctly never decayed.
    _NO_WD_EMBED_NAMES = ("position_embeddings", "token_type_embeddings", "relative_attention_bias", "embed_positions")

    def no_weight_decay_patterns(self):
        """Glob patterns for parameters to exclude from weight decay (the optimizer scopes them to this tower).

        HF models declare no such set themselves (their Trainer excludes norms/biases by module *type*, which our
        optimizer's 1-D rule already covers). We additionally exclude the learned positional / type embeddings in
        :attr:`_NO_WD_EMBED_NAMES` -- matching the CLIP/ViT convention rather than HF's (which decays them). Each
        name is wrapped in ``*`` so it matches anywhere in the tower-scoped path: the leading ``*`` bridges the
        intermediate ``transformer.embeddings.``/``transformer.block.{i}...`` path, the trailing ``*`` covers the
        ``.weight`` leaf. LayerNorms/biases are left to the optimizer's dimensional rule.
        """
        return [f"*{name}*" for name in self._NO_WD_EMBED_NAMES]

    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass
