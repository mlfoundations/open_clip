""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""

import re

import torch
import torch.nn as nn
from torch import TensorType
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    transformers = None
    class BaseModelOutput: pass
    class PretrainedConfig: pass

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
    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)        
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)

@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""
    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values

@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""
    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        
        if (self.use_pooler_output and 
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
            ):
            return x.pooler_output
        
        return x.last_hidden_state[:, self.cls_token_position, :]

class PreTrainedTextEncoder(nn.Module):
    """HuggingFace model adapter"""
    def __init__(
            self, 
            model_name_or_path:str,
            output_dim:int,
            config: PretrainedConfig=None,
            pooler_type:str=None,
            proj:str=None):
        super().__init__()

        self.output_dim = output_dim

        # TODO: find better way to get this information
        uses_transformer_pooler = (pooler_type == "cls_pooler")

        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            self.transformer = AutoModel.from_pretrained(model_name_or_path, add_pooling_layer=uses_transformer_pooler)
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)

        if pooler_type is None: # get default arch pooler
            self.pooler = _POOLERS[(arch_dict[self.config.model_type]["pooler"])]()
        else:
            self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
        if (d_model == output_dim) and (proj is None): # do we always need a proj?
            self.proj = nn.Identity()
        elif proj == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj == 'mlp':
            hidden_size = (d_model + output_dim)//2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )

    def forward(self, x:TensorType) -> TensorType:
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)

        return self.proj(pooled_out)

    def lock(self, unlocked_layers:int=0, freeze_layer_norm:bool=True):
        if not unlocked_layers: # full freezing
             for n, p in self.transformer.named_parameters():
                 p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
             return

        n_layers = len(self.transformer.encoder.layer) - unlocked_layers - 1 # -1 for embeddings
        modules = [self.transformer.embeddings, self.transformer.encoder.layer[:n_layers]]
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass
