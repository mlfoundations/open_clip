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
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
except ImportError as e:
    transformers = None

from timm.models.layers import Mlp

# utils
def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

# TODO: ?last - for gpt-like models
_POOLERS = {}

def register_pooler(cls):
    "Decorator registering pooler class"
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    "Mean pooling"

    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)        
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)

@register_pooler
class MaxPooler(nn.Module):
    "Max pooling"

    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values

@register_pooler
class ClsPooler(nn.Module):
    "CLS token pooling"

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        
        if (self.use_pooler_output and 
            isinstance(x, BaseModelOutputWithPooling) and
            (x.pooler_output is not None)
            ):
            return x.pooler_output
        
        return x.last_hidden_state[:, self.cls_token_position, :]


# arch-to-pooler mapping
_DEFAULT_POOLER = {}

def get_pooler(pooler_type:str):
    if pooler_type is None:
        # pooler_type = _DEFAULT_POOLER[self.config]
        return MeanPooler()
    else:
        _POOLERS[pooler_type]()

class PreTrainedTextEncoder(nn.Module):
    """HuggingFace model adapter
    
    # TODO: add dockstring here
    """
    def __init__(
            self, 
            model_name_or_path:str,
            output_dim:int,
            config: PretrainedConfig=None,
            pooler_type:str=None,
            proj:str=None):
        super().__init__()

        self.output_dim = output_dim

        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        if config is None:
          self.config = AutoConfig.from_pretrained(model_name_or_path)
          self.transformer = AutoModel.from_pretrained(model_name_or_path)
        else:
          self.config = config
          self.transformer = AutoModel.from_config(config)
        
        self.pooler = get_pooler(pooler_type)
        d_model = self.config.hidden_size # TODO: get d_model from config
        # different models can have different names for it
        # ?? do we use separate classes for different archs or handle it with helper funcs  
        if (d_model == output_dim) and (proj is None): # do we always need a proj?
            self.proj = nn.Identity()
        elif proj == 'linear':
            self.proj == nn.Linear(d_model, output_dim, bias=False)
        elif proj == 'mlp':
            self.proj = Mlp(d_model, (d_model + output_dim)//2, output_dim, bias=False)

    def forward(self, x:TensorType) -> TensorType:
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)

        return self.proj(pooled_out)

    def lock(self, unlocked_layers:int=0, freeze_layer_norm:bool=True):
        # TODO: add support for partial freezing
        for n, p in self.transformer.named_parameters():
            if True: #mb optional LayerNorm params etc.
                p.requires_grad = False
