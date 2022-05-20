""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""

import torch
import torch.nn as nn
from torch import TensorType
from traitlets import default
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
except ImportError as e:
    transformers = None


# utils
# TODO: cls, max, mean, last
_POOLERS = {}

def register_pooler(cls):
    "Register pooler class"
    pass

class DummyPooler(nn.Module):
    "Fetches first of output hidden state"

    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        return x.last_hidden_state[:, 0, :]

# arch-to-pooler mapping
_DEFAULT_POOLER = {}

def get_pooler(pooler_type:str):
    if pooler_type is None:
        # pooler_type = _DEFAULT_POOLER[self.config]
        pass
    return DummyPooler()

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
            # TODO: add me
            pass

    def forward(self, x:TensorType) -> TensorType:
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out)

        return self.proj(pooled_out)

    def lock(self, unlocked_layers:int=0, freeze_layer_norm:bool=True):
        # TODO: add support for partial freezing
        for n, p in self.transformer.named_parameters():
            if True: #mb optional LayerNorm params etc.
                p.requires_grad = False
