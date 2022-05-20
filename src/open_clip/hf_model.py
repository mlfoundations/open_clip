""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""

import torch
import torch.nn as nn
from torch import TensorType
from traitlets import default
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig
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

            context_length:int = 77,
            vocab_size:int = 49408,
            width:int = 512,
            heads:int = 8,
            layers:int = 12,
            output_dim:int = 512,
            pooler_type:str=None,
            proj:str=None):
        super().__init__()

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim

        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        
        self.pooler = get_pooler(pooler_type)
        d_model = self.config.hidden_size # TODO: get d_model from config
        # different models can have different names for it
        # ?? do we use separate classes for different archs or handle it with helper funcs  
        if (d_model == width) and (proj is None): # do we always need a proj?
            self.proj = nn.Identity()
        elif proj == 'linear':
            self.proj == nn.Linear(d_model, width, bias=False)
        elif proj == 'mlp':
            # TODO: add me
            pass

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, x:TensorType) -> TensorType:
        # TODO: add attention mask?
        pad = torch.full((x.shape[0], self.context_length), self.config.pad_token_id)
        pad[:, :x.shape[1]] = x
        x = pad

        out = self.transformer(input_ids=x, attention_mask=self.attn_mask[:x.shape[0]])
        pooled_out = self.pooler(out)

        return self.proj(pooled_out)

    def lock(self, unlocked_layers:int=0, freeze_layer_norm:bool=True):
        # TODO: add support for partial freezing
        for n, p in self.transformer.named_parameters():
            if True: #mb optional LayerNorm params etc.
                p.requires_grad = False
