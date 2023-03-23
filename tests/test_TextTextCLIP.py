import torch
from open_clip.factory import get_tokenizer
import pytest
import open_clip
import os
import sys
import pandas as pd
from training.data import get_data
from training.params import parse_args
from training.main import main

os.environ["CUDA_VISIBLE_DEVICES"] = ""

@pytest.mark.parametrize("model_cfg", [("roberta-roberta")])
def test_inference_simple(model_cfg, pretrained=None):
    model, _, preprocess = open_clip.create_model_and_transforms(model_cfg, pretrained=pretrained, jit=False, model_type='text_dual_encoder')
    tokenizer = get_tokenizer(model_cfg)

    text_a = tokenizer(['this', 'is', 'a', 'document'])
    text_b = tokenizer(['this', 'is', 'a', 'summary'])

    with torch.no_grad():
        text_a_features = model.encode_text_a(text_a)
        text_b_features = model.encode_text_b(text_b)
    
    print(text_a_features.shape)
    print(text_b_features.shape)

    text_probs = (100.0 * text_a_features @ text_b_features.T).softmax(dim=-1)
    print(text_probs)

