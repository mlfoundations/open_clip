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

@pytest.mark.parametrize("model_type", [("roberta-roberta")])
def test_inference_simple(model_type, pretrained=None):
    model, _, preprocess = open_clip.create_model_and_transforms(model_type, pretrained=pretrained, jit=False, text_to_text=True)
    tokenizer = get_tokenizer(model_type)

    doc = tokenizer(['this', 'is', 'a', 'document'])
    query = tokenizer(['this', 'is', 'a', 'summary'])

    with torch.no_grad():
        doc_features = model.encode_doc(doc)
        query_features = model.encode_query(query)
    
    print(doc_features.shape)
    print(query_features.shape)

    text_probs = (100.0 * doc_features @ query_features.T).softmax(dim=-1)
    print(text_probs)

    '''
if __name__ == "__main__":
    
    #open_clip.factory._rescan_model_configs()
    #print(open_clip.factory._MODEL_CONFIG_PATHS)
    #test_inference_simple("roberta-roberta",'')

    main([
    '--save-frequency', '1',
    '--zeroshot-frequency', '1',
    '--dataset-type', "synthetic",
    '--train-num-samples', '16',
    '--warmup', '1',
    '--batch-size', '2',
    '--lr', '1e-3',
    '--wd', '0.1',
    '--epochs', '1',
    '--workers', '2',
    '--model', 'roberta-roberta', 
    '--dataset-type', 'textpair', 
    '--train-data', 'text_pairs.parquet.gzip',
    '--text-to-text', 'True',
    ])
    '''