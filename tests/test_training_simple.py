
import torch
from PIL import Image
from training.main import main
import pytest
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def test_training():
    main([
    '--save-frequency', '1',
    '--zeroshot-frequency', '1',
    '--dataset-type', "synthetic",
    '--train-num-samples', '16',
    '--warmup', '1',
    '--batch-size', '4',
    '--lr', '1e-3',
    '--wd', '0.1',
    '--epochs', '1',
    '--workers', '2',
    '--model', 'RN50'
    ])
