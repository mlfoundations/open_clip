
import os
import sys
import pytest
from PIL import Image
import torch
from training.main import main

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if hasattr(torch._C, '_jit_set_profiling_executor'):
    # legacy executor is too slow to compile large models for unit tests
    # no need for the fusion performance here
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)

@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
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

@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_coca():
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
    '--model', 'coca_ViT-B-32'
    ])

@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_mt5():
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
    '--model', 'mt5-base-ViT-B-32',
    '--lock-text',
    '--lock-text-unlocked-layers', '2'
    ])



@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_unfreezing_vit():
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
    '--model', 'ViT-B-32',
    '--lock-image',
    '--lock-image-unlocked-groups', '5',
    '--accum-freq', '2'
    ])


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_clip_with_jit():
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
    '--model', 'ViT-B-32',
    '--torchscript'
    ])
