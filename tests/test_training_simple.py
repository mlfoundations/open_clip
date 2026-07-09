
import os
import sys
import pytest
import torch
from open_clip_train.main import main

os.environ["CUDA_VISIBLE_DEVICES"] = ""


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
@pytest.mark.parametrize("model_name", ['mammut2_ViT-B-32', 'mammut2-moderntext_ViT-B-32'])
def test_training_mammut(model_name):
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
    '--model', model_name,
    '--coca-caption-loss-weight', '1.0'
    ])

@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_text_attention_mask_rejected_for_clip():
    """Explicit --text-attention-mask with a non-consumer model must fail fast (CLIPTask would
    silently drop the key in training_forward and crash in the grad-accumulation path)."""
    with pytest.raises(ValueError, match='text-attention-mask'):
        main([
        '--dataset-type', "synthetic",
        '--train-num-samples', '16',
        '--warmup', '1',
        '--batch-size', '4',
        '--epochs', '1',
        '--workers', '2',
        '--model', 'RN50',
        '--text-attention-mask',
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


