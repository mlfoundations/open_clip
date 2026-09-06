import sys

import pytest

from open_clip_train.main import main


def _train(model_name, tmp_path, *extra_args):
    main([
        '--device', 'cpu',
        '--logs', str(tmp_path),
        '--save-frequency', '1',
        '--zeroshot-frequency', '1',
        '--dataset-type', 'synthetic',
        '--train-num-samples', '16',
        '--warmup', '1',
        '--batch-size', '4',
        '--lr', '1e-3',
        '--wd', '0.1',
        '--epochs', '1',
        '--workers', '2',
        '--model', model_name,
        *extra_args,
    ])


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training(tmp_path):
    _train('RN50', tmp_path)


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_coca(tmp_path):
    _train('coca_ViT-B-32', tmp_path)


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
@pytest.mark.parametrize("model_name", ['mammut2_ViT-B-32', 'mammut2-moderntext_ViT-B-32'])
def test_training_mammut(model_name, tmp_path):
    _train(model_name, tmp_path, '--coca-caption-loss-weight', '1.0')


def test_training_text_attention_mask_rejected_for_clip(tmp_path):
    """An explicit text mask with a non-consumer model must fail before training."""
    with pytest.raises(ValueError, match='text-attention-mask'):
        _train('RN50', tmp_path, '--text-attention-mask')


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_mt5(tmp_path):
    _train('mt5-base-ViT-B-32', tmp_path, '--lock-text', '--lock-text-unlocked-layers', '2')


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_unfreezing_vit(tmp_path):
    _train('ViT-B-32', tmp_path, '--lock-image', '--lock-image-unlocked-groups', '5', '--accum-freq', '2')
