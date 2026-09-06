"""Public trainer helper behavior across current and legacy import paths."""
import os
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from open_clip.model_traits import CLIP_TRAITS, get_model_traits
from open_clip.task import unwrap_model
from open_clip_train import file_utils, legacy_main, legacy_train, main, train
from open_clip_train.audio_data import AudioCaptionTokenizer
from util_test import VariableTokenizer


def test_tuple_output_helpers_remain_callable():
    output = (torch.randn(2, 4), torch.randn(2, 4), torch.tensor(1.5))
    for helper in (train.postprocess_clip_output, legacy_train.postprocess_clip_output):
        result = helper(output)
        assert list(result) == ["image_features", "text_features", "logit_scale"]
        assert all(actual is expected for actual, expected in zip(result.values(), output))


def test_unwrap_model_supports_both_wrapper_orders():
    model = SimpleNamespace(traits=CLIP_TRAITS)
    for first, second in [("module", "_orig_mod"), ("_orig_mod", "module")]:
        wrapped = SimpleNamespace(**{first: SimpleNamespace(**{second: model})})
        assert unwrap_model(wrapped) is model
        assert get_model_traits(wrapped) == CLIP_TRAITS


def test_seed_and_meter_helpers_agree_across_trainers():
    main.random_seed(19, rank=2)
    expected = (torch.rand(2), np.random.rand(2), random.random())
    legacy_main.random_seed(19, rank=2)
    actual = (torch.rand(2), np.random.rand(2), random.random())
    torch.testing.assert_close(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    for meter_cls in (train.AverageMeter, legacy_train.AverageMeter):
        meter = meter_cls()
        meter.update(2, 3)
        meter.update(6, 1)
        assert (meter.val, meter.avg, meter.count) == (6, 3, 4)
        meter.reset()
        assert (meter.val, meter.avg, meter.count) == (0, 0, 0)


def test_checkpoint_selection_preserves_legacy_sharded_exclusion(tmp_path):
    checkpoint_root = str(tmp_path) + os.sep
    assert main.get_latest_checkpoint(checkpoint_root, False) is None
    nested = tmp_path / "archive"
    nested.mkdir()
    for epoch in (2, 10):
        (nested / f"epoch_{epoch}.pt").touch()
    incomplete = tmp_path / "epoch_30"
    incomplete.mkdir()
    sharded = tmp_path / "epoch_20"
    sharded.mkdir()
    (sharded / ".metadata").touch()
    assert main.get_latest_checkpoint(checkpoint_root, False) == str(sharded)
    assert legacy_main.get_latest_checkpoint(checkpoint_root, False) == str(nested / "epoch_10.pt")


@pytest.mark.parametrize("entrypoint", [main, legacy_main])
def test_remote_checkpoint_selection(entrypoint, monkeypatch):
    response = SimpleNamespace(returncode=0, stdout=b"date 100 epoch_2.pt\ndate 100 epoch_10.pt\n")
    monkeypatch.setattr(file_utils.subprocess, "run", lambda *args, **kwargs: response)
    assert entrypoint.get_latest_checkpoint("s3://bucket/run", True) == "s3://bucket/run/epoch_10.pt"
    response.returncode = 1
    assert entrypoint.get_latest_checkpoint("s3://bucket/run", True) is None


@pytest.mark.parametrize("variable", [False, True])
def test_audio_tokenizer_pickle(variable):
    tokenizer = VariableTokenizer()
    expected = tokenizer("a sound", pad=not variable)[0]
    encode = pickle.loads(pickle.dumps(AudioCaptionTokenizer(tokenizer, variable=variable)))
    torch.testing.assert_close(encode(b'{"caption": "a sound"}'), expected)
