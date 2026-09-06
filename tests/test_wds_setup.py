"""Epoch accounting and evaluation exhaustion across current and legacy WDS builders."""
import io
import tarfile
from functools import partial

import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from open_clip.model_traits import CLAP_TRAITS, CLIP_TRAITS
from open_clip_train import audio_data, data, legacy_data
from open_clip_train.params import parse_args
from util_test import VariableTokenizer


def _decode_audio_bytes(payload):
    return torch.zeros(1, 8), 8


def _decode_audio(key, payload):
    return _decode_audio_bytes(payload) if key.endswith("wav") else None


def _transform_audio(decoded):
    return {"waveform": decoded[0].squeeze(0), "longer": False}


@pytest.fixture(params=["image", "audio", "legacy_image", "legacy_audio"])
def pipeline(request, tmp_path, monkeypatch):
    monkeypatch.setattr(audio_data, "_decode_audio", _decode_audio)
    monkeypatch.setattr(audio_data, "_decode_audio_bytes", _decode_audio_bytes)
    audio = "audio" in request.param
    image = io.BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="PNG")
    shard = tmp_path / "samples.tar"
    with tarfile.open(shard, "w") as tar:
        for i in range(7):
            for ext, payload in [("wav" if audio else "png", b"audio" if audio else image.getvalue()),
                                 ("txt", b"caption")]:
                member = tarfile.TarInfo(f"{i}.{ext}")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
    builders = {
        "image": partial(data.get_wds_dataset, model_traits=CLIP_TRAITS),
        "audio": partial(audio_data.get_wds_audio_dataset, model_traits=CLAP_TRAITS),
        "legacy_image": legacy_data.get_wds_dataset_legacy,
        "legacy_audio": legacy_data.get_wds_audio_dataset_legacy,
    }
    args = parse_args([])
    args.train_data = args.val_data = str(shard)
    args.train_num_samples, args.val_num_samples = 5, None
    args.workers, args.world_size, args.batch_size = 0, 1, 2
    args.audio_ext = "wav"
    return partial(
        builders[request.param], args, _transform_audio if audio else ToTensor(), tokenizer=VariableTokenizer(),
    )


@pytest.mark.parametrize("floor,expected_batches", [(False, 3), (True, 2)])
def test_fixed_training_epoch_counts(pipeline, floor, expected_batches):
    info = pipeline(is_train=True, floor=floor)
    batches = list(info.dataloader)
    assert len(batches) == info.dataloader.num_batches == expected_batches
    assert sum(len(b["text"]) for b in batches) == info.dataloader.num_samples == expected_batches * 2


def test_evaluation_without_size_exhausts_partial_batch(pipeline):
    info = pipeline(is_train=False)
    assert info.dataloader.num_samples == info.dataloader.num_batches == 0
    assert [len(b["text"]) for b in info.dataloader] == [2, 2, 2, 1]


@pytest.mark.parametrize("floor,batches,samples,per_worker", [(False, 6, 24, 2), (True, 3, 12, 1)])
def test_epoch_rounding_accounts_for_workers_and_ranks(floor, batches, samples, per_worker):
    args = parse_args([])
    args.batch_size, args.world_size, args.workers = 2, 2, 3
    dataset = data.wds.DataPipeline([])
    info = data.create_wds_loader(dataset, args, True, 17, data.SharedEpoch(), floor=floor)
    assert (info.dataloader.num_batches, info.dataloader.num_samples) == (batches, samples)
    assert dataset.nsamples == per_worker


def test_token_budget_bypasses_sample_count_lookup():
    args = parse_args([])
    args.train_num_samples = None
    assert data.get_wds_sizes(args, "missing.tar", True, num_tokens=128) == (None, None)
    args.train_num_samples = 8
    with pytest.raises(ValueError, match="Specify only one"):
        data.get_wds_sizes(args, "missing.tar", True, num_tokens=128)
