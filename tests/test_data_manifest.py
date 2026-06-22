"""Unit tests for the WebDataset multi-source manifest resolver (open_clip_train.data_manifest)."""

import json
import os

import pytest

from open_clip_train.data_manifest import (
    apply_manifest_to_args,
    load_manifest,
    looks_like_manifest,
    resolve_manifest,
)
from open_clip_train.params import parse_args

try:
    import yaml  # noqa: F401

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _write(tmp_path, name, obj):
    p = os.path.join(tmp_path, name)
    with open(p, "w") as f:
        if name.endswith((".yaml", ".yml")):
            yaml.safe_dump(obj, f)
        else:
            json.dump(obj, f)
    return p


RAW_MANIFEST = {
    "version": 1,
    "sources": [
        {"name": "a", "shards": "/data/a/{000..001}.tar", "samples": 100, "sampling_weight": 1.0},
        {"name": "b", "shards": ["/data/b/000.tar"], "samples": 50, "sampling_weight": 0.2},
    ],
}

FRACTION_MANIFEST = {
    "version": 1,
    "weight_mode": "mixture_fraction",
    "sources": [
        {"name": "small", "shards": "/data/small/000.tar", "samples": 1_000_000, "mixture_fraction": 0.5},
        {"name": "big", "shards": "/data/big/000.tar", "samples": 9_000_000, "mixture_fraction": 0.5},
    ],
}


# --- resolve_manifest -------------------------------------------------------


def test_resolve_raw_to_per_shard_lists():
    shards, weights, num = resolve_manifest(RAW_MANIFEST)
    assert shards == ["/data/a/000.tar", "/data/a/001.tar", "/data/b/000.tar"]
    # source weight replicated per shard, normalized so max == 1.0
    assert weights == [1.0, 1.0, 0.2]
    assert num == 150


def test_resolve_raw_uniform_yields_none():
    manifest = {
        "version": 1,
        "sources": [
            {"shards": "/data/a/000.tar", "samples": 10},
            {"shards": "/data/b/000.tar", "samples": 10},
        ],
    }
    shards, weights, num = resolve_manifest(manifest)
    assert shards == ["/data/a/000.tar", "/data/b/000.tar"]
    assert weights is None  # uniform raw mode => no upsampling factors
    assert num == 20


def test_resolve_mixture_fraction_upweights_small_source():
    # equal target fractions, 9x size gap => small source weighted 9x the big one
    shards, weights, num = resolve_manifest(FRACTION_MANIFEST)
    assert shards == ["/data/small/000.tar", "/data/big/000.tar"]
    assert weights is not None
    assert weights[0] == pytest.approx(1.0)  # normalized max
    assert weights[1] == pytest.approx(1.0 / 9.0)  # 0.5/9e6 vs 0.5/1e6
    assert num == 10_000_000


# --- looks_like_manifest ----------------------------------------------------


def test_detect_real_manifest(tmp_path):
    p = _write(str(tmp_path), "mix.json", RAW_MANIFEST)
    assert looks_like_manifest(p) is True


def test_detect_rejects_tar_spec():
    assert looks_like_manifest("/data/a/{000..009}.tar") is False
    assert looks_like_manifest("/data/a/000.tar::/data/b/000.tar") is False


def test_detect_rejects_json_without_sources(tmp_path):
    p = _write(str(tmp_path), "notes.json", {"version": 1, "hello": "world"})
    assert looks_like_manifest(p) is False


def test_detect_rejects_list_input():
    assert looks_like_manifest(["/data/a/000.tar"]) is False


def test_detect_syntax_error_json_raises(tmp_path):
    # a .json that exists but is malformed should fail clearly, not fall through as a shard path
    p = os.path.join(str(tmp_path), "broken.json")
    with open(p, "w") as f:
        f.write("{ this is not valid json,,, ")
    with pytest.raises(ValueError, match="valid JSON"):
        looks_like_manifest(p)


@pytest.mark.skipif(not HAS_YAML, reason="pyyaml not installed")
def test_detect_syntax_error_yaml_raises(tmp_path):
    p = os.path.join(str(tmp_path), "broken.yaml")
    with open(p, "w") as f:
        f.write("sources: [unclosed\n  - bad: : :\n")
    with pytest.raises(ValueError, match="valid YAML"):
        looks_like_manifest(p)


# --- load_manifest validation ----------------------------------------------


def test_validation_missing_samples(tmp_path):
    bad = {"version": 1, "sources": [{"shards": "/d/0.tar"}]}
    p = _write(str(tmp_path), "bad.json", bad)
    with pytest.raises(ValueError, match="samples"):
        load_manifest(p)


def test_validation_empty_sources(tmp_path):
    p = _write(str(tmp_path), "empty.json", {"version": 1, "sources": []})
    with pytest.raises(ValueError, match="non-empty"):
        load_manifest(p)


def test_validation_fraction_mode_requires_fraction(tmp_path):
    bad = {
        "version": 1,
        "weight_mode": "mixture_fraction",
        "sources": [{"shards": "/d/0.tar", "samples": 10}],
    }
    p = _write(str(tmp_path), "bad.json", bad)
    with pytest.raises(ValueError, match="mixture_fraction"):
        load_manifest(p)


def test_validation_bad_version(tmp_path):
    p = _write(str(tmp_path), "v2.json", {"version": 2, "sources": [{"shards": "/d/0.tar", "samples": 1}]})
    with pytest.raises(ValueError, match="version"):
        load_manifest(p)


# --- apply_manifest_to_args (policy) ---------------------------------------


def _args(extra=None):
    return parse_args(["--model", "ViT-B-32", "--dataset-type", "webdataset"] + (extra or []))


def test_apply_sets_list_and_samples():
    args = _args(["--dataset-resampled"])
    apply_manifest_to_args(args, RAW_MANIFEST)
    assert args.train_data == ["/data/a/000.tar", "/data/a/001.tar", "/data/b/000.tar"]
    assert args.train_num_samples == 150


def test_apply_manifest_weights_ignored_without_resampled():
    # non-uniform manifest weights without --dataset-resampled => ignored (in-order reading), not an error
    args = _args()
    apply_manifest_to_args(args, RAW_MANIFEST)
    assert args.dataset_resampled is False
    assert args.train_data_upsampling_factors is None
    assert args.train_data == ["/data/a/000.tar", "/data/a/001.tar", "/data/b/000.tar"]

    # with --dataset-resampled => weights applied per-shard
    args = _args(["--dataset-resampled"])
    apply_manifest_to_args(args, RAW_MANIFEST)
    assert args.dataset_resampled is True
    assert args.train_data_upsampling_factors == [1.0, 1.0, 0.2]


def test_apply_uniform_without_resampled_ok():
    # uniform raw manifest has no weights, so no resampling required
    uniform = {
        "version": 1,
        "sources": [
            {"shards": "/data/a/000.tar", "samples": 10},
            {"shards": "/data/b/000.tar", "samples": 10},
        ],
    }
    args = _args()
    apply_manifest_to_args(args, uniform)
    assert args.dataset_resampled is False
    assert args.train_data_upsampling_factors is None
    assert args.train_data == ["/data/a/000.tar", "/data/b/000.tar"]


def test_apply_manifest_resampled_field_ignored():
    # a stray 'resampled' key in the manifest does NOT enable resampling (CLI-only); weights are then ignored.
    manifest = dict(RAW_MANIFEST, resampled=True)
    args = _args()
    apply_manifest_to_args(args, manifest)
    assert args.dataset_resampled is False
    assert args.train_data_upsampling_factors is None


def test_apply_cli_factors_source_count_expands_per_shard():
    # RAW_MANIFEST: source a has 2 shards, b has 1 => 2 sources / 3 shards.
    args = _args(["--dataset-resampled", "--train-data-upsampling-factors", "5::1"])  # one per source
    apply_manifest_to_args(args, RAW_MANIFEST)
    assert args.train_data_upsampling_factors == [5.0, 5.0, 1.0]  # a-weight replicated across a's 2 shards


def test_apply_cli_factors_shard_count_used_as_is():
    args = _args(["--dataset-resampled", "--train-data-upsampling-factors", "5::5::1"])  # one per shard
    apply_manifest_to_args(args, RAW_MANIFEST)
    assert args.train_data_upsampling_factors == [5.0, 5.0, 1.0]


def test_apply_cli_factors_bad_count_errors():
    args = _args(["--dataset-resampled", "--train-data-upsampling-factors", "5::5::5::5"])  # 4 != 2 or 3
    with pytest.raises(ValueError, match="expected"):
        apply_manifest_to_args(args, RAW_MANIFEST)


def test_apply_cli_factors_require_resampled():
    # CLI factors without --dataset-resampled => rejected early (clear error, with manifest context)
    args = _args(["--train-data-upsampling-factors", "5::1"])
    with pytest.raises(ValueError, match="dataset-resampled"):
        apply_manifest_to_args(args, RAW_MANIFEST)


def test_apply_cli_num_samples_wins():
    args = _args(["--dataset-resampled", "--train-num-samples", "999"])
    apply_manifest_to_args(args, RAW_MANIFEST)
    assert args.train_num_samples == 999


def test_apply_rejects_auto_dataset_type():
    args = parse_args(["--model", "ViT-B-32", "--dataset-type", "auto"])
    with pytest.raises(ValueError, match="explicit --dataset-type"):
        apply_manifest_to_args(args, RAW_MANIFEST)


# --- end-to-end through parse_args (overloaded --train-data) ----------------


def test_parse_args_expands_manifest(tmp_path):
    p = _write(str(tmp_path), "mix.json", RAW_MANIFEST)
    args = parse_args(
        [
            "--model",
            "ViT-B-32",
            "--dataset-type",
            "webdataset",
            "--dataset-resampled",
            "--train-data",
            p,
        ]
    )
    assert args.train_data == ["/data/a/000.tar", "/data/a/001.tar", "/data/b/000.tar"]
    assert args.train_data_upsampling_factors == [1.0, 1.0, 0.2]
    assert args.train_num_samples == 150


def test_parse_args_leaves_plain_shards_untouched():
    spec = "/data/a/{000..009}.tar"
    args = parse_args(["--model", "ViT-B-32", "--dataset-type", "webdataset", "--train-data", spec])
    assert args.train_data == spec  # unchanged string, no manifest expansion


@pytest.mark.skipif(not HAS_YAML, reason="pyyaml not installed")
def test_yaml_manifest(tmp_path):
    p = _write(str(tmp_path), "mix.yaml", RAW_MANIFEST)
    assert looks_like_manifest(p) is True
    shards, weights, num = resolve_manifest(load_manifest(p))
    assert shards == ["/data/a/000.tar", "/data/a/001.tar", "/data/b/000.tar"]
    assert num == 150
