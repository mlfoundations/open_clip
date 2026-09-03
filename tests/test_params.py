import pytest

from open_clip_train.params import parse_args


def test_opt_kwargs_parse_timm_style_key_values():
    args = parse_args([
        "--opt-kwargs",
        "foreach=False",
        "amsgrad=True",
        "max_grad_norm=1.0",
        "mode=fast",
    ])

    assert args.opt_kwargs == {
        "foreach": False,
        "amsgrad": True,
        "max_grad_norm": 1.0,
        "mode": "fast",
    }


def test_val_retrieval_chunk_size_parse():
    args = parse_args(["--val-retrieval-chunk-size", "128"])

    assert args.val_retrieval_chunk_size == 128


def test_val_retrieval_precision_parse():
    args = parse_args(["--val-retrieval-precision", "model"])

    assert args.val_retrieval_precision == "model"


def test_caption_loss_options_parse():
    args = parse_args([
        "--caption-z-loss-weight", "1e-4",
        "--caption-loss-compute-dtype", "model",
        "--caption-loss-chunk-size", "512",
    ])

    assert args.caption_z_loss_weight == 1e-4
    assert args.caption_loss_compute_dtype == "model"
    assert args.caption_loss_chunk_size == 512


@pytest.mark.parametrize("option", [
    ["--caption-z-loss-weight=-1e-4"],
    ["--caption-loss-chunk-size", "0"],
])
def test_caption_loss_options_reject_invalid_values(option):
    with pytest.raises(ValueError):
        parse_args(option)
