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
