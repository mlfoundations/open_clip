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
