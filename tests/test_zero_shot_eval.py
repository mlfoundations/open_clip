from types import SimpleNamespace

import torch

from open_clip_train import zero_shot as zero_shot_module
from open_clip_train.naflex_data import create_naflex_dummy_image


class _BareModel:
    pass


class _WrappedModel:
    def __init__(self, module):
        self.module = module


class _DataWrapper:
    def __init__(self):
        self.dataloader = object()


def test_zero_shot_eval_handles_already_unwrapped_model(monkeypatch):
    bare_model = _BareModel()
    build_models = []
    run_models = []

    monkeypatch.setattr(zero_shot_module, "get_tokenizer", lambda _model: object())

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(zero_shot_module, "get_autocast", lambda *_args, **_kwargs: _NullContext)
    monkeypatch.setattr(
        zero_shot_module,
        "build_zero_shot_classifier",
        lambda model, **_kwargs: build_models.append(model) or object(),
    )

    def _run(model, classifier, dataloader, args, **kwargs):
        run_models.append(model)
        return 1.0, 1.0

    monkeypatch.setattr(zero_shot_module, "run_zero_shot_classifier", _run)

    args = SimpleNamespace(
        distributed=False,
        zeroshot_frequency=1,
        epochs=1,
        model="ViT-B-32",
        device="cpu",
        precision="fp32",
        batch_size=1,
        rank=0,
        fsdp=False,
    )
    data = {"imagenet-val": _DataWrapper()}

    results = zero_shot_module.zero_shot_eval(bare_model, data, epoch=1, args=args)

    assert results["imagenet-zeroshot-val-top1"] == 1.0
    # get_model_from_task passes the bare model through unchanged
    assert build_models == [bare_model]
    assert run_models == [bare_model]


def test_zero_shot_eval_unwraps_wrapped_model_once(monkeypatch):
    bare_model = _BareModel()
    wrapped_model = _WrappedModel(bare_model)
    build_models = []
    run_models = []

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(zero_shot_module, "get_autocast", lambda *_args, **_kwargs: _NullContext)
    monkeypatch.setattr(zero_shot_module, "get_tokenizer", lambda _model: object())
    monkeypatch.setattr(
        zero_shot_module,
        "build_zero_shot_classifier",
        lambda model, **_kwargs: build_models.append(model) or object(),
    )

    def _run(model, classifier, dataloader, args, **kwargs):
        run_models.append(model)
        return 1.0, 1.0

    monkeypatch.setattr(zero_shot_module, "run_zero_shot_classifier", _run)

    args = SimpleNamespace(
        distributed=True,
        zeroshot_frequency=1,
        epochs=1,
        model="ViT-B-32",
        device="cpu",
        precision="fp32",
        batch_size=1,
        rank=0,
        fsdp=False,
    )
    data = {"imagenet-val": _DataWrapper()}

    results = zero_shot_module.zero_shot_eval(wrapped_model, data, epoch=1, args=args)

    assert results["imagenet-zeroshot-val-top1"] == 1.0
    # get_model_from_task unwraps .module from the DDP-like wrapper
    assert build_models == [wrapped_model]  # build_zero_shot_classifier gets model_or_task
    assert run_models == [wrapped_model]  # run_zero_shot_classifier gets model_or_task


def test_run_zero_shot_classifier_accepts_naflex_image_dict():
    class _Model:
        def __call__(self, image):
            features = torch.zeros(image["patches"].shape[0], 2)
            features[:, 0] = 1
            return {"image_features": features}

    classifier = torch.zeros(2, 5)
    classifier[0, 0] = 1
    images = create_naflex_dummy_image(2, max_seq_len=4, patch_size=16)
    target = torch.zeros(2, dtype=torch.long)
    args = SimpleNamespace(
        device="cpu",
        precision="fp32",
        batch_size=2,
        rank=0,
        fsdp=False,
        use_naflex=True,
    )

    top1, top5 = zero_shot_module.run_zero_shot_classifier(_Model(), classifier, [(images, target)], args)

    assert top1 == 1.0
    assert top5 == 1.0
