from types import SimpleNamespace

import torch

from open_clip_train import zero_shot as zero_shot_module


class _BareModel:
    pass


class _WrappedModel:
    def __init__(self, module):
        self.module = module


class _DataWrapper:
    def __init__(self):
        self.dataloader = object()


def test_accuracy_returns_python_scalars():
    logits = torch.tensor([
        [0.1, 0.9, 0.0],
        [0.8, 0.1, 0.2],
    ])
    target = torch.tensor([1, 2])

    acc1, acc2 = zero_shot_module.accuracy(logits, target, topk=(1, 2))

    assert acc1 == 1.0
    assert acc2 == 2.0
    assert isinstance(acc1, float)
    assert isinstance(acc2, float)


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
