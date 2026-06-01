from types import SimpleNamespace

import pytest
import torch

from open_clip_train import zero_shot as zero_shot_module


class _BareModel:
    visual = object()

    def encode_image(self, image, normalize=False):
        raise NotImplementedError

    def encode_text(self, text, normalize=False):
        raise NotImplementedError


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


def test_zero_shot_eval_skips_generative_model_without_text_tower():
    """A generative VLM (image tower, but no encode_text) skips the contrastive zero-shot path."""
    class _GenerativeModel:
        visual = object()

        def encode_image(self, image, normalize=False):
            raise NotImplementedError
        # no encode_text -> no contrastive text tower (e.g. GenLIP)

    args = SimpleNamespace(
        distributed=False,
        zeroshot_frequency=1,
        epochs=1,
        model="naflexgenlip_b16_224",
        device="cpu",
        precision="fp32",
        batch_size=1,
        rank=0,
        fsdp=False,
    )
    data = {"imagenet-val": _DataWrapper()}

    results = zero_shot_module.zero_shot_eval(_GenerativeModel(), data, epoch=1, args=args)

    assert results == {}


def test_zero_shot_eval_rejects_non_image_model():
    class _AudioModel:
        audio = object()

    args = SimpleNamespace(
        distributed=False,
        zeroshot_frequency=1,
        epochs=1,
        model="CLAP-test",
        device="cpu",
        precision="fp32",
        batch_size=1,
        rank=0,
        fsdp=False,
    )
    data = {"imagenet-val": _DataWrapper()}

    with pytest.raises(ValueError, match="ImageNet zero-shot"):
        zero_shot_module.zero_shot_eval(_AudioModel(), data, epoch=1, args=args)


def test_accuracy_returns_python_floats():
    output = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
        ]
    )
    target = torch.tensor([0, 1])

    top1, top2 = zero_shot_module.accuracy(output, target, topk=(1, 2))

    assert isinstance(top1, float)
    assert isinstance(top2, float)
    assert top1 == 1.0
    assert top2 == 2.0


def test_run_zero_shot_classifier_accepts_naflex_image_dict():
    class _Model:
        def __call__(self, image):
            features = torch.zeros(image["patches"].shape[0], 2)
            features[:, 0] = 1
            return {"image_features": features}

    classifier = torch.zeros(2, 5)
    classifier[0, 0] = 1
    images = {
        "patches": torch.zeros(2, 4, 16 * 16 * 3),
        "patch_coord": torch.zeros(2, 4, 2, dtype=torch.long),
        "patch_valid": torch.ones(2, 4, dtype=torch.bool),
        "seq_len": 4,
    }
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


def test_run_zero_shot_classifier_uses_task_dummy_batch_for_fsdp_non_rank0(monkeypatch):
    calls = []

    class _Task:
        def create_dummy_batch(self, batch_size, device, dtype):
            calls.append((batch_size, device, dtype))
            return {
                "image": {
                    "patches": torch.zeros(batch_size, 4, 16 * 16 * 3, device=device, dtype=dtype),
                    "patch_coord": torch.zeros(batch_size, 4, 2, device=device, dtype=torch.long),
                    "patch_valid": torch.ones(batch_size, 4, device=device, dtype=torch.bool),
                    "seq_len": 4,
                },
                "text": torch.zeros(batch_size, 77, device=device, dtype=torch.long),
            }

        def __call__(self, image):
            raise AssertionError("rank 1 should stop before forward when broadcast signal is zero")

    monkeypatch.setattr(zero_shot_module.dist, "broadcast", lambda *_args, **_kwargs: None)
    args = SimpleNamespace(
        device="cpu",
        precision="fp32",
        batch_size=2,
        rank=1,
        fsdp=True,
        use_naflex=True,
    )

    top1, top5 = zero_shot_module.run_zero_shot_classifier(
        _Task(),
        torch.zeros(2, 5),
        [],
        args,
        use_fsdp_eval=True,
    )

    assert (top1, top5) == (0., 0.)
    assert calls == [(1, torch.device("cpu"), None)]
