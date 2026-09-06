"""Standalone loss construction and the legacy trainer's args/model adapter."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from open_clip import create_loss
from open_clip.loss import ClipLoss, CoCaLoss, DistillClipLoss, GenLipLoss, SigLipLoss
from open_clip.model_traits import CLAP_TRAITS, CLIP_TRAITS, GENLAP_TRAITS, GENLIP_TRAITS
from open_clip_train.loss import create_loss_from_args
from open_clip_train.params import parse_args


@pytest.mark.parametrize("loss_type,loss_class,options", [
    ("clip", ClipLoss, dict(local_loss=True, gather_with_grad=True, cache_labels=True)),
    ("distill_clip", DistillClipLoss, dict(local_loss=True, gather_with_grad=True, cache_labels=True)),
    ("siglip", SigLipLoss, dict(dist_impl="gather", chunk_size=2, cache_labels=True)),
    ("coca", CoCaLoss, dict(
        caption_loss_weight=0.7, clip_loss_weight=1.3, pad_id=6, z_loss_weight=0.1,
        compute_dtype="model", local_loss=True, gather_with_grad=True, cache_labels=True,
    )),
    ("genlip", GenLipLoss, dict(ignore_index=6, z_loss_weight=0.1, compute_dtype="model")),
])
def test_standalone_loss_matches_constructor_forward_and_backward(loss_type, loss_class, options):
    """Only tensors and explicit options are needed; all objective-specific options reach the loss."""
    generator = torch.Generator().manual_seed(12)
    inputs = dict(
        image_features=torch.randn(3, 4, generator=generator, dtype=torch.float64),
        text_features=torch.randn(3, 4, generator=generator, dtype=torch.float64),
        logit_scale=torch.tensor(2.0, dtype=torch.float64),
    )
    if loss_type == "siglip":
        inputs['logit_bias'] = torch.tensor(-1.0, dtype=torch.float64)
    elif loss_type == "distill_clip":
        inputs.update({f'dist_{key}': value * 0.5 for key, value in inputs.copy().items()})
    elif loss_type in ("coca", "genlip"):
        if loss_type == "genlip":
            inputs = {}
        inputs.update(
            logits=torch.randn(3, 3, 7, generator=generator, dtype=torch.float64),
            labels=torch.tensor([[0, 2, 6], [1, 6, 6], [0, 3, 4]]),
        )

    def run(loss):
        batch = {key: value.clone().requires_grad_(value.is_floating_point()) for key, value in inputs.items()}
        losses = loss(**batch, output_dict=True)
        sum(value for key, value in losses.items() if key.endswith('_loss')).backward()
        return losses, {key: value.grad for key, value in batch.items()}

    actual_loss = create_loss(loss_type, **options)
    assert type(actual_loss) is loss_class
    expected, expected_grads = run(loss_class(**options))
    actual, actual_grads = run(actual_loss)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grads, expected_grads)


@pytest.mark.parametrize("pad_id", [6, None])
def test_caption_padding_preserves_real_token_zero(pad_id):
    """Raw nonzero padding and pre-masked labels must produce the same CE and gradients."""
    logits = torch.tensor([[[4., 0., 1., 2., 0., 1., 0.], [0., 1., 2., 0., 1., 0., 3.]]], requires_grad=True)
    labels = torch.tensor([[0, 6 if pad_id is not None else -100]])
    loss = create_loss("coca", pad_id=pad_id, caption_loss_weight=1.0, clip_loss_weight=0.0)
    result = loss(
        image_features=torch.ones(1, 2), text_features=torch.ones(1, 2),
        logits=logits, labels=labels, logit_scale=torch.tensor(1.), output_dict=True,
    )['caption_loss']
    expected = F.cross_entropy(logits[:, 0], torch.tensor([0]))
    torch.testing.assert_close(result, expected)
    result.backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 1].count_nonzero() == 0


def test_caption_loss_requires_explicit_padding_convention():
    with pytest.raises(ValueError, match="explicit pad_id"):
        create_loss("coca")


@pytest.mark.parametrize("loss_type", ["mammut", "ViT-B-32", "unknown", None])
def test_unknown_loss_type_rejected(loss_type):
    with pytest.raises(ValueError, match="Unknown loss_type"):
        create_loss(loss_type)


@pytest.mark.parametrize("loss_type,options", [
    ("clip", dict(pad_id=None)),
    ("clip", dict(caption_loss_weight=0.7)),
    ("clip", dict(clip_loss_weight=0.0)),
    ("clip", dict(z_loss_weight=0.1)),
    ("clip", dict(compute_dtype="model")),
    ("clip", dict(ignore_index=6)),
    ("clip", dict(dist_impl="gather")),
    ("clip", dict(chunk_size=2)),
    ("siglip", dict(local_loss=True)),
    ("siglip", dict(gather_with_grad=True)),
    ("genlip", dict(cache_labels=True)),
    ("genlip", dict(rank=1)),
    ("genlip", dict(world_size=2)),
])
def test_incompatible_loss_options_rejected(loss_type, options):
    with pytest.raises(ValueError, match=next(iter(options))):
        create_loss(loss_type, **options)


@pytest.mark.parametrize("loss_type", ["clip", "distill_clip", "siglip", "coca"])
def test_standalone_distributed_options(loss_type):
    options = dict(pad_id=None) if loss_type == "coca" else {}
    loss = create_loss(loss_type, rank=1, world_size=2, cache_labels=True, **options)
    assert (loss.rank, loss.world_size, loss.cache_labels) == (1, 2, True)


@pytest.mark.parametrize("traits,siglip,expected_class", [
    (CLIP_TRAITS, False, ClipLoss),
    (CLAP_TRAITS, False, ClipLoss),
    (CLIP_TRAITS, True, SigLipLoss),
    (GENLIP_TRAITS, True, GenLipLoss),
    (GENLAP_TRAITS, True, GenLipLoss),
])
def test_legacy_adapter_selects_and_translates_only_relevant_options(traits, siglip, expected_class):
    # CLI defaults include options for every objective; incompatible ones must stay out of the public call.
    args = parse_args(["--model", "renamed", "--caption-z-loss-weight", "0.2", "--caption-loss-compute-dtype", "model"])
    args.distill, args.siglip, args.rank, args.world_size = False, siglip, 1, 2
    args.loss_dist_impl = "gather"
    model = SimpleNamespace(traits=traits)
    loss = create_loss_from_args(args, model)
    assert type(loss) is expected_class
    if expected_class is GenLipLoss:
        assert loss.z_loss_weight == 0.2 and loss.compute_dtype is None
    else:
        assert (loss.rank, loss.world_size, loss.cache_labels) == (1, 2, True)
        if expected_class is SigLipLoss:
            assert loss.dist_impl == "gather"
