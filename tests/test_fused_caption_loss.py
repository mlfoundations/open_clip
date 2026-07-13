"""Fused caption loss (CoCa / MaMMUT) parity with the legacy logits path.

The fused path (model ``forward(labels=...)`` -> ``fused_linear_cross_entropy``) must produce
the same caption loss and the same gradients as the legacy path (materialized ``[B, L, V]``
logits -> ``CoCaLoss`` CE), for every decoder head variant (legacy Parameter heads, modern
Linear heads).
"""
import pytest
import torch
import torch.nn.functional as F

import open_clip
from open_clip.loss import caption_cross_entropy, fused_linear_cross_entropy

MODELS = [
    'coca_ViT-B-32',                # classic MultimodalTransformer head (text_projection Parameter)
    'coca2_ViT-B-32',               # coca2, classic text_arch
    'coca2-moderntext_ViT-B-32',    # coca2, modern text_arch (ModernMultimodalTransformer, nn.Linear head)
    'mammut2_ViT-B-32',              # classic MultimodalDecoder head (lm_head Parameter)
    'mammut2-moderntext_ViT-B-32',   # ModernMultimodalDecoder head (nn.Linear, maybe tied)
]


def _make_batch(model, batch_size=3, seed=0):
    torch.manual_seed(seed)
    ctx = model.context_length
    image_size = model.visual.image_size
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]
    image = torch.randn(batch_size, 3, image_size, image_size)
    text = torch.randint(1, 400, (batch_size, ctx))
    # right-padded captions of varying length
    text_valid = torch.zeros(batch_size, ctx, dtype=torch.bool)
    for i, n in enumerate((ctx, ctx * 2 // 3, 5)):
        text_valid[i, :n] = True
    text = text.masked_fill(~text_valid, getattr(model, 'pad_id', 0) or 0)
    labels = text[:, 1:].masked_fill(~text_valid[:, 1:], -100)
    return image, text, text_valid, labels


@pytest.mark.parametrize('model_name', MODELS)
def test_fused_caption_loss_matches_legacy(model_name):
    model = open_clip.create_model(model_name)
    model.train()
    image, text, text_valid, labels = _make_batch(model)

    # legacy: materialized logits, CE with the same shift/mask CoCaTask applies
    out_legacy = model(image=image, text=text, text_valid=text_valid)
    logits = out_legacy['logits'][:, :-1]
    z_weight = 1e-4
    ce_legacy, z_legacy = caption_cross_entropy(logits, labels, ignore_index=-100, z_loss=True)
    loss_legacy = ce_legacy + z_weight * z_legacy

    # fused: labels into forward, model returns the reduced unweighted components
    out_fused = model(
        image=image, text=text, text_valid=text_valid, labels=labels,
        caption_z_loss=True,
    )
    assert 'logits' not in out_fused
    assert out_fused['caption_loss_z'] > 0
    loss_fused = out_fused['caption_loss_ce'] + z_weight * out_fused['caption_loss_z']

    torch.testing.assert_close(loss_fused, loss_legacy, rtol=1e-5, atol=1e-5)

    # gradient parity on a shared trunk parameter
    trunk_param = next(p for n, p in model.named_parameters() if 'visual' in n and p.dim() > 1)
    model.zero_grad()
    loss_legacy.backward(retain_graph=False)
    g_legacy = trunk_param.grad.clone()
    model.zero_grad()
    out_fused2 = model(
        image=image, text=text, text_valid=text_valid, labels=labels,
        caption_z_loss=True,
    )
    (out_fused2['caption_loss_ce'] + z_weight * out_fused2['caption_loss_z']).backward()
    torch.testing.assert_close(trunk_param.grad, g_legacy, rtol=1e-4, atol=1e-6)


def test_fused_linear_cross_entropy_chunking():
    """Chunked reduction must be exact regardless of chunk size (incl. ignored positions)."""
    torch.manual_seed(0)
    n, d, v = 100, 16, 50
    hidden = torch.randn(n, d, requires_grad=True)
    weight = torch.randn(v, d, requires_grad=True)
    target = torch.randint(0, v, (n,))
    target[::7] = -100
    ref = F.cross_entropy(hidden @ weight.t(), target, ignore_index=-100)
    for chunk in (7, 32, 1000):
        ce, z = fused_linear_cross_entropy(hidden, weight, target, chunk_size=chunk)
        assert z is None  # z_loss off -> no z component
        torch.testing.assert_close(ce, ref, rtol=1e-6, atol=1e-6)


def test_fused_linear_cross_entropy_z_loss_matches_materialized():
    """CE, z-loss, ignored-token handling, and gradients match the materialized objective."""
    torch.manual_seed(1)
    n, d, v = 37, 12, 41
    target = torch.randint(0, v, (n,))
    target[::6] = -100
    z_weight = 1e-4

    hidden = torch.randn(n, d, requires_grad=True)
    weight = torch.randn(v, d, requires_grad=True)
    fused_ce, fused_z = fused_linear_cross_entropy(
        hidden, weight, target,
        chunk_size=7,
        z_loss=True,
    )
    fused_grads = torch.autograd.grad(fused_ce + z_weight * fused_z, (hidden, weight))

    hidden_ref = hidden.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    ref_ce, ref_z = caption_cross_entropy(
        hidden_ref @ weight_ref.t(), target,
        z_loss=True,
    )
    ref_grads = torch.autograd.grad(ref_ce + z_weight * ref_z, (hidden_ref, weight_ref))

    for actual, expected in ((fused_ce, ref_ce), (fused_z, ref_z)):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    for actual, expected in zip(fused_grads, ref_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fused_linear_cross_entropy_model_compute():
    """Model mode preserves bf16 loss-logit compute while returning fp32 reduced scalars."""
    torch.manual_seed(2)
    n, d, v = 23, 8, 29
    hidden = torch.randn(n, d).bfloat16()
    weight = torch.randn(v, d).bfloat16()
    target = torch.randint(0, v, (n,))
    target[::5] = -100

    ce, z = fused_linear_cross_entropy(
        hidden, weight, target,
        chunk_size=6,
        z_loss=True,
        compute_dtype="model",
    )
    valid = target != -100
    logits = hidden[valid] @ weight.t()
    log_z = torch.logsumexp(logits, dim=-1)
    ref_ce = (log_z - logits.gather(-1, target[valid, None]).squeeze(-1)).float().mean()
    ref_z = log_z.float().square().mean()

    assert ce.dtype == z.dtype == torch.float32
    torch.testing.assert_close(ce, ref_ce, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(z, ref_z, rtol=1e-6, atol=1e-6)


def test_all_targets_ignored_backward():
    """An all-padding batch must yield a graph-attached zero loss and zero (not missing) grads."""
    torch.manual_seed(3)
    n, d, v = 11, 8, 19
    hidden = torch.randn(n, d, requires_grad=True)
    weight = torch.randn(v, d, requires_grad=True)
    bias = torch.randn(v, requires_grad=True)
    target = torch.full((n,), -100)

    ce, z = fused_linear_cross_entropy(hidden, weight, target, bias=bias, z_loss=True)
    loss = ce + 1e-4 * z
    assert loss.item() == 0.0
    loss.backward()
    for param in (hidden, weight, bias):
        assert param.grad is not None
        assert torch.all(param.grad == 0)

    logits = (torch.randn(n, v)).requires_grad_()
    for z_loss in (False, True):
        ce, z = caption_cross_entropy(logits, target, z_loss=z_loss)
        loss = ce if z is None else ce + 1e-4 * z
        assert loss.item() == 0.0
        loss.backward()
        assert torch.all(logits.grad == 0)
        logits.grad = None


def test_caption_cross_entropy_masked_positions_stay_out():
    """Extreme (finite) logits at ignored positions must not poison the loss or its grads."""
    torch.manual_seed(4)
    n, v = 10, 17
    logits = torch.randn(n, v)
    target = torch.randint(0, v, (n,))
    target[::3] = -100
    logits[::3] = 1e30  # padded rows can carry garbage activations
    logits.requires_grad_()

    ce, z = caption_cross_entropy(logits, target, z_loss=True)
    loss = ce + 1e-4 * z
    assert torch.isfinite(loss) and torch.isfinite(ce) and torch.isfinite(z)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[::3] == 0)


def test_compute_mode_respects_explicit_upcast_and_ambient_autocast():
    """float32 upcasts loss logits; model mode preserves their dtype and ambient autocast."""
    torch.manual_seed(5)
    n, d, v = 23, 8, 29
    hidden = torch.randn(n, d)
    weight = torch.randn(v, d)
    logits = hidden @ weight.t()
    target = torch.randint(0, v, (n,))
    target[::5] = -100
    z_weight = 2e-4

    def compose(ce_z):
        ce, z = ce_z
        return ce + z_weight * z

    logits_bf16 = logits.bfloat16()
    model_ref = compose(caption_cross_entropy(
        logits_bf16, target, z_loss=True, compute_dtype="model"))
    fp32_ref = compose(caption_cross_entropy(
        logits_bf16, target, z_loss=True, compute_dtype="float32"))
    assert model_ref.dtype == fp32_ref.dtype == torch.float32
    assert not torch.allclose(model_ref, fp32_ref)  # the loss-logit compute mode is observable
    with torch.autocast("cpu", dtype=torch.bfloat16):
        # The fused head GEMM follows AMP (bf16 here). float32 only upcasts its output before
        # CE/logsumexp; model leaves that bf16 output under the ambient policy.
        fused_model = compose(fused_linear_cross_entropy(
            hidden, weight, target, z_loss=True, compute_dtype="model"))
        fused_fp32 = compose(fused_linear_cross_entropy(
            hidden, weight, target, z_loss=True, compute_dtype="float32"))
    assert fused_model.dtype == fused_fp32.dtype == torch.float32
    torch.testing.assert_close(fused_model, model_ref, rtol=2e-2, atol=0)
    torch.testing.assert_close(fused_fp32, fp32_ref, rtol=2e-2, atol=0)


def test_caption_cross_entropy_meta_device():
    """Materialized and fused caption losses support meta-device dry runs."""
    logits = torch.randn(6, 13, device='meta')
    target = torch.zeros(6, dtype=torch.long, device='meta')
    for z_loss in (False, True):
        ce, z = caption_cross_entropy(logits, target, z_loss=z_loss)
        assert ce.device.type == 'meta'
        assert (z is None) == (not z_loss)

    # fused path: the data-dependent valid-mask filtering cannot run on meta; a graph-attached
    # scalar short-circuit keeps shape/memory dry runs working
    hidden = torch.randn(6, 8, device='meta', requires_grad=True)
    weight = torch.randn(13, 8, device='meta')
    bias = torch.randn(13, device='meta')
    ce, z = fused_linear_cross_entropy(hidden, weight, target, bias=bias, z_loss=True)
    assert ce.device.type == 'meta' and z.device.type == 'meta'
    assert ce.dtype == torch.float32
    assert ce.requires_grad


def test_coca_z_loss_weight_independent_of_caption_weight():
    """The effective z weight is exactly z_loss_weight, not scaled by caption_loss_weight."""
    from open_clip.loss import CoCaLoss
    torch.manual_seed(6)
    z_weight, caption_weight = 1e-2, 2.0
    b, seq_len, v, d = 4, 9, 32, 8
    img_f = F.normalize(torch.randn(b, d), dim=-1)
    txt_f = F.normalize(torch.randn(b, d), dim=-1)
    logits = torch.randn(b, seq_len, v)
    labels = torch.randint(0, v, (b, seq_len))
    scale = torch.tensor(10.0)

    loss_fn = CoCaLoss(
        caption_loss_weight=caption_weight, clip_loss_weight=1.0, pad_id=None,
        z_loss_weight=z_weight,
    )
    ce, z = caption_cross_entropy(logits, labels, z_loss=True)
    for kwargs in (
            {"logits": logits, "labels": labels},               # legacy path
            {"caption_loss_ce": ce, "caption_loss_z": z},       # fused path
    ):
        out = loss_fn(img_f, txt_f, logit_scale=scale, output_dict=True, **kwargs)
        torch.testing.assert_close(out["caption_loss"], caption_weight * ce + z_weight * z)


def test_negative_z_loss_weight_rejected():
    """Composition owners validate the z weight; a negative value would subtract the regularizer."""
    from open_clip.loss import CoCaLoss, GenLipLoss
    from open_clip.task.coca_task import CoCaTask
    from open_clip.task.genlip_task import GenLipTask
    import torch.nn as nn

    with pytest.raises(ValueError, match='non-negative'):
        CoCaLoss(caption_loss_weight=2.0, clip_loss_weight=1.0, pad_id=None, z_loss_weight=-1e-4)
    with pytest.raises(ValueError, match='non-negative'):
        GenLipLoss(z_loss_weight=-1e-4)

    class Stub(nn.Module):
        pad_id = 0

    with pytest.raises(ValueError, match='non-negative'):
        CoCaTask(Stub(), caption_z_loss_weight=-1e-4, default_loss=False)
    with pytest.raises(ValueError, match='non-negative'):
        GenLipTask(Stub(), caption_z_loss_weight=-1e-4, default_loss=False)


def test_genlip_task_missing_z_component_raises():
    """A configured z weight must not silently train without the z term (model ignored the flag)."""
    import torch.nn as nn
    from open_clip.task.genlip_task import GenLipTask

    class NoZModel(nn.Module):
        pad_id = 0

        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(()))

        def forward(self, image=None, text=None, text_valid=None, compute_loss=False, **kwargs):
            return {'caption_loss_ce': self.p + 1.0}  # never returns caption_loss_z

    batch = {
        'image': torch.zeros(2, 4),
        'text': torch.ones(2, 5, dtype=torch.long),
        'text_valid': torch.ones(2, 5, dtype=torch.bool),
    }
    task = GenLipTask(NoZModel(), caption_z_loss_weight=1e-4)
    with pytest.raises(ValueError, match='caption_loss_z'):
        task.training_forward(batch)

    # with z disabled the same model is a complete contract
    task = GenLipTask(NoZModel())
    losses, _ = task.training_forward(batch)
    assert losses['loss'].item() == 1.0
    assert 'caption_loss_z' not in losses


def test_coca_loss_precomputed_ce_without_z_raises():
    """z enabled but the model-supplied caption term lacks caption_loss_z -- fail loudly."""
    from open_clip.loss import CoCaLoss
    torch.manual_seed(7)
    d = 8
    img_f = F.normalize(torch.randn(4, d), dim=-1)
    txt_f = F.normalize(torch.randn(4, d), dim=-1)
    scale = torch.tensor(10.0)

    loss_fn = CoCaLoss(
        caption_loss_weight=2.0, clip_loss_weight=1.0, pad_id=None, z_loss_weight=1e-4)
    with pytest.raises(ValueError, match='caption_loss_z'):
        loss_fn(img_f, txt_f, caption_loss_ce=torch.tensor(1.5), logit_scale=scale, output_dict=True)

    # z disabled: a bare precomputed CE is a complete caption term
    loss_fn = CoCaLoss(caption_loss_weight=2.0, clip_loss_weight=1.0, pad_id=None)
    out = loss_fn(img_f, txt_f, caption_loss_ce=torch.tensor(1.5), logit_scale=scale, output_dict=True)
    torch.testing.assert_close(out['caption_loss'], torch.tensor(3.0))


@pytest.mark.parametrize("compute_mode", ["float32", "model"])
def test_coca_loss_module_replacement_honored(compute_mode):
    """Replacing the public caption_loss module must work in either loss-compute mode."""
    import torch.nn as nn
    from open_clip.loss import CoCaLoss
    torch.manual_seed(8)
    b, seq_len, v, d = 4, 9, 32, 8
    img_f = F.normalize(torch.randn(b, d), dim=-1)
    txt_f = F.normalize(torch.randn(b, d), dim=-1)
    logits = torch.randn(b, seq_len, v).bfloat16()
    labels = torch.randint(0, v, (b, seq_len))
    labels[:, -2:] = -100
    scale = torch.tensor(10.0)

    loss_fn = CoCaLoss(
        caption_loss_weight=2.0,
        clip_loss_weight=0.0,
        pad_id=None,
        compute_dtype=compute_mode,
    )
    default = loss_fn(img_f, txt_f, logits=logits, labels=labels, logit_scale=scale, output_dict=True)
    loss_fn.caption_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.2)
    smoothed = loss_fn(img_f, txt_f, logits=logits, labels=labels, logit_scale=scale, output_dict=True)
    assert not torch.allclose(default['caption_loss'], smoothed['caption_loss'])
    expected_logits = logits.float() if compute_mode == "float32" else logits
    expected = 2.0 * F.cross_entropy(
        expected_logits.permute(0, 2, 1), labels, ignore_index=-100, label_smoothing=0.2).float()
    torch.testing.assert_close(smoothed['caption_loss'], expected)


def test_coca_loss_dual_mode():
    """CoCaLoss accepts either (logits, labels) or precomputed caption CE/z components."""
    from open_clip.loss import CoCaLoss
    torch.manual_seed(0)
    z_weight = 1e-4
    loss_fn = CoCaLoss(
        caption_loss_weight=2.0, clip_loss_weight=1.0, pad_id=None,
        z_loss_weight=z_weight,
    )
    b, seq_len, v, d = 4, 9, 32, 8
    img_f = F.normalize(torch.randn(b, d), dim=-1)
    txt_f = F.normalize(torch.randn(b, d), dim=-1)
    logits = torch.randn(b, seq_len, v)
    labels = torch.randint(0, v, (b, seq_len))
    scale = torch.tensor(10.0)

    legacy = loss_fn(
        img_f, txt_f, logits=logits, labels=labels, logit_scale=scale,
        output_dict=True, return_components=True)
    ce, z = caption_cross_entropy(logits, labels, z_loss=True)
    fused = loss_fn(
        img_f, txt_f, caption_loss_ce=ce, caption_loss_z=z,
        logit_scale=scale, output_dict=True, return_components=True)
    torch.testing.assert_close(legacy['caption_loss'], fused['caption_loss'])
    torch.testing.assert_close(legacy['contrastive_loss'], fused['contrastive_loss'])
    torch.testing.assert_close(legacy['caption_loss_ce'], fused['caption_loss_ce'])
    torch.testing.assert_close(legacy['caption_loss_z'], fused['caption_loss_z'])
    # legacy positional call order still works
    positional = loss_fn(img_f, txt_f, logits, labels, scale, output_dict=True)
    torch.testing.assert_close(positional['caption_loss'], legacy['caption_loss'])
