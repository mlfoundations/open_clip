"""Gradient-accumulation equivalence (github.com/mlfoundations/open_clip issue #761).

open_clip's accumulation is the cached-feature scheme: every accumulation step computes the loss
over the FULL effective batch with only that step's microbatch features live. The per-step backward
therefore yields one term of the chain-rule partition d(L_full)/dtheta = sum_j (dL/df_j)(df_j/dtheta),
and the accumulated gradient must EQUAL the single full-batch gradient -- no 1/accum_freq scaling
(adding one, as #761 proposed, would be wrong). logit_scale/logit_bias are the exception: they are
live on every step, so without the detach fix their gradient is over-counted by accum_freq.
"""
import contextlib
import types

import pytest
import torch

from open_clip.model import CLIP
from open_clip.coca_model import CoCa
from open_clip.mammut_model import MaMMUT
from open_clip.loss import CoCaLoss
from open_clip.task import CLIPTask, CoCaTask
from open_clip_train.train import _train_step_eager

ACCUM_FREQ = 4
MICRO_BS = 2


def _make_args(accum_freq=1):
    return types.SimpleNamespace(
        accum_freq=accum_freq,
        grad_clip_norm=None,
        naflex_loss_scale='none',
        batch_size=MICRO_BS,
    )


def _tiny_task():
    torch.manual_seed(0)
    model = CLIP(
        embed_dim=32,
        vision_cfg=dict(image_size=32, layers=2, width=64, patch_size=16),
        text_cfg=dict(context_length=8, vocab_size=64, width=32, heads=2, layers=2),
        output_dict=True,
    ).double()
    task = CLIPTask(model, verbose=False)
    task.train()
    return task, model


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_grad_accum_matches_full_batch():
    task, model = _tiny_task()
    # lr=0 optimizer: _train_step_eager steps internally; keep params fixed so grads are comparable
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    torch.manual_seed(1)
    images = torch.randn(ACCUM_FREQ * MICRO_BS, 3, 32, 32, dtype=torch.float64)
    texts = torch.randint(1, 60, (ACCUM_FREQ * MICRO_BS, 8))

    # reference: single full-batch backward
    optimizer.zero_grad()
    losses, _ = task.training_forward({'image': images, 'text': texts})
    losses['loss'].backward()
    grads_full = _grads(model)

    # accumulation through the real train-step code
    optimizer.zero_grad()
    args = _make_args(accum_freq=ACCUM_FREQ)
    accum_state = ([], {})
    result = None
    for j in range(ACCUM_FREQ):
        batch = {
            'image': images[j * MICRO_BS:(j + 1) * MICRO_BS],
            'text': texts[j * MICRO_BS:(j + 1) * MICRO_BS],
        }
        result = _train_step_eager(
            task, batch, accum_state, optimizer, scaler=None,
            autocast=contextlib.nullcontext, args=args,
        )
        if result is not None:
            accum_state = result[-1]
    assert result is not None, 'final accumulation step must produce a result'
    grads_accum = _grads(model)

    assert grads_full.keys() == grads_accum.keys()
    for name in grads_full:
        g_full, g_accum = grads_full[name], grads_accum[name]
        ratio = (g_accum.norm() / g_full.norm()).item()
        # the whole point of #761: NO 1/accum_freq scaling is needed -- gradients must match 1:1,
        # including logit_scale (over-counted by exactly accum_freq before the detach fix)
        assert torch.allclose(g_accum, g_full, rtol=1e-9, atol=1e-12), (
            f'{name}: accumulated gradient diverges from full-batch gradient (norm ratio {ratio:.4f})'
        )


def test_grad_accum_no_accum_path_unchanged():
    """accum_freq=1 goes through the direct path and produces the same grads as a plain backward."""
    task, model = _tiny_task()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    torch.manual_seed(2)
    batch = {
        'image': torch.randn(MICRO_BS, 3, 32, 32, dtype=torch.float64),
        'text': torch.randint(1, 60, (MICRO_BS, 8)),
    }

    optimizer.zero_grad()
    losses, _ = task.training_forward(batch)
    losses['loss'].backward()
    grads_ref = _grads(model)

    optimizer.zero_grad()
    result = _train_step_eager(
        task, batch, None, optimizer, scaler=None,
        autocast=contextlib.nullcontext, args=_make_args(accum_freq=1),
    )
    assert result is not None
    grads_step = _grads(model)
    for name in grads_ref:
        assert torch.allclose(grads_step[name], grads_ref[name], rtol=1e-9, atol=1e-12), name


def _tiny_caption_task(arch, z_loss_weight):
    torch.manual_seed(0)
    vision_cfg = dict(image_size=32, layers=1, width=32, head_width=16, patch_size=16, output_tokens=True)
    text_cfg = dict(context_length=12, vocab_size=64, width=32, heads=2, layers=1, pad_id=63, variable_text=True)
    if arch == 'coca':
        model = CoCa(
            embed_dim=32, vision_cfg=vision_cfg,
            text_cfg=dict(text_cfg, text_arch='modern', pool_type='mean', output_tokens=True),
            multimodal_cfg=dict(text_cfg, text_arch='modern'),
        )
    else:
        model = MaMMUT(
            embed_dim=32, vision_cfg=vision_cfg,
            multimodal_cfg=dict(text_cfg, text_arch='modern' if arch == 'mammut-modern' else 'clip'),
        )
    task = CoCaTask(
        model.double(), caption_loss_weight=0.7, clip_loss_weight=1.3,
        caption_z_loss_weight=z_loss_weight, caption_loss_compute_dtype='model', verbose=False,
    )
    task.train()
    return task, model


def _variable_caption_batches(mixed_masks=False):
    torch.manual_seed(1)
    # Unequal microbatch sizes, padded lengths, and valid-token counts catch averaging each
    # microbatch's caption mean equally instead of weighting by the effective batch's valid tokens.
    lengths = torch.tensor([5, 3, 9, 7, 2])
    valid = torch.arange(9).unsqueeze(0) < lengths.unsqueeze(1)
    full = {
        'image': torch.randn(5, 3, 32, 32, dtype=torch.float64),
        'text': torch.randint(1, 63, (5, 9)).masked_fill(~valid, 63),
    }
    if mixed_masks:
        full['text_valid'] = valid
        full['text'][0, 1] = 63  # a real target equal to pad_id: only the explicit mask can preserve it
    batches = []
    for start, end, seq_len in ((0, 2, 5), (2, 3, 9), (3, 5, 7)):
        batch = {'image': full['image'][start:end], 'text': full['text'][start:end, :seq_len]}
        if mixed_masks and start == 0:
            batch['text_valid'] = valid[start:end, :seq_len]
        batches.append(batch)
    return full, batches


@pytest.mark.parametrize('arch', ['coca', 'mammut', 'mammut-modern'])
@pytest.mark.parametrize('mixed_masks', [False, True])
@pytest.mark.parametrize('z_loss_weight', [0.0, 1e-4])
def test_caption_variable_text_accum_matches_full_batch(arch, mixed_masks, z_loss_weight):
    task, model = _tiny_caption_task(arch, z_loss_weight)
    full, batches = _variable_caption_batches(mixed_masks)
    losses_full, _ = task.training_forward(full)
    losses_full['loss'].backward()
    grads_full = _grads(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    optimizer.zero_grad()
    forward_lengths = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: forward_lengths.append(kwargs['text'].shape[1]), with_kwargs=True,
    )
    accum_state = ([], {})
    try:
        for batch in batches:
            result = _train_step_eager(
                task, batch, accum_state, optimizer, scaler=None,
                autocast=contextlib.nullcontext, args=_make_args(accum_freq=len(batches)),
            )
    finally:
        hook.remove()
    assert result is not None
    losses_accum = result[0]
    for key in ('contrastive_loss', 'caption_loss', 'loss'):
        torch.testing.assert_close(losses_accum[key], losses_full[key], rtol=1e-6, atol=1e-7)
    grads_accum = _grads(model)
    assert grads_accum.keys() == grads_full.keys()
    for name, grad in grads_full.items():
        torch.testing.assert_close(grads_accum[name], grad, rtol=1e-6, atol=1e-8, msg=name)
    # Padding is limited to the loss inputs, preserving the shorter model forwards on cache and replay.
    assert forward_lengths == [5, 9, 7, 5, 9, 7]


@pytest.mark.parametrize('arch', ['coca', 'mammut', 'mammut-modern'])
@pytest.mark.parametrize('z_loss_weight', [0.0, 1e-4])
def test_legacy_caption_variable_text_accum_matches_full_batch(arch, z_loss_weight):
    from open_clip_train.legacy_train import train_one_epoch

    task, model = _tiny_caption_task(arch, z_loss_weight)
    full, batches = _variable_caption_batches()
    losses_full, _ = task.training_forward(full)
    losses_full['loss'].backward()
    grads_full = _grads(model)

    class Loader(list):
        num_batches = len(batches)
        num_samples = len(full['text'])

    data = {'train': types.SimpleNamespace(dataloader=Loader(batches), set_epoch=lambda epoch: None)}
    args = types.SimpleNamespace(
        device='cpu', precision='fp32', distill=False, accum_freq=len(batches), skip_scheduler=True,
        grad_clip_norm=None, rank=0, world_size=1, batch_size=MICRO_BS, log_every_n_steps=1, wandb=False,
    )
    loss = CoCaLoss(
        caption_loss_weight=0.7, clip_loss_weight=1.3, pad_id=model.pad_id,
        z_loss_weight=z_loss_weight, compute_dtype='model',
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    optimizer.zero_grad()
    train_one_epoch(model, data, loss, 0, optimizer, None, None, None, args)
    grads_accum = _grads(model)
    assert grads_accum.keys() == grads_full.keys()
    for name, grad in grads_full.items():
        torch.testing.assert_close(grads_accum[name], grad, rtol=1e-6, atol=1e-8, msg=name)
