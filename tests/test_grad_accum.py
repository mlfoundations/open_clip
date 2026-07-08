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
from open_clip.task import CLIPTask
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
