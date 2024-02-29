import logging
from functools import partial
from typing import Optional

import numpy as np
from torch import optim


def _assign_lrs(
    optimizer: optim.Optimizer, groups2ids: dict[int, int], newlrs: dict[int, float]
) -> None:
    for i, pgroup in enumerate(optimizer.param_groups):
        lrid = groups2ids[i]
        newlr = newlrs[lrid]
        pgroup['lr'] = newlr


def _map_param_groups_to_lrs(
    optimizer: optim.Optimizer, baselr: float
) -> tuple[dict[int, int], dict[int, float]]:

    _lr_counter = 0
    lrs2ids = {}
    groups2lrids = {}
    for i, pgroup in enumerate(optimizer.param_groups):
        lr = pgroup.get('lr', baselr)
        if lr in lrs2ids:
            lrid = lrs2ids[lr]
        else:
            lrid = _lr_counter
            lrs2ids[lr] = lrid
            _lr_counter += 1
        groups2lrids[i] = lrid

    return groups2lrids, {v: k for k, v in lrs2ids.items()}


def _warmup_lr(baselr: float, step: int, warmup_steps: int) -> float:
    return baselr * (step + 1) / warmup_steps


def _constant_lr(baselr: float, step: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return _warmup_lr(baselr, warmup_steps, step)
    return baselr


def _constant_lr_with_cooldown(
    baselr: float,
    step: int,
    warmup_steps: int,
    total_steps: int,
    cooldown_steps: int,
    cooldown_power: float = 1.0,
    cooldown_end_lr: float = 0.0,
) -> float:

    start_cooldown_step = total_steps - cooldown_steps
    if step < warmup_steps:
        return _warmup_lr(baselr, warmup_steps, step)

    if step < start_cooldown_step:
        return baselr

    e = step - start_cooldown_step
    es = total_steps - start_cooldown_step
    # linear decay if power == 1; polynomial decay otherwise;
    decay = (1 - (e / es)) ** cooldown_power
    return decay * (baselr - cooldown_end_lr) + cooldown_end_lr


def _cosine_lr(baselr: float, step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return _warmup_lr(baselr, warmup_steps, step)
    e = step - warmup_steps
    es = total_steps - warmup_steps
    return 0.5 * (1 + np.cos(np.pi * e / es)) * baselr


def create_scheduler(
    optimizer: optim.Optimizer,
    baselr: float,
    warmup_steps: int,
    total_steps: int,
    cooldown_steps: Optional[int] = None,
    cooldown_power: float = 1.0,
    cooldown_end_lr: float = 0.0,
    scheduler_type: str = 'cosine',
):
    if scheduler_type == 'cosine':
        _f_scheduler = partial(
            _cosine_lr, warmup_steps=warmup_steps, total_steps=total_steps
        )
    elif scheduler_type == 'const':
        _f_scheduler = partial(
            _constant_lr, warmup_steps=warmup_steps,
        )
    elif scheduler_type == 'const-cooldown':
        assert cooldown_steps is not None
        _f_scheduler = partial(
            _constant_lr_with_cooldown,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            cooldown_steps=cooldown_steps,
            cooldown_power=cooldown_power,
            cooldown_end_lr=cooldown_end_lr,
        )
    else:
        logging.error(
            f'Unknown scheduler, {scheduler_type}. Available options are: cosine, '
            f'const, const-cooldown'
        )
        exit(1)

    groups2lrids, lrids2lrs = _map_param_groups_to_lrs(optimizer, baselr=baselr)
    _lr_scheduler_funcs = {
        _id: partial(_f_scheduler, baselr=lr)
        for _id, lr in lrids2lrs.items()
    }

    def _lr_scheduler(step: int):
        _new_lrs = {
            _id: _lr_scheduler_func(step=step)
            for _id, _lr_scheduler_func in _lr_scheduler_funcs.items()
        }
        _assign_lrs(optimizer, groups2ids=groups2lrids, newlrs=_new_lrs)

    return _lr_scheduler
