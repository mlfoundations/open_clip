"""Exponential moving average for FSDP2-sharded models."""
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


def _clean_name(name: str) -> str:
    return name.replace('._orig_mod.', '.').removeprefix('_orig_mod.')


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


class FSDPModelEma(nn.Module):
    """Keep an EMA of an FSDP2 model in a second module sharded the same way.

    Mirrors timm's ``ModelEmaV3``: same decay schedule and arguments, ``module``, ``get_decay()``, ``update()``, and
    ``set()``. Each rank holds 1/N of the EMA and updates its own shards without communication. ``module`` is a normal
    FSDP root, so evaluating it is a collective like any FSDP forward. It can't be created by copying a sharded model;
    pass a copy that has already been passed through ``fully_shard`` (see ``TrainingTask.prepare_fsdp``).
    """

    def __init__(
            self,
            module: nn.Module,
            decay: float = 0.9999,
            min_decay: float = 0.0,
            update_after_step: int = 0,
            use_warmup: bool = False,
            warmup_gamma: float = 1.0,
            warmup_power: float = 2 / 3,
    ):
        super().__init__()
        self.module = module
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.module.eval()
        self.module.requires_grad_(False)
        # Non-persistent buffers are derived, not learned: lerping the -inf entries of an attn_mask would give NaN.
        persistent = {_clean_name(k) for k in self.module.state_dict()}
        self._buffer_names = [n for n, _ in self._named_tensors(self.module.named_buffers()) if n in persistent]
        self._init_fsdp()

    @staticmethod
    def _named_tensors(named: Iterable[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
        return [(_clean_name(n), t) for n, t in named]

    @torch.no_grad()
    def _init_fsdp(self):
        """Run FSDP's lazy init now and allocate its unsharded storage.

        The first forward would otherwise happen inside eval's ``inference_mode``. FSDP would create the sharded
        params and unsharded storage as inference tensors, and the next ``update()`` would fail writing to them.
        """
        from torch.distributed.fsdp import FSDPModule
        if not isinstance(self.module, FSDPModule):
            return
        self.module._get_fsdp_state()._lazy_init()
        for mod in self.module.modules():
            if isinstance(mod, FSDPModule):
                mod.unshard()
                mod.reshard()

    def _param_pairs(self, model: nn.Module) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # named_parameters() drops aliases, so tied params are updated once. Pair by name since the trained model's
        # blocks may be wrapped by torch.compile while the EMA's are not.
        model_params = dict(self._named_tensors(model.named_parameters()))
        ema_params = self._named_tensors(self.module.named_parameters())
        return [_to_local(p) for _, p in ema_params], [_to_local(model_params[n]) for n, _ in ema_params]

    def _buffer_pairs(self, model: nn.Module) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        model_buffers = dict(self._named_tensors(model.named_buffers()))
        ema_buffers = dict(self._named_tensors(self.module.named_buffers()))
        return [(ema_buffers[n], model_buffers[n]) for n in self._buffer_names]

    def get_decay(self, step: Optional[int] = None) -> float:
        """Decay for update number ``step``, matching timm's ``ModelEmaV3``. Constant ``decay`` if no step is given."""
        if step is None:
            return self.decay

        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            return max(min(decay, self.decay), self.min_decay)
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module, step: Optional[int] = None):
        """Update from ``model``, which must be sharded (true after backward unless reshard_after_backward is off)."""
        decay = self.get_decay(step)
        torch._foreach_lerp_(*self._param_pairs(model), weight=1. - decay)
        for ema_buffer, model_buffer in self._buffer_pairs(model):
            if ema_buffer.is_floating_point():
                ema_buffer.lerp_(model_buffer, 1. - decay)
            else:
                ema_buffer.copy_(model_buffer)

    @torch.no_grad()
    def set(self, model: nn.Module):
        """Copy ``model``'s weights into the EMA."""
        torch._foreach_copy_(*self._param_pairs(model))
        for ema_buffer, model_buffer in self._buffer_pairs(model):
            ema_buffer.copy_(model_buffer)
