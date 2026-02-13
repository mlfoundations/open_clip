import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap model from DDP and/or torch.compile wrappers."""
    unwrapped = model
    if hasattr(unwrapped, 'module'):
        unwrapped = unwrapped.module
    if hasattr(unwrapped, '_orig_mod'):
        unwrapped = unwrapped._orig_mod
    return unwrapped


def get_model_from_task(task_or_model: nn.Module) -> nn.Module:
    """Extract the raw model from a task or model.

    Used by evaluate() and zero_shot_eval() to avoid duplicated duck-typing.
    """
    if isinstance(task_or_model, CLIPTrainingTask):
        return task_or_model.get_trainable_module(use_ema=False)
    return unwrap_model(task_or_model)


class CLIPTrainingTask(nn.Module):
    """Base class for CLIP training tasks.

    Wraps model + loss into a single nn.Module, providing utilities for
    EMA, DDP, FSDP2, checkpointing, and logit scale clamping.

    Note: ``state_dict()`` returns ``{'state_dict': ..., 'state_dict_ema': ...}``
    rather than the standard ``nn.Module.state_dict()`` flat dict. Callers should
    use the task methods for checkpoint save/load rather than raw PyTorch APIs.
    """

    trainable_module: nn.Module
    trainable_module_ema: Optional[nn.Module]

    def __init__(
            self,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__()
        self.trainable_module_ema = None
        self._device = device
        self._dtype = dtype
        self._verbose = verbose
        self._fsdp_enabled = False
        # When True, state dicts squeeze [1] scalars back to 0-D for checkpoint
        # compatibility with non-FSDP models (e.g. logit_scale, logit_bias).
        self.normalize_checkpoint_scalars = True

    def get_trainable_module(self, use_ema: bool = True) -> nn.Module:
        """Get the trainable module, optionally returning EMA version."""
        if use_ema and self.trainable_module_ema is not None:
            return self.trainable_module_ema.module
        return unwrap_model(self.trainable_module)

    def setup_ema(self, decay: float = 0.9999, device: Optional[torch.device] = None) -> 'CLIPTrainingTask':
        """Set up exponential moving average for the trainable module."""
        assert not self._fsdp_enabled, (
            'EMA must be set up before prepare_fsdp(). '
            'FSDP2 sharded parameters are not compatible with ModelEmaV3.'
        )
        from timm.utils import ModelEmaV3
        self.trainable_module_ema = ModelEmaV3(
            unwrap_model(self.trainable_module),
            decay=decay,
            device=device,
        )
        return self

    def update_ema(self, step: Optional[int] = None):
        """Update EMA weights."""
        if self.trainable_module_ema is not None:
            self.trainable_module_ema.update(unwrap_model(self.trainable_module), step=step)

    @property
    def has_ema(self) -> bool:
        return self.trainable_module_ema is not None

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs,
    ) -> 'CLIPTrainingTask':
        """Wrap trainable_module with DistributedDataParallel."""
        assert not self._fsdp_enabled, \
            'Cannot wrap FSDP2-sharded module with DDP. Use --fsdp OR DDP, not both.'
        self.trainable_module = torch.nn.parallel.DistributedDataParallel(
            self.trainable_module,
            device_ids=device_ids,
            **ddp_kwargs,
        )
        return self

    def _get_fsdp_shard_modules(self) -> List[Tuple[str, nn.Module]]:
        """Discover modules to shard with FSDP2.

        Default: finds all ResidualAttentionBlock, CustomResidualAttentionBlock,
        and Bottleneck instances within the trainable module. Models can override
        this by defining a ``fsdp_shard_modules()`` method.
        """
        model = unwrap_model(self.trainable_module)
        if hasattr(model, 'fsdp_shard_modules'):
            return model.fsdp_shard_modules()

        from open_clip.transformer import ResidualAttentionBlock, CustomResidualAttentionBlock
        from open_clip.modified_resnet import Bottleneck

        shard_types = (ResidualAttentionBlock, CustomResidualAttentionBlock, Bottleneck)

        modules = []
        for name, mod in model.named_modules():
            if isinstance(mod, shard_types):
                modules.append((name, mod))
        return modules

    def prepare_fsdp(
            self,
            mesh: Optional['DeviceMesh'] = None,
            reshard_after_forward: bool = True,
            mp_policy: Optional['MixedPrecisionPolicy'] = None,
            offload_policy: Optional['CPUOffloadPolicy'] = None,
    ) -> 'CLIPTrainingTask':
        """Apply FSDP2 sharding to trainable module.

        Reshapes any 0-D params to 1-D (FSDP2 requires dim-0 for sharding),
        then applies fully_shard() to discovered submodules and the root.
        """
        assert not isinstance(self.trainable_module, torch.nn.parallel.DistributedDataParallel), \
            'Cannot apply FSDP2 to a DDP-wrapped module. Use --fsdp OR DDP, not both.'
        from torch.distributed._composable.fsdp import fully_shard

        fsdp_kwargs = dict(
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
        )
        if mp_policy is not None:
            fsdp_kwargs['mp_policy'] = mp_policy
        if offload_policy is not None:
            fsdp_kwargs['offload_policy'] = offload_policy

        # Reshape any 0-D (scalar) params to 1-D so FSDP2 can shard them.
        # This affects logit_scale (always present) and logit_bias (SigLIP only).
        # Both are accessed via .exp() or multiplication, which broadcast
        # identically on [1] tensors.
        model = unwrap_model(self.trainable_module)
        for name, param in list(model.named_parameters()):
            if param.ndim == 0:
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    attr_name = parts[1]
                else:
                    parent = model
                    attr_name = parts[0]
                setattr(parent, attr_name, nn.Parameter(
                    param.data.unsqueeze(0), requires_grad=param.requires_grad,
                ))
                if self._verbose:
                    logging.info(f'FSDP2: reshaped 0-D param {name!r} to [1]')

        # Shard discovered submodules (transformer blocks) first
        shard_modules = self._get_fsdp_shard_modules()
        if self._verbose:
            logging.info(f'FSDP2: sharding {len(shard_modules)} submodules')
        if not shard_modules:
            logging.warning(
                'FSDP2: no submodules matched default shard types. '
                'Consider defining fsdp_shard_modules() on your model for optimal sharding.'
            )
        for name, mod in shard_modules:
            fully_shard(mod, **fsdp_kwargs)

        # Shard root trainable module (covers remaining params: embeddings, projections, logit_scale)
        fully_shard(self.trainable_module, **fsdp_kwargs)
        self._fsdp_enabled = True
        return self

    @staticmethod
    def _normalize_scalar_params(sd: dict) -> dict:
        """Squeeze [1] tensors back to 0-D scalars for checkpoint compatibility.

        FSDP2 requires 1-D params but standard checkpoints store logit_scale
        and logit_bias as 0-D scalars. This normalizes the state dict so saved
        checkpoints are loadable by non-FSDP models without reconciliation.
        """
        for key, val in sd.items():
            if val.ndim == 1 and val.shape[0] == 1:
                sd[key] = val.squeeze(0)
        return sd

    def state_dict(self, *args, **kwargs) -> dict:
        """Return state dict with both main and EMA weights."""
        if self._fsdp_enabled:
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict,
                StateDictOptions,
            )
            # full_state_dict=True gathers shards into full tensors
            # cpu_offload=True to avoid GPU OOM during gather
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            model_sd = get_model_state_dict(
                unwrap_model(self.trainable_module), options=options,
            )
            if self.normalize_checkpoint_scalars:
                model_sd = self._normalize_scalar_params(model_sd)
            sd = {'state_dict': model_sd}
        else:
            sd = {'state_dict': unwrap_model(self.trainable_module).state_dict()}
        if self.trainable_module_ema is not None:
            sd['state_dict_ema'] = self.trainable_module_ema.module.state_dict()
        return sd

    @staticmethod
    def _reconcile_state_dict_shapes(
            model: nn.Module,
            sd: dict,
    ) -> dict:
        """Reconcile 0-D vs 1-D tensor shapes between checkpoint and model.

        When crossing the FSDP/non-FSDP boundary, logit_scale and logit_bias
        may differ in shape (0-D scalar vs 1-D [1]). This reshapes checkpoint
        tensors to match the model's current parameter shapes.
        """
        model_sd = {k: p for k, p in model.named_parameters()}
        for key, val in sd.items():
            if key in model_sd:
                param = model_sd[key]
                if val.ndim == 0 and param.ndim == 1 and param.shape[0] == 1:
                    sd[key] = val.unsqueeze(0)
                elif val.ndim == 1 and val.shape[0] == 1 and param.ndim == 0:
                    sd[key] = val.squeeze(0)
        return sd

    def load_state_dict(self, state_dict: dict, strict: bool = True, **kwargs):
        """Load state dict for both main and EMA weights."""
        if 'state_dict' in state_dict:
            model = unwrap_model(self.trainable_module)
            sd = self._reconcile_state_dict_shapes(model, state_dict['state_dict'])
            if self._fsdp_enabled:
                from torch.distributed.checkpoint.state_dict import (
                    set_model_state_dict,
                    StateDictOptions,
                )
                options = StateDictOptions(strict=strict)
                set_model_state_dict(model, sd, options=options)
            else:
                model.load_state_dict(sd, strict=strict)
        if 'state_dict_ema' in state_dict and self.trainable_module_ema is not None:
            self.trainable_module_ema.module.load_state_dict(
                state_dict['state_dict_ema'], strict=strict,
            )

    def get_optim_state_dict(self, optimizer: torch.optim.Optimizer) -> dict:
        """Get optimizer state dict, handling FSDP2 sharded state."""
        if self._fsdp_enabled:
            from torch.distributed.checkpoint.state_dict import (
                get_optimizer_state_dict,
                StateDictOptions,
            )
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            return get_optimizer_state_dict(
                unwrap_model(self.trainable_module), optimizer, options=options,
            )
        return optimizer.state_dict()

    def load_optim_state_dict(self, optimizer: torch.optim.Optimizer, state_dict: dict):
        """Load optimizer state dict, handling FSDP2 sharded state."""
        if self._fsdp_enabled:
            from torch.distributed.checkpoint.state_dict import (
                set_optimizer_state_dict,
                StateDictOptions,
            )
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            set_optimizer_state_dict(
                unwrap_model(self.trainable_module),
                optimizer,
                optim_state_dict=state_dict,
                options=options,
            )
        else:
            optimizer.load_state_dict(state_dict)

    def state_dict_for_inference(self) -> dict:
        """Return state dict for inference (prefers EMA if available)."""
        if self.trainable_module_ema is not None:
            return self.trainable_module_ema.module.state_dict()
        if self._fsdp_enabled:
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict,
                StateDictOptions,
            )
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            sd = get_model_state_dict(
                unwrap_model(self.trainable_module), options=options,
            )
            if self.normalize_checkpoint_scalars:
                sd = self._normalize_scalar_params(sd)
            return sd
        return unwrap_model(self.trainable_module).state_dict()

    def clamp_logit_scale(self, max_val: float = math.log(100)):
        """Clamp logit_scale parameter to [0, max_val].

        With FSDP2, logit_scale is a replicated DTensor. In-place clamp_
        dispatches to the local tensor on each rank, which is correct for
        a single-element replicated param.
        """
        model = unwrap_model(self.trainable_module)
        if hasattr(model, 'logit_scale'):
            with torch.no_grad():
                model.logit_scale.clamp_(0, max_val)

    def compute_accum_loss(self, inputs, inputs_no_accum, accum_texts):
        """Compute loss from accumulated features for gradient accumulation.

        Override in subclasses that need to derive training targets from
        raw text tokens (e.g. autoregressive label creation in CoCa).
        """
        return self.loss(**inputs, **inputs_no_accum, output_dict=True)

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
