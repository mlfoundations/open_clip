import logging

_logger = logging.getLogger(__name__)
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from ..model_traits import unwrap_model
from ..utils import move_to_device

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy
    from ..naflex_config import NaFlexDataConfig


def get_model_from_task(task_or_model: nn.Module) -> nn.Module:
    """Extract the raw model from a task, compiled task, or plain model.

    Unwraps torch.compile / DDP wrappers, then if the result has a
    ``trainable_module`` attribute (i.e. it's a task), unwraps that too.
    """
    unwrapped = unwrap_model(task_or_model)
    if hasattr(unwrapped, 'trainable_module'):
        return unwrap_model(unwrapped.trainable_module)
    return unwrapped


class TrainingTask(nn.Module):
    """Modality-agnostic base for training tasks.

    Wraps a model + (optional) loss into a single nn.Module, providing
    utilities for EMA, DDP, FSDP2, and checkpointing. Does NOT make any
    assumption about the shape or keys of training batches — that contract
    lives in modality-specific subclasses (e.g. ``ImageTextTask``).

    Note: ``state_dict()`` returns ``{'state_dict': ..., 'state_dict_ema': ...}``
    rather than the standard ``nn.Module.state_dict()`` flat dict. Callers should
    use the task methods for checkpoint save/load rather than raw PyTorch APIs.
    """

    trainable_module: nn.Module
    trainable_module_ema: Optional[nn.Module]

    def __init__(
            self,
            model: nn.Module,
            *,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__()
        self.trainable_module = model
        self.trainable_module_ema = None
        self._device = device
        self._dtype = dtype
        self._verbose = verbose
        self._fsdp_enabled = False
        # When True, state dicts squeeze [1] scalars back to 0-D for checkpoint
        # compatibility with non-FSDP models (e.g. logit_scale, logit_bias).
        self.normalize_checkpoint_scalars = True
        self._compiled_training_forward = None
        self._compiled_eval_forward = None
        # Optional NaFlex data policy (image OR audio); shared with data loaders and dummy batches. NaFlex is a
        # cross-modal data concern, so it lives on the modality-agnostic base, not just the image-text layer.
        self._naflex_data_config = None

    @property
    def naflex_data_config(self) -> Optional["NaFlexDataConfig"]:
        return self._naflex_data_config

    @property
    def naflex_eval_config(self) -> Optional[Tuple[Tuple[int, int], int]]:
        return self._naflex_data_config.eval_config if self._naflex_data_config is not None else None

    def set_naflex_data_config(
            self,
            naflex_data_config: Optional["NaFlexDataConfig"],
    ) -> "TrainingTask":
        """Configure the NaFlex train/eval data policy (image or audio); shared by loaders and dummy batches."""
        self._naflex_data_config = naflex_data_config
        return self

    @staticmethod
    def _compile_kwargs(
            backend: Optional[str] = None,
            mode: Optional[str] = None,
            **compile_kwargs,
    ) -> dict:
        kwargs = {k: v for k, v in compile_kwargs.items() if v is not None}
        if backend is not None:
            kwargs['backend'] = backend
        if mode is not None:
            kwargs['mode'] = mode
        return kwargs

    def compile(
            self,
            *,
            target: str = 'task',
            backend: Optional[str] = None,
            mode: Optional[str] = None,
            compile_train: bool = True,
            compile_eval: bool = True,
            **compile_kwargs,
    ) -> 'TrainingTask':
        """Compile task-owned hot paths without replacing the task object.

        ``target="model"`` compiles only ``trainable_module`` and is intended
        for pre-DDP use. ``target="task"`` compiles the task training/eval
        forward callables so model + loss can be captured while the task's
        public methods remain available to training, eval, and checkpoint code.
        """
        kwargs = self._compile_kwargs(backend=backend, mode=mode, **compile_kwargs)
        if target == 'model':
            self.trainable_module = torch.compile(self.trainable_module, **kwargs)
        elif target == 'blocks':
            # Compile transformer blocks in place; surrounding code (incl. any inline
            # torch.utils.checkpoint call in the trunks) stays eager. Grad-checkpoint recompute
            # is then sequenced block-by-block by eager autograd instead of being traced into
            # the compiled backward, which bounds backward working-set to ~one block. Use when
            # whole-graph compile schedules checkpointed recomputes poorly (torch 2.13 regression).
            blocks = self._get_fsdp_shard_modules()
            if not blocks:
                raise ValueError("compile target 'blocks' found no blocks to compile "
                                 "(see _get_fsdp_shard_modules)")
            model = unwrap_model(self.trainable_module)
            for name, mod in blocks:
                parent_name, _, attr = name.rpartition('.')
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, attr, torch.compile(mod, **kwargs))
        elif target == 'task':
            if compile_train:
                self._compiled_training_forward = torch.compile(self.training_forward, **kwargs)
            if compile_eval:
                self._compiled_eval_forward = torch.compile(self.eval_forward, **kwargs)
        else:
            raise ValueError(f"Unsupported task compile target: {target}")
        return self

    def prepare_batch(
            self,
            batch: dict,
            device: torch.device,
            input_dtype: Optional[torch.dtype] = None,
    ) -> dict:
        """Move batch dict tensors to device with correct dtypes.

        Float tensors get ``input_dtype``; integer tensors stay as-is. Recurses
        into nested dicts (e.g. NaFlex patch dicts under ``"image"``).
        """
        return move_to_device(batch, device, input_dtype)

    def get_trainable_module(self, use_ema: bool = True) -> nn.Module:
        """Get the trainable module, optionally returning EMA version."""
        if use_ema and self.trainable_module_ema is not None:
            return self.trainable_module_ema.module
        return unwrap_model(self.trainable_module)

    def setup_ema(self, decay: float = 0.9999, device: Optional[torch.device] = None) -> 'TrainingTask':
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

    @property
    def primary_key(self) -> str:
        """Primary non-text modality key.

        Eval expects model output dicts to expose ``f"{primary_key}_features"``
        plus ``"text_features"`` for paired retrieval metrics.
        """
        return self.data_keys[0]

    def batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        """Return local batch size for logging/accounting."""
        data_keys = getattr(self, 'data_keys', None)
        key = data_keys[0] if data_keys else next(iter(batch))
        value = batch[key]
        if isinstance(value, dict):
            for preferred in ('patches', 'waveform'):
                tensor = value.get(preferred)
                if isinstance(tensor, torch.Tensor):
                    return tensor.shape[0]
            for tensor in value.values():
                if isinstance(tensor, torch.Tensor):
                    return tensor.shape[0]
            raise ValueError(f"Cannot infer batch size from nested batch key {key!r}.")
        if isinstance(value, torch.Tensor):
            return value.shape[0]
        return len(value)

    def ddp_extra_kwargs(self) -> dict:
        """Additional DistributedDataParallel kwargs required by this task."""
        return {}

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs,
    ) -> 'TrainingTask':
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
        ModernTextBlock, Bottleneck, and timm ViT ``Block`` (timm towers, incl. NaFlexVit
        trunks) instances within the trainable module. Also drives the 'blocks' compile
        strategy, so timm vision trunks get per-block compile rather than staying eager.
        Models can override this by defining a ``fsdp_shard_modules()`` method.
        """
        model = unwrap_model(self.trainable_module)
        if hasattr(model, 'fsdp_shard_modules'):
            return model.fsdp_shard_modules()

        from open_clip.transformer import ResidualAttentionBlock, CustomResidualAttentionBlock, ModernTextBlock
        from open_clip.modified_resnet import Bottleneck

        shard_types = [ResidualAttentionBlock, CustomResidualAttentionBlock, ModernTextBlock, Bottleneck]
        try:
            from timm.models.vision_transformer import Block as TimmViTBlock
            shard_types.append(TimmViTBlock)
        except ImportError:
            pass
        shard_types = tuple(shard_types)

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
            compile_blocks: bool = False,
            compile_kwargs: Optional[dict] = None,
            grad_checkpointing: bool = False,
    ) -> 'TrainingTask':
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
                    _logger.info(f'FSDP2: reshaped 0-D param {name!r} to [1]')

        # Shard discovered submodules (transformer blocks) first
        shard_modules = self._get_fsdp_shard_modules()
        if self._verbose:
            _logger.info(f'FSDP2: sharding {len(shard_modules)} submodules')
        if not shard_modules:
            _logger.warning(
                'FSDP2: no submodules matched default shard types. '
                'Consider defining fsdp_shard_modules() on your model for optimal sharding.'
            )

        # Per-block torch.compile (before AC hooks and FSDP sharding).
        # Required ordering: compile → composable AC → FSDP.
        #
        # Composable AC hooks must be registered on the OptimizedModule wrapper,
        # NOT on the inner block. If hooks are on the inner block, AC recomputation
        # replays inside the compiled graph where FSDP DTensor metadata has changed,
        # causing "Recomputed values have different metadata" errors.
        #
        # With hooks on the OptimizedModule, AC recomputation triggers at the module
        # boundary (outside the compiled graph), which also triggers FSDP all-gather
        # hooks — so parameters are correctly gathered during recompute.
        if compile_blocks:
            if self._verbose:
                _logger.info(f'FSDP2: compiling {len(shard_modules)} blocks with torch.compile')
            if grad_checkpointing:
                from torch.distributed._composable import checkpoint as composable_checkpoint
            compile_kwargs = compile_kwargs or {}
            compiled_modules = []
            for name, mod in shard_modules:
                compiled_mod = torch.compile(mod, **compile_kwargs)
                # Re-register compiled module in parent so FSDP sees it
                parts = name.rsplit('.', 1)
                parent_name, child_name = (parts[0], parts[1]) if len(parts) == 2 else ('', name)
                parent = model.get_submodule(parent_name) if parent_name else model
                parent.register_module(child_name, compiled_mod)
                # Apply composable AC to the compiled wrapper (not the inner block)
                if grad_checkpointing:
                    composable_checkpoint(compiled_mod)
                compiled_modules.append((name, compiled_mod))
            shard_modules = compiled_modules

        for name, mod in shard_modules:
            fully_shard(mod, **fsdp_kwargs)

        # Shard root trainable module (covers remaining params: embeddings, projections, logit_scale)
        fully_shard(self.trainable_module, **fsdp_kwargs)

        # Register encode_text/encode_image as FSDP forward methods so they trigger
        # the same all-gather/reshard hooks as __call__. Without this, direct calls
        # like build_zero_shot_classifier → model.encode_text() fail with DTensor errors.
        from torch.distributed.fsdp import register_fsdp_forward_method
        # Only register methods the model actually defines (e.g. GenLIP has encode_image but no encode_text).
        unwrapped = unwrap_model(self.trainable_module)
        for method_name in ("encode_text", "encode_image"):
            if hasattr(unwrapped, method_name):
                register_fsdp_forward_method(self.trainable_module, method_name)

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

    @staticmethod
    def _strip_compiled_keys(sd: dict) -> dict:
        """Remove ``_orig_mod.`` segments that torch.compile wrappers insert into state_dict keys.

        Handles both the root prefix (compile strategy 'model') and nested per-block segments
        (strategy 'blocks', e.g. ``text.blocks.9._orig_mod.norm1.weight``) so saved checkpoints
        are always clean and portable across compile strategies.
        """
        return {k.replace('._orig_mod.', '.').removeprefix('_orig_mod.'): v for k, v in sd.items()}

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
            sd = {'state_dict': self._strip_compiled_keys(model_sd)}
        else:
            sd = {'state_dict': self._strip_compiled_keys(
                unwrap_model(self.trainable_module).state_dict())}
        if self.trainable_module_ema is not None:
            sd['state_dict_ema'] = self._strip_compiled_keys(
                self.trainable_module_ema.module.state_dict())
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
            # normalize away compile-wrapper segments from the incoming checkpoint, then remap
            # clean names onto the live model's keys when its blocks are compiled in place
            sd = self._strip_compiled_keys(state_dict['state_dict'])
            model_keys = list(model.state_dict().keys())
            if any('_orig_mod' in k for k in model_keys):
                clean_to_model = {
                    k.replace('._orig_mod.', '.').removeprefix('_orig_mod.'): k for k in model_keys
                }
                # unmapped (unexpected) keys keep their name so strict-mode reporting is unchanged
                sd = {clean_to_model.get(k, k): v for k, v in sd.items()}
            sd = self._reconcile_state_dict_shapes(model, sd)
            if self._fsdp_enabled:
                from torch.distributed.checkpoint.state_dict import (
                    set_model_state_dict,
                    StateDictOptions,
                )
                options = StateDictOptions(full_state_dict=True, cpu_offload=True, strict=strict)
                set_model_state_dict(model, sd, options=options)
            else:
                model.load_state_dict(sd, strict=strict)
        if 'state_dict_ema' in state_dict and self.trainable_module_ema is not None:
            self.trainable_module_ema.module.load_state_dict(
                self._strip_compiled_keys(state_dict['state_dict_ema']), strict=strict,
            )

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

    @staticmethod
    def _report(source: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Per-step scalars to LOG (not loss terms): ``logit_scale``, and ``logit_bias`` when present.

        Returned as the second element of ``training_forward``/``compute_accum_loss``'s ``(losses, report)`` so the
        train loop logs them once, instead of smuggling them through the loss dict (where the generic loss meter
        would double-log the scale).
        """
        return {key: source[key] for key in ("logit_scale", "logit_bias") if key in source}

    def concat_accum_features(self, features: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Combine cached and live microbatch outputs; override for outputs with variable sequence lengths."""
        return {key: torch.cat(values) for key, values in features.items()}

    def compute_accum_loss(self, inputs, inputs_no_accum, accum_batches):
        """Compute loss from accumulated features for gradient accumulation.

        Override in subclasses that need to derive training targets from
        raw batch dicts (e.g. autoregressive label creation in CoCa).

        Returns ``(losses, report)`` — see :meth:`_report`.
        """
        losses = self.loss(**inputs, **inputs_no_accum, output_dict=True)
        return losses, self._report(inputs_no_accum)

    def eval_forward(self, batch: Dict[str, torch.Tensor]):
        return self.get_trainable_module(use_ema=True)(**batch)

    def forward(self, *args, **kwargs):
        """Normalize task call conventions, then run train or eval forward."""
        if len(args) == 1 and isinstance(args[0], dict):
            batch = args[0]
            if kwargs:
                batch = {**batch, **kwargs}
        elif args and kwargs:
            batch = dict(zip(self.data_keys, args))
            batch.update(kwargs)
        elif args:
            batch = dict(zip(self.data_keys, args))
        else:
            batch = kwargs

        if not self.training:
            forward_fn = self._compiled_eval_forward or self.eval_forward
        else:
            forward_fn = self._compiled_training_forward or self.training_forward
        return forward_fn(batch)

    def training_forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        """Run the train-time forward. Subclasses return ``(losses, report)`` — see :meth:`_report`."""
        raise NotImplementedError
