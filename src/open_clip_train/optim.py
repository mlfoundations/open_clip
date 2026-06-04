"""Optimizer construction for open_clip training.

Single home for: weight-decay filters, param-group builders (a plain weight-decay split and layer-wise LR decay),
and the builtin-torch-vs-timm creation dispatch. Layer-wise LR decay (LLRD) reuses each text tower's
``layer_groups()`` enumeration -- the same model-specific partition that ``lock`` consumes for freezing -- so the
per-layer ordering lives in exactly one place per model.
"""

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import nn, optim

from open_clip.task import unwrap_model
from open_clip.transformer import _text_layer_groups
from open_clip_train.scheduler import assign_learning_rate, tensorize_learning_rate

_logger = logging.getLogger(__name__)


@dataclass
class OptimizerCfg:
    """Typed optimizer configuration, decoupling :func:`create_optimizer` from the loose training ``args``.

    The caller (``main``) maps CLI args onto these fields once; everything downstream is typed.

    Attributes:
        opt: optimizer name -- ``"adamw"``, ``"nadamw"`` (builtin torch NAdam with decoupled weight decay), or
            any timm optimizer as ``"timm/{name}"``.
        lr: base learning rate.
        weight_decay: weight decay for decayed param groups.
        beta1, beta2: Adam betas (both set, or both None).
        eps: optimizer epsilon (None -> optimizer default).
        momentum: momentum (timm optimizers only; None -> default).
        opt_kwargs: extra optimizer kwargs, merged last.
        text_layer_decay: per-depth LR decay for the text tower (None/1.0 -> off).
        image_layer_decay: per-depth LR decay for the image tower / ``model.visual`` (None/1.0 -> off).
        audio_layer_decay: per-depth LR decay for the audio tower / ``model.audio`` (None/1.0 -> off).
        pooler_in_head: text pooler placement for layer-wise LR decay / lock (see ``layer_groups``).
        wd_exclude_patterns: extra glob patterns (matched against full param names) whose params skip weight decay,
            on top of the 1-D rule and the model's ``no_weight_decay()``.
        fallback_list: param-name glob patterns routed to a hybrid optimizer's fallback (Muon-family timm opts
            only, e.g. ``timm/nadamuon``): matched params are flagged ``use_fallback=True`` so they use the AdamW
            fallback instead of Muon. Not valid for torch optimizers. The fallback LR scale and other
            optimizer-specific fallback settings go through ``opt_kwargs`` (e.g. ``fallback_lr_scale=0.5``).
    """

    opt: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 0.2
    beta1: Optional[float] = None
    beta2: Optional[float] = None
    eps: Optional[float] = None
    momentum: Optional[float] = None
    opt_kwargs: dict = field(default_factory=dict)
    text_layer_decay: Optional[float] = None
    image_layer_decay: Optional[float] = None
    audio_layer_decay: Optional[float] = None
    pooler_in_head: bool = True
    wd_exclude_patterns: List[str] = field(default_factory=list)
    fallback_list: Optional[List[str]] = None


def exclude_from_wd(name: str, param: nn.Parameter) -> bool:
    """Base no-weight-decay rule: any 1-D parameter.

    Dimensionality is the robust signal -- it covers biases, every kind of norm scale/gain, and scalars like
    ``logit_scale`` -- without the brittle substring matching (``"bn"``/``"ln"``/``"bias" in name``) it replaces.
    Non-1-D parameters that should still skip decay (e.g. position/class embeddings) are caught by the model's
    ``no_weight_decay()`` set in :func:`make_wd_exclude`.
    """
    return param.ndim <= 1


def collect_no_weight_decay(model: nn.Module) -> "tuple[set, list]":
    """Recursively gather declared no-weight-decay info from the model and every submodule.

    Returns ``(names, patterns)`` -- two scoped signals, both lined up against ``model.named_parameters()``:

    * ``names`` -- exact parameter names from each submodule's ``no_weight_decay()``, prefixed by that submodule's
      path (matched later by exact equality; e.g. ``positional_embedding``, ``trunk.pos_embed``).
    * ``patterns`` -- glob patterns (matched later with :func:`fnmatch.fnmatchcase`), each scoped to the declaring
      submodule, from two sources:

        - ``no_weight_decay_patterns()`` (the preferred convention): globs relative to the module, prefixed by its
          path -- e.g. a tower returning ``"*relative_position_bias_table"`` at path ``audio.encoder`` becomes
          ``"audio.encoder.*relative_position_bias_table"``.
        - ``no_weight_decay_keywords()`` (legacy timm/Swin compat): each keyword ``kw`` is converted to the
          **suffix-anchored, submodule-scoped** glob ``f"{path}.*{kw}"``. Anchoring at the suffix (not substring)
          means a sloppy keyword like ``"bias"`` matches only params *ending* in ``bias`` (already 1-D) rather
          than ``relative_position_bias_table`` mid-name; scoping by path stops a keyword leaking into siblings.

    ``fnmatch``'s ``*`` spans dots, so a scoped pattern bridges the intermediate block path. Walking every
    submodule -- rather than relying on each wrapper to surface its children -- avoids the fragile surfacing
    chains (e.g. ``AudioTower`` forwarded ``no_weight_decay()`` but not the keywords, silently re-enabling decay
    on Swin's relative-position-bias tables). Overlap with wrappers that already aggregate their children
    (``CLIP``/``CLAP``/...) is harmless: identical absolute strings collapse in the set / dedup.
    """
    names: set = set()
    patterns: list = []
    seen: set = set()

    def _add_pattern(pattern: str):
        if pattern not in seen:
            seen.add(pattern)
            patterns.append(pattern)

    for mod_name, module in model.named_modules():
        prefix = f"{mod_name}." if mod_name else ""
        get_names = getattr(module, "no_weight_decay", None)
        if callable(get_names):
            try:
                names |= {prefix + n for n in get_names()}
            except Exception:  # a submodule's hook may assume context we don't have; skip it rather than fail
                pass
        get_patterns = getattr(module, "no_weight_decay_patterns", None)
        if callable(get_patterns):
            try:
                for pat in get_patterns():
                    _add_pattern(f"{prefix}{pat}")
            except Exception:
                pass
        get_keywords = getattr(module, "no_weight_decay_keywords", None)
        if callable(get_keywords):
            try:
                for kw in get_keywords():
                    _add_pattern(f"{prefix}*{kw}")
            except Exception:
                pass
    return names, patterns


def make_wd_exclude(
        model: nn.Module,
        extra_patterns: Optional[Sequence[str]] = None,
) -> Callable[[str, nn.Parameter], bool]:
    """Build a ``(name, param) -> bool`` predicate selecting parameters that skip weight decay.

    Combines the 1-D rule (:func:`exclude_from_wd`), the model's declared exact ``no_weight_decay()`` names, the
    submodule-scoped glob patterns gathered by :func:`collect_no_weight_decay` (from ``no_weight_decay_patterns()``
    and legacy ``no_weight_decay_keywords()``), and the user's ``extra_patterns`` -- the last two share a single
    glob-matching mechanism. User patterns are matched against the full parameter name with
    :func:`fnmatch.fnmatchcase` (e.g. ``"*.bias"``, ``"visual.proj*"``, ``"*pos_embed*"`` -- use ``*`` for
    substring matches since patterns are anchored).
    """
    no_wd_names, declared_patterns = collect_no_weight_decay(model)
    patterns = declared_patterns + list(extra_patterns or [])

    def exclude(name: str, param: nn.Parameter) -> bool:
        if exclude_from_wd(name, param) or (name in no_wd_names):
            return True
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)

    return exclude


def lr_scales_for_groups(layer_groups: List, layer_decay: float) -> Dict[str, float]:
    """Map each named layer group to ``lr_scale = layer_decay ** depth_from_head``, purely by position.

    The grouping order *is* the depth order (input -> output), so the head (last group) gets ``lr_scale = 1.0``
    and each step deeper multiplies by ``layer_decay``. All policy (e.g. pooler placement) is already baked into
    the group ordering returned by the tower's ``layer_groups(pooler_in_head=...)``.

    Args:
        layer_groups: ordered ``[(name, members), ...]`` from a tower's ``layer_groups()``, input -> output.
        layer_decay: per-depth decay factor in (0, 1]; 1.0 disables decay (every group -> 1.0).

    Returns:
        Mapping of group name to lr_scale.
    """
    n = len(layer_groups)
    return {name: layer_decay ** (n - 1 - idx) for idx, (name, _) in enumerate(layer_groups)}


def wd_param_groups(
        model: nn.Module,
        weight_decay: float,
        exclude_fn: Optional[Callable[[str, nn.Parameter], bool]] = None,
        fallback_fn: Optional[Callable[[str], bool]] = None,
) -> List[Dict]:
    """Two-group weight-decay split (no-decay gains/biases/declared vs decayed rest) -- the long-standing default.

    When ``fallback_fn`` is given, each group is further split by it and the matched groups are flagged
    ``use_fallback=True`` so hybrid optimizers (e.g. Muon/nadamuon) route those params to their fallback. With no
    ``fallback_fn`` this is exactly the historical two-group split.
    """
    exclude_fn = exclude_fn or make_wd_exclude(model)
    buckets: Dict[tuple, Dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = exclude_fn(name, p)
        fb = bool(fallback_fn(name)) if fallback_fn else False
        key = (no_wd, fb)
        if key not in buckets:
            group = {
                "params": [],
                "weight_decay": 0.0 if no_wd else weight_decay,
                "group_name": ("no_wd" if no_wd else "wd") + ("/fallback" if fb else ""),
            }
            if fb:
                group["use_fallback"] = True
            buckets[key] = group
        buckets[key]["params"].append(p)
    return list(buckets.values())


def _text_layer_groups_of(model: nn.Module, pooler_in_head: bool):
    """Resolve the text tower's ``layer_groups`` for either packaging: a ``model.text`` tower exposing
    ``layer_groups`` (CustomTextCLIP / CLAP), or a standard CLIP that unpacks the text attributes directly onto
    the model (``token_embedding``/``transformer``/...), handled by the duck-typed :func:`_text_layer_groups`.
    Returns the ordered ``[(name, members), ...]`` or None if no text tower is found.
    """
    text = getattr(model, "text", None)
    if text is not None and hasattr(text, "layer_groups"):
        return text.layer_groups(pooler_in_head)
    if getattr(model, "token_embedding", None) is not None or getattr(model, "transformer", None) is not None:
        return _text_layer_groups(model, pooler_in_head)
    return None


def _tower_layer_groups_of(model: nn.Module, attr: str, pooler_in_head: bool):
    """Resolve a modality tower's ``layer_groups`` (``model.visual`` / ``model.audio``), or None."""
    tower = getattr(model, attr, None)
    if tower is not None and hasattr(tower, "layer_groups"):
        return tower.layer_groups(pooler_in_head)
    return None


def _assign_group_scales(groups: List, layer_decay: float, param_scale: Dict[int, float]):
    """Fill ``param_scale[id(param)] = lr_scale`` for every parameter in the ordered ``groups``.

    ``groups`` is ``[(name, members), ...]`` where members are ``nn.Module`` or raw ``nn.Parameter``.
    """
    scales = lr_scales_for_groups(groups, layer_decay)
    for name, members in groups:
        scale = scales[name]
        for member in members:
            params = [member] if isinstance(member, nn.Parameter) else member.parameters()
            for p in params:
                param_scale[id(p)] = scale


def layer_decay_param_groups(
        model: nn.Module,
        weight_decay: float,
        text_layer_decay: Optional[float] = None,
        image_layer_decay: Optional[float] = None,
        audio_layer_decay: Optional[float] = None,
        pooler_in_head: bool = True,
        exclude_fn: Optional[Callable[[str, nn.Parameter], bool]] = None,
        fallback_fn: Optional[Callable[[str], bool]] = None,
) -> List[Dict]:
    """Build optimizer param groups with layer-wise LR decay applied to the text / image / audio tower(s).

    Each decayed tower's parameters get an ``lr_scale`` from their ``layer_groups()`` depth (head/projection and
    adapter at 1.0, deeper layers geometrically smaller); everything else (an un-decayed tower, ``logit_scale``,
    ...) stays at ``lr_scale = 1.0``. Within each distinct ``lr_scale`` the usual weight-decay split is preserved,
    so groups are keyed by ``(lr_scale, no_wd[, fallback])``.

    Args:
        model: the full model (e.g. ``CLIP``/``CLAP``); the decayed tower(s) must expose ``layer_groups()``.
        weight_decay: weight decay applied to decayed groups.
        text_layer_decay: per-depth decay in (0, 1] for the text tower (``model.text`` or unpacked standard CLIP);
            None or 1.0 to skip.
        image_layer_decay: per-depth decay in (0, 1] for the image tower (``model.visual``); None or 1.0 to skip.
        audio_layer_decay: per-depth decay in (0, 1] for the audio tower (``model.audio``); None or 1.0 to skip.
        pooler_in_head: text pooler placement policy, see the towers' ``layer_groups``.
        exclude_fn: predicate selecting parameters that skip weight decay.
        fallback_fn: predicate selecting parameters routed to a hybrid optimizer's fallback (Muon-family).

    Returns:
        A list of optimizer param-group dicts, each carrying ``params``, ``weight_decay``, ``lr_scale`` and a
        descriptive ``group_name``.

    Raises:
        ValueError: if decay is requested for a tower that does not expose ``layer_groups()``.
    """
    exclude_fn = exclude_fn or make_wd_exclude(model)
    param_scale: Dict[int, float] = {}
    if text_layer_decay:
        groups = _text_layer_groups_of(model, pooler_in_head)
        if groups is None:
            raise ValueError("text layer-wise LR decay requested but the model has no text tower with layer_groups().")
        _assign_group_scales(groups, text_layer_decay, param_scale)
    if image_layer_decay:
        groups = _tower_layer_groups_of(model, "visual", pooler_in_head)
        if groups is None:
            raise ValueError("image layer-wise LR decay requested but model.visual has no layer_groups().")
        _assign_group_scales(groups, image_layer_decay, param_scale)
    if audio_layer_decay:
        groups = _tower_layer_groups_of(model, "audio", pooler_in_head)
        if groups is None:
            raise ValueError("audio layer-wise LR decay requested but model.audio has no layer_groups().")
        _assign_group_scales(groups, audio_layer_decay, param_scale)

    # Bucket every trainable parameter by (lr_scale, no_wd, fallback); params of un-decayed towers fall back to
    # lr_scale 1.0, and fallback-matched groups are flagged so hybrid optimizers (Muon) route them to the fallback.
    buckets: Dict[tuple, Dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scale = param_scale.get(id(p), 1.0)
        no_wd = exclude_fn(name, p)
        fb = bool(fallback_fn(name)) if fallback_fn else False
        key = (round(scale, 8), no_wd, fb)
        if key not in buckets:
            group = {
                "params": [],
                "weight_decay": 0.0 if no_wd else weight_decay,
                "lr_scale": scale,
                "group_name": f"lr_scale={scale:.4g}/{'no_wd' if no_wd else 'wd'}{'/fallback' if fb else ''}",
            }
            if fb:
                group["use_fallback"] = True
            buckets[key] = group
        buckets[key]["params"].append(p)

    # Order groups head -> embeddings (largest lr_scale first) for readable logs / get_learning_rate.
    return sorted(buckets.values(), key=lambda g: (-g["lr_scale"], g["group_name"]))


def describe_param_groups(optimizer: optim.Optimizer) -> str:
    """One-line summary of the optimizer's param groups (count, lr_scale span, param totals) for logging."""
    parts = []
    for g in optimizer.param_groups:
        n = sum(p.numel() for p in g["params"])
        scale = g.get("lr_scale", 1.0)
        parts.append(f"{g.get('group_name', '?')}: {n / 1e6:.2f}M params, lr_scale={scale:.4g}, wd={g['weight_decay']}")
    return f"{len(optimizer.param_groups)} param groups -- " + " | ".join(parts)


def create_optimizer(
        model: nn.Module,
        cfg: OptimizerCfg,
        device: Optional[torch.device] = None,
        tensorize: bool = False,
) -> optim.Optimizer:
    """Create the training optimizer, dispatching between builtin torch and timm and applying optional LLRD.

    Param groups are always pre-built here (via :func:`make_wd_exclude`) and handed to the optimizer, so the full
    no-weight-decay policy and any layer-wise LR decay apply uniformly to both the AdamW and timm paths. The
    two-group weight-decay split remains the default when no LLRD is requested.

    Args:
        model: the trainable module (e.g. ``task.trainable_module``), possibly wrapped by DDP / torch.compile /
            FSDP2 ``fully_shard`` -- it is unwrapped here for structure discovery and clean parameter names.
        cfg: typed optimizer configuration, see :class:`OptimizerCfg`.
        device: target device, used only when ``tensorize`` is True.
        tensorize: convert per-group LR to a device tensor (for step-compile, avoids optimizer recompiles).

    Returns:
        The constructed optimizer.
    """
    # Unwrap DDP / torch.compile so tower discovery and parameter names (used for LLRD grouping and the
    # no-weight-decay / fallback matching) see the real module structure with clean names. DDP keeps the original
    # parameter objects and FSDP2 (fully_shard) shards them in-place on the same modules, so the unwrapped model's
    # parameters ARE the ones the optimizer should own. (FSDP1's flat parameters are unsupported.)
    model = unwrap_model(model)
    opt = cfg.opt.lower()
    weight_decay = cfg.weight_decay
    # Validate, then treat a decay of 1.0 (or None) as "off". A factor outside (0, 1] would produce negative or
    # growing per-layer LRs once scaled, so reject it up front (0.0 and out-of-range raise; use 1.0/None for off).
    for name, decay in (
        ("text", cfg.text_layer_decay),
        ("image", cfg.image_layer_decay),
        ("audio", cfg.audio_layer_decay),
    ):
        if decay is not None and not (0.0 < decay <= 1.0):
            raise ValueError(f"--{name}-layer-decay must be in (0, 1], got {decay}.")
    text_layer_decay = cfg.text_layer_decay if (cfg.text_layer_decay and cfg.text_layer_decay != 1.0) else None
    image_layer_decay = cfg.image_layer_decay if (cfg.image_layer_decay and cfg.image_layer_decay != 1.0) else None
    audio_layer_decay = cfg.audio_layer_decay if (cfg.audio_layer_decay and cfg.audio_layer_decay != 1.0) else None
    use_llrd = bool(text_layer_decay) or bool(image_layer_decay) or bool(audio_layer_decay)

    # Hybrid-optimizer fallback routing (Muon-family timm opts, e.g. nadamuon). ``fallback_list`` is a
    # *param-group-building* concept, not an optimizer ctor arg, and timm only applies it when handed an nn.Module;
    # since we always pre-build groups, we consume it here and flag the matched groups ``use_fallback=True``
    # ourselves (fnmatch on the full param name, as timm does). It comes from the first-class cfg field, falling
    # back to ``opt_kwargs`` for compatibility. The fallback LR scale (``fallback_lr_scale``) and any other
    # optimizer-specific fallback settings stay in ``opt_kwargs`` and are forwarded to the optimizer constructor,
    # so they only take effect for an optimizer that supports them.
    extra_opt_kwargs = dict(cfg.opt_kwargs or {})
    fallback_list = cfg.fallback_list or extra_opt_kwargs.pop("fallback_list", None)
    if fallback_list and not opt.startswith("timm/"):
        raise ValueError(
            f"--opt-fallback-list only applies to Muon-family timm optimizers (e.g. timm/nadamuon), not '{cfg.opt}'."
        )
    fallback_fn = None
    if fallback_list:
        fallback_fn = lambda name: any(fnmatch.fnmatchcase(name, pat) for pat in fallback_list)

    # Always pre-build the param groups with our own filter so the full no-weight-decay policy (the 1-D rule,
    # declared names/patterns incl. Swin's relative-position-bias keywords, and user patterns) and any LLRD apply
    # uniformly -- including the timm path, whose internal split honours neither keywords nor user patterns.
    # make_wd_exclude is a superset of timm's split (it subsumes the model's no_weight_decay()), so this never
    # decays a parameter timm would have spared.
    exclude_fn = make_wd_exclude(model, cfg.wd_exclude_patterns)
    if use_llrd:
        param_groups = layer_decay_param_groups(
            model,
            weight_decay,
            text_layer_decay,
            image_layer_decay,
            audio_layer_decay,
            cfg.pooler_in_head,
            exclude_fn,
            fallback_fn,
        )
    else:
        param_groups = wd_param_groups(model, weight_decay, exclude_fn, fallback_fn)

    if opt.startswith("timm/"):
        from timm.optim import create_optimizer_v2

        timm_opt = opt.split("timm/")[-1]
        opt_kwargs = {}
        assert (cfg.beta1 is None) == (cfg.beta2 is None), (
            "When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified)."
        )
        if cfg.beta1 is not None:
            opt_kwargs["betas"] = (cfg.beta1, cfg.beta2)
        if cfg.eps is not None:
            opt_kwargs["eps"] = cfg.eps
        if cfg.momentum is not None:
            opt_kwargs["momentum"] = cfg.momentum
        opt_kwargs.update(extra_opt_kwargs)  # incl. any fallback_lr_scale / optimizer-specific kwargs, forwarded below
        # timm optimizers consume pre-built param groups exactly like torch (per-group lr/weight_decay and the
        # use_fallback flag honoured, extra keys such as lr_scale carried through).
        optimizer = create_optimizer_v2(
            param_groups,
            timm_opt,
            lr=cfg.lr,
            weight_decay=weight_decay,
            **opt_kwargs,
        )
    elif opt in ("adamw", "nadamw"):
        # Only pass betas/eps when set, so a bare OptimizerCfg() (all None) falls back to the optimizer's own
        # defaults rather than AdamW(betas=(None, None)). The CLI always fills these via get_default_params.
        assert (cfg.beta1 is None) == (cfg.beta2 is None), "BOTH beta1 and beta2 must be specified (or neither)."
        opt_kwargs = {"lr": cfg.lr}
        if cfg.beta1 is not None:
            opt_kwargs["betas"] = (cfg.beta1, cfg.beta2)
        if cfg.eps is not None:
            opt_kwargs["eps"] = cfg.eps
        if opt == "nadamw":
            # builtin torch NAdam with AdamW-style decoupled weight decay (overridable via opt_kwargs).
            opt_kwargs["decoupled_weight_decay"] = True
        opt_kwargs.update(extra_opt_kwargs)
        optim_cls = optim.AdamW if opt == "adamw" else optim.NAdam
        optimizer = optim_cls(param_groups, **opt_kwargs)
    else:
        raise ValueError(f"Unknown optimizer {opt}")

    # Apply the per-group lr_scale to the initial LRs now (groups are constructed at the base lr). The scheduler
    # re-applies it each step, but doing it here means LLRD is correct before the first step and survives
    # --skip-scheduler (which otherwise leaves every group at the unscaled base lr).
    assign_learning_rate(optimizer, cfg.lr)
    if tensorize:
        tensorize_learning_rate(optimizer, device)

    if use_llrd:
        _logger.info(
            f"Layer-wise LR decay (text={text_layer_decay}, image={image_layer_decay}, "
            f"audio={audio_layer_decay}, pooler_in_head={cfg.pooler_in_head}): "
            f"{describe_param_groups(optimizer)}"
        )

    return optimizer
