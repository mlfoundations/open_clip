"""Model traits: what kind of model was built, and what the data / training pipeline must do for it.

A :class:`ModelTraits` describes the *built* model only. No field depends on a data-pipeline or run flag;
run-level decisions that combine a trait with a user flag (NaFlex batching for convertible towers, variable
text overrides, the ``--text-attention-mask`` default, ...) live in the trainer
(``open_clip_train.params.apply_model_traits``), not here.

Resolution has three layers:

1. **Class defaults.** Every model class declares ``traits`` (the family-level row of the table below), so a
   model built directly, outside the factory, still carries correct family facts.
2. **Factory override.** ``create_model`` attaches ``model.traits = traits_from_model(model)`` after
   instantiation, filling the tower-level fields (``image_input`` / ``audio_input`` / ``variable_text``) from
   the towers it just built. This is the single resolution point for built-in names, local config files,
   ``local-dir:`` and ``hf-hub:`` sources alike -- the factory has the final config for all of them.
3. **Accessor.** :func:`get_model_traits` sees through DDP / ``torch.compile`` wrappers, returns the attached
   instance, and falls back to :func:`traits_from_model` for models that were never through the factory.

:func:`traits_from_config` predicts the same facts from a model config, for the places inside the factory that
only have a config (tokenizer validation, the ``force_naflex_vision`` no-op decision). It must agree with
:func:`traits_from_model` on every built-in config; ``tests/test_model_traits.py`` enforces that.

Family table (intrinsic fields; tower-level fields vary per config):

    family   generative  contrastive  requires_naflex_data  text_in_budget  consumes_mask  cached_accum
    CLIP     no          yes          no                    no              no             yes
    COCA     yes         yes          no                    no              yes            yes
    MAMMUT   yes         yes          no                    no              yes            yes
    GENLIP   yes         no           yes                   yes             yes            no
    GENLAP   yes         no           yes                   yes             yes            no
    CLAP     no          yes          NaFlexCLAP only       no              no             yes

This module is intentionally torch-free so the argument-parsing and data layers can import it.
"""
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional

__all__ = [
    "ModelFamily",
    "InputMode",
    "ModelTraits",
    "CLIP_TRAITS",
    "COCA_TRAITS",
    "MAMMUT_TRAITS",
    "GENLIP_TRAITS",
    "GENLAP_TRAITS",
    "CLAP_TRAITS",
    "traits_from_config",
    "traits_from_model",
    "get_model_traits",
    "validate_distillation",
]


class ModelFamily(str, Enum):
    CLIP = "clip"
    COCA = "coca"
    MAMMUT = "mammut"
    GENLIP = "genlip"
    GENLAP = "genlap"
    CLAP = "clap"


class InputMode(str, Enum):
    NONE = "none"      # modality absent
    FIXED = "fixed"    # fixed-size tensors
    NAFLEX = "naflex"  # patch dicts: patches / patch_coord / patch_valid


@dataclass(frozen=True)
class ModelTraits:
    """Facts about a built model. See the module docstring for the family table."""

    # -- family-level: the class default on each model class --
    family: ModelFamily
    generative: bool = False            # has a caption / LM loss
    contrastive: bool = True            # has an image-text or audio-text contrastive loss
    requires_naflex_data: bool = False  # cannot consume fixed batches at all (GenLIP, GenLAP, NaFlexCLAP)
    naflex_text_in_token_budget: bool = False  # captions count toward the NaFlex row token budget
    consumes_text_mask: bool = False    # forward takes a text validity mask
    supports_cached_grad_accum: bool = True    # False where no contrastive feature cache exists (accum_freq > 1)

    # -- tower-level: read off the built towers by the factory --
    image_input: InputMode = InputMode.FIXED   # NAFLEX = patch-dict native (timm NaFlexVit, GenLIP embed)
    audio_input: InputMode = InputMode.NONE    # NAFLEX = spectrogram-ViT patch dicts (NaFlexCLAP, GenLAP)
    variable_text: bool = False                # text tower expects per-batch padded text with a reserved pad

    @property
    def wants_text_valid_key(self) -> bool:
        """The standard image-text collator should emit ``text_valid`` for this model.

        GenLIP / GenLAP consume a mask too, but they get it from the NaFlex row collator that also budgets
        their captions, so the separate batch key is only for CoCa / MaMMUT.
        """
        return self.consumes_text_mask and not self.naflex_text_in_token_budget


CLIP_TRAITS = ModelTraits(family=ModelFamily.CLIP)
COCA_TRAITS = ModelTraits(family=ModelFamily.COCA, generative=True, consumes_text_mask=True)
MAMMUT_TRAITS = ModelTraits(family=ModelFamily.MAMMUT, generative=True, consumes_text_mask=True)
GENLIP_TRAITS = ModelTraits(
    family=ModelFamily.GENLIP,
    generative=True,
    contrastive=False,
    requires_naflex_data=True,
    naflex_text_in_token_budget=True,
    consumes_text_mask=True,
    supports_cached_grad_accum=False,
    image_input=InputMode.NAFLEX,
    variable_text=True,
)
GENLAP_TRAITS = replace(
    GENLIP_TRAITS,
    family=ModelFamily.GENLAP,
    image_input=InputMode.NONE,
    audio_input=InputMode.NAFLEX,
)
CLAP_TRAITS = ModelTraits(family=ModelFamily.CLAP, image_input=InputMode.NONE, audio_input=InputMode.FIXED)


def validate_distillation(traits: ModelTraits) -> None:
    """Reject model families unsupported by the contrastive distillation task and loss."""
    if traits.generative:
        raise ValueError(f"distillation is not supported for generative models ({traits.family.value}).")
    if traits.family is ModelFamily.CLAP:
        raise ValueError("CLAP distillation is not supported in this integration.")


def _is_naflex_audio_type(model_type: Any) -> bool:
    return str(model_type or "").lower() == "naflexvit"


def _vision_cfg_input(vision_cfg: Optional[Dict[str, Any]]) -> InputMode:
    if not vision_cfg:
        return InputMode.NONE
    timm_model_name = vision_cfg.get("timm_model_name") or ""
    timm_model_kwargs = vision_cfg.get("timm_model_kwargs") or {}
    if timm_model_name.startswith("naflexvit") or timm_model_kwargs.get("use_naflex", False):
        return InputMode.NAFLEX
    return InputMode.FIXED


def traits_from_config(model_cfg: Dict[str, Any]) -> ModelTraits:
    """Predict the traits of the model ``model_cfg`` would build.

    Keys on the same config structure ``create_model`` dispatches on (``genlip_cfg`` / ``genlap_cfg`` /
    ``audio_cfg`` / ``multimodal_cfg``). Accepts a bare model config or a full ``open_clip_config.json`` dict.
    Pass the *final* config (after ``force_naflex_vision`` etc.) to predict the tower-level fields correctly.
    """
    cfg = model_cfg.get("model_cfg", model_cfg) if isinstance(model_cfg, dict) else model_cfg
    if "genlip_cfg" in cfg:
        return GENLIP_TRAITS
    if "genlap_cfg" in cfg:
        return GENLAP_TRAITS
    # MaMMUT carries its (only) text tower config in multimodal_cfg
    text_cfg = cfg.get("text_cfg") or cfg.get("multimodal_cfg") or {}
    variable_text = bool(text_cfg.get("variable_text", False))
    if "audio_cfg" in cfg:
        naflex_audio = _is_naflex_audio_type((cfg.get("audio_cfg") or {}).get("model_type"))
        return replace(
            CLAP_TRAITS,
            audio_input=InputMode.NAFLEX if naflex_audio else InputMode.FIXED,
            requires_naflex_data=naflex_audio,
            variable_text=variable_text,
        )
    image_input = _vision_cfg_input(cfg.get("vision_cfg"))
    if "multimodal_cfg" in cfg:
        if "text_cfg" not in cfg:
            return replace(MAMMUT_TRAITS, image_input=image_input, variable_text=variable_text)
        return replace(COCA_TRAITS, image_input=image_input, variable_text=variable_text)
    return replace(CLIP_TRAITS, image_input=image_input, variable_text=variable_text)


def traits_from_model(model: Any) -> ModelTraits:
    """Resolve traits from a built (unwrapped) model: its class default plus the tower-level fields."""
    base = getattr(type(model), "traits", None)
    if not isinstance(base, ModelTraits):
        raise TypeError(f"{type(model).__name__} declares no ModelTraits; is it an open_clip model class?")

    image_input = base.image_input
    if base.family in (ModelFamily.CLIP, ModelFamily.COCA, ModelFamily.MAMMUT):
        visual = getattr(model, "visual", None)
        trunk = getattr(visual, "trunk", None)
        image_input = InputMode.NAFLEX if type(trunk).__name__ == "NaFlexVit" else InputMode.FIXED

    audio_input = base.audio_input
    if base.family is ModelFamily.CLAP:
        audio_cfg = getattr(getattr(model, "audio", None), "cfg", None)
        audio_input = (
            InputMode.NAFLEX if _is_naflex_audio_type(getattr(audio_cfg, "model_type", None)) else InputMode.FIXED
        )

    variable_text = base.variable_text
    text = getattr(model, "text", None)
    if text is not None and hasattr(text, "variable_text"):
        variable_text = bool(text.variable_text)

    return replace(
        base,
        image_input=image_input,
        audio_input=audio_input,
        variable_text=variable_text,
        requires_naflex_data=base.requires_naflex_data or audio_input is InputMode.NAFLEX,
    )


def unwrap_model(model: Any) -> Any:
    """Unwrap nested DDP and torch.compile wrappers without importing torch."""
    unwrapped = model
    for _ in range(4):
        inner = getattr(unwrapped, "module", None)
        if inner is None:
            inner = getattr(unwrapped, "_orig_mod", None)
        if inner is None:
            break
        unwrapped = inner
    return unwrapped


def get_model_traits(model: Any) -> ModelTraits:
    """Return the traits attached by the factory, resolving them for models that never went through it."""
    unwrapped = unwrap_model(model)
    attached = unwrapped.__dict__.get("traits") if hasattr(unwrapped, "__dict__") else None
    if isinstance(attached, ModelTraits):
        return attached
    return traits_from_model(unwrapped)
