"""ModelTraits: family table, config vs model resolution, factory attachment, and the run-level combination."""
import types
from copy import deepcopy

import pytest
import torch.nn as nn

import open_clip
from open_clip import factory
from open_clip.model_traits import (
    CLAP_TRAITS,
    CLIP_TRAITS,
    COCA_TRAITS,
    GENLAP_TRAITS,
    GENLIP_TRAITS,
    MAMMUT_TRAITS,
    InputMode,
    ModelFamily,
    ModelTraits,
    get_model_traits,
    traits_from_config,
    traits_from_model,
)
from open_clip_train.params import apply_model_traits, parse_args
from open_clip_train.loss import create_loss_from_args
from open_clip.transform import AugmentationCfg

_NAFLEX_AVAILABLE = True
try:
    from timm.data import naflex_transforms  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    _NAFLEX_AVAILABLE = False


# ---------------------------------------------------------------------------------------------------------------------
# The family table
# ---------------------------------------------------------------------------------------------------------------------
def test_family_table():
    rows = {
        # family:        (generative, contrastive, requires_naflex, text_in_budget, consumes_mask, cached_accum)
        CLIP_TRAITS:     (False, True, False, False, False, True),
        COCA_TRAITS:     (True, True, False, False, True, True),
        MAMMUT_TRAITS:   (True, True, False, False, True, True),
        GENLIP_TRAITS:   (True, False, True, True, True, False),
        GENLAP_TRAITS:   (True, False, True, True, True, False),
        CLAP_TRAITS:     (False, True, False, False, False, True),
    }
    for traits, expected in rows.items():
        got = (
            traits.generative, traits.contrastive, traits.requires_naflex_data,
            traits.naflex_text_in_token_budget, traits.consumes_text_mask, traits.supports_cached_grad_accum,
        )
        assert got == expected, traits.family
    # only CoCa / MaMMUT take the standard collator's text_valid key; GenLIP/GenLAP mask via the NaFlex rows
    assert [t.family for t in rows if t.wants_text_valid_key] == [ModelFamily.COCA, ModelFamily.MAMMUT]


def test_class_defaults_match_table():
    from open_clip import CLAP, CLIP, CoCa, CustomTextCLIP, MaMMUT, NaFlexGenLap, NaFlexGenLip

    assert CLIP.traits == CLIP_TRAITS and CustomTextCLIP.traits == CLIP_TRAITS
    assert CoCa.traits == COCA_TRAITS
    assert MaMMUT.traits == MAMMUT_TRAITS
    assert NaFlexGenLip.traits == GENLIP_TRAITS
    assert NaFlexGenLap.traits == GENLAP_TRAITS
    assert CLAP.traits == CLAP_TRAITS


def test_traits_are_frozen_and_enum_values_are_strings():
    with pytest.raises(Exception):
        CLIP_TRAITS.generative = True  # type: ignore[misc]
    assert ModelFamily.GENLIP == "genlip" and InputMode.NAFLEX == "naflex"


# ---------------------------------------------------------------------------------------------------------------------
# Config resolver
# ---------------------------------------------------------------------------------------------------------------------
def test_traits_from_config_keys_on_config_structure():
    assert traits_from_config({"genlip_cfg": {}, "vision_cfg": {}, "text_cfg": {}}) == GENLIP_TRAITS
    assert traits_from_config({"genlap_cfg": {}, "audio_naflex_cfg": {}, "text_cfg": {}}) == GENLAP_TRAITS

    clap = traits_from_config({"audio_cfg": {"model_type": "HTSAT"}, "text_cfg": {}})
    assert clap.family is ModelFamily.CLAP and clap.audio_input is InputMode.FIXED and not clap.requires_naflex_data
    naflexclap = traits_from_config({"audio_cfg": {"model_type": "naflexvit"}, "text_cfg": {"variable_text": True}})
    assert naflexclap.audio_input is InputMode.NAFLEX and naflexclap.requires_naflex_data and naflexclap.variable_text

    coca = traits_from_config({"multimodal_cfg": {}, "text_cfg": {"variable_text": True}, "vision_cfg": {"layers": 12}})
    assert coca.family is ModelFamily.COCA and coca.image_input is InputMode.FIXED and coca.variable_text
    mammut = traits_from_config({"multimodal_cfg": {"variable_text": True}, "vision_cfg": {"layers": 12}})
    assert mammut.family is ModelFamily.MAMMUT and mammut.variable_text  # MaMMUT's text cfg is multimodal_cfg

    naflex_clip = traits_from_config({"vision_cfg": {"timm_model_name": "naflexvit_base_patch16_gap"}, "text_cfg": {}})
    assert naflex_clip.family is ModelFamily.CLIP and naflex_clip.image_input is InputMode.NAFLEX
    assert not naflex_clip.requires_naflex_data  # a NaFlexVit tower still accepts fixed images
    converted = traits_from_config(
        {"vision_cfg": {"timm_model_name": "vit_base_patch16_224", "timm_model_kwargs": {"use_naflex": True}}, "text_cfg": {}})
    assert converted.image_input is InputMode.NAFLEX

    # a full open_clip_config.json wrapper is accepted too
    assert traits_from_config({"model_cfg": {"genlip_cfg": {}}, "preprocess_cfg": {}}) == GENLIP_TRAITS


# ---------------------------------------------------------------------------------------------------------------------
# Model resolver, accessor, and factory attachment
# ---------------------------------------------------------------------------------------------------------------------
class _Trunk(nn.Module):
    pass


class _NaFlexVit(nn.Module):  # name is what the resolver keys on
    pass


NaFlexVit = _NaFlexVit
NaFlexVit.__name__ = "NaFlexVit"


def test_traits_from_model_reads_towers_and_get_model_traits_unwraps():
    from open_clip import CLIP

    class FakeCLIP(CLIP):
        def __init__(self, trunk, variable_text):  # noqa: D401 - minimal stand-in, skip the real ctor
            nn.Module.__init__(self)
            self.visual = types.SimpleNamespace(trunk=trunk)
            self.text = types.SimpleNamespace(variable_text=variable_text)

    fixed = FakeCLIP(_Trunk(), False)
    t = traits_from_model(fixed)
    assert t.family is ModelFamily.CLIP and t.image_input is InputMode.FIXED and not t.variable_text

    naflex = FakeCLIP(NaFlexVit(), True)
    t = traits_from_model(naflex)
    assert t.image_input is InputMode.NAFLEX and t.variable_text and not t.requires_naflex_data

    # accessor: attached instance wins, and wrappers are seen through
    attached = ModelTraits(family=ModelFamily.CLIP, variable_text=True)
    naflex.traits = attached
    wrapped = types.SimpleNamespace(module=types.SimpleNamespace(_orig_mod=naflex))
    assert get_model_traits(wrapped) is attached
    # no attached instance: resolved from the towers, not the class default
    assert get_model_traits(fixed).image_input is InputMode.FIXED

    with pytest.raises(TypeError, match="declares no ModelTraits"):
        traits_from_model(nn.Linear(2, 2))


@pytest.mark.parametrize(
    "name",
    [
        "RN50",
        "ViT-B-32",
        "naflex_ViT-B-32",
        "coca_ViT-B-32",
        "mammut2-naflex_ViT-B-32",
        "mammut2-moderntext_ViT-B-32",
        "naflexgenlip_test",
        "naflexgenlap_test_1d",
        "naflexclap_base_pf8_pt16_moderntext",
        "CLAP-HTSAT-tiny",
    ],
)
def test_factory_attaches_traits_that_agree_with_config(name):
    """Drift test: the config prediction equals what the factory attached to the built model."""
    if "naflex" in name.lower() and not _NAFLEX_AVAILABLE:
        pytest.skip("timm NaFlex support not available")
    if "clap" in name.lower() and "naflex" not in name.lower():
        pytest.importorskip("torchaudio")
    model = open_clip.create_model(name, load_weights=False)
    attached = model.__dict__["traits"]
    assert isinstance(attached, ModelTraits)
    assert get_model_traits(model) is attached
    assert traits_from_config(factory.get_model_config(name)) == attached
    if name == "mammut2-moderntext_ViT-B-32":
        args = apply_model_traits(parse_args(["--model", name]), attached)
        assert model.text.variable_text and args.variable_text


@pytest.mark.skipif(not _NAFLEX_AVAILABLE, reason="timm NaFlex support not available")
def test_force_naflex_vision_converts_only_convertible_towers():
    converted = open_clip.create_model("ViT-B-32", load_weights=False, force_naflex_vision=True)
    assert get_model_traits(converted).image_input is InputMode.NAFLEX
    assert traits_from_config(factory.get_model_config("ViT-B-32")).image_input is InputMode.FIXED  # pre-conversion

    # NaFlex-native and audio models: --use-naflex passes the flag for every model, it must be a no-op here
    genlip = open_clip.create_model("naflexgenlip_test", load_weights=False, force_naflex_vision=True)
    assert get_model_traits(genlip).requires_naflex_data
    genlap = open_clip.create_model("naflexgenlap_test_1d", load_weights=False, force_naflex_vision=True)
    assert get_model_traits(genlap).family is ModelFamily.GENLAP
    naflexclap = open_clip.create_model(
        "naflexclap_base_pf8_pt16_moderntext", load_weights=False, force_naflex_vision=True)
    assert get_model_traits(naflexclap).audio_input is InputMode.NAFLEX

    with pytest.raises(RuntimeError):  # a ResNet tower is not convertible
        open_clip.create_model("RN50", load_weights=False, force_naflex_vision=True)


@pytest.mark.skipif(not _NAFLEX_AVAILABLE, reason="timm NaFlex support not available")
@pytest.mark.parametrize("aug_cfg", [
    None,
    {"naflex": True},
    {"naflex": True, "use_timm": False},
    {"naflex": True, "use_timm": True},
    {"scale": (0.6, 1.0)},
    AugmentationCfg(naflex=True, scale=(0.6, 1.0)),
])
def test_genlip_always_gets_naflex_transforms(aug_cfg):
    """Partial NaFlex options must still produce factories, preserving other settings and caller input."""
    original = deepcopy(aug_cfg)
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "naflexgenlip_test", load_weights=False, aug_cfg=aug_cfg,
    )
    assert getattr(preprocess_train, "is_naflex_transform_factory", False)
    assert getattr(preprocess_val, "is_naflex_eval_transform_factory", False)
    assert aug_cfg == original
    expected_scale = (
        aug_cfg.get("scale", (0.9, 1.0)) if isinstance(aug_cfg, dict)
        else aug_cfg.scale if aug_cfg is not None else (0.9, 1.0)
    )
    assert preprocess_train.transform_kwargs["scale"] == expected_scale


def test_legacy_loss_dispatches_on_traits_never_names():
    from open_clip.loss import ClipLoss, CoCaLoss, GenLipLoss

    args = types.SimpleNamespace(
        model="coca_ViT-B-32", distill=False, siglip=False, local_loss=False, gather_with_grad=False, rank=0,
        world_size=1, coca_caption_loss_weight=2.0, coca_contrastive_loss_weight=1.0, loss_dist_impl=None,
        torchcompile=False, torchcompile_strategy="task",
    )
    # the name says coca; only the model / traits decide
    assert isinstance(create_loss_from_args(args, model=types.SimpleNamespace(traits=CLIP_TRAITS)), ClipLoss)
    assert isinstance(create_loss_from_args(args, model=types.SimpleNamespace(traits=COCA_TRAITS, pad_id=7)), CoCaLoss)
    assert isinstance(create_loss_from_args(args, model=types.SimpleNamespace(traits=GENLIP_TRAITS)), GenLipLoss)


# ---------------------------------------------------------------------------------------------------------------------
# Run-level combination (trainer side)
# ---------------------------------------------------------------------------------------------------------------------
def _run_args(*argv):
    return parse_args(["--model", "ViT-B-32", *argv])


@pytest.mark.parametrize("traits", [COCA_TRAITS, MAMMUT_TRAITS, GENLIP_TRAITS, GENLAP_TRAITS, CLAP_TRAITS])
@pytest.mark.parametrize("entrypoint", ["trainer", "legacy_loss", "task"])
def test_distillation_rejected_consistently(traits, entrypoint):
    args = _run_args()
    args.distill = True
    args.rank, args.world_size = 0, 1
    # The factories must reject this before constructing a task or requiring any model internals.
    model = nn.Module()
    model.traits = traits
    wrapped = types.SimpleNamespace(module=types.SimpleNamespace(_orig_mod=model))
    message = (
        f"distillation is not supported for generative models ({traits.family.value})."
        if traits.generative else "CLAP distillation is not supported in this integration."
    )
    with pytest.raises(ValueError) as exc:
        if entrypoint == "trainer":
            apply_model_traits(args, traits)
        elif entrypoint == "legacy_loss":
            create_loss_from_args(args, model=wrapped)
        else:
            open_clip.create_task(args, model=wrapped)
    assert str(exc.value) == message


@pytest.mark.parametrize("siglip", [False, True])
def test_contrastive_image_distillation_remains_supported(siglip):
    from open_clip.loss import DistillClipLoss

    args = _run_args()
    args.distill, args.siglip = True, siglip
    args.rank, args.world_size = 0, 1
    apply_model_traits(args, CLIP_TRAITS)
    assert args.text_attention_mask is False
    assert isinstance(create_loss_from_args(args, model=types.SimpleNamespace(traits=CLIP_TRAITS)), DistillClipLoss)


def test_parse_args_model_name_does_not_imply_training_flags():
    args = parse_args(["--model", "naflexgenlip_test"])
    assert args.use_naflex is False
    args = parse_args(["--model", "ViT-B-32", "--use-naflex"])
    assert args.force_naflex_vision is True and args.aug_cfg["naflex"] is True


def test_apply_model_traits_combines_flags_with_traits():
    # CLIP: nothing implied; text mask defaults off
    args = apply_model_traits(_run_args(), CLIP_TRAITS)
    assert args.use_naflex is False and args.variable_text is False and args.text_attention_mask is False

    # NaFlexCLAP implies NaFlex data + aug_cfg toggles; variable text from the tower
    naflexclap = ModelTraits(
        family=ModelFamily.CLAP, audio_input=InputMode.NAFLEX, requires_naflex_data=True, variable_text=True,
        image_input=InputMode.NONE)
    args = apply_model_traits(_run_args(), naflexclap)
    assert args.use_naflex and args.aug_cfg["naflex"] and args.variable_text

    # CoCa: text_valid key wanted unless distilling
    args = apply_model_traits(_run_args(), COCA_TRAITS)
    assert args.text_attention_mask is True
    args = _run_args(); args.distill = True
    with pytest.raises(ValueError, match="distillation"):
        apply_model_traits(args, COCA_TRAITS)

    # GenLIP: mask consumed via NaFlex rows, so the batch key stays off and an explicit request fails fast
    args = apply_model_traits(_run_args(), GENLIP_TRAITS)
    assert args.text_attention_mask is False and args.use_naflex and args.variable_text
    with pytest.raises(ValueError, match="text-attention-mask"):
        apply_model_traits(_run_args("--text-attention-mask"), GENLIP_TRAITS)
    with pytest.raises(ValueError, match="accum-freq"):
        apply_model_traits(_run_args("--accum-freq", "2"), GENLIP_TRAITS)
    apply_model_traits(_run_args("--accum-freq", "2"), COCA_TRAITS)  # cached features: fine

    # --use-naflex needs a NaFlex-capable tower: a converted / NaFlexVit image tower passes, fixed-audio CLAP fails
    from dataclasses import replace
    args = apply_model_traits(_run_args("--use-naflex"), replace(CLIP_TRAITS, image_input=InputMode.NAFLEX))
    assert args.use_naflex is True
    with pytest.raises(ValueError, match="NaFlex-capable"):
        apply_model_traits(_run_args("--use-naflex"), CLAP_TRAITS)
    with pytest.raises(ValueError, match="NaFlex-capable"):
        apply_model_traits(_run_args("--use-naflex"), CLIP_TRAITS)  # unconverted fixed image tower

    # a user-set variable_text (no CLI flag; set on the namespace) still wins on a fixed-text tower
    args = _run_args(); args.variable_text = True
    assert apply_model_traits(args, CLIP_TRAITS).variable_text is True
