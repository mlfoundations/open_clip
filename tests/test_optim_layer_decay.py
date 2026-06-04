"""Tests for the consolidated optimizer factory and the shared text-tower ``layer_groups`` enumeration that
both ``lock`` (freezing) and layer-wise LR decay leverage."""

import pytest
import torch

import open_clip
from open_clip_train.optim import (
    OptimizerCfg,
    collect_no_weight_decay,
    create_optimizer,
    exclude_from_wd,
    layer_decay_param_groups,
    lr_scales_for_groups,
    make_wd_exclude,
    wd_param_groups,
)
from open_clip_train.scheduler import assign_learning_rate, cosine_lr, get_learning_rate

NATIVE_MODEL = "naflexclap_test"  # CLAP with a tiny native TextTransformer text tower (hermetic)


def _cfg(opt="adamw", text_layer_decay=None, image_layer_decay=None, audio_layer_decay=None, pooler_in_head=True):
    return OptimizerCfg(
        opt=opt,
        lr=5e-4,
        weight_decay=0.2,
        beta1=0.9,
        beta2=0.98,
        eps=1e-6,
        text_layer_decay=text_layer_decay,
        image_layer_decay=image_layer_decay,
        audio_layer_decay=audio_layer_decay,
        pooler_in_head=pooler_in_head,
    )


def _frozen_text_param_names(model):
    return [n for n, p in model.text.named_parameters() if not p.requires_grad]


def test_lr_scales_positional():
    groups = [("embeddings", []), ("layer.0", []), ("layer.1", []), ("proj", [])]
    scales = lr_scales_for_groups(groups, 0.5)
    assert scales["proj"] == 1.0
    assert scales["layer.1"] == pytest.approx(0.5)
    assert scales["layer.0"] == pytest.approx(0.25)
    assert scales["embeddings"] == pytest.approx(0.125)
    # decay == 1.0 disables decay entirely.
    assert all(v == 1.0 for v in lr_scales_for_groups(groups, 1.0).values())


def test_native_layer_groups_structure():
    model = open_clip.create_model(NATIVE_MODEL)
    groups = model.text.layer_groups()
    names = [n for n, _ in groups]
    # embeddings, one group per resblock, then the projection head.
    assert names[0] == "embeddings"
    assert names[-1] == "proj"
    assert [n for n in names if n.startswith("layer.")] == ["layer.0", "layer.1"]
    # native tower has no pooler, so head/own are identical.
    assert [n for n, _ in model.text.layer_groups(pooler_in_head=False)] == names


def test_native_lock_freezes_all_at_zero():
    model = open_clip.create_model(NATIVE_MODEL)
    # unlocked_layers=0 freezes the whole tower, including the projection head (the top group).
    model.lock_text_tower(unlocked_layers=0, freeze_layer_norm=True)
    assert all(not p.requires_grad for _, p in model.text.named_parameters())


def test_native_lock_unlocks_top_groups():
    # unlocked_layers=1 -> only the projection head trainable; all encoder blocks + embeddings frozen.
    model = open_clip.create_model(NATIVE_MODEL)
    model.lock_text_tower(unlocked_layers=1, freeze_layer_norm=True)
    frozen = set(_frozen_text_param_names(model))
    assert not any("text_projection" in n for n in frozen), "projection head should be trainable"
    assert any(n.startswith("transformer.resblocks.") for n in frozen), "encoder blocks should be frozen"
    # unlocked_layers=2 -> projection + top block trainable; bottom block frozen.
    model2 = open_clip.create_model(NATIVE_MODEL)
    n_blocks = len(model2.text.transformer.resblocks)
    model2.lock_text_tower(unlocked_layers=2, freeze_layer_norm=True)
    frozen2 = set(_frozen_text_param_names(model2))
    assert any(n.startswith("transformer.resblocks.0.") for n in frozen2), "bottom block frozen"
    assert not any(n.startswith(f"transformer.resblocks.{n_blocks - 1}.") for n in frozen2), "top block trainable"


def test_create_optimizer_default_path_unchanged():
    model = open_clip.create_model(NATIVE_MODEL)
    opt = create_optimizer(model, _cfg(opt="adamw", text_layer_decay=None))
    # The long-standing default: a two-group weight-decay split, no per-group lr_scale.
    assert len(opt.param_groups) == 2
    assert all("lr_scale" not in g for g in opt.param_groups)
    wds = sorted(g["weight_decay"] for g in opt.param_groups)
    assert wds == [0.0, 0.2]
    # text_layer_decay == 1.0 is treated as "off" (no decay groups).
    opt1 = create_optimizer(model, _cfg(opt="adamw", text_layer_decay=1.0))
    assert len(opt1.param_groups) == 2


@pytest.mark.parametrize("opt", ["adamw", "timm/adamw"])
def test_create_optimizer_llrd_full_coverage(opt):
    model = open_clip.create_model(NATIVE_MODEL)
    optimizer = create_optimizer(model, _cfg(opt=opt, text_layer_decay=0.65))
    # Every trainable parameter lands in exactly one group.
    seen = {id(p) for g in optimizer.param_groups for p in g["params"]}
    trainable = [p for _, p in model.named_parameters() if p.requires_grad]
    assert len(seen) == len(trainable)
    scales = {round(g.get("lr_scale", 1.0), 6) for g in optimizer.param_groups}
    assert max(scales) == 1.0 and min(scales) < 1.0  # head/from-scratch at 1.0, text encoder decayed
    # Non-text params (audio tower, logit_scale) must be at lr_scale 1.0.
    id_scale = {id(p): g.get("lr_scale", 1.0) for g in optimizer.param_groups for p in g["params"]}
    for n, p in model.named_parameters():
        if p.requires_grad and (n == "logit_scale" or n.startswith("audio.")):
            assert id_scale[id(p)] == 1.0


def test_scheduler_honors_lr_scale():
    model = open_clip.create_model(NATIVE_MODEL)
    optimizer = create_optimizer(model, _cfg(text_layer_decay=0.65))
    sched = cosine_lr(optimizer, base_lr=5e-4, warmup_length=0, steps=100)
    sched(0)
    for g in optimizer.param_groups:
        assert g["lr"] == pytest.approx(5e-4 * g["lr_scale"])
    # get_learning_rate reports the (unscaled) base LR == the lr_scale==1.0 group.
    assert get_learning_rate(optimizer) == pytest.approx(5e-4)


def test_assign_learning_rate_flat_without_lr_scale():
    # Backward-compat: groups without lr_scale all receive the same LR.
    model = open_clip.create_model(NATIVE_MODEL)
    optimizer = create_optimizer(model, _cfg(text_layer_decay=None))
    assign_learning_rate(optimizer, 1e-3)
    assert all(g["lr"] == pytest.approx(1e-3) for g in optimizer.param_groups)


# ---- HF text tower: pooler placement policy must be consistent across lock and decay ----


def _build_roberta_tiny():
    pytest.importorskip("transformers")
    from open_clip.hf_model import HFTextEncoder

    try:
        return HFTextEncoder("arampacha/roberta-tiny", 64, pooler_type="cls_pooler", proj_type="clap_mlp")
    except Exception as e:  # offline / hub unavailable
        pytest.skip(f"could not load roberta-tiny: {e}")


def test_hf_pooler_in_head_vs_own():
    text = _build_roberta_tiny()
    head_names = [n for n, _ in text.layer_groups(pooler_in_head=True)]
    own_names = [n for n, _ in text.layer_groups(pooler_in_head=False)]
    # pooler_in_head=True: pooler folded into the proj head (no separate pooler group).
    assert "pooler" not in head_names and "proj" in head_names
    # pooler_in_head=False: pooler is its own group, just below proj.
    assert "pooler" in own_names and own_names.index("pooler") == own_names.index("proj") - 1
    # Decay scales: pooler at 1.0 when in head, at decay^1 when on its own.
    assert lr_scales_for_groups(text.layer_groups(pooler_in_head=False), 0.65)["pooler"] == pytest.approx(0.65)


@pytest.mark.parametrize("pooler_in_head,pooler_frozen", [(True, False), (False, True)])
def test_hf_lock_pooler_policy_consistent(pooler_in_head, pooler_frozen):
    text = _build_roberta_tiny()
    for p in text.parameters():
        p.requires_grad_(True)
    # unlocked_layers=1 frees the top group: with pooler_in_head=True the pooler rides in the proj head
    # (trainable); with False it sits in its own group just below proj and stays frozen. Mirrors decay's split.
    text.lock(unlocked_layers=1, freeze_layer_norm=True, pooler_in_head=pooler_in_head)
    frozen = {n for n, p in text.named_parameters() if not p.requires_grad}
    assert any("pooler" in n for n in frozen) is pooler_frozen
    # Projection head is trainable once unlocked_layers >= 1, regardless of policy.
    assert not any(n.startswith("proj.") for n in frozen)


def test_hf_lock_freezes_all_at_zero():
    text = _build_roberta_tiny()
    for p in text.parameters():
        p.requires_grad_(True)
    # unlocked_layers=0 freezes the whole tower, including pooler and projection head.
    text.lock(unlocked_layers=0, freeze_layer_norm=True)
    assert all(not p.requires_grad for p in text.parameters())


def test_hf_excludes_position_and_token_type_embeddings():
    text = _build_roberta_tiny()
    pats = text.no_weight_decay_patterns()
    assert any("position_embeddings" in p for p in pats)
    assert any("token_type_embeddings" in p for p in pats)
    # through the filter: the 2-D position/token-type embeddings are excluded, the 2-D word table is still decayed.
    exclude = make_wd_exclude(text)
    named = dict(text.named_parameters())
    pe = next(n for n in named if n.endswith("position_embeddings.weight"))
    tt = next(n for n in named if n.endswith("token_type_embeddings.weight"))
    we = next(n for n in named if n.endswith("word_embeddings.weight"))
    assert named[pe].ndim >= 2 and exclude(pe, named[pe])
    assert named[tt].ndim >= 2 and exclude(tt, named[tt])
    assert named[we].ndim >= 2 and not exclude(we, named[we])


def test_hf_mt5_excludes_relative_attention_bias():
    pytest.importorskip("transformers")
    from transformers import MT5Config
    from open_clip.hf_model import HFTextEncoder

    cfg = MT5Config(d_model=32, d_ff=64, num_layers=2, num_heads=2, vocab_size=128, relative_attention_num_buckets=8)
    enc = HFTextEncoder("mt5", 16, config=cfg, proj_type="linear")
    # config= construction normalizes encoder-decoder models to .encoder, so layer_groups sees encoder attrs.
    names = [n for n, _ in enc.layer_groups()]
    assert names[0] == "embeddings" and names[-1] == "proj"
    assert any("relative_attention_bias" in p for p in enc.no_weight_decay_patterns())  # T5's learned rel-pos bias
    exclude = make_wd_exclude(enc)
    named = dict(enc.named_parameters())
    rab = next(n for n in named if n.endswith("relative_attention_bias.weight"))
    ffn = next(n for n in named if "DenseReluDense.wo" in n)
    assert named[rab].ndim >= 2 and exclude(rab, named[rab])  # rel-pos bias excluded
    assert named[ffn].ndim >= 2 and not exclude(ffn, named[ffn])  # ordinary FFN weight decayed


def test_hf_bert_layer_groups_supported():
    # bert's arch_dict entry was missing layer_attr/token_embeddings_attr -> layer_groups() used to KeyError.
    pytest.importorskip("transformers")
    from transformers import BertConfig
    from open_clip.hf_model import HFTextEncoder

    cfg = BertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=64,
    )
    enc = HFTextEncoder("bert", 16, config=cfg, proj_type="linear")
    names = [n for n, _ in enc.layer_groups()]
    assert names[0] == "embeddings" and names[1] == "layer.0" and "proj" in names


# ---- vision towers: shared layer_groups + lock + image-tower LLRD ----


def test_vit_layer_groups_and_lock():
    model = open_clip.create_model("ViT-B-32")
    names = [n for n, _ in model.visual.layer_groups()]
    assert names[0] == "embeddings" and names[1] == "layer.0" and names[-1] == "proj"
    # unlocked_groups=0 freezes the whole image tower (including the projection).
    model.lock_image_tower(unlocked_groups=0)
    assert all(not p.requires_grad for p in model.visual.parameters())
    # unlocked_groups=2 -> projection + top block trainable, bottom block frozen.
    model2 = open_clip.create_model("ViT-B-32")
    model2.lock_image_tower(unlocked_groups=2)
    assert model2.visual.proj.requires_grad
    blocks = model2.visual.transformer.resblocks
    assert all(not p.requires_grad for p in blocks[0].parameters())
    assert any(p.requires_grad for p in blocks[-1].parameters())


def test_resnet_layer_groups_partial_lock():
    model = open_clip.create_model("RN50")
    names = [n for n, _ in model.visual.layer_groups()]
    assert names == ["embeddings", "layer.0", "layer.1", "layer.2", "layer.3", "proj"]
    # Partial lock is now supported (previously asserted off): unlocked_groups=1 -> only the attn-pool head trainable.
    model.lock_image_tower(unlocked_groups=1)
    assert any(p.requires_grad for p in model.visual.attnpool.parameters())
    assert all(not p.requires_grad for p in model.visual.layer1.parameters())


def test_timm_adapter_is_head():
    pytest.importorskip("timm")
    from open_clip.timm_model import TimmModel

    # proj='linear' -> trunk built with num_classes=0, the adapter carries the projection -> adapter is the head.
    with_adapter = TimmModel("vit_tiny_patch16_224", 512, proj="linear", pretrained=False)
    groups = with_adapter.layer_groups()
    assert groups[-1][0] == "proj" and groups[-1][1] == [with_adapter.head]
    # Head within the trunk (default proj) -> adapter empty, the trunk's own top group is the head (no extra 'proj').
    in_trunk = TimmModel("vit_tiny_patch16_224", 512, proj=None, pretrained=False)
    assert next(in_trunk.head.parameters(), None) is None
    assert not any(n == "proj" for n, _ in in_trunk.layer_groups())


# ---- review fixes: standard CLIP text LLRD / initial LRs / decay validation / reentrant lock / audio LLRD ----


def test_standard_clip_text_llrd():
    # Standard CLIP unpacks the text attrs directly onto the model (no model.text); text LLRD must still work.
    model = open_clip.create_model("ViT-B-32")
    assert not hasattr(model, "text")
    opt = create_optimizer(model, _cfg(text_layer_decay=0.65))
    id2s = {id(p): g.get("lr_scale", 1.0) for g in opt.param_groups for p in g["params"]}
    named = dict(model.named_parameters())
    blk = next(n for n in named if n.startswith("transformer.resblocks.0."))
    proj = next(n for n in named if "text_projection" in n)
    assert id2s[id(named[blk])] < 1.0  # text encoder block decayed
    assert id2s[id(named[proj])] == 1.0  # text projection (head) at full LR


def test_llrd_applied_to_initial_lrs():
    # LLRD must hold before the first scheduler step, so --skip-scheduler keeps it (not silently disabled).
    model = open_clip.create_model(NATIVE_MODEL)
    opt = create_optimizer(model, _cfg(text_layer_decay=0.65))  # base lr 5e-4
    lrs = [g["lr"] for g in opt.param_groups]
    assert max(lrs) == pytest.approx(5e-4)  # head/from-scratch groups at base lr
    assert min(lrs) < 5e-4  # deeper text groups already scaled down


def test_decay_out_of_range_rejected():
    model = open_clip.create_model(NATIVE_MODEL)
    for bad in (-0.5, 0.0, 1.5):
        with pytest.raises(ValueError):
            create_optimizer(model, _cfg(text_layer_decay=bad))


def test_vit_lock_reentrant():
    # Progressive unfreezing / multi-stage: lock(0) then lock(2) must end with the top 2 groups trainable,
    # not stay fully frozen.
    model = open_clip.create_model("ViT-B-32")
    model.lock_image_tower(unlocked_groups=0)
    model.lock_image_tower(unlocked_groups=2)
    blocks = model.visual.transformer.resblocks
    assert model.visual.proj.requires_grad  # proj head re-unlocked
    assert any(p.requires_grad for p in blocks[-1].parameters())  # top block re-unlocked
    assert all(not p.requires_grad for p in blocks[0].parameters())  # bottom still frozen


def test_audio_tower_llrd():
    # NaFlexClap: --audio-layer-decay targets model.audio (CLAP has no model.visual).
    model = open_clip.create_model("naflexclap_little")
    assert not hasattr(model, "visual")
    opt = create_optimizer(model, _cfg(audio_layer_decay=0.65))
    id2s = {id(p): g.get("lr_scale", 1.0) for g in opt.param_groups for p in g["params"]}
    named = dict(model.named_parameters())
    blk = next(n for n, p in named.items() if n.startswith("audio.encoder.vit") and "blocks.0" in n and p.ndim >= 2)
    txt = next(n for n in named if n.startswith("text."))
    assert id2s[id(named[blk])] < 1.0  # audio trunk decayed
    assert id2s.get(id(named[txt]), 1.0) == 1.0  # text tower left at full LR


class _ModuleWrap(torch.nn.Module):
    """Minimal DDP-style wrapper: the real model lives under ``.module`` (and attrs/params get a module. prefix)."""

    def __init__(self, model):
        super().__init__()
        self.module = model


def test_create_optimizer_unwraps_wrapper():
    # Under DDP the optimizer is built after wrapping; create_optimizer must unwrap for tower discovery + names.
    model = open_clip.create_model("ViT-B-32")
    opt = create_optimizer(_ModuleWrap(model), _cfg(text_layer_decay=0.65))  # raised pre-fix (no .text on wrapper)
    scales = {round(g.get("lr_scale", 1.0), 4) for g in opt.param_groups}
    assert min(scales) < 1.0 and max(scales) == 1.0  # LLRD applied through the wrapper
    seen = {id(p) for g in opt.param_groups for p in g["params"]}
    assert len(seen) == sum(1 for _, p in model.named_parameters() if p.requires_grad)  # full coverage, same objects


def test_create_optimizer_unwrap_preserves_name_matching():
    # The no-weight-decay / positional-embedding matching is name-based; the module. prefix from the wrapper must
    # not leak (i.e. roberta position_embeddings must still be excluded from weight decay through the wrapper).
    model = open_clip.create_model("naflexclap_little_roberta")
    opt = create_optimizer(_ModuleWrap(model), _cfg(text_layer_decay=0.65))
    id2wd = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    pe = dict(model.named_parameters())["text.transformer.embeddings.position_embeddings.weight"]
    assert id2wd[id(pe)] == 0.0


def test_create_optimizer_default_cfg_self_consistent():
    # A bare OptimizerCfg() (no betas/eps) must build, falling back to the optimizer's own defaults.
    opt = create_optimizer(open_clip.create_model(NATIVE_MODEL), OptimizerCfg())
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["betas"] == (0.9, 0.999) and opt.defaults["eps"] == 1e-8


# ---- robust weight-decay exclusion filter ----


def test_exclude_from_wd_is_dimensional():
    assert exclude_from_wd("x", torch.nn.Parameter(torch.zeros(8)))  # 1-D (bias / norm scale)
    assert exclude_from_wd("x", torch.nn.Parameter(torch.zeros([])))  # 0-D scalar (e.g. logit_scale)
    assert not exclude_from_wd("x", torch.nn.Parameter(torch.zeros(8, 8)))  # 2-D weight -> decayed


def test_make_wd_exclude_respects_no_weight_decay():
    # positional_embedding is 2-D (so not caught by the 1-D rule) but declared in the model's no_weight_decay().
    model = open_clip.create_model("ViT-B-32")
    named = dict(model.named_parameters())
    exclude = make_wd_exclude(model)
    pe = named["visual.positional_embedding"]
    assert pe.ndim >= 2 and exclude("visual.positional_embedding", pe)
    assert not exclude("visual.conv1.weight", named["visual.conv1.weight"])  # ordinary weight stays decayed


def test_make_wd_exclude_user_patterns():
    model = open_clip.create_model("ViT-B-32")
    w = dict(model.named_parameters())["visual.conv1.weight"]
    # A 4-D weight is normally decayed, but a matching glob pattern excludes it.
    assert not make_wd_exclude(model)("visual.conv1.weight", w)
    assert make_wd_exclude(model, ["visual.conv1*"])("visual.conv1.weight", w)


class _Leaf(torch.nn.Module):
    def __init__(self, declare, mechanism):
        super().__init__()
        # a 2-D param whose name ends in 'table' (so the 1-D rule does NOT exclude it)
        self.relative_position_bias_table = torch.nn.Parameter(torch.zeros(9, 4))
        self._declare = declare
        self._mechanism = mechanism

    def no_weight_decay_keywords(self):
        return {self._declare} if self._mechanism == "keywords" else set()

    def no_weight_decay_patterns(self):
        return [self._declare] if self._mechanism == "patterns" else []


class _Root(torch.nn.Module):
    def __init__(self, declare, mechanism):
        super().__init__()
        self.a = _Leaf(declare, mechanism)  # declares the exclusion
        self.b = _Leaf(declare, "none")  # sibling with the same param, declares nothing


def test_keyword_scoped_to_declaring_submodule():
    # 'a' declares the keyword; 'b' (same param name) does not -> only a's table is excluded.
    model = _Root("relative_position_bias_table", "keywords")
    names, patterns = collect_no_weight_decay(model)
    assert "a.*relative_position_bias_table" in patterns
    exclude = make_wd_exclude(model)
    assert exclude("a.relative_position_bias_table", model.a.relative_position_bias_table)
    assert not exclude("b.relative_position_bias_table", model.b.relative_position_bias_table)


def test_keyword_suffix_anchored_not_substring():
    # A sloppy generic keyword 'bias' must NOT match the 2-D relative_position_bias_table (ends in 'table');
    # suffix anchoring restricts it to params literally ending in 'bias' (which are 1-D and already excluded).
    model = _Root("bias", "keywords")
    exclude = make_wd_exclude(model)
    assert not exclude("a.relative_position_bias_table", model.a.relative_position_bias_table)


def test_no_weight_decay_patterns_convention():
    # The preferred glob convention: a relative pattern is scoped by prefixing the submodule path.
    model = _Root("*relative_position_bias_table", "patterns")
    names, patterns = collect_no_weight_decay(model)
    assert "a.*relative_position_bias_table" in patterns
    exclude = make_wd_exclude(model)
    assert exclude("a.relative_position_bias_table", model.a.relative_position_bias_table)
    assert not exclude("b.relative_position_bias_table", model.b.relative_position_bias_table)


def _build_clap_htsat():
    try:
        return open_clip.create_model("CLAP-HTSAT-tiny")
    except Exception as e:  # optional deps / config issues
        pytest.skip(f"could not build CLAP-HTSAT-tiny: {e}")


@pytest.mark.parametrize("opt", ["adamw", "timm/adamw"])
def test_htsat_relative_position_bias_excluded(opt):
    # Regression: Swin's relative_position_bias_table (2-D) is declared via no_weight_decay_keywords(); ensure it
    # lands in a no-decay group through create_optimizer for both the torch and timm paths.
    model = _build_clap_htsat()
    rpbt = [(n, p) for n, p in model.named_parameters() if n.endswith("relative_position_bias_table")]
    assert rpbt and rpbt[0][1].ndim >= 2, "expected 2-D relative_position_bias_table params"
    optimizer = create_optimizer(model, _cfg(opt=opt))
    no_wd_ids = {id(p) for g in optimizer.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    assert all(id(p) in no_wd_ids for _, p in rpbt)


def test_timm_surfaces_trunk_no_weight_decay():
    pytest.importorskip("timm")
    from open_clip.timm_model import TimmModel

    tm = TimmModel("vit_tiny_patch16_224", 512, proj="linear", pretrained=False)
    assert "trunk.pos_embed" in tm.no_weight_decay()
    pe = dict(tm.named_parameters())["trunk.pos_embed"]
    # pos_embed is 3-D, so it is excluded only because the trunk's no_weight_decay() is now surfaced.
    assert pe.ndim >= 2 and make_wd_exclude(tm)("trunk.pos_embed", pe)


@pytest.mark.parametrize("opt", ["adamw", "timm/adamw"])
def test_create_optimizer_wd_patterns_applied(opt):
    model = open_clip.create_model("ViT-B-32")
    cfg = _cfg(opt=opt)
    cfg.wd_exclude_patterns = ["visual.conv1*"]
    optimizer = create_optimizer(model, cfg)
    conv1 = dict(model.named_parameters())["visual.conv1.weight"]
    for g in optimizer.param_groups:
        if any(p is conv1 for p in g["params"]):
            assert g["weight_decay"] == 0.0  # the pattern moved it into the no-decay group
            break
    else:
        raise AssertionError("visual.conv1.weight not found in any optimizer group")


def test_create_optimizer_nadamw():
    model = open_clip.create_model(NATIVE_MODEL)
    optimizer = create_optimizer(model, _cfg(opt="nadamw"))
    assert isinstance(optimizer, torch.optim.NAdam)
    # decoupled weight decay defaults on, and the two-group weight-decay split is preserved.
    assert all(g.get("decoupled_weight_decay") for g in optimizer.param_groups)
    assert sorted(g["weight_decay"] for g in optimizer.param_groups) == [0.0, 0.2]
    # LLRD composes with nadamw too.
    llrd = create_optimizer(model, _cfg(opt="nadamw", text_layer_decay=0.65))
    assert isinstance(llrd, torch.optim.NAdam)
    scales = sorted({round(g.get("lr_scale", 1.0), 4) for g in llrd.param_groups})
    assert max(scales) == 1.0 and min(scales) < 1.0


def test_create_optimizer_fallback_list_routing():
    pytest.importorskip("timm")
    model = open_clip.create_model(NATIVE_MODEL)
    cfg = _cfg(opt="timm/nadamuon")
    cfg.fallback_list = ["*token_embedding*"]  # first-class --opt-fallback-list
    cfg.opt_kwargs = {"fallback_lr_scale": 0.5}  # fallback LR scale stays in opt-kwargs -> optimizer ctor
    try:
        opt = create_optimizer(model, cfg)
    except Exception as e:  # nadamuon may be absent in older timm
        pytest.skip(f"nadamuon unavailable: {e}")
    id2fb = {id(p): g.get("use_fallback", False) for g in opt.param_groups for p in g["params"]}
    named = dict(model.named_parameters())
    te = next(n for n in named if "token_embedding" in n)
    body = next(n for n, p in named.items() if "resblocks" in n and p.ndim == 2)
    assert id2fb[id(named[te])] is True  # matched param routed to the Adam fallback
    assert id2fb[id(named[body])] is False  # transformer body stays on Muon (auto)
    assert opt.defaults.get("fallback_lr_scale") == 0.5  # opt-kwargs reached the optimizer ctor
    assert not any("fallback_list" in g for g in opt.param_groups)  # consumed at build time, not leaked


def test_fallback_list_rejected_for_torch_opt():
    # Muon fallback routing is meaningless for torch optimizers -> fail fast rather than silently no-op.
    model = open_clip.create_model(NATIVE_MODEL)
    cfg = _cfg(opt="adamw")
    cfg.fallback_list = ["*token_embedding*"]
    with pytest.raises(ValueError):
        create_optimizer(model, cfg)


@pytest.mark.parametrize("opt", ["adamw", "timm/adamw"])
def test_create_optimizer_image_decay(opt):
    model = open_clip.create_model("ViT-B-32")
    optimizer = create_optimizer(model, _cfg(opt=opt, image_layer_decay=0.65))
    seen = {id(p) for g in optimizer.param_groups for p in g["params"]}
    trainable = [p for _, p in model.named_parameters() if p.requires_grad]
    assert len(seen) == len(trainable)  # full coverage
    id_scale = {id(p): g.get("lr_scale", 1.0) for g in optimizer.param_groups for p in g["params"]}
    # The (un-decayed) text tower stays at lr_scale 1.0; the image tower is decayed with a 1.0 head.
    text_scales = {
        round(id_scale[id(p)], 4)
        for n, p in model.named_parameters()
        if n.startswith("transformer.") and p.requires_grad
    }
    assert text_scales == {1.0}
    img_scales = [id_scale[id(p)] for n, p in model.named_parameters() if n.startswith("visual.") and p.requires_grad]
    assert min(img_scales) < 1.0 and max(img_scales) == 1.0
