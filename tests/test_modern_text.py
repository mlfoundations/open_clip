"""Tests for the ModernTextTransformer text tower and the variable-length text collators.

Covers, in order:
- attention-path numerics: causal SDPA with ``is_causal=True`` (no ``[B, 1, L, L]`` mask) is numerically
  identical to a full additive mask for the pooled output; bidirectional uses a ``[B, 1, 1, L]`` key-pad mask
  that correctly excludes padding; padding invariance; init scheme,
- pooling semantics and eos_id/pad_id validation (tower, ``_validate_special_tokens``),
- the public tower surface, CustomTextCLIP routing, builtin/local config registration, tokenizer wiring,
- the variable-text collators: per-batch padding, ``pad_multiple`` rounding clamped at ``pad_cap``, and the
  always-emitted ``text_valid`` mask (tasks select the batch keys they use).
"""
import json
import types

import pytest
import torch

import open_clip
from open_clip import CLIPTextCfg, CLIPVisionCfg, ModernTextTransformer
from open_clip.factory import _validate_special_tokens
from open_clip.model import CLIP
from open_clip.transformer import LayerNorm
from open_clip.tokenizer import SimpleTokenizer
from open_clip_train.audio_data import _audio_collate
from open_clip_train.data import collate_variable_text_dicts
from open_clip_train.naflex_data import collate_variable_text


PAD_ID = 100
EOS_ID = 99


def _modern_cfg(**kwargs):
    """Small modern text_cfg for surface / integration tests."""
    cfg = CLIPTextCfg(
        text_arch="modern",
        context_length=8,
        vocab_size=64,
        width=32,
        heads=4,
        layers=2,
        mlp_ratio=2.0,
        pad_id=0,
        eos_id=2,
        pool_type="eos",
        attention_mode="causal",
        attn_gated=True,
        qk_norm=True,
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_modern_text(
        attention_mode="causal", pool_type="eos", qk_norm=True, attn_gated=True, layers=3, eos_id=EOS_ID,
        **overrides,
):
    """Tower builder for the attention/pooling numerics tests (wider, variable-text, swiglu/rope defaults)."""
    cfg = CLIPTextCfg(
        text_arch="modern",
        variable_text=True,
        context_length=32,
        vocab_size=120,
        width=64,
        heads=4,
        layers=layers,
        mlp_ratio=2.6666667,
        pad_id=PAD_ID,
        bos_id=101,
        eos_id=eos_id,
        attention_mode=attention_mode,
        pool_type=pool_type,
        qk_norm=qk_norm,
        attn_gated=attn_gated,
        proj_bias=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return ModernTextTransformer(cfg, output_dim=48).eval()


def _right_padded_batch(lengths, total_len):
    text = torch.randint(0, EOS_ID, (len(lengths), total_len))
    for i, n in enumerate(lengths):
        text[i, n - 1] = EOS_ID  # eos at the last real token
        if n < total_len:
            text[i, n:] = PAD_ID  # right-pad the tail
    return text


def _old_attn_inputs(self, text, dtype, num_prefix=0):
    """Reference: a full ``[B, 1, L, L]`` additive mask (causal AND valid), is_causal=False."""
    assert num_prefix == 0  # reference path is register-free
    b, l = text.shape
    valid = self._valid_mask(text)
    allowed = valid[:, None, None, :].expand(b, 1, l, l).clone()
    if self.cfg.attention_mode == "causal":
        causal = torch.ones(l, l, dtype=torch.bool).tril()
        allowed = allowed & causal[None, None]
    bias = torch.zeros((b, 1, l, l), dtype=dtype)
    bias.masked_fill_(~allowed, torch.finfo(dtype).min)
    return False, bias, valid


# ---------------------------------------------------------------------------
# Attention path numerics
# ---------------------------------------------------------------------------

def test_causal_nomask_matches_full_mask():
    """Causal is_causal-only path == full [B,1,L,L] mask for the pooled output (right-padded input)."""
    torch.manual_seed(0)
    text = _right_padded_batch([14, 9, 6, 11, 3], total_len=14)
    for pool in ("eos", "mean"):
        m = _make_modern_text(attention_mode="causal", pool_type=pool)
        with torch.no_grad():
            new = m(text)
            m._attn_inputs = types.MethodType(_old_attn_inputs, m)
            old = m(text)
        assert torch.equal(new, old), f"pool={pool}: causal no-mask diverged from full-mask reference"


def test_bidirectional_keypad_mask_excludes_padding():
    """A real token's output is invariant to whether a pad tail is present (the key-pad mask masks pads)."""
    torch.manual_seed(0)
    m = _make_modern_text(attention_mode="bidirectional", pool_type="mean", qk_norm=False, attn_gated=False)
    prefix_len = 6
    text = torch.randint(0, EOS_ID, (2, 10))
    text[:, prefix_len:] = PAD_ID
    with torch.no_grad():
        with_pad = m(text)
        truncated = m(text[:, :prefix_len])  # same prefix, no pad tail at all
    assert torch.allclose(with_pad, truncated, atol=1e-5)


def test_modern_text_causal_rope_padding_invariant():
    model = ModernTextTransformer(_modern_cfg(pool_type="eos", attention_mode="causal"), output_dim=16).eval()
    short = torch.tensor([[1, 7, 2]], dtype=torch.long)
    padded = torch.tensor([[1, 7, 2, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        out_short = model(short)
        out_padded = model(padded)

    assert out_short.shape == (1, 16)
    assert torch.isfinite(out_padded).all()
    assert torch.allclose(out_short, out_padded, atol=1e-5)


def test_modern_text_bidirectional_map_padding_invariant():
    model = ModernTextTransformer(
        _modern_cfg(pool_type="map", attention_mode="bidirectional", attn_gated=True),
        output_dim=16,
    ).eval()
    short = torch.tensor([[1, 5, 9, 2]], dtype=torch.long)
    padded = torch.tensor([[1, 5, 9, 2, 0, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        out_short = model(short)
        out_padded = model(padded)

    assert out_short.shape == (1, 16)
    assert torch.isfinite(out_padded).all()
    assert torch.allclose(out_short, out_padded, atol=1e-5)


def test_modern_text_arch_options_padding_invariant():
    """pre_norm / sandwich norms / registers / value residual / relu2 all preserve causal padding invariance
    (pooled output unchanged by a pad tail), individually and combined."""
    torch.manual_seed(0)
    option_sets = (
        {"pre_norm": True},
        {"norm_placement": "sandwich"},
        {"reg_tokens": 4},
        {"value_residual": True},
        {"mlp_type": "relu2"},
        {"pre_norm": True, "norm_placement": "sandwich", "reg_tokens": 2, "value_residual": True,
         "mlp_type": "relu2"},
    )
    short = _right_padded_batch([6, 4], total_len=6)
    padded = torch.cat([short, torch.full((2, 4), PAD_ID)], dim=1)
    for opts in option_sets:
        m = _make_modern_text(**opts)
        with torch.no_grad():
            assert torch.allclose(m(short), m(padded), atol=1e-5), f"options {opts} broke padding invariance"


def test_modern_text_registers_excluded_from_outputs():
    """Registers are prepended internally but never leak: token outputs, intermediates, and pooling all cover
    text positions only; registers surface via output_extra_tokens and are excluded from weight decay."""
    m = _make_modern_text(reg_tokens=3, output_tokens=True)
    text = _right_padded_batch([5, 7], total_len=8)
    with torch.no_grad():
        pooled, tokens = m(text)
        inter = m.forward_intermediates(text, indices=1, output_extra_tokens=True)
    assert pooled.shape == (2, 48)
    assert tokens.shape[:2] == (2, text.shape[1])
    assert inter["text_intermediates"][0].shape[1] == text.shape[1]
    assert inter["text_intermediates_extra"][0].shape[1] == 3
    assert "reg_tokens" in m.no_weight_decay()
    assert any(p is m.reg_tokens for _, members in m.layer_groups() for p in members)


def test_modern_text_value_residual_params_and_grads():
    """Layer 0 produces v_first (no lambda -- an unused param breaks DDP); later layers mix with a learned
    scalar that receives gradient."""
    m = _make_modern_text(value_residual=True, layers=3).train()
    assert m.blocks[0].attn.vr_lambda is None
    assert all(b.attn.vr_lambda is not None for b in m.blocks[1:])
    m(_right_padded_batch([5, 3], total_len=6)).sum().backward()
    assert all(b.attn.vr_lambda.grad is not None for b in m.blocks[1:])


def test_modern_text_low_precision_conversion():
    """convert_weights_to_lp converts module weights but not standalone params (vr_lambda, pool.query,
    reg_tokens); those stay fp32 and must cast at use, so fp16/bf16-converted towers still run."""
    from open_clip.model import convert_weights_to_lp

    text = _right_padded_batch([6, 4], total_len=8)
    for dtype in (torch.bfloat16, torch.float16):
        m = _make_modern_text(pool_type="map", value_residual=True, reg_tokens=2, norm_placement="sandwich")
        convert_weights_to_lp(m, dtype=dtype)
        assert m.blocks[1].attn.vr_lambda.dtype == torch.float32  # standalone params stay fp32 by design
        assert m.pool.query.dtype == torch.float32
        with torch.no_grad():
            out = m(text)
        assert out.dtype == dtype
        assert torch.isfinite(out.float()).all()


def test_all_pad_row_is_finite():
    """Degenerate all-pad row must not produce NaNs (the _valid_mask fallback forces token 0 valid)."""
    m = _make_modern_text(attention_mode="causal", pool_type="eos")
    all_pad = torch.full((1, 8), PAD_ID)
    with torch.no_grad():
        out = m(all_pad)
    assert torch.isfinite(out).all()


def test_modern_text_without_pad_id_treats_zero_as_valid():
    model = ModernTextTransformer(_modern_cfg(pad_id=None, pool_type="mean"), output_dim=16).eval()
    text = torch.tensor([[1, 0, 2]], dtype=torch.long)

    with torch.no_grad():
        out = model(text)

    assert out.shape == (1, 16)
    assert torch.isfinite(out).all()
    assert model._valid_mask(text).all()


def test_init_scheme_depth_scaled_output_projections():
    """Block weights follow the GPT-2/CLIP scheme: input projs ~ width**-0.5, residual output projs additionally
    scaled by (2*layers)**-0.5; norms = 1.0; biases = 0. (Catches a fallback to PyTorch's default Linear init.)"""
    torch.manual_seed(0)
    layers = 12
    # Force biases on so the zero-init checks below exercise that path (they default off now).
    m = _make_modern_text(
        attention_mode="causal", pool_type="map", qk_norm=True, attn_gated=True, layers=layers,
        attention_bias=True, mlp_bias=True,
    )
    width = m.width
    attn_std = width ** -0.5
    proj_std = attn_std * ((2 * layers) ** -0.5)
    fc_std = (2 * width) ** -0.5

    qkv = torch.stack([b.attn.qkv.weight for b in m.blocks])
    attn_proj = torch.stack([b.attn.proj.weight for b in m.blocks])
    w12 = torch.stack([b.mlp.w12.weight for b in m.blocks])
    w3 = torch.stack([b.mlp.w3.weight for b in m.blocks])

    # Empirical std within ~15% of target (averaged over all layers -> tight). w12 carries the 1.22x SwiGLU
    # correction (gate product u*silu(v) loses ~2.4x variance vs the GELU hidden the scheme assumes).
    assert qkv.std().item() == pytest.approx(attn_std, rel=0.15)
    assert attn_proj.std().item() == pytest.approx(proj_std, rel=0.15)
    assert w12.std().item() == pytest.approx(fc_std * 1.22, rel=0.15)
    assert w3.std().item() == pytest.approx(proj_std, rel=0.15)
    # The depth term must actually shrink the output projections relative to the input projections.
    assert attn_proj.std().item() < 0.5 * qkv.std().item()

    # RMSNorm / qk_norm weights are ones; block + pool biases are zero except the attention gate.
    for b in m.blocks:
        assert torch.equal(b.norm1.weight, torch.ones_like(b.norm1.weight))
        assert torch.equal(b.attn.q_norm.weight, torch.ones_like(b.attn.q_norm.weight))
        for bias in (b.attn.qkv.bias, b.attn.proj.bias, b.mlp.w12.bias, b.mlp.w3.bias):
            assert torch.count_nonzero(bias) == 0
    assert torch.count_nonzero(m.pool.q.bias) == 0
    assert torch.count_nonzero(m.pool.kv.bias) == 0
    # Gated attention starts mostly open: gate bias 1 -> sigmoid(1) ~= 0.73 (near-transparent at init).
    assert torch.equal(m.blocks[0].attn.gate.bias, torch.ones_like(m.blocks[0].attn.gate.bias))


def test_linear_bias_flags():
    """attention_bias / mlp_bias gate every Linear bias (default off = modern); proj_bias is the head, separate."""
    # Defaults are tri-state None on the shared cfg; the modern tower resolves None -> off (no bias).
    assert CLIPTextCfg().attention_bias is None and CLIPTextCfg().mlp_bias is None
    off = _make_modern_text(pool_type="map", attn_gated=True)  # None -> modern default off
    b = off.blocks[0]
    for layer in (b.attn.qkv, b.attn.proj, b.attn.gate, off.pool.q, off.pool.kv, b.mlp.w12, b.mlp.w3):
        assert layer.bias is None

    on = _make_modern_text(pool_type="map", attn_gated=True, attention_bias=True, mlp_bias=True)
    b = on.blocks[0]
    for layer in (b.attn.qkv, b.attn.proj, b.attn.gate, on.pool.q, on.pool.kv):  # attention_bias group
        assert layer.bias is not None
    for layer in (b.mlp.w12, b.mlp.w3):  # mlp_bias group
        assert layer.bias is not None

    # The two flags are independent.
    attn_only = _make_modern_text(pool_type="map", attn_gated=True, attention_bias=True, mlp_bias=False)
    assert attn_only.blocks[0].attn.qkv.bias is not None
    assert attn_only.blocks[0].mlp.w12.bias is None


def test_gate_bias_override():
    """gate_bias is tri-state: None inherits attention_bias; True/False force the gate's own bias on/off."""
    # Default (None) follows attention_bias.
    assert _make_modern_text(attn_gated=True, attention_bias=False).blocks[0].attn.gate.bias is None
    assert _make_modern_text(attn_gated=True, attention_bias=True).blocks[0].attn.gate.bias is not None

    # Force ON with attention_bias off: gate keeps a bias, and it is inited mostly-open (==1 -> sigmoid~=0.73)
    # while qkv/proj stay bias-free.
    forced_on = _make_modern_text(attn_gated=True, attention_bias=False, gate_bias=True).blocks[0].attn
    assert forced_on.qkv.bias is None and forced_on.proj.bias is None
    assert forced_on.gate.bias is not None
    assert torch.allclose(forced_on.gate.bias, torch.ones_like(forced_on.gate.bias))

    # Force OFF with attention_bias on: qkv/proj keep biases, gate does not.
    forced_off = _make_modern_text(attn_gated=True, attention_bias=True, gate_bias=False).blocks[0].attn
    assert forced_off.qkv.bias is not None
    assert forced_off.gate.bias is None


def test_map_pool_qk_norm():
    """cfg.qk_norm threads into the MAP pool's attention (same bf16-stability guard as the blocks)."""
    on = _make_modern_text(pool_type="map", qk_norm=True)
    assert type(on.pool.q_norm).__name__ == "RMSNorm"
    assert type(on.pool.k_norm).__name__ == "RMSNorm"
    off = _make_modern_text(pool_type="map", qk_norm=False)
    assert isinstance(off.pool.q_norm, torch.nn.Identity)
    assert isinstance(off.pool.k_norm, torch.nn.Identity)
    # The pool no longer carries a redundant kv pre-norm (ln_final already normalises the pool input).
    assert not hasattr(on.pool, "norm")
    # qk-norm follows the model's norm type rather than being hardcoded to RMSNorm.
    ln = _make_modern_text(pool_type="map", qk_norm=True, norm_type="layernorm")
    assert type(ln.pool.q_norm).__name__ == "LayerNorm"
    assert type(ln.pool.k_norm).__name__ == "LayerNorm"
    # forward still finite with the pool qk-norm active
    with torch.no_grad():
        out = on(_right_padded_batch([14, 9, 6], total_len=14))
    assert torch.isfinite(out).all()


def test_block_qk_norm_follows_norm_type():
    """The block attention's qk-norm uses the model's norm_layer (default RMSNorm), not a hardcoded type."""
    # norm_type is tri-state on the shared cfg; the modern tower resolves None -> rmsnorm.
    assert CLIPTextCfg().norm_type is None
    attn = _make_modern_text(qk_norm=True).blocks[0].attn  # no norm_type override -> None -> rmsnorm
    assert type(attn.q_norm).__name__ == "RMSNorm"
    assert type(attn.k_norm).__name__ == "RMSNorm"
    ln = _make_modern_text(qk_norm=True, norm_type="layernorm").blocks[0].attn
    assert type(ln.q_norm).__name__ == "LayerNorm"
    assert type(ln.k_norm).__name__ == "LayerNorm"
    off = _make_modern_text(qk_norm=False).blocks[0].attn
    assert isinstance(off.q_norm, torch.nn.Identity)


def test_all_norms_use_config_eps():
    """Every norm in the modern text tower is built from norm_layer, so all carry cfg.norm_eps -- including the
    block/pool qk-norm, which previously hardcoded RMSNorm's 1e-6 default and ignored the config."""
    eps = 4.2e-5  # distinct from any native norm default (e.g. LayerNorm 1e-5) so a match is meaningful
    for norm_type in ("rmsnorm", "layernorm"):
        # All norm-bearing features on: sandwich (norm*_post), pre_norm (norm_pre), map pool + qk-norm.
        m = _make_modern_text(
            pool_type="map", qk_norm=True, attn_gated=True, norm_type=norm_type, norm_eps=eps,
            norm_placement="sandwich", pre_norm=True,
        )
        norms = [mod for _, mod in m.named_modules() if isinstance(mod, (torch.nn.RMSNorm, LayerNorm))]
        assert norms, "expected norm modules in the tower"
        assert all(mod.eps == eps for mod in norms), f"{norm_type}: a norm did not pick up cfg.norm_eps"
        # The two that used to bypass cfg.norm_eps, called out explicitly.
        assert m.blocks[0].attn.q_norm.eps == eps
        assert m.pool.q_norm.eps == eps


def test_init_scheme_sandwich_uses_flat_init():
    """Sandwich norm placement switches block matrices to flat N(0, 0.02): the post-norms undo depth scaling,
    so the depth-scaled out-proj std must NOT be applied."""
    torch.manual_seed(0)
    layers = 12
    m = _make_modern_text(pool_type="map", layers=layers, norm_placement="sandwich")
    qkv = torch.stack([b.attn.qkv.weight for b in m.blocks])
    attn_proj = torch.stack([b.attn.proj.weight for b in m.blocks])
    w12 = torch.stack([b.mlp.w12.weight for b in m.blocks])
    assert qkv.std().item() == pytest.approx(0.02, rel=0.15)
    assert attn_proj.std().item() == pytest.approx(0.02, rel=0.15)  # flat, not 0.02 * (2*12)**-0.5
    assert w12.std().item() == pytest.approx(0.02, rel=0.15)  # no swiglu correction under flat init
    # Head/readout init is scheme-independent (CLIP convention), not flattened.
    assert m.pool.q.weight.std().item() == pytest.approx(m.width ** -0.5, rel=0.2)
    assert m.text_projection.weight.std().item() == pytest.approx(m.width ** -0.5, rel=0.2)


def test_init_scheme_zero_init_residual():
    """zero_init_residual zeroes the residual out-projs (blocks start as identity); inputs keep their std."""
    m = _make_modern_text(zero_init_residual=True)
    for b in m.blocks:
        assert torch.count_nonzero(b.attn.proj.weight) == 0
        assert torch.count_nonzero(b.mlp.w3.weight) == 0
        assert torch.count_nonzero(b.attn.qkv.weight) > 0
        assert torch.count_nonzero(b.mlp.w12.weight) > 0
    # Identity blocks at init: token outputs equal the (normed) embeddings regardless of depth.
    text = _right_padded_batch([5, 3], total_len=6)
    one = _make_modern_text(zero_init_residual=True, layers=1)
    deep = _make_modern_text(zero_init_residual=True, layers=4)
    deep.load_state_dict({k: v for k, v in one.state_dict().items() if k in deep.state_dict()
                          and deep.state_dict()[k].shape == v.shape}, strict=False)
    with torch.no_grad():
        x1, _, _ = one._encode_tokens(text)
        x4, _, _ = deep._encode_tokens(text)
    assert torch.allclose(x1, x4, atol=1e-6)


def test_register_tokens_blank_slate_init():
    """Registers follow the timm cls/reg convention: near-zero init (std 1e-6), not 0.02."""
    m = _make_modern_text(reg_tokens=4)
    assert m.reg_tokens.abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# Pooling semantics + special-token validation
# ---------------------------------------------------------------------------

def test_modern_pool_rejects_physical_last():
    """In open_clip 'last' means the last *physical* position (SigLIP/CLIPA); the masked tower rejects it."""
    with pytest.raises(ValueError, match="pool_type"):
        _make_modern_text(pool_type="last")


def test_modern_eos_pooling_requires_eos_id():
    """eos_id has no config default; 'eos' (and the 'argmax' remap) must fail fast without it."""
    for pool in ("eos", "argmax"):
        with pytest.raises(ValueError, match="eos_id"):
            _make_modern_text(pool_type=pool, eos_id=None)


def test_argmax_remaps_to_eos_pooling():
    """'argmax' (classic CLIP EOT pooling) and 'eos' produce identical features for the same weights."""
    torch.manual_seed(0)
    text = _right_padded_batch([10, 5, 7], total_len=10)
    m_eos = _make_modern_text(pool_type="eos")
    m_argmax = _make_modern_text(pool_type="argmax")
    m_argmax.load_state_dict(m_eos.state_dict())
    with torch.no_grad():
        assert torch.equal(m_eos(text), m_argmax(text))


def test_validate_special_tokens():
    """get_tokenizer's config/tokenizer cross-checks: eos for eos/argmax pooling, pad for explicit/variable text."""
    tok = types.SimpleNamespace(eot_token_id=50257, pad_token_id=50258)

    # Consistent config passes; non-eos pooling without eos_id passes.
    _validate_special_tokens({"pool_type": "eos", "eos_id": 50257, "pad_id": 50258, "variable_text": True}, tok)
    _validate_special_tokens({"pool_type": "argmax"}, tok)

    # Modern 'argmax' remaps to eos pooling, so it requires (and checks) eos_id.
    with pytest.raises(ValueError, match="eos_id"):
        _validate_special_tokens({"text_arch": "modern", "pool_type": "argmax"}, tok)
    with pytest.raises(ValueError, match="eos_id"):
        _validate_special_tokens({"pool_type": "eos", "eos_id": 2}, tok)

    # Explicit pad_id must match the tokenizer; variable_text requires a reserved pad id at all.
    with pytest.raises(ValueError, match="pad_id"):
        _validate_special_tokens({"pad_id": 0}, tok)
    with pytest.raises(ValueError, match="pad_token_id"):
        _validate_special_tokens({"variable_text": True}, types.SimpleNamespace(pad_token_id=None))
    # Tokenizers that expose no ids at all (e.g. SimpleTokenizer) skip the comparisons.
    _validate_special_tokens({"pool_type": "eos", "eos_id": 2, "pad_id": 0}, object())


# ---------------------------------------------------------------------------
# Public surface, model routing, configs, tokenizer wiring
# ---------------------------------------------------------------------------

def test_modern_text_public_surface_and_lock():
    model = ModernTextTransformer(_modern_cfg(pool_type="argmax"), output_dim=16)

    assert model.context_length == 8
    assert model.num_pos == 8
    assert model.cfg.context_length == 8
    assert model.variable_text is False
    assert model.vocab_size == 64
    assert model.width == 32
    assert model.layers == 2
    assert model.output_dim == 16
    assert model.pad_id == 0
    assert model.eos_id == 2
    assert model.token_embedding.num_embeddings == 64
    assert model.text_projection.out_features == 16
    assert not hasattr(model, "text_arch")
    assert not hasattr(model, "bos_id")
    assert not hasattr(model, "attention_mode")
    assert not hasattr(model, "pool_type")
    assert not hasattr(model, "attn_mask")
    assert [n for n, _ in model.layer_groups()] == ["embeddings", "layer.0", "layer.1", "proj"]

    model.lock(unlocked_layers=1)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert any(n.startswith("blocks.0.") for n in frozen)
    assert not any(n.startswith("text_projection.") for n in frozen)


def test_modern_text_validates_attention_geometry():
    with pytest.raises(ValueError, match=r"width \(30\) must be divisible by heads \(8\)"):
        ModernTextTransformer(_modern_cfg(width=30, heads=8), output_dim=16)

    with pytest.raises(ValueError, match=r"RoPE head dim must be even"):
        ModernTextTransformer(_modern_cfg(width=36, heads=4, pos_embed="rope"), output_dim=16)

    ModernTextTransformer(_modern_cfg(width=36, heads=4, pos_embed="none"), output_dim=16)


def _write_model_config(tmp_path, *, custom_text: bool):
    model_cfg = {
        "embed_dim": 16,
        "vision_cfg": {
            "image_size": 16,
            "patch_size": 8,
            "width": 32,
            "layers": 1,
            "head_width": 16,
        },
        "text_cfg": {
            "text_arch": "modern",
            "context_length": 8,
            "vocab_size": 64,
            "width": 32,
            "heads": 4,
            "layers": 1,
            "mlp_ratio": 2.0,
            "pad_id": 0,
            "eos_id": 2,
            "pool_type": "eos",
            "attention_mode": "causal",
            "attn_gated": True,
        },
    }
    if custom_text:
        model_cfg["custom_text"] = True
    config_dir = tmp_path / ("modern_custom" if custom_text else "modern_legacy")
    config_dir.mkdir()
    (config_dir / "open_clip_config.json").write_text(json.dumps({"model_cfg": model_cfg}), encoding="utf-8")
    return f"local-dir:{config_dir}"


def _write_tiktoken_model_config(tmp_path):
    model_cfg = {
        "embed_dim": 16,
        "vision_cfg": {
            "image_size": 16,
            "patch_size": 8,
            "width": 32,
            "layers": 1,
            "head_width": 16,
        },
        "text_cfg": {
            "text_arch": "modern",
            "context_length": 16,
            "vocab_size": 100280,
            "width": 32,
            "heads": 4,
            "layers": 1,
            "mlp_ratio": 2.0,
            "pad_id": 100278,
            "bos_id": 100279,
            "eos_id": 100277,
            "pool_type": "eos",
            "attention_mode": "causal",
            "tokenizer_type": "tiktoken",
            "tiktoken_name": "cl100k_base",
        },
    }
    config_dir = tmp_path / "modern_tiktoken"
    config_dir.mkdir()
    (config_dir / "open_clip_config.json").write_text(json.dumps({"model_cfg": model_cfg}), encoding="utf-8")
    return f"local-dir:{config_dir}"


def test_modern_text_auto_uses_custom_text_clip(tmp_path):
    model = open_clip.create_model(_write_model_config(tmp_path, custom_text=False), load_weights=False)

    assert isinstance(model, open_clip.CustomTextCLIP)
    assert isinstance(model.text, ModernTextTransformer)


def test_modern_text_rejects_direct_legacy_clip():
    with pytest.raises(ValueError, match="requires CustomTextCLIP"):
        CLIP(
            embed_dim=16,
            vision_cfg=CLIPVisionCfg(image_size=16, patch_size=8, width=32, layers=1, head_width=16),
            text_cfg=_modern_cfg(),
        )


def test_modern_text_custom_text_clip_forward(tmp_path):
    model = open_clip.create_model(
        _write_model_config(tmp_path, custom_text=True),
        load_weights=False,
        output_dict=True,
    ).eval()

    assert isinstance(model, open_clip.CustomTextCLIP)
    assert isinstance(model.text, ModernTextTransformer)
    image = torch.randn(2, 3, 16, 16)
    text = torch.tensor([[1, 6, 2, 0, 0], [1, 8, 9, 2, 0]], dtype=torch.long)

    with torch.no_grad():
        out = model(image=image, text=text)

    assert out["image_features"].shape == (2, 16)
    assert out["text_features"].shape == (2, 16)
    assert torch.isfinite(out["image_features"]).all()
    assert torch.isfinite(out["text_features"]).all()


def test_modern_text_tiktoken_config_builds_model_and_tokenizer(tmp_path):
    pytest.importorskip("tiktoken")
    model_name = _write_tiktoken_model_config(tmp_path)

    model = open_clip.create_model(model_name, load_weights=False)
    tokenizer = open_clip.get_tokenizer(model_name)
    text = tokenizer(["a short caption"])

    assert isinstance(model, open_clip.CustomTextCLIP)
    assert isinstance(model.text, ModernTextTransformer)
    assert model.text.vocab_size == tokenizer.vocab_size
    assert model.text.pad_id == tokenizer.pad_token_id
    assert model.text.eos_id == tokenizer.eot_token_id
    assert text.shape == (1, model.context_length)
    assert text[0, 0].item() == tokenizer.bos_token_id
    assert text[0, (text[0] != tokenizer.pad_token_id).nonzero()[-1].item()].item() == model.text.eos_id


def test_modern_text_and_alt_builtin_configs_registered():
    modern = open_clip.get_model_config("moderntext-ViT-B-32-256")
    gte = open_clip.get_model_config("gte-modernbert-base-ViT-B-32-256")
    naflex = open_clip.get_model_config("moderntext-naflex_ViT-B-32")

    assert modern["vision_cfg"]["image_size"] == 256
    assert modern["text_cfg"]["text_arch"] == "modern"
    assert modern["text_cfg"]["variable_text"] is True
    assert modern["text_cfg"]["tiktoken_name"] == "r50k_base"
    assert modern["text_cfg"]["vocab_size"] == 50260
    assert gte["text_cfg"]["hf_model_name"] == "Alibaba-NLP/gte-modernbert-base"
    assert gte["text_cfg"]["variable_text"] is True
    assert gte["text_cfg"]["hf_pooler_type"] == "cls_pooler"
    assert gte["text_cfg"]["hf_model_config"]["max_position_embeddings"] == 256
    assert naflex["vision_cfg"]["image_size"] == 224
    assert naflex["vision_cfg"]["image_seq_len"] == 49
    assert naflex["vision_cfg"]["timm_model_kwargs"]["pos_embed_grid_size"] == [7, 7]
    assert naflex["text_cfg"]["text_arch"] == "modern"
    assert naflex["text_cfg"]["variable_text"] is True

    model = open_clip.create_model("moderntext-ViT-B-32-256", load_weights=False)
    assert isinstance(model, open_clip.CustomTextCLIP)
    assert isinstance(model.text, ModernTextTransformer)
    assert model.text.variable_text is True
    assert model.context_length == 256
    assert model.vocab_size == 50260


def test_simple_tokenizer_rejects_variable_text():
    tokenizer = SimpleTokenizer(context_length=8)

    with pytest.raises(ValueError, match="does not support variable-length"):
        tokenizer("hello world", pad=False)


# ---------------------------------------------------------------------------
# Variable-length text collators
# ---------------------------------------------------------------------------

def test_collate_variable_text_pad_multiple():
    seqs = [torch.arange(5), torch.arange(11), torch.arange(3)]
    text, valid = collate_variable_text(seqs, pad_id=PAD_ID, pad_multiple=8)
    assert text.shape == (3, 16)  # max real length 11 rounded up to a multiple of 8
    assert valid.sum().item() == 5 + 11 + 3  # only the real tokens are valid
    # No multiple -> exact batch max.
    text2, _ = collate_variable_text(seqs, pad_id=PAD_ID)
    assert text2.shape == (3, 11)


def test_collate_variable_text_pad_cap_clamps_rounding():
    # A max-length caption (= the truncation cap) must not be rounded past the cap: absolute-position towers
    # index positional_embedding[:seq_len] and would crash on seq_len > context_length.
    seqs = [torch.arange(13), torch.arange(3)]
    text, _ = collate_variable_text(seqs, pad_id=PAD_ID, pad_multiple=8, pad_cap=13)
    assert text.shape == (2, 13)  # ceil(13/8)*8 = 16, clamped at the cap
    # Cap above the rounded value is a no-op.
    text2, _ = collate_variable_text(seqs, pad_id=PAD_ID, pad_multiple=8, pad_cap=32)
    assert text2.shape == (2, 16)


def test_collate_variable_text_pad_cap_rejects_overlong_sequences():
    # pad_cap is an upper-bound contract: a sequence longer than the cap means the tokenizer failed to
    # truncate. Raise loudly (with or without pad_multiple) instead of silently chopping the tail.
    with pytest.raises(ValueError, match="exceeds pad_cap"):
        collate_variable_text([torch.arange(20)], pad_id=PAD_ID, pad_multiple=8, pad_cap=13)
    with pytest.raises(ValueError, match="exceeds pad_cap"):
        collate_variable_text([torch.arange(20)], pad_id=PAD_ID, pad_cap=13)


def test_collate_variable_text_dicts_emits_text_valid():
    batch = [
        {"image": torch.zeros(3, 4, 4), "text": torch.arange(5)},
        {"image": torch.zeros(3, 4, 4), "text": torch.arange(9)},
    ]
    out = collate_variable_text_dicts(batch, pad_id=PAD_ID, text_pad_multiple=8)
    assert out["text"].shape[1] == 16  # 9 rounded up to a multiple of 8
    # text_valid is always present; tasks pick the keys they consume.
    assert torch.equal(out["text_valid"], out["text"] != PAD_ID)


def _audio_sample(text):
    return {"audio": {"waveform": torch.zeros(16), "longer": False}, "text": text}


def test_audio_collate_variable_text_pad_multiple():
    batch = [_audio_sample(torch.arange(5)), _audio_sample(torch.arange(9))]
    out = _audio_collate(batch, pad_id=PAD_ID, text_pad_multiple=8)
    assert out["text"].shape[1] == 16  # 9 rounded up to a multiple of 8
    assert torch.equal(out["text_valid"], out["text"] != PAD_ID)
    # The cap bounds rounding at the tokenizer truncation length.
    out2 = _audio_collate(batch, pad_id=PAD_ID, text_pad_multiple=8, text_pad_cap=9)
    assert out2["text"].shape[1] == 9
