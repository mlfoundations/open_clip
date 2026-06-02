""" NaFlex GenLIP model.

A standalone implementation of GenLIP ("Let ViT Speak: Generative Language-Image Pre-training",
arXiv 2605.00809) adapted to open_clip's NaFlex data pipeline.

GenLIP pretrains a ViT vision encoder with a single Transformer and a single autoregressive language
modeling objective -- no contrastive loss, no dual tower, no separate text decoder. For an (image, caption)
pair the image is patchified to patch embeddings ``v0..vM``, the caption tokenized to ``t0..tL``, and the two
are concatenated into one sequence ``S = [v0..vM, t0..tL]`` run through a unified trunk with:

- **prefix-LM attention**: image patches attend bidirectionally among themselves; text attends causally;
  text attends to image; image never attends to text.
- **MRoPE**: 3-axis (T/H/W) interleaved rotary positions on Q/K. Absolute position embeddings are discarded.
- **gated attention**: ``q_proj`` emits ``2*width`` channels split into a query and a sigmoid gate applied to
  the attention output (the paper's attention-sink fix).
- **SwiGLU** FFN, **LayerScale**, and **DropPath**.

The cross-entropy LM loss is applied only to text tokens (image patches and padding are ``ignore_index``).
Used as a pure vision encoder the LM head / text embedding are discarded and prefix-LM degrades to full
attention over the image patches.

Differences from the reference (by design): the image side reuses open_clip's NaFlex data pipeline with a
timm-``NaFlexVit``-style linear patch embedding on pre-patchified inputs, and there is **no** cross-sample
sequence packing -- a batch is a set of padded rows, each row ``[one image's patches ; its caption tokens]``.
Attention uses torch SDPA with a dense per-sample boolean prefix-LM mask.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

try:
    from timm.layers import DropPath
except ImportError:
    DropPath = None

from .loss import fused_linear_cross_entropy


@dataclass
class NaFlexGenLipVisionCfg:
    image_size: int = 256
    patch_size: int = 16
    in_chans: int = 3
    proj_bias: bool = True
    input_norm: bool = False  # apply LayerNorm to the flat patch input before projection
    pool_type: str = 'avg'  # masked pooling used for vision-encoder-only feature extraction


@dataclass
class NaFlexGenLipTextCfg:
    vocab_size: int = 100280
    context_length: int = 256  # default/max caption length used for dummy batches
    pad_id: int = 100278
    bos_id: int = 100279
    eos_id: int = 100277
    tokenizer_type: str = 'tiktoken'
    tiktoken_name: str = 'cl100k_base'


@dataclass
class NaFlexGenLipTrunkCfg:
    width: int = 1152  # transformer hidden size
    depth: int = 27
    num_heads: int = 16
    intermediate_size: int = 3072
    text_embed_dim: int = 1024  # token embedding / lm_head dim (projected to/from width)
    mrope_section: Tuple[int, int, int] = (12, 12, 12)
    rope_theta: float = 10000.0
    ls_init_value: float = 0.1
    drop_path_rate: float = 0.0
    gated_attention: bool = True
    use_swiglu_ffn: bool = True
    mrope_interleaved: bool = True
    hidden_act: str = 'silu'
    layer_norm_eps: float = 1e-6
    max_position_embeddings: int = 16384
    # When True, the generative (compute_loss) path compacts each row to [valid prefix ; valid text ; PAD]
    # instead of the fixed [prefix-block ; text-block] layout -- removes wasted padding between the prefix and
    # text (big win for variable-length audio). Default False = current block layout (existing runs unchanged).
    pack_prefix: bool = False


_ACT_FN = {
    'silu': F.silu,
    'gelu': F.gelu,
    'relu': F.relu,
}


# ---------------------------------------------------------------------------------------------------------------------
# Rotary (interleaved MRoPE)
# ---------------------------------------------------------------------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to the first (negated)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: Tuple[int, int, int]) -> torch.Tensor:
    """Reorganize chunked ``[T..H..W..]`` frequencies into interleaved ``[THWTHW..]``.

    Args:
        freqs: Per-axis frequencies of shape ``(3, B, S, head_dim // 2)``.
        mrope_section: Channel split for the temporal, height and width axes.

    Returns:
        Interleaved frequencies of shape ``(B, S, head_dim // 2)``.
    """
    freqs_t = freqs[0].clone()  # start from the temporal axis, overwrite H/W channels
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply interleaved-MRoPE cos/sin to query and key tensors.

    Args:
        q: Query of shape ``(B, num_heads, S, head_dim)``.
        k: Key of shape ``(B, num_heads, S, head_dim)``.
        cos: Cosine of shape ``(B, S, head_dim)``.
        sin: Sine of shape ``(B, S, head_dim)``.
        unsqueeze_dim: Dimension to unsqueeze cos/sin on so they broadcast over heads.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GenLipRotaryEmbedding(nn.Module):
    """3-axis multimodal rotary position embedding (interleaved MRoPE)."""

    def __init__(self, cfg: NaFlexGenLipTrunkCfg):
        super().__init__()
        head_dim = cfg.width // cfg.num_heads
        if sum(cfg.mrope_section) != head_dim // 2:
            raise ValueError(
                f"sum(mrope_section)={sum(cfg.mrope_section)} must equal head_dim//2={head_dim // 2} "
                f"(head_dim={head_dim}). Check num_heads/width vs mrope_section."
            )
        self.mrope_section = tuple(cfg.mrope_section)
        self.mrope_interleaved = cfg.mrope_interleaved
        inv_freq = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for the given position ids.

        Args:
            x: A tensor only used for device/dtype reference.
            position_ids: Long tensor of shape ``(3, B, S)`` with T/H/W positions.

        Returns:
            ``(cos, sin)`` each of shape ``(B, S, head_dim)`` cast to ``x.dtype``.
        """
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()  # (3, B, 1, S)
        device_type = x.device.type if x.device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ pos).transpose(2, 3)  # (3, B, S, head_dim//2)
            if self.mrope_interleaved:
                freqs = apply_interleaved_mrope(freqs, self.mrope_section)  # (B, S, head_dim//2)
            else:
                freqs = freqs[0]
            emb = torch.cat((freqs, freqs), dim=-1)  # (B, S, head_dim)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# Attention / FFN / blocks
# ---------------------------------------------------------------------------------------------------------------------
class GenLipAttention(nn.Module):
    """Multi-head self-attention with optional gating, MRoPE, and an SDPA backend."""

    def __init__(self, cfg: NaFlexGenLipTrunkCfg):
        super().__init__()
        self.width = cfg.width
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.width // cfg.num_heads
        self.scale = self.head_dim ** -0.5
        self.gated_attention = cfg.gated_attention

        self.k_proj = nn.Linear(self.width, self.width)
        self.v_proj = nn.Linear(self.width, self.width)
        self.q_proj = nn.Linear(self.width, self.width * 2 if self.gated_attention else self.width)
        self.out_proj = nn.Linear(self.width, self.width)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            cos: torch.Tensor,
            sin: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            x: Hidden states of shape ``(B, S, width)``.
            attn_mask: Boolean mask ``(B, 1, S, S)`` where True means the pair may attend.
            cos: MRoPE cosine of shape ``(B, S, head_dim)``.
            sin: MRoPE sine of shape ``(B, S, head_dim)``.
        """
        bs, seq_len, _ = x.shape

        if self.gated_attention:
            q, gate = self.q_proj(x).chunk(2, dim=-1)
            gate = gate.reshape(bs, seq_len, self.num_heads, self.head_dim)
        else:
            q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, H, S, head_dim)
        attn_out = attn_out.transpose(1, 2)  # (B, S, H, head_dim)

        if self.gated_attention:
            attn_out = attn_out * torch.sigmoid(gate)

        attn_out = attn_out.reshape(bs, seq_len, self.width)
        return self.out_proj(attn_out)


class GenLipSwiGLUFFN(nn.Module):
    def __init__(self, cfg: NaFlexGenLipTrunkCfg):
        super().__init__()
        self.act = _ACT_FN[cfg.hidden_act]
        self.fc1 = nn.Linear(cfg.width, cfg.intermediate_size)
        self.gate_fc = nn.Linear(cfg.width, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.gate_fc(x)) * self.fc1(x))


class GenLipMLP(nn.Module):
    def __init__(self, cfg: NaFlexGenLipTrunkCfg):
        super().__init__()
        self.act = _ACT_FN[cfg.hidden_act]
        self.fc1 = nn.Linear(cfg.width, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GenLipLayerScale(nn.Module):
    def __init__(self, width: int, init_value: float = 0.1):
        super().__init__()
        self.lambda1 = nn.Parameter(init_value * torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lambda1


class GenLipBlock(nn.Module):
    """Pre-norm transformer block: gated attention + SwiGLU, with LayerScale and DropPath."""

    def __init__(self, cfg: NaFlexGenLipTrunkCfg):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.width, eps=cfg.layer_norm_eps)
        self.self_attn = GenLipAttention(cfg)
        self.layer_norm2 = nn.LayerNorm(cfg.width, eps=cfg.layer_norm_eps)
        self.mlp = GenLipSwiGLUFFN(cfg) if cfg.use_swiglu_ffn else GenLipMLP(cfg)

        use_ls = cfg.ls_init_value is not None and cfg.ls_init_value > 1e-6
        self.layer_scale1 = GenLipLayerScale(cfg.width, cfg.ls_init_value) if use_ls else nn.Identity()
        self.layer_scale2 = GenLipLayerScale(cfg.width, cfg.ls_init_value) if use_ls else nn.Identity()

        if cfg.drop_path_rate > 1e-6:
            if DropPath is None:
                raise ImportError("timm is required for drop_path_rate > 0 (timm.layers.DropPath).")
            self.drop_path = DropPath(cfg.drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            cos: torch.Tensor,
            sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale1(self.self_attn(self.layer_norm1(x), attn_mask, cos, sin)))
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.layer_norm2(x))))
        return x


class GenLipTrunk(nn.Module):
    """Stack of GenLIP blocks followed by a final LayerNorm (``ln_post``)."""

    def __init__(self, cfg: NaFlexGenLipTrunkCfg):
        super().__init__()
        self.layers = nn.ModuleList([GenLipBlock(cfg) for _ in range(cfg.depth)])
        self.ln_post = nn.LayerNorm(cfg.width, eps=cfg.layer_norm_eps)
        self.grad_checkpointing = False

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            cos: torch.Tensor,
            sin: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
                x = torch.utils.checkpoint.checkpoint(layer, x, attn_mask, cos, sin, use_reentrant=False)
            else:
                x = layer(x, attn_mask, cos, sin)
        return self.ln_post(x)


# ---------------------------------------------------------------------------------------------------------------------
# Patch embedding (timm NaFlexVit linear-patch-embed style)
# ---------------------------------------------------------------------------------------------------------------------
class GenLipPatchEmbed(nn.Module):
    """Linear patch embedding over pre-patchified inputs ``[B, N, P*P*C]`` (no position embedding)."""

    def __init__(self, vision_cfg: NaFlexGenLipVisionCfg, width: int):
        super().__init__()
        patch_dim = vision_cfg.patch_size * vision_cfg.patch_size * vision_cfg.in_chans
        self.norm_input = nn.LayerNorm(patch_dim) if vision_cfg.input_norm else None
        self.proj = nn.Linear(patch_dim, width, bias=vision_cfg.proj_bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if self.norm_input is not None:
            patches = self.norm_input(patches)
        return self.proj(patches)


# ---------------------------------------------------------------------------------------------------------------------
# Mask / position-id builders for the no-packing "rows" batch
# ---------------------------------------------------------------------------------------------------------------------
def build_prefix_lm_mask(patch_valid: torch.Tensor, text_valid: torch.Tensor) -> torch.Tensor:
    """Build a boolean prefix-LM attention mask for ``[image_patches ; text_tokens]`` rows.

    Allowed pairs: image<->image (bidirectional), text->text (causal), text->image. Image never attends to
    text. Padding keys are removed. The diagonal is forced True so SDPA never sees a fully-masked query row.

    Args:
        patch_valid: ``(B, Ni)`` bool, True for valid image patches.
        text_valid: ``(B, Lt)`` bool, True for valid (non-pad) caption tokens.

    Returns:
        Boolean mask of shape ``(B, 1, S, S)`` with ``S = Ni + Lt`` (True means the pair may attend).
    """
    pv = patch_valid.bool()
    tv = text_valid.bool()
    b, ni = pv.shape
    lt = tv.shape[1]
    s = ni + lt
    device = pv.device

    valid = torch.cat([pv, tv], dim=1)  # (B, S)
    is_img = torch.zeros(s, dtype=torch.bool, device=device)
    is_img[:ni] = True
    is_txt = ~is_img

    causal = torch.tril(torch.ones(s, s, dtype=torch.bool, device=device))
    allowed = (
        (is_img[:, None] & is_img[None, :])
        | (is_txt[:, None] & is_txt[None, :] & causal)
        | (is_txt[:, None] & is_img[None, :])
    )  # (S, S)
    allowed = allowed[None].expand(b, s, s).clone()
    allowed &= valid[:, None, :]  # remove padding keys

    idx = torch.arange(s, device=device)
    allowed[:, idx, idx] = True  # guard against all-masked query rows
    return allowed.unsqueeze(1)


def build_packed_prefix_lm_mask(prefix_pos: torch.Tensor, text_pos: torch.Tensor) -> torch.Tensor:
    """Prefix-LM mask for the packed ``[valid prefix ; valid text ; PAD]`` layout, shape ``(B, 1, T, T)``.

    Same allowed pairs as :func:`build_prefix_lm_mask` (prefix<->prefix bidirectional, text->text causal,
    text->prefix), but the prefix/text split is per-row (``prefix_pos``/``text_pos`` are ``(B, T)`` boolean
    masks marking each row's prefix and text positions); trailing PAD positions are masked, diagonal forced.
    """
    b, t = prefix_pos.shape
    device = prefix_pos.device
    valid = prefix_pos | text_pos
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))
    allowed = (
        (prefix_pos[:, :, None] & prefix_pos[:, None, :])
        | (text_pos[:, :, None] & text_pos[:, None, :] & causal[None])
        | (text_pos[:, :, None] & prefix_pos[:, None, :])
    )
    allowed = allowed & valid[:, None, :]  # remove padding keys
    idx = torch.arange(t, device=device)
    allowed[:, idx, idx] = True  # guard against all-masked query rows (the trailing PAD)
    return allowed.unsqueeze(1)


def pack_prefix_sequence(
        prefix_emb: torch.Tensor,
        prefix_valid: torch.Tensor,
        block_pos: torch.Tensor,
        text_emb: torch.Tensor,
        text_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact each row to ``[valid prefix ; valid text ; PAD]`` (drops the padding between prefix and text).

    Assumes valid tokens are front-contiguous (the NaFlex collate / variable-text collate pad at the end).

    Args:
        prefix_emb: ``(B, Np, W)`` embedded prefix (image/audio) tokens.
        prefix_valid: ``(B, Np)`` bool.
        block_pos: ``(3, B, Np + Nt)`` MRoPE position ids from the standard block builder (prefix positions
            then continuing text positions) -- compacted here rather than recomputed.
        text_emb: ``(B, Nt, W)`` embedded caption tokens.
        text_valid: ``(B, Nt)`` bool.

    Returns:
        ``(combined_emb (B,T,W), combined_pos (3,B,T), attn_mask (B,1,T,T), prefix_lens (B,), text_lens (B,))``
        with ``T = max_i(k_i + m_i)``.
    """
    b, n_prefix, width = prefix_emb.shape
    n_text = text_emb.shape[1]
    device = prefix_emb.device
    k = prefix_valid.bool().sum(dim=1)  # (B,) valid prefix lengths
    m = text_valid.bool().sum(dim=1)    # (B,) valid text lengths
    seq_t = int((k + m).amax().item())

    cols = torch.arange(seq_t, device=device)
    prefix_dst = cols[None, :] < k[:, None]                                          # (B, T)
    text_dst = (cols[None, :] >= k[:, None]) & (cols[None, :] < (k + m)[:, None])     # (B, T)
    prefix_src = torch.arange(n_prefix, device=device)[None, :] < k[:, None]          # (B, Np) front-valid
    text_src = torch.arange(n_text, device=device)[None, :] < m[:, None]              # (B, Nt) front-valid

    combined = prefix_emb.new_zeros(b, seq_t, width)
    combined[prefix_dst] = prefix_emb[prefix_src]
    combined[text_dst] = text_emb[text_src]

    pos = block_pos.new_zeros(3, b, seq_t)
    pos[:, prefix_dst] = block_pos[:, :, :n_prefix][:, prefix_src]
    pos[:, text_dst] = block_pos[:, :, n_prefix:][:, text_src]

    attn_mask = build_packed_prefix_lm_mask(prefix_dst, text_dst)
    return combined, pos, attn_mask, k, m


def packed_caption_loss(model, prefix_emb, prefix_valid, block_pos, text, text_valid):
    """Fused autoregressive caption CE over the packed ``[valid prefix ; valid text ; PAD]`` layout.

    Shared by GenLIP and GenLAP (``model`` supplies ``in_proj``/``text_embed``/``rotary``/``trunk``/``out_proj``/
    ``lm_head``). ``prefix_emb`` is the embedded image/audio tokens; ``block_pos`` is the standard block MRoPE
    position ids ``(3, B, Np+Nt)``. The trunk runs on the compacted sequence (length ``max_i(k_i+m_i)``), and
    the loss gathers each row's text-predicting window ``[k_i-1, k_i+m_i-1)`` -- so the first caption token is
    predicted from the last *valid* prefix token (not a padding slot, as the fixed-block layout would).
    """
    text_emb = model.in_proj(model.text_embed(text))
    combined, pos, attn_mask, k, m = pack_prefix_sequence(
        prefix_emb, prefix_valid, block_pos, text_emb, text_valid,
    )
    cos, sin = model.rotary(combined, pos)
    hidden = model.out_proj(model.trunk(combined, attn_mask, cos, sin))  # (B, T, text_embed_dim)

    cols = torch.arange(hidden.shape[1], device=hidden.device)
    pred_dst = (cols[None, :] >= (k - 1)[:, None]) & (cols[None, :] < (k + m - 1)[:, None])  # (B, T)
    text_src = torch.arange(text.shape[1], device=text.device)[None, :] < m[:, None]         # (B, Nt)
    pred = hidden[pred_dst]   # (sum(m), D) row-major
    target = text[text_src]   # (sum(m),) valid caption tokens, row-major aligned (all valid -> no ignore)
    return fused_linear_cross_entropy(
        pred, model.lm_head.weight, target, bias=model.lm_head.bias, ignore_index=-100,
    )


def build_image_attn_mask(patch_valid: torch.Tensor) -> torch.Tensor:
    """Full bidirectional mask over valid image patches (vision-encoder-only mode), ``(B, 1, Ni, Ni)``."""
    pv = patch_valid.bool()
    b, ni = pv.shape
    allowed = (pv[:, :, None] & pv[:, None, :]).clone()
    idx = torch.arange(ni, device=pv.device)
    allowed[:, idx, idx] = True
    return allowed.unsqueeze(1)


def build_image_position_ids(patch_coord: torch.Tensor, patch_valid: torch.Tensor) -> torch.Tensor:
    """3-axis MRoPE position ids for image patches: ``t=0, h=coord_y, w=coord_x``. Shape ``(3, B, Ni)``."""
    b, ni, _ = patch_coord.shape
    pos = patch_coord.new_zeros((3, b, ni), dtype=torch.long)
    pos[1] = patch_coord[..., 0].long()  # height (y)
    pos[2] = patch_coord[..., 1].long()  # width (x)
    return pos


def build_mrope_position_ids(
        patch_coord: torch.Tensor,
        patch_valid: torch.Tensor,
        text_valid: torch.Tensor,
) -> torch.Tensor:
    """3-axis MRoPE position ids for the concatenated ``[image ; text]`` sequence, shape ``(3, B, S)``.

    Image patches use ``t=0`` and their ``(h, w)`` grid coordinates. Text tokens use a single running 1-D index
    (broadcast to all three axes) that starts just after the image's spatial extent, per Qwen2-VL MRoPE.
    """
    b, ni, _ = patch_coord.shape
    lt = text_valid.shape[1]
    device = patch_coord.device

    pos = torch.zeros(3, b, ni + lt, dtype=torch.long, device=device)
    h = patch_coord[..., 0].long()
    w = patch_coord[..., 1].long()
    pos[1, :, :ni] = h
    pos[2, :, :ni] = w

    # text starts after max spatial position over valid patches: max(H, W)
    pv = patch_valid.bool()
    h_valid = torch.where(pv, h, torch.zeros_like(h))
    w_valid = torch.where(pv, w, torch.zeros_like(w))
    text_start = torch.maximum(h_valid.amax(dim=1), w_valid.amax(dim=1)) + 1  # (B,)
    text_pos = text_start[:, None] + torch.arange(lt, device=device)[None, :]  # (B, Lt)
    pos[:, :, ni:] = text_pos[None].expand(3, b, lt)
    return pos


# ---------------------------------------------------------------------------------------------------------------------
# Vision-encoder-only adapter + top-level model
# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def init_genlm_weights(model: nn.Module, std: float = 0.02) -> None:
    """Initialize GenLIP/GenLAP trunk weights (modality-agnostic; shared by the image and audio models)."""
    def _init(module: nn.Module):
        if isinstance(module, GenLipAttention):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            for lin in (module.q_proj, module.k_proj, module.v_proj, module.out_proj):
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
        elif isinstance(module, (GenLipSwiGLUFFN, GenLipMLP)):
            for name, lin in module.named_children():
                if isinstance(lin, nn.Linear):
                    nn.init.xavier_uniform_(lin.weight)
                    if lin.bias is not None:
                        nn.init.normal_(lin.bias, std=1e-6)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # apply generic init first, then specialize attention/ffn (so nn.Linear default doesn't clobber them)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            _init(module)
    for module in model.modules():
        if isinstance(module, (GenLipAttention, GenLipSwiGLUFFN, GenLipMLP)):
            _init(module)


class NaFlexGenLipVisualAdapter(nn.Module):
    """Vision-encoder face of the model: runs image patches through the shared trunk and pools features.

    Holds references (not copies) to the shared patch-embed / trunk / rotary so the LM and the vision encoder
    use the same weights. Exposes the attributes open_clip's NaFlex pipeline and factory expect on ``visual``.
    """

    def __init__(
            self,
            patch_embed: GenLipPatchEmbed,
            trunk: GenLipTrunk,
            rotary: GenLipRotaryEmbedding,
            vision_cfg: NaFlexGenLipVisionCfg,
            width: int,
            embed_dim: int,
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.trunk = trunk
        self.rotary = rotary
        self.pool_type = vision_cfg.pool_type
        self.patch_size = (vision_cfg.patch_size, vision_cfg.patch_size)
        self.image_size = (vision_cfg.image_size, vision_cfg.image_size)
        self.image_seq_len = (vision_cfg.image_size // vision_cfg.patch_size) ** 2
        self.preprocess_cfg: Dict = {}
        # Optional projector to a downstream feature dim (Identity when embed_dim == width).
        self.proj = nn.Linear(width, embed_dim) if embed_dim != width else nn.Identity()

    def get_patch_size(self) -> Tuple[int, int]:
        return self.patch_size

    def forward(self, image: Dict[str, torch.Tensor]) -> torch.Tensor:
        patches = image['patches']
        patch_coord = image['patch_coord']
        patch_valid = image['patch_valid']

        x = self.patch_embed(patches)
        attn_mask = build_image_attn_mask(patch_valid)
        pos = build_image_position_ids(patch_coord, patch_valid)
        cos, sin = self.rotary(x, pos)
        x = self.trunk(x, attn_mask, cos, sin)  # ln_post applied inside trunk

        pv = patch_valid.to(x.dtype)
        summed = (x * pv.unsqueeze(-1)).sum(dim=1)
        count = pv.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = summed / count
        return self.proj(pooled)


class NaFlexGenLip(nn.Module):
    """GenLIP unified vision-language model with a NaFlex linear patch-embed image side."""

    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: Union[NaFlexGenLipVisionCfg, Dict],
            text_cfg: Union[NaFlexGenLipTextCfg, Dict],
            genlip_cfg: Union[NaFlexGenLipTrunkCfg, Dict],
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = True,
            **kwargs,
    ):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = NaFlexGenLipVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = NaFlexGenLipTextCfg(**text_cfg)
        if isinstance(genlip_cfg, dict):
            genlip_cfg = NaFlexGenLipTrunkCfg(**genlip_cfg)

        self.vision_cfg = vision_cfg
        self.text_cfg = text_cfg
        self.trunk_cfg = genlip_cfg
        self.output_dict = output_dict
        self.embed_dim = embed_dim
        self.pack_prefix = genlip_cfg.pack_prefix

        width = genlip_cfg.width
        text_embed_dim = genlip_cfg.text_embed_dim
        self.pad_id = text_cfg.pad_id
        self.context_length = text_cfg.context_length

        # image side
        self.patch_embed = GenLipPatchEmbed(vision_cfg, width)
        # text side
        self.text_embed = nn.Embedding(text_cfg.vocab_size, text_embed_dim, padding_idx=text_cfg.pad_id)
        self.in_proj = nn.Linear(text_embed_dim, width) if text_embed_dim != width else nn.Identity()
        self.out_proj = nn.Linear(width, text_embed_dim) if text_embed_dim != width else nn.Identity()
        self.lm_head = nn.Linear(text_embed_dim, text_cfg.vocab_size, bias=False)  # untied

        # shared trunk + rotary
        self.rotary = GenLipRotaryEmbedding(genlip_cfg)
        self.trunk = GenLipTrunk(genlip_cfg)

        self.visual = NaFlexGenLipVisualAdapter(
            self.patch_embed, self.trunk, self.rotary, vision_cfg, width, embed_dim,
        )

        self.init_weights()
        if cast_dtype is not None:
            self.to(dtype=cast_dtype)

    @torch.no_grad()
    def init_weights(self, std: float = 0.02):
        """Initialize weights following the GenLIP reference scheme."""
        init_genlm_weights(self, std)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        if impl == 'composable' and enable:
            from torch.distributed._composable import checkpoint as composable_checkpoint
            for block in self.trunk.layers:
                composable_checkpoint(block)
        else:
            self.trunk.grad_checkpointing = enable

    def fsdp_shard_modules(self) -> List[Tuple[str, nn.Module]]:
        """``(name, module)`` pairs to wrap individually for FSDP / activation checkpointing.

        Matches the contract of ``TrainingTask._get_fsdp_shard_modules`` / ``prepare_fsdp`` (which iterates
        ``for name, mod in shard_modules``).
        """
        return [(f"trunk.layers.{i}", block) for i, block in enumerate(self.trunk.layers)]

    def encode_image(self, image: Dict[str, torch.Tensor], normalize: bool = False) -> torch.Tensor:
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def _encode(
            self,
            image: Dict[str, torch.Tensor],
            text: torch.Tensor,
            text_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Run the unified trunk over ``[image_patches ; caption_tokens]``.

        Returns ``(hidden, image_seq_len)`` where ``hidden`` is the post-``ln_post``/``out_proj`` sequence of
        shape ``(B, S, text_embed_dim)`` and ``image_seq_len`` is the per-batch image patch count ``Ni``.
        """
        patch_valid = image['patch_valid']
        img_emb = self.patch_embed(image['patches'])  # (B, Ni, width)
        txt_emb = self.in_proj(self.text_embed(text))  # (B, Lt, width)
        h = torch.cat([img_emb, txt_emb], dim=1)  # (B, S, width)

        attn_mask = build_prefix_lm_mask(patch_valid, text_valid)
        pos = build_mrope_position_ids(image['patch_coord'], patch_valid, text_valid)
        cos, sin = self.rotary(h, pos)

        h = self.trunk(h, attn_mask, cos, sin)  # ln_post applied inside trunk
        return self.out_proj(h), img_emb.shape[1]

    def forward(
            self,
            image: Dict[str, torch.Tensor],
            text: torch.Tensor,
            text_valid: Optional[torch.Tensor] = None,
            compute_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run the unified trunk over ``[image_patches ; caption_tokens]``.

        Args:
            image: Dict with ``patches`` ``(B, Ni, P*P*C)``, ``patch_coord`` ``(B, Ni, 2)`` and
                ``patch_valid`` ``(B, Ni)``.
            text: Caption token ids of shape ``(B, Lt)`` padded with ``pad_id``.
            text_valid: Optional ``(B, Lt)`` bool mask; derived from ``text != pad_id`` when omitted.
            compute_loss: When True, return a memory-efficient autoregressive caption ``loss`` (fused linear
                cross-entropy over the text-predicting positions only, no full-vocabulary logits are
                materialized). When False, return full ``logits`` ``(B, S, vocab)`` (for inference/generation).

        Returns:
            Dict with ``image_seq_len`` and either ``loss`` (``compute_loss=True``) or ``logits``.
        """
        if text_valid is None:
            text_valid = text != self.pad_id

        if compute_loss and self.pack_prefix:
            # Packed layout: compact [valid image ; valid text ; PAD] per row (no padding between the two).
            return {'loss': packed_caption_loss(
                self,
                self.patch_embed(image['patches']), image['patch_valid'],
                build_mrope_position_ids(image['patch_coord'], image['patch_valid'], text_valid),
                text, text_valid,
            )}

        hidden, ni = self._encode(image, text, text_valid)  # (B, S, D), Ni

        if compute_loss:
            # Position p predicts token p+1. Caption tokens occupy positions [Ni, S); they are predicted by
            # hidden states at [Ni-1, S-1). Restricting the LM head to this window drops all image positions
            # (never predicted) and avoids the dominant memory cost of full-sequence logits.
            pred = hidden[:, ni - 1:-1, :]  # (B, Lt, D)
            target = torch.where(text_valid, text, torch.full_like(text, -100))  # (B, Lt)
            loss = fused_linear_cross_entropy(
                pred.reshape(-1, pred.shape[-1]),
                self.lm_head.weight,
                target.reshape(-1),
                bias=self.lm_head.bias,
                ignore_index=-100,
            )
            # NOTE: return only tensors here. Returning a Python int (e.g. image_seq_len) breaks
            # torch.compile under the DDP graph-splitter (AOTAutograd expects FX nodes with .meta).
            return {'loss': loss}

        logits = self.lm_head(hidden)
        return {'logits': logits, 'image_seq_len': ni}
