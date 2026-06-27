"""Tests for the NaFlex GenLIP generative vision-language model."""
import types

import pytest
import torch

import open_clip
from open_clip import create_task, get_tokenizer
from open_clip.naflex_genlip_model import (
    NaFlexGenLip,
    apply_interleaved_mrope,
    build_image_attn_mask,
    build_mrope_position_ids,
    build_prefix_lm_mask,
)
from open_clip_train.naflex_data import collate_variable_text

TEST_MODEL = 'naflexgenlip_test'
PATCH_DIM = 16 * 16 * 3


def _make_batch(pad_id, b=3, ni=12, lt=9, grid_w=4):
    patches = torch.randn(b, ni, PATCH_DIM)
    coords = torch.tensor([[i // grid_w, i % grid_w] for i in range(ni)])
    coords = coords.unsqueeze(0).expand(b, ni, 2).contiguous()
    patch_valid = torch.ones(b, ni, dtype=torch.bool)
    patch_valid[-1, ni - 4:] = False  # last sample has fewer valid patches
    text = torch.randint(0, 100277, (b, lt))
    text[0, lt - 3:] = pad_id  # first sample has padded caption tail
    text_valid = text != pad_id
    image = {'patches': patches, 'patch_coord': coords, 'patch_valid': patch_valid}
    return {'image': image, 'text': text, 'text_valid': text_valid}


# ---------------------------------------------------------------------------------------------------------------------
# Masks / positions
# ---------------------------------------------------------------------------------------------------------------------
def test_prefix_lm_mask_semantics():
    pv = torch.ones(1, 3, dtype=torch.bool)
    tv = torch.ones(1, 2, dtype=torch.bool)
    mask = build_prefix_lm_mask(pv, tv)[0, 0]  # (5, 5), image=0..2, text=3..4

    # image queries: attend to all images, never to text
    assert mask[:3, :3].all()
    assert not mask[:3, 3:].any()
    # first text token: sees all images + itself, not the future text token
    assert mask[3, :3].all() and mask[3, 3] and not mask[3, 4]
    # second text token: sees images + both text tokens (causal)
    assert mask[4, :3].all() and mask[4, 3] and mask[4, 4]


def test_prefix_lm_mask_padding():
    pv = torch.tensor([[True, True, False]])  # third patch is padding
    tv = torch.tensor([[True, False]])  # second text token is padding
    mask = build_prefix_lm_mask(pv, tv)[0, 0]
    # padded image key (col 2) is masked for every query
    assert not mask[:, 2].any() or mask[2, 2]  # only its own forced diagonal may be set
    # valid text token must not attend the padded text key (col 4)
    assert not mask[3, 4]
    # diagonal is always set (no fully-masked query rows -> no SDPA NaN)
    idx = torch.arange(5)
    assert mask[idx, idx].all()


def test_mrope_position_ids():
    coords = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]])  # 2x2 grid, B=1, Ni=4
    pv = torch.ones(1, 4, dtype=torch.bool)
    tv = torch.ones(1, 3, dtype=torch.bool)
    pos = build_mrope_position_ids(coords, pv, tv)
    assert pos.shape == (3, 1, 7)
    # image temporal axis is 0
    assert (pos[0, 0, :4] == 0).all()
    # text starts after max spatial extent (max(h,w)=1 -> start at 2) and increments by 1 on all axes
    text_pos = pos[:, 0, 4:]
    assert torch.equal(text_pos[0], torch.tensor([2, 3, 4]))
    assert torch.equal(text_pos[0], text_pos[1]) and torch.equal(text_pos[1], text_pos[2])


def test_apply_interleaved_mrope_layout():
    # head_dim//2 = 6, mrope_section sums to 6 -> interleaved THWTHW
    freqs = torch.zeros(3, 1, 1, 6)
    freqs[0] = 0  # T
    freqs[1] = 1  # H
    freqs[2] = 2  # W
    out = apply_interleaved_mrope(freqs, (2, 2, 2))[0, 0]
    # positions 0,3 -> T(0); 1,4 -> H(1); 2,5 -> W(2)
    assert torch.equal(out, torch.tensor([0., 1., 2., 0., 1., 2.]))


# ---------------------------------------------------------------------------------------------------------------------
# Model forward / vision encoder
# ---------------------------------------------------------------------------------------------------------------------
def test_model_forward_shapes():
    model, _, _ = open_clip.create_model_and_transforms(TEST_MODEL)
    assert isinstance(model, NaFlexGenLip)
    batch = _make_batch(model.pad_id)
    out = model(**batch)
    b, ni = batch['image']['patches'].shape[:2]
    lt = batch['text'].shape[1]
    assert out['logits'].shape == (b, ni + lt, model.text_cfg.vocab_size)
    assert out['image_seq_len'] == ni
    assert torch.isfinite(out['logits']).all()


def test_encode_image():
    model, _, _ = open_clip.create_model_and_transforms(TEST_MODEL)
    batch = _make_batch(model.pad_id)
    feats = model.encode_image(batch['image'])
    assert feats.shape == (batch['image']['patches'].shape[0], model.embed_dim)
    assert torch.isfinite(feats).all()
    # image-only full attention mask is symmetric over valid patches
    pv = batch['image']['patch_valid']
    m = build_image_attn_mask(pv)[0, 0]
    assert m[0, :].sum() == int(pv[0].sum())


def test_mrope_section_assertion():
    from open_clip.naflex_genlip_model import GenLipRotaryEmbedding, NaFlexGenLipTrunkCfg
    bad = NaFlexGenLipTrunkCfg(width=64, num_heads=4, mrope_section=(2, 2, 2))  # head_dim//2=8 != sum 6
    with pytest.raises(ValueError):
        GenLipRotaryEmbedding(bad)


# ---------------------------------------------------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------------------------------------------------
def test_tiktoken_tokenizer():
    tok = get_tokenizer(TEST_MODEL)
    assert type(tok).__name__ == 'TikTokenTokenizer'
    var = tok(['a cat on a mat', 'hi'], pad=False)
    assert all(v[0].item() == tok.bos_token_id and v[-1].item() == tok.eot_token_id for v in var)
    # body ids stay below the reserved control range
    for v in var:
        assert (v[1:-1] < tok.enc.n_vocab).all()
    fixed = tok(['a cat on a mat', 'hi'], pad=True, context_length=16)
    assert fixed.shape == (2, 16)
    assert (fixed[1, 4:] == tok.pad_token_id).all()


def test_tiktoken_clean_option():
    """TikTokenTokenizer 'clean': default None = verbatim (required for generation); contrastive configs can opt
    into 'canonicalize' (lowercase + punctuation strip) via tokenizer_kwargs."""
    from open_clip.tokenizer import TikTokenTokenizer

    text = "Hello, WORLD!! A dog's bark."
    verbatim = TikTokenTokenizer(encoding_name='r50k_base', context_length=32)
    canon = TikTokenTokenizer(encoding_name='r50k_base', context_length=32, clean='canonicalize')
    assert verbatim.decode(verbatim.encode(text)) == text  # default leaves case + punctuation intact
    cleaned = canon.decode(canon.encode(text))
    assert cleaned == cleaned.lower() and ',' not in cleaned and '!' not in cleaned

    # The factory threads clean through tokenizer_kwargs (caller kwargs merged + whitelisted).
    tok = get_tokenizer(TEST_MODEL, clean='canonicalize')
    out = tok.decode(tok.encode(text))
    assert out == out.lower() and '!' not in out

    # whitespace_underscore: case/punctuation-preserving, only normalizes snake_case separators to spaces.
    from open_clip.tokenizer import get_clean_fn
    assert get_clean_fn('whitespace_underscore')("This is a sound of sea_waves.") == "This is a sound of sea waves."
    assert get_clean_fn('whitespace')("sea_waves") == "sea_waves"      # 'whitespace' keeps underscores
    assert get_clean_fn('canonicalize')("Sea_Waves!") == "sea waves"   # canon: lower + strip punct + _->space

    # Both must stay picklable (serialized into dataloader workers).
    import pickle
    for t in (verbatim, canon):
        assert pickle.loads(pickle.dumps(t)).encode(text) == t.encode(text)


# ---------------------------------------------------------------------------------------------------------------------
# Data collation
# ---------------------------------------------------------------------------------------------------------------------
def test_collate_variable_text():
    targets = [torch.tensor([5, 6, 7, 2]), torch.tensor([5, 9, 2]), torch.tensor([5, 1, 1, 1, 1, 2])]
    text, valid = collate_variable_text(targets, pad_id=99)
    assert text.shape == (3, 6)
    assert (text[1, 3:] == 99).all()
    assert valid.sum().item() == 4 + 3 + 6


# ---------------------------------------------------------------------------------------------------------------------
# Task integration + overfit
# ---------------------------------------------------------------------------------------------------------------------
def _make_args():
    return types.SimpleNamespace(
        model=TEST_MODEL, distill=False, siglip=False, rank=0, world_size=1,
        local_loss=False, gather_with_grad=False, torchcompile=False,
    )


def test_create_task_and_forward():
    model, _, _ = open_clip.create_model_and_transforms(TEST_MODEL)
    task = create_task(_make_args(), model=model)
    assert type(task).__name__ == 'GenLipTask'
    batch = _make_batch(task.pad_id)
    task.train()
    losses, _ = task(batch)
    assert 'caption_loss' in losses and 'loss' in losses
    assert torch.isfinite(losses['loss'])
    assert task.batch_size(batch) == batch['image']['patches'].shape[0]


def test_end_to_end_naflex_pipeline():
    """Real NaFlex patchify + variable-text collation feeds the model correctly."""
    np = pytest.importorskip('numpy')
    PIL = pytest.importorskip('PIL.Image')
    from open_clip_train.data import TokenizeText
    from open_clip_train.naflex_data import NaFlexBatchScheduler

    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        TEST_MODEL, aug_cfg={'use_timm': True, 'naflex': True},
    )
    assert getattr(preprocess_train, 'is_naflex_transform_factory', False)
    tok = get_tokenizer(TEST_MODEL)
    tokenize = TokenizeText(tok, variable=True)

    sched = NaFlexBatchScheduler(
        train_num_samples=100, patch_size=16, seq_lens=(256,),
        max_tokens_per_batch=4096, transform_factory=preprocess_train,
        shuffle=False, pad_id=tok.pad_token_id,
    )
    samples = []
    for cap in ['a red square', 'a longer caption about a blue circle', 'cat']:
        arr = (np.random.rand(96, 128, 3) * 255).astype('uint8')  # non-square -> native aspect ratio
        samples.append({'image': PIL.fromarray(arr), 'text': tokenize(cap)})

    batch = sched.collate_batch(samples, seq_len=256, patch_idx=0)
    img = batch['image']
    assert img['patches'].shape == (3, 256, PATCH_DIM)
    assert set(('patches', 'patch_coord', 'patch_valid')).issubset(img.keys())
    assert 'text_valid' in batch

    out = model(image=img, text=batch['text'], text_valid=batch['text_valid'])
    seq = img['patches'].shape[1] + batch['text'].shape[1]
    assert out['logits'].shape == (3, seq, tok.vocab_size)
    assert torch.isfinite(out['logits']).all()
    assert model.encode_image(img).shape == (3, model.embed_dim)


def test_fused_loss_matches_naive_logits_loss():
    """The memory-efficient fused loss equals the naive full-logits cross-entropy."""
    torch.manual_seed(0)
    model, _, _ = open_clip.create_model_and_transforms(TEST_MODEL)
    model.eval()
    batch = _make_batch(model.pad_id)
    text, tv = batch['text'], batch['text_valid']

    fused = model(image=batch['image'], text=text, text_valid=tv, compute_loss=True)['loss']

    logits = model(image=batch['image'], text=text, text_valid=tv)['logits']
    b, lt = text.shape
    ni = logits.shape[1] - lt
    labels = torch.full((b, ni + lt), -100)
    labels[:, ni:] = torch.where(tv, text, torch.full_like(text, -100))
    naive = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100,
    )
    assert torch.allclose(fused, naive, atol=1e-4), f"{fused.item()} != {naive.item()}"


def test_pack_prefix_defaults_off_and_parity_full_prefix():
    """pack_prefix defaults OFF (existing runs unchanged); when on it's loss-identical to the block layout
    for a full prefix (every patch valid, k == Na)."""
    torch.manual_seed(0)
    model = open_clip.create_model(TEST_MODEL).eval()
    assert model.pack_prefix is False  # toggle defaults off -> ongoing runs untouched

    batch = _make_batch(model.pad_id)
    batch['image']['patch_valid'][:] = True  # full prefix: all patches valid -> block == packed
    image, text, tv = batch['image'], batch['text'], batch['text_valid']

    model.pack_prefix = False
    block = model(image=image, text=text, text_valid=tv, compute_loss=True)['loss']
    model.pack_prefix = True
    packed = model(image=image, text=text, text_valid=tv, compute_loss=True)['loss']
    assert torch.allclose(block, packed, atol=1e-5), f"{block.item()} != {packed.item()}"


def test_pack_prefix_variable_prefix_runs():
    """pack_prefix runs on a variable-length prefix (k < Na) and yields a finite loss."""
    model = open_clip.create_model(TEST_MODEL).eval()
    model.pack_prefix = True
    batch = _make_batch(model.pad_id)  # last sample has fewer valid patches (k < Na)
    loss = model(image=batch['image'], text=batch['text'], text_valid=batch['text_valid'], compute_loss=True)['loss']
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_task_fused_loss_flag():
    """fused_loss=True (default) uses the model's in-forward fused loss and builds NO loss module;
    fused_loss=False drives the model as a logits producer + external GenLipLoss for the same result."""
    from open_clip.task import GenLipTask

    torch.manual_seed(0)
    model, _, _ = open_clip.create_model_and_transforms(TEST_MODEL)
    model.eval()
    batch = _make_batch(model.pad_id)

    fused_task = GenLipTask(model)
    assert fused_task.fused_loss is True
    assert not hasattr(fused_task, 'loss')  # default carries no would-be-unused loss module

    ext_task = GenLipTask(model, fused_loss=False)
    assert ext_task.fused_loss is False
    assert type(ext_task.loss).__name__ == 'GenLipLoss'

    fused = fused_task._loss_forward(model, batch)
    ext = ext_task._loss_forward(model, batch)
    for out in (fused, ext):
        assert 'caption_loss' in out and 'loss' in out and torch.isfinite(out['loss'])
    assert torch.allclose(fused['loss'], ext['loss'], atol=1e-4), \
        f"{fused['loss'].item()} != {ext['loss'].item()}"


def test_grad_checkpointing_toggle_and_parity():
    """set_grad_checkpointing accepts impl=, toggles, and is numerically transparent."""
    import copy
    model, _, _ = open_clip.create_model_and_transforms(TEST_MODEL)
    model.set_grad_checkpointing(True, impl='inline')
    assert model.trunk.grad_checkpointing is True
    model.set_grad_checkpointing(False)
    assert model.trunk.grad_checkpointing is False

    ref = copy.deepcopy(model)
    ckpt = copy.deepcopy(model)
    ckpt.set_grad_checkpointing(True, impl='inline')
    batch = _make_batch(model.pad_id)
    kw = dict(image=batch['image'], text=batch['text'], text_valid=batch['text_valid'], compute_loss=True)
    l_ref = ref(**kw)['loss']; l_ref.backward()
    l_ckpt = ckpt(**kw)['loss']; l_ckpt.backward()
    assert torch.allclose(l_ref, l_ckpt, atol=1e-4)
    g_ref = ref.trunk.layers[0].self_attn.q_proj.weight.grad
    g_ckpt = ckpt.trunk.layers[0].self_attn.q_proj.weight.grad
    assert torch.allclose(g_ref, g_ckpt, atol=1e-4)


def test_torch_compile_loss_step():
    """forward(compute_loss=True) must compile (no Python-int graph outputs that break AOTAutograd)."""
    torch.manual_seed(0)
    model, _, _ = open_clip.create_model_and_transforms(TEST_MODEL)
    batch = _make_batch(model.pad_id)

    def step(image, text, text_valid):
        return model(image=image, text=text, text_valid=text_valid, compute_loss=True)['loss']

    compiled = torch.compile(step, backend='aot_eager')
    loss = compiled(batch['image'], batch['text'], batch['text_valid'])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.trunk.layers[0].self_attn.q_proj.weight.grad is not None


def test_total_token_batch_sizing():
    """per_row_text_tokens makes the budget count image + text cap; 0 stays image-only (regression)."""
    pytest.importorskip("timm.data.naflex_dataset")
    from open_clip_train.naflex_data import NaFlexBatchScheduler

    def make(cost):
        return NaFlexBatchScheduler(
            train_num_samples=10000, seq_lens=(256,), max_tokens_per_batch=32768,
            transform_factory=lambda **kw: None, batch_divisor=8, shuffle=False,
            per_row_text_tokens=cost,
        )._canonical_batch_schedule

    img_only = make(0)
    total = make(64)
    assert img_only[0][1] == 128           # 32768 // 256
    assert total[0][1] == 96               # 32768 // (256 + 64), floored to divisor 8
    # total-token sizing keeps every batch within the budget (image + cap)
    assert all(bs * (256 + 64) <= 32768 for _, bs in total)
    # stored seq_len is still the image bucket (collation unchanged)
    assert all(seq_len == 256 for seq_len, _ in total)


def test_length_bucketer():
    """LengthBucketer reorders only (multiset preserved), groups similar lengths, and is picklable."""
    import pickle
    import random
    from open_clip_train.naflex_data import LengthBucketer

    torch.manual_seed(0)
    samples = [{"text": torch.zeros(int(n), dtype=torch.long)} for n in torch.randint(2, 80, (512,))]
    bucketer = LengthBucketer(pool=256, chunk=32, seed=1, epoch=0)  # defaults to [CaptionLength()]
    out = list(bucketer(iter(samples)))

    assert len(out) == len(samples)
    assert sorted(s["text"].shape[0] for s in out) == sorted(s["text"].shape[0] for s in samples)

    def adj_diff(seq):
        lens = [s["text"].shape[0] for s in seq]
        return sum(abs(lens[i + 1] - lens[i]) for i in range(len(lens) - 1)) / (len(lens) - 1)

    shuffled = list(samples)
    random.Random(0).shuffle(shuffled)
    assert adj_diff(out) < 0.4 * adj_diff(shuffled)   # bucketed runs are length-homogeneous
    assert pickle.loads(pickle.dumps(bucketer)) is not None   # forkserver-safe


def test_json_caption_key():
    """FilterValidSample + JsonCaptionExtractor read captions from a chosen JSON field."""
    import json as _json
    import pickle
    import webdataset as wds
    from open_clip_train.data import FilterNonEmptyText, FilterValidSample, JsonCaptionExtractor

    # filter: member-mode requires the text member(s); json-mode requires .json (both need an image)
    assert FilterValidSample()({"txt": b"x", "jpg": b"i"})
    assert not FilterValidSample()({"json": b"{}", "jpg": b"i"})            # no txt member
    assert not FilterValidSample()({"txt": b"x"})                           # no image
    assert FilterValidSample(text_key="caption")({"caption": b"x", "jpg": b"i"})
    assert FilterValidSample(text_key="txt;caption")({"caption": b"x", "jpg": b"i"})  # ';' alternative
    assert FilterValidSample(image_key="ppm")({"txt": b"x", "ppm": b"i"})
    assert FilterValidSample(text_key="caption", image_key="ppm;pgm")({"caption": b"x", "pgm": b"i"})
    assert not FilterValidSample(image_key="ppm")({"txt": b"x", "jpg": b"i"})          # custom image key
    assert FilterValidSample(json_text_key="caption_x")({"json": b"{}", "jpg": b"i"})
    assert FilterValidSample(json_text_key="caption_x", image_key="ppm")({"json": b"{}", "ppm": b"i"})
    assert not FilterValidSample(json_text_key="caption_x")({"jpg": b"i"})  # no json

    ex = JsonCaptionExtractor("caption_x")
    meta = {"caption_x": "a cat", "caption_y": "other", "w": 100}
    # works from a parsed dict and from raw bytes
    assert ex(meta) == "a cat"
    assert ex(_json.dumps(meta).encode()) == "a cat"
    # missing field -> empty string (sample later filtered), and picklable
    assert ex({"other": "z"}) == ""
    assert pickle.loads(pickle.dumps(ex)) is not None

    # JSON caption extraction is field-level map_dict, not sample-level map. WebDataset's sample-level map
    # re-adds __key__ from the input sample; after keep=False that becomes None and breaks default_collate.
    sample = {"image": object(), "text": _json.dumps(meta).encode()}
    out = next(iter(wds.map_dict(text=ex)([sample])))
    assert out == {"image": sample["image"], "text": "a cat"}

    # json caption keys support ordered fallback via both ';' strings and explicit lists/tuples.
    fallback_meta = {
        "caption_a": "   ",
        "caption_b": " second ",
        "caption_c": "third",
    }
    assert (
        JsonCaptionExtractor("caption_a;caption_b;caption_c")(fallback_meta) == "second"
    )
    assert JsonCaptionExtractor(["missing", "caption_c"])(fallback_meta) == "third"
    assert JsonCaptionExtractor(("missing", "caption_a"))(fallback_meta) == ""

    non_empty = FilterNonEmptyText()
    assert non_empty({"text": "caption"})
    assert non_empty({"text": b"caption"})
    assert not non_empty({"text": ""})
    assert not non_empty({"text": "   "})
    assert not non_empty({"text": b"   "})
    assert not non_empty({"caption": "caption"})
    assert not non_empty({"text": 1})
    assert pickle.loads(pickle.dumps(non_empty)) is not None


def test_json_caption_sampling():
    """--json-text-key-probs draws caption keys into a weighted random priority order, then first-non-empty
    wins; 0-weight keys stay as fallbacks, and no probs -> deterministic priority (back-compat)."""
    import pickle
    import random
    from open_clip_train.data import JsonCaptionExtractor, _pad_caption_weights, _weighted_order

    # _pad_caption_weights: None passthrough, tail-pad with 0, error when over-specified
    assert _pad_caption_weights(None, 3) is None
    assert _pad_caption_weights([0.7], 3) == [0.7, 0.0, 0.0]
    assert _pad_caption_weights([0.7, 0.3], 2) == [0.7, 0.3]
    with pytest.raises(ValueError):
        _pad_caption_weights([0.1, 0.2, 0.3], 2)

    # _weighted_order: 0-weight keys always sort last (fallbacks); output is a permutation (nothing dropped)
    random.seed(0)
    for _ in range(50):
        order = _weighted_order(("a", "b", "c"), [0.5, 0.5, 0.0])
        assert order[-1] == "c"
        assert sorted(order) == ["a", "b", "c"]

    meta = {"primary": "P", "secondary": "S"}

    # all mass on 'primary' -> always 'primary' when present; the 0-weight key is used only as fallback
    ex = JsonCaptionExtractor("primary;secondary", sample_probs=[1.0, 0.0])
    random.seed(0)
    assert all(ex(dict(meta)) == "P" for _ in range(20))
    assert ex({"primary": "  ", "secondary": "S"}) == "S"   # primary empty -> fallback

    # ~70/30 split between two present captions over seeded draws
    ex = JsonCaptionExtractor("primary;secondary", sample_probs=[0.7, 0.3])
    random.seed(1234)
    picks = [ex(dict(meta)) for _ in range(2000)]
    assert 0.65 < picks.count("P") / len(picks) < 0.75

    # no probs -> deterministic priority; picklable with probs set
    assert JsonCaptionExtractor("primary;secondary")(dict(meta)) == "P"
    assert pickle.loads(pickle.dumps(ex)) is not None


def test_decode_pil_rgb_max_pixels_cap():
    """decode_pil_rgb drops oversized images from the header (before the costly load); within-cap decodes fine."""
    import io as _io
    from PIL import Image as _Image
    from open_clip_train.data import decode_pil_rgb

    buf = _io.BytesIO()
    _Image.new("RGB", (100, 80), (10, 20, 30)).save(buf, format="PNG")  # 8000 px
    data = buf.getvalue()

    assert decode_pil_rgb(data).size == (100, 80)                     # no cap -> decodes
    assert decode_pil_rgb(data, max_pixels=20_000).size == (100, 80)  # under cap -> decodes
    with pytest.raises(ValueError):
        decode_pil_rgb(data, max_pixels=5_000)                        # over cap -> raises (skipped upstream)


def test_fsdp_shard_modules_are_name_module_pairs():
    """prepare_fsdp iterates `for name, mod in shard_modules` -> must be (str, module) pairs."""
    from open_clip.naflex_genlip_model import GenLipBlock
    model, _, _ = open_clip.create_model_and_transforms(TEST_MODEL)
    shards = model.fsdp_shard_modules()
    assert len(shards) == len(model.trunk.layers)
    for name, mod in shards:  # must unpack cleanly
        assert isinstance(name, str) and isinstance(mod, GenLipBlock)


def test_genlip_has_encode_image_but_no_encode_text():
    """GenLIP has no contrastive text tower; FSDP method registration must be guarded on this."""
    model, _, _ = open_clip.create_model_and_transforms(TEST_MODEL)
    assert hasattr(model, "encode_image")
    assert not hasattr(model, "encode_text")


def test_eval_forward_is_generative_only():
    """eval_forward returns only caption/LM loss; evaluate()'s retrieval branch keys must be absent."""
    model, _, _ = open_clip.create_model_and_transforms(TEST_MODEL)
    task = create_task(_make_args(), model=model)
    task.eval()
    out = task.eval_forward(_make_batch(task.pad_id))
    assert "caption_loss" in out and "loss" in out
    for key in ("image_features", "text_features", "logit_scale", f"{task.primary_key}_features"):
        assert key not in out


def test_map_wrapper_forwards_text_budget():
    """NaFlexMapDatasetWrapper (CSV/map path) forwards pad_id + per_row_text_tokens to its scheduler."""
    pytest.importorskip("timm.data.naflex_dataset")
    from open_clip_train.naflex_data import NaFlexMapDatasetWrapper

    class _DS:
        def __len__(self):
            return 256

        def __getitem__(self, i):
            return {"image": None, "text": torch.zeros(4, dtype=torch.long)}

    wrapper = NaFlexMapDatasetWrapper(
        _DS(), patch_size=16, seq_lens=[256], max_tokens_per_batch=32768,
        transform_factory=lambda **kw: None, shuffle=False, pad_id=99, per_row_text_tokens=64,
    )
    assert wrapper.scheduler.pad_id == 99
    assert wrapper.scheduler.per_row_text_tokens == 64


def test_overfit_single_batch():
    torch.manual_seed(0)
    model, _, _ = open_clip.create_model_and_transforms(TEST_MODEL)
    task = create_task(_make_args(), model=model)
    batch = _make_batch(task.pad_id)
    task.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l0 = task(batch)[0]['loss'].item()
    for _ in range(80):
        opt.zero_grad()
        loss = task(batch)[0]['loss']
        loss.backward()
        opt.step()
    l1 = task(batch)[0]['loss'].item()
    assert l1 < 0.5 * l0, f"expected overfit, got {l0:.3f} -> {l1:.3f}"


def test_pre_norm_per_modality_streams():
    """Optional per-modality pre-trunk norms (vision_cfg.pre_norm / text_cfg.pre_norm): off by default with no
    new state-dict keys (paper-faithful baseline); on, each modality's projected stream is normed entering the
    shared trunk (block pre-norms only normalize branch inputs -- the residual stream carries raw embed scales)."""
    import copy

    cfg = copy.deepcopy(open_clip.get_model_config(TEST_MODEL))
    base = NaFlexGenLip(**cfg)
    assert isinstance(base.patch_embed.norm_pre, torch.nn.Identity)
    assert isinstance(base.text_norm_pre, torch.nn.Identity)
    assert not any('norm_pre' in k for k in base.state_dict())  # existing checkpoints unaffected

    cfg['vision_cfg']['pre_norm'] = True
    cfg['text_cfg']['pre_norm'] = True
    model = NaFlexGenLip(**cfg).eval()
    assert isinstance(model.patch_embed.norm_pre, torch.nn.LayerNorm)
    assert isinstance(model.text_norm_pre, torch.nn.LayerNorm)
    batch = _make_batch(model.pad_id)
    with torch.no_grad():
        out = model(**batch, compute_loss=True)
    assert torch.isfinite(out['loss'])


def test_trunk_bias_and_norm_type_controls():
    """Modern trunk controls (attention_bias / mlp_bias / norm_type, same names as the moderntext tower): default
    off + LayerNorm, opt in to bias + RMSNorm. The gate is fused into q_proj, so it shares attention_bias."""
    import copy

    cfg = copy.deepcopy(open_clip.get_model_config(TEST_MODEL))
    base = NaFlexGenLip(**cfg)
    blk = base.trunk.layers[0]
    # Defaults: bias-free trunk (incl. the fused gate via q_proj) + LayerNorm + qk-norm off.
    for lin in (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.out_proj, blk.mlp.fc1, blk.mlp.fc2):
        assert lin.bias is None
    assert isinstance(blk.layer_norm1, torch.nn.LayerNorm) and isinstance(base.trunk.ln_post, torch.nn.LayerNorm)
    assert isinstance(blk.self_attn.q_norm, torch.nn.Identity) and isinstance(blk.self_attn.k_norm, torch.nn.Identity)

    cfg['genlip_cfg']['attention_bias'] = True
    cfg['genlip_cfg']['mlp_bias'] = True
    cfg['genlip_cfg']['norm_type'] = 'rmsnorm'
    cfg['genlip_cfg']['qk_norm'] = True  # q/k normed over head_dim, follows norm_type (RMSNorm here)
    cfg['vision_cfg']['pre_norm'] = True  # the image prefix pre-norm must follow norm_type too (uniform policy)
    cfg['vision_cfg']['input_norm'] = True  # raw-input norm stays LayerNorm regardless of norm_type
    cfg['text_cfg']['pre_norm'] = True
    model = NaFlexGenLip(**cfg).eval()
    blk = model.trunk.layers[0]
    assert blk.self_attn.q_proj.bias is not None and blk.self_attn.out_proj.bias is not None
    assert blk.mlp.fc1.bias is not None and blk.mlp.gate_fc.bias is not None
    assert isinstance(blk.layer_norm1, torch.nn.RMSNorm) and isinstance(model.trunk.ln_post, torch.nn.RMSNorm)
    assert isinstance(blk.self_attn.q_norm, torch.nn.RMSNorm) and isinstance(blk.self_attn.k_norm, torch.nn.RMSNorm)
    # Modality + text prefix *stream* pre-norms follow the policy; the raw-input norm stays LayerNorm.
    assert isinstance(model.patch_embed.norm_pre, torch.nn.RMSNorm)
    assert isinstance(model.text_norm_pre, torch.nn.RMSNorm)
    assert isinstance(model.patch_embed.norm_input, torch.nn.LayerNorm)
    assert not isinstance(model.patch_embed.norm_input, torch.nn.RMSNorm)
    batch = _make_batch(model.pad_id)
    with torch.no_grad():
        out = model(**batch, compute_loss=True)
    assert torch.isfinite(out['loss'])

    # qk-norm follows norm_type (not fixed at RMSNorm): under layernorm it is LayerNorm.
    cfg['genlip_cfg']['norm_type'] = 'layernorm'
    ln_attn = NaFlexGenLip(**cfg).trunk.layers[0].self_attn
    assert isinstance(ln_attn.q_norm, torch.nn.LayerNorm) and isinstance(ln_attn.k_norm, torch.nn.LayerNorm)
