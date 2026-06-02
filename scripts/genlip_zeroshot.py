""" Generative zero-shot image classification for GenLIP (naflexgenlip).

GenLIP has no contrastive image/text embedding, so CLIP-style "cosine of image vs text embeddings" does not
apply. Instead we classify *generatively*: for each class we form a templated caption with the standard CLIP
ImageNet templates (e.g. "a photo of a {classname}.") and score the model's conditional log-likelihood of that
caption given the image, ``log P(caption | image)`` (teacher-forced, length-normalized). The predicted class is
the argmax over classes of the mean per-template score.

This is a research probe of the LM head, NOT how the GenLIP paper evaluates classification (that uses an
attentive probe on frozen features). It is also expensive: with no KV-cache the image prefix is recomputed for
every batch of candidate captions, so cost ~= images * (num_classes * num_templates / score_batch) forwards.
Use --num-images / --templates to keep it tractable; full 50k-val x 80-templates needs a KV-cache (not here).

Example:
    python scripts/genlip_zeroshot.py \
        --model naflexgenlip_b16_224 --checkpoint /path/to/ckpt.pt \
        --imagenet-val /data/x/imagenet/validation \
        --seq-len 256 --templates simple --num-images 2000 --device cuda --precision amp_bf16
"""
import argparse
import time
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import open_clip
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.naflex_config import NaFlexDataConfig
from open_clip.zero_shot_metadata import (
    IMAGENET_CLASSNAMES,
    OPENAI_IMAGENET_TEMPLATES,
    SIMPLE_IMAGENET_TEMPLATES,
)
from open_clip_train.naflex_data import collate_naflex_tuples, create_naflex_eval_transform

SINGLE_TEMPLATE = (lambda c: f"a photo of a {c}.",)
TEMPLATE_SETS = {"single": SINGLE_TEMPLATE, "simple": SIMPLE_IMAGENET_TEMPLATES, "openai": OPENAI_IMAGENET_TEMPLATES}


def strip_prefix(key: str) -> str:
    for prefix in ("module.", "_orig_mod.", "trainable_module."):
        while key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_weights(model, path: str, use_ema: bool = False) -> None:
    """Load GenLIP weights from a raw param dict or a (possibly nested) task checkpoint."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # Descend through task wrappers: {'state_dict': {...}} or {'state_dict_ema': {...}}.
    while isinstance(obj, dict):
        if use_ema and isinstance(obj.get("state_dict_ema"), dict):
            obj = obj["state_dict_ema"]
            continue
        if isinstance(obj.get("state_dict"), dict):
            obj = obj["state_dict"]
            continue
        break
    state_dict = {strip_prefix(k): v for k, v in obj.items() if torch.is_tensor(v)}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {len(state_dict)} tensors from {path} (missing={len(missing)}, unexpected={len(unexpected)}).")
    if missing:
        print(f"  e.g. missing: {missing[:5]}")


def build_caption_chunks(tokenizer, classnames, templates, pad_id, chunk_size, device):
    """Tokenize every (class, template) caption once; return chunked, padded tensors on ``device``.

    Returns a list of ``(text [b, Lmax], text_valid [b, Lmax], class_idx [b])`` and the template count.
    """
    captions, class_idx = [], []
    for ci, name in enumerate(classnames):
        for template in templates:
            ids = tokenizer([template(name)], pad=False)[0]
            captions.append(ids)
            class_idx.append(ci)

    chunks = []
    for start in range(0, len(captions), chunk_size):
        cap = captions[start:start + chunk_size]
        idx = class_idx[start:start + chunk_size]
        max_len = max(c.shape[0] for c in cap)
        text = torch.full((len(cap), max_len), pad_id, dtype=torch.long)
        for i, c in enumerate(cap):
            text[i, :c.shape[0]] = c
        chunks.append((
            text.to(device),
            (text != pad_id).to(device),
            torch.tensor(idx, dtype=torch.long, device=device),
        ))
    return chunks, len(templates)


@torch.no_grad()
def score_image(model, image_row, chunks, num_classes, device, autocast):
    """Return per-class summed length-normalized log P(caption | image) for one image."""
    patches, coord, valid = image_row  # each [Ni, ...]
    ni = patches.shape[0]
    class_logprob = torch.full((num_classes,), 0.0, device=device)
    for text, text_valid, class_idx in chunks:
        b = text.shape[0]
        image = {
            "patches": patches.unsqueeze(0).expand(b, *patches.shape),
            "patch_coord": coord.unsqueeze(0).expand(b, *coord.shape),
            "patch_valid": valid.unsqueeze(0).expand(b, *valid.shape),
        }
        with autocast():
            out = model(image=image, text=text, text_valid=text_valid, compute_loss=False)
        # Caption token text[:, j] (position ni+j) is predicted by logits at position ni-1+j.
        lt = text.shape[1]
        pred = out["logits"][:, ni - 1:ni - 1 + lt, :].float()
        token_lp = F.log_softmax(pred, dim=-1).gather(-1, text.unsqueeze(-1)).squeeze(-1)  # [b, Lt]
        token_lp = token_lp.masked_fill(~text_valid, 0.0)
        seq_lp = token_lp.sum(1) / text_valid.sum(1).clamp(min=1)  # length-normalized
        class_logprob.index_add_(0, class_idx, seq_lp)  # equal templates/class -> argmax == mean
    return class_logprob


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="naflexgenlip_b16", help="open_clip model name (a genlip config).")
    parser.add_argument("--checkpoint", default=None, help="Path to trained weights (raw or task checkpoint).")
    parser.add_argument("--use-ema", action="store_true", help="Prefer EMA weights if present in the checkpoint.")
    parser.add_argument("--imagenet-val", required=True, help="ImageFolder val dir (class subdirs, standard order).")
    parser.add_argument("--seq-len", type=int, default=256, help="NaFlex image patch tokens (eval bucket).")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--templates", choices=tuple(TEMPLATE_SETS), default="simple")
    parser.add_argument("--pmi", action="store_true",
                        help="PMI debias: subtract each class caption's unconditional (no-image) log-likelihood "
                             "from its image-conditioned score, cancelling surface-form / string-prior bias.")
    parser.add_argument("--num-images", type=int, default=2000, help="Random subset of val images to evaluate.")
    parser.add_argument("--image-batch", type=int, default=16, help="Images loaded per dataloader batch.")
    parser.add_argument("--score-batch", type=int, default=1024, help="Candidate captions scored per forward.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "amp_bf16", "amp"), default="amp_bf16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torchvision

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    use_amp = args.precision in ("amp_bf16", "amp")
    amp_dtype = torch.bfloat16 if args.precision != "amp" else torch.float16
    autocast = (lambda: torch.autocast(device.type, dtype=amp_dtype)) if use_amp else nullcontext

    print(f"Building {args.model} ...")
    model, _, preprocess_val = create_model_and_transforms(
        args.model, aug_cfg={"use_timm": True, "naflex": True},
    )
    if args.checkpoint:
        load_weights(model, args.checkpoint, use_ema=args.use_ema)
    else:
        print("WARNING: no --checkpoint given; running with random weights (expect ~chance accuracy).")
    model = model.to(device).eval()

    tokenizer = get_tokenizer(args.model)
    pad_id = getattr(tokenizer, "pad_token_id")

    # NaFlex eval transform (PIL -> {patches, patch_coord, patch_valid}) at a fixed seq-len bucket.
    ndc = NaFlexDataConfig.resolve(
        patch_sizes=[args.patch_size], seq_lens=[args.seq_len],
        eval_patch_size=args.patch_size, eval_seq_len=args.seq_len,
    )
    eval_tf, max_seq_len, _ = create_naflex_eval_transform(preprocess_val, ndc)

    classnames = list(IMAGENET_CLASSNAMES)
    templates = TEMPLATE_SETS[args.templates]
    print(f"Tokenizing {len(classnames)} classes x {len(templates)} templates ...")
    chunks, n_templates = build_caption_chunks(tokenizer, classnames, templates, pad_id, args.score_batch, device)
    n_forward_per_image = len(chunks)
    print(f"  {len(classnames) * n_templates} captions -> {n_forward_per_image} forward(s)/image "
          f"(score-batch {args.score_batch}).")

    # PMI baseline: unconditional log P(caption | no image), computed once over all class captions.
    # A "null image" (zeros patches, patch_valid all False) makes the image contribute nothing to attention,
    # so this isolates the per-class string/surface-form prior to subtract from the conditioned scores.
    uncond = None
    if args.pmi:
        vc = model.vision_cfg
        pdim = vc.patch_size * vc.patch_size * vc.in_chans
        null_row = (
            torch.zeros(max_seq_len, pdim, device=device),
            torch.zeros(max_seq_len, 2, dtype=torch.long, device=device),
            torch.zeros(max_seq_len, dtype=torch.bool, device=device),  # no visible image
        )
        uncond = score_image(model, null_row, chunks, len(classnames), device, autocast)
        print(f"PMI on: unconditional baseline computed (null image), uncond std={uncond.std():.3f}")

    dataset = torchvision.datasets.ImageFolder(args.imagenet_val, transform=eval_tf)
    assert len(dataset.classes) == len(classnames), \
        f"ImageFolder has {len(dataset.classes)} classes but {len(classnames)} classnames."
    loader = DataLoader(
        dataset, batch_size=args.image_batch, shuffle=True, num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=partial(collate_naflex_tuples, max_seq_len=max_seq_len),
    )

    top1 = top5 = n = 0
    t0 = time.time()
    for image_dict, labels in loader:
        patches = image_dict["patches"].to(device, non_blocking=True)
        coord = image_dict["patch_coord"].to(device, non_blocking=True)
        valid = image_dict["patch_valid"].to(device, non_blocking=True)
        labels = labels.to(device)
        for b in range(patches.shape[0]):
            scores = score_image(model, (patches[b], coord[b], valid[b]), chunks, len(classnames), device, autocast)
            if uncond is not None:
                scores = scores - uncond  # PMI: log P(cap|img) - log P(cap|null)
            pred5 = scores.topk(5).indices
            top1 += int(pred5[0] == labels[b])
            top5 += int((pred5 == labels[b]).any())
            n += 1
            if n % 100 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n} imgs | top1 {100 * top1 / n:.2f}% top5 {100 * top5 / n:.2f}% | {rate:.1f} img/s")
            if n >= args.num_images:
                break
        if n >= args.num_images:
            break

    print(f"\n=== {args.model} generative zero-shot ({args.templates} templates, pmi={args.pmi}, n={n}) ===")
    print(f"  top-1: {100 * top1 / max(n, 1):.2f}%   top-5: {100 * top5 / max(n, 1):.2f}%")


if __name__ == "__main__":
    main()
