""" Attentive-probe (AttentionPoolLatent) ImageNet-1k evaluation of a frozen GenLIP image encoder.

GenLIP has no [CLS] token, so we follow the AIM/DINOv2-style frozen-backbone protocol: freeze the trunk,
extract last-layer image patch features (post-ln_post), and train a small attention-pooling head
(``timm.AttentionPoolLatent``: a learnable latent query that cross-attends the patch tokens, padding-masked)
+ a linear classifier. The backbone is frozen, so features are extracted ONCE and cached; only the head trains
-> fast, many epochs. No train-time augmentation (cached features are deterministic) -- a plain frozen probe.

Example:
    python scripts/genlip_attentive_probe.py \
        --model naflexgenlip_b16 --checkpoint /path/epoch_32.pt \
        --imagenet-train /data/f/imagenet/train --imagenet-val /data/f/imagenet/val \
        --seq-len 256 --train-per-class 100 --epochs 20 --lr 1e-3 \
        --device cuda --precision amp_bf16
"""

import argparse
import random
import time
from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from open_clip import create_model_and_transforms
from open_clip.naflex_config import NaFlexDataConfig
from open_clip.naflex_genlip_model import build_image_attn_mask, build_image_position_ids
from open_clip_train.naflex_data import collate_naflex_tuples, create_naflex_eval_transform
from timm.layers import AttentionPoolLatent


def strip_prefix(key: str) -> str:
    for prefix in ("module.", "_orig_mod.", "trainable_module."):
        while key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def load_weights(model, path: str, use_ema: bool = False) -> None:
    obj = torch.load(path, map_location="cpu", weights_only=False)
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


@torch.no_grad()
def extract_patch_features(visual, image, device, autocast):
    """Frozen GenLIP -> last-layer image patch hidden ``[B, Ni, width]`` (post-ln_post) + patch_valid."""
    patches = image["patches"].to(device, non_blocking=True)
    coord = image["patch_coord"].to(device, non_blocking=True)
    valid = image["patch_valid"].to(device, non_blocking=True)
    with autocast():
        x = visual.patch_embed(patches)
        cos, sin = visual.rotary(x, build_image_position_ids(coord, valid))
        x = visual.trunk(x, build_image_attn_mask(valid), cos, sin)  # [B, Ni, width], ln_post inside
    return x, valid


class ProbeHead(nn.Module):
    """AttentionPoolLatent (padding-masked) -> BN(affine=False) -> linear classifier."""

    def __init__(self, dim, num_classes, num_heads=12, q_proj=False, mlp_ratio=0.0, use_bn=True, bn_affine=False):
        super().__init__()
        import inspect

        pool_kwargs = dict(
            in_features=dim,
            embed_dim=dim,
            num_heads=num_heads,
            latent_len=1,
            mlp_ratio=mlp_ratio,
            out_features=0,  # out_features=0 -> pooled dim-vector (no proj/mlp)
        )
        if "q_proj" in inspect.signature(AttentionPoolLatent.__init__).parameters:
            pool_kwargs["q_proj"] = q_proj
        elif not q_proj:
            print(
                "note: installed timm AttentionPoolLatent has no q_proj arg; using the default query "
                "projection (expressivity-equivalent for a learnable latent query)."
            )
        self.pool = AttentionPoolLatent(**pool_kwargs)
        # affine=False = AIM-style pure standardizer (no learnable scale/shift); affine=True adds learnable gamma/beta.
        self.bn = nn.BatchNorm1d(dim, affine=bn_affine) if use_bn else nn.Identity()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, feats, valid):
        # additive mask so the latent query ignores padding patches: 0 for valid keys, -inf for padding
        attn_mask = torch.zeros(feats.shape[0], 1, 1, feats.shape[1], device=feats.device, dtype=feats.dtype)
        attn_mask = attn_mask.masked_fill(~valid[:, None, None, :], float("-inf"))
        pooled = self.pool(feats, attn_mask=attn_mask)  # [B, dim]
        return self.fc(self.bn(pooled))


def build_loader(root, eval_tf, max_seq_len, per_class, batch_size, workers, seed):
    import torchvision

    dataset = torchvision.datasets.ImageFolder(root, transform=eval_tf)
    if per_class:
        by_class = defaultdict(list)
        for idx, (_, cls) in enumerate(dataset.samples):
            by_class[cls].append(idx)
        rng = random.Random(seed)
        keep = []
        for cls, idxs in by_class.items():
            rng.shuffle(idxs)
            keep.extend(idxs[:per_class])
        dataset = Subset(dataset, keep)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=partial(collate_naflex_tuples, max_seq_len=max_seq_len),
    )
    return loader, len(dataset)


@torch.no_grad()
def cache_features(visual, loader, n, seq_len, dim, device, cache_device, autocast, tag):
    feats = torch.empty(n, seq_len, dim, dtype=torch.bfloat16, device=cache_device)
    valid = torch.empty(n, seq_len, dtype=torch.bool, device=cache_device)
    labels = torch.empty(n, dtype=torch.long, device=cache_device)
    i, t0 = 0, time.time()
    for image, y in loader:
        x, v = extract_patch_features(visual, image, device, autocast)
        b = x.shape[0]
        feats[i:i + b] = x.to(cache_device, torch.bfloat16)
        valid[i:i + b] = v.to(cache_device)
        labels[i:i + b] = y.to(cache_device)
        i += b
        if (i // b) % 50 == 0:
            print(f"  [{tag}] cached {i}/{n}  ({i / (time.time() - t0):.0f} img/s)", flush=True)
    return feats[:i], valid[:i], labels[:i]


def accuracy(logits, target, topk=(1, 5)):
    pred = logits.topk(max(topk), 1, True, True).indices.t()
    correct = pred.eq(target.view(1, -1))
    return [correct[:k].reshape(-1).float().sum().item() for k in topk]


def evaluate(head, feats, valid, labels, batch_size, device):
    head.eval()
    top1 = top5 = 0.0
    with torch.no_grad():
        for i in range(0, feats.shape[0], batch_size):
            x = feats[i:i + batch_size].to(device, torch.float32)
            v = valid[i:i + batch_size].to(device)
            logits = head(x, v)
            a1, a5 = accuracy(logits, labels[i:i + batch_size].to(device))
            top1 += a1
            top5 += a5
    n = feats.shape[0]
    return 100 * top1 / n, 100 * top5 / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="naflexgenlip_b16")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--imagenet-train", required=True)
    p.add_argument("--imagenet-val", required=True)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--train-per-class", type=int, default=100, help="Images/class to cache for training (0=all).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pool-num-heads", type=int, default=12)
    p.add_argument("--no-q-proj", dest="q_proj", action="store_false", help="AIM-style: latent used directly as Q.")
    p.add_argument("--mlp-ratio", type=float, default=0.0, help=">0 adds the MAP-head residual MLP.")
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    p.add_argument("--head-batch", type=int, default=512, help="Batch size for head train/eval on cached features.")
    p.add_argument("--extract-batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache-device", default="cuda", help="Where to keep cached features (cuda|cpu).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="amp_bf16")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    cache_device = torch.device(args.cache_device if torch.cuda.is_available() else "cpu")
    use_amp = args.precision.startswith("amp")
    amp_dtype = torch.bfloat16 if "bf16" in args.precision or args.precision == "amp_bfloat16" else torch.float16
    from contextlib import nullcontext

    autocast = (lambda: torch.autocast(device.type, dtype=amp_dtype)) if use_amp else nullcontext

    print(f"Building {args.model} (frozen backbone) ...")
    model, _, preprocess_val = create_model_and_transforms(args.model, aug_cfg={"use_timm": True, "naflex": True})
    load_weights(model, args.checkpoint, use_ema=args.use_ema)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    visual = model.visual
    dim = model.trunk_cfg.width

    ndc = NaFlexDataConfig.resolve(
        patch_sizes=[args.patch_size],
        seq_lens=[args.seq_len],
        eval_patch_size=args.patch_size,
        eval_seq_len=args.seq_len,
    )
    eval_tf, max_seq_len, _ = create_naflex_eval_transform(preprocess_val, ndc)

    print(f"Caching features (dim={dim}, seq_len={max_seq_len}, cache_device={cache_device}) ...")
    tr_loader, n_tr = build_loader(
        args.imagenet_train,
        eval_tf,
        max_seq_len,
        args.train_per_class,
        args.extract_batch,
        args.workers,
        args.seed,
    )
    va_loader, n_va = build_loader(
        args.imagenet_val,
        eval_tf,
        max_seq_len,
        0,
        args.extract_batch,
        args.workers,
        args.seed,
    )
    print(f"  train images: {n_tr} ({args.train_per_class}/class) | val images: {n_va}")
    tr_feats, tr_valid, tr_labels = cache_features(
        visual, tr_loader, n_tr, max_seq_len, dim, device, cache_device, autocast, "train"
    )
    va_feats, va_valid, va_labels = cache_features(
        visual, va_loader, n_va, max_seq_len, dim, device, cache_device, autocast, "val"
    )

    head = ProbeHead(
        dim,
        num_classes=1000,
        num_heads=args.pool_num_heads,
        q_proj=args.q_proj,
        mlp_ratio=args.mlp_ratio,
        use_bn=args.use_bn,
    ).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    print(
        f"Head: AttentionPoolLatent(q_proj={args.q_proj}, mlp_ratio={args.mlp_ratio}) + "
        f"{'BN' if args.use_bn else 'noBN'} + Linear  ({n_head / 1e6:.2f}M params)"
    )

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best1 = 0.0
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(tr_feats.shape[0], device=cache_device)
        t0 = time.time()
        for i in range(0, perm.shape[0], args.head_batch):
            idx = perm[i:i + args.head_batch]
            x = tr_feats[idx].to(device, torch.float32)
            v = tr_valid[idx].to(device)
            y = tr_labels[idx].to(device)
            logits = head(x, v)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        top1, top5 = evaluate(head, va_feats, va_valid, va_labels, args.head_batch, device)
        best1 = max(best1, top1)
        print(
            f"epoch {epoch + 1:2d}/{args.epochs} | loss {loss.item():.3f} | val top1 {top1:.2f}% top5 {top5:.2f}% "
            f"| {time.time() - t0:.1f}s",
            flush=True,
        )

    print(f"\n=== {args.model} attentive probe (epochs={args.epochs}, {args.train_per_class}/class) ===")
    print(f"  best val top-1: {best1:.2f}%")


if __name__ == "__main__":
    main()
