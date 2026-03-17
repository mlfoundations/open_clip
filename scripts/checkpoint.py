#!/usr/bin/env python3
"""
Minimal script to verify loading the base SigLIP model + PEFT checkpoint (e.g. epoch_9.pt).
Runs a tiny forward pass (one image, one text) and prints success/failure.
"""

import argparse
import os
import sys

import torch
from PIL import Image as PILImage

# Optional: add open_clip src if running from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPEN_CLIP_SRC = os.path.join(REPO_ROOT, "open_clip", "src")
if os.path.isdir(OPEN_CLIP_SRC) and OPEN_CLIP_SRC not in sys.path:
    sys.path.insert(0, OPEN_CLIP_SRC)

from open_clip import create_model_from_pretrained, get_tokenizer


def load_checkpoint_peft(model, state_dict, lora_r=128, lora_alpha=256, lora_dropout=0.05, target_modules=None):
    """Load PEFT checkpoint: base weights first, then wrap with LoRA and load adapters."""
    if target_modules is None:
        target_modules = ["out_proj", "c_fc", "c_proj"]
    prefix = "base_model.model."
    base_sd = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        if "lora_A" in k or "lora_B" in k or "lora_dropout" in k:
            continue
        key = k[len(prefix):]
        key = key.replace(".base_layer.weight", ".weight").replace(".base_layer.bias", ".bias")
        base_sd[key] = v
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    print(f"  Base: loaded {len(base_sd)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  Base missing (first 3): {missing[:3]}")
    if unexpected:
        print(f"  Base unexpected (first 3): {unexpected[:3]}")

    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        task_type=None,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    adapter_sd = {k: v for k, v in state_dict.items() if "lora_A" in k or "lora_B" in k}
    missing2, unexpected2 = model.load_state_dict(adapter_sd, strict=False)
    print(f"  Adapters: loaded {len(adapter_sd)} keys, missing={len(missing2)}, unexpected={len(unexpected2)}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Check loading SigLIP + PEFT checkpoint.")
    parser.add_argument("--model_name", type=str, default="ViT-SO400M-14-SigLIP2")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(
            REPO_ROOT,
            "/home/varsha.redla/ai-search/open_clip/logs_full/full-no-hard-20260314-213423/checkpoints/epoch_9.pt",
        ),
    )
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return 1

    full_model_name = f"hf-hub:timm/{args.model_name}"
    print("Loading base model and tokenizer...")
    model, preprocess = create_model_from_pretrained(full_model_name, device=device)
    tokenizer = get_tokenizer(full_model_name, device=device)

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # DDP-saved checkpoints have "module." prefix (e.g. module.base_model.model.*); strip it first
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        first_key = next(iter(state_dict.keys()))
    print(f"  First key: {first_key[:60]}...")

    if first_key.startswith("base_model.model."):
        print("  PEFT checkpoint: loading base then adapters...")
        model = load_checkpoint_peft(
            model,
            state_dict,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    else:
        model.load_state_dict(state_dict, strict=True)
        print("  Loaded state_dict (strict).")

    model.eval()

    # Tiny forward pass: one image, one text
    print("Running forward pass (1 image, 1 text)...")
    with torch.no_grad():
        # Dummy image 224x224 (or use a real path if you pass one)
        dummy_img = PILImage.new("RGB", (224, 224), color=(128, 128, 128))
        img_tensor = preprocess(dummy_img).unsqueeze(0).to(device)
        text_tensor = tokenizer(["a person smiling"], context_length=model.context_length).to(device)
        img_emb = model.encode_image(img_tensor, normalize=True)
        text_emb = model.encode_text(text_tensor, normalize=True)
        sim = (img_emb @ text_emb.T).item()
    print(f"  Image embedding shape: {img_emb.shape}")
    print(f"  Text embedding shape: {text_emb.shape}")
    print(f"  Cosine similarity (image–text): {sim:.4f}")

    print("OK: Model loaded and forward pass succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
