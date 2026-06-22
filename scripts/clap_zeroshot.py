""" Audio zero-shot classification for CLAP / NaFlexClap checkpoints.

Loads a trained CLAP (HTSAT) or NaFlexClap (spectrogram-ViT) checkpoint and runs audio classification
zero-shot on a Hugging Face audio dataset (e.g. ESC-50): builds a text classifier from templated class names
and scores ``audio_features @ text_classifier``. Reuses ``open_clip_train.audio_zero_shot`` (now NaFlex-aware),
so the same script works for both tower types -- NaFlexClap gets the NaFlex mel-patch transform, HTSAT the
fixed-clip ``AudioPreprocess``.

Needs ``datasets[audio]`` and the eval set (downloaded to ``HF_HOME``).

Example:
    python scripts/clap_zeroshot.py \
        --model naflexclap_little --checkpoint /path/to/checkpoints/epoch_18.pt \
        --audio-zeroshot-dataset ashraq/esc50 --audio-zeroshot-split train \
        --audio-zeroshot-class-key category --audio-zeroshot-target-key target \
        --batch-size 16 --device cuda --precision amp_bf16
"""
import argparse

import torch

from open_clip import create_model, get_tokenizer
from open_clip_train.audio_zero_shot import audio_zero_shot_eval, build_hf_audio_zero_shot_dataset


def strip_prefix(key: str) -> str:
    for prefix in ("module.", "_orig_mod.", "trainable_module."):
        while key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_weights(model, path: str, use_ema: bool = False) -> None:
    """Load CLAP/NaFlexClap weights from a raw param dict or a (possibly nested) task checkpoint."""
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
    if missing:
        print(f"  e.g. missing: {missing[:5]}")
    if unexpected:
        print(f"  e.g. unexpected: {unexpected[:5]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="open_clip model name (CLAP-* or naflexclap_*).")
    parser.add_argument("--checkpoint", required=True, help="Path to trained weights (raw or task checkpoint).")
    parser.add_argument("--use-ema", action="store_true", help="Prefer EMA weights if present.")
    parser.add_argument("--audio-zeroshot-dataset", required=True, help="HF dataset id, e.g. ashraq/esc50.")
    parser.add_argument("--audio-zeroshot-split", default="train")
    parser.add_argument("--audio-zeroshot-audio-key", default="audio")
    parser.add_argument("--audio-zeroshot-target-key", default="target")
    parser.add_argument("--audio-zeroshot-class-key", default="category")
    parser.add_argument("--audio-zeroshot-workers", type=int, default=0)
    parser.add_argument("--naflex-seq-lens", type=int, nargs="+", default=None,
                        help="NaFlexClap audio-token cap for the eval clips (ignored for HTSAT CLAP). Unset -> the "
                             "model config's audio_seq_len, else a ~10s geometry default.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="amp_bf16")
    args = parser.parse_args()

    # Fields that audio_zero_shot_eval / build_hf_audio_zero_shot_dataset read off args.
    args.rank = 0
    args.world_size = 1
    args.distributed = False
    args.zeroshot_frequency = 1
    args.epochs = 1

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"Building {args.model} ...")
    model = create_model(args.model)
    load_weights(model, args.checkpoint, use_ema=args.use_ema)
    model = model.to(device).eval()
    tokenizer = get_tokenizer(args.model)

    print(f"Loading {args.audio_zeroshot_dataset} (split={args.audio_zeroshot_split}) ...")
    audio_data = build_hf_audio_zero_shot_dataset(args, model)
    print(f"  {len(audio_data.classnames)} classes; scoring ...")

    metrics = audio_zero_shot_eval(model, audio_data, epoch=1, args=args, tokenizer=tokenizer)
    print(f"\n=== {args.model} audio zero-shot on {audio_data.dataset_name} ===")
    for key, value in metrics.items():
        print(f"  {key}: {100 * value:.2f}%")


if __name__ == "__main__":
    main()
