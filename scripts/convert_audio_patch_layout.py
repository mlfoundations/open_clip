#!/usr/bin/env python3
"""Remap a NaFlexClap audio checkpoint from the legacy ``(C, p_t, p_f)`` patch-embed layout to the canonical
``(C, p_f, p_t)`` layout, in place.

The audio patchifier's legacy flatten order ``(C, p_t, p_f)`` is spatial-transposed relative to the declared
``patch_size = (patch_freq, patch_time)``; the canonical order ``(C, p_f, p_t)`` matches it (timm-canonical,
patch-resize / FlexiViT ready). Switching the patchifier therefore requires permuting the patch-embed Linear's
input columns so the model stays numerically unchanged (validated end-to-end: bit-identical CLAP-suite eval;
~1e-7 embedding delta).

For each checkpoint the original is renamed to ``<name>_legacy.<ext>`` and the remapped checkpoint is written
under the original name, so existing configs/paths keep working -- now expecting the canonical patchifier.

    python scripts/convert_audio_patch_layout.py <model_name> <ckpt.pt> [<ckpt2.pt> ...]
"""
import argparse
import os

import torch

import open_clip


def _geom(model_name):
    cfg = open_clip.get_model_config(model_name) or {}
    audio = cfg.get("audio_cfg")
    if not audio:
        raise SystemExit(f"no audio_cfg for model {model_name!r}")
    return int(audio["in_chans"]), int(audio["patch_time"]), int(audio["patch_freq"])


def convert(path, C, pt, pf):
    stem, ext = os.path.splitext(path)
    legacy = f"{stem}_legacy{ext}"
    if os.path.exists(legacy):
        raise SystemExit(f"backup {legacy} already exists; refusing to re-convert {path}")

    ck = torch.load(path, map_location="cpu", weights_only=False)
    wrapper = isinstance(ck, dict) and "state_dict" in ck
    if wrapper and ck.get("audio_patch_layout") == "canonical":
        raise SystemExit(f"{path} already marked canonical")
    sd = ck["state_dict"] if wrapper else ck

    keys = [k for k in sd if k.endswith("embeds.proj.weight") and "audio" in k]
    if len(keys) != 1:
        raise SystemExit(f"expected exactly one audio patch-embed weight, found {keys}")
    k = keys[0]
    W = sd[k]
    embed, D = W.shape
    if D != C * pt * pf:
        raise SystemExit(f"{k} input dim {D} != C*pt*pf = {C * pt * pf} (C={C} pt={pt} pf={pf})")

    # legacy (C, p_t, p_f) -> canonical (C, p_f, p_t): transpose the within-patch spatial axes.
    W_new = W.reshape(embed, C, pt, pf).transpose(-1, -2).contiguous().reshape(embed, C * pf * pt)
    sd[k] = W_new
    if wrapper:
        ck["audio_patch_layout"] = "canonical"  # provenance marker

    tmp = path + ".convert_tmp"
    torch.save(ck, tmp)         # write canonical first; original untouched until both renames succeed
    os.rename(path, legacy)     # original -> *_legacy
    os.rename(tmp, path)        # canonical -> original name
    print(f"converted {os.path.basename(path)}  [{k} {tuple(W.shape)}, C={C} pt={pt} pf={pf}, "
          f"max|dW|={(W - W_new).abs().max().item():.3g}]  backup -> {os.path.basename(legacy)}")


def main():
    ap = argparse.ArgumentParser(
        description="Remap NaFlexClap audio patch-embed: legacy (C,pt,pf) -> canonical (C,pf,pt), in place.")
    ap.add_argument("model_name", help="open_clip config name (reads in_chans/patch_time/patch_freq)")
    ap.add_argument("ckpts", nargs="+", help="checkpoint(s); each original -> *_legacy, canonical -> original name")
    args = ap.parse_args()
    C, pt, pf = _geom(args.model_name)
    print(f"{args.model_name}: in_chans={C} patch_time={pt} patch_freq={pf}")
    ok = skip = 0
    for path in args.ckpts:
        try:
            convert(path, C, pt, pf)
            ok += 1
        except SystemExit as exc:  # per-file: skip and continue (e.g. backup already exists / already canonical)
            print(f"SKIP {os.path.basename(path)}: {exc}")
            skip += 1
    print(f"done: {ok} converted, {skip} skipped")


if __name__ == "__main__":
    main()
