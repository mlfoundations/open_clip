"""State-dict conversion helpers for CLAP audio checkpoints."""

import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Union

import torch


_BLOCK_RE = re.compile(r"audio_model\.audio_encoder\.layers\.(\d+)\.blocks\.(\d+)\.(.+)")
_QKV_RE = re.compile(
    r"audio_model\.audio_encoder\.layers\.(\d+)\.blocks\.(\d+)\.attention\.self\.(query|key|value)\.(weight|bias)"
)


def _load_checkpoint(path: Union[str, Path]) -> Mapping[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _map_audio_block_key(key: str) -> str:
    match = _BLOCK_RE.match(key)
    if not match:
        return ""

    layer_idx, block_idx, suffix = match.groups()
    target = f"audio.encoder.layers.{layer_idx}.blocks.{block_idx}."
    replacements = (
        ("layernorm_before.", "norm1."),
        ("layernorm_after.", "norm2."),
        ("attention.self.relative_position_bias_table", "attn.relative_position_bias_table"),
        ("attention.self.relative_position_index", "attn.relative_position_index"),
        ("attention.output.dense.", "attn.proj."),
        ("intermediate.dense.", "mlp.fc1."),
        ("output.dense.", "mlp.fc2."),
    )
    for old, new in replacements:
        if suffix.startswith(old):
            return target + suffix.replace(old, new, 1)
    return ""


def _convert_hf_clap_qkv(
        state_dict: Mapping[str, torch.Tensor],
        output: MutableMapping[str, torch.Tensor],
) -> None:
    grouped = {}
    for key, value in state_dict.items():
        match = _QKV_RE.match(key)
        if not match:
            continue
        layer_idx, block_idx, qkv_name, param_name = match.groups()
        grouped.setdefault((layer_idx, block_idx, param_name), {})[qkv_name] = value

    for (layer_idx, block_idx, param_name), tensors in grouped.items():
        if not all(name in tensors for name in ("query", "key", "value")):
            continue
        output[f"audio.encoder.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.{param_name}"] = torch.cat(
            [tensors["query"], tensors["key"], tensors["value"]],
            dim=0,
        )


def convert_hf_clap_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Convert a Transformers ``ClapModel`` checkpoint to OpenCLIP CLAP keys."""
    state_dict = _strip_module_prefix(state_dict)
    output = OrderedDict()
    _convert_hf_clap_qkv(state_dict, output)

    for key, value in state_dict.items():
        if _QKV_RE.match(key):
            continue

        if key == "logit_scale_a":
            # HF CLAP has directional scales. In the released LAION checkpoints,
            # logit_scale_t remains at init while logit_scale_a is learned, so map
            # the audio->text scale onto OpenCLIP's single symmetric logit_scale.
            output["logit_scale"] = value
        elif key == "audio_model.audio_encoder.batch_norm.num_batches_tracked":
            output["audio.encoder.bn0.num_batches_tracked"] = value
        elif key.startswith("audio_model.audio_encoder.batch_norm."):
            output[key.replace("audio_model.audio_encoder.batch_norm.", "audio.encoder.bn0.", 1)] = value
        elif key.startswith("audio_model.audio_encoder.patch_embed."):
            output[key.replace("audio_model.audio_encoder.patch_embed.", "audio.encoder.patch_embed.", 1)] = value
        elif key.startswith("audio_model.audio_encoder.norm."):
            output[key.replace("audio_model.audio_encoder.norm.", "audio.encoder.norm.", 1)] = value
        elif key.startswith("audio_model.audio_encoder.layers.") and ".blocks." in key:
            mapped_key = _map_audio_block_key(key)
            if mapped_key:
                output[mapped_key] = value
        elif key.startswith("audio_model.audio_encoder.layers."):
            output[key.replace("audio_model.audio_encoder.layers.", "audio.encoder.layers.", 1)] = value
        elif key.startswith("audio_projection.linear1."):
            output[key.replace("audio_projection.linear1.", "audio.proj.0.", 1)] = value
        elif key.startswith("audio_projection.linear2."):
            output[key.replace("audio_projection.linear2.", "audio.proj.2.", 1)] = value
        elif key.startswith("text_model."):
            text_key = key.replace("text_model.", "text.transformer.", 1)
            if not text_key.endswith((".position_ids", ".token_type_ids")):
                output[text_key] = value
        elif key.startswith("text_projection.linear1."):
            output[key.replace("text_projection.linear1.", "text.proj.0.", 1)] = value
        elif key.startswith("text_projection.linear2."):
            output[key.replace("text_projection.linear2.", "text.proj.2.", 1)] = value

    return output


def load_hf_clap_state_dict(path: Union[str, Path]) -> OrderedDict[str, torch.Tensor]:
    """Load and convert a Transformers CLAP checkpoint from disk."""
    return convert_hf_clap_state_dict(_load_checkpoint(path))
