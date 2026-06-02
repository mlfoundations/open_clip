import importlib.util

from .config import CLIPAudioCfg
from .naflex_audio import (
    AudioNaFlexCfg,
    AudioNaFlexPatchify,
    AudioNaFlexTransformFactory,
    MelPatchEmbed,
    mel_to_patches,
)
from .tower import AudioTower
from .transform import AudioAugmentationCfg, audio_transform_v2

_AUDIO_DEPS = {
    "torchaudio": "torchaudio",
    "torchlibrosa": "torchlibrosa",
    "whisper": "openai-whisper",
}
AUDIO_AVAILABLE = all(importlib.util.find_spec(dep) is not None for dep in _AUDIO_DEPS)


def require_audio():
    missing = [package for module, package in _AUDIO_DEPS.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(
            "CLAP audio support requires optional audio dependencies: " + ", ".join(missing)
        )


__all__ = [
    "AUDIO_AVAILABLE",
    "AudioAugmentationCfg",
    "AudioNaFlexCfg",
    "AudioNaFlexPatchify",
    "AudioNaFlexTransformFactory",
    "AudioTower",
    "CLIPAudioCfg",
    "MelPatchEmbed",
    "audio_transform_v2",
    "mel_to_patches",
    "require_audio",
]
