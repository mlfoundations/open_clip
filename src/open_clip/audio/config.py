from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CLIPAudioCfg:
    model_type: str = "HTSAT"
    model_name: str = "tiny"
    audio_length: int = 1024
    clip_samples: int = 480000
    sample_rate: int = 48000
    mel_bins: int = 64
    window_size: int = 1024
    hop_size: int = 480
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    enable_fusion: bool = False
    fusion_type: str = "aff_2d"
    pre_norm: bool = False
    proj_act: str = "gelu"
    training_head: bool = False
    pretrained: bool = False

    # NaFlexClap (model_type == "naflexvit"): spectrogram-ViT encoder geometry. Mel fields above are reused.
    patch_freq: int = 64   # p_f: freq bins per patch (== mel_bins -> full-height strips)
    patch_time: int = 4    # p_t: time frames per patch
    in_chans: int = 1
    rope_type: str = "axial"  # 2-D (freq, time) RoPE; 'mrope'/'' overridable (MRoPE needs naflexvit_cfg section)
    naflexvit_cfg: Dict[str, Any] = field(default_factory=dict)  # NaFlexVitCfg overrides (embed_dim/depth/...)
