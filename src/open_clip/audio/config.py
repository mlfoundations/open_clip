from dataclasses import dataclass


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
