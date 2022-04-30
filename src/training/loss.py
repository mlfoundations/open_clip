from dataclasses import dataclass


@dataclass
class LossCfg:
    type: str = 'clip'
    local_loss: bool = True
    gather_with_grad: bool = True
    cache_labels: bool = True
