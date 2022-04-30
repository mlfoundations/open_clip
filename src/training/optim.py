from dataclasses import dataclass
from typing import Tuple


@dataclass
class OptimCfg:
    type: str = 'adamw'
    lr: float = 5e-4
    wd: float = 0.2
    betas: Tuple[int, int] = (0.9, 0.999)
    eps: float = 1e-8
