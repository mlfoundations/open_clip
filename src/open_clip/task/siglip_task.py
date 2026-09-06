from typing import Optional

import torch.nn as nn

from .clip_task import CLIPTask


class SigLIPTask(CLIPTask):
    """SigLIP task. Inherits forward() from CLIPTask (loss handles differences).

    Separate class for type distinction and to construct SigLipLoss by default.
    """

    def __init__(
            self,
            model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            default_loss: bool = True,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
            **kwargs,
    ):
        if loss is None and default_loss:
            from open_clip.loss import SigLipLoss
            loss = SigLipLoss(rank=rank, world_size=world_size, dist_impl=dist_impl)
        super().__init__(model, loss=loss, default_loss=False, rank=rank, world_size=world_size, **kwargs)
