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
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
            **kwargs,
    ):
        if loss is not None:
            super().__init__(model, loss=loss, **kwargs)
        else:
            from open_clip.loss import SigLipLoss
            super().__init__(
                model,
                loss=SigLipLoss(
                    rank=rank,
                    world_size=world_size,
                    dist_impl=dist_impl,
                ),
                **kwargs,
            )
