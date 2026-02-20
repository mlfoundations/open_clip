from typing import Dict, Optional

import torch
import torch.nn as nn

from .task import TrainingTask


class DistillCLIPTask(TrainingTask):
    """Distillation task wrapping student model + frozen teacher + DistillClipLoss."""

    def __init__(
            self,
            student_model: nn.Module,
            teacher_model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            local_loss: bool = False,
            gather_with_grad: bool = False,
            cache_labels: bool = True,
            rank: int = 0,
            world_size: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)
        self.trainable_module = student_model
        self.teacher = teacher_model
        self.teacher.eval()
        # Freeze teacher parameters
        for p in self.teacher.parameters():
            p.requires_grad = False
        if loss is not None:
            self.loss = loss
        else:
            from open_clip.loss import DistillClipLoss
            self.loss = DistillClipLoss(
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=cache_labels,
                rank=rank,
                world_size=world_size,
            )

    def train(self, mode: bool = True):
        """Override to keep teacher always in eval mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def training_forward(self, images: torch.Tensor, texts: torch.Tensor) -> Dict[str, torch.Tensor]:
        model_out = self.trainable_module(images, texts)
        logit_scale = model_out["logit_scale"]
        with torch.no_grad():
            teacher_out = self.teacher(images, texts)
        model_out.update({f'dist_{k}': v for k, v in teacher_out.items()})
        losses = self.loss(**model_out, output_dict=True)
        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
        losses["loss"] = total_loss
        losses["logit_scale"] = logit_scale
        return losses
