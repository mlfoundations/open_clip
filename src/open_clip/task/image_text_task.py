"""Intermediate task layer for image+text contrastive tasks.

Holds the CLIP-family modality contract: ``data_keys = ("image", "text")``,
positional ``task(image, text)`` backward-compat in ``forward()``,
``create_dummy_batch`` for FSDP eval scaffolding, and ``clamp_logit_scale``.
Concrete tasks (CLIPTask, SigLIPTask, CoCaTask, DistillCLIPTask) inherit
from this layer.

Future modalities (NaFlex, CLAP, MamMuT) should derive directly from
``TrainingTask`` and supply their own contract.
"""
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..naflex_config import NaFlexDataConfig
from .base_task import TrainingTask, unwrap_model


class ImageTextTask(TrainingTask):
    """Image + text contract shared by CLIP-family tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._naflex_data_config = None

    @property
    def data_keys(self) -> Tuple[str, ...]:
        """Keys expected in the batch dict from the data pipeline."""
        return ("image", "text")

    @property
    def naflex_data_config(self) -> Optional[NaFlexDataConfig]:
        return self._naflex_data_config

    @property
    def naflex_eval_config(self) -> Optional[Tuple[Tuple[int, int], int]]:
        return self._naflex_data_config.eval_config if self._naflex_data_config is not None else None

    def set_naflex_data_config(
            self,
            naflex_data_config: Optional[NaFlexDataConfig],
    ) -> 'ImageTextTask':
        """Configure NaFlex train/eval data policy shared by data loaders and dummy batches."""
        self._naflex_data_config = naflex_data_config
        return self

    def create_dummy_batch(
            self,
            image_size=None,
            context_length: Optional[int] = None,
            batch_size: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Create a dummy batch for FSDP eval on non-rank-0 workers."""
        model = unwrap_model(self.trainable_module)
        if context_length is None:
            context_length = model.context_length

        if self._naflex_data_config is not None:
            image = self._create_naflex_dummy_image(
                batch_size=batch_size,
                max_seq_len=self._naflex_data_config.eval_seq_len,
                patch_size=self._naflex_data_config.eval_patch_size,
                device=device,
                dtype=dtype,
            )
        else:
            if image_size is None:
                image_size = model.visual.image_size
            if not isinstance(image_size, tuple):
                image_size = (image_size, image_size)
            image = torch.zeros(batch_size, 3, *image_size, device=device, dtype=dtype)

        return {
            "image": image,
            "text": torch.zeros(batch_size, context_length, device=device, dtype=torch.long),
        }

    @staticmethod
    def _create_naflex_dummy_image(
            batch_size: int,
            max_seq_len: int,
            patch_size: Tuple[int, int],
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            num_channels: int = 3,
    ) -> Dict[str, torch.Tensor]:
        patch_dim = patch_size[0] * patch_size[1] * num_channels
        patches = torch.zeros(batch_size, max_seq_len, patch_dim, device=device, dtype=dtype)

        width = math.ceil(math.sqrt(max_seq_len))
        patch_idx = torch.arange(max_seq_len, device=device)
        patch_coord = torch.stack((patch_idx // width, patch_idx % width), dim=-1).long()
        patch_coord = patch_coord.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        patch_valid = torch.ones(batch_size, max_seq_len, device=device, dtype=torch.bool)

        return {
            "patches": patches,
            "patch_coord": patch_coord,
            "patch_valid": patch_valid,
            "seq_len": max_seq_len,
        }

    def clamp_logit_scale(self, max_val: float = math.log(100)):
        """Clamp logit_scale parameter to [0, max_val].

        With FSDP2, logit_scale is a replicated DTensor. In-place clamp_
        dispatches to the local tensor on each rank, which is correct for
        a single-element replicated param.
        """
        model = unwrap_model(self.trainable_module)
        if hasattr(model, 'logit_scale'):
            with torch.no_grad():
                model.logit_scale.clamp_(0, max_val)
