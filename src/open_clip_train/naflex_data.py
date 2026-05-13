import math
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import webdataset as wds
from torch.utils.data.dataloader import default_collate

try:
    from timm.data.naflex_dataset import calculate_naflex_batch_size
    from timm.data.naflex_transforms import Patchify

    NAFLEX_AVAILABLE = True
except ImportError:
    calculate_naflex_batch_size = None
    Patchify = None
    NAFLEX_AVAILABLE = False


PatchSize = Union[int, Tuple[int, int]]
Sample = Dict[str, Any]


def require_naflex() -> None:
    if not NAFLEX_AVAILABLE:
        raise RuntimeError(
            "NaFlex batching requires a timm version with NaFlex data support. "
            "Install timm>=1.0.16 or a recent timm main checkout."
        )


def _to_2tuple(value: PatchSize) -> Tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Patch size tuples must have exactly two values.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def resolve_patch_cfg(
        patch_size: Optional[PatchSize] = None,
        patch_size_choices: Optional[Sequence[PatchSize]] = None,
        patch_size_choice_probs: Optional[Sequence[float]] = None,
) -> Tuple[List[Tuple[int, int]], List[float], bool]:
    # Mirrors timm.data.naflex_dataset._resolve_patch_cfg without importing a private helper.
    if patch_size is None and patch_size_choices is None:
        patch_size = 16
    if (patch_size is None) == (patch_size_choices is None):
        raise ValueError("Specify exactly one of `patch_size` or `patch_size_choices`.")

    if patch_size is not None:
        return [_to_2tuple(patch_size)], [1.0], False

    sizes = [_to_2tuple(size) for size in patch_size_choices]
    if not sizes:
        raise ValueError("`patch_size_choices` must contain at least one value.")
    if patch_size_choice_probs is None:
        probs = [1.0 / len(sizes)] * len(sizes)
    else:
        if len(patch_size_choice_probs) != len(sizes):
            raise ValueError("`patch_size_choice_probs` must match `patch_size_choices` length.")
        prob_sum = float(sum(patch_size_choice_probs))
        if prob_sum <= 0:
            raise ValueError("`patch_size_choice_probs` must sum to a positive value.")
        probs = [float(prob) / prob_sum for prob in patch_size_choice_probs]
    return sizes, probs, True


def _padded_per_rank(total: int, distributed: bool, world_size: int) -> int:
    if total <= 0:
        raise ValueError("NaFlex schedule size must be positive.")
    if distributed and world_size > 1:
        return math.ceil(total / world_size)
    return total


class NaFlexBatcher(wds.PipelineStage):
    """WebDataset stage that turns image/text samples into NaFlex dict batches."""

    def __init__(
            self,
            train_num_samples: Optional[int] = None,
            train_num_tokens: Optional[int] = None,
            patch_size: Optional[PatchSize] = None,
            patch_size_choices: Optional[Sequence[PatchSize]] = None,
            patch_size_choice_probs: Optional[Sequence[float]] = None,
            seq_lens: Sequence[int] = (128, 256, 576, 784, 1024),
            max_tokens_per_batch: int = 4096 * 4,
            transform_factory: Optional[Callable[..., Callable]] = None,
            seed: int = 42,
            shuffle: bool = True,
            distributed: bool = False,
            rank: int = 0,
            world_size: int = 1,
            epoch=-1,
            batch_divisor: int = 8,
            image_key: str = "image",
            target_key: str = "text",
    ) -> None:
        super().__init__()
        require_naflex()

        if (train_num_samples is None) == (train_num_tokens is None):
            raise ValueError("Specify exactly one of `train_num_samples` or `train_num_tokens` for NaFlex batching.")
        if transform_factory is None:
            raise ValueError("NaFlex batching requires a transform factory.")

        self.seq_lens = sorted(set(int(seq_len) for seq_len in seq_lens))
        if not self.seq_lens:
            raise ValueError("NaFlex batching requires at least one sequence length.")
        if not all(seq_len > 0 for seq_len in self.seq_lens):
            raise ValueError("NaFlex sequence lengths must be positive.")
        self.max_tokens_per_batch = int(max_tokens_per_batch)
        if self.max_tokens_per_batch <= 0:
            raise ValueError("`max_tokens_per_batch` must be positive.")
        self.transform_factory = transform_factory
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.distributed = bool(distributed)
        self.rank = int(rank) if distributed else 0
        self.world_size = int(world_size) if distributed else 1
        self.epoch = epoch
        self.batch_divisor = int(batch_divisor)
        if self.batch_divisor <= 0:
            raise ValueError("`batch_divisor` must be positive.")
        self.image_key = image_key
        self.target_key = target_key

        self.patch_sizes, self.patch_size_probs, self.variable_patch_size = resolve_patch_cfg(
            patch_size=patch_size,
            patch_size_choices=patch_size_choices,
            patch_size_choice_probs=patch_size_choice_probs,
        )
        self.transforms: Dict[Tuple[int, int], Optional[Callable]] = {}
        self.patchifiers: List[Callable] = []

        for patch_idx, patch_size_tuple in enumerate(self.patch_sizes):
            self.patchifiers.append(
                Patchify(
                    patch_size=patch_size_tuple,
                    flatten_patches=not self.variable_patch_size,
                )
            )
            for seq_len in self.seq_lens:
                self.transforms[(seq_len, patch_idx)] = transform_factory(
                    max_seq_len=seq_len,
                    patch_size=patch_size_tuple,
                )

        if train_num_samples is not None:
            self._create_schedule_from_num_samples(int(train_num_samples))
        else:
            self._create_schedule_from_num_tokens(int(train_num_tokens))

    def _next_seq_len(self, generator: torch.Generator) -> int:
        seq_idx = torch.randint(0, len(self.seq_lens), (1,), generator=generator).item()
        return self.seq_lens[seq_idx]

    def _create_schedule_from_num_samples(self, num_samples: int) -> None:
        samples_per_rank = _padded_per_rank(num_samples, self.distributed, self.world_size)
        remaining_samples = samples_per_rank
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        schedule = []
        while remaining_samples > 0:
            seq_len = self._next_seq_len(generator)
            batch_size = calculate_naflex_batch_size(
                tokens_per_batch=self.max_tokens_per_batch,
                seq_len=seq_len,
                max_size=remaining_samples,
                divisor=self.batch_divisor,
                rounding="floor",
            )
            batch_size = min(max(1, int(batch_size)), remaining_samples)
            schedule.append((seq_len, batch_size))
            remaining_samples -= batch_size

        self._canonical_batch_schedule = schedule
        self._num_batches_per_rank = len(schedule)
        self._num_samples_per_rank = sum(batch_size for _, batch_size in schedule)

    def _create_schedule_from_num_tokens(self, num_tokens: int) -> None:
        tokens_per_rank = _padded_per_rank(num_tokens, self.distributed, self.world_size)
        remaining_tokens = tokens_per_rank
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        schedule = []
        while remaining_tokens > 0:
            seq_len = self._next_seq_len(generator)
            batch_size = calculate_naflex_batch_size(
                tokens_per_batch=min(self.max_tokens_per_batch, remaining_tokens),
                seq_len=seq_len,
                divisor=self.batch_divisor,
                rounding="floor",
            )
            batch_size = max(1, int(batch_size))
            schedule.append((seq_len, batch_size))
            remaining_tokens -= batch_size * seq_len

        self._canonical_batch_schedule = schedule
        self._num_batches_per_rank = len(schedule)
        self._num_samples_per_rank = sum(batch_size for _, batch_size in schedule)

    @property
    def num_batches(self) -> int:
        return self._num_batches_per_rank

    @property
    def num_samples(self) -> int:
        if self.distributed:
            return self._num_samples_per_rank * self.world_size
        return self._num_samples_per_rank

    def __len__(self) -> int:
        return self.num_batches

    def _epoch_schedule(self, epoch: int) -> List[Tuple[int, int]]:
        schedule = list(self._canonical_batch_schedule)
        if self.shuffle:
            random.Random(self.seed + epoch).shuffle(schedule)
        return schedule

    @staticmethod
    def _pad_schedule_for_workers(
            schedule: List[Tuple[int, int]],
            num_workers: int,
    ) -> List[Tuple[int, int]]:
        if num_workers <= 1 or not schedule:
            return schedule
        target_batches = math.ceil(len(schedule) / num_workers) * num_workers
        pad_batches = target_batches - len(schedule)
        if pad_batches > 0:
            repeats = math.ceil(pad_batches / len(schedule))
            schedule = schedule + (schedule * repeats)[:pad_batches]
        return schedule

    def _schedule_for_worker(self, epoch: int) -> List[Tuple[int, int]]:
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        schedule = self._pad_schedule_for_workers(self._epoch_schedule(epoch), num_workers)
        return schedule[worker_id::num_workers]

    def num_batches_for_workers(self, num_workers: int) -> int:
        schedule = self._pad_schedule_for_workers(list(self._canonical_batch_schedule), max(1, num_workers))
        return len(schedule)

    def num_samples_for_workers(self, num_workers: int) -> int:
        schedule = self._pad_schedule_for_workers(list(self._canonical_batch_schedule), max(1, num_workers))
        num_samples_per_rank = sum(batch_size for _, batch_size in schedule)
        if self.distributed:
            return num_samples_per_rank * self.world_size
        return num_samples_per_rank

    def _sample_patch_idx(self, generator: torch.Generator) -> int:
        if not self.variable_patch_size:
            return 0
        probs = torch.tensor(self.patch_size_probs, dtype=torch.float32)
        return int(torch.multinomial(probs, 1, generator=generator).item())

    @staticmethod
    def _collate_images(patch_dicts: List[Dict[str, torch.Tensor]], max_seq_len: int) -> Dict[str, torch.Tensor]:
        max_patches = max_seq_len or max(patch_dict["patches"].shape[0] for patch_dict in patch_dicts)
        batch_size = len(patch_dicts)
        first_patches = patch_dicts[0]["patches"]
        patches = first_patches.new_zeros((batch_size, max_patches) + tuple(first_patches.shape[1:]))
        first_coord = patch_dicts[0]["patch_coord"]
        patch_coord = first_coord.new_zeros((batch_size, max_patches, 2))
        patch_valid = torch.zeros((batch_size, max_patches), dtype=torch.bool)

        for sample_idx, patch_dict in enumerate(patch_dicts):
            num_patches = min(patch_dict["patches"].shape[0], max_patches)
            patches[sample_idx, :num_patches] = patch_dict["patches"][:num_patches]
            patch_coord[sample_idx, :num_patches] = patch_dict["patch_coord"][:num_patches]
            patch_valid[sample_idx, :num_patches] = patch_dict["patch_valid"][:num_patches].bool()

        return {
            "patches": patches,
            "patch_coord": patch_coord,
            "patch_valid": patch_valid,
            "seq_len": max_patches,
        }

    def _collate_batch(
            self,
            samples: List[Sample],
            seq_len: int,
            patch_idx: int,
    ) -> Dict[str, Any]:
        transform = self.transforms[(seq_len, patch_idx)]
        patchify = self.patchifiers[patch_idx]
        patch_dicts = []
        targets = []

        for sample in samples:
            image = sample[self.image_key]
            image = transform(image) if transform is not None else image
            patch_dict = image if isinstance(image, dict) else patchify(image)
            patch_dicts.append(patch_dict)
            targets.append(sample[self.target_key])

        return {
            self.image_key: self._collate_images(patch_dicts, seq_len),
            self.target_key: default_collate(targets),
        }

    def run(self, src: Iterable[Sample]):
        if hasattr(self.epoch, "get_value"):
            epoch = int(self.epoch.get_value())
        else:
            self.epoch += 1
            epoch = int(self.epoch)

        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        samples_iter = iter(src)

        for seq_len, batch_size in self._schedule_for_worker(epoch):
            patch_idx = self._sample_patch_idx(generator)
            samples = []
            for _ in range(batch_size):
                sample = next(samples_iter)
                if not isinstance(sample, dict):
                    raise TypeError("NaFlexBatcher expects dictionary samples from the data pipeline.")
                samples.append(sample)
            if samples:
                yield self._collate_batch(samples, seq_len, patch_idx)
