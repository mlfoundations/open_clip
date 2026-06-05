import logging
import math
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import default_collate

from open_clip.naflex_config import NaFlexDataConfig, PatchSize, to_2tuple

try:
    from timm.data.naflex_dataset import NaFlexCollator, calculate_naflex_batch_size
    from timm.data.naflex_transforms import Patchify

    NAFLEX_AVAILABLE = True
except ImportError:
    NaFlexCollator = None
    calculate_naflex_batch_size = None
    Patchify = None
    NAFLEX_AVAILABLE = False


Sample = Dict[str, Any]


__all__ = [
    "NAFLEX_AVAILABLE",
    "AudioTokenLength",
    "CaptionLength",
    "LengthBucketer",
    "NaFlexBatchScheduler",
    "NaFlexBatcher",
    "NaFlexMapDatasetWrapper",
    "collate_naflex_dicts",
    "collate_naflex_tuples",
    "create_naflex_data_config_from_args",
    "create_naflex_eval_transform",
    "get_naflex_model_image_seq_len",
    "get_naflex_model_patch_size",
    "require_naflex",
    "resolve_patch_cfg",
]


def require_naflex() -> None:
    if not NAFLEX_AVAILABLE:
        raise RuntimeError(
            "NaFlex requires a timm version with NaFlex data support, including eval patchify transforms. "
            "Install timm>=1.0.16 or a recent timm main checkout."
        )


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
        return [to_2tuple(patch_size)], [1.0], False

    sizes = [to_2tuple(size) for size in patch_size_choices]
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


def get_naflex_model_patch_size(model) -> Optional[Tuple[int, int]]:
    visual = getattr(model, 'visual', None)
    trunk = getattr(visual, 'trunk', visual)
    if trunk is not None and hasattr(trunk, 'get_patch_size'):
        return to_2tuple(trunk.get_patch_size())
    return None


def get_naflex_model_image_seq_len(model) -> Optional[int]:
    image_seq_len = getattr(getattr(model, 'visual', None), 'image_seq_len', None)
    return int(image_seq_len) if image_seq_len is not None else None


def create_naflex_data_config_from_args(
        args,
        default_patch_size: Optional[PatchSize] = None,
        default_eval_seq_len: Optional[int] = None,
) -> NaFlexDataConfig:
    patch_sizes = getattr(args, 'naflex_patch_sizes', None)
    if patch_sizes is None and default_patch_size is not None:
        patch_sizes = [default_patch_size]
    seq_lens = getattr(args, 'naflex_seq_lens', None)
    return NaFlexDataConfig.resolve(
        patch_sizes=patch_sizes,
        patch_size_probs=getattr(args, 'naflex_patch_size_probs', None),
        seq_lens=seq_lens,
        train_num_image_tokens=getattr(args, 'naflex_num_train_image_tokens', None),
        max_tokens_per_batch=getattr(args, 'naflex_max_tokens_per_batch', 4096 * 4),
        batch_divisor=getattr(args, 'naflex_batch_divisor', 8),
        eval_seq_len=default_eval_seq_len if seq_lens is None else None,
    )


def create_naflex_eval_transform(
        transform_factory,
        naflex_data_config: NaFlexDataConfig,
) -> Tuple[Callable, int, Tuple[int, int]]:
    require_naflex()
    if not getattr(transform_factory, 'is_naflex_eval_transform_factory', False):
        raise ValueError("NaFlex eval requires `--aug-cfg use_timm=True naflex=True`.")

    patch_size, max_seq_len = naflex_data_config.eval_config
    return transform_factory(max_seq_len=max_seq_len, patch_size=patch_size), max_seq_len, patch_size


def collate_naflex_tuples(
        batch: List[Tuple[Dict[str, torch.Tensor], Any]],
        max_seq_len: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    require_naflex()
    return NaFlexCollator(max_seq_len=max_seq_len)(batch)


def collate_variable_text(
        targets: List[Any],
        pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length token id sequences into a batch.

    Args:
        targets: List of 1-D token id tensors (or sequences) of differing lengths.
        pad_id: Padding token id used to fill, and to derive the validity mask.

    Returns:
        Tuple ``(text, text_valid)`` of shape ``[B, Lmax]``; ``text_valid`` is ``text != pad_id``.
    """
    tensors = [t if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.long) for t in targets]
    max_len = max((t.shape[0] for t in tensors), default=0)
    text = torch.full((len(tensors), max_len), pad_id, dtype=torch.long)
    for i, tokens in enumerate(tensors):
        text[i, :tokens.shape[0]] = tokens
    return text, text != pad_id


def collate_naflex_dicts(
        batch: List[Sample],
        image_key: str = "image",
        target_key: str = "text",
        max_seq_len: Optional[int] = None,
        pad_id: Optional[int] = None,
) -> Dict[str, Any]:
    if pad_id is not None:
        # Generative (GenLIP) path: variable-length captions padded within the batch.
        images = NaFlexBatchScheduler._collate_images(
            [sample[image_key] for sample in batch], max_seq_len,
        )
        text, text_valid = collate_variable_text([sample[target_key] for sample in batch], pad_id)
        return {
            image_key: images,
            target_key: text,
            f"{target_key}_valid": text_valid,
        }

    images, targets = collate_naflex_tuples(
        [(sample[image_key], sample[target_key]) for sample in batch],
        max_seq_len=max_seq_len,
    )
    return {
        image_key: images,
        target_key: targets,
    }


def _padded_per_rank(total: int, distributed: bool, world_size: int) -> int:
    if total <= 0:
        raise ValueError("NaFlex schedule size must be positive.")
    if distributed and world_size > 1:
        return math.ceil(total / world_size)
    return total


class CaptionLength:
    """Length-fn: caption (text) token count of a sample. Picklable (forkserver-safe DataLoader workers)."""

    def __init__(self, key: str = "text"):
        self.key = key

    def __call__(self, sample: Sample) -> int:
        value = sample.get(self.key)
        if value is None:
            return 0
        return value.shape[0] if hasattr(value, "shape") else len(value)


class AudioTokenLength:
    """Length-fn: estimated NaFlex audio-patch count ``k`` before patchify.

    Bucketing sees decoded ``(waveform, sr)`` samples, so this mirrors ``AudioNaFlexPatchify``: resample-aware
    frame count, ceil to time patches, multiply by freq tokens, and optionally clamp to the largest cap.
    """

    def __init__(
            self,
            audio_key: str = "audio",
            freq_tokens: int = 1,
            patch_time: int = 1,
            hop_size: int = 1,
            window_size: int = 0,
            sample_rate: int = 0,
            max_audio_tokens: int = 0,
    ):
        self.audio_key = audio_key
        self.freq_tokens = max(1, int(freq_tokens))
        self.patch_time = max(1, int(patch_time))
        self.hop_size = max(1, int(hop_size))
        self.window_size = max(0, int(window_size))  # mel STFT floor: short clips padded up to one window
        self.sample_rate = int(sample_rate)
        self.max_audio_tokens = max(0, int(max_audio_tokens))  # cap (== max seq-len bucket); 0 = uncapped

    def __call__(self, sample: Sample) -> int:
        audio = sample.get(self.audio_key)
        if not (isinstance(audio, (tuple, list)) and audio and hasattr(audio[0], "shape")):
            return 0
        waveform, sr = audio[0], (audio[1] if len(audio) > 1 else 0)
        num_samples = waveform.shape[-1]
        if self.sample_rate and sr and sr != self.sample_rate:
            num_samples = num_samples * self.sample_rate / sr  # mel runs at sample_rate -> resampled frame count
        num_samples = max(num_samples, self.window_size)  # mirror the transform's pad of sub-window clips
        frames = int(num_samples // self.hop_size) + 1
        time_tokens = max(1, math.ceil(frames / self.patch_time))  # match the transform's ceil-pad (not floor-crop)
        tokens = self.freq_tokens * time_tokens
        return min(tokens, self.max_audio_tokens) if self.max_audio_tokens else tokens


class LengthBucketer:
    """WebDataset stage that reorders samples so similar sequence lengths batch together.

    Sort key is ``sum(fn(sample) for fn in length_fns)``: NaFlexClap ``[audio]``, GenLAP ``[audio, caption]``,
    GenLIP ``[caption]``. Reorder-only; every sample and step count is preserved.

    Module-level and picklable (no closures) for forkserver-safe DataLoader workers, mirroring ``TokenizeText``.
    """

    def __init__(
            self,
            length_fns: Optional[Sequence[Callable[[Sample], int]]] = None,
            pool: int = 2048,
            chunk: int = 128,
            seed: int = 42,
            epoch=-1,
    ):
        self.length_fns = list(length_fns) if length_fns else [CaptionLength()]
        self.pool = max(1, int(pool))
        self.chunk = max(1, int(chunk))
        self.seed = int(seed)
        self.epoch = epoch

    def _epoch(self) -> int:
        if hasattr(self.epoch, "get_value"):
            return int(self.epoch.get_value())
        return int(self.epoch)

    def _length(self, sample: Sample) -> int:
        return sum(fn(sample) for fn in self.length_fns)

    def _flush(self, buffer: List[Sample], rng: random.Random):
        buffer.sort(key=self._length)
        chunks = [buffer[i:i + self.chunk] for i in range(0, len(buffer), self.chunk)]
        rng.shuffle(chunks)
        for chunk in chunks:
            yield from chunk

    def __call__(self, src: Iterable[Sample]):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(self.seed + self._epoch() * 131 + worker_id)
        buffer: List[Sample] = []
        for sample in src:
            buffer.append(sample)
            if len(buffer) >= self.pool:
                yield from self._flush(buffer, rng)
                buffer = []
        if buffer:
            yield from self._flush(buffer, rng)


class NaFlexBatchScheduler:
    """Shared NaFlex schedule, patchify, and collation logic."""

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
            batch_divisor: int = 8,
            image_key: str = "image",
            target_key: str = "text",
            pad_id: Optional[int] = None,
            per_row_text_tokens: int = 0,
            pad_multiple: Optional[int] = None,
    ) -> None:
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
        self.batch_divisor = int(batch_divisor)
        if self.batch_divisor <= 0:
            raise ValueError("`batch_divisor` must be positive.")
        self.image_key = image_key
        self.target_key = target_key
        self.pad_id = pad_id
        # Per-row text token cost added when sizing batches so the token budget counts image + text (the
        # worst-case caption length, i.e. the tokenizer context-length cap). 0 = image-only batch sizing
        # (default; preserves CLIP/SigLIP behavior). GenLIP sets this to the caption cap so the budget bounds
        # total (image+text) tokens per batch. It does NOT change the image bucket used for patchify/collation.
        self.per_row_text_tokens = int(per_row_text_tokens or 0)

        # Native audio pads to batch max, optionally modulus-rounded for a smaller compile shape set.
        # Image paths keep pad-to-seq_len because their transforms resize to the sampled bucket.
        if pad_multiple is not None and int(pad_multiple) <= 0:
            raise ValueError(f"`pad_multiple` (--naflex-pad-multiple) must be > 0 when set, got {pad_multiple}.")
        self.pad_multiple = int(pad_multiple) if pad_multiple else None
        self.pad_freq_tokens = getattr(getattr(transform_factory, "cfg", None), "freq_tokens", None)
        if self.pad_multiple and self.pad_freq_tokens and any(s % self.pad_multiple for s in self.seq_lens):
            logging.info(
                "NaFlex pad-multiple=%d does not divide all seq_lens %s; a few extra cap-aligned batch shapes "
                "will appear under torch.compile (pick seq_lens divisible by %d to avoid).",
                self.pad_multiple, self.seq_lens, self.pad_multiple,
            )

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
                seq_len=seq_len + self.per_row_text_tokens,  # row cost = image bucket + text cap
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
                seq_len=seq_len + self.per_row_text_tokens,  # row cost = image bucket + text cap
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

    def epoch_schedule(self, epoch: int, num_workers: int = 1) -> List[Tuple[int, int]]:
        schedule = list(self._canonical_batch_schedule)
        if self.shuffle:
            random.Random(self.seed + epoch).shuffle(schedule)
        schedule = self.pad_schedule_for_workers(schedule, max(1, num_workers))
        return schedule

    @staticmethod
    def pad_schedule_for_workers(
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

    def worker_schedule(self, epoch: int) -> List[Tuple[int, int]]:
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        schedule = self.epoch_schedule(epoch, num_workers=num_workers)
        return schedule[worker_id::num_workers]

    def num_batches_for_workers(self, num_workers: int) -> int:
        schedule = self.pad_schedule_for_workers(list(self._canonical_batch_schedule), max(1, num_workers))
        return len(schedule)

    def num_samples_for_workers(self, num_workers: int) -> int:
        schedule = self.pad_schedule_for_workers(list(self._canonical_batch_schedule), max(1, num_workers))
        num_samples_per_rank = sum(batch_size for _, batch_size in schedule)
        if self.distributed:
            return num_samples_per_rank * self.world_size
        return num_samples_per_rank

    def sample_patch_idx(self, generator: torch.Generator) -> int:
        if not self.variable_patch_size:
            return 0
        probs = torch.tensor(self.patch_size_probs, dtype=torch.float32)
        return int(torch.multinomial(probs, 1, generator=generator).item())

    @staticmethod
    def _collate_images(
            patch_dicts: List[Dict[str, torch.Tensor]],
            max_seq_len: int,
            pad_multiple: Optional[int] = None,
            freq_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if freq_tokens is None:
            # Image / fixed-bucket path: pad to the sampled seq-len bucket (or batch-max if seq_len is 0/None).
            max_patches = max_seq_len or max(pd["patches"].shape[0] for pd in patch_dicts)
        else:
            # Native audio: pad to batch max, optionally modulus-rounded, then clamp to the whole-column cap.
            batch_max = max(pd["patches"].shape[0] for pd in patch_dicts)
            max_patches = math.ceil(batch_max / pad_multiple) * pad_multiple if pad_multiple else batch_max
            if max_seq_len:
                max_patches = min(max_patches, freq_tokens * (max_seq_len // freq_tokens))
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

    def collate_batch(
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

        images = self._collate_images(
            patch_dicts, seq_len, pad_multiple=self.pad_multiple, freq_tokens=self.pad_freq_tokens,
        )
        if self.pad_id is not None:
            # Generative (GenLIP) path: variable-length captions padded within the batch.
            text, text_valid = collate_variable_text(targets, self.pad_id)
            return {
                self.image_key: images,
                self.target_key: text,
                f"{self.target_key}_valid": text_valid,
            }

        return {
            self.image_key: images,
            self.target_key: default_collate(targets),
        }


class NaFlexBatcher:
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
            pad_id: Optional[int] = None,
            per_row_text_tokens: int = 0,
            pad_multiple: Optional[int] = None,
    ) -> None:
        self.epoch = epoch
        self.scheduler = NaFlexBatchScheduler(
            train_num_samples=train_num_samples,
            train_num_tokens=train_num_tokens,
            patch_size=patch_size,
            patch_size_choices=patch_size_choices,
            patch_size_choice_probs=patch_size_choice_probs,
            seq_lens=seq_lens,
            max_tokens_per_batch=max_tokens_per_batch,
            transform_factory=transform_factory,
            seed=seed,
            shuffle=shuffle,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            batch_divisor=batch_divisor,
            image_key=image_key,
            target_key=target_key,
            pad_id=pad_id,
            per_row_text_tokens=per_row_text_tokens,
            pad_multiple=pad_multiple,
        )

    @property
    def num_batches(self) -> int:
        return self.scheduler.num_batches

    @property
    def num_samples(self) -> int:
        return self.scheduler.num_samples

    def __len__(self) -> int:
        return self.num_batches

    def __call__(self, src: Iterable[Sample]):
        return self.run(src)

    def num_batches_for_workers(self, num_workers: int) -> int:
        return self.scheduler.num_batches_for_workers(num_workers)

    def num_samples_for_workers(self, num_workers: int) -> int:
        return self.scheduler.num_samples_for_workers(num_workers)

    def _epoch(self) -> int:
        if hasattr(self.epoch, "get_value"):
            return int(self.epoch.get_value())
        self.epoch += 1
        return int(self.epoch)

    def run(self, src: Iterable[Sample]):
        epoch = self._epoch()
        generator = torch.Generator()
        generator.manual_seed(self.scheduler.seed + epoch)
        samples_iter = iter(src)

        for seq_len, batch_size in self.scheduler.worker_schedule(epoch):
            patch_idx = self.scheduler.sample_patch_idx(generator)
            samples = []
            for _ in range(batch_size):
                sample = next(samples_iter)
                if not isinstance(sample, dict):
                    raise TypeError("NaFlexBatcher expects dictionary samples from the data pipeline.")
                samples.append(sample)
            if samples:
                yield self.scheduler.collate_batch(samples, seq_len, patch_idx)


class NaFlexMapDatasetWrapper(IterableDataset):
    """Map-style dataset wrapper that yields NaFlex dict batches."""

    def __init__(
            self,
            base_dataset: Dataset,
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
            pad_id: Optional[int] = None,
            per_row_text_tokens: int = 0,
    ) -> None:
        if not hasattr(base_dataset, '__len__') or not hasattr(base_dataset, '__getitem__'):
            raise TypeError("NaFlex map batching requires a map-style dataset.")

        self.base_dataset = base_dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.distributed = bool(distributed)
        self.rank = int(rank) if distributed else 0
        self.world_size = int(world_size) if distributed else 1
        self.epoch = epoch

        train_num_samples = None if train_num_tokens is not None else len(base_dataset)
        self.scheduler = NaFlexBatchScheduler(
            train_num_samples=train_num_samples,
            train_num_tokens=train_num_tokens,
            patch_size=patch_size,
            patch_size_choices=patch_size_choices,
            patch_size_choice_probs=patch_size_choice_probs,
            seq_lens=seq_lens,
            max_tokens_per_batch=max_tokens_per_batch,
            transform_factory=transform_factory,
            seed=seed,
            shuffle=shuffle,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            batch_divisor=batch_divisor,
            pad_id=pad_id,
            per_row_text_tokens=per_row_text_tokens,
        )

    def __len__(self) -> int:
        return self.scheduler.num_batches

    def num_batches_for_workers(self, num_workers: int) -> int:
        return self.scheduler.num_batches_for_workers(num_workers)

    def num_samples_for_workers(self, num_workers: int) -> int:
        return self.scheduler.num_samples_for_workers(num_workers)

    def _epoch(self) -> int:
        if hasattr(self.epoch, "get_value"):
            return int(self.epoch.get_value())
        self.epoch += 1
        return int(self.epoch)

    def _epoch_indices(self, epoch: int, samples_per_rank: int) -> List[int]:
        dataset_len = len(self.base_dataset)
        if dataset_len <= 0:
            raise ValueError("NaFlex map batching requires at least one sample.")

        total_samples = samples_per_rank * self.world_size if self.distributed else samples_per_rank
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        indices = []
        while len(indices) < total_samples:
            if self.shuffle:
                indices.extend(torch.randperm(dataset_len, generator=generator).tolist())
            else:
                indices.extend(range(dataset_len))
        indices = indices[:total_samples]

        if self.distributed:
            return indices[self.rank::self.world_size]
        return indices

    def __iter__(self):
        epoch = self._epoch()
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        schedule = self.scheduler.epoch_schedule(epoch, num_workers=num_workers)
        samples_per_rank = sum(batch_size for _, batch_size in schedule)
        rank_indices = self._epoch_indices(epoch, samples_per_rank)

        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)

        index_offset = 0
        for batch_idx, (seq_len, batch_size) in enumerate(schedule):
            batch_indices = rank_indices[index_offset:index_offset + batch_size]
            index_offset += batch_size
            if batch_idx % num_workers != worker_id:
                continue

            patch_idx = self.scheduler.sample_patch_idx(generator)
            samples = [self.base_dataset[idx] for idx in batch_indices]
            yield self.scheduler.collate_batch(samples, seq_len, patch_idx)
