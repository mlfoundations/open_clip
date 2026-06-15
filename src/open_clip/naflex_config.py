from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union


PatchSize = Union[int, Tuple[int, int]]


def to_2tuple(value: PatchSize) -> Tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Patch size tuples must have exactly two values.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


@dataclass(frozen=True)
class NaFlexDataConfig:
    train_patch_sizes: Tuple[Tuple[int, int], ...] = ((16, 16),)
    train_patch_size_probs: Optional[Tuple[float, ...]] = None
    train_seq_lens: Tuple[int, ...] = (128, 256, 576, 784, 1024)
    train_seq_len_probs: Optional[Tuple[float, ...]] = None
    train_num_image_tokens: Optional[int] = None
    max_tokens_per_batch: int = 4096 * 4
    batch_divisor: int = 8
    eval_patch_size: Tuple[int, int] = (16, 16)
    eval_seq_len: int = 1024

    @classmethod
    def resolve(
            cls,
            patch_sizes: Optional[Sequence[PatchSize]] = None,
            patch_size_probs: Optional[Sequence[float]] = None,
            seq_lens: Optional[Sequence[int]] = None,
            seq_len_probs: Optional[Sequence[float]] = None,
            train_num_image_tokens: Optional[int] = None,
            max_tokens_per_batch: int = 4096 * 4,
            batch_divisor: int = 8,
            eval_patch_size: Optional[PatchSize] = None,
            eval_seq_len: Optional[int] = None,
    ) -> 'NaFlexDataConfig':
        patch_sizes = (16,) if patch_sizes is None else patch_sizes
        train_patch_sizes = tuple(to_2tuple(size) for size in patch_sizes)
        if not train_patch_sizes:
            raise ValueError("NaFlex patch sizes must contain at least one value.")
        if not all(size[0] > 0 and size[1] > 0 for size in train_patch_sizes):
            raise ValueError("NaFlex patch sizes must be positive.")

        seq_lens = (128, 256, 576, 784, 1024) if seq_lens is None else seq_lens
        train_seq_lens = tuple(int(seq_len) for seq_len in seq_lens)
        if not train_seq_lens:
            raise ValueError("NaFlex sequence lengths must contain at least one value.")
        if not all(seq_len > 0 for seq_len in train_seq_lens):
            raise ValueError("NaFlex sequence lengths must be positive.")

        # Weights stay aligned to ``train_seq_lens`` (user order here); the scheduler pairs + sorts them with the
        # seq-lens, so the alignment survives its ``sorted(set(...))``. Unset -> uniform sampling (legacy).
        train_seq_len_probs = None
        if seq_len_probs is not None:
            if len(seq_len_probs) != len(train_seq_lens):
                raise ValueError("NaFlex seq-len probabilities must match seq-lens length.")
            if not all(prob >= 0 for prob in seq_len_probs):
                raise ValueError("NaFlex seq-len probabilities must be non-negative.")
            prob_sum = float(sum(seq_len_probs))
            if prob_sum <= 0:
                raise ValueError("NaFlex seq-len probabilities must sum to a positive value.")
            train_seq_len_probs = tuple(float(prob) / prob_sum for prob in seq_len_probs)

        train_patch_size_probs = None
        if patch_size_probs is not None:
            if len(patch_size_probs) != len(train_patch_sizes):
                raise ValueError("NaFlex patch size probabilities must match patch sizes length.")
            if not all(prob >= 0 for prob in patch_size_probs):
                raise ValueError("NaFlex patch size probabilities must be non-negative.")
            prob_sum = float(sum(patch_size_probs))
            if prob_sum <= 0:
                raise ValueError("NaFlex patch size probabilities must sum to a positive value.")
            train_patch_size_probs = tuple(float(prob) / prob_sum for prob in patch_size_probs)

        train_num_image_tokens = (
            int(train_num_image_tokens) if train_num_image_tokens is not None else None
        )
        if train_num_image_tokens is not None and train_num_image_tokens <= 0:
            raise ValueError("NaFlex train image token count must be positive.")

        max_tokens_per_batch = int(max_tokens_per_batch)
        if max_tokens_per_batch <= 0:
            raise ValueError("NaFlex max image tokens per batch must be positive.")

        batch_divisor = int(batch_divisor)
        if batch_divisor <= 0:
            raise ValueError("NaFlex batch divisor must be positive.")

        eval_patch_size = to_2tuple(eval_patch_size) if eval_patch_size is not None else train_patch_sizes[0]
        if eval_patch_size[0] <= 0 or eval_patch_size[1] <= 0:
            raise ValueError("NaFlex eval patch size must be positive.")

        eval_seq_len = int(eval_seq_len) if eval_seq_len is not None else max(train_seq_lens)
        if eval_seq_len <= 0:
            raise ValueError("NaFlex eval sequence length must be positive.")

        return cls(
            train_patch_sizes=train_patch_sizes,
            train_patch_size_probs=train_patch_size_probs,
            train_seq_lens=train_seq_lens,
            train_seq_len_probs=train_seq_len_probs,
            train_num_image_tokens=train_num_image_tokens,
            max_tokens_per_batch=max_tokens_per_batch,
            batch_divisor=batch_divisor,
            eval_patch_size=eval_patch_size,
            eval_seq_len=eval_seq_len,
        )

    @property
    def variable_patch_size(self) -> bool:
        return len(self.train_patch_sizes) > 1

    @property
    def eval_config(self) -> Tuple[Tuple[int, int], int]:
        return self.eval_patch_size, self.eval_seq_len
