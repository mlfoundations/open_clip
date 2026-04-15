import ast
import json
import logging
import math
import os
import random
import sys
import warnings
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
from typing import Any, Iterator, List, Tuple, Dict, Optional, Union, Callable

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    from timm.data.naflex_transforms import Patchify
    from timm.data.naflex_dataset import NaFlexCollator, calculate_naflex_batch_size, _resolve_patch_cfg
    naflex_available = True
except ImportError:
    naflex_available = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if "fname" not in filesample or "data" not in filesample:
            continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])

class NaFlexBatching(wds.PipelineStage):
    """
    
    WebDataset PipelineStage that implements the NaFlex batching strategy for image-text datasets.
    Adapted directly from timm's NaFlexDatasetWrapper 
    (https://github.com/huggingface/pytorch-image-models/blob/c769a585e0915c58fd7bb92f2355377aab7fc0e0/timm/data/naflex_dataset.py#L157).

    Yields batches with variable sequence lengths. It calculates a canonical
    batch schedule (sequence length, batch size pairs) once based on the
    total dataset size (padded for distribution). Each epoch, it shuffles
    the order of this canonical schedule and the dataset indices.
    This ensures a consistent number of batches and samples per epoch
    across all ranks. Handles distributed training and multiple workers.

    Supports both specification the total number of samples (`train_num_samples`) or total number of vision
    tokens (`train_num_tokens`) seen per epoch, and calculates a canonical batch schedule accordingly.

    """

    def __init__(
            self,
            train_num_samples: int = None,
            train_num_tokens: int = None,
            patch_size: Optional[Union[int, Tuple[int, int]]] = None,
            patch_size_choices: Optional[List[int]] = None,
            patch_size_choice_probs: Optional[List[float]] = None,
            seq_lens: Tuple[int, ...] = (128, 256, 576, 784, 1024),
            max_tokens_per_batch: int = 4096 * 4,
            transform_factory: Optional[Callable] = None,
            seed: int = 42,
            shuffle: bool = True,
            distributed: bool = False,
            rank: int = 0,
            world_size: int = 1,
            epoch: int = -1,
            batch_divisor: int = 8,
    ):
        """
        Args:
            train_num_samples: Total number of samples in the training dataset. Adjust the canonical batch schedule based on this.
            train_num_tokens: Total number of vision tokens in the training dataset. Adjust the canonical batch schedule based on this. Either `train_num_tokens` or `train_num_samples` must be specified, not both.
            patch_size: Single patch size to use.
            patch_size_choices: List of patch sizes to randomly select from.
            patch_size_choice_probs: Probabilities for each patch size.
            seq_lens: Sequence lengths to use for batching.
            max_tokens_per_batch: Target tokens per batch.
            transform_factory: Factory function for creating transforms.
            seed: Random seed.
            shuffle: Whether to shuffle data.
            distributed: Whether using distributed training.
            rank: Process rank for distributed training.
            world_size: Total number of processes.
            epoch: Starting epoch.
            batch_divisor: Ensure batch size is divisible by this.
        """
        super().__init__()
        self.seq_lens = sorted(list(set(seq_lens))) # Ensure unique and sorted
        self.max_tokens_per_batch = max_tokens_per_batch
        self.seed = seed
        self.shuffle = shuffle
        self.distributed = distributed
        self.rank = rank if distributed else 0
        self.world_size = world_size if distributed else 1
        self.epoch = epoch
        self.batch_divisor = batch_divisor

        # Resolve patch size configuration
        self.patch_sizes, self.patch_size_probs, self.variable_patch_size = _resolve_patch_cfg(
            patch_size,
            patch_size_choices,
            patch_size_choice_probs
        )
        # Pre-initialize transforms and collate fns for each (seq_len, patch_idx) combination
        self.transforms: Dict[Tuple[int, int], Optional[Callable]] = {}
        self.collate_fns: Dict[int, Callable] = {}
        self.patchifiers: List[Callable] = []

        for seq_len in self.seq_lens:
            self.collate_fns[seq_len] = NaFlexCollator(seq_len)

        for patch_idx, patch_size_tuple in enumerate(self.patch_sizes):
            # Pre-initialize patchifiers for each patch size (indexed by patch_idx)
            self.patchifiers.append(Patchify(
                patch_size=patch_size_tuple,
                flatten_patches=not self.variable_patch_size
            ))

            # Create transforms for each (seq_len, patch_idx) combination
            for seq_len in self.seq_lens:
                key = (seq_len, patch_idx)
                if transform_factory:
                    self.transforms[key] = transform_factory(max_seq_len=seq_len, patch_size=patch_size_tuple)
                else:
                    self.transforms[key] = None # No transform

        # Canonical Schedule Calculation (Done Once)
        self._canonical_batch_schedule: List[Tuple[int, int]] = []
        self._num_batches_per_rank: int = 0
        self._num_samples_per_rank: int = 0

        if train_num_samples is not None and train_num_tokens is None:
            self._create_canonical_schedule_from_num_samples(train_num_samples)
        elif train_num_tokens is not None and train_num_samples is None:
            self._create_canonical_schedule_from_num_tokens(train_num_tokens)
        else:
            raise ValueError("Must specify either `train_num_samples` or `train_num_tokens` for NaFlexBatching to create a canonical schedule.")
        
    def _create_canonical_schedule_from_num_tokens(self, num_tokens: int):
        """
        Alternative method to calculate canonical schedule based on total tokens instead of samples.
        This can be used if we want to target a specific number of tokens per epoch rather than samples.
        """
        current_schedule: List[Tuple[int, int]] = []

        if self.distributed and self.world_size > 1:
            # Calculate padding needed for even distribution
            if num_tokens % self.world_size != 0:
                 pad_size = self.world_size - (num_tokens % self.world_size)
                 padded_num_tokens = num_tokens + pad_size
            else:
                 pad_size = 0
                 padded_num_tokens = num_tokens
            if padded_num_tokens % self.world_size != 0:
                 # This should not happen with the padding logic, but safeguard
                 raise RuntimeError(f"Internal Error: Padded total length {padded_num_tokens} not divisible by world size {self.world_size}")
            num_tokens_per_rank = padded_num_tokens // self.world_size
        else:
             # Distributed flag set but world_size is 1, treat as non-distributed
             num_tokens_per_rank = num_tokens
        
        remaining_tokens = num_tokens_per_rank
        g = torch.Generator()
        g.manual_seed(self.seed) # Use base seed for deterministic schedule structure
        while remaining_tokens > 0:
            # Sample sequence length deterministically based on base seed
            seq_idx = torch.randint(0, len(self.seq_lens), (1,), generator=g).item()
            seq_len = self.seq_lens[seq_idx]

            # Calculate batch size
            batch_size = calculate_naflex_batch_size(
                tokens_per_batch=min(self.max_tokens_per_batch, remaining_tokens), # Don't exceed remaining tokens
                seq_len=seq_len,
                divisor=self.batch_divisor,
                rounding='floor',
            )
            batch_size = int(batch_size)
            if batch_size == 0:
                break

            current_schedule.append((seq_len, batch_size))
            remaining_tokens -= (batch_size * seq_len) # Account for all ranks

        self._canonical_batch_schedule = current_schedule
        self._num_batches_per_rank = len(current_schedule)
        self._num_samples_per_rank = sum(batch_size for _, batch_size in current_schedule)

    def _create_canonical_schedule_from_num_samples(self, num_samples):
        """
        Calculates the canonical batch schedule (seq_len, batch_size pairs)
        based on the dataset size, padded for distributed training.
        This schedule is the *same* for all ranks and ensures consistent
        epoch length. It is calculated once during initialization.
        """
        total_len = num_samples
        padded_total_len = total_len
        num_samples_per_rank = total_len

        if self.distributed and self.world_size > 1:
            # Calculate padding needed for even distribution
            if total_len % self.world_size != 0:
                 pad_size = self.world_size - (total_len % self.world_size)
                 padded_total_len += pad_size
            else:
                 pad_size = 0

            if padded_total_len % self.world_size != 0:
                 # This should not happen with the padding logic, but safeguard
                 raise RuntimeError(f"Internal Error: Padded total length {padded_total_len} not divisible by world size {self.world_size}")

            num_samples_per_rank = padded_total_len // self.world_size
        elif self.distributed and self.world_size <= 1:
             # Distributed flag set but world_size is 1, treat as non-distributed
             pass # num_samples_per_rank remains total_len

        self._num_samples_per_rank = num_samples_per_rank

        if num_samples_per_rank == 0:
             self._canonical_batch_schedule = []
             self._num_batches_per_rank = 0
             return

        # Use a fixed seed for generating the canonical schedule structure
        g = torch.Generator()
        g.manual_seed(self.seed) # Use base seed, NOT epoch seed

        current_schedule: List[Tuple[int, int]] = []
        remaining_samples = num_samples_per_rank
        total_scheduled_samples = 0

        while remaining_samples > 0:
            # Sample sequence length deterministically based on base seed
            seq_idx = torch.randint(0, len(self.seq_lens), (1,), generator=g).item()
            seq_len = self.seq_lens[seq_idx]

            # Calculate batch size
            batch_size = calculate_naflex_batch_size(
                tokens_per_batch=self.max_tokens_per_batch,
                seq_len=seq_len,
                # max_size should be remaining_samples to avoid overshooting
                max_size=remaining_samples,
                divisor=self.batch_divisor,
                rounding='floor',
            )
            # Ensure batch size is positive and doesn't exceed remaining samples
            batch_size = max(1, batch_size)
            batch_size = min(batch_size, remaining_samples)

            if batch_size <= 0:
                 warnings.warn(f"Calculated batch size <= 0 (seq_len={seq_len}, remaining={remaining_samples}). Stopping schedule generation early.")
                 break # Avoid infinite loop if something goes wrong

            current_schedule.append((seq_len, batch_size))
            remaining_samples -= batch_size
            total_scheduled_samples += batch_size

        # Sanity check: Ensure the schedule covers all samples for the rank
        if total_scheduled_samples != num_samples_per_rank:
            warnings.warn(
                f"Rank {self.rank}: Canonical schedule accounts for {total_scheduled_samples} samples, "
                f"but expected {num_samples_per_rank} samples per rank. "
                f"This might happen if min_batch_size or batch_divisor constraints prevent utilizing all samples. "
                f"Check parameters. Remaining samples: {remaining_samples}"
            )
            # Adjust if needed? Could add a final small batch, but might violate constraints.
            # Current behavior: some samples might be dropped if schedule logic fails.

        self._canonical_batch_schedule = current_schedule
        self._num_batches_per_rank = len(current_schedule)
        self._num_samples_per_rank = total_scheduled_samples

    @property
    def num_samples(self) -> int:
        """Returns the total number of samples on training."""
        return self._num_samples_per_rank * self.world_size if self.distributed else self._num_samples_per_rank

    def __len__(self) -> int:
        """Returns the number of batches per worker for the current epoch.

        Returns:
            Number of batches this worker will process.
        """
        return self._num_batches_per_rank

    def run(self, src):
        """Iterates through pre-calculated batches for the current epoch.

        Yields:
            Tuple of (input_dict, targets) for each batch.
        """
        g = torch.Generator()
        if isinstance(self.epoch, SharedEpoch):
            # if Epoch value is shared across workers, use that for deterministic shuffling across workers. Otherwise, increment local epoch counter for shuffling.
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        g.manual_seed(self.seed + epoch) # Ensure same shuffling for all workers

        schedule = list(self._canonical_batch_schedule)
        # shuffle the schedule for this epoch
        if self.shuffle:
            random.Random(self.seed + epoch).shuffle(schedule) # Use Python random for shuffling schedule to avoid affecting torch generator state used for patch size sampling
        
        # divide up schedule based on dataloader worker id
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Split the schedule into roughly equal parts for each worker
            batches_per_worker = (self._num_batches_per_rank + num_workers - 1) // num_workers
            start_idx = worker_id * batches_per_worker
            end_idx = min(start_idx + batches_per_worker, self._num_batches_per_rank)
            schedule = schedule[start_idx:end_idx]
        else:
            print(f"Warning: Could not get worker info in NaFlexBatching. Processing full schedule in single worker mode.")
        
        samples = iter(src)
        for _, (seq_len, batch_size) in enumerate(schedule):
            batch_imgs = []
            batch_targets = []
            for _ in range(batch_size):
                try:
                    sample = next(samples)
                except StopIteration:
                    # This can happen if the underlying dataset is smaller than expected
                    # due to padding issues or if schedule generation had problems.
                    # In this case, we stop yielding batches for this epoch.
                    warnings.warn(f"Rank {self.rank}: Reached end of dataset samples while processing batch (seq_len={seq_len}, batch_size={batch_size}). Stopping iteration.")
                    return
                # Select patch size for this batch
                patch_idx = 0
                if self.variable_patch_size:
                    # Use torch multinomial for weighted random choice
                    patch_idx = torch.multinomial(torch.tensor(self.patch_size_probs), 1, generator=g).item()

                # Get the pre-initialized transform and patchifier using patch_idx
                transform_key = (seq_len, patch_idx)
                transform = self.transforms.get(transform_key)
                batch_patchifier = self.patchifiers[patch_idx]
                
                img, label = sample

                # Apply transform if available
                processed_img = transform(img) if transform else img
                batch_imgs.append(processed_img)
                batch_targets.append(label)
                
            batch_imgs = [batch_patchifier(img) for img in batch_imgs]
            batch_samples = list(zip(batch_imgs, batch_targets))
            if batch_samples: # Only yield if we successfully processed samples
                # Collate the processed samples into a batch
                yield self.collate_fns[seq_len](batch_samples)


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    if args.use_naflex:
        assert naflex_available, "NaFlexBatching requires timm version with NaFlex support. Please install timm>=1.0.16 to use this feature."

    num_shards = None
    if is_train:
        if args.naflex_num_train_image_tokens is not None and args.train_num_samples is None:
            num_image_tokens = args.naflex_num_train_image_tokens
            num_samples = None
        elif args.train_num_samples is not None:
            num_samples = args.train_num_samples
            num_image_tokens = None
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            num_image_tokens = None
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt")
    ])
    if args.use_naflex:
        # for naflex we need to keep the original image for patching, and we will apply the preprocess_img transform inside the naflex batching 
        # see below, where we pass the transform factory to the NaFlexBatching instance)
        pipeline.append(wds.map_dict(image=lambda img: img, text=lambda text: tokenizer(text)[0]))
    else:
        # for non-naflex we can just apply the preprocess_img transform here as usualß
        pipeline.append(wds.map_dict(image=lambda img: preprocess_img(img), text=lambda text: tokenizer(text)[0]))
    pipeline.append(wds.to_tuple("image", "text"))

    if args.use_naflex:
        # When we use NaFlex, batching is handled by NaFlexBatching which:
        # randomly selects a patch size and sequence length for each batch, applies the appropriate image transformations, and dynamically 
        # adjusts the batch size to fit within the specified max tokens per batch.
        if is_train:
            naflex_batching = NaFlexBatching(
                epoch=shared_epoch,
                train_num_samples=num_samples,
                train_num_tokens=num_image_tokens,
                rank=args.rank, 
                world_size=args.world_size, 
                distributed=args.distributed,
                max_tokens_per_batch=args.naflex_max_image_tokens_per_batch, 
                patch_size_choices=args.naflex_patch_sizes, 
                seq_lens=args.naflex_seq_lens, 
                transform_factory=preprocess_img
            )
            pipeline.append(naflex_batching)
            dataset = wds.DataPipeline(*pipeline)
            num_batches = len(naflex_batching)
            num_samples = naflex_batching.num_samples
        else:
            raise NotImplementedError("NaFlex batching is currently only implemented for training.")
    else:
        pipeline.append(wds.batched(args.batch_size, partial=not is_train))
        dataset = wds.DataPipeline(*pipeline)
        if is_train:
            if not resampled:
                num_shards = num_shards or len(expand_urls(input_shards)[0])
                assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
            # roll over and repeat a few samples to get same number of full batches on each node
            round_fn = math.floor if floor else math.ceil
            global_batch_size = args.batch_size * args.world_size
            num_batches = round_fn(num_samples / global_batch_size)
            num_workers = max(1, args.workers)
            num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
            num_batches = num_worker_batches * num_workers
            num_samples = num_batches * global_batch_size
            dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
        else:
            # last batches are partial, eval is done on single (master) node
            num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
