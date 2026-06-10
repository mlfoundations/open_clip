import ast
import io
import json
import logging
import math

_logger = logging.getLogger(__name__)
import os
import random
import sys
import warnings
from typing import Optional

import braceexpand
from dataclasses import dataclass
from functools import partial
from multiprocessing import Value, get_context

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample


class TokenizeText:
    # Module-level callable replaces inline lambdas in webdataset pipelines so
    # they survive pickling — required under forkserver multiprocessing
    # (Python 3.14+ default on POSIX).
    def __init__(self, tokenizer, variable: bool = False):
        self.tokenizer = tokenizer
        self.variable = variable

    def __call__(self, text):
        # Bucketed pipelines tokenize before `wds.decode` runs (the bucket pool holds raw, undecoded samples),
        # so the caption may arrive as raw utf-8 bytes rather than a decoded str.
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        # `variable=True` returns a per-sample 1-D tensor (no fixed-length padding); collators pad captions
        # to the per-batch max for text towers that support variable length.
        if self.variable:
            return self.tokenizer(text, pad=False)[0]
        return self.tokenizer(text)[0]

from open_clip_train.naflex_data import (
    CaptionLength,
    LengthBucketer,
    NaFlexBatcher,
    NaFlexMapDatasetWrapper,
    collate_naflex_dicts,
    collate_naflex_tuples,
    collate_variable_text,
    create_naflex_eval_transform,
    require_naflex,
)


def get_text_pad_id(tokenizer) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise ValueError("variable_text=True requires a tokenizer with a reserved `pad_token_id`.")
    return int(pad_id)


def collate_variable_text_dicts(
        batch,
        *,
        pad_id: int,
        target_key: str = "text",
        text_pad_multiple: Optional[int] = None,
        text_pad_cap: Optional[int] = None,
):
    text, text_valid = collate_variable_text(
        [sample[target_key] for sample in batch], pad_id,
        pad_multiple=text_pad_multiple, pad_cap=text_pad_cap,
    )
    others = [{k: v for k, v in sample.items() if k != target_key} for sample in batch]
    out = default_collate(others) if others and others[0] else {}
    out[target_key] = text
    out[f"{target_key}_valid"] = text_valid
    return out


class CsvDataset(Dataset):
    def __init__(
            self,
            input_filename,
            transforms,
            img_key,
            caption_key,
            sep="\t",
            tokenizer=None,
            variable_text: bool = False,
    ):
        _logger.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        # Keep as pandas Series rather than converting to Python lists.
        # Python lists of strings cause copy-on-write memory duplication in
        # forked DataLoader workers because the cyclic GC walks every element
        # and modifies reference counts.  Pandas Series are backed by
        # numpy/pyarrow buffers that the GC does not traverse.
        self.images = df[img_key]
        self.captions = df[caption_key]
        self.transforms = transforms
        _logger.debug('Done loading data.')

        self.tokenize = tokenizer
        self.variable_text = variable_text

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(str(self.images.iloc[idx]))
        if self.transforms is not None:
            image = self.transforms(image)
        caption = str(self.captions.iloc[idx])
        if self.variable_text:
            text = self.tokenize(caption, pad=False)[0]
        else:
            text = self.tokenize([caption])[0]
        return {"image": image, "text": text}


class SharedEpoch:
    def __init__(self, epoch: int = 0, mp_context: Optional[str] = None):
        # Create the shared counter in the SAME multiprocessing context as the DataLoader workers. A fork-context
        # SemLock can't be shipped to forkserver/spawn workers ("A SemLock created in a fork context is being
        # shared with a process in a spawn context"), which bites the audio loader (forkserver) once this rides
        # inside ResampledShards2 and is pickled to a worker. mp_context=None -> default context (fork on Linux).
        self.shared_epoch = get_context(mp_context).Value('i', epoch)

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


def get_imagenet(args, preprocess_fns, split, naflex_data_config=None):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    use_naflex_eval = naflex_data_config is not None and not is_train
    collate_fn = None

    if is_train and naflex_data_config is not None:
        raise ValueError("NaFlex is only wired for validation and zero-shot ImageNet loaders, not --imagenet-train.")

    if use_naflex_eval:
        preprocess_val, naflex_max_seq_len, _ = create_naflex_eval_transform(preprocess_val, naflex_data_config)
        collate_fn = partial(collate_naflex_tuples, max_seq_len=naflex_max_seq_len)

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
        collate_fn=collate_fn,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for batch in dataloader:
        n_batches += 1
        image = batch["image"]
        image_batch_size = image["patches"].shape[0] if isinstance(image, dict) else len(image)
        n_elements += image_batch_size
        assert image_batch_size == len(batch["text"])
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def _has_image(sample):
    return ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)


class FilterValidSample:
    """WebDataset filter keeping samples that have an image and a caption source.

    Module-level + picklable (forkserver-safe). Two mutually-exclusive caption sources:
      - ``json_text_key`` set  -> require a ``.json`` member (caption read from a field of it).
      - otherwise              -> require one of the ``text_key`` member suffixes (``;``-separated alternatives).
    """

    def __init__(self, text_key: str = "txt", json_text_key: Optional[str] = None):
        self.json_text_key = json_text_key
        self.text_keys = None if json_text_key else tuple(text_key.split(";"))

    def __call__(self, sample):
        if not _has_image(sample):
            return False
        if self.json_text_key is not None:
            return 'json' in sample
        return any(key in sample for key in self.text_keys)


class JsonCaptionExtractor:
    """Set ``sample['text']`` from a field of the sample's JSON metadata (datasets without a ``.txt`` member).

    Robust to the JSON being either a parsed dict or raw bytes/str (depending on the decoder), and drops the
    raw metadata afterwards. Module-level + picklable, mirroring ``TokenizeText``.
    """

    def __init__(self, caption_key: str, json_key: str = "json"):
        self.caption_key = caption_key
        self.json_key = json_key

    def __call__(self, sample):
        meta = sample.get(self.json_key)
        if isinstance(meta, (bytes, bytearray, str)):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                meta = {}
        caption = meta.get(self.caption_key) if isinstance(meta, dict) else None
        sample["text"] = caption if isinstance(caption, str) else ""
        sample.pop(self.json_key, None)
        return sample


def decode_pil_rgb(data):
    """Decode raw image bytes to an RGB PIL image (what ``wds.decode('pilrgb')`` does, as a per-key map).

    Bucketed pipelines reorder samples *before* decoding so the bucket pool holds raw bytes instead of decoded
    images (10-50x smaller); this runs after the reorder. PIL sniffs the format from the byte signature, so the
    extension-keyed dispatch of ``wds.decode`` is not needed.
    """
    with io.BytesIO(data) as stream:
        img = Image.open(stream)
        img.load()
        return img.convert("RGB")


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    _logger.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
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


def wds_shuffle_sizes():
    """WebDataset shuffle buffer sizes ``(shard_size, shard_initial, sample_size, sample_initial)``.

    Read at pipeline-build time so the ``OPENCLIP_WDS_*`` env vars can override the module defaults — e.g. set
    tiny values when iterating a small test shard so the first batch doesn't wait on a multi-thousand-sample
    buffer fill. Unset env vars preserve prior behavior.
    """
    env = os.environ.get
    return (
        int(env("OPENCLIP_WDS_SHARD_SHUFFLE_SIZE", _SHARD_SHUFFLE_SIZE)),
        int(env("OPENCLIP_WDS_SHARD_SHUFFLE_INITIAL", _SHARD_SHUFFLE_INITIAL)),
        int(env("OPENCLIP_WDS_SAMPLE_SHUFFLE_SIZE", _SAMPLE_SHUFFLE_SIZE)),
        int(env("OPENCLIP_WDS_SAMPLE_SHUFFLE_INITIAL", _SAMPLE_SHUFFLE_INITIAL)),
    )


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


class RepeatedShardList(IterableDataset):
    """An iterable dataset that repeats a finite shard list."""

    def __init__(self, urls):
        super().__init__()
        self.urls, _ = expand_urls(urls)
        assert isinstance(self.urls[0], str)

    def __iter__(self):
        while True:
            for url in self.urls:
                yield dict(url=url)


def append_naflex_train_stages(
        pipeline,
        *,
        naflex_data_config,
        transform_factory,
        tokenize_text,
        modality_key,
        num_samples,
        num_tokens,
        args,
        shared_epoch,
        pad_id,
        per_row_text_tokens,
        bucketer=None,
        decode_stage=None,
        pad_multiple=None,
        text_pad_multiple=None,
        text_pad_cap=None,
):
    """Append the modality-agnostic NaFlex train stages to ``pipeline`` and return the ``NaFlexBatcher``.

    Shared by the image (``get_wds_dataset``) and audio (``get_wds_audio_dataset``) pipelines: tokenize text ->
    optional length bucketing -> ``decode_stage`` -> ``NaFlexBatcher``.
    ``decode_stage`` is the caller's modality-specific decode map (image bytes -> PIL / audio bytes ->
    ``(waveform, sr)``); it runs after the bucketer so the bucket pool holds raw, undecoded samples.
    The batcher reads ``sample[modality_key]`` (``"image"`` or ``"audio"``) plus ``sample['text']`` and applies
    ``transform_factory`` to produce the ``{patches, patch_coord, patch_valid}`` rows -- so audio reuses the
    whole batching/scheduling/collation path unchanged via ``modality_key="audio"``.
    """
    patch_size = None
    patch_size_choices = naflex_data_config.train_patch_sizes
    if not naflex_data_config.variable_patch_size:
        patch_size = patch_size_choices[0]
        patch_size_choices = None

    stages = [wds.map_dict(text=tokenize_text)]
    if bucketer is not None:
        # Reorder samples so similar lengths batch together (text for image, audio_tokens for audio),
        # tightening per-batch-max padding. Reorder-only -> schedule / num_batches / DDP unchanged. The caller
        # owns the bucketer choice + policy (a LengthBucketer with the right length_fns per the model type).
        stages.append(bucketer)
    if decode_stage is not None:
        stages.append(decode_stage)
    stages.append(NaFlexBatcher(
        train_num_samples=num_samples,
        train_num_tokens=num_tokens,
        patch_size=patch_size,
        patch_size_choices=patch_size_choices,
        patch_size_choice_probs=naflex_data_config.train_patch_size_probs,
        seq_lens=naflex_data_config.train_seq_lens,
        max_tokens_per_batch=naflex_data_config.max_tokens_per_batch,
        transform_factory=transform_factory,
        seed=args.seed,
        shuffle=True,
        distributed=args.distributed,
        rank=args.rank,
        world_size=args.world_size,
        epoch=shared_epoch,
        batch_divisor=naflex_data_config.batch_divisor,
        image_key=modality_key,
        pad_id=pad_id,
        per_row_text_tokens=per_row_text_tokens,
        pad_multiple=pad_multiple,
        text_pad_multiple=text_pad_multiple,
        text_pad_cap=text_pad_cap,
    ))
    if pad_id is not None:
        _logger.info(
            f"NaFlex batch budget = {naflex_data_config.max_tokens_per_batch} modality tokens/row-batch "
            f"({modality_key} bucket"
            + (f" + text cap {per_row_text_tokens}" if per_row_text_tokens else "")
            + ")"
            + ("; length bucketing ON" if bucketer is not None else "")
        )
    pipeline.extend(stages)
    return pipeline[-1]


def naflex_loader_counts(batcher, args):
    """NaFlex epoch counts come from the batcher's deterministic schedule (no with_epoch / fixed-batch math)."""
    num_workers = max(1, args.workers)
    return batcher.num_batches_for_workers(num_workers), batcher.num_samples_for_workers(num_workers)


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, naflex_data_config=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    use_naflex_train = naflex_data_config is not None and is_train
    use_naflex_eval = naflex_data_config is not None and not is_train

    num_shards = None
    if is_train:
        num_image_tokens = naflex_data_config.train_num_image_tokens if use_naflex_train else None
        if use_naflex_train and num_image_tokens is not None and args.train_num_samples is not None:
            raise ValueError("Specify only one of `--train-num-samples` or `--naflex-num-train-image-tokens`.")
        if use_naflex_train and num_image_tokens is not None:
            num_samples = None
        elif args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, (
            "--train_data_upsampling_factors is only supported when sampling with replacement "
            "(with --dataset-resampled)."
        )

    if use_naflex_train:
        require_naflex()
        if not getattr(preprocess_img, 'is_naflex_transform_factory', False):
            raise ValueError("NaFlex WebDataset training requires `--aug-cfg use_timm=True naflex=True`.")
    elif use_naflex_eval:
        preprocess_img, naflex_max_seq_len, _ = create_naflex_eval_transform(preprocess_img, naflex_data_config)

    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    elif use_naflex_train:
        num_shards = num_shards or len(expand_urls(input_shards)[0])
        assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        pipeline = [RepeatedShardList(input_shards)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        shard_shuffle_size, shard_shuffle_initial, sample_shuffle_size, sample_shuffle_initial = wds_shuffle_sizes()
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=shard_shuffle_size,
                    initial=shard_shuffle_initial,
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
                bufsize=sample_shuffle_size,
                initial=sample_shuffle_initial,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    # GenLIP budgets variable captions with image tokens. Other variable-text towers only need batch-time padding.
    genlip_text = getattr(args, 'genlip', False)
    variable_text = bool(getattr(args, 'variable_text', False))
    text_variable = genlip_text or variable_text
    text_pad_id = get_text_pad_id(tokenizer) if text_variable else None
    text_pad_multiple = getattr(args, 'text_pad_multiple', None)
    text_pad_cap = getattr(tokenizer, 'context_length', None)
    naflex_pad_id = text_pad_id if text_variable else None
    naflex_text_cost = (getattr(tokenizer, 'context_length', 0) or 0) if genlip_text else 0
    tokenize_text = TokenizeText(tokenizer, variable=text_variable)

    # Length bucketing reorders by caption length (train-only; only meaningful for variable text).
    use_bucketing = is_train and text_variable and getattr(args, 'length_bucketing', False)

    # Image decode runs *after* tokenize and the optional length bucketer (see decode_pil_rgb below): the
    # bucketer pools `--bucket-pool` complete samples per worker, so it must see raw, undecoded samples (the
    # same regime as the raw-sample shuffle above) -- a pool of decoded full-resolution images is 10-50x
    # larger and can OOM dataloader workers on hi-res data. One ordering for all branches, bucketed or not;
    # the decode-first assembly lives in legacy_data.py.
    text_key = getattr(args, 'text_key', 'txt') or 'txt'
    json_text_key = getattr(args, 'json_text_key', None)
    if json_text_key:
        # Read the caption from a field of each sample's .json (datasets without a text member, e.g. monet).
        pipeline.extend([
            wds.select(FilterValidSample(json_text_key=json_text_key)),
            wds.rename(image="jpg;png;jpeg;webp", json="json", keep=False),
            wds.map(JsonCaptionExtractor(json_text_key), handler=log_and_continue),  # parses raw bytes itself
        ])
    else:
        # Read the caption from a tar member (default 'txt'; --text-key allows alternatives like 'txt;caption').
        # Plain-text members only (TokenizeText utf-8 decodes the raw bytes); json captions use --json-text-key.
        pipeline.extend([
            wds.select(FilterValidSample(text_key=text_key)),
            wds.rename(image="jpg;png;jpeg;webp", text=text_key, keep=False),
        ])
    decode_image = wds.map_dict(image=decode_pil_rgb, handler=log_and_continue)

    if use_naflex_train:
        # Image NaFlex resizes images to ~fill the bucket, so caption length is the only optional bucketing signal.
        image_bucketer = None
        if use_bucketing:
            image_bucketer = LengthBucketer(
                length_fns=[CaptionLength(key="text")],
                pool=args.bucket_pool,
                chunk=args.bucket_chunk,
                seed=args.seed,
                epoch=shared_epoch,
            )
        naflex_batcher = append_naflex_train_stages(
            pipeline,
            naflex_data_config=naflex_data_config,
            transform_factory=preprocess_img,
            tokenize_text=tokenize_text,
            modality_key="image",
            num_samples=num_samples,
            num_tokens=num_image_tokens,
            args=args,
            shared_epoch=shared_epoch,
            pad_id=naflex_pad_id,
            per_row_text_tokens=naflex_text_cost,
            bucketer=image_bucketer,
            decode_stage=decode_image,
            pad_multiple=getattr(args, 'naflex_pad_multiple', None),
            text_pad_multiple=text_pad_multiple,
            text_pad_cap=text_pad_cap,
        )
        dataset = wds.DataPipeline(*pipeline)
        num_batches, num_samples = naflex_loader_counts(naflex_batcher, args)
    elif use_naflex_eval:
        pipeline.extend([
            decode_image,
            wds.map_dict(image=preprocess_img, text=tokenize_text),
            wds.batched(
                args.batch_size,
                partial=True,
                collation_fn=partial(
                    collate_naflex_dicts,
                    max_seq_len=naflex_max_seq_len,
                    pad_id=naflex_pad_id,
                    text_pad_multiple=text_pad_multiple,
                    text_pad_cap=text_pad_cap,
                ),
            ),
        ])
        dataset = wds.DataPipeline(*pipeline)
    else:
        collate_fn = (
            partial(
                collate_variable_text_dicts,
                pad_id=text_pad_id,
                text_pad_multiple=text_pad_multiple,
                text_pad_cap=text_pad_cap,
            )
            if text_variable else default_collate
        )
        # Tokenize -> [bucket] -> decode -> transform -> batch. Bucketing reorders by caption length so
        # similar-length captions batch together: tighter per-batch-max text padding and fewer distinct
        # text shapes (stacks with --text-pad-multiple).
        stages = [wds.map_dict(text=tokenize_text)]
        if use_bucketing:
            stages.append(LengthBucketer(
                length_fns=[CaptionLength(key="text")],
                pool=args.bucket_pool,
                chunk=args.bucket_chunk,
                seed=args.seed,
                epoch=shared_epoch,
            ))
        stages.extend([
            decode_image,
            wds.map_dict(image=preprocess_img),
            wds.batched(args.batch_size, partial=not is_train, collation_fn=collate_fn),
        ])
        pipeline.extend(stages)
        dataset = wds.DataPipeline(*pipeline)

    if is_train and not use_naflex_train:
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
    elif not is_train:
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


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, naflex_data_config=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    shared_epoch = SharedEpoch(epoch=epoch)
    use_naflex_train = naflex_data_config is not None and is_train
    use_naflex_eval = naflex_data_config is not None and not is_train
    variable_text = bool(getattr(args, 'variable_text', False))
    genlip_text = getattr(args, 'genlip', False)
    text_variable = variable_text or genlip_text
    text_pad_id = get_text_pad_id(tokenizer) if text_variable else None
    text_pad_multiple = getattr(args, 'text_pad_multiple', None)
    text_pad_cap = getattr(tokenizer, 'context_length', None)
    collate_fn = default_collate

    if use_naflex_train:
        require_naflex()
        if not getattr(preprocess_fn, 'is_naflex_transform_factory', False):
            raise ValueError("NaFlex CSV training requires `--aug-cfg use_timm=True naflex=True`.")
        dataset_transform = None
    elif use_naflex_eval:
        dataset_transform, naflex_max_seq_len, _ = create_naflex_eval_transform(preprocess_fn, naflex_data_config)
        collate_fn = partial(
            collate_naflex_dicts, max_seq_len=naflex_max_seq_len, pad_id=text_pad_id,
            text_pad_multiple=text_pad_multiple, text_pad_cap=text_pad_cap,
        )
    else:
        dataset_transform = preprocess_fn
        if text_variable:
            collate_fn = partial(
                collate_variable_text_dicts, pad_id=text_pad_id,
                text_pad_multiple=text_pad_multiple, text_pad_cap=text_pad_cap,
            )

    dataset = CsvDataset(
        input_filename,
        dataset_transform,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        variable_text=text_variable,
    )

    if use_naflex_train:
        naflex_patch_size = None
        naflex_patch_size_choices = naflex_data_config.train_patch_sizes
        if not naflex_data_config.variable_patch_size:
            naflex_patch_size = naflex_patch_size_choices[0]
            naflex_patch_size_choices = None
        # GenLIP includes caption-token cap in the batch budget; other variable-text towers only need padding.
        naflex_pad_id = text_pad_id if (genlip_text or variable_text) else None
        naflex_text_cost = (getattr(tokenizer, 'context_length', 0) or 0) if genlip_text else 0
        dataset = NaFlexMapDatasetWrapper(
            dataset,
            train_num_tokens=naflex_data_config.train_num_image_tokens,
            patch_size=naflex_patch_size,
            patch_size_choices=naflex_patch_size_choices,
            patch_size_choice_probs=naflex_data_config.train_patch_size_probs,
            seq_lens=naflex_data_config.train_seq_lens,
            max_tokens_per_batch=naflex_data_config.max_tokens_per_batch,
            transform_factory=preprocess_fn,
            seed=args.seed,
            shuffle=True,
            distributed=args.distributed,
            rank=args.rank,
            world_size=args.world_size,
            epoch=shared_epoch,
            batch_divisor=naflex_data_config.batch_divisor,
            pad_id=naflex_pad_id,
            per_row_text_tokens=naflex_text_cost,
            text_pad_multiple=text_pad_multiple,
            text_pad_cap=text_pad_cap,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        num_workers = max(1, args.workers)
        dataloader.num_samples = dataset.num_samples_for_workers(num_workers)
        dataloader.num_batches = dataset.num_batches_for_workers(num_workers)
        return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

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
        collate_fn=collate_fn,
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
            variable_text: bool = False,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = TokenizeText(tokenizer, variable=variable_text)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return {"image": image, "text": self.preprocess_txt(self.caption)}


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, naflex_data_config=None):
    image_size = preprocess_fn.transforms[0].size
    variable_text = bool(getattr(args, 'variable_text', False))
    collate_fn = (
        partial(
            collate_variable_text_dicts, pad_id=get_text_pad_id(tokenizer),
            text_pad_multiple=getattr(args, 'text_pad_multiple', None),
            text_pad_cap=getattr(tokenizer, 'context_length', None),
        )
        if variable_text else default_collate
    )
    dataset = SyntheticDataset(
        transform=preprocess_fn,
        image_size=image_size,
        dataset_size=args.train_num_samples,
        tokenizer=tokenizer,
        variable_text=variable_text,
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
        collate_fn=collate_fn,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "webdataset-audio":
        from open_clip_train.audio_data import get_wds_audio_dataset
        return get_wds_audio_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "synthetic-audio":
        from open_clip_train.audio_data import get_synthetic_audio_dataset
        return get_synthetic_audio_dataset
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
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None, naflex_data_config=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if getattr(args, 'use_naflex', False) and naflex_data_config is None:
        raise ValueError("NaFlex data loaders require a NaFlex data config.")
    if not getattr(args, 'use_naflex', False) and getattr(args, 'naflex_num_train_image_tokens', None) is not None:
        warnings.warn("`--naflex-num-train-image-tokens` is ignored unless `--use-naflex` is set.")

    if args.train_data or args.dataset_type in ("synthetic", "synthetic-audio"):
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args,
            preprocess_train,
            is_train=True,
            epoch=epoch,
            tokenizer=tokenizer,
            naflex_data_config=naflex_data_config,
        )

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args,
            preprocess_val,
            is_train=False,
            tokenizer=tokenizer,
            naflex_data_config=naflex_data_config,
        )

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(
            args,
            preprocess_fns,
            "val",
            naflex_data_config=naflex_data_config,
        )

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(
            args,
            preprocess_fns,
            "v2",
            naflex_data_config=naflex_data_config,
        )

    return data
