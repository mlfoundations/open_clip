import ast
import json
import logging
import math
import os
import pickle
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

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


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
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
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

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


def _collate_attr_jsonl(batch):
    """Collate batch of (image, text, img_attr, txt_attr, txt_mask[, caption]) into stacked tensors + optional captions."""
    images = torch.stack([b[0] for b in batch])
    texts = torch.stack([b[1] for b in batch])
    img_attr = torch.stack([b[2] for b in batch])
    txt_attr = torch.stack([b[3] for b in batch])
    txt_mask = torch.stack([b[4] for b in batch])
    if len(batch[0]) > 5:
        captions = [b[5] for b in batch]
        return images, texts, img_attr, txt_attr, txt_mask, captions
    return images, texts, img_attr, txt_attr, txt_mask


def _build_jsonl_offset_index(captions_path, index_path):
    """Stream the JSONL once and build a list of byte offsets (one per non-empty line)."""
    offsets = []
    with open(captions_path, 'rb') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    with open(index_path, 'wb') as f:
        pickle.dump(offsets, f, protocol=pickle.HIGHEST_PROTOCOL)
    return offsets


# When image index space is dense (max_index not huge, most slots used), use a dense numpy
# array of offsets so it can be memory-mapped and shared across DataLoader workers.
_IMAGE_ATTR_DENSE_MAX_INDEX = 50_000_000  # use dense array only if max_index <= this
_IMAGE_ATTR_DENSE_FILL_RATIO = 0.5  # and at least this fraction of [0, max_index] is present


def _build_image_attr_index(image_attr_path):
    """Stream image_attr JSONL and build index: image_index -> byte offset.
    Also builds image_path -> byte offset when rows have image_path (so img_attr matches caption's image).
    Uses a dense numpy array (memmap-friendly) when indices are dense, else a pickle dict."""
    base = image_attr_path
    index_pickle_path = base + '.attr_index.pkl'
    path_index_path = base + '.attr_path_index.pkl'
    offsets_npy_path = base + '.attr_offsets.npy'
    meta_path = base + '.attr_offsets_meta.json'

    if os.path.exists(offsets_npy_path) and os.path.exists(meta_path):
        pass  # index already built; may still need path index
    elif os.path.exists(index_pickle_path):
        pass
    else:
        pairs = []
        path_pairs = []
        max_index = -1
        with open(image_attr_path, 'rb') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line.decode('utf-8'))
                idx = int(row['index'])
                pairs.append((idx, pos))
                max_index = max(max_index, idx)
                path_str = (row.get('image_path') or '').strip()
                if path_str:
                    path_pairs.append((os.path.normpath(path_str), pos))

        if not pairs:
            with open(index_pickle_path, 'wb') as f:
                pickle.dump({}, f, protocol=pickle.HIGHEST_PROTOCOL)
            return None

        n = len(pairs)
        use_dense = (
            max_index <= _IMAGE_ATTR_DENSE_MAX_INDEX
            and (n / (max_index + 1)) >= _IMAGE_ATTR_DENSE_FILL_RATIO
        )
        if use_dense:
            offsets_arr = np.full(max_index + 1, -1, dtype=np.int64)
            for idx, pos in pairs:
                offsets_arr[idx] = pos
            np.save(offsets_npy_path, offsets_arr)
            with open(meta_path, 'w') as f:
                json.dump({'max_index': max_index, 'n': n}, f)
            logging.info(f'Built dense image-attr index: {n} entries, max_index={max_index}, saved to {offsets_npy_path}.')
        else:
            index_to_offset = {idx: pos for idx, pos in pairs}
            with open(index_pickle_path, 'wb') as f:
                pickle.dump(index_to_offset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f'Built sparse image-attr index: {n} entries, saved to {index_pickle_path}.')

        if path_pairs and not os.path.exists(path_index_path):
            path_to_offset = {p: pos for p, pos in path_pairs}
            with open(path_index_path, 'wb') as f:
                pickle.dump(path_to_offset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f'Built image-attr path index: {len(path_to_offset)} entries, saved to {path_index_path}.')

    # Ensure path index exists if image_attr file has image_path (build from current file)
    if not os.path.exists(path_index_path):
        path_pairs = []
        with open(image_attr_path, 'rb') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line.decode('utf-8'))
                path_str = (row.get('image_path') or '').strip()
                if path_str:
                    path_pairs.append((os.path.normpath(path_str), pos))
        if path_pairs:
            path_to_offset = {p: pos for p, pos in path_pairs}
            with open(path_index_path, 'wb') as f:
                pickle.dump(path_to_offset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f'Built image-attr path index: {len(path_to_offset)} entries, saved to {path_index_path}.')
    return None


def _load_image_attr_index(image_attr_path):
    """Load image_index -> byte offset. Returns (index_lookup, is_dense).
    index_lookup: for dense, np.ndarray (use with mmap); for sparse, dict."""
    base = image_attr_path
    offsets_npy_path = base + '.attr_offsets.npy'
    meta_path = base + '.attr_offsets_meta.json'
    index_pickle_path = base + '.attr_index.pkl'

    if os.path.exists(offsets_npy_path) and os.path.exists(meta_path):
        # Dense: memory-mapped so workers share pages
        index_lookup = np.load(offsets_npy_path, mmap_mode='r')
        return index_lookup, True
    if os.path.exists(index_pickle_path):
        with open(index_pickle_path, 'rb') as f:
            index_lookup = pickle.load(f)
        return index_lookup, False
    return None, None


def _load_image_attr_path_index(image_attr_path):
    """Load image_path -> byte offset if .attr_path_index.pkl exists. Returns dict or None."""
    path_index_path = image_attr_path + '.attr_path_index.pkl'
    if not os.path.exists(path_index_path):
        return None
    with open(path_index_path, 'rb') as f:
        return pickle.load(f)


def ensure_attr_jsonl_indices(captions_path, image_attr_path, is_master, distributed):
    """Build .index and .attr_* index files if missing. Only rank 0 builds when distributed to avoid races."""
    if distributed and not is_master:
        # Non-master ranks wait; rank 0 will build and then we all proceed
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return
    # Rank 0 or single process: build any missing indices
    captions_index_path = captions_path + '.index'
    if not os.path.exists(captions_index_path):
        logging.info(f'Building captions index for {captions_path} (one-time pass)...')
        _build_jsonl_offset_index(captions_path, captions_index_path)
        logging.info(f'Built captions index, saved to {captions_index_path}.')
    attr_npy = image_attr_path + '.attr_offsets.npy'
    attr_pkl = image_attr_path + '.attr_index.pkl'
    path_pkl = image_attr_path + '.attr_path_index.pkl'
    if not os.path.exists(attr_npy) and not os.path.exists(attr_pkl):
        logging.info(f'Building image-attr index for {image_attr_path} (one-time pass)...')
        _build_image_attr_index(image_attr_path)
    elif not os.path.exists(path_pkl):
        logging.info(f'Building image-attr path index for {image_attr_path} (one-time pass)...')
        _build_image_attr_index(image_attr_path)
    if distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()


class AttrCaptionJsonlDataset(Dataset):
    """Dataset for masked attribute alignment: captions JSONL + image attribute vectors JSONL.
    Uses offset indices for both files so neither is loaded fully into memory (suitable for very large JSONL)."""

    def __init__(self, captions_path, image_attr_path, transforms, tokenizer=None):
        logging.debug(f'Loading captions index from {captions_path}, image attrs from {image_attr_path}.')
        self.captions_path = captions_path
        self.image_attr_path = image_attr_path
        self.transforms = transforms
        self.tokenize = tokenizer
        self._file = None  # lazy per-process open (safe for DataLoader workers)
        self._attr_file = None

        # Load image_attr index (must already exist; built by ensure_attr_jsonl_indices on rank 0)
        self.image_attr_index, self.image_attr_dense = _load_image_attr_index(image_attr_path)
        if self.image_attr_index is None:
            raise FileNotFoundError(
                f'No image-attr index found for {image_attr_path}. '
                'Indices are built by rank 0 before dataset creation; ensure ensure_attr_jsonl_indices was called.'
            )
        self.image_attr_path_index = _load_image_attr_path_index(image_attr_path)
        if self.image_attr_path_index is not None:
            logging.debug(f'Using image-attr path index: {len(self.image_attr_path_index)} entries (img_attr by image_path).')
        if self.image_attr_dense:
            logging.debug(f'Using dense (mmap) image-attr index, size={len(self.image_attr_index)}.')
        else:
            logging.debug(f'Using sparse image-attr index, {len(self.image_attr_index)} entries.')

        # Load captions offset index (must already exist; built by ensure_attr_jsonl_indices on rank 0)
        index_path = captions_path + '.index'
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f'Captions index not found: {index_path}. '
                'Indices are built by rank 0 before dataset creation; ensure ensure_attr_jsonl_indices was called.'
            )
        with open(index_path, 'rb') as f:
            self.offsets = pickle.load(f)
        logging.debug(f'Loaded captions index: {len(self.offsets)} samples from {index_path}.')

    def __len__(self):
        return len(self.offsets)

    def _read_sample(self, idx):
        if self._file is None:
            self._file = open(self.captions_path, 'rb')
        self._file.seek(self.offsets[idx])
        line = self._file.readline()
        return json.loads(line.decode('utf-8'))

    def _get_image_attr(self, image_index, image_path=None):
        """Resolve to attr_values by image_path (if path index exists) or image_index.
        Prefer image_path so img_attr matches the same image as the caption regardless of index ordering."""
        offset = -1
        if self.image_attr_path_index is not None and image_path:
            path_norm = os.path.normpath((image_path or '').strip())
            offset = self.image_attr_path_index.get(path_norm, -1)
        if offset < 0:
            if self.image_attr_dense:
                if image_index < 0 or image_index >= len(self.image_attr_index):
                    return [0] * 9
                offset = int(self.image_attr_index[image_index])
            else:
                offset = self.image_attr_index.get(image_index, -1)
        if offset < 0:
            return [0] * 9
        if self._attr_file is None:
            self._attr_file = open(self.image_attr_path, 'rb')
        self._attr_file.seek(offset)
        line = self._attr_file.readline()
        row = json.loads(line.decode('utf-8'))
        return row.get('attr_values', [0] * 9)

    def __getitem__(self, idx):
        s = self._read_sample(idx)
        image_path = s['image_path']
        image_index = s['image_index']
        caption = s['caption']
        caption_mask = s['caption_mask']
        caption_value = s['caption_value']

        image = self.transforms(Image.open(str(image_path)).convert('RGB'))
        text = self.tokenize([str(caption)])[0]

        img_attr = self._get_image_attr(image_index, image_path=image_path)
        if len(img_attr) != 9:
            img_attr = (list(img_attr) + [0] * 9)[:9]

        img_attr_t = torch.tensor(img_attr, dtype=torch.long)
        txt_attr_t = torch.tensor(caption_value if len(caption_value) >= 9 else (list(caption_value) + [0] * 9)[:9], dtype=torch.long)
        txt_mask_t = torch.tensor(caption_mask if len(caption_mask) >= 9 else (list(caption_mask) + [0] * 9)[:9], dtype=torch.bool)

        return image, text, img_attr_t, txt_attr_t, txt_mask_t, caption


def get_attr_jsonl_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    image_attr_path = getattr(args, 'image_attr_vectors', None)
    assert input_filename, 'train_data or val_data must be set for attr_jsonl'
    assert image_attr_path, 'image_attr_vectors must be set for attr_jsonl'

    # Only rank 0 builds index files; then all ranks synchronize before loading (avoids DDP write races)
    is_master = (not getattr(args, 'distributed', False)) or (getattr(args, 'rank', 0) == 0)
    ensure_attr_jsonl_indices(input_filename, image_attr_path, is_master, getattr(args, 'distributed', False))

    dataset = AttrCaptionJsonlDataset(
        input_filename,
        image_attr_path,
        preprocess_fn,
        tokenizer=tokenizer,
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
        collate_fn=_collate_attr_jsonl,
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
    elif dataset_type == "attr_jsonl":
        return get_attr_jsonl_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1].lower()
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        elif ext == 'jsonl':
            return get_attr_jsonl_dataset
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
