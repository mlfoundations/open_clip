import ast
import json
import logging
import math
import os
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


def _expand_dirs(urls):
    """Expand directory paths to sorted lists of .tar files within them."""
    result = []
    for url in urls:
        if os.path.isdir(url):
            tars = sorted(
                [os.path.join(url, f) for f in os.listdir(url) if f.endswith('.tar')],
                key=lambda p: (int(os.path.splitext(os.path.basename(p))[0])
                               if os.path.splitext(os.path.basename(p))[0].isdigit()
                               else float('inf'), os.path.basename(p)),
            )
            if not tars:
                logging.warning(f'Directory {url} contains no .tar files, skipping.')
            result.extend(tars)
        else:
            result.append(url)
    return result


def expand_urls(urls, weights=None):
    if weights is None:
        if isinstance(urls, str):
            expanded_urls = wds.shardlists.expand_urls(urls)
        else:
            expanded_urls = list(urls)
        expanded_urls = _expand_dirs(expanded_urls)
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
            expanded_url = _expand_dirs(expanded_url)
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        all_urls = _expand_dirs(all_urls)
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
        expanded_shards, _ = expand_urls(input_shards)
        pipeline = [wds.SimpleShardList(expanded_shards)]

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


def filter_no_caption_or_no_audio(sample):
    has_caption = ('txt' in sample or 'json' in sample or 'cls' in sample)
    has_audio = ('wav' in sample or 'flac' in sample or 'mp3' in sample)
    return has_caption and has_audio


def _decode_audio(key, data):
    """WebDataset audio decoder using torchaudio."""
    import io
    import torchaudio
    ext = key.rsplit('.', 1)[-1] if '.' in key else key
    if ext not in ('wav', 'flac', 'mp3', 'ogg'):
        return None
    waveform, sr = torchaudio.load(io.BytesIO(data))
    return waveform, sr


def _extract_caption(text_data):
    """Extract caption string from text or JSON annotation data.

    Handles:
    - Plain text string (from .txt files)
    - JSON dict with 'text' key containing string or list of strings
    - Falls back to str() for unknown formats
    """
    if isinstance(text_data, str):
        return text_data
    if isinstance(text_data, dict):
        texts = text_data.get('text', text_data.get('caption', ''))
        if isinstance(texts, list) and texts:
            return random.choice(texts)
        elif isinstance(texts, str):
            return texts
    return str(text_data)


def int16_to_float32_torch(x):
    """Convert int16 tensor to float32 in [-1, 1] range."""
    return (x / 32767.0).type(torch.float32)


def float32_to_int16_torch(x):
    """Clamp float32 tensor to [-1, 1] and quantize to int16."""
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)


def _get_mel(audio_data, audio_cfg):
    """Compute log-mel spectrogram for fusion mode.

    Parameters
    ----------
    audio_data : torch.Tensor, shape (T,)
        Mono waveform.
    audio_cfg : dict
        Audio configuration (sample_rate, window_size, hop_size, mel_bins, fmin, fmax).

    Returns
    -------
    torch.Tensor, shape (T_frames, n_mels)
    """
    import torchaudio
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg.get('sample_rate', 48000),
        n_fft=audio_cfg.get('window_size', 1024),
        win_length=audio_cfg.get('window_size', 1024),
        hop_length=audio_cfg.get('hop_size', 480),
        center=True, pad_mode="reflect", power=2.0, norm=None, onesided=True,
        n_mels=audio_cfg.get('mel_bins', 64),
        f_min=audio_cfg.get('fmin', 50),
        f_max=audio_cfg.get('fmax', 14000),
    )
    mel = mel_tf(audio_data)
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T  # (T_frames, n_mels)


def make_audio_preprocess(audio_cfg, data_filling="pad", data_truncating="rand_trunc", int16_normalize=False):
    """Create audio preprocessing function from audio config.

    Returns a function that takes (waveform, sr) tuple from the decoder
    and returns a dict suitable for the audio encoder.

    Parameters
    ----------
    audio_cfg : dict
        Audio configuration from model config (sample_rate, clip_samples, etc.)
    data_filling : str
        How to fill audio shorter than clip_samples:
        - "pad": zero-pad to clip_samples (default)
        - "repeat": loop waveform until clip_samples
        - "repeatpad": loop waveform then zero-pad remainder
    data_truncating : str
        How to truncate audio longer than clip_samples:
        - "rand_trunc": random crop of clip_samples each call (default, matches Marianna fork)
        - "trunc": always take first clip_samples (deterministic, old baseline behavior)
        - "fusion": compute 4-channel mel_fusion (global + 3 local chunks) for HTSAT fusion mode
    int16_normalize : bool
        If True, apply int16 quantization roundtrip (clamp→int16→float32) matching CLAP v1/v2.
        Default False: ablation showed it degrades Clotho text R@5 by ~1.7pp and UrbanSound8K by ~2.8pp.
    """
    import torchaudio
    import numpy as np

    target_sr = audio_cfg.get('sample_rate', 48000)
    clip_samples = audio_cfg.get('clip_samples', 480000)
    hop_size = audio_cfg.get('hop_size', 480)
    mel_bins = audio_cfg.get('mel_bins', 64)

    def _fill_waveform(waveform):
        """Apply filling strategy to waveform shorter than clip_samples."""
        if len(waveform) >= clip_samples:
            return waveform
        if data_filling == "repeatpad":
            n_repeat = int(clip_samples / len(waveform))
            if n_repeat > 1:
                waveform = waveform.repeat(n_repeat)
            waveform = torch.nn.functional.pad(waveform, (0, clip_samples - len(waveform)))
        elif data_filling == "repeat":
            n_repeat = int(clip_samples / len(waveform)) + 1
            waveform = waveform.repeat(n_repeat)[:clip_samples]
        else:  # "pad"
            waveform = torch.nn.functional.pad(waveform, (0, clip_samples - len(waveform)))
        return waveform

    def preprocess(audio_data):
        waveform, sr = audio_data
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        waveform = waveform.squeeze(0)  # (samples,)
        # Int16 normalization: clamp → quantize → dequantize (matches CLAP v1/v2 convention)
        if int16_normalize:
            waveform = int16_to_float32_torch(float32_to_int16_torch(waveform))

        longer = len(waveform) > clip_samples
        result = {}

        if len(waveform) > clip_samples:
            if data_truncating == "fusion":
                # Fusion: create 4-channel mel spectrogram
                mel = _get_mel(waveform, audio_cfg)  # (T_frames, n_mels)
                chunk_frames = clip_samples // hop_size + 1
                total_frames = mel.shape[0]
                if chunk_frames >= total_frames:
                    # Corner case: audio barely longer than clip_samples
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    longer = False
                else:
                    # Split frame range into 3 regions, sample one chunk from each
                    ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                    if len(ranges[1]) == 0:
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        ranges[2] = [0]
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])
                    mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
                    # Global view: resize full mel to chunk size
                    import torchvision.transforms
                    mel_shrink = torchvision.transforms.Resize(
                        size=[chunk_frames, mel_bins]
                    )(mel[None])[0]
                    mel_fusion = torch.stack(
                        [mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0
                    )
                result["mel_fusion"] = mel_fusion
                # Also crop waveform (for non-fusion path compatibility)
                overflow = len(waveform) - clip_samples
                idx = np.random.randint(0, overflow + 1)
                waveform = waveform[idx: idx + clip_samples]
            elif data_truncating == "rand_trunc":
                overflow = len(waveform) - clip_samples
                idx = random.randint(0, overflow)
                waveform = waveform[idx: idx + clip_samples]
            else:  # "trunc"
                waveform = waveform[:clip_samples]
        else:
            # Audio is shorter or equal — apply filling
            waveform = _fill_waveform(waveform)
            if data_truncating == "fusion":
                # Fusion for short audio: all 4 channels identical
                mel = _get_mel(waveform, audio_cfg)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                result["mel_fusion"] = mel_fusion

        result["waveform"] = waveform
        result["longer"] = longer
        return result

    return preprocess


def _audio_collate(batch):
    """Custom collation for audio batches: list of (audio_dict, text) -> (batched_dict, text_tensor)."""
    audios, texts = zip(*batch)
    waveforms = torch.stack([a["waveform"] for a in audios])
    longers = torch.tensor([a["longer"] for a in audios], dtype=torch.bool)
    texts = torch.stack(list(texts))
    result = {"waveform": waveforms, "longer": longers}
    if "mel_fusion" in audios[0]:
        result["mel_fusion"] = torch.stack([a["mel_fusion"] for a in audios])
    return result, texts


def get_wds_audio_dataset(args, preprocess_audio, is_train, epoch=0, floor=False, tokenizer=None):
    """WebDataset loader for audio-text pairs. Mirrors get_wds_dataset."""
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
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(epoch=epoch)

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported with --dataset-resampled."

    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        expanded_shards, _ = expand_urls(input_shards)
        pipeline = [wds.SimpleShardList(expanded_shards)]

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
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    audio_ext = getattr(args, 'audio_ext', 'flac')
    pipeline.extend([
        wds.select(filter_no_caption_or_no_audio),
        wds.decode(_decode_audio, handler=log_and_continue),
        wds.rename(audio=audio_ext, text="json;txt;cls"),
        wds.map_dict(
            audio=preprocess_audio,
            text=lambda t: tokenizer(_extract_caption(t))[0],
        ),
        wds.to_tuple("audio", "text"),
        wds.batched(args.batch_size, partial=not is_train, collation_fn=_audio_collate),
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)
    else:
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

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


class SyntheticAudioDataset(Dataset):

    def __init__(self, audio_cfg, dataset_size=100, tokenizer=None):
        self.clip_samples = audio_cfg.get('clip_samples', 480000)
        self.dataset_size = dataset_size
        self.preprocess_txt = lambda text: tokenizer(text)[0]
        self.caption = "Dummy caption"

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        waveform = torch.randn(self.clip_samples)
        audio = {"waveform": waveform, "longer": False}
        return audio, self.preprocess_txt(self.caption)


def get_synthetic_audio_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    from open_clip import get_model_config
    model_cfg = get_model_config(args.model)
    audio_cfg = model_cfg.get('audio_cfg', {})
    dataset = SyntheticAudioDataset(
        audio_cfg=audio_cfg, dataset_size=args.train_num_samples, tokenizer=tokenizer)
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
        collate_fn=_audio_collate,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "webdataset-audio":
        return get_wds_audio_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "synthetic-audio":
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
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type in ("synthetic", "synthetic-audio"):
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
