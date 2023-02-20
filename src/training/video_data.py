"""video dataset creation"""
import io
import math
import torchvision
import tempfile
import webdataset as wds

from dataclasses import dataclass
from multiprocessing import Value
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample


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
    shared_epoch: SharedEpoch = None
    sampler = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


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

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples

def create_webdataset(
    args,
    video_transform,
    tokenizer=None,
):
    pipeline = [wds.SimpleShardList(args.train_data)]
    is_train = True

    pipeline.extend([
        wds.split_by_node,
        wds.split_by_worker,
        tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
    ])

    pipeline.extend([
        wds.decode(wds.torch_video, handler=log_and_continue),
        wds.rename(video="mp4", text="txt"),
        wds.map_dict(video=video_transform, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("video", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)
    return dataset


def get_wds_dataset(args, preprocess_vid, is_train, epoch=0, floor=False, tokenizer=None):
    num_samples = args.train_num_samples

    dataset = create_webdataset(
        args,
        preprocess_vid,
        tokenizer=tokenizer,
    )

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers,
        persistent_workers=True,
        prefetch_factor=8,
        pin_memory=True,
    )

    round_fn = math.floor
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    shared_epoch = SharedEpoch(epoch=epoch)

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_video_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_wds_dataset(args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    return data
