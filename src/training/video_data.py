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


def create_webdataset(
    urls,
    video_transform,
    enable_text=True,
    enable_video=True,
    video_key="mp4",
    caption_key="txt",
    enable_metadata=False,
    cache_path=None,
    input_sampler=lambda a: a,
    tokenizer=None,
):
    """Create a WebDataset reader, it can read a webdataset of video, text and json"""

    urls = input_sampler(urls)

    dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10**10, handler=wds.handlers.warn_and_continue)

    def filter_dataset(item):
        if enable_text and caption_key not in item:
            return False
        if enable_video and video_key not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        output = {}
        if enable_video:
            video_data = item[video_key]
            with tempfile.NamedTemporaryFile() as f:
                f.write(video_data)
                video, audio, meta = torchvision.io.read_video(f.name, output_format="TCHW")
            video_tensor = video_transform(video)
            print(video_tensor.shape)
            output["video_filename"] = item["__key__"]
            output["video_tensor"] = video_tensor

        if enable_text:
            text = item[caption_key]
            caption = text.decode("utf-8")
            tokenized_text = tokenizer(caption)[0]
            output["text_tokens"] = tokenized_text
            output["text"] = caption

        if enable_metadata:
            metadata_file = item["json"]
            metadata = metadata_file.decode("utf-8")
            output["metadata"] = metadata
        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, input_format):
    """Create a pytorch dataloader from a dataset"""

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    #     pin_memory=True,
    #     prefetch_factor=2,
    data = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_prepro_workers,
        collate_fn=collate_fn if input_format == "files" else None,
    )
    return data


class VideoDatasetReader:
    """WebdatasetReader is a reader that reads samples from a webdataset"""

    def __init__(
        self,
        sampler,
        preprocess,
        input_dataset,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_video=True,
        enable_metadata=False,
        wds_video_key="mp4",
        wds_caption_key="txt",
        cache_path=None,
        tokenizer=None,
    ):
        self.batch_size = batch_size
        dataset = create_webdataset(
            input_dataset,
            preprocess,
            enable_text=enable_text,
            enable_video=enable_video,
            video_key=wds_video_key,
            caption_key=wds_caption_key,
            enable_metadata=enable_metadata,
            cache_path=cache_path,
            input_sampler=sampler,
            tokenizer=tokenizer,
        )
        self.dataloader = dataset_to_dataloader(dataset, batch_size, num_prepro_workers, "webdataset")
        self.num_batches = 0
        self.num_samples = 0


    def __iter__(self):
        for batch in self.dataloader:
            yield batch


def get_wds_dataset(args, preprocess_vid, is_train, epoch=0, floor=False, tokenizer=None):
    num_samples = args.train_num_samples

    wds = VideoDatasetReader(
        sampler=lambda a: a,
        preprocess=preprocess_vid,
        input_dataset=args.train_data,
        batch_size=args.batch_size,
        num_prepro_workers=args.workers,
        enable_metadata=True,
        tokenizer=tokenizer,
    )

    round_fn = math.floor
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    wds.num_batches = num_batches
    wds.num_samples = num_samples

    shared_epoch = SharedEpoch(epoch=epoch)

    return DataInfo(dataloader=wds, shared_epoch=shared_epoch)


def get_video_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_wds_dataset(args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    return data

