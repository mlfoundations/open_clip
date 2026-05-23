import io
import json
import math
import random
from typing import Dict, List

import torch
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from open_clip import get_model_config
from open_clip_train.data import (
    DataInfo,
    ResampledShards2,
    SharedEpoch,
    _SAMPLE_SHUFFLE_INITIAL,
    _SAMPLE_SHUFFLE_SIZE,
    _SHARD_SHUFFLE_INITIAL,
    _SHARD_SHUFFLE_SIZE,
    detshuffle2,
    expand_urls,
    get_dataset_size,
    log_and_continue,
    tarfile_to_samples_nothrow,
)


def filter_no_caption_or_no_audio(sample):
    has_caption = "txt" in sample or "json" in sample or "cls" in sample
    has_audio = "wav" in sample or "flac" in sample or "mp3" in sample or "ogg" in sample
    return has_caption and has_audio


def _decode_audio(key, data):
    import torchaudio

    ext = key.rsplit(".", 1)[-1] if "." in key else key
    if ext not in ("wav", "flac", "mp3", "ogg"):
        return None
    return torchaudio.load(io.BytesIO(data))


def _extract_caption(text_data):
    if isinstance(text_data, bytes):
        text_data = text_data.decode("utf-8")
    if isinstance(text_data, str):
        try:
            text_data = json.loads(text_data)
        except json.JSONDecodeError:
            return text_data
    if isinstance(text_data, dict):
        texts = text_data.get("text", text_data.get("caption", ""))
        if isinstance(texts, list) and texts:
            return random.choice(texts)
        if isinstance(texts, str):
            return texts
    return str(text_data)


def _audio_collate(batch: List[Dict]):
    audios = [sample["audio"] for sample in batch]
    texts = [sample["text"] for sample in batch]
    waveforms = torch.stack([audio["waveform"] for audio in audios])
    longers = torch.as_tensor([bool(audio["longer"]) for audio in audios], dtype=torch.bool)
    text_tensor = torch.stack(list(texts))
    audio_batch = {
        "waveform": waveforms,
        "longer": longers,
    }
    if "mel_fusion" in audios[0]:
        audio_batch["mel_fusion"] = torch.stack([audio["mel_fusion"] for audio in audios])
    return {"audio": audio_batch, "text": text_tensor}


def get_wds_audio_dataset(
        args,
        preprocess_audio,
        is_train,
        epoch=0,
        floor=False,
        tokenizer=None,
        naflex_data_config=None,
):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    "Currently, the number of dataset samples must be specified for the training dataset. "
                    "Please specify it via `--train-num-samples` if no dataset length info is present."
                )
    else:
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(epoch=epoch)
    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, (
            "--train_data_upsampling_factors is only supported when sampling with replacement "
            "(with --dataset-resampled)."
        )

    if resampled:
        pipeline = [
            ResampledShards2(
                input_shards,
                weights=args.train_data_upsampling_factors,
                deterministic=True,
                epoch=shared_epoch,
            )
        ]
    else:
        expanded_shards, _ = expand_urls(input_shards)
        pipeline = [wds.SimpleShardList(expanded_shards)]

    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=args.seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                tarfile_to_samples_nothrow,
                wds.shuffle(bufsize=_SAMPLE_SHUFFLE_SIZE, initial=_SAMPLE_SHUFFLE_INITIAL),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )

    audio_ext = getattr(args, "audio_ext", "flac")
    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_audio),
            wds.decode(_decode_audio, handler=log_and_continue),
            wds.rename(audio=audio_ext, text="json;txt;cls", keep=False),
            wds.map_dict(
                audio=preprocess_audio,
                text=lambda text: tokenizer(_extract_caption(text))[0],
            ),
            wds.map(lambda sample: {"audio": sample["audio"], "text": sample["text"]}),
            wds.batched(args.batch_size, partial=not is_train, collation_fn=_audio_collate),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, "number of shards must be >= total workers"
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)
    else:
        num_batches = math.ceil(num_samples / args.batch_size) if num_samples else 0

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


class SyntheticAudioDataset(Dataset):
    def __init__(self, audio_cfg, transform=None, dataset_size=100, tokenizer=None):
        self.audio_cfg = audio_cfg
        self.transform = transform
        self.clip_samples = audio_cfg.get("clip_samples", 480000)
        self.sample_rate = audio_cfg.get("sample_rate", 48000)
        self.dataset_size = dataset_size
        self.preprocess_txt = lambda text: tokenizer(text)[0]
        self.caption = "Dummy caption"

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        waveform = torch.randn(1, self.clip_samples)
        if self.transform is not None:
            audio = self.transform((waveform, self.sample_rate))
        else:
            audio = {"waveform": waveform.squeeze(0), "longer": False}
        return {"audio": audio, "text": self.preprocess_txt(self.caption)}


def get_synthetic_audio_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, naflex_data_config=None):
    model_cfg = get_model_config(args.model) or {}
    audio_cfg = model_cfg.get("audio_cfg", {})
    dataset = SyntheticAudioDataset(
        audio_cfg=audio_cfg,
        transform=preprocess_fn,
        dataset_size=args.train_num_samples,
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
        collate_fn=_audio_collate,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)
