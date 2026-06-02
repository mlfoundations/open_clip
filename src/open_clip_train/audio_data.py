import io
import json
import math
import random
from functools import partial
from typing import Dict, List

import torch
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from open_clip import get_model_config
from open_clip_train.data import (
    DataInfo,
    RepeatedShardList,
    ResampledShards2,
    SharedEpoch,
    TokenizeText,
    _SAMPLE_SHUFFLE_INITIAL,
    _SAMPLE_SHUFFLE_SIZE,
    _SHARD_SHUFFLE_INITIAL,
    _SHARD_SHUFFLE_SIZE,
    append_naflex_train_stages,
    detshuffle2,
    expand_urls,
    get_dataset_size,
    log_and_continue,
    naflex_loader_counts,
    tarfile_to_samples_nothrow,
)
from open_clip_train.naflex_data import collate_naflex_dicts, create_naflex_eval_transform


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


class _TokenizeAudioCaption:
    # Module-level callable (picklable for forkserver workers).
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        return self.tokenizer(_extract_caption(text))[0]


class AudioCaptionTokenizer:
    """Picklable caption-member -> token tensor for the NaFlex audio path.

    Reuses ``_extract_caption`` (json/txt/cls + multi-caption handling); ``variable=True`` returns a 1-D
    unpadded sequence (the generative contract the NaFlex collator pads per-batch), mirroring ``TokenizeText``.
    """

    def __init__(self, tokenizer, variable: bool = False):
        self.tokenizer = tokenizer
        self.variable = variable

    def __call__(self, text):
        caption = _extract_caption(text)
        if self.variable:
            return self.tokenizer(caption, pad=False)[0]
        return self.tokenizer(caption)[0]


def _keep_audio_text(sample):
    return {"audio": sample["audio"], "text": sample["text"]}


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

    # NaFlex audio path (GenLAP generative + NaFlexClap contrastive) mirrors the image NaFlex path
    # (RepeatedShardList head + batcher-derived epoch); otherwise fall through to the standard CLAP fixed-batch
    # loader. `generative` (GenLAP) => variable-length caption concatenated to audio (pad_id, text budget);
    # contrastive (NaFlexClap) => fixed-length text in a separate tower (pad_id=None, no text budget).
    naflex_audio = naflex_data_config is not None and (
        getattr(args, "genlap", False) or getattr(args, "naflexclap", False)
    )
    generative = getattr(args, "genlap", False)
    naflex_train = naflex_audio and is_train
    naflex_eval = naflex_audio and not is_train

    if resampled:
        pipeline = [
            ResampledShards2(
                input_shards,
                weights=args.train_data_upsampling_factors,
                deterministic=True,
                epoch=shared_epoch,
            )
        ]
    elif naflex_train:
        num_shards = num_shards or len(expand_urls(input_shards)[0])
        assert num_shards >= args.workers * args.world_size, "number of shards must be >= total workers"
        pipeline = [RepeatedShardList(input_shards)]
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
    # Shared decode/filter/rename head (sample -> {audio: (waveform, sr), text: <raw caption member>}).
    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_audio),
            wds.decode(_decode_audio, handler=log_and_continue),
            wds.rename(audio=audio_ext, text="json;txt;cls", keep=False),
        ]
    )

    # Generative (GenLAP) concatenates a variable-length caption to the audio (needs pad_id + a per-row text
    # budget); contrastive (NaFlexClap) uses fixed-length text in a separate tower (pad_id=None -> NaFlexBatcher
    # fixed-text collate; no text in the audio token budget).
    naflex_pad_id = getattr(tokenizer, "pad_token_id", None) if generative else None
    naflex_text_cost = (getattr(tokenizer, "context_length", 0) or 0) if generative else 0
    naflex_tokenize = AudioCaptionTokenizer(tokenizer, variable=generative)

    naflex_batcher = None
    if naflex_train:
        # Reuse the modality-agnostic NaFlex tail; the batcher applies preprocess_audio (an
        # AudioNaFlexTransformFactory) to sample["audio"] and reads sample["text"]. Epoch by num_samples
        # (the --naflex-num-train-image-tokens token-budget mode stays image-only for now).
        naflex_batcher = append_naflex_train_stages(
            pipeline,
            naflex_data_config=naflex_data_config,
            transform_factory=preprocess_audio,
            tokenize_text=naflex_tokenize,
            modality_key="audio",
            num_samples=num_samples,
            num_tokens=None,
            args=args,
            shared_epoch=shared_epoch,
            pad_id=naflex_pad_id,
            per_row_text_tokens=naflex_text_cost,
        )
    elif naflex_eval:
        eval_transform, eval_seq_len, _ = create_naflex_eval_transform(preprocess_audio, naflex_data_config)
        pipeline.extend(
            [
                wds.map_dict(audio=eval_transform, text=naflex_tokenize),
                wds.batched(
                    args.batch_size,
                    partial=True,
                    collation_fn=partial(
                        collate_naflex_dicts,
                        image_key="audio",
                        max_seq_len=eval_seq_len,
                        pad_id=naflex_pad_id,
                    ),
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.map_dict(
                    audio=preprocess_audio,
                    text=_TokenizeAudioCaption(tokenizer),
                ),
                wds.map(_keep_audio_text),
                wds.batched(args.batch_size, partial=not is_train, collation_fn=_audio_collate),
            ]
        )

    dataset = wds.DataPipeline(*pipeline)
    if naflex_train:
        # NaFlex epoch length comes from the batcher's deterministic schedule (no with_epoch / fixed batch).
        num_batches, num_samples = naflex_loader_counts(naflex_batcher, args)
    elif is_train:
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
        self.preprocess_txt = TokenizeText(tokenizer)
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
