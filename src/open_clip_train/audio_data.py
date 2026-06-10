import io
import json
import math
import random
from dataclasses import asdict, is_dataclass
from functools import partial
from typing import Dict, List, Optional

import torch
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from open_clip_train.data import (
    DataInfo,
    RepeatedShardList,
    ResampledShards2,
    SharedEpoch,
    TokenizeText,
    append_naflex_train_stages,
    detshuffle2,
    expand_urls,
    get_dataset_size,
    get_text_pad_id,
    log_and_continue,
    naflex_loader_counts,
    tarfile_to_samples_nothrow,
    wds_shuffle_sizes,
)
from open_clip_train.naflex_data import (
    AudioTokenLength,
    CaptionLength,
    LengthBucketer,
    collate_naflex_dicts,
    collate_variable_text,
    create_naflex_eval_transform,
)


def _audio_loader_kwargs(args):
    """Extra DataLoader kwargs for audio loaders: a forkserver multiprocessing context (only when workers > 0).

    Audio decode/mel work in the workers pulls in torchaudio, which spawns threads; the default ``fork`` start
    method then deadlocks the child. forkserver (configurable via --audio-multiprocessing-context) avoids it.
    """
    if getattr(args, "workers", 0) and args.workers > 0:
        return {"multiprocessing_context": getattr(args, "audio_multiprocessing_context", "forkserver")}
    return {}


def filter_no_caption_or_no_audio(sample):
    has_caption = "txt" in sample or "json" in sample or "cls" in sample
    has_audio = "wav" in sample or "flac" in sample or "mp3" in sample or "ogg" in sample
    return has_caption and has_audio


def _decode_audio(key, data):
    # Extension-keyed `wds.decode` handler; used by the legacy (decode-first) pipeline assembly.
    import torchaudio

    ext = key.rsplit(".", 1)[-1] if "." in key else key
    if ext not in ("wav", "flac", "mp3", "ogg"):
        return None
    return torchaudio.load(io.BytesIO(data))


def _decode_audio_bytes(data):
    """Decode raw audio bytes to ``(waveform, sr)`` (per-key map for the post-rename/post-bucket pipelines).

    The format is sniffed from the byte stream (extension keys are gone after ``wds.rename``), matching what
    the extension-keyed ``_decode_audio`` handler produces.
    """
    import torchaudio

    return torchaudio.load(io.BytesIO(data))


class _TokenizeAudioCaption:
    # Module-level callable (picklable for forkserver workers).
    def __init__(self, tokenizer, variable: bool = False):
        self.tokenizer = tokenizer
        self.variable = variable

    def __call__(self, text):
        caption = _extract_caption(text)
        if self.variable:
            return self.tokenizer(caption, pad=False)[0]
        return self.tokenizer(caption)[0]


class AudioCaptionTokenizer:
    """Picklable caption-member -> token tensor for the NaFlex audio path.

    Reuses ``_extract_caption`` (json/txt/cls + multi-caption handling); ``variable=True`` returns a 1-D
    unpadded sequence for per-batch text padding, mirroring ``TokenizeText``.
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


def _audio_collate(
        batch: List[Dict],
        pad_id: Optional[int] = None,
        text_pad_multiple: Optional[int] = None,
        text_pad_cap: Optional[int] = None,
):
    audios = [sample["audio"] for sample in batch]
    texts = [sample["text"] for sample in batch]
    waveforms = torch.stack([audio["waveform"] for audio in audios])
    longers = torch.as_tensor([bool(audio["longer"]) for audio in audios], dtype=torch.bool)
    if pad_id is None:
        text_tensor = torch.stack(list(texts))
        text_valid = None
    else:
        text_tensor, text_valid = collate_variable_text(
            texts, pad_id, pad_multiple=text_pad_multiple, pad_cap=text_pad_cap,
        )
    audio_batch = {
        "waveform": waveforms,
        "longer": longers,
    }
    if "mel_fusion" in audios[0]:
        audio_batch["mel_fusion"] = torch.stack([audio["mel_fusion"] for audio in audios])
    out = {"audio": audio_batch, "text": text_tensor}
    if text_valid is not None:
        out["text_valid"] = text_valid
    return out


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

    # Match the shared-epoch counter's mp context to the loader's (forkserver for audio when workers>0), so its
    # SemLock can be pickled into the workers; otherwise a fork-context lock crossing to forkserver workers errors.
    shared_epoch = SharedEpoch(epoch=epoch, mp_context=_audio_loader_kwargs(args).get("multiprocessing_context"))
    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, (
            "--train_data_upsampling_factors is only supported when sampling with replacement "
            "(with --dataset-resampled)."
        )

    # NaFlex audio uses the shared batcher path; standard CLAP keeps the fixed-batch loader.
    # GenLAP has variable text in the audio row budget; contrastive variable text is padded separately.
    naflex_audio = naflex_data_config is not None and (
        getattr(args, "genlap", False) or getattr(args, "naflexclap", False)
    )
    generative = getattr(args, "genlap", False)
    variable_text = bool(getattr(args, "variable_text", False))
    text_variable = generative or variable_text
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
        shard_shuffle_size, shard_shuffle_initial, sample_shuffle_size, sample_shuffle_initial = wds_shuffle_sizes()
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=shard_shuffle_size,
                        initial=shard_shuffle_initial,
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
                wds.shuffle(bufsize=sample_shuffle_size, initial=sample_shuffle_initial),
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
    # Shared filter/rename head (sample -> {audio: <raw bytes>, text: <raw caption member>}). Audio decode runs
    # later, after tokenize and the optional length bucketer: the bucketer pools `--bucket-pool` complete
    # samples per worker, so it must see raw compressed bytes -- pre-crop decoded waveforms are far larger
    # (a 2-minute 48kHz stereo clip is ~45MB) and can OOM dataloader workers. One ordering for all branches;
    # the decode-first assembly lives in legacy_data.py.
    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_audio),
            wds.rename(audio=audio_ext, text="json;txt;cls", keep=False),
        ]
    )
    decode_audio = wds.map_dict(audio=_decode_audio_bytes, handler=log_and_continue)

    # GenLAP budgets variable text with audio; contrastive variable text only changes padding/collation.
    text_pad_id = get_text_pad_id(tokenizer) if text_variable else None
    text_pad_multiple = getattr(args, "text_pad_multiple", None)
    text_pad_cap = getattr(tokenizer, "context_length", None)
    naflex_pad_id = text_pad_id if text_variable else None
    naflex_text_cost = (getattr(tokenizer, "context_length", 0) or 0) if generative else 0
    naflex_tokenize = AudioCaptionTokenizer(tokenizer, variable=text_variable)

    naflex_batcher = None
    if naflex_train:
        # Native audio is variable-length. NaFlexClap buckets by audio tokens; GenLAP buckets by audio+caption
        # tokens for both block and packed-prefix layouts. Geometry comes from the resolved transform factory.
        audio_bucketer = None
        audio_naflex_cfg = getattr(preprocess_audio, "cfg", None)
        if getattr(args, "length_bucketing", False) and audio_naflex_cfg is not None:
            # Always include audio length; variable-text towers add caption length below.
            length_fns = [
                AudioTokenLength(
                    audio_key="audio",
                    freq_tokens=audio_naflex_cfg.freq_tokens,
                    patch_time=audio_naflex_cfg.patch_time,
                    hop_size=audio_naflex_cfg.hop_size,
                    window_size=audio_naflex_cfg.window_size,  # mirror the transform's sub-window pad
                    sample_rate=audio_naflex_cfg.sample_rate,
                    max_audio_tokens=max(naflex_data_config.train_seq_lens),  # clamp the sort key for outliers
                )
            ]
            if text_variable:
                # Variable text adds caption length to the reorder key; only GenLAP also adds its cap to row budget.
                length_fns.append(CaptionLength(key="text"))
            audio_bucketer = LengthBucketer(
                length_fns=length_fns,
                pool=args.bucket_pool,
                chunk=args.bucket_chunk,
                seed=args.seed,
                epoch=shared_epoch,
            )
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
            bucketer=audio_bucketer,
            decode_stage=decode_audio,
            pad_multiple=getattr(args, "naflex_pad_multiple", None),
            text_pad_multiple=text_pad_multiple,
            text_pad_cap=text_pad_cap,
        )
    elif naflex_eval:
        eval_transform, eval_seq_len, _ = create_naflex_eval_transform(preprocess_audio, naflex_data_config)
        pipeline.extend(
            [
                decode_audio,
                wds.map_dict(audio=eval_transform, text=naflex_tokenize),
                wds.batched(
                    args.batch_size,
                    partial=True,
                    collation_fn=partial(
                        collate_naflex_dicts,
                        image_key="audio",
                        max_seq_len=eval_seq_len,
                        pad_id=naflex_pad_id,
                        text_pad_multiple=text_pad_multiple,
                        text_pad_cap=text_pad_cap,
                    ),
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                decode_audio,
                wds.map_dict(
                    audio=preprocess_audio,
                    text=_TokenizeAudioCaption(tokenizer, variable=variable_text),
                ),
                wds.map(_keep_audio_text),
                wds.batched(
                    args.batch_size,
                    partial=not is_train,
                    collation_fn=partial(
                        _audio_collate, pad_id=text_pad_id,
                        text_pad_multiple=text_pad_multiple,
                        text_pad_cap=text_pad_cap,
                    ) if variable_text else _audio_collate,
                ),
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
        **_audio_loader_kwargs(args),
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


class SyntheticAudioDataset(Dataset):
    def __init__(self, audio_cfg, transform=None, dataset_size=100, tokenizer=None, variable_text: bool = False):
        self.audio_cfg = audio_cfg
        self.transform = transform
        self.sample_rate = int(audio_cfg.get("sample_rate", 48000))
        # NaFlex audio configs are variable-duration and do not define clip_samples. If they do provide a
        # resolved sample_rate, default synthetic clips to one second at that rate; preserve the legacy 10s/48k
        # fallback when no resolved audio config was available at all.
        self.clip_samples = int(audio_cfg.get("clip_samples", self.sample_rate if "sample_rate" in audio_cfg else 480000))
        self.dataset_size = dataset_size
        self.preprocess_txt = TokenizeText(tokenizer, variable=variable_text)
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
    # NOT YET supported for NaFlex audio models (GenLAP / NaFlexClap): SyntheticAudioDataset feeds the transform a
    # (waveform, sample_rate) tuple, but the NaFlex audio transform is a factory that patchifies on demand
    # (__call__(max_seq_len, patch_size)) -- so it would mis-handle the tuple and fail downstream in the collate.
    # Fail loudly with guidance rather than producing a confusing error. Use webdataset-audio for these models.
    if getattr(preprocess_fn, "is_naflex_transform_factory", False):
        raise NotImplementedError(
            "synthetic-audio is not supported for NaFlex audio models (GenLAP / NaFlexClap) yet; the synthetic "
            "dataset emits raw (waveform, sample_rate) but the NaFlex transform expects to patchify on demand. "
            "Use --dataset-type webdataset-audio for these models."
        )
    # Source the audio cfg (dummy clip length / sample rate) from the resolved preprocess transform, which is
    # built from the actual instantiated model. get_model_config(args.model) resolves *built-in* config names
    # ONLY -- it would miss hf-hub:/local-dir: models and any runtime config overrides. AudioPreprocess.cfg is
    # exactly the audio-cfg dict; NaFlex audio factories carry an AudioNaFlexCfg dataclass.
    audio_cfg = getattr(preprocess_fn, "cfg", None)
    if isinstance(audio_cfg, dict):
        audio_cfg = dict(audio_cfg)
    elif is_dataclass(audio_cfg):
        audio_cfg = asdict(audio_cfg)
    elif audio_cfg is not None and hasattr(audio_cfg, "__dict__"):
        audio_cfg = {
            key: value for key, value in vars(audio_cfg).items()
            if not key.startswith("_") and isinstance(value, (str, int, float, bool, tuple, list, dict, type(None)))
        }
    else:
        audio_cfg = {}
    variable_text = bool(getattr(args, "variable_text", False))
    dataset = SyntheticAudioDataset(
        audio_cfg=audio_cfg,
        transform=preprocess_fn,
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
        collate_fn=partial(
            _audio_collate, pad_id=get_text_pad_id(tokenizer),
            text_pad_multiple=getattr(args, "text_pad_multiple", None),
            text_pad_cap=getattr(tokenizer, "context_length", None),
        ) if variable_text else _audio_collate,
        **_audio_loader_kwargs(args),
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)
