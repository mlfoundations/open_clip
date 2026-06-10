"""Legacy (decode-first) WebDataset pipeline assembly, kept for reference and as a fallback.

FROZEN -- new features go in ``data.py`` / ``audio_data.py``; do not extend this module.

The default builders assemble wds pipelines as ``select -> rename -> tokenize -> [length bucketer] ->
decode -> transform -> batch`` so the length bucketer (and anything else between tokenize and decode) pools
raw, undecoded samples. This module preserves the original assembly for familiarity / what-changed reference:

    select -> wds.decode (extension-keyed handlers) -> rename -> transform + tokenize -> batch

Differences vs the default builders:
  - decode-first via extension-keyed ``wds.decode`` handlers (``pilrgb`` / ``_decode_audio``),
  - no ``--length-bucketing`` support,
  - no NaFlex support (``get_data_legacy`` raises if a NaFlex data config is passed),
  - json caption members (``--json-text-key``) and variable-length text are supported as before.

Only the pipeline *assembly* is duplicated here; all stages (filters, extractors, tokenizers, collators,
loader helpers) are imported from the default modules so the building blocks cannot drift.
"""
import math

from functools import partial

import webdataset as wds
from torch.utils.data.dataloader import default_collate

# Audio helpers are referenced through the module (not from-imported) so test monkeypatching of the decode
# stubs (audio_data._decode_audio) applies here too.
from open_clip_train import audio_data as _audio_data
from open_clip_train.data import (
    DataInfo,
    FilterValidSample,
    JsonCaptionExtractor,
    ResampledShards2,
    SharedEpoch,
    TokenizeText,
    collate_variable_text_dicts,
    detshuffle2,
    expand_urls,
    get_dataset_size,
    get_imagenet,
    get_text_pad_id,
    log_and_continue,
    tarfile_to_samples_nothrow,
    wds_shuffle_sizes,
)


def _legacy_shard_head(pipeline_seed_args, input_shards, is_train, resampled, shared_epoch):
    """Shared shard-list + shuffle head (identical to the default builders)."""
    args = pipeline_seed_args
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

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
            tarfile_to_samples_nothrow,
            wds.shuffle(bufsize=sample_shuffle_size, initial=sample_shuffle_initial),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    return pipeline


def _legacy_wds_sizes(args, input_shards, is_train):
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
    return num_samples, num_shards


def _legacy_text_collate(args, tokenizer):
    """Variable-text collate wiring shared by the legacy image and audio builders."""
    variable_text = bool(getattr(args, 'variable_text', False))
    if not variable_text:
        return False, None, default_collate
    collate_fn = partial(
        collate_variable_text_dicts,
        pad_id=get_text_pad_id(tokenizer),
        text_pad_multiple=getattr(args, 'text_pad_multiple', None),
        text_pad_cap=getattr(tokenizer, 'context_length', None),
    )
    return True, get_text_pad_id(tokenizer), collate_fn


def get_wds_dataset_legacy(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    """Decode-first image wds pipeline: select -> wds.decode('pilrgb') -> rename -> transform+tokenize -> batch."""
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    num_samples, num_shards = _legacy_wds_sizes(args, input_shards, is_train)
    shared_epoch = SharedEpoch(epoch=epoch)

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, (
            "--train_data_upsampling_factors is only supported when sampling with replacement "
            "(with --dataset-resampled)."
        )

    pipeline = _legacy_shard_head(args, input_shards, is_train, resampled, shared_epoch)

    text_key = getattr(args, 'text_key', 'txt') or 'txt'
    json_text_key = getattr(args, 'json_text_key', None)
    if json_text_key:
        pipeline.extend([
            wds.select(FilterValidSample(json_text_key=json_text_key)),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", json="json", keep=False),
            wds.map(JsonCaptionExtractor(json_text_key), handler=log_and_continue),
        ])
    else:
        pipeline.extend([
            wds.select(FilterValidSample(text_key=text_key)),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text=text_key, keep=False),
        ])

    variable_text, _, collate_fn = _legacy_text_collate(args, tokenizer)
    pipeline.extend([
        wds.map_dict(image=preprocess_img, text=TokenizeText(tokenizer, variable=variable_text)),
        wds.batched(args.batch_size, partial=not is_train, collation_fn=collate_fn),
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


def get_wds_audio_dataset_legacy(args, preprocess_audio, is_train, epoch=0, floor=False, tokenizer=None):
    """Decode-first audio wds pipeline: select -> wds.decode(_decode_audio) -> rename -> transform+tokenize -> batch."""
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False) and is_train
    num_samples, num_shards = _legacy_wds_sizes(args, input_shards, is_train)
    shared_epoch = SharedEpoch(epoch=epoch)

    pipeline = _legacy_shard_head(args, input_shards, is_train, resampled, shared_epoch)

    audio_ext = getattr(args, "audio_ext", "flac")
    variable_text, text_pad_id, _ = _legacy_text_collate(args, tokenizer)
    audio_collate = (
        partial(
            _audio_data._audio_collate,
            pad_id=text_pad_id,
            text_pad_multiple=getattr(args, "text_pad_multiple", None),
            text_pad_cap=getattr(tokenizer, "context_length", None),
        )
        if variable_text else _audio_data._audio_collate
    )
    pipeline.extend([
        wds.select(_audio_data.filter_no_caption_or_no_audio),
        wds.decode(_audio_data._decode_audio, handler=log_and_continue),
        wds.rename(audio=audio_ext, text="json;txt;cls", keep=False),
        wds.map_dict(
            audio=preprocess_audio,
            text=_audio_data._TokenizeAudioCaption(tokenizer, variable=variable_text),
        ),
        wds.map(_audio_data._keep_audio_text),
        wds.batched(args.batch_size, partial=not is_train, collation_fn=audio_collate),
    ])
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
        **_audio_data._audio_loader_kwargs(args),
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_data_legacy(args, preprocess_fns, epoch=0, tokenizer=None, naflex_data_config=None):
    """``get_data`` counterpart using the legacy decode-first wds builders (used by ``legacy_main``).

    WebDataset types route to the legacy assemblies above; csv/synthetic types have no wds pipeline and
    delegate to the default builders. NaFlex is not supported here -- use ``main.py`` / ``data.get_data``.
    """
    from open_clip_train.data import get_dataset_fn

    if naflex_data_config is not None or getattr(args, 'use_naflex', False):
        raise ValueError("legacy data pipelines do not support NaFlex; use open_clip_train.main / data.get_data.")
    if getattr(args, 'length_bucketing', False):
        raise ValueError("legacy data pipelines do not support --length-bucketing; use data.get_data.")

    def dataset_fn(data_path, dataset_type):
        if dataset_type == "webdataset" or (
                dataset_type == "auto" and data_path and data_path.split('.')[-1] == 'tar'
        ):
            return get_wds_dataset_legacy
        if dataset_type == "webdataset-audio":
            return get_wds_audio_dataset_legacy
        return get_dataset_fn(data_path, dataset_type)

    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type in ("synthetic", "synthetic-audio"):
        fn = dataset_fn(args.train_data, args.dataset_type)
        data["train"] = fn(args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        fn = dataset_fn(args.val_data, args.dataset_type)
        data["val"] = fn(args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
