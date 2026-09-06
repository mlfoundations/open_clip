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
from functools import partial

import webdataset as wds
from torch.utils.data.dataloader import default_collate

# Audio helpers are referenced through the module (not from-imported) so test monkeypatching of the decode
# stubs (audio_data._decode_audio) applies here too.
from open_clip.model_traits import CLIP_TRAITS
from open_clip_train import audio_data as _audio_data
from open_clip_train.data import (
    DEFAULT_IMAGE_KEY,
    FilterNonEmptyText,
    FilterValidSample,
    JsonCaptionExtractor,
    SharedEpoch,
    TokenizeText,
    collate_variable_text_dicts,
    create_wds_loader,
    get_csv_dataset,
    get_imagenet,
    get_text_pad_id,
    get_wds_sizes,
    log_and_continue,
    wds_shard_head,
)


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
    num_samples, num_shards = get_wds_sizes(args, input_shards, is_train)
    shared_epoch = SharedEpoch(epoch=epoch)

    pipeline = wds_shard_head(args, input_shards, is_train, resampled, shared_epoch, num_shards=num_shards)

    image_key = getattr(args, 'image_key', DEFAULT_IMAGE_KEY) or DEFAULT_IMAGE_KEY
    text_key = getattr(args, 'text_key', 'txt') or 'txt'
    json_text_key = getattr(args, 'json_text_key', None)
    if json_text_key:
        pipeline.extend([
            wds.select(FilterValidSample(json_text_key=json_text_key, image_key=image_key)),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image=image_key, text="json", keep=False),
            wds.map_dict(
                text=JsonCaptionExtractor(json_text_key, sample_probs=getattr(args, 'json_text_key_probs', None)),
                handler=log_and_continue,
            ),
            wds.select(FilterNonEmptyText()),
        ])
    else:
        pipeline.extend([
            wds.select(FilterValidSample(text_key=text_key, image_key=image_key)),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image=image_key, text=text_key, keep=False),
            wds.select(FilterNonEmptyText()),
        ])

    variable_text, _, collate_fn = _legacy_text_collate(args, tokenizer)
    pipeline.extend([
        wds.map_dict(image=preprocess_img, text=TokenizeText(tokenizer, variable=variable_text)),
        wds.batched(args.batch_size, partial=not is_train, collation_fn=collate_fn),
    ])
    dataset = wds.DataPipeline(*pipeline)

    return create_wds_loader(
        dataset, args, is_train, num_samples, shared_epoch, floor=floor,
    )


def get_wds_audio_dataset_legacy(args, preprocess_audio, is_train, epoch=0, floor=False, tokenizer=None):
    """Decode-first audio wds pipeline: select -> wds.decode(_decode_audio) -> rename -> transform+tokenize -> batch."""
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False) and is_train
    num_samples, num_shards = get_wds_sizes(args, input_shards, is_train)
    shared_epoch = SharedEpoch(
        epoch=epoch, mp_context=_audio_data._audio_loader_kwargs(args).get("multiprocessing_context"),
    )

    pipeline = wds_shard_head(args, input_shards, is_train, resampled, shared_epoch, num_shards=num_shards)

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
            text=_audio_data.AudioCaptionTokenizer(tokenizer, variable=variable_text),
        ),
        wds.batched(args.batch_size, partial=not is_train, collation_fn=audio_collate),
    ])
    dataset = wds.DataPipeline(*pipeline)

    return create_wds_loader(
        dataset, args, is_train, num_samples, shared_epoch, floor=floor,
        **_audio_data._audio_loader_kwargs(args),
    )


# The legacy pipeline is args-driven and fixed-batch by contract (it rejects NaFlex below), so shared CSV and
# synthetic builders use constant text traits: no caption token budget, ``variable_text`` decided by args alone.
_LEGACY_TEXT_TRAITS = CLIP_TRAITS


def get_csv_dataset_legacy(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    """Args-only csv builder for the legacy pipeline (``get_csv_dataset`` without a model traits argument)."""
    return get_csv_dataset(
        args, preprocess_fn, is_train, epoch=epoch, tokenizer=tokenizer, model_traits=_LEGACY_TEXT_TRAITS,
    )


def get_data_legacy(args, preprocess_fns, epoch=0, tokenizer=None, naflex_data_config=None):
    """``get_data`` counterpart using the legacy decode-first wds builders (used by ``legacy_main``).

    Args-only, like the pre-traits pipeline: WebDataset types route to the legacy assemblies above, csv types to
    :func:`get_csv_dataset_legacy`, synthetic types to args-only adapters of the default builders.
    NaFlex is not supported here -- use ``main.py`` / ``data.get_data``.
    """
    from open_clip_train.data import get_dataset_fn

    if naflex_data_config is not None or getattr(args, 'use_naflex', False):
        raise ValueError("legacy data pipelines do not support NaFlex; use open_clip_train.main / data.get_data.")
    if getattr(args, 'length_bucketing', False):
        raise ValueError("legacy data pipelines do not support --length-bucketing; use data.get_data.")

    def dataset_fn(data_path, dataset_type):
        ext = data_path.split('.')[-1] if data_path else ''
        if dataset_type == "webdataset" or (dataset_type == "auto" and ext == 'tar'):
            return get_wds_dataset_legacy
        if dataset_type == "webdataset-audio":
            return get_wds_audio_dataset_legacy
        if dataset_type == "csv" or (dataset_type == "auto" and ext in ('csv', 'tsv')):
            return get_csv_dataset_legacy
        if dataset_type in ("synthetic", "synthetic-audio"):
            return partial(get_dataset_fn(data_path, dataset_type), model_traits=_LEGACY_TEXT_TRAITS)
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
