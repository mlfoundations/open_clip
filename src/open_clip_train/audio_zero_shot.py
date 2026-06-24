import logging
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from open_clip import build_zero_shot_classifier, get_input_dtype, get_tokenizer
from open_clip.audio.transform import audio_transform_v2
from open_clip.task import get_model_from_task
from open_clip_train.precision import get_autocast
from open_clip_train.zero_shot import accuracy

_logger = logging.getLogger(__name__)

AUDIO_ZEROSHOT_TEMPLATES_ALT = (
    "A sound of {}.",
    "The sound of {}.",
    "Audio of {}.",
    "A recording of {}.",
    "The sound of a {}.",
    "This is a sound of {}.",
)

AUDIO_ZEROSHOT_TEMPLATES = (
    "This is a sound of {}.",
)


@dataclass
class AudioZeroShotData:
    dataloader: Optional[DataLoader]
    classnames: List[str]
    dataset_name: str


def is_audio_zeroshot_compatible(model_or_task) -> bool:
    """Return True if the audio zero-shot path can call ``model(audio=...)``."""
    model = get_model_from_task(model_or_task)
    return hasattr(model, "audio") and hasattr(model, "encode_audio")


def validate_audio_zeroshot_compatible(model_or_task):
    if not is_audio_zeroshot_compatible(model_or_task):
        raise ValueError("Audio zero-shot evaluation requires a CLAP-style audio model.")


def _normalise_class_name(name: str) -> str:
    return str(name).replace("_", " ")


def _get_dataset_column(dataset, key: str):
    try:
        return dataset[key]
    except (KeyError, TypeError, AttributeError):
        return None


def _get_classnames_and_target_map(dataset, target_key: str, class_key: Optional[str]):
    features = getattr(dataset, "features", {})
    target_feature = features.get(target_key) if hasattr(features, "get") else None
    names = getattr(target_feature, "names", None)
    if names:
        classnames = [_normalise_class_name(name) for name in names]
        return classnames, {label: label for label in range(len(classnames))}

    target_values = _get_dataset_column(dataset, target_key)
    class_values = _get_dataset_column(dataset, class_key) if class_key else None
    if class_key:
        by_target = {}
        if target_values is not None and class_values is not None:
            for target, class_name in zip(target_values, class_values):
                by_target[int(target)] = _normalise_class_name(class_name)
        else:
            for sample in dataset:
                by_target[int(sample[target_key])] = _normalise_class_name(sample[class_key])
        labels = sorted(by_target)
        return [by_target[idx] for idx in labels], {label: index for index, label in enumerate(labels)}

    labels = (
        sorted({int(target) for target in target_values})
        if target_values is not None
        else sorted(
            {int(sample[target_key]) for sample in dataset}
        )
    )
    return [str(label) for label in labels], {label: index for index, label in enumerate(labels)}


def _get_classnames(dataset, target_key: str, class_key: Optional[str]) -> List[str]:
    return _get_classnames_and_target_map(dataset, target_key, class_key)[0]


def _get_target_map(dataset, target_key: str) -> Dict[int, int]:
    return _get_classnames_and_target_map(dataset, target_key, None)[1]


def _move_to_device(value, device, input_dtype=None):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(device=device, dtype=input_dtype, non_blocking=True)
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(val, device, input_dtype) for key, val in value.items()}
    return value


def _prepare_audio(model_or_task, audio, device, input_dtype=None):
    if hasattr(model_or_task, "prepare_batch"):
        return model_or_task.prepare_batch({"audio": audio}, device, input_dtype)["audio"]
    return _move_to_device(audio, device, input_dtype)


def _create_dummy_audio(model_or_task, device, input_dtype=None):
    if hasattr(model_or_task, "create_dummy_batch"):
        return model_or_task.create_dummy_batch(batch_size=1, device=device, dtype=input_dtype)["audio"]

    model = get_model_from_task(model_or_task)
    audio_cfg = model.audio.cfg
    dummy_audio = {
        "waveform": torch.zeros(1, audio_cfg.clip_samples, device=device, dtype=input_dtype),
        "longer": torch.zeros(1, dtype=torch.bool, device=device),
    }
    if audio_cfg.enable_fusion:
        from open_clip.audio.transform import get_audio_frame_count

        dummy_audio["mel_fusion"] = torch.zeros(
            1,
            4,
            get_audio_frame_count(audio_cfg),
            audio_cfg.mel_bins,
            device=device,
            dtype=input_dtype,
        )
    return dummy_audio


def _validate_audio_templates(templates: Sequence[str]) -> None:
    for template in templates:
        if "{}" not in template:
            raise ValueError(f"Audio zero-shot template missing '{{}}' placeholder: {template!r}")


def _extract_audio_array_and_rate(sample: Dict, audio_key: str):
    audio = sample[audio_key]
    if isinstance(audio, tuple) and len(audio) == 2:
        return audio

    if isinstance(audio, dict):
        array = audio["array"]
        sample_rate = audio["sampling_rate"]
    else:
        try:
            array = audio["array"]
            sample_rate = audio["sampling_rate"]
        except (KeyError, TypeError, AttributeError):
            array = audio
            sample_rate = sample.get("sampling_rate")

    if sample_rate is None:
        raise KeyError(
            "Audio sample must include a sampling rate. Expected an audio dict/decoder with "
            "`sampling_rate` or a top-level `sampling_rate` field."
        )
    return array, sample_rate


class HFAudioClassificationDataset(Dataset):
    """Hugging Face audio classification dataset wrapper for CLAP zero-shot."""

    def __init__(
            self,
            dataset,
            transform,
            *,
            audio_key: str = "audio",
            target_key: str = "target",
            target_map: Optional[Dict[int, int]] = None,
    ):
        self.dataset = dataset
        self.transform = transform
        self.audio_key = audio_key
        self.target_key = target_key
        self.target_map = target_map or {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        array, sample_rate = _extract_audio_array_and_rate(sample, self.audio_key)

        waveform = torch.as_tensor(array, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return {
            "audio": self.transform((waveform, sample_rate)),
            "target": self.target_map.get(int(sample[self.target_key]), int(sample[self.target_key])),
        }


def _collate_audio_zero_shot(batch: Sequence[Dict]):
    audio_items = [sample["audio"] for sample in batch]
    audio_batch = {
        "waveform": torch.stack([audio["waveform"] for audio in audio_items]),
        "longer": torch.as_tensor([bool(audio["longer"]) for audio in audio_items], dtype=torch.bool),
    }
    if "mel_fusion" in audio_items[0]:
        audio_batch["mel_fusion"] = torch.stack([audio["mel_fusion"] for audio in audio_items])
    return {
        "audio": audio_batch,
        "target": torch.as_tensor([sample["target"] for sample in batch], dtype=torch.long),
    }


def build_hf_audio_zero_shot_dataset(args, model_or_task):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Audio zero-shot evaluation with Hugging Face datasets requires `datasets`. "
            "Install with `pip install datasets[audio]`."
        ) from e

    dataset_name = args.audio_zeroshot_dataset
    split = getattr(args, "audio_zeroshot_split", "train")
    audio_key = getattr(args, "audio_zeroshot_audio_key", "audio")
    target_key = getattr(args, "audio_zeroshot_target_key", "target")
    class_key = getattr(args, "audio_zeroshot_class_key", "category")
    dataset = load_dataset(dataset_name, split=split)
    classnames, target_map = _get_classnames_and_target_map(dataset, target_key=target_key, class_key=class_key)

    model = get_model_from_task(model_or_task)
    if getattr(model.audio.cfg, "model_type", "").lower() == "naflexvit":
        # NaFlexClap: the spectrogram-ViT tower consumes patchified mel ({patches, patch_coord, patch_valid}),
        # NOT the HTSAT {waveform, longer}. Build the NaFlex audio transform + the pad-to-seq-len collate.
        from open_clip.audio.naflex_audio import (
            AudioNaFlexCfg, AudioNaFlexTransformFactory, naflex_audio_eval_seq_len)
        from open_clip_train.naflex_data import collate_naflex_dicts

        naflex_cfg = AudioNaFlexCfg.from_clip_audio_cfg(model.audio.cfg)
        # Audio-token cap for the eval clips: explicit --naflex-seq-lens > the model config's audio_seq_len
        # (mirror of vision_cfg.image_seq_len) > a geometry-derived ~10s default (vs the old hardcoded 256,
        # which truncated clips past ~6s).
        seq_lens = getattr(args, "naflex_seq_lens", None)
        if seq_lens:
            seq_len = max(seq_lens)
        elif getattr(model.audio.cfg, "audio_seq_len", None):
            seq_len = int(model.audio.cfg.audio_seq_len)
        else:
            seq_len = naflex_audio_eval_seq_len(naflex_cfg, 10.0)
        secs = (seq_len / naflex_cfg.freq_tokens) * naflex_cfg.patch_time / 100.0
        _logger.info(
            "NaFlex audio zero-shot: capping eval clips at %d tokens (~%.0fs at this model's geometry); longer "
            "clips are truncated. Override with --naflex-seq-lens.", seq_len, secs,
        )
        transform = AudioNaFlexTransformFactory(naflex_cfg)(max_seq_len=seq_len, patch_size=None)
        collate_fn = partial(collate_naflex_dicts, primary_key="audio", target_key="target", max_seq_len=seq_len)
    else:
        audio_aug_cfg = {
            "data_trunc": getattr(args, "audio_trunc", "rand_trunc"),
            "data_fill": getattr(args, "audio_fill", "repeatpad"),
            "enable_fusion": getattr(args, "audio_fusion", False),
            "int16_normalize": getattr(args, "audio_int16_normalize", False),
        }
        transform = audio_transform_v2(model.audio.cfg, is_train=False, audio_aug_cfg=audio_aug_cfg)
        collate_fn = _collate_audio_zero_shot
    wrapped = HFAudioClassificationDataset(
        dataset,
        transform,
        audio_key=audio_key,
        target_key=target_key,
        target_map=target_map,
    )
    num_workers = getattr(args, "audio_zeroshot_workers", 0)
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["multiprocessing_context"] = getattr(
            args,
            "audio_zeroshot_multiprocessing_context",
            "forkserver",
        )
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(
        wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=getattr(args, "device", "cpu") != "cpu",
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    return AudioZeroShotData(dataloader=dataloader, classnames=classnames, dataset_name=dataset_name)


def run_audio_zero_shot_classifier(model, classifier, dataloader, args, use_fsdp_eval=False):
    device = torch.device(args.device)
    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, "fsdp", False),
    )
    input_dtype = get_input_dtype(args.precision)
    is_rank0 = args.rank == 0

    if use_fsdp_eval and not is_rank0:
        dummy_audio = _create_dummy_audio(model, device=device, input_dtype=input_dtype)

    with torch.inference_mode():
        top1, top5, n = 0.0, 0.0, 0.0
        top5_k = min(5, classifier.shape[1])

        if use_fsdp_eval:
            signal = torch.zeros(1, device=device, dtype=torch.long)
            if is_rank0:
                dataloader_iter = iter(dataloader)

            while True:
                if is_rank0:
                    batch = next(dataloader_iter, None)
                    signal.fill_(0 if batch is None else 1)
                dist.broadcast(signal, src=0)
                if signal.item() == 0:
                    break

                if is_rank0:
                    audio = _prepare_audio(model, batch["audio"], device, input_dtype)
                    target = batch["target"].to(device, non_blocking=True)
                else:
                    audio = dummy_audio

                with autocast():
                    output = model(audio=audio)
                    audio_features = output["audio_features"] if isinstance(output, dict) else output[0]

                if is_rank0:
                    logits = 100.0 * audio_features @ classifier
                    acc1, acc5 = accuracy(logits, target, topk=(1, top5_k))
                    top1 += acc1
                    top5 += acc5
                    n += audio_features.shape[0]
        else:
            for batch in tqdm(dataloader, unit_scale=args.batch_size):
                audio = _prepare_audio(model, batch["audio"], device, input_dtype)
                target = batch["target"].to(device, non_blocking=True)

                with autocast():
                    output = model(audio=audio)
                    audio_features = output["audio_features"] if isinstance(output, dict) else output[0]
                    logits = 100.0 * audio_features @ classifier

                acc1, acc5 = accuracy(logits, target, topk=(1, top5_k))
                top1 += acc1
                top5 += acc5
                n += audio_features.shape[0]

    top1 = (top1 / n) if n else 0.0
    top5 = (top5 / n) if n else 0.0
    return top1, top5


def audio_zero_shot_eval(model_or_task, audio_data: Optional[AudioZeroShotData], epoch, args, tokenizer=None):
    if audio_data is None:
        return {}
    validate_audio_zeroshot_compatible(model_or_task)
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    use_fsdp_eval = getattr(args, "fsdp", False) and getattr(args, "distributed", False)
    is_rank0 = args.rank == 0

    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    classnames = audio_data.classnames if is_rank0 else []

    if use_fsdp_eval:
        object_list = [classnames]
        dist.broadcast_object_list(object_list, src=0)
        classnames = object_list[0]

    templates = getattr(args, "audio_zeroshot_templates", None) or AUDIO_ZEROSHOT_TEMPLATES
    _validate_audio_templates(templates)
    device = torch.device(args.device)
    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, "fsdp", False),
    )
    with autocast():
        classifier = build_zero_shot_classifier(
            model_or_task,
            tokenizer=tokenizer,
            classnames=classnames,
            templates=templates,
            num_classes_per_batch=10,
            device=device,
            use_tqdm=is_rank0,
        )

    top1, top5 = run_audio_zero_shot_classifier(
        model_or_task,
        classifier,
        audio_data.dataloader,
        args,
        use_fsdp_eval=use_fsdp_eval,
    )
    if not is_rank0:
        return {}

    dataset_slug = audio_data.dataset_name.rstrip("/").split("/")[-1]
    return {
        f"{dataset_slug}-zeroshot-top1": top1,
        f"{dataset_slug}-zeroshot-top5": top5,
    }
