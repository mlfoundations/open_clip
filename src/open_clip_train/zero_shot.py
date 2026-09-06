import logging

_logger = logging.getLogger(__name__)

import torch

from open_clip import get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from open_clip.task import get_model_from_task
from open_clip.utils import move_to_device
from open_clip_train.eval_utils import accuracy, run_classification_eval
from open_clip_train.precision import get_autocast


def is_imagenet_zeroshot_compatible(model_or_task) -> bool:
    """Return True if the ImageNet zero-shot path can call ``model(image=...)``."""
    model = get_model_from_task(model_or_task)
    return hasattr(model, "visual") and hasattr(model, "encode_image")


def validate_imagenet_zeroshot_compatible(model_or_task):
    if not is_imagenet_zeroshot_compatible(model_or_task):
        raise ValueError("ImageNet zero-shot evaluation is image-only and requires an image model.")


def run_zero_shot_classifier(model, classifier, dataloader, args, use_fsdp_eval=False):
    def prepare(batch, device, dtype):
        images, target = batch
        return move_to_device(images, device, dtype), target.to(device, non_blocking=True)

    def dummy(device, dtype):
        if hasattr(model, "create_dummy_batch"):
            return model.create_dummy_batch(batch_size=1, device=device, dtype=dtype)["image"]
        if getattr(args, 'use_naflex', False):
            raise ValueError("NaFlex FSDP zero-shot eval requires an ImageTextTask dummy batch interface.")
        image_size = get_model_from_task(model).visual.image_size
        if not isinstance(image_size, tuple):
            image_size = (image_size, image_size)
        return torch.zeros(1, 3, *image_size, device=device, dtype=dtype)

    return run_classification_eval(
        model, classifier, dataloader, args, input_key="image", prepare_batch=prepare, create_dummy=dummy,
        use_fsdp_eval=use_fsdp_eval,
    )


def zero_shot_eval(model_or_task, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    # Reject non-image models (e.g. audio) first, then skip image models that lack a contrastive text
    # tower (generative VLMs such as GenLIP): the text-classifier zero-shot path requires encode_text.
    validate_imagenet_zeroshot_compatible(model_or_task)
    if not hasattr(get_model_from_task(model_or_task), 'encode_text'):
        _logger.warning(
            "Skipping zero-shot ImageNet eval: model has no `encode_text` "
            "(generative models such as GenLIP have no contrastive text tower)."
        )
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    use_fsdp_eval = getattr(args, 'fsdp', False) and getattr(args, 'distributed', False)
    is_rank0 = (args.rank == 0)

    if is_rank0:
        _logger.info('Starting zero-shot imagenet.')

    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    if is_rank0:
        _logger.info('Building zero-shot classifier')

    device = torch.device(args.device)
    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, 'fsdp', False),
    )

    # All ranks must call forward() for FSDP collective ops.
    # build_zero_shot_classifier is deterministic — same number of forward calls on all ranks.
    with autocast():
        classifier = build_zero_shot_classifier(
            model_or_task,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=device,
            use_tqdm=is_rank0,
        )

    if is_rank0:
        _logger.info('Using classifier')

    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run_zero_shot_classifier(
            model_or_task, classifier, data['imagenet-val'].dataloader, args,
            use_fsdp_eval=use_fsdp_eval,
        )
        if is_rank0:
            results['imagenet-zeroshot-val-top1'] = top1
            results['imagenet-zeroshot-val-top5'] = top5

    if 'imagenet-v2' in data:
        top1, top5 = run_zero_shot_classifier(
            model_or_task, classifier, data['imagenet-v2'].dataloader, args,
            use_fsdp_eval=use_fsdp_eval,
        )
        if is_rank0:
            results['imagenetv2-zeroshot-val-top1'] = top1
            results['imagenetv2-zeroshot-val-top5'] = top5

    if is_rank0:
        _logger.info('Finished zero-shot imagenet.')

    return results
