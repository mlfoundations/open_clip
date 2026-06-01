import logging

_logger = logging.getLogger(__name__)

import torch
import torch.distributed as dist
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from open_clip.task import get_model_from_task
from open_clip_train.precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum().item() for k in topk]


def _move_to_device(value, device, input_dtype=None):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(device=device, dtype=input_dtype, non_blocking=True)
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(val, device, input_dtype) for key, val in value.items()}
    return value


def _image_batch_size(images) -> int:
    if isinstance(images, dict):
        return images["patches"].shape[0]
    return images.size(0)


def _get_dummy_batch_creator(model_or_task):
    if hasattr(model_or_task, "create_dummy_batch"):
        return model_or_task.create_dummy_batch
    return None


def is_imagenet_zeroshot_compatible(model_or_task) -> bool:
    """Return True if the ImageNet zero-shot path can call ``model(image=...)``."""
    model = get_model_from_task(model_or_task)
    return hasattr(model, "visual") and hasattr(model, "encode_image")


def validate_imagenet_zeroshot_compatible(model_or_task):
    if not is_imagenet_zeroshot_compatible(model_or_task):
        raise ValueError("ImageNet zero-shot evaluation is image-only and requires an image model.")


def run_zero_shot_classifier(model, classifier, dataloader, args, use_fsdp_eval=False):
    device = torch.device(args.device)
    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, 'fsdp', False),
    )
    input_dtype = get_input_dtype(args.precision)
    is_rank0 = (args.rank == 0)

    if use_fsdp_eval and not is_rank0:
        dummy_batch_creator = _get_dummy_batch_creator(model)
        if dummy_batch_creator is not None:
            dummy_images = dummy_batch_creator(batch_size=1, device=device, dtype=input_dtype)["image"]
        else:
            if getattr(args, 'use_naflex', False):
                raise ValueError("NaFlex FSDP zero-shot eval requires an ImageTextTask dummy batch interface.")
            raw_model = get_model_from_task(model)
            image_size = raw_model.visual.image_size
            if not isinstance(image_size, tuple):
                image_size = (image_size, image_size)
            dummy_images = torch.zeros(1, 3, *image_size, device=device, dtype=input_dtype)

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.

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
                    images, target = batch
                    images = _move_to_device(images, device=device, input_dtype=input_dtype)
                    target = target.to(device, non_blocking=True)
                else:
                    images = dummy_images

                with autocast():
                    output = model(image=images)
                    image_features = output['image_features'] if isinstance(output, dict) else output[0]

                if is_rank0:
                    logits = 100. * image_features @ classifier
                    acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                    top1 += acc1
                    top5 += acc5
                    n += _image_batch_size(images)
        else:
            for images, target in tqdm(dataloader, unit_scale=args.batch_size):
                images = _move_to_device(images, device=device, input_dtype=input_dtype)
                target = target.to(device, non_blocking=True)

                with autocast():
                    # predict
                    output = model(image=images)
                    image_features = output['image_features'] if isinstance(output, dict) else output[0]
                    logits = 100. * image_features @ classifier

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += _image_batch_size(images)

    top1 = (top1 / n) if n else 0.
    top5 = (top5 / n) if n else 0.
    return top1, top5


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
