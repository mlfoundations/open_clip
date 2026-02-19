import logging

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
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args, use_fsdp_eval=False):
    device = torch.device(args.device)
    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, 'fsdp', False),
    )
    input_dtype = get_input_dtype(args.precision)
    is_rank0 = (args.rank == 0)

    if use_fsdp_eval and not is_rank0:
        # Pre-allocate dummy image tensor for non-master ranks
        image_size = model.visual.image_size
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
                    images = images.to(device=device, dtype=input_dtype)
                    target = target.to(device)
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
                    n += images.size(0)
        else:
            for images, target in tqdm(dataloader, unit_scale=args.batch_size):
                images = images.to(device=device, dtype=input_dtype)
                target = target.to(device)

                with autocast():
                    # predict
                    output = model(image=images)
                    image_features = output['image_features'] if isinstance(output, dict) else output[0]
                    logits = 100. * image_features @ classifier

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

    top1 = (top1 / n) if n else 0.
    top5 = (top5 / n) if n else 0.
    return top1, top5


def zero_shot_eval(model_or_task, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    use_fsdp_eval = getattr(args, 'fsdp', False) and getattr(args, 'distributed', False)
    is_rank0 = (args.rank == 0)

    model = get_model_from_task(model_or_task)

    if is_rank0:
        logging.info('Starting zero-shot imagenet.')

    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    if is_rank0:
        logging.info('Building zero-shot classifier')

    device = torch.device(args.device)
    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, 'fsdp', False),
    )

    # All ranks must call encode_text() for FSDP collective ops.
    # build_zero_shot_classifier is deterministic — same number of forward calls on all ranks.
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=device,
            use_tqdm=is_rank0,
        )

    if is_rank0:
        logging.info('Using classifier')

    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(
            model, classifier, data['imagenet-val'].dataloader, args,
            use_fsdp_eval=use_fsdp_eval,
        )
        if is_rank0:
            results['imagenet-zeroshot-val-top1'] = top1
            results['imagenet-zeroshot-val-top5'] = top5

    if 'imagenet-v2' in data:
        top1, top5 = run(
            model, classifier, data['imagenet-v2'].dataloader, args,
            use_fsdp_eval=use_fsdp_eval,
        )
        if is_rank0:
            results['imagenetv2-zeroshot-val-top1'] = top1
            results['imagenetv2-zeroshot-val-top5'] = top5

    if is_rank0:
        logging.info('Finished zero-shot imagenet.')

    return results
