import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from open_clip import create_model_and_transforms, trace_model
from .data import get_data
from .device import is_master, init_device, world_info_from_env
from .evaluate import evaluate
from .logger import setup_logging
from .loss import LossCfg
from .optim import OptimCfg
from .params import parse_args
from .scheduler import cosine_lr
from .train import train_one_epoch
from .train_jig import TrainJig

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    args = parse_args()

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize device + distributed environment
    assert args.precision in ['amp', 'fp16', 'fp32']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    dev_env = init_device(args)
    if dev_env.cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if dev_env.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {dev_env.device}.'
            f'Process (global: {dev_env.rank}, local {dev_env.local_rank}), total {dev_env.world_size}.')
    elif dev_env.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {dev_env.device}.'
            f'Process (global: {dev_env.rank}, local {dev_env.local_rank}), total {dev_env.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {dev_env.device}.')

    # setup logging services
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if dev_env.is_master():
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    if args.copy_codebase:
        copy_codebase(args)

    # create model, load pretrained checkpoints
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=dev_env.device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
    )

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=dev_env.device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)

    if args.grad_cache_chunk_size:
        assert args.batch_size % args.grad_cache_chunk_size == 0,\
            'Gradient caching batch size must be divisible by chunk size'
        if args.val_batch_size is None:
            # set batch size for evaluation to smaller chunk size if not already set
            args.val_batch_size = args.grad_cache_chunk_size

    if dev_env.is_master():
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
        if args.grad_cache_chunk_size:
            logging.info(
                f'Enabling gradient caching with chunk_size: {args.grad_cache_chunk_size}, '
                f'batch_size: {args.batch_size}, val_batch_size: {args.val_batch_size}.')

    is_training = args.train_data is not None
    start_epoch = 0
    train_jig = None
    if is_training:
        # train specific setup
        loss_cfg = LossCfg(
            type='clip',  # TODO support other CLIP-like image-text losses
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad)

        optim_cfg = OptimCfg(
            type='adamw',   # TODO support other optimizers
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

        train_jig = TrainJig(
            model=model,
            dev_env=dev_env,
            loss_cfg=loss_cfg,
            optim_cfg=optim_cfg,
            grad_cache_chunk_size=args.grad_cache_chunk_size,
        )
        if args.resume is not None:
            train_jig.resume(args.resume)
        start_epoch = train_jig.epoch  # get_data needs epoch for wds.detshuffle seeding if we are training

    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch)
    assert len(data), 'At least one train or eval dataset must be specified.'

    if not is_training:
        # evaluate only, specify checkpoint via --checkpoint arg, not --resume arg
        # TODO possibly update evaluate to use jig / loss cfg?
        evaluate(model, data, start_epoch, args)
        return

    assert 'train' in data
    assert train_jig is not None
    assert not args.trace, "cannot train with traced model, please disable tracing"
    total_steps = data["train"].dataloader.num_batches * args.epochs
    scheduler = cosine_lr(train_jig.optimizer, args.lr, args.warmup, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    tb_writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        tb_writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="open-clip",
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    for epoch in range(start_epoch, args.epochs):
        if dev_env.is_master():
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(train_jig, data, scheduler, args, tb_writer)
        completed_epoch = epoch + 1
        assert completed_epoch == train_jig.epoch

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            eval_metrics = evaluate(train_jig.model, data, completed_epoch, args)

            if args.save_logs:
                for name, val in eval_metrics.items():
                    if tb_writer is not None:
                        tb_writer.add_scalar(f"val/{name}", val, epoch)

                with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
                    f.write(json.dumps(eval_metrics))
                    f.write("\n")

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                for name, val in eval_metrics.items():
                    wandb.log({f"val/{name}": val, 'epoch': epoch})

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = train_jig.state_dict(name=args.name)
            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main()
