import os
import time
import logging
from time import gmtime, strftime
from datetime import datetime
from pathlib import Path
import json

import wandb
import torch
from torch import optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

try:
    import horovod.torch as hvd
except ImportError:
    print("Horovod not installed")

from clip.openai_clip import _transform, load
from clip.model import convert_weights_to_fp16, CLIP
from training.train import train_one_epoch, evaluate
from training.data import get_data
from training.params import parse_args
from training.logger import setup_logging
from training.scheduler import cosine_lr


def is_master(args):
    return args.local_rank == 0


def is_using_horovod():
    # FIXME will this be distinct for horovod or overlap with other uses of Slurm (ie slurm + DDP)?
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all([var in os.environ for var in pmi_vars]):
        return True
    else:
        return False


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    return False


def main():
    args = parse_args()

    # Distributed training = training on more than one GPU.
    # Also easily possible to extend to multiple nodes & multiple GPUs.
    args.distributed = False
    args.dp = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.horovod:
        hvd.init()
        local_rank = int(hvd.local_rank())
        args.world_size = hvd.size()
        args.rank = hvd.rank()
        args.distributed = True
    elif is_using_distributed():
        if 'SLURM_PROCID' in os.environ:
            # FIXME this needs debugging
            args.rank = int(os.environ['SLURM_PROCID'])
            os.environ['RANK'] = os.environ['SLURM_PROCID']
            os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
            args.world_size = int(os.environ['WORLD_SIZE'])
            for k, v in os.environ.items():
                if 'SLURM' in k:
                    print(k, v)
        assert 'LOCAL_RANK' in os.environ  # current torch uses env var only for passing args to workers
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('DDP init', args.world_size, args.rank, args.local_rank)  # FIXME debug
        args.distributed = True
    else:
        args.world_size = 1  # DP is still a world-size of 1
        local_rank = 0
        args.multigpu = []
        if args.dp and not args.multigpu:
            args.multigpu = list(range(torch.cuda.device_count()))

    args.local_rank = local_rank
    if torch.cuda.is_available():
        device = 'cuda'
        if args.distributed:
            device += ':%d' % local_rank
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)

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

    # Log and save params.
    args.log_path = os.path.join(args.logs, args.name, f"out-{local_rank}.log")
    if os.path.exists(args.log_path):
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    if args.copy_codebase:
        copy_codebase(args)

    assert args.precision in ['amp', 'fp16', 'fp32']
    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {local_rank}), total {args.world_size}.')
    else:
        if args.dp:
            logging.info(f'Running with a single process, DataParallel on {len(args.multigpu)} GPUs.')
        else:
            logging.info(f'Running with a single process. Device {args.device}.')

    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # Do not use skip_reset unless you want to use on of the CLIP model
    if args.openai_pretrained:
        model, preprocess_train, preprocess_val = load(
            args.model,
            device=args.device,
            jit=False,
            is_train=True)
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if args.precision == "amp" or args.precision == "fp32":
            model = model.float()
    else:
        model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
        print('Loading model from', model_config_file)
        assert os.path.exists(model_config_file)
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        model = CLIP(**model_info)
        preprocess_train = _transform(model.visual.image_size, is_train=True)
        preprocess_val = _transform(model.visual.image_size, is_train=False)

        model.to(device=device)
        if args.precision == "fp16":
            convert_weights_to_fp16(model)

        print(model)

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    if args.dp:
        model = torch.nn.DataParallel(model, device_ids=args.multigpu)

    data = get_data(args, (preprocess_train, preprocess_val))
    assert 'train' in data or 'val' in data, 'At least one of train or val datasets must be specified'

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if 'cuda' in args.device:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume, map_location=device)
            else:
                checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False

    # determine if this worker should save logs and checkpoints.
    # only do so if it is the 0th worker.
    args.save_logs = args.logs and args.logs.lower() != 'none' and args.local_rank == 0
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
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

    if 'train' not in data:
        evaluate(model, data, start_epoch, args, writer)
        return
    elif start_epoch == 0 and 'val' in data:
        evaluate(model, data, 0, args, writer)

    for epoch in range(start_epoch, args.epochs):
        if args.local_rank == 0:
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        completed_epoch = epoch + 1

        if any([v in data for v in ('val', 'imagenet-val', 'imagenet-v2')]):
            evaluate(model, data, completed_epoch, args, writer)

        # Saving checkpoints.
        if args.save_logs:
            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    {
                        "epoch": completed_epoch,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    {
                        "epoch": completed_epoch,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )

    if args.wandb and args.local_rank == 0:
        wandb.finish()


def copy_codebase(args):
    import sys
    import subprocess
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
    # os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{os.path.join(new_code_path, 'src')}"
    # main_file = os.path.join(new_code_path, "src", "training", "main.py")
    # argv = sys.argv
    # argv.remove('--copy-codebase')
    # argv.extend(['--name', args.name])
    # command = [sys.executable] + argv
    # print("Executing command:", " ".join(command))
    # subprocess.check_call(command)
    return 1


if __name__ == "__main__":
    main()
