import random

import numpy as np
import torch


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def init_device(args):
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = "cuda:%d" % args.local_rank
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    args.device = device
    device = torch.device(device)
    return device
