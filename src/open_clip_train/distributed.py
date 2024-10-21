import os
import warnings
from typing import Optional

import torch
import torch.distributed as dist

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_device_available(device):
    device_type = torch.device(device).type
    is_avail = False
    is_known = False
    if device_type == 'cuda':
        is_avail = torch.cuda.is_available()
        is_known = True
    elif device_type == 'npu':
        # NOTE autoload device extension needed for this not to error out on this check
        is_avail = torch.npu.is_available()
        is_known = True
    elif device_type == 'mps':
        is_avail = torch.backends.mps.is_available()
        is_known = True
    elif device_type == 'cpu':
        is_avail = True
        is_known = True

    return is_avail, is_known


def set_device(device):
    if device.startswith('cuda:'):
        torch.cuda.set_device(device)
    elif device.startswith('npu:'):
        torch.npu.set_device(device)


def is_using_horovod():
    # NOTE w/ horovod run, OMPI vars should be set, but w/ SLURM PMI vars will be set
    # Differentiating between horovod and DDP use via SLURM may not be possible, so horovod arg still required...
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all([var in os.environ for var in pmi_vars]):
        return True
    else:
        return False


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    result = init_distributed_device_so(
        device=getattr(args, 'device', 'cuda'),
        dist_backend=getattr(args, 'dist_backend', None),
        dist_url=getattr(args, 'dist_url', None),
        horovod=getattr(args, 'horovod', False),
        no_set_device_rank=getattr(args, 'no_set_device_rank', False),
    )
    args.device = result['device']
    args.world_size = result['world_size']
    args.rank = result['global_rank']
    args.local_rank = result['local_rank']
    args.distributed = result['distributed']
    device = torch.device(args.device)
    return device


def init_distributed_device_so(
        device: str = 'cuda',
        dist_backend: Optional[str] = None,
        dist_url: Optional[str] = None,
        horovod: bool = False,
        no_set_device_rank: bool = False,
):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    distributed = False
    world_size = 1
    global_rank = 0
    local_rank = 0
    device_type, *device_idx = device.split(':', maxsplit=1)
    is_avail, is_known = is_device_available(device_type)
    if not is_known:
        warnings.warn(f"Device {device} was not known and checked for availability, trying anyways.")
    elif not is_avail:
        warnings.warn(f"Device {device} was not available, falling back to CPU.")
        device_type = device = 'cpu'

    if horovod:
        import horovod.torch as hvd
        assert hvd is not None, "Horovod is not installed"
        hvd.init()
        local_rank = int(hvd.local_rank())
        global_rank = hvd.rank()
        world_size = hvd.size()
        distributed = True
    elif is_using_distributed():
        if dist_backend is None:
            dist_backends = {
                "cuda": "nccl",
                "hpu": "hccl",
                "npu": "hccl",
                "xpu": "ccl",
            }
            dist_backend = dist_backends.get(device_type, 'gloo')

        dist_url = dist_url or 'env://'

        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            local_rank, global_rank, world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['RANK'] = str(global_rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=world_size,
                rank=global_rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
            )
            world_size = torch.distributed.get_world_size()
            global_rank = torch.distributed.get_rank()
        distributed = True

    if distributed and not no_set_device_rank and device_type not in ('cpu', 'mps'):
        # Ignore manually specified device index in distributed mode and
        # override with resolved local rank, fewer headaches in most setups.
        if device_idx:
            warnings.warn(f'device index {device_idx[0]} removed from specified ({device}).')
        device = f'{device_type}:{local_rank}'
        set_device(device)

    return dict(
        device=device,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
    )


def broadcast_object(args, obj, src=0):
    # broadcast a pickle-able python object from rank-0 to all ranks
    if args.horovod:
        return hvd.broadcast_object(obj, root_rank=src)
    else:
        if args.rank == src:
            objects = [obj]
        else:
            objects = [None]
        dist.broadcast_object_list(objects, src=src)
        return objects[0]


def all_gather_object(args, obj, dst=0):
    # gather a pickle-able python object across all ranks
    if args.horovod:
        return hvd.allgather_object(obj)
    else:
        objects = [None for _ in range(args.world_size)]
        dist.all_gather_object(objects, obj)
        return objects
