import argparse
import logging
from logging import Filter
from logging.handlers import QueueHandler, QueueListener

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue


def setup_primary_logging(log_file, level):
    log_queue = Queue(-1)

    file_handler = logging.FileHandler(filename=log_file)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', 
        datefmt='%Y-%m-%d,%H:%M:%S')

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    file_handler.setLevel(level)
    stream_handler.setLevel(level)

    listener = QueueListener(log_queue, file_handler, stream_handler)

    listener.start()

    return log_queue


class WorkerLogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"Rank {self._rank} | {record.msg}"
        return True


def setup_worker_logging(rank, log_queue, level):
    queue_handler = QueueHandler(log_queue)

    worker_filter = WorkerLogFilter(rank)
    queue_handler.addFilter(worker_filter)

    queue_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)

    root_logger.setLevel(level)


def fake_worker(rank: int, world_size: int, log_queue: Queue):
    setup_worker_logging(rank, log_queue, logging.DEBUG)
    logging.info("Test worker log")
    logging.error("Test worker error log")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:6100',
        world_size=world_size,
        rank=rank,
    )

if __name__ == "__main__":
    # Set multiprocessing type to spawn
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu-list", type=int, help="List of GPU IDs", nargs="+", required=True)

    args = parser.parse_args()

    world_size = len(args.gpu_list)

    # Initialize the primary logging handlers. Use the returned `log_queue`
    # to which the worker processes would use to push their messages
    log_queue = setup_primary_logging("/usr/lusers/gamaga/out.log", logging.DEBUG)

    if world_size == 1:
        worker(0, world_size, log_queue)
    else:
        mp.spawn(fake_worker, args=(world_size, log_queue), nprocs=world_size)