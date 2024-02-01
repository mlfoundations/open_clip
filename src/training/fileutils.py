import logging
import os
import multiprocessing
import subprocess
import time
import fsspec
import torch
from tqdm import tqdm

def remote_sync_s3(local_dir, remote_dir):
    # skip epoch_latest which can change during sync.
    result = subprocess.run(["aws", "s3", "sync", local_dir, remote_dir, '--exclude', '*epoch_latest.pt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error(f"Error: Failed to sync with S3 bucket {result.stderr.decode('utf-8')}")
        return False
        
    logging.info(f"Successfully synced with S3 bucket")
    return True

def remote_sync_fsspec(local_dir, remote_dir):
    # FIXME currently this is slow and not recommended. Look into speeding up.
    a = fsspec.get_mapper(local_dir)
    b = fsspec.get_mapper(remote_dir)

    for k in a:
        # skip epoch_latest which can change during sync.
        if 'epoch_latest.pt' in k:
            continue

        logging.info(f'Attempting to sync {k}')
        if k in b and len(a[k]) == len(b[k]):
            logging.debug(f'Skipping remote sync for {k}.')
            continue

        try:
            logging.info(f'Successful sync for {k}.')
            b[k] = a[k]
        except Exception as e:
            logging.info(f'Error during remote sync for {k}: {e}')
            return False

    return True

def remote_sync(local_dir, remote_dir, protocol):
    logging.info('Starting remote sync.')
    if protocol == 's3':
        return remote_sync_s3(local_dir, remote_dir)
    elif protocol == 'fsspec':
        return remote_sync_fsspec(local_dir, remote_dir)
    else:
        logging.error('Remote protocol not known')
        return False

def keep_running_remote_sync(sync_every, local_dir, remote_dir, protocol):
    while True:
        time.sleep(sync_every)
        remote_sync(local_dir, remote_dir, protocol)

def start_sync_process(sync_every, local_dir, remote_dir, protocol):
    p = multiprocessing.Process(target=keep_running_remote_sync, args=(sync_every, local_dir, remote_dir, protocol))
    return p

# Note: we are not currently using this save function.
def pt_save(pt_obj, file_path):
    of = fsspec.open(file_path, "wb")
    with of as f:
        torch.save(pt_obj, file_path)

def pt_load(file_path, map_location=None):
    if file_path.startswith('s3'):
        logging.info('Loading remote checkpoint, which may take a bit.')
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out

def check_exists(file_path):
    try:
        with fsspec.open(file_path):
            pass
    except FileNotFoundError:
        return False
    return True
