import glob
import logging
import multiprocessing
import os
import re
import subprocess
import time
import fsspec
import torch

_logger = logging.getLogger(__name__)

def remote_sync_s3(local_dir, remote_dir):
    # skip epoch_latest which can change during sync.
    result = subprocess.run(["aws", "s3", "sync", local_dir, remote_dir, '--exclude', '*epoch_latest.pt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        _logger.error(f"Error: Failed to sync with S3 bucket {result.stderr.decode('utf-8')}")
        return False
        
    _logger.info(f"Successfully synced with S3 bucket")
    return True

def remote_sync_fsspec(local_dir, remote_dir):
    # FIXME currently this is slow and not recommended. Look into speeding up.
    a = fsspec.get_mapper(local_dir)
    b = fsspec.get_mapper(remote_dir)

    for k in a:
        # skip epoch_latest which can change during sync.
        if 'epoch_latest.pt' in k:
            continue

        _logger.info(f'Attempting to sync {k}')
        if k in b and len(a[k]) == len(b[k]):
            _logger.debug(f'Skipping remote sync for {k}.')
            continue

        try:
            _logger.info(f'Successful sync for {k}.')
            b[k] = a[k]
        except Exception as e:
            _logger.info(f'Error during remote sync for {k}: {e}')
            return False

    return True

def remote_sync(local_dir, remote_dir, protocol):
    _logger.info('Starting remote sync.')
    if protocol == 's3':
        return remote_sync_s3(local_dir, remote_dir)
    elif protocol == 'fsspec':
        return remote_sync_fsspec(local_dir, remote_dir)
    else:
        _logger.error('Remote protocol not known')
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
    """Load a checkpoint via fsspec. Uses ``weights_only=True``.

    If a checkpoint contains optimizer state with custom (non-allowlisted)
    types, the caller must register them via
    ``torch.serialization.add_safe_globals([...])`` before calling.
    """
    if file_path.startswith('s3'):
        _logger.info('Loading remote checkpoint, which may take a bit.')
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location, weights_only=True)
    return out

def check_exists(file_path):
    try:
        with fsspec.open(file_path):
            pass
    except FileNotFoundError:
        return False
    return True


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool, *, include_sharded: bool = True):
    # The glob recurses, so it can pick up checkpoints across multiple subfolders.
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _logger.debug("Remote checkpoint listing: %s", result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
        # Also find DCP checkpoint dirs (contain .metadata file from DCP)
        if include_sharded:
            for d in glob.glob(os.path.join(path, 'epoch_*')):
                if os.path.isdir(d) and os.path.exists(os.path.join(d, '.metadata')):
                    checkpoints.append(d)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


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
