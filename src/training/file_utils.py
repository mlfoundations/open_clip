import logging
import os
import multiprocessing
import subprocess
import time
import fsspec
import torch

def sync_s3(local_dir, s3_dir):
    result = subprocess.run(["aws", "s3", "sync", local_dir, s3_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.info(f"Error: Failed to sync with S3 bucket")
        return False
        
    logging.info(f"Successfully synced with S3 bucket")
    return True

def keep_running_sync_s3(sync_every, local_dir, s3_dir):
    while True:
        time.sleep(sync_every)
        sync_s3(local_dir, s3_dir)

def start_sync_process(sync_every, local_dir, s3_dir):
    p = multiprocessing.Process(target=keep_running_sync_s3, args=(sync_every, local_dir, s3_dir))
    return p

# Note: we are not currently using this save function.
def pt_save(pt_obj, file_path):
    of = fsspec.open(file_path, "wb")
    with of as f:
        torch.save(pt_obj, file_path)

def pt_load(file_path, map_location=None):
    if file_path.startswith('s3'):
        logging.info('Loading checkpoint from S3, which may take a bit.')
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f)
    return out

def check_exists(file_path):
    if file_path.startswith('s3'):
        result = subprocess.run(["aws", "s3", "ls", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    else:
        return os.path.isfile(file_path)