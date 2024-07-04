import os
import pytest
import util_test
import collections
import tarfile
import io
from PIL import Image

from open_clip_train.data import get_wds_dataset
from open_clip_train.params import parse_args
from open_clip_train.main import random_seed

TRAIN_NUM_SAMPLES = 10_000
RTOL = 0.2

# NOTE: we use two test tar files, which are created on the fly and saved to data/input.
# 000.tar has 10 samples, and the captions are 000_0, 000_1, ..., 000_9
# 001.tar has 5 samples, and the captions are 001_0, 001_1, ..., 001_4
def build_inputs(test_name):
    base_input_dir, _ = util_test.get_data_dirs()
    input_dir = os.path.join(base_input_dir, test_name)
    os.makedirs(input_dir, exist_ok=True)
    
    def save_tar(idx, num_samples):
        filename = os.path.join(input_dir, f'test_data_{idx:03d}.tar')
        tar = tarfile.open(filename, 'w')
        
        for sample_idx in range(num_samples):
            # Image
            image = Image.new('RGB', (32, 32))
            info = tarfile.TarInfo(f'{sample_idx}.png')
            bio = io.BytesIO()
            image.save(bio, format='png')
            size = bio.tell()
            bio.seek(0)
            info.size = size
            tar.addfile(info, bio)
            
            # Caption
            info = tarfile.TarInfo(f'{sample_idx}.txt')
            bio = io.BytesIO()
            bio.write(f'{idx:03d}_{sample_idx}'.encode('utf-8'))
            size = bio.tell()
            bio.seek(0)
            info.size = size
            tar.addfile(info, bio)
        
        tar.close()          

    save_tar(0, 10)
    save_tar(1, 5)

    return input_dir


def build_params(input_shards, seed=0):
    args = parse_args([])
    args.train_data = input_shards
    args.train_num_samples = TRAIN_NUM_SAMPLES
    args.dataset_resampled = True
    args.seed = seed
    args.workers = 1
    args.world_size = 1
    args.batch_size = 1
    random_seed(seed)

    preprocess_img = lambda x: x
    tokenizer = lambda x: [x.strip()]

    return args, preprocess_img, tokenizer


def get_dataloader(input_shards):
    args, preprocess_img, tokenizer = build_params(input_shards)
    dataset = get_wds_dataset(args, preprocess_img, is_train=True, tokenizer=tokenizer)
    dataloader = dataset.dataloader
    return dataloader


def test_single_source():
    """Test webdataset with a single tar file."""
    input_dir = build_inputs('single_source')    
    input_shards = os.path.join(input_dir, 'test_data_000.tar')
    dataloader = get_dataloader(input_shards)
    
    counts = collections.defaultdict(int)
    for sample in dataloader:
        txts = sample[1]
        for txt in txts:
            counts[txt] += 1
    
    for key, count in counts.items():
        assert count == pytest.approx(TRAIN_NUM_SAMPLES / 10, RTOL)


def test_two_sources():
    """Test webdataset with a single two tar files."""
    input_dir = build_inputs('two_sources')
    input_shards = os.path.join(input_dir, 'test_data_{000..001}.tar')
    dataloader = get_dataloader(input_shards)

    counts = collections.defaultdict(int)
    for sample in dataloader:
        txts = sample[1]
        for txt in txts:
            counts[txt] += 1
    
    for key, count in counts.items():
        assert count == pytest.approx(TRAIN_NUM_SAMPLES / 15, RTOL), f'{key}, {count}'


def test_two_sources_same_weights():
    """Test webdataset with a two tar files, using --train-data-weights=1::1."""
    input_dir = build_inputs('two_sources_same_weights')
    input_shards = f"{os.path.join(input_dir, 'test_data_000.tar')}::{os.path.join(input_dir, 'test_data_001.tar')}"
    args, preprocess_img, tokenizer = build_params(input_shards)
    args.train_data_upsampling_factors = '1::1'
    dataset = get_wds_dataset(args, preprocess_img, is_train=True, tokenizer=tokenizer)
    dataloader = dataset.dataloader

    counts = collections.defaultdict(int)
    for sample in dataloader:
        txts = sample[1]
        for txt in txts:
            counts[txt] += 1
    
    for key, count in counts.items():
        assert count == pytest.approx(TRAIN_NUM_SAMPLES / 15, RTOL), f'{key}, {count}'

def test_two_sources_with_upsampling():
    """Test webdataset with a two tar files with upsampling."""
    input_dir = build_inputs('two_sources_with_upsampling')
    input_shards = f"{os.path.join(input_dir, 'test_data_000.tar')}::{os.path.join(input_dir, 'test_data_001.tar')}"
    args, preprocess_img, tokenizer = build_params(input_shards)
    args.train_data_upsampling_factors = '1::2'
    dataset = get_wds_dataset(args, preprocess_img, is_train=True, tokenizer=tokenizer)
    dataloader = dataset.dataloader

    counts = collections.defaultdict(int)
    for sample in dataloader:
        txts = sample[1]
        for txt in txts:
            counts[txt] += 1
    
    for key, count in counts.items():
        if key.startswith('000'):
            assert count == pytest.approx(TRAIN_NUM_SAMPLES / 20, RTOL), f'{key}, {count}'
        else:
            assert count == pytest.approx(TRAIN_NUM_SAMPLES / 10, RTOL), f'{key}, {count}'
