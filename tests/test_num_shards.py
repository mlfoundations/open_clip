import json

import pytest

from open_clip_train.data import get_dataset_size

@pytest.mark.parametrize(
    "shards,expected_size",
    [
        ('/path/to/shard.tar', 1),
        ('/path/to/shard_{000..000}.tar', 1),
        ('/path/to/shard_{000..009}.tar', 10),
        ('/path/to/shard_{000..009}_{000..009}.tar', 100),
        ('/path/to/shard.tar::/path/to/other_shard_{000..009}.tar', 11),
        ('/path/to/shard_{000..009}.tar::/path/to/other_shard_{000..009}.tar', 20),
        (['/path/to/shard.tar'], 1),
        (['/path/to/shard.tar', '/path/to/other_shard.tar'], 2),
    ]
)
def test_num_shards(shards, expected_size):
    _, size = get_dataset_size(shards)
    assert size == expected_size, f'Expected {expected_size} for {shards} but found {size} instead.'


def _write_sizes(dir_path, sizes):
    dir_path.mkdir()
    (dir_path / 'sizes.json').write_text(json.dumps(sizes))


def test_dataset_size_reads_each_source_dir(tmp_path):
    # Two `::`-combined sources whose shards share basenames but have their own sizes.json.
    _write_sizes(tmp_path / 'a', {'00000.tar': 10, '00001.tar': 10})
    _write_sizes(tmp_path / 'b', {'00000.tar': 3, '00001.tar': 4})
    shards = f'{tmp_path}/a/{{00000..00001}}.tar::{tmp_path}/b/{{00000..00001}}.tar'
    assert get_dataset_size(shards) == (27, 4)


def test_dataset_size_sources_with_different_shard_names(tmp_path):
    _write_sizes(tmp_path / 'a', {'00000.tar': 10, '00001.tar': 10})
    _write_sizes(tmp_path / 'c', {'x0.tar': 5})
    shards = f'{tmp_path}/a/{{00000..00001}}.tar::{tmp_path}/c/x0.tar'
    assert get_dataset_size(shards) == (25, 3)


def test_dataset_size_mixed_len_files(tmp_path):
    _write_sizes(tmp_path / 'a', {'00000.tar': 10})
    (tmp_path / 'd').mkdir()
    (tmp_path / 'd' / '__len__').write_text('100')
    shards = f'{tmp_path}/a/00000.tar::{tmp_path}/d/00000.tar'
    assert get_dataset_size(shards) == (110, 2)


def test_dataset_size_unknown_when_any_source_lacks_info(tmp_path):
    _write_sizes(tmp_path / 'a', {'00000.tar': 10})
    (tmp_path / 'e').mkdir()
    shards = f'{tmp_path}/a/00000.tar::{tmp_path}/e/00000.tar'
    assert get_dataset_size(shards) == (None, 2)
