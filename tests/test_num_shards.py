import pytest

from training.data import get_dataset_size

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
