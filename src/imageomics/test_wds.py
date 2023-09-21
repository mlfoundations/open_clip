import multiprocessing
import tempfile

from . import wds


def test_choose_next_shard_no_missing():
    existing = [0, 1, 2]
    init_counter = 0
    expected_seq = [3, 4]

    counter = multiprocessing.Value("I", init_counter, lock=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        for expected in expected_seq:
            writer = wds.ShardWriter(tmpdir, counter, no_init=True)
            with counter.get_lock():
                writer.choose_next_shard(existing)

            assert writer.shard == expected


def test_choose_next_shard_one_missing():
    existing = [0, 1, 2, 4, 5]
    init_counter = 0
    expected_first = 3
    expected_second = 6

    counter = multiprocessing.Value("I", init_counter, lock=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = wds.ShardWriter(tmpdir, counter, no_init=True)
        with counter.get_lock():
            writer.choose_next_shard(existing)

        assert writer.shard == expected_first

        writer = wds.ShardWriter(tmpdir, counter, no_init=True)
        with counter.get_lock():
            writer.choose_next_shard(existing)

        assert writer.shard == expected_second


def test_choose_next_shard_two_missing():
    existing = [0, 1, 2, 5]
    init_counter = 0
    expected_seq = [3, 4, 6]

    counter = multiprocessing.Value("I", init_counter, lock=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        for expected in expected_seq:
            writer = wds.ShardWriter(tmpdir, counter, no_init=True)
            with counter.get_lock():
                writer.choose_next_shard(existing)
            print(expected, writer.shard, writer.next_shard.value)
            assert writer.shard == expected


def test_choose_next_shard_two_missing_non_sequential():
    existing = [0, 1, 2, 4, 6]
    init_counter = 0
    expected_seq = [3, 5, 7]

    counter = multiprocessing.Value("I", init_counter, lock=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        for expected in expected_seq:
            writer = wds.ShardWriter(tmpdir, counter, no_init=True)
            with counter.get_lock():
                writer.choose_next_shard(existing)

            assert writer.shard == expected
