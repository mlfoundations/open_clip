import logging
import multiprocessing
import os
import re

import webdataset


class ShardWriter:
    """Like TarWriter but splits into multiple shards."""

    def __init__(
        self,
        outdir: str,
        next_shard: multiprocessing.Value,
        *,
        digits: int = 6,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        no_init: bool = False,
        **kw,
    ):
        """Create a ShardWriter.

        :param pattern: output file pattern
        :param maxcount: maximum number of records per shard (Default value = 100000)
        :param maxsize: maximum size of each shard (Default value = 3e9)
        :param kw: other options passed to TarWriter
        """
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize

        self.tarstream = None
        self.outdir = outdir
        self.pattern = f"shard-%0{digits}d.tar"
        self.parse_pattern = re.compile(f"shard-(\\d{{{digits}}})\\.tar")
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None

        self.next_shard = next_shard
        self.logger = logging.getLogger(f"shard-writer(p{os.getpid()})")

        if not no_init:
            self.next_stream()

    def choose_next_shard(self, existing):
        if not existing:
            self.shard = 0
            self.next_shard.value = self.shard + 1
            return

        existing = set(existing)
        for i in range(max(existing)):
            if i < self.next_shard.value:
                continue

            if i in existing:
                continue

            self.shard = i
            self.next_shard.value = self.shard + 1
            return

        self.shard = max(self.next_shard.value, max(existing) + 1)
        self.next_shard.value = self.shard + 1

    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()

        with self.next_shard.get_lock():
            # Get a list of existing files
            existing = []
            for file in os.listdir(self.outdir):
                match = self.parse_pattern.match(file)
                if match is None:
                    continue
                num = int(match.group(1), base=10)
                existing.append(num)
            existing = sorted(existing)

            # Choose next shard
            self.choose_next_shard(existing)

        self.fname = os.path.join(self.outdir, self.pattern % self.shard)
        stream = open(self.fname, "wb")
        self.logger.info("Opened shard %s.", self.fname)
        self.tarstream = webdataset.TarWriter(stream, **self.kw)
        self.count = 0
        self.size = 0

    def write(self, obj):
        """Write a sample.

        :param obj: sample to be written
        """
        if (
            self.tarstream is None
            or self.count >= self.maxcount
            or self.size >= self.maxsize
        ):
            self.next_stream()
        size = self.tarstream.write(obj)
        self.count += 1
        self.total += 1
        self.size += size

    def finish(self):
        """Finish all writing (use close instead)."""
        if self.tarstream is not None:
            self.tarstream.close()
            self.logger.info("Closed shard %s.", self.fname)
            assert self.fname is not None
            self.tarstream = None

    def close(self):
        """Close the stream."""
        self.finish()
        del self.tarstream
        del self.shard
        del self.count
        del self.size

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, *args, **kw):
        """Exit context."""
        self.close()
