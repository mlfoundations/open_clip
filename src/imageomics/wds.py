import multiprocessing

import webdataset


class ShardWriter:
    """Like TarWriter but splits into multiple shards."""

    def __init__(
        self,
        pattern: str,
        shard_counter: multiprocessing.Value,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post=None,
        start_shard: int = 0,
        verbose=True,
        **kw,
    ):
        """Create a ShardWriter.

        :param pattern: output file pattern
        :param maxcount: maximum number of records per shard (Default value = 100000)
        :param maxsize: maximum size of each shard (Default value = 3e9)
        :param kw: other options passed to TarWriter
        """
        self.verbose = verbose
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.post = post

        self.tarstream = None
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None

        self.shard_counter = shard_counter

        self.next_stream()

    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()

        with self.shard_counter.get_lock():
            self.shard = self.shard_counter.value
            self.shard_counter.value += 1

        self.fname = self.pattern % self.shard
        if self.verbose:
            print(
                "# writing",
                self.fname,
                self.count,
                "%.1f GB" % (self.size / 1e9),
                self.total,
            )
        stream = open(self.fname, "wb")
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
            assert self.fname is not None
            if callable(self.post):
                self.post(self.fname)
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
