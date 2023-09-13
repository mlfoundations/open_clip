import concurrent.futures
import typing

from tqdm.auto import tqdm

T = typing.TypeVar("T")


class BoundedExecutor:
    def __init__(self, pool_cls=concurrent.futures.ThreadPoolExecutor, **kwargs):
        self._pool = pool_cls(**kwargs)
        self._futures = []

    def submit(self, *args, **kwargs):
        self._futures.append(self._pool.submit(*args, **kwargs))

    def shutdown(self, **kwargs):
        self._pool.shutdown(wait=False, cancel_futures=True, **kwargs)

    def finish(self, *, desc: str = ""):
        return [
            future.result()
            for future in tqdm(
                concurrent.futures.as_completed(self._futures),
                total=len(self._futures),
                desc=desc,
            )
        ]


def finish_all(futures: list[concurrent.futures.Future[T]]) -> list[T]:
    return [
        future.result()
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures))
    ]
