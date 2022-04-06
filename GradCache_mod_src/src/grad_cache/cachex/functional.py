from typing import Iterable, Any
from functools import partial

import jax
import jax.numpy as jnp

from .tree_utils import tree_unchunk

Array = Any


def grad_with_cache(f, **grad_kwargs):
    def cache_f(params, cache, *args, **kwargs):
        return jnp.sum(f(params, *args, **kwargs) * cache)
    return jax.grad(cache_f, **grad_kwargs)


def encode_scan_fn(f, carry, x):
    return carry, f(**x)


def cache_grad_scan_fn(f, params, acc, x):
    cached_grad, kwargs = x

    def fwd_fn(w):
        return f(params=w, **kwargs)

    chunk_grad = grad_with_cache(fwd_fn)(params, cached_grad)
    acc = jax.tree_multimap(lambda u, v: u + v, acc, chunk_grad)
    return acc, None


def chunk_encode(encode_fn):
    def f(**xx):
        _, hh = jax.lax.scan(partial(encode_scan_fn, encode_fn), 0, xx)
        return hh
    return f


def cache_grad(encode_fn):
    def f(params, grad_accumulator, cached_grad, **xx):
        grads, _ = jax.lax.scan(
            partial(cache_grad_scan_fn, encode_fn, params), grad_accumulator, [cached_grad, xx]
        )
        return grads
    return f


def unchunk_args(axis: int = 0, argnums: Iterable[int] = ()):
    def decorator_unchunk(f):
        def g(*args, **kwargs):
            new_args = list(args)
            for i in argnums:
                new_args[i] = tree_unchunk(args[i], axis)
            return f(*new_args, **kwargs)

        return g

    return decorator_unchunk
