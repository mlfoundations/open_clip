from typing import Any

import jax


def tree_chunk(tree: Any, n_chunk: int, axis: int = 0) -> Any:
    return jax.tree_map(
        lambda v: v.reshape(v.shape[:axis] + (n_chunk, -1) + v.shape[axis + 1:]),
        tree
    )


def tree_unchunk(tree: Any, axis: int = 0) -> Any:
    return jax.tree_map(
        lambda x: x.reshape(x.shape[:axis] + (-1,) + x.shape[axis + 2:]),
        tree
    )
