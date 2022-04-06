from functools import partial

import jax
import jax.numpy as jnp

from .functional import chunk_encode, cache_grad, unchunk_args


def cache_train_step(loss_fn, state, ss, tt, axis='device'):
    def encode_with_params(params, **kwargs):
        return state.apply_fn(params=params, **kwargs)

    encode_fn = chunk_encode(partial(encode_with_params, state.params))
    grad_fn = cache_grad(encode_with_params)

    s_reps = encode_fn(**ss)
    t_reps = encode_fn(**tt)

    @unchunk_args(axis=0, argnums=(0, 1))
    def grad_cache_fn(xx, yy):
        return jnp.mean(loss_fn(xx, yy, axis=axis))
    loss, (s_grads, t_grads) = jax.value_and_grad(grad_cache_fn, argnums=(0, 1))(s_reps, t_reps)

    grads = jax.tree_map(lambda v: jnp.zeros_like(v), state.params)
    grads = grad_fn(state.params, grads, s_grads, **ss)
    grads = grad_fn(state.params, grads, t_grads, **tt)

    loss, grads = jax.lax.pmean([loss, grads], axis)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state
