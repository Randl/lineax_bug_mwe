from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.core
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar  # pyright:ignore


class NoneAux(eqx.Module):
    fn: Callable

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs), None


class _NoAuxIn(eqx.Module):
    fn: Callable
    args: Any

    def __call__(self, x):
        return self.fn(x, self.args)


class _NoAuxOut(eqx.Module):
    fn: Callable

    def __call__(self, x):
        f, _ = self.fn(x)
        return f


class _Unwrap(eqx.Module):
    fn: Callable

    def __call__(self, x):
        (f,) = self.fn(x)
        return f

def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def _asarray(dtype, x):
    return jnp.asarray(x, dtype=dtype)


def inexact_asarray(x):
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        dtype = default_floating_dtype()
    return _asarray(dtype, x)


def jacobian(fn, in_size, out_size, holomorphic=False, has_aux=False):
    # Heuristic for which is better in each case
    # These could probably be tuned a lot more.
    if (in_size < 100) or (in_size <= 1.5 * out_size):
        return jax.jacfwd(fn, holomorphic=holomorphic, has_aux=has_aux)
    else:
        return jax.jacrev(fn, holomorphic=holomorphic, has_aux=has_aux)

