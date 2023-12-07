from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp


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


def _asarray(dtype, x):
    return jnp.asarray(x, dtype=dtype)


def inexact_asarray(x):
    dtype = jnp.result_type(x)
    return _asarray(dtype, x)
