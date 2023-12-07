import math

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr

from tests.bug_opers import JacobianLinearOperator


def make_jac_operator(getkey, matrix, tags):
    out_size, in_size = matrix.shape
    x = jr.normal(getkey(), (in_size,), dtype=matrix.dtype)
    a = jr.normal(getkey(), (out_size,), dtype=matrix.dtype)
    b = jr.normal(getkey(), (out_size, in_size), dtype=matrix.dtype)
    c = jr.normal(getkey(), (out_size, in_size), dtype=matrix.dtype)
    fn_tmp = lambda x, _: a + b @ x + c @ x**2
    jac = jax.jacfwd(fn_tmp, holomorphic=jnp.iscomplexobj(x))(x, None)
    diff = matrix - jac
    fn = lambda x, _: a + (b + diff) @ x + c @ x**2
    return JacobianLinearOperator(fn, x, None, tags)


def _construct_matrix_impl(getkey, cond_cutoff, tags, size, dtype):
    while True:
        matrix = jr.normal(getkey(), (size, size), dtype=dtype)
        matrix = -matrix @ matrix.T.conj()
        jax.debug.print("matrix {matrix}", matrix=jnp.linalg.cond(matrix))
        if eqxi.unvmap_all(jnp.linalg.cond(matrix) < cond_cutoff):  # pyright: ignore
            break
    return matrix


def construct_matrix(getkey, solver, tags, num=1, *, size=3, dtype=jnp.float64):
    cond_cutoff = math.sqrt(1000)
    return tuple(
        _construct_matrix_impl(getkey, cond_cutoff, tags, size, dtype)
        for _ in range(num)
    )
