from typing import Any

import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar  # pyright:ignore

from tests.bug_opers import IdentityLinearOperator


def preconditioner_and_y0(
    operator, vector: PyTree[Array], options: dict[str, Any] = None
):
    structure = operator.in_structure()
    preconditioner = IdentityLinearOperator(structure)
    y0 = jtu.tree_map(jnp.zeros_like, vector)
    return preconditioner, y0

def tree_where(
    pred: Bool[ArrayLike, ""], true: PyTree[ArrayLike], false: PyTree[ArrayLike]
) -> PyTree[Array]:
    keep = lambda a, b: jnp.where(pred, a, b)
    return jtu.tree_map(keep, true, false)


def max_norm(x: PyTree) -> Scalar:
    # a standard python max will fail when jax tracers are introduced.
    return jtu.tree_reduce(
        jnp.maximum,
        [jnp.max(jnp.abs(xi)) for xi in jtu.tree_leaves(x)],
    )


def resolve_rcond(rcond, n, m, dtype):
    return jnp.finfo(dtype).eps * max(n, m)
