import functools as ft
from typing import Any
from typing import Optional, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import ArrayLike, PyTree

from tests.bug_oper_misc import inexact_asarray
from tests.bug_opers import (
    linearise,
    TangentLinearOperator,
)


#
# _linear_solve_p
#


def _to_shapedarray(x):
    return jax.core.ShapedArray(x.shape, x.dtype)


def _to_struct(x):
    if isinstance(x, jax.core.ShapedArray):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)
    else:
        return x


def _is_none(x):
    return x is None


def _sum(*args):
    return sum(args)


def _linear_solve_impl(_, state, vector, options, solver, throw, *, check_closure):
    out = solver.compute(state, vector, options)
    solution, result, stats = out
    return solution


@eqxi.filter_primitive_def
def _linear_solve_abstract_eval(operator, state, vector, options, solver, throw):
    state, vector, options, solver = jtu.tree_map(
        _to_struct, (state, vector, options, solver)
    )
    out = eqx.filter_eval_shape(
        _linear_solve_impl,
        operator,
        state,
        vector,
        options,
        solver,
        throw,
        check_closure=False,
    )
    out = jtu.tree_map(_to_shapedarray, out)
    return out


@eqxi.filter_primitive_jvp
def _linear_solve_jvp(primals, tangents):
    operator, state, vector, options, solver, throw = primals
    t_operator, t_state, t_vector, t_options, t_solver, t_throw = tangents
    del t_state, t_options, t_solver, t_throw

    # Note that we pass throw=True unconditionally to all the tangent solves, as there
    # is nowhere we can pipe their error to.
    # This is the primal solve so we can respect the original `throw`.
    solution = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vector, options, solver, throw
    )

    vecs = []
    sols = []
    if any(t is not None for t in jtu.tree_leaves(t_vector, is_leaf=_is_none)):
        # b' term
        vecs.append(
            jtu.tree_map(eqxi.materialise_zeros, vector, t_vector, is_leaf=_is_none)
        )
    if any(t is not None for t in jtu.tree_leaves(t_operator, is_leaf=_is_none)):
        t_operator = TangentLinearOperator(operator, t_operator)
        t_operator = linearise(t_operator)  # optimise for matvecs
        # -A'x term
        vec = (-t_operator.mv(solution) ** ω).ω
        vecs.append(vec)

    vecs = jtu.tree_map(_sum, *vecs)
    # the A^ term at the very beginning
    sol = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vecs, options, solver, True
    )
    return solution, sol


# Call with `check_closure=False` so that the autocreated vmap rule works.
linear_solve_p = eqxi.create_vprim(
    "linear_solve",
    eqxi.filter_primitive_def(ft.partial(_linear_solve_impl, check_closure=False)),
    _linear_solve_abstract_eval,
    _linear_solve_jvp,
    lambda x: None,
)
# Then rebind so that the impl rule catches leaked-in tracers.
linear_solve_p.def_impl(
    eqxi.filter_primitive_def(ft.partial(_linear_solve_impl, check_closure=True))
)
eqxi.register_impl_finalisation(linear_solve_p)

#
# linear_solve
#


_SolverState = TypeVar("_SolverState")


@eqx.filter_jit
def linear_solve(
    operator,
    vector: PyTree[ArrayLike],
    solver=None,
    *,
    options: Optional[dict[str, Any]] = None,
    state: PyTree[Any] = None,
    throw: bool = True,
):
    if options is None:
        options = {}
    vector = jtu.tree_map(inexact_asarray, vector)

    state = eqxi.nondifferentiable(state, name="`lineax.linear_solve(..., state=...)`")
    options = eqxi.nondifferentiable(
        options, name="`lineax.linear_solve(..., options=...)`"
    )
    solver = eqxi.nondifferentiable(
        solver, name="`lineax.linear_solve(..., solver=...)`"
    )
    solution = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vector, options, solver, throw
    )
    return solution
