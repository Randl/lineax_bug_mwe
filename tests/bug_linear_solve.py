import abc
import functools as ft
from typing import Any
from typing import Generic, Optional, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, PyTree
from typing_extensions import TypeAlias

from tests.bug_oper_misc import inexact_asarray
from tests.bug_opers import (
    linearise,
    TangentLinearOperator,
)
from tests.bug_solution import RESULTS, Solution


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
    has_nonfinites = jnp.any(
        jnp.stack(
            [jnp.any(jnp.invert(jnp.isfinite(x))) for x in jtu.tree_leaves(solution)]
        )
    )
    result = RESULTS.where(
        (result == RESULTS.successful) & has_nonfinites,
        RESULTS.singular,
        result,
    )
    if throw:
        solution, result, stats = result.error_if(
            (solution, result, stats),
            result != RESULTS.successful,
        )
    return solution, result, stats


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
    solution, result, stats = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vector, options, solver, throw
    )

    jax.debug.print("In JVP {sol}", sol=solution)

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
    sol, _, _ = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vecs, options, solver, True
    )
    sols.append(sol)
    t_solution = jtu.tree_map(_sum, *sols)

    out = solution, result, stats
    t_out = (
        t_solution,
        jtu.tree_map(lambda _: None, result),
        jtu.tree_map(lambda _: None, stats),
    )
    return out, t_out


@eqxi.filter_primitive_transpose(materialise_zeros=True)  # pyright: ignore
def _linear_solve_transpose(inputs, cts_out):
    return None, None, None, None, None, None


# Call with `check_closure=False` so that the autocreated vmap rule works.
linear_solve_p = eqxi.create_vprim(
    "linear_solve",
    eqxi.filter_primitive_def(ft.partial(_linear_solve_impl, check_closure=False)),
    _linear_solve_abstract_eval,
    _linear_solve_jvp,
    _linear_solve_transpose,
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


class AbstractLinearSolver(eqx.Module, Generic[_SolverState]):
    """Abstract base class for all linear solvers."""

    @abc.abstractmethod
    def init(self, operator, options: dict[str, Any]) -> _SolverState:
        """Do any initial computation on just the `operator`.

        For example, an LU solver would compute the LU decomposition of the operator
        (and this does not require knowing the vector yet).

        It is common to need to solve the linear system `Ax=b` multiple times in
        succession, with the same operator `A` and multiple vectors `b`. This method
        improves efficiency by making it possible to re-use the computation performed
        on just the operator.

        !!! Example

            ```python
            operator = lx.MatrixLinearOperator(...)
            vector1 = ...
            vector2 = ...
            solver = lx.LU()
            state = solver.init(operator, options={})
            solution1 = lx.linear_solve(operator, vector1, solver, state=state)
            solution2 = lx.linear_solve(operator, vector2, solver, state=state)
            ```

        **Arguments:**

        - `operator`: a linear operator.
        - `options`: a dictionary of any extra options that the solver may wish to
            accept.

        **Returns:**

        A PyTree of arbitrary Python objects.
        """

    @abc.abstractmethod
    def compute(
        self, state: _SolverState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        """Solves a linear system.

        **Arguments:**

        - `state`: as returned from [`lineax.AbstractLinearSolver.init`][].
        - `vector`: the vector to solve against.
        - `options`: a dictionary of any extra options that the solver may wish to
            accept. For example, [`lineax.CG`][] accepts a `preconditioner` option.

        **Returns:**

        A 3-tuple of:

        - The solution to the linear system.
        - An integer indicating the success or failure of the solve. This is an integer
            which may be converted to a human-readable error message via
            `lx.RESULTS[...]`.
        - A dictionary of an extra statistics about the solve, e.g. the number of steps
            taken.
        """

    @abc.abstractmethod
    def allow_dependent_columns(self, operator) -> bool:
        """Does this method ever produce non-NaN outputs for operators with linearly
        dependent columns? (Even if only sometimes.)

        If `True` then a more expensive backward pass is needed, to account for the
        extra generality.

        If you do not need to autodifferentiate through a custom linear solver then you
        simply define this method as
        ```python
        class MyLinearSolver(AbstractLinearsolver):
            def allow_dependent_columns(self, operator):
                raise NotImplementedError
        ```

        **Arguments:**

        - `operator`: a linear operator.

        **Returns:**

        Either `True` or `False`.
        """

    @abc.abstractmethod
    def allow_dependent_rows(self, operator) -> bool:
        """Does this method ever produce non-NaN outputs for operators with
        linearly dependent rows? (Even if only sometimes)

        If `True` then a more expensive backward pass is needed, to account for the
        extra generality.

        If you do not need to autodifferentiate through a custom linear solver then you
        simply define this method as
        ```python
        class MyLinearSolver(AbstractLinearsolver):
            def allow_dependent_rows(self, operator):
                raise NotImplementedError
        ```

        **Arguments:**

        - `operator`: a linear operator.

        **Returns:**

        Either `True` or `False`.
        """

    @abc.abstractmethod
    def transpose(
        self, state: _SolverState, options: dict[str, Any]
    ) -> tuple[_SolverState, dict[str, Any]]:
        """Transposes the result of [`lineax.AbstractLinearSolver.init`][].

        That is, it should be the case that
        ```python
        state_transpose, _ = solver.transpose(solver.init(operator, options), options)
        state_transpose2 = solver.init(operator.T, options)
        ```
        must be identical to each other.

        It is relatively common (in particular when differentiating through a linear
        solve) to need to solve both `Ax = b` and `A^T x = b`. This method makes it
        possible to avoid computing both `solver.init(operator)` and
        `solver.init(operator.T)` if one can be cheaply computed from the other.

        **Arguments:**

        - `state`: as returned from `solver.init`.
        - `options`: any extra options that were passed to `solve.init`.

        **Returns:**

        A 2-tuple of:

        - The state of the transposed operator.
        - The options for the transposed operator.
        """

    @abc.abstractmethod
    def conj(
        self, state: _SolverState, options: dict[str, Any]
    ) -> tuple[_SolverState, dict[str, Any]]:
        """Conjugate the result of [`lineax.AbstractLinearSolver.init`][].

        That is, it should be the case that
        ```python
        state_conj, _ = solver.conj(solver.init(operator, options), options)
        state_conj2 = solver.init(conj(operator), options)
        ```
        must be identical to each other.

        **Arguments:**

        - `state`: as returned from `solver.init`.
        - `options`: any extra options that were passed to `solve.init`.

        **Returns:**

        A 2-tuple of:

        - The state of the conjugated operator.
        - The options for the conjugated operator.
        """


_qr_token = eqxi.str2jax("qr_token")
_diagonal_token = eqxi.str2jax("diagonal_token")
_well_posed_diagonal_token = eqxi.str2jax("well_posed_diagonal_token")
_tridiagonal_token = eqxi.str2jax("tridiagonal_token")
_triangular_token = eqxi.str2jax("triangular_token")
_cholesky_token = eqxi.str2jax("cholesky_token")
_lu_token = eqxi.str2jax("lu_token")
_svd_token = eqxi.str2jax("svd_token")

_AutoLinearSolverState: TypeAlias = tuple[Any, Any]


@eqx.filter_jit
def linear_solve(
    operator,
    vector: PyTree[ArrayLike],
    solver: AbstractLinearSolver = None,
    *,
    options: Optional[dict[str, Any]] = None,
    state: PyTree[Any] = None,
    throw: bool = True,
) -> Solution:
    r"""Solves a linear system.

    Given an operator represented as a matrix $A$, and a vector $b$: if the operator is
    square and nonsingular (so that the problem is well-posed), then this returns the
    usual solution $x$ to $Ax = b$, defined as $A^{-1}b$.

    If the operator is overdetermined, then this either returns the least-squares
    solution $\min_x \| Ax - b \|_2$, or throws an error. (Depending on the choice of
    solver.)

    If the operator is underdetermined, then this either returns the minimum-norm
    solution $\min_x \|x\|_2 \text{ subject to } Ax = b$, or throws an error. (Depending
    on the choice of solver.)

    !!! info

        This function is equivalent to either `numpy.linalg.solve`, or to its
        generalisation `numpy.linalg.lstsq`, depending on the choice of solver.

    The default solver is `lineax.AutoLinearSolver(well_posed=True)`. This
    automatically selects a solver depending on the structure (e.g. triangular) of your
    problem, and will throw an error if your system is overdetermined or
    underdetermined.

    Use `lineax.AutoLinearSolver(well_posed=False)` if your system is known to be
    overdetermined or underdetermined (although handling this case implies greater
    computational cost).

    !!! tip

        These three kinds of solution to a linear system are collectively known as the
        "pseudoinverse solution" to a linear system. That is, given our matrix $A$, let
        $A^\dagger$ denote the
        [Moore--Penrose pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)
        of $A$. Then the usual/least-squares/minimum-norm solution are all equal to
        $A^\dagger b$.

    **Arguments:**

    - `operator`: a linear operator. This is the '$A$' in '$Ax = b$'.

        Most frequently this operator is simply represented as a JAX matrix (i.e. a
        rank-2 JAX array), but any [`lineax.AbstractLinearOperator`][] is supported.

        Note that if it is a matrix, then it should be passed as an
        [`lineax.MatrixLinearOperator`][], e.g.
        ```python
        matrix = jax.random.normal(key, (5, 5))  # JAX array of shape (5, 5)
        operator = lx.MatrixLinearOperator(matrix)  # Wrap into a linear operator
        solution = lx.linear_solve(operator, ...)
        ```
        rather than being passed directly.

    - `vector`: the vector to solve against. This is the '$b$' in '$Ax = b$'.

    - `solver`: the solver to use. Should be any [`lineax.AbstractLinearSolver`][].
        The default is [`lineax.AutoLinearSolver`][] which behaves as discussed
        above.

        If the operator is overdetermined or underdetermined , then passing
        [`lineax.SVD`][] is typical.

    - `options`: Individual solvers may accept additional runtime arguments; for example
        [`lineax.CG`][] allows for specifying a preconditioner. See each individual
        solver's documentation for more details. Keyword only argument.

    - `state`: If performing multiple linear solves with the same operator, then it is
        possible to save re-use some computation between these solves, and to pass the
        result of any intermediate computation in as this argument. See
        [`lineax.AbstractLinearSolver.init`][] for more details. Keyword only
        argument.

    - `throw`: How to report any failures. (E.g. an iterative solver running out of
        steps, or a well-posed-only solver being run with a singular operator.)

        If `True` then a failure will raise an error. Note that errors are only reliably
        raised on CPUs. If on GPUs then the error may only be printed to stderr, whilst
        on TPUs then the behaviour is undefined.

        If `False` then the returned solution object will have a `result` field
        indicating whether any failures occured. (See [`lineax.Solution`][].)

        Keyword only argument.

    **Returns:**

    An [`lineax.Solution`][] object containing the solution to the linear system.
    """  # noqa: E501

    if options is None:
        options = {}
    vector = jtu.tree_map(inexact_asarray, vector)
    vector_struct = jax.eval_shape(lambda: vector)
    operator_out_structure = operator.out_structure()

    state = eqxi.nondifferentiable(state, name="`lineax.linear_solve(..., state=...)`")
    options = eqxi.nondifferentiable(
        options, name="`lineax.linear_solve(..., options=...)`"
    )
    solver = eqxi.nondifferentiable(
        solver, name="`lineax.linear_solve(..., solver=...)`"
    )
    solution, result, stats = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vector, options, solver, throw
    )
    # TODO: prevent forward-mode autodiff through stats
    stats = eqxi.nondifferentiable_backward(stats)
    return Solution(value=solution, result=result, state=state, stats=stats)
