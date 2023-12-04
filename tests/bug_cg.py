# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from collections.abc import Callable
from typing import Any, Generic, Optional, TypeVar
from typing import ClassVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree
from jaxtyping import Scalar
from typing_extensions import TYPE_CHECKING
from typing_extensions import TypeAlias

from tests.bug_misc import preconditioner_and_y0, max_norm, resolve_rcond, tree_dot, tree_where
from tests.bug_opers import linearise, AbstractLinearOperator, conj
from tests.bug_solution import RESULTS

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox.internal import AbstractClassVar


_CGState: TypeAlias = tuple[AbstractLinearOperator, bool]

_SolverState = TypeVar("_SolverState")


class AbstractLinearSolver(eqx.Module, Generic[_SolverState]):
    """Abstract base class for all linear solvers."""

    @abc.abstractmethod
    def init(
            self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _SolverState:
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
    def allow_dependent_columns(self, operator: AbstractLinearOperator) -> bool:
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
    def allow_dependent_rows(self, operator: AbstractLinearOperator) -> bool:
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


# TODO(kidger): this is pretty slow to compile.
# - CG evaluates `operator.mv` three times.
# - Normal CG evaluates `operator.mv` seven (!) times.
# Possibly this can be cheapened a bit somehow?
class _CG(AbstractLinearSolver[_CGState]):
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    stabilise_every: Optional[int] = 10
    max_steps: Optional[int] = None

    _normal: AbstractClassVar[bool]

    def __check_init__(self):
        pass

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        is_nsd = True
        return operator, is_nsd

    # This differs from jax.scipy.sparse.linalg.cg in:
    # 1. Every few steps we calculate the residual directly, rather than by cheaply
    #    using the existing quantities. This improves numerical stability.
    # 2. We use a more sophisticated termination condition. To begin with we have an
    #    rtol and atol in the conventional way, inducing a vector-valued scale. This is
    #    then checked in both the `y` and `b` domains (for `Ay = b`).
    # 3. We return the number of steps, and whether or not the solve succeeded, as
    #    additional information.
    # 4. We don't try to support complex numbers. (Yet.)
    def compute(
            self, state: _CGState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        operator, is_nsd = state
        if self._normal:
            # Linearise if JacobianLinearOperator, to avoid computing the forward
            # pass separately for mv and transpose_mv.
            # This choice is "fast by default", even at the expense of memory.
            # If a downstream user wants to avoid this then they can call
            # ```
            # linear_solve(
            #     conj(operator.T) @ operator, operator.mv(b), solver=CG()
            # )
            # ```
            # directly.
            operator = linearise(operator)

            _mv = operator.mv
            _transpose_mv = conj(operator.transpose()).mv

            def mv(vector: PyTree) -> PyTree:
                return _transpose_mv(_mv(vector))

            vector = _transpose_mv(vector)

        preconditioner, y0 = preconditioner_and_y0(operator, vector, options)
        leaves, _ = jtu.tree_flatten(vector)
        size = sum(leaf.size for leaf in leaves)
        max_steps = 10 * size  # Copied from SciPy!
        r0 = (vector ** ω - mv(y0) ** ω).ω
        p0 = preconditioner.mv(r0)
        gamma0 = tree_dot(r0, p0)
        rcond = resolve_rcond(None, size, size, jnp.result_type(*leaves))
        initial_value = (
            ω(y0).call(lambda x: jnp.full_like(x, jnp.inf)).ω,
            y0,
            r0,
            p0,
            gamma0,
            0,
        )
        has_scale = not (
                isinstance(self.atol, (int, float))
                and isinstance(self.rtol, (int, float))
                and self.atol == 0
                and self.rtol == 0
        )
        if has_scale:
            b_scale = (self.atol + self.rtol * ω(vector).call(jnp.abs)).ω

        def not_converged(r, diff, y):
            # The primary tolerance check.
            # Given Ay=b, then we have to be doing better than `scale` in both
            # the `y` and the `b` spaces.
            if has_scale:
                y_scale = (self.atol + self.rtol * ω(y).call(jnp.abs)).ω
                norm1 = self.norm((r ** ω / b_scale ** ω).ω)  # pyright: ignore
                norm2 = self.norm((diff ** ω / y_scale ** ω).ω)
                return (norm1 > 1) | (norm2 > 1)
            else:
                return True

        def cond_fun(value):
            diff, y, r, _, gamma, step = value
            out = gamma > 0
            out = out & (step < max_steps)
            out = out & not_converged(r, diff, y)
            return out

        def body_fun(value):
            _, y, r, p, gamma, step = value
            mat_p = mv(p)
            jax.debug.print("inside {st} compute {p} {mp}", st=step, mp=mat_p, p=p)

            inner_prod = tree_dot(p, mat_p)
            alpha = gamma / inner_prod
            alpha = tree_where(
                jnp.abs(inner_prod) > 100 * rcond * gamma, alpha, jnp.nan
            )
            diff = (alpha * p ** ω).ω
            y = (y ** ω + diff ** ω).ω
            step = step + 1

            # E.g. see B.2 of
            # https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            # We compute the residual the "expensive" way every now and again, so as to
            # correct numerical rounding errors.
            def stable_r():
                return (vector ** ω - mv(y) ** ω).ω

            def cheap_r():
                return (r ** ω - alpha * mat_p ** ω).ω

            stable_step = (eqxi.unvmap_max(step) % self.stabilise_every) == 0
            stable_step = eqxi.nonbatchable(stable_step)
            r = lax.cond(stable_step, stable_r, cheap_r)

            z = preconditioner.mv(r)
            gamma_prev = gamma
            gamma = tree_dot(r, z)
            beta = gamma / gamma_prev
            p = (z ** ω + beta * p ** ω).ω
            return diff, y, r, p, gamma, step

        _, solution, _, _, _, num_steps = lax.while_loop(
            cond_fun, body_fun, initial_value
        )

        result = RESULTS.where(
            num_steps == max_steps,
            RESULTS.singular,
            RESULTS.successful,
        )

        stats = {"num_steps": num_steps, "max_steps": self.max_steps}
        return solution, result, stats

    def transpose(self, state: _CGState, options: dict[str, Any]):
        del options
        psd_op, is_nsd = state
        transpose_state = psd_op.transpose(), is_nsd
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _CGState, options: dict[str, Any]):
        del options
        psd_op, is_nsd = state
        conj_state = conj(psd_op), is_nsd
        conj_options = {}
        return conj_state, conj_options


class CG(_CG):
    """Conjugate gradient solver for linear systems.

    The operator should be positive or negative definite.

    Equivalent to `scipy.sparse.linalg.cg`.

    This supports the following `options` (as passed to
    `lx.linear_solve(..., options=...)`).

    - `preconditioner`: A positive definite [`lineax.AbstractLinearOperator`][]
        to be used as preconditioner. Defaults to
        [`lineax.IdentityLinearOperator`][].
    - `y0`: The initial estimate of the solution to the linear system. Defaults to all
        zeros.

    !!! info


    """

    _normal: ClassVar[bool] = False

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False


class NormalCG(_CG):
    """Conjugate gradient applied to the normal equations:

    `A^T A = A^T b`

    of a system of linear equations. Note that this squares the condition
    number, so it is not recommended. This is a fast but potentially inaccurate
    method, especially in 32 bit floating point precision.

    This can handle nonsquare operators provided they are full-rank.

    This supports the following `options` (as passed to
    `lx.linear_solve(..., options=...)`).

    - `preconditioner`: A positive definite [`lineax.AbstractLinearOperator`][]
        to be used as preconditioner. Defaults to
        [`lineax.IdentityLinearOperator`][].
    - `y0`: The initial estimate of the solution to the linear system. Defaults to all
        zeros.

    !!! info


    """

    _normal: ClassVar[bool] = True

    def allow_dependent_columns(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        return columns > rows

    def allow_dependent_rows(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        return rows > columns


CG.__init__.__doc__ = r"""**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `norm`: The norm to use when computing whether the error falls within the tolerance.
    Defaults to the max norm.
- `stabilise_every`: The conjugate gradient is an iterative method that produces
    candidate solutions $x_1, x_2, \ldots$, and terminates once $r_i = \| Ax_i - b \|$
    is small enough. For computational efficiency, the values $r_i$ are computed using
    other internal quantities, and not by directly evaluating the formula above.
    However, this computation of $r_i$ is susceptible to drift due to limited
    floating-point precision. Every `stabilise_every` steps, then $r_i$ is computed
    directly using the formula above, in order to stabilise the computation.
- `max_steps`: The maximum number of iterations to run the solver for. If more steps
    than this are required, then the solve is halted with a failure.
"""

NormalCG.__init__.__doc__ = r"""**Arguments:**
- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `norm`: The norm to use when computing whether the error falls within the tolerance.
    Defaults to the max norm.
- `stabilise_every`: The conjugate gradient is an iterative method that produces
    candidate solutions $x_1, x_2, \ldots$, and terminates once $r_i = \| Ax_i - b \|$
    is small enough. For computational efficiency, the values $r_i$ are computed using
    other internal quantities, and not by directly evaluating the formula above.
    However, this computation of $r_i$ is susceptible to drift due to limited
    floating-point precision. Every `stabilise_every` steps, then $r_i$ is computed
    directly using the formula above, in order to stabilise the computation.
- `max_steps`: The maximum number of iterations to run the solver for. If more steps
    than this are required, then the solve is halted with a failure.
"""
