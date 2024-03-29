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

from collections.abc import Callable
from typing import Any, Optional
from typing import ClassVar

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import AbstractClassVar
from equinox.internal import ω
from jaxtyping import Array, PyTree
from jaxtyping import Scalar

from tests.bug_misc import (
    preconditioner_and_y0,
    max_norm,
    resolve_rcond,
    tree_where,
)
from tests.bug_opers import linearise, conj


class _CG(eqx.Module):
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    stabilise_every: Optional[int] = 10
    max_steps: Optional[int] = None

    _normal: AbstractClassVar[bool]

    def init(self, operator, options: dict[str, Any]):
        del options
        return operator

    def compute(self, state, vector: PyTree[Array], options: dict[str, Any]):
        operator = state

        operator = linearise(operator)

        _mv = operator.mv
        _transpose_mv = conj(operator.transpose()).mv

        def mv(vector):
            return _transpose_mv(_mv(vector))

        vector = _transpose_mv(vector)

        preconditioner, y0 = preconditioner_and_y0(operator, vector, options)
        leaves, _ = jtu.tree_flatten(vector)
        size = sum(leaf.size for leaf in leaves)
        max_steps = 10 * size  # Copied from SciPy!
        r0 = (vector**ω - mv(y0) ** ω).ω
        p0 = preconditioner.mv(r0)
        gamma0 = jnp.vdot(r0, p0)
        rcond = resolve_rcond(None, size, size, jnp.result_type(*leaves))
        initial_value = (
            ω(y0).call(lambda x: jnp.full_like(x, jnp.inf)).ω,
            y0,
            r0,
            p0,
            gamma0,
            0,
        )

        b_scale = (self.atol + self.rtol * ω(vector).call(jnp.abs)).ω

        def not_converged(r, diff, y):
            # The primary tolerance check.
            # Given Ay=b, then we have to be doing better than `scale` in both
            # the `y` and the `b` spaces.
            y_scale = (self.atol + self.rtol * ω(y).call(jnp.abs)).ω
            norm1 = self.norm((r**ω / b_scale**ω).ω)  # pyright: ignore
            norm2 = self.norm((diff**ω / y_scale**ω).ω)
            return (norm1 > 1) | (norm2 > 1)

        def cond_fun(value):
            diff, y, r, _, gamma, step = value
            out = gamma > 0
            out = out & (step < max_steps)
            out = out & not_converged(r, diff, y)
            return out

        def body_fun(value):
            _, y, r, p, gamma, step = value
            mat_p = mv(p)

            inner_prod = jnp.vdot(p, mat_p)
            alpha = gamma / inner_prod
            alpha = tree_where( # TODO: replacing with jnp.where reduces probability of fail
                jnp.abs(inner_prod) > 100 * rcond * gamma, alpha, jnp.nan
            )
            diff = (alpha * p**ω).ω
            y = (y**ω + diff**ω).ω
            step = step + 1

            def stable_r():
                return (vector**ω - mv(y) ** ω).ω

            r = stable_r()

            z = preconditioner.mv(r)
            gamma_prev = gamma
            gamma = jnp.vdot(r, z)
            beta = gamma / gamma_prev
            p = (z**ω + beta * p**ω).ω
            return diff, y, r, p, gamma, step

        _, solution, _, _, _, _ = lax.while_loop(cond_fun, body_fun, initial_value)

        return solution


class NormalCG(_CG):
    _normal: ClassVar[bool] = True
