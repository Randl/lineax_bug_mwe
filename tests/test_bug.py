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

import functools as ft

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from tests.bug_cg import NormalCG
from tests.bug_helpers import construct_matrix, make_jac_operator
from tests.bug_linear_solve import linear_solve


def test_jvp_jvp(getkey):
    tol = 1e-12
    solver = NormalCG(rtol=tol, atol=tol)
    dtype = jnp.complex128

    matrix, t_matrix = construct_matrix(getkey, solver, None, num=2, dtype=dtype)

    make_op = ft.partial(make_jac_operator, getkey)
    t_make_operator = lambda p, t_p: eqx.filter_jvp(make_op, (p, None), (t_p, None))

    operator, t_operator = t_make_operator(matrix, t_matrix)

    out_size, _ = matrix.shape
    vec = jr.normal(getkey(), (out_size,), dtype=dtype)
    t_vec = jr.normal(getkey(), (out_size,), dtype=dtype)

    def linear_solve1(operator, vector):
        state = solver.init(operator, options={})
        state_dynamic, state_static = eqx.partition(state, eqx.is_inexact_array)
        state_dynamic = lax.stop_gradient(state_dynamic)
        state = eqx.combine(state_dynamic, state_static)

        sol = linear_solve(operator, vector, state=state, solver=solver)
        return sol

    jnp_solve1 = jnp.linalg.solve

    linear_solve2 = ft.partial(eqx.filter_jvp, linear_solve1)
    jnp_solve2 = ft.partial(eqx.filter_jvp, jnp_solve1)

    linear_solve3 = lambda v: linear_solve2((operator, v), (t_operator, t_vec))
    jnp_solve3 = lambda v: jnp_solve2((matrix, v), (t_matrix, t_vec))

    linear_solve3 = eqx.filter_jit(linear_solve3)
    lowered = jax.jit(linear_solve3).lower(vec)
    compiled = lowered.compile()
    print(vec)
    o1, o2 = compiled(vec)
    print(compiled.as_text())

    jnp_solve3 = eqx.filter_jit(jnp_solve3)
    to1, to2 = jnp_solve3(vec)

    assert np.allclose(o1, to1, atol=1e-4)
    assert np.allclose(o2, to2, atol=1e-4)
