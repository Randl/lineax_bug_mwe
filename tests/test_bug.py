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
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr

from tests.bug_cg import NormalCG
from tests.bug_helpers import construct_matrix, make_jac_operator, shaped_allclose
from tests.bug_linear_solve import linear_solve


def test_jvp_jvp(getkey):
    tol = 1e-12
    solver, pseudoinverse = (NormalCG(rtol=tol, atol=tol), False)
    make_operator = make_jac_operator
    make_matrix = construct_matrix
    dtype = jnp.complex128
    t_tags = None
    if (make_matrix is construct_matrix) or pseudoinverse:
        matrix, t_matrix, tt_matrix, tt_t_matrix = construct_matrix(
            getkey, solver, None, num=4, dtype=dtype
        )

        make_op = ft.partial(make_operator, getkey)
        t_make_operator = lambda p, t_p: eqx.filter_jvp(
            make_op, (p, None), (t_p, t_tags)
        )
        tt_make_operator = lambda p, t_p, tt_p, tt_t_p: eqx.filter_jvp(
            t_make_operator, (p, t_p), (tt_p, tt_t_p)
        )
        (operator, t_operator), (tt_operator, tt_t_operator) = tt_make_operator(
            matrix, t_matrix, tt_matrix, tt_t_matrix
        )

        out_size, _ = matrix.shape
        vec = jr.normal(getkey(), (out_size,), dtype=dtype)
        t_vec = jr.normal(getkey(), (out_size,), dtype=dtype)
        tt_vec = jr.normal(getkey(), (out_size,), dtype=dtype)
        jr.normal(getkey(), (out_size,), dtype=dtype)

        def linear_solve1(operator, vector):
            state = solver.init(operator, options={})
            state_dynamic, state_static = eqx.partition(state, eqx.is_inexact_array)
            state_dynamic = lax.stop_gradient(state_dynamic)
            state = eqx.combine(state_dynamic, state_static)

            sol = linear_solve(operator, vector, state=state, solver=solver)
            return sol.value

        jnp_solve1 = jnp.linalg.solve  # pyright: ignore

        linear_solve2 = ft.partial(eqx.filter_jvp, linear_solve1)
        jnp_solve2 = ft.partial(eqx.filter_jvp, jnp_solve1)

        def _make_primal_tangents():
            lx_args = ([], [], operator, t_operator, tt_operator, tt_t_operator)
            jnp_args = ([], [], matrix, t_matrix, tt_matrix, tt_t_matrix)
            for primals, ttangents, op, t_op, tt_op, tt_t_op in (lx_args, jnp_args):
                primals.append(vec)
                ttangents.append(tt_vec)
            lx_out = tuple(lx_args[0]), tuple(lx_args[1])
            jnp_out = tuple(jnp_args[0]), tuple(jnp_args[1])
            return lx_out, jnp_out

        linear_solve3 = lambda v: linear_solve2((operator, v), (t_operator, t_vec))
        jnp_solve3 = lambda v: jnp_solve2((matrix, v), (t_matrix, t_vec))

        linear_solve3 = ft.partial(eqx.filter_jvp, linear_solve3)
        linear_solve3 = eqx.filter_jit(linear_solve3)
        jnp_solve3 = ft.partial(eqx.filter_jvp, jnp_solve3)
        jnp_solve3 = eqx.filter_jit(jnp_solve3)

        (primal, tangent), (jnp_primal, jnp_tangent) = _make_primal_tangents()
        (out, t_out), (minus_out, tt_out) = linear_solve3(primal, tangent)
        (true_out, true_t_out), (minus_true_out, true_tt_out) = jnp_solve3(
            jnp_primal, jnp_tangent
        )

        assert shaped_allclose(out, true_out, atol=1e-4)
        assert shaped_allclose(t_out, true_t_out, atol=1e-4)
        assert shaped_allclose(tt_out, true_tt_out, atol=1e-4)
        assert shaped_allclose(minus_out, minus_true_out, atol=1e-4)
