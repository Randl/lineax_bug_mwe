import functools as ft
from collections.abc import Callable
from typing import (
    Any,
    Iterable,
    NoReturn,
    TypeVar,
    Union,
)

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import (  # pyright: ignore
    Array,
    ArrayLike,
    Inexact,
    PyTree,
)

from tests.bug_oper_misc import (
    inexact_asarray,
    NoneAux,
    _NoAuxOut,
    _NoAuxIn,
)

# Needed as static fields must be hashable and eq-able, and custom pytrees might have
# e.g. define custom __eq__ methods.
_T = TypeVar("_T")
_FlatPyTree = tuple[list[_T], jtu.PyTreeDef]


def _inexact_structure_impl(x):
    return x


def _inexact_structure(x: PyTree[jax.ShapeDtypeStruct]) -> PyTree[jax.ShapeDtypeStruct]:
    return jax.eval_shape(_inexact_structure_impl, x)


# `structure` must be static as with `JacobianLinearOperator`
class IdentityLinearOperator(eqx.Module):
    """Represents the identity transformation `X -> X`, where each `x in X` is some
    PyTree of floating-point JAX arrays.
    """

    input_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    output_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)

    def __init__(
        self,
        input_structure=None,
        output_structure=None,
    ):
        self.input_structure = None
        self.output_structure = None

    def mv(self, vector):
        return vector


class JacobianLinearOperator(eqx.Module):
    """Given a function `fn: X -> Y`, and a point `x in X`, then this defines the
    linear operator (also a function `X -> Y`) given by the Jacobian `(d(fn)/dx)(x)`.

    For example if the inputs and outputs are just arrays, then this is equivalent to
    `MatrixLinearOperator(jax.jacfwd(fn)(x))`.

    The Jacobian is not materialised; matrix-vector products, which are in fact
    Jacobian-vector products, are computed using autodifferentiation, specifically
    `jax.jvp`. Thus, `JacobianLinearOperator(fn, x).mv(v)` is equivalent to
    `jax.jvp(fn, (x,), (v,))`.

    See also [`lineax.linearise`][], which caches the primal computation, i.e.
    it returns `_, lin = jax.linearize(fn, x); FunctionLinearOperator(lin, ...)`

    See also [`lineax.materialise`][], which materialises the whole Jacobian in
    memory.
    """

    fn: Callable[
        [PyTree[Inexact[Array, "..."]], PyTree[Any]], PyTree[Inexact[Array, "..."]]
    ]
    x: PyTree[Inexact[Array, "..."]]
    args: PyTree[Any]
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        fn: Callable,
        x: PyTree[ArrayLike],
        args: PyTree[Any] = None,
        tags: Union[object, Iterable[object]] = (),
        _has_aux: bool = False,
    ):
        """**Arguments:**

        - `fn`: A function `(x, args) -> y`. The Jacobian `d(fn)/dx` is used as the
            linear operator, and `args` are just any other arguments that should not be
            differentiated.
        - `x`: The point to evaluate `d(fn)/dx` at: `(d(fn)/dx)(x, args)`.
        - `args`: As `x`; this is the point to evaluate `d(fn)/dx` at:
            `(d(fn)/dx)(x, args)`.
        - `tags`: any tags indicating whether this operator has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if these
            tags are wrong.
        """
        fn = NoneAux(fn)

        x = jtu.tree_map(inexact_asarray, x)
        fn = eqx.filter_closure_convert(fn, x, args)
        self.fn = fn
        self.x = x
        self.args = None
        self.tags = None

    def in_structure(self):
        return jax.eval_shape(lambda: self.x)


# `input_structure` must be static as with `JacobianLinearOperator`
class FunctionLinearOperator(eqx.Module):
    """Wraps a *linear* function `fn: X -> Y` into a linear operator. (So that
    `self.mv(x)` is defined by `self.mv(x) == fn(x)`.)

    See also [`lineax.materialise`][], which materialises the whole linear operator
    in memory. (Similar to `.as_matrix()`.)
    """

    fn: Callable[[PyTree[Inexact[Array, "..."]]], PyTree[Inexact[Array, "..."]]]
    input_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        fn: Callable[[PyTree[Inexact[Array, "..."]]], PyTree[Inexact[Array, "..."]]],
        input_structure: PyTree[jax.ShapeDtypeStruct],
        tags: Union[object, Iterable[object]] = (),
    ):
        """**Arguments:**

        - `fn`: a linear function. Should accept a PyTree of floating-point JAX arrays,
            and return a PyTree of floating-point JAX arrays.
        - `input_structure`: A PyTree of `jax.ShapeDtypeStruct`s specifying the
            structure of the input to the function. (When later calling `self.mv(x)`
            then this should match the structure of `x`, i.e.
            `jax.eval_shape(lambda: x)`.)
        - `tags`: any tags indicating whether this operator has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if these
            tags are wrong.
        """
        # See matching comment in JacobianLinearOperator.
        fn = eqx.filter_closure_convert(fn, input_structure)
        input_structure = _inexact_structure(input_structure)
        self.fn = fn
        self.input_structure = jtu.tree_flatten(input_structure)
        self.tags = None

    def mv(self, vector):
        return self.fn(vector)

    def transpose(self):
        transpose_fn = jax.linear_transpose(self.fn, self.in_structure())

        def _transpose_fn(vector):
            (out,) = transpose_fn(vector)
            return out

        # Works because transpose_fn is a PyTree
        return FunctionLinearOperator(_transpose_fn, self.out_structure(), self.tags)

    def in_structure(self):
        leaves, treedef = self.input_structure
        return jtu.tree_unflatten(treedef, leaves)

    def out_structure(self):
        return eqxi.cached_filter_eval_shape(self.fn, self.in_structure())


def _is_none(x):
    return x is None


class TangentLinearOperator(eqx.Module):
    """Internal to lineax. Used to represent the tangent (jvp) computation with
    respect to the linear operator in a linear solve.
    """

    primal: Any
    tangent: Any

    def mv(self, vector):
        mv = lambda operator: operator.mv(vector)
        out, t_out = eqx.filter_jvp(mv, (self.primal,), (self.tangent,))
        return jtu.tree_map(eqxi.materialise_zeros, out, t_out, is_leaf=_is_none)


def _default_not_implemented(name: str, operator) -> NoReturn:
    msg = f"`lineax.{name}` has not been implemented for {type(operator)}"
    if type(operator).__module__.startswith("lineax"):
        assert False, msg + ". Please file a bug against Lineax."
    else:
        raise NotImplementedError(msg)


# linearise


@ft.singledispatch
def linearise(operator):
    """Linearises a linear operator. This returns another linear operator.

    Mathematically speaking this is just the identity function. And indeed most linear
    operators will be returned unchanged.

    For specifically [`lineax.JacobianLinearOperator`][], then this will cache the
    primal pass, so that it does not need to be recomputed each time. That is, it uses
    some memory to improve speed. (This is the precisely same distinction as `jax.jvp`
    versus `jax.linearize`.)

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Another linear operator. Mathematically it performs matrix-vector products
    (`operator.mv`) that produce the same results as the input `operator`.
    """
    _default_not_implemented("linearise", operator)


@linearise.register(JacobianLinearOperator)
def _(operator):
    fn = _NoAuxIn(operator.fn, operator.args)
    # print(f"l, {operator.as_matrix()}")
    (_, aux), lin = jax.linearize(fn, operator.x)
    lin = _NoAuxOut(lin)
    out = FunctionLinearOperator(lin, operator.in_structure(), operator.tags)
    return out


@linearise.register(TangentLinearOperator)
def _(operator):
    primal_out, tangent_out = eqx.filter_jvp(
        linearise, (operator.primal,), (operator.tangent,)
    )
    return TangentLinearOperator(primal_out, tangent_out)


# conj


@ft.singledispatch
def conj(operator):
    """Elementwise conjugate of a linear operator. This returns another linear operator.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Another linear operator.
    """
    _default_not_implemented("conj", operator)


@conj.register(FunctionLinearOperator)
def _(operator):
    return FunctionLinearOperator(
        lambda vec: jtu.tree_map(jnp.conj, operator.mv(jtu.tree_map(jnp.conj, vec))),
        operator.in_structure(),
        operator.tags,
    )
