import functools as ft
import math
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
import jax.flatten_util as jfu
import jax.lax as lax
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
from tests.bug_type import sentinel

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
        input_structure: PyTree[jax.ShapeDtypeStruct],
        output_structure: PyTree[jax.ShapeDtypeStruct] = sentinel,
    ):
        """**Arguments:**

        - `input_structure`: A PyTree of `jax.ShapeDtypeStruct`s specifying the
            structure of the the input space. (When later calling `self.mv(x)`
            then this should match the structure of `x`, i.e.
            `jax.eval_shape(lambda: x)`.)
        - `output_structure`: A PyTree of `jax.ShapeDtypeStruct`s specifying the
            structure of the the output space. If not passed then this defaults to the
            same as `input_structure`. If passed then it must have the same number of
            elements as `input_structure`, so that the operator is square.
        """
        if output_structure is sentinel:
            output_structure = input_structure
        input_structure = _inexact_structure(input_structure)
        output_structure = _inexact_structure(output_structure)
        self.input_structure = jtu.tree_flatten(input_structure)
        self.output_structure = jtu.tree_flatten(output_structure)

    def mv(self, vector):
        if jax.eval_shape(lambda: vector) != self.in_structure():
            raise ValueError("Vector and operator structures do not match")
        elif self.input_structure == self.output_structure:
            return vector  # fast-path for common special case
        else:
            raise ValueError()

    def in_structure(self):
        leaves, treedef = self.input_structure
        return jtu.tree_unflatten(treedef, leaves)


class AuxLinearOperator(eqx.Module):
    """Internal to lineax. Used to represent a linear operator with additional
    metadata attached.
    """

    operator: Any
    aux: PyTree[Array]

    def as_matrix(self):
        return materialise(self).as_matrix()

    def mv(self, vector):
        return self.operator.mv(vector)

    def transpose(self):
        return self.operator.transpose()

    def in_structure(self):
        return self.operator.in_structure()


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
        if not _has_aux:
            fn = NoneAux(fn)
        # Flush out any closed-over values, so that we can safely pass `self`
        # across API boundaries. (In particular, across `linear_solve_p`.)
        # We don't use `jax.closure_convert` as that only flushes autodiffable
        # (=floating-point) constants. It probably doesn't matter, but if `fn` is a
        # PyTree capturing non-floating-point constants, we should probably continue
        # to respect that, and keep any non-floating-point constants as part of the
        # PyTree structure.
        x = jtu.tree_map(inexact_asarray, x)
        fn = eqx.filter_closure_convert(fn, x, args)
        self.fn = fn
        self.x = x
        self.args = args
        self.tags = None

    def in_structure(self):
        return jax.eval_shape(lambda: self.x)

    def out_structure(self):
        fn = _NoAuxOut(_NoAuxIn(self.fn, self.args))
        return eqxi.cached_filter_eval_shape(fn, self.x)


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

    def as_matrix(self):
        return materialise(self).as_matrix()

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


class _Leaf:  # not a pytree
    def __init__(self, value):
        self.value = value


def _matmul(matrix: ArrayLike, vector: ArrayLike) -> Array:
    # matrix has structure [leaf(out), leaf(in)]
    # vector has structure [leaf(in)]
    # return has structure [leaf(out)]
    return jnp.tensordot(
        matrix, vector, axes=jnp.ndim(vector), precision=lax.Precision.HIGHEST
    )


def _tree_matmul(matrix: PyTree[ArrayLike], vector: PyTree[ArrayLike]) -> PyTree[Array]:
    # matrix has structure [tree(in), leaf(out), leaf(in)]
    # vector has structure [tree(in), leaf(in)]
    # return has structure [leaf(out)]
    matrix = jtu.tree_leaves(matrix)
    vector = jtu.tree_leaves(vector)
    assert len(matrix) == len(vector)
    return sum([_matmul(m, v) for m, v in zip(matrix, vector)])


# The `{input,output}_structure`s have to be static because otherwise abstract
# evaluation rules will promote them to ShapedArrays.
class PyTreeLinearOperator(eqx.Module):
    """Represents a PyTree of floating-point JAX arrays as a linear operator.

    This is basically a generalisation of [`lineax.MatrixLinearOperator`][], from
    taking just a single array to take a PyTree-of-arrays. (And likewise from returning
    a single array to returning a PyTree-of-arrays.)

    Specifically, suppose we want this to be a linear operator `X -> Y`, for which
    elements of `X` are PyTrees with structure `T` whose `i`th leaf is a floating-point
    JAX array of shape `x_shape_i`, and elements of `Y` are PyTrees with structure `S`
    whose `j`th leaf is a floating-point JAX array of has shape `y_shape_j`. Then the
    input PyTree should have structure `T`-compose-`S`, and its `(i, j)`-th  leaf should
    be a floating-point JAX array of shape `(*x_shape_i, *y_shape_j)`.

    !!! Example

        ```python
        # Suppose `x` is a member of our input space, with the following pytree
        # structure:
        eqx.tree_pprint(x)  # [f32[5, 9], f32[3]]

        # Suppose `y` is a member of our output space, with the following pytree
        # structure:
        eqx.tree_pprint(y)
        # {"a": f32[1, 2]}

        # then `pytree` should be a pytree with the following structure:
        eqx.tree_pprint(pytree)  # {"a": [f32[1, 2, 5, 9], f32[1, 2, 3]]}
        ```
    """

    pytree: PyTree[Inexact[Array, "..."]]
    output_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)
    input_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)

    def __init__(
        self,
        pytree: PyTree[ArrayLike],
        output_structure: PyTree[jax.ShapeDtypeStruct],
        tags: Union[object, frozenset[object]] = (),
    ):
        """**Arguments:**

        - `pytree`: this should be a PyTree, with structure as specified in
            [`lineax.PyTreeLinearOperator`][].
        - `out_structure`: the structure of the output space. This should be a PyTree of
            `jax.ShapeDtypeStruct`s. (The structure of the input space is then
            automatically derived from the structure of `pytree`.)
        - `tags`: any tags indicating whether this operator has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if these
            tags are wrong.
        """
        output_structure = _inexact_structure(output_structure)
        self.pytree = jtu.tree_map(inexact_asarray, pytree)
        self.output_structure = jtu.tree_flatten(output_structure)
        self.tags = None

        # self.out_structure() has structure [tree(out)]
        # self.pytree has structure [tree(out), tree(in), leaf(out), leaf(in)]
        def get_structure(struct, subpytree):
            # subpytree has structure [tree(in), leaf(out), leaf(in)]
            def sub_get_structure(leaf):
                shape = jnp.shape(leaf)  # [leaf(out), leaf(in)]
                ndim = len(struct.shape)
                if shape[:ndim] != struct.shape:
                    raise ValueError(
                        "`pytree` and `output_structure` are not consistent"
                    )
                return jax.ShapeDtypeStruct(shape=shape[ndim:], dtype=jnp.dtype(leaf))

            return _Leaf(jtu.tree_map(sub_get_structure, subpytree))

        if output_structure is None:
            # Implies that len(input_structures) > 0
            raise ValueError("Cannot have trivial output_structure")
        input_structures = jtu.tree_map(get_structure, output_structure, self.pytree)
        input_structures = jtu.tree_leaves(input_structures)
        input_structure = input_structures[0].value
        for val in input_structures[1:]:
            if eqx.tree_equal(input_structure, val.value) is not True:
                raise ValueError(
                    "`pytree` does not have a consistent `input_structure`"
                )
        self.input_structure = jtu.tree_flatten(input_structure)

    def mv(self, vector):
        # vector has structure [tree(in), leaf(in)]
        # self.out_structure() has structure [tree(out)]
        # self.pytree has structure [tree(out), tree(in), leaf(out), leaf(in)]
        # return has struture [tree(out), leaf(out)]
        def matmul(_, matrix):
            return _tree_matmul(matrix, vector)

        return jtu.tree_map(matmul, self.out_structure(), self.pytree)

    def as_matrix(self):
        dtype = jnp.result_type(*jtu.tree_leaves(self.pytree))

        def concat_in(struct, subpytree):
            leaves = jtu.tree_leaves(subpytree)
            assert all(
                leaf.shape[: len(struct.shape)] == struct.shape for leaf in leaves
            )
            size = math.prod(struct.shape)
            leaves = [
                leaf.astype(dtype).reshape(size, -1) if leaf.size > 0 else leaf
                for leaf in leaves
            ]
            return jnp.concatenate(leaves, axis=1)

        matrix = jtu.tree_map(concat_in, self.out_structure(), self.pytree)
        matrix = jtu.tree_leaves(matrix)
        return jnp.concatenate(matrix, axis=0)

    def transpose(self):
        def _transpose(struct, subtree):
            def _transpose_impl(leaf):
                return jnp.moveaxis(leaf, source, dest)

            source = list(range(struct.ndim))
            dest = list(range(-struct.ndim, 0))
            return jtu.tree_map(_transpose_impl, subtree)

        pytree_transpose = jtu.tree_map(_transpose, self.out_structure(), self.pytree)
        pytree_transpose = jtu.tree_transpose(
            jtu.tree_structure(self.out_structure()),
            jtu.tree_structure(self.in_structure()),
            pytree_transpose,
        )
        return PyTreeLinearOperator(
            pytree_transpose, self.in_structure(), self.tags
        )

    def in_structure(self):
        leaves, treedef = self.input_structure
        return jtu.tree_unflatten(treedef, leaves)

    def out_structure(self):
        leaves, treedef = self.output_structure
        return jtu.tree_unflatten(treedef, leaves)


def _is_none(x):
    return x is None


class TangentLinearOperator(eqx.Module):
    """Internal to lineax. Used to represent the tangent (jvp) computation with
    respect to the linear operator in a linear solve.
    """

    primal: Any
    tangent: Any

    def __check_init__(self):
        assert type(self.primal) is type(self.tangent)  # noqa: E721

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
    return AuxLinearOperator(out, aux)


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


# materialise


@ft.singledispatch
def materialise(operator):
    """Materialises a linear operator. This returns another linear operator.

    Mathematically speaking this is just the identity function. And indeed most linear
    operators will be returned unchanged.

    For specifically [`lineax.JacobianLinearOperator`][] and
    [`lineax.FunctionLinearOperator`][] then the linear operator is materialised in
    memory. That is, it becomes defined as a matrix (or pytree of arrays), rather
    than being defined only through its matrix-vector product
    ([`lineax.AbstractLinearOperator.mv`][]).

    Materialisation sometimes improves compile time or run time. It usually increases
    memory usage.

    For example:
    ```python
    large_function = ...
    operator = lx.FunctionLinearOperator(large_function, ...)

    # Option 1
    out1 = operator.mv(vector1)  # Traces and compiles `large_function`
    out2 = operator.mv(vector2)  # Traces and compiles `large_function` again!
    out3 = operator.mv(vector3)  # Traces and compiles `large_function` a third time!
    # All that compilation might lead to long compile times.
    # If `large_function` takes a long time to run, then this might also lead to long
    # run times.

    # Option 2
    operator = lx.materialise(operator)  # Traces and compiles `large_function` and
                                           # stores the result as a matrix.
    out1 = operator.mv(vector1)  # Each of these just computes a matrix-vector product
    out2 = operator.mv(vector2)  # against the stored matrix.
    out3 = operator.mv(vector3)  #
    # Now, `large_function` is only compiled once, and only ran once.
    # However, storing the matrix might take a lot of memory, and the initial
    # computation may-or-may-not take a long time to run.
    ```
    Generally speaking it is worth first setting up your problem without
    `lx.materialise`, and using it as an optional optimisation if you find that it
    helps your particular problem.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Another linear operator. Mathematically it performs matrix-vector products
    (`operator.mv`) that produce the same results as the input `operator`.
    """
    _default_not_implemented("materialise", operator)


@materialise.register(PyTreeLinearOperator)
@materialise.register(IdentityLinearOperator)
def _(operator):
    return operator


@materialise.register(AuxLinearOperator)  # pyright: ignore
def _(operator):
    return materialise(operator.operator)


@materialise.register(JacobianLinearOperator)
def _(operator):
    fn = _NoAuxIn(operator.fn, operator.args)
    jac, aux = jacobian(
        fn,
        operator.in_size(),
        operator.out_size(),
        holomorphic=jnp.iscomplexobj(operator.x),
        has_aux=True,
    )(operator.x)
    out = PyTreeLinearOperator(jac, operator.out_structure(), operator.tags)
    return AuxLinearOperator(out, aux)


@materialise.register(FunctionLinearOperator)
def _(operator):
    flat, unravel = eqx.filter_eval_shape(jfu.ravel_pytree, operator.in_structure())
    eye = jnp.eye(flat.size, dtype=flat.dtype)
    jac = jax.vmap(lambda x: operator.fn(unravel(x)), out_axes=-1)(eye)

    def batch_unravel(x):
        assert x.ndim > 0
        unravel_ = unravel
        for _ in range(x.ndim - 1):
            unravel_ = jax.vmap(unravel_)
        return unravel_(x)

    jac = jtu.tree_map(batch_unravel, jac)
    return PyTreeLinearOperator(jac, operator.out_structure(), operator.tags)
