"""A compute-oriented float64 subset of NumPy backed by Mojo kernels."""

import numpy as _np

from . import linalg, random
from .core import (
    abs,
    absolute,
    add,
    amax,
    amin,
    argmax,
    argmin,
    argsort,
    clip,
    cos,
    divide,
    exp,
    log,
    max,
    maximum,
    mean,
    min,
    minimum,
    multiply,
    negative,
    prod,
    product,
    sin,
    sort,
    sqrt,
    square,
    std,
    subtract,
    sum,
    tanh,
    true_divide,
    var,
)
from .linalg import dot, matmul

__version__ = "0.1.0"

array = _np.array
asarray = _np.asarray
ascontiguousarray = _np.ascontiguousarray
arange = _np.arange
empty = _np.empty
empty_like = _np.empty_like
full = _np.full
full_like = _np.full_like
ones = _np.ones
ones_like = _np.ones_like
zeros = _np.zeros
zeros_like = _np.zeros_like
eye = _np.eye
linspace = _np.linspace
reshape = _np.reshape
transpose = _np.transpose
concatenate = _np.concatenate
stack = _np.stack
allclose = _np.allclose
isclose = _np.isclose
array_equal = _np.array_equal

ndarray = _np.ndarray
float64 = _np.float64
int64 = _np.int64
uint64 = _np.uint64
nan = _np.nan
inf = _np.inf
pi = _np.pi

__all__ = [
    "abs", "absolute", "add", "amax", "amin", "argmax", "argmin", "argsort",
    "clip", "cos", "divide", "dot", "exp", "linalg", "log", "matmul", "max",
    "maximum", "mean", "min", "minimum", "multiply", "negative", "prod",
    "product", "random", "sin", "sort", "sqrt", "square", "std", "subtract",
    "sum", "tanh", "true_divide", "var",
]
