from __future__ import annotations

import operator
import builtins as _builtins
from functools import reduce

import numpy as _np

from ._lib import addr, f64, lib, parallel_calls, parallel_workers


def _dtype_ok(dtype) -> None:
    if dtype is not None and _np.dtype(dtype) != _np.dtype(_np.float64):
        raise TypeError("the Mojo kernel subset currently supports float64 output")


def _out(result, output):
    if output is None:
        if isinstance(result, _np.ndarray) and result.ndim == 0:
            return result[()]
        return result
    target = output[0] if isinstance(output, tuple) else output
    if target is None:
        return result
    _np.copyto(target, result, casting="same_kind")
    return target


def _binary(name, x1, x2, output=None, *, dtype=None, where=True):
    _dtype_ok(dtype)
    if where is not True:
        raise NotImplementedError("where masks are not covered")
    a, b = _np.broadcast_arrays(_np.asarray(x1), _np.asarray(x2))
    a = f64(a)
    b = f64(b)
    result = _np.empty(a.shape, dtype=_np.float64)
    if result.size:
        getattr(lib(), name)(addr(a), addr(b), addr(result), result.size)
    return _out(result, output)


def _unary(name, x, output=None, *, dtype=None, where=True):
    _dtype_ok(dtype)
    if where is not True:
        raise NotImplementedError("where masks are not covered")
    a = f64(x)
    result = _np.empty(a.shape, dtype=_np.float64)
    if result.size:
        getattr(lib(), name)(addr(a), addr(result), result.size)
    return _out(result, output)


def add(x1, x2, out=None, *, where=True, dtype=None, **kwargs):
    return _binary("mn_add", x1, x2, out, dtype=dtype, where=where)


def subtract(x1, x2, out=None, *, where=True, dtype=None, **kwargs):
    return _binary("mn_subtract", x1, x2, out, dtype=dtype, where=where)


def multiply(x1, x2, out=None, *, where=True, dtype=None, **kwargs):
    return _binary("mn_multiply", x1, x2, out, dtype=dtype, where=where)


def divide(x1, x2, out=None, *, where=True, dtype=None, **kwargs):
    return _binary("mn_divide", x1, x2, out, dtype=dtype, where=where)


true_divide = divide


def maximum(x1, x2, out=None, *, where=True, dtype=None, **kwargs):
    return _binary("mn_maximum", x1, x2, out, dtype=dtype, where=where)


def minimum(x1, x2, out=None, *, where=True, dtype=None, **kwargs):
    return _binary("mn_minimum", x1, x2, out, dtype=dtype, where=where)


def negative(x, out=None, *, where=True, dtype=None, **kwargs):
    return _unary("mn_negative", x, out, dtype=dtype, where=where)


def absolute(x, out=None, *, where=True, dtype=None, **kwargs):
    return _unary("mn_absolute", x, out, dtype=dtype, where=where)


abs = absolute


def square(x, out=None, *, where=True, dtype=None, **kwargs):
    return _unary("mn_square", x, out, dtype=dtype, where=where)


def sqrt(x, out=None, *, where=True, dtype=None, **kwargs):
    return _unary("mn_sqrt", x, out, dtype=dtype, where=where)


def exp(x, out=None, *, where=True, dtype=None, **kwargs):
    return _unary("mn_exp", x, out, dtype=dtype, where=where)


def log(x, out=None, *, where=True, dtype=None, **kwargs):
    return _unary("mn_log", x, out, dtype=dtype, where=where)


def sin(x, out=None, *, where=True, dtype=None, **kwargs):
    return _unary("mn_sin", x, out, dtype=dtype, where=where)


def cos(x, out=None, *, where=True, dtype=None, **kwargs):
    return _unary("mn_cos", x, out, dtype=dtype, where=where)


def tanh(x, out=None, *, where=True, dtype=None, **kwargs):
    return _unary("mn_tanh", x, out, dtype=dtype, where=where)


def clip(a, a_min=None, a_max=None, out=None, **kwargs):
    if a_min is None or a_max is None:
        raise NotImplementedError("both a_min and a_max are required")
    source = f64(a)
    result = _np.empty_like(source)
    if result.size:
        lib().mn_clip(addr(source), addr(result), result.size, a_min, a_max)
    return _out(result, out)


def _axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    raw = (axis,) if isinstance(axis, int) else tuple(axis)
    normalized = tuple(_normalize_axis(a, ndim) for a in raw)
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate value in 'axis'")
    return normalized


def _normalize_axis(axis, ndim):
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise _np.exceptions.AxisError(axis, ndim=ndim)
    return normalized


def _reduction_layout(a, axis):
    axes = _axes(axis, a.ndim)
    keep = tuple(i for i in range(a.ndim) if i not in axes)
    perm = keep + axes
    moved = _np.ascontiguousarray(a.transpose(perm) if perm else a, dtype=_np.float64)
    rows = reduce(operator.mul, (a.shape[i] for i in keep), 1)
    cols = reduce(operator.mul, (a.shape[i] for i in axes), 1)
    shape = tuple(a.shape[i] for i in keep)
    return moved.reshape(rows, cols), axes, shape


def _reduce(
    a,
    operation,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
    ddof=0,
):
    _dtype_ok(dtype)
    if where is not True:
        raise NotImplementedError("where masks are not covered")
    source = _np.asarray(a)
    matrix, axes, shape = _reduction_layout(source, axis)
    if not axes:
        return _out(f64(source, copy=True), out)
    if matrix.shape[1] == 0:
        if operation == 0:
            result = _np.zeros(shape, dtype=_np.float64)
        elif operation == 1:
            result = _np.ones(shape, dtype=_np.float64)
        elif operation in (4, 5):
            result = _np.full(shape, _np.nan, dtype=_np.float64)
        elif initial is not None:
            result = _np.full(shape, initial, dtype=_np.float64)
        else:
            raise ValueError("zero-size array to reduction operation which has no identity")
    elif operation == 5 and ddof >= matrix.shape[1]:
        result = _np.full(shape, _np.nan, dtype=_np.float64)
    else:
        result = _np.empty(matrix.shape[0], dtype=_np.float64)
        lib().mn_reduce(
            addr(matrix), addr(result), matrix.shape[0], matrix.shape[1], operation, ddof
        )
        result = result.reshape(shape)
        if initial is not None:
            if operation == 0:
                result += initial
            elif operation == 1:
                result *= initial
            elif operation == 2:
                result = minimum(result, initial)
            elif operation == 3:
                result = maximum(result, initial)
    if keepdims:
        final_shape = tuple(1 if i in axes else source.shape[i] for i in range(source.ndim))
        result = result.reshape(final_shape)
    return _out(result, out)


def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True):
    return _reduce(a, 0, axis, dtype, out, keepdims, initial, where)


def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True):
    return _reduce(a, 1, axis, dtype, out, keepdims, initial, where)


product = prod


def min(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    return _reduce(a, 2, axis, None, out, keepdims, initial, where)


amin = min


def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    return _reduce(a, 3, axis, None, out, keepdims, initial, where)


amax = max


def mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    return _reduce(a, 4, axis, dtype, out, keepdims, None, where)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
    return _reduce(a, 5, axis, dtype, out, keepdims, None, where, ddof)


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
    result = var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
    return sqrt(result, out=out)


def _argreduce(a, operation, axis=None, out=None, keepdims=False):
    source = _np.asarray(a)
    matrix, axes, shape = _reduction_layout(source, axis)
    if len(axes) != 1 and axis is not None:
        raise TypeError("'axis' must be None or an integer")
    if matrix.shape[1] == 0:
        raise ValueError("attempt to get argmin/argmax of an empty sequence")
    result = _np.empty(matrix.shape[0], dtype=_np.int64)
    lib().mn_argreduce(
        addr(matrix), addr(result), matrix.shape[0], matrix.shape[1], operation
    )
    result = result.reshape(shape)
    if keepdims:
        final_shape = tuple(1 if i in axes else source.shape[i] for i in range(source.ndim))
        result = result.reshape(final_shape)
    return _out(result, out)


def argmin(a, axis=None, out=None, *, keepdims=False):
    return _argreduce(a, 0, axis, out, keepdims)


def argmax(a, axis=None, out=None, *, keepdims=False):
    return _argreduce(a, 1, axis, out, keepdims)


def _sort_layout(a, axis):
    source = _np.asarray(a)
    if axis is None:
        return f64(source).reshape(1, -1), None
    normalized = _normalize_axis(axis, source.ndim)
    moved = _np.moveaxis(source, normalized, -1)
    return f64(moved).reshape(-1, moved.shape[-1]), normalized


def sort(a, axis=-1, kind=None, order=None, *, stable=None):
    if order is not None:
        raise NotImplementedError("structured-field ordering is not covered")
    if stable is True or kind in ("stable", "mergesort"):
        source = f64(a)
        indices = argsort(source, axis=axis, stable=True)
        if axis is None:
            return source.ravel()[indices]
        normalized = _normalize_axis(axis, source.ndim)
        return _np.take_along_axis(source, indices, axis=normalized)
    matrix, normalized = _sort_layout(a, axis)
    result = _np.empty_like(matrix)
    if matrix.size:
        cols = matrix.shape[1]
        if cols >= 262_144 and not _np.isnan(matrix).any():
            result[...] = matrix
            work = _np.empty_like(matrix)
            workers = _builtins.min(
                parallel_workers(), _builtins.max(1, cols // 16_384)
            )
            chunks = workers * 2
            kernel = lib().mn_sort_inplace
            for row in range(matrix.shape[0]):
                row_addr = addr(result[row])
                chunk_width = (cols + chunks - 1) // chunks
                parallel_calls(
                    kernel,
                    [
                        (
                            row_addr + start * 8,
                            _builtins.min(chunk_width, cols - start),
                        )
                        for start in range(0, cols, chunk_width)
                    ],
                )
                src_addr = row_addr
                tmp_addr = addr(work[row])
                width = chunk_width
                while width < cols:
                    merges = (cols + 2 * width - 1) // (2 * width)
                    parallel_calls(
                        lib().mn_merge_numeric,
                        [
                            (
                                src_addr,
                                tmp_addr,
                                start,
                                _builtins.min(start + width, cols),
                                _builtins.min(start + 2 * width, cols),
                            )
                            for start in range(0, cols, 2 * width)
                        ],
                    )
                    src_addr, tmp_addr = tmp_addr, src_addr
                    width *= 2
                if src_addr != row_addr:
                    _np.copyto(result[row], work[row])
        else:
            work = _np.empty_like(matrix) if cols >= 262_144 else None
            lib().mn_sort(
                addr(matrix),
                addr(result),
                addr(work) if work is not None else 0,
                matrix.shape[0],
                cols,
            )
    if axis is None:
        return result.ravel()
    moved_shape = _np.moveaxis(_np.asarray(a), normalized, -1).shape
    return _np.moveaxis(result.reshape(moved_shape), -1, normalized)


def argsort(a, axis=-1, kind=None, order=None, *, stable=None):
    if order is not None:
        raise NotImplementedError("structured-field ordering is not covered")
    matrix, normalized = _sort_layout(a, axis)
    result = _np.empty(matrix.shape, dtype=_np.int64)
    values = _np.empty_like(matrix)
    work_values = _np.empty_like(matrix)
    work_idx = _np.empty(matrix.shape, dtype=_np.int64)
    if matrix.size:
        lib().mn_argsort(
            addr(matrix),
            addr(result),
            addr(values),
            addr(work_values),
            addr(work_idx),
            matrix.shape[0],
            matrix.shape[1],
        )
    if axis is None:
        return result.ravel()
    moved_shape = _np.moveaxis(_np.asarray(a), normalized, -1).shape
    return _np.moveaxis(result.reshape(moved_shape), -1, normalized)
