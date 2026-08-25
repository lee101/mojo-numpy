from __future__ import annotations

import numpy as _np

from ._lib import addr, blas_matmul, f64, lib, parallel_calls, parallel_workers


class LinAlgError(Exception):
    pass


def _matrix(value):
    result = f64(value)
    if result.ndim != 2:
        raise ValueError("expected a two-dimensional array")
    return result


def matmul(x1, x2, out=None, *, dtype=None, device="cpu", **kwargs):
    if dtype is not None and _np.dtype(dtype) != _np.dtype(_np.float64):
        raise TypeError("the Mojo kernel subset currently supports float64 output")
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    a = _np.asarray(x1)
    b = _np.asarray(x2)
    a_was_vector = a.ndim == 1
    b_was_vector = b.ndim == 1
    if a.ndim < 1 or b.ndim < 1:
        raise ValueError("matmul input operand does not have enough dimensions")
    a = a.reshape((1, a.shape[0])) if a_was_vector else a
    b = b.reshape((b.shape[0], 1)) if b_was_vector else b
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("matmul core dimension mismatch")
    batch = _np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    aa = _np.broadcast_to(a, batch + a.shape[-2:])
    bb = _np.broadcast_to(b, batch + b.shape[-2:])
    result = _np.empty(batch + (a.shape[-2], b.shape[-1]), dtype=_np.float64)
    batches = _np.ndindex(batch) if batch else [()]
    for index in batches:
        left = f64(aa[index])
        right_matrix = f64(bb[index])
        target = result[index]
        if target.size == 0:
            continue
        if left.shape[1] == 0:
            target.fill(0.0)
            continue
        gpu_bytes = left.nbytes + right_matrix.nbytes + target.nbytes
        if device == "gpu" and gpu_bytes < 2 * 1024**3 and lib().mn_matmul_gpu(
            addr(left),
            addr(right_matrix),
            addr(target),
            left.shape[0],
            left.shape[1],
            right_matrix.shape[1],
        ):
            continue
        if blas_matmul(left, right_matrix, target):
            continue
        right = f64(_np.swapaxes(right_matrix, -1, -2))
        m, k, n = left.shape[0], left.shape[1], right.shape[0]
        kernel = lib().mn_matmul
        if 2 * m * k * n >= 1_048_576 and m > 1:
            workers = min(parallel_workers(), m)
            arguments = []
            left_addr = addr(left)
            right_addr = addr(right)
            target_addr = addr(target)
            for worker in range(workers):
                start = worker * m // workers
                end = (worker + 1) * m // workers
                arguments.append(
                    (
                        left_addr + start * k * 8,
                        right_addr,
                        target_addr + start * n * 8,
                        end - start,
                        k,
                        n,
                    )
                )
            parallel_calls(kernel, arguments)
        else:
            kernel(addr(left), addr(right), addr(target), m, k, n)
    if a_was_vector:
        result = _np.squeeze(result, axis=-2)
    if b_was_vector:
        result = _np.squeeze(result, axis=-1)
    if out is not None:
        _np.copyto(out, result, casting="same_kind")
        return out
    return result[()] if result.ndim == 0 else result


def dot(a, b, out=None):
    left = _np.asarray(a)
    right = _np.asarray(b)
    if left.ndim == 0 or right.ndim == 0:
        from .core import multiply

        return multiply(left, right, out=out)
    if left.ndim <= 2 and right.ndim <= 2:
        return matmul(left, right, out=out)
    raise NotImplementedError("dot currently covers scalar, vector, and matrix operands")


def norm(x, ord=None, axis=None, keepdims=False):
    source = _np.asarray(x)
    if ord not in (None, 2, "fro"):
        raise NotImplementedError("covered norms are vector 2-norm and matrix Frobenius")
    if axis is None:
        data = f64(source)
        result = lib().mn_norm(addr(data), data.size) if data.size else 0.0
        if keepdims:
            return _np.asarray(result).reshape((1,) * source.ndim)
        return result
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    squared = _np.asarray(source, dtype=_np.float64) ** 2
    result = _np.sqrt(_np.sum(squared, axis=axes, keepdims=keepdims))
    return result


def cholesky(a, *, upper=False):
    result = _matrix(a).copy()
    if result.shape[0] != result.shape[1]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    if not lib().mn_cholesky(addr(result), result.shape[0]):
        raise LinAlgError("Matrix is not positive definite")
    return result.T if upper else result


def solve(a, b):
    matrix = _matrix(a).copy()
    if matrix.shape[0] != matrix.shape[1]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    rhs = f64(b, copy=True)
    vector = rhs.ndim == 1
    if vector:
        rhs = rhs.reshape(-1, 1)
    if rhs.ndim != 2 or rhs.shape[0] != matrix.shape[0]:
        raise ValueError("solve input operand dimension mismatch")
    if not lib().mn_solve(addr(matrix), addr(rhs), matrix.shape[0], rhs.shape[1]):
        raise LinAlgError("Singular matrix")
    return rhs[:, 0] if vector else rhs


def inv(a):
    matrix = _matrix(a)
    if matrix.shape[0] != matrix.shape[1]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    return solve(matrix, _np.eye(matrix.shape[0], dtype=_np.float64))


def det(a):
    matrix = _matrix(a).copy()
    if matrix.shape[0] != matrix.shape[1]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    return lib().mn_det(addr(matrix), matrix.shape[0])
