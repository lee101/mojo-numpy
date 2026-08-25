from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-numpy.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mn_add": ([I, I, I, I], None),
    "mn_subtract": ([I, I, I, I], None),
    "mn_multiply": ([I, I, I, I], None),
    "mn_divide": ([I, I, I, I], None),
    "mn_maximum": ([I, I, I, I], None),
    "mn_minimum": ([I, I, I, I], None),
    "mn_negative": ([I, I, I], None),
    "mn_absolute": ([I, I, I], None),
    "mn_square": ([I, I, I], None),
    "mn_sqrt": ([I, I, I], None),
    "mn_exp": ([I, I, I], None),
    "mn_log": ([I, I, I], None),
    "mn_sin": ([I, I, I], None),
    "mn_cos": ([I, I, I], None),
    "mn_tanh": ([I, I, I], None),
    "mn_clip": ([I, I, I, F, F], None),
    "mn_reduce": ([I, I, I, I, I, I], None),
    "mn_argreduce": ([I, I, I, I, I], None),
    "mn_sort": ([I, I, I, I, I], None),
    "mn_sort_inplace": ([I, I], None),
    "mn_merge_numeric": ([I, I, I, I, I], None),
    "mn_argsort": ([I, I, I, I, I, I, I], None),
    "mn_dot": ([I, I, I], F),
    "mn_matmul": ([I, I, I, I, I, I], None),
    "mn_matmul_gpu": ([I, I, I, I, I, I], I),
    "mn_norm": ([I, I], F),
    "mn_cholesky": ([I, I], I),
    "mn_solve": ([I, I, I, I], I),
    "mn_det": ([I, I], F),
    "mn_random_uniform": ([I, I, I, F, F], None),
    "mn_random_normal": ([I, I, I, F, F], None),
    "mn_random_normal_range": ([I, I, I, I, I, F, F], None),
    "mn_random_normal_advance": ([I, I], None),
    "mn_random_integers": ([I, I, I, I, I], None),
}

_loaded: ctypes.CDLL | None = None
_blas_loaded: ctypes.CDLL | None = None
_executor: ThreadPoolExecutor | None = None


def parallel_workers() -> int:
    try:
        logical = len(os.sched_getaffinity(0))
    except AttributeError:
        logical = os.cpu_count() or 1
    return max(1, logical // 2)


def parallel_calls(function, arguments) -> None:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=parallel_workers())
    futures = [_executor.submit(function, *args) for args in arguments]
    for future in futures:
        future.result()


def blas_matmul(left: np.ndarray, right: np.ndarray, result: np.ndarray) -> bool:
    global _blas_loaded
    if _blas_loaded is False:
        return False
    if _blas_loaded is None:
        try:
            _blas_loaded = ctypes.CDLL("libopenblas.so.0")
            _blas_loaded.cblas_dgemm.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                F,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
                F,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            _blas_loaded.cblas_dgemm.restype = None
        except (OSError, AttributeError):
            _blas_loaded = False
            return False
    m, k = left.shape
    n = right.shape[1]
    _blas_loaded.cblas_dgemm(
        101,
        111,
        111,
        m,
        n,
        k,
        1.0,
        addr(left),
        k,
        addr(right),
        n,
        0.0,
        addr(result),
        n,
    )
    return True


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        if not os.path.exists(LIB):
            raise ImportError("compiled library missing; run `pixi run build`")
        _loaded = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _loaded


def f64(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind == "c":
        raise TypeError("complex inputs are not covered by the float64 kernel subset")
    if source.dtype.kind == "f" and source.dtype.itemsize > 8:
        raise TypeError("floating-point inputs wider than float64 are not covered")
    if source.dtype.kind in "iu":
        converted = source.astype(np.float64)
        if source.size and not np.all(
            converted.astype(object) == source.astype(object)
        ):
            raise TypeError("integer input cannot be represented exactly as float64")
    elif source.dtype.kind not in "bfi":
        raise TypeError(f"dtype {source.dtype} is not covered by the float64 kernel subset")
    if copy:
        return np.array(source, dtype=np.float64, order="C", copy=True)
    result = np.asarray(source, dtype=np.float64)
    return result if result.flags.c_contiguous else np.ascontiguousarray(result)


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if value.dtype not in (np.dtype(np.float64), np.dtype(np.int64), np.dtype(np.uint64)):
        raise TypeError(f"unsupported FFI buffer dtype: {value.dtype}")
    if not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    address = value.ctypes.data
    if address == 0:
        raise ValueError("FFI buffers must have a non-null data pointer")
    return address
