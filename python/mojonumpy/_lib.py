from __future__ import annotations

import ctypes
import os

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
    "mn_argsort": ([I, I, I, I, I, I, I], None),
    "mn_dot": ([I, I, I], F),
    "mn_matmul": ([I, I, I, I, I, I], None),
    "mn_norm": ([I, I], F),
    "mn_cholesky": ([I, I], I),
    "mn_solve": ([I, I, I, I], I),
    "mn_det": ([I, I], F),
    "mn_random_uniform": ([I, I, I, F, F], None),
    "mn_random_normal": ([I, I, I, F, F], None),
    "mn_random_integers": ([I, I, I, I, I], None),
}

_loaded: ctypes.CDLL | None = None


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
