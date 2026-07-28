from __future__ import annotations

import secrets
import operator

import numpy as _np

from ._lib import addr, lib


def _shape(size):
    if size is None:
        return ()
    if isinstance(size, int):
        return (size,)
    return tuple(size)


def _finish(result, output=None):
    if output is not None:
        _np.copyto(output, result)
        return output
    return result[()] if result.ndim == 0 else result


class Generator:
    def __init__(self, seed=None):
        if seed is None:
            seed = secrets.randbits(64)
        if not isinstance(seed, (int, _np.integer)):
            raise TypeError("SeedSequence and BitGenerator inputs are not covered")
        self._state = _np.array([int(seed) & ((1 << 64) - 1)], dtype=_np.uint64)

    def random(self, size=None, dtype=_np.float64, out=None):
        if _np.dtype(dtype) != _np.dtype(_np.float64):
            raise TypeError("only float64 random output is covered")
        result = _np.empty(_shape(size), dtype=_np.float64)
        if result.size:
            lib().mn_random_uniform(
                addr(self._state), addr(result), result.size, 0.0, 1.0
            )
        return _finish(result, out)

    def uniform(self, low=0.0, high=1.0, size=None):
        if high < low:
            raise ValueError("high - low < 0")
        result = _np.empty(_shape(size), dtype=_np.float64)
        if result.size:
            lib().mn_random_uniform(
                addr(self._state), addr(result), result.size, low, high
            )
        return _finish(result)

    def standard_normal(self, size=None, dtype=_np.float64, out=None):
        if _np.dtype(dtype) != _np.dtype(_np.float64):
            raise TypeError("only float64 random output is covered")
        result = _np.empty(_shape(size), dtype=_np.float64)
        if result.size:
            lib().mn_random_normal(
                addr(self._state), addr(result), result.size, 0.0, 1.0
            )
        return _finish(result, out)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if scale < 0:
            raise ValueError("scale < 0")
        result = _np.empty(_shape(size), dtype=_np.float64)
        if result.size:
            lib().mn_random_normal(
                addr(self._state), addr(result), result.size, loc, scale
            )
        return _finish(result)

    def integers(self, low, high=None, size=None, dtype=_np.int64, endpoint=False):
        if _np.dtype(dtype) != _np.dtype(_np.int64):
            raise TypeError("only int64 integer output is covered")
        if high is None:
            high = low
            low = 0
        try:
            low = operator.index(low)
            high = operator.index(high)
        except TypeError as error:
            raise TypeError("low and high must be integers") from error
        if endpoint:
            if high == _np.iinfo(_np.int64).max:
                raise ValueError("endpoint=True would exceed the supported int64 range")
            high += 1
        bounds = _np.iinfo(_np.int64)
        if low < bounds.min or high > bounds.max:
            raise ValueError("low and high must fit in int64")
        if high <= low:
            raise ValueError("low >= high")
        result = _np.empty(_shape(size), dtype=_np.int64)
        if result.size:
            lib().mn_random_integers(
                addr(self._state), addr(result), result.size, low, high
            )
        return _finish(result)


def default_rng(seed=None):
    return Generator(seed)


_global = Generator()


def seed(value=None):
    global _global
    _global = Generator(value)


def random(size=None):
    return _global.random(size)


random_sample = random
sample = random
ranf = random


def uniform(low=0.0, high=1.0, size=None):
    return _global.uniform(low, high, size)


def standard_normal(size=None):
    return _global.standard_normal(size)


def normal(loc=0.0, scale=1.0, size=None):
    return _global.normal(loc, scale, size)


def randint(low, high=None, size=None, dtype=_np.int64):
    return _global.integers(low, high, size, dtype)
