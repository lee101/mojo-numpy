import numpy as np
import pytest

import mojonumpy as mnp


@pytest.fixture
def values():
    rng = np.random.default_rng(42)
    return rng.normal(size=(31, 17))


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (mnp.add, np.add),
        (mnp.subtract, np.subtract),
        (mnp.multiply, np.multiply),
        (mnp.divide, np.divide),
        (mnp.maximum, np.maximum),
        (mnp.minimum, np.minimum),
    ],
)
def test_binary_ufunc_parity(values, ours, theirs):
    other = np.linspace(0.5, 2.0, values.shape[1])
    assert np.allclose(ours(values, other), theirs(values, other), equal_nan=True)


@pytest.mark.parametrize(
    ("ours", "theirs", "transform"),
    [
        (mnp.negative, np.negative, lambda x: x),
        (mnp.absolute, np.absolute, lambda x: x),
        (mnp.square, np.square, lambda x: x),
        (mnp.sqrt, np.sqrt, np.abs),
        (mnp.exp, np.exp, lambda x: x / 4),
        (mnp.log, np.log, lambda x: np.abs(x) + 0.1),
        (mnp.sin, np.sin, lambda x: x),
        (mnp.cos, np.cos, lambda x: x),
        (mnp.tanh, np.tanh, lambda x: x),
    ],
)
def test_unary_ufunc_parity(values, ours, theirs, transform):
    x = transform(values)
    assert np.allclose(ours(x), theirs(x), rtol=2e-9, atol=2e-9)


def test_scalar_and_out_semantics():
    assert isinstance(mnp.add(1.5, 2.5), np.float64)
    target = np.empty(4)
    returned = mnp.square(np.arange(4.0), out=target)
    assert returned is target
    assert np.array_equal(target, np.square(np.arange(4.0)))


def test_clip_parity(values):
    assert np.array_equal(mnp.clip(values, -0.5, 0.75), np.clip(values, -0.5, 0.75))


def test_noncontiguous_and_nan_inputs(values):
    x = values[:, ::2]
    y = x.copy()
    y[2, 3] = np.nan
    assert np.allclose(mnp.add(x, y), np.add(x, y), equal_nan=True)
    assert np.allclose(mnp.maximum(x, y), np.maximum(x, y), equal_nan=True)


def test_simd_tail_and_parallel_threshold():
    rng = np.random.default_rng(91)
    tail = rng.normal(size=37)
    assert np.allclose(mnp.add(tail, tail), np.add(tail, tail))
    large = rng.normal(size=262_147)
    assert np.allclose(mnp.tanh(large), np.tanh(large), rtol=2e-9, atol=2e-9)


def test_lossy_input_conversion_is_rejected():
    with pytest.raises(TypeError, match="complex"):
        mnp.add(np.array([1.0 + 2.0j]), 1.0)
    with pytest.raises(TypeError, match="exactly"):
        mnp.add(np.array([2**63 - 1], dtype=np.int64), 1.0)


def test_forwarded_array_plumbing():
    assert np.array_equal(mnp.array([1, 2]), np.array([1, 2]))
    assert mnp.asarray([1, 2]).shape == (2,)
    assert mnp.ascontiguousarray(np.arange(4)[::2]).flags.c_contiguous
    assert np.array_equal(mnp.arange(3), np.arange(3))
    assert mnp.empty(2).shape == (2,)
    assert mnp.empty_like(np.zeros(2)).shape == (2,)
    assert np.array_equal(mnp.full(2, 3), np.full(2, 3))
    assert np.array_equal(mnp.full_like(np.zeros(2), 3), np.full_like(np.zeros(2), 3))
    assert np.array_equal(mnp.ones(2), np.ones(2))
    assert np.array_equal(mnp.ones_like(np.zeros(2)), np.ones(2))
    assert np.array_equal(mnp.zeros(2), np.zeros(2))
    assert np.array_equal(mnp.zeros_like(np.ones(2)), np.zeros(2))
    assert np.array_equal(mnp.eye(2), np.eye(2))
    assert np.array_equal(mnp.linspace(0, 1, 3), np.linspace(0, 1, 3))
    assert mnp.reshape(np.arange(4), (2, 2)).shape == (2, 2)
    assert mnp.transpose(np.zeros((2, 3))).shape == (3, 2)
    assert np.array_equal(mnp.concatenate(([1], [2])), [1, 2])
    assert mnp.stack(([1], [2])).shape == (2, 1)
    assert mnp.allclose([1.0], [1.0])
    assert mnp.isclose(1.0, 1.0)
    assert mnp.array_equal([1], [1])
