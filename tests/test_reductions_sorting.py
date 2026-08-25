import numpy as np
import pytest

import mojonumpy as mnp


@pytest.fixture
def data():
    return np.random.default_rng(7).normal(size=(9, 7, 5))


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2)])
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (mnp.sum, np.sum),
        (mnp.prod, np.prod),
        (mnp.min, np.min),
        (mnp.max, np.max),
        (mnp.mean, np.mean),
        (mnp.var, np.var),
        (mnp.std, np.std),
    ],
)
def test_reduction_parity(data, axis, ours, theirs):
    actual = ours(data, axis=axis)
    expected = theirs(data, axis=axis)
    assert np.allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_keepdims_initial_and_out(data):
    target = np.empty((1, 7, 1))
    returned = mnp.sum(data, axis=(0, 2), keepdims=True, initial=3.5, out=target)
    assert returned is target
    assert np.allclose(target, np.sum(data, axis=(0, 2), keepdims=True, initial=3.5))
    assert np.allclose(mnp.max(data, axis=1, initial=10), np.max(data, axis=1, initial=10))


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_arg_reduction_parity(data, axis):
    assert np.array_equal(mnp.argmin(data, axis=axis), np.argmin(data, axis=axis))
    assert np.array_equal(mnp.argmax(data, axis=axis), np.argmax(data, axis=axis))


def test_nan_reduction_behavior(data):
    changed = data.copy()
    changed[3, 2, 1] = np.nan
    assert np.isnan(mnp.min(changed))
    assert mnp.argmin(changed) == np.argmin(changed)
    assert mnp.argmax(changed) == np.argmax(changed)


def test_empty_reduction_identities():
    empty = np.empty((3, 0))
    assert np.array_equal(mnp.sum(empty, axis=1), np.sum(empty, axis=1))
    assert np.array_equal(mnp.prod(empty, axis=1), np.prod(empty, axis=1))
    with pytest.raises(ValueError):
        mnp.min(empty, axis=1)
    assert np.isnan(mnp.mean(empty, axis=1)).all()
    assert np.isnan(mnp.var(empty, axis=1)).all()


def test_variance_nonpositive_degrees_of_freedom():
    assert np.isnan(mnp.var([1.0], ddof=1))
    assert np.isnan(mnp.var([1.0], ddof=2))


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_sort_parity(data, axis):
    assert np.array_equal(mnp.sort(data, axis=axis), np.sort(data, axis=axis))


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_argsort_parity(data, axis):
    actual = mnp.argsort(data, axis=axis, stable=True)
    expected = np.argsort(data, axis=axis, stable=True)
    assert np.array_equal(actual, expected)


def test_stable_sort_and_nan_placement():
    values = np.array([3.0, np.nan, -0.0, 3.0, 1.0, np.nan, 0.0])
    assert np.array_equal(mnp.sort(values), np.sort(values), equal_nan=True)
    stable = mnp.sort(values, stable=True)
    expected = np.sort(values, stable=True)
    assert np.array_equal(stable, expected, equal_nan=True)
    assert np.array_equal(np.signbit(stable), np.signbit(expected))
    assert np.array_equal(
        mnp.argsort(values, stable=True), np.argsort(values, stable=True)
    )


def test_parallel_sort_threshold_and_tail():
    values = np.random.default_rng(92).normal(size=262_147)
    values[19] = np.nan
    values[-7] = np.nan
    assert np.array_equal(mnp.sort(values), np.sort(values), equal_nan=True)


def test_parallel_sort_without_nan():
    values = np.random.default_rng(95).normal(size=262_147)
    assert np.array_equal(mnp.sort(values), np.sort(values))
