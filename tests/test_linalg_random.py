import numpy as np
import pytest

import mojonumpy as mnp


@pytest.fixture
def matrices():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(7, 5))
    b = rng.normal(size=(5, 4))
    return a, b


def test_dot_and_matmul_parity(matrices):
    a, b = matrices
    assert np.allclose(mnp.matmul(a, b), np.matmul(a, b))
    assert np.allclose(mnp.dot(a, b), np.dot(a, b))
    assert np.allclose(mnp.dot(a[0], b[:, 0]), np.dot(a[0], b[:, 0]))
    assert np.allclose(mnp.dot(a, b[:, 0]), np.dot(a, b[:, 0]))
    assert np.allclose(mnp.dot(a[0], b), np.dot(a[0], b))


def test_batched_broadcast_matmul():
    rng = np.random.default_rng(9)
    a = rng.normal(size=(2, 1, 4, 3))
    b = rng.normal(size=(5, 3, 6))
    assert np.allclose(mnp.matmul(a, b), np.matmul(a, b))


def test_matmul_simd_tails():
    rng = np.random.default_rng(93)
    a = rng.normal(size=(9, 11))
    b = rng.normal(size=(11, 7))
    assert np.allclose(mnp.matmul(a, b), np.matmul(a, b))
    assert np.array_equal(
        mnp.matmul(np.empty((2, 0)), np.empty((0, 4))),
        np.matmul(np.empty((2, 0)), np.empty((0, 4))),
    )
    assert mnp.matmul(np.empty((0, 3)), np.empty((3, 4))).shape == (0, 4)


def test_norm_parity(matrices):
    a, _ = matrices
    assert mnp.linalg.norm(a) == pytest.approx(np.linalg.norm(a))
    assert np.allclose(mnp.linalg.norm(a, axis=1), np.linalg.norm(a, axis=1))


def test_solve_inverse_and_determinant():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(8, 8))
    a += np.eye(8) * 2
    b = rng.normal(size=(8, 3))
    assert np.allclose(mnp.linalg.solve(a, b), np.linalg.solve(a, b), rtol=1e-12)
    assert np.allclose(mnp.linalg.inv(a), np.linalg.inv(a), rtol=1e-12)
    assert mnp.linalg.det(a) == pytest.approx(np.linalg.det(a), rel=1e-12)


def test_cholesky_parity():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 5))
    a = x.T @ x + np.eye(5)
    assert np.allclose(mnp.linalg.cholesky(a), np.linalg.cholesky(a))
    assert np.allclose(mnp.linalg.cholesky(a, upper=True), np.linalg.cholesky(a).T)
    with pytest.raises(mnp.linalg.LinAlgError):
        mnp.linalg.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_singular_solve_raises():
    with pytest.raises(mnp.linalg.LinAlgError):
        mnp.linalg.solve(np.ones((3, 3)), np.ones(3))


def test_rng_reproducibility_shapes_and_ranges():
    a = mnp.random.default_rng(123)
    b = mnp.random.default_rng(123)
    assert np.array_equal(a.random(100), b.random(100))
    assert np.array_equal(a.standard_normal((4, 5)), b.standard_normal((4, 5)))
    integers = a.integers(-3, 8, size=(20, 10))
    assert integers.shape == (20, 10)
    assert integers.dtype == np.int64
    assert integers.min() >= -3 and integers.max() < 8
    assert isinstance(a.random(), np.float64)


def test_rng_distribution_moments():
    rng = mnp.random.default_rng(99)
    uniform = rng.uniform(-2.0, 4.0, 200_000)
    normal = rng.normal(3.0, 2.0, 200_000)
    assert np.mean(uniform) == pytest.approx(1.0, abs=0.02)
    assert np.var(uniform) == pytest.approx(3.0, abs=0.03)
    assert np.mean(normal) == pytest.approx(3.0, abs=0.02)
    assert np.std(normal) == pytest.approx(2.0, abs=0.02)


def test_parallel_normal_threshold_odd_tail_is_reproducible():
    a = mnp.random.default_rng(94)
    b = mnp.random.default_rng(94)
    actual = a.standard_normal(262_147)
    assert np.array_equal(actual, b.standard_normal(262_147))
    assert abs(float(np.mean(actual))) < 0.01
    assert abs(float(np.std(actual)) - 1.0) < 0.01


def test_module_level_random_api_and_integer_bounds():
    mnp.random.seed(123)
    first = mnp.random.random(4)
    mnp.random.seed(123)
    assert np.array_equal(first, mnp.random.random_sample(4))
    assert mnp.random.uniform(size=3).shape == (3,)
    assert mnp.random.normal(size=3).shape == (3,)
    assert mnp.random.standard_normal(3).shape == (3,)
    assert mnp.random.randint(2, 5, 10).min() >= 2
    with pytest.raises(ValueError, match="int64"):
        mnp.random.default_rng(1).integers(0, 2**63)
