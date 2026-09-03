import numpy as np
import pytest

from al_dvc.utils.grid_interp import interp_grid_field, smooth_grid_field
from al_dvc.utils.inpaint import fill_nan_grid, fill_nan_nearest, fill_nan_nodes
from al_dvc.utils.outlier_detection import convergence_outliers, universal_median_test


def test_fill_nan_grid_harmonic_reproduces_linear_field():
    z, y, x = np.mgrid[0:6, 0:7, 0:8].astype(np.float64)
    field = 0.5 * x - 0.25 * y + 0.1 * z
    holed = field.copy()
    holed[2:4, 2:5, 3:6] = np.nan
    filled = fill_nan_grid(holed)
    assert np.allclose(filled, field, atol=1e-8)
    assert np.array_equal(fill_nan_grid(field), field)


def test_fill_nan_nearest_and_all_nan():
    a = np.array([[1.0, np.nan, 3.0]])
    assert np.allclose(fill_nan_nearest(a), [[1.0, 1.0, 3.0]])
    with pytest.warns(UserWarning):
        out = fill_nan_grid(np.full((3, 3, 3), np.nan))
    assert np.all(out == 0)
    nodes = fill_nan_nodes(np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0], [5.0, 6.0]]), (2, 2, 1))
    assert np.all(np.isfinite(nodes))


def test_median_test_flags_spike():
    rng = np.random.default_rng(0)
    field = rng.normal(0, 0.05, (8, 9, 10, 3))
    field[4, 4, 5, :] += 5.0
    flag = universal_median_test(field, None, threshold=2.0)
    assert flag[4, 4, 5]
    assert flag.sum() <= 3
    assert not universal_median_test(field, None, threshold=0.0).any()


def test_convergence_outliers():
    n_iter = np.array([3, 4, 3, 5, 4, 40, 3], dtype=np.int32)
    good = np.ones(7, dtype=bool)
    flag = convergence_outliers(n_iter, good, sigma_factor=1.0, min_threshold=6)
    assert flag[5] and flag.sum() == 1


def test_interp_grid_field_linear_exact():
    x0 = np.arange(10.0, 50.0, 8.0)
    y0 = np.arange(12.0, 52.0, 8.0)
    z0 = np.arange(14.0, 46.0, 8.0)
    Z, Y, X = np.meshgrid(z0, y0, x0, indexing="ij")
    field = 0.2 * X - 0.1 * Y + 0.05 * Z
    q = np.array([[13.5, 20.2, 30.1], [40.0, 44.0, 22.0]])
    got = interp_grid_field(field, (x0, y0, z0), q, order=1)
    exact = 0.2 * q[:, 0] - 0.1 * q[:, 1] + 0.05 * q[:, 2]
    assert np.allclose(got, exact, atol=1e-10)
    got3 = interp_grid_field(field, (x0, y0, z0), q, order=3)
    assert np.allclose(got3, exact, atol=1e-5)


def test_smooth_grid_field_preserves_constant_and_nan():
    f = np.full((6, 6, 6), 2.0)
    f[0, 0, 0] = np.nan
    s = smooth_grid_field(f, 1.0)
    assert np.isnan(s[0, 0, 0]) and np.allclose(s[1:], 2.0)
    assert np.array_equal(smooth_grid_field(f, 0.0), f, equal_nan=True)
