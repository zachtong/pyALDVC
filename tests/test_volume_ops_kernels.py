"""Numba preprocessing kernels (normalisation, gradients) against the NumPy references."""

import numpy as np
import pytest

from al_dvc.core.data_structures import VOIRange
from al_dvc.io.volume_ops import (
    GRADIENT_BORDER,
    compute_gradients,
    compute_gradients_np,
    normalize_volume,
    voi_mean_std,
)


def test_gradients_match_reference():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(12, 14, 16)).astype(np.float32)
    grads = compute_gradients(f)
    refs = compute_gradients_np(f)
    b = GRADIENT_BORDER
    for a, r in zip(grads, refs):
        assert a.dtype == np.float32 and a.shape == f.shape
        assert np.allclose(a, r, atol=2e-6)
        assert not a[:b].any() and not a[-b:].any()
        assert not a[:, :b].any() and not a[:, -b:].any()
        assert not a[:, :, :b].any() and not a[:, :, -b:].any()
    # exact for a linear ramp away from the border
    z, y, x = np.meshgrid(np.arange(12), np.arange(14), np.arange(16), indexing="ij")
    lin = (2.0 * x - 3.0 * y + 0.5 * z).astype(np.float32)
    gx, gy, gz = compute_gradients(lin)
    inner = (slice(b, -b),) * 3
    assert np.allclose(gx[inner], 2.0, atol=1e-5)
    assert np.allclose(gy[inner], -3.0, atol=1e-5)
    assert np.allclose(gz[inner], 0.5, atol=1e-5)


def test_gradients_small_volume_and_errors():
    f = np.random.default_rng(1).normal(size=(5, 6, 7)).astype(np.float32)
    grads = compute_gradients(f)
    refs = compute_gradients_np(f)
    for a, r in zip(grads, refs):
        assert a.shape == f.shape and np.array_equal(a, r)
    with pytest.raises(ValueError):
        compute_gradients(np.zeros((4, 4), dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32, np.float64])
def test_normalize_matches_numpy(dtype):
    rng = np.random.default_rng(2)
    if np.issubdtype(dtype, np.integer):
        lo = 0 if np.issubdtype(dtype, np.unsignedinteger) else -5000
        vol = rng.integers(lo, 60000, size=(20, 24, 28)).astype(dtype)
    else:
        vol = (rng.normal(size=(20, 24, 28)) * 100.0 + 50.0).astype(dtype)
    voi = VOIRange(x=(3, 20), y=(2, 21), z=(4, 15))
    for v in (None, voi):
        out = normalize_volume(vol, v)
        patch = vol if v is None else vol[v.clamp(vol.shape).slices]
        mean = float(np.mean(patch, dtype=np.float64))
        std = float(np.std(patch, dtype=np.float64))
        ref = ((vol.astype(np.float64) - mean) / std).astype(np.float32)
        assert out.dtype == np.float32 and out.shape == vol.shape
        assert np.allclose(out, ref, atol=1e-5, rtol=1e-5)
        m, s = voi_mean_std(vol, v)
        assert np.isclose(m, mean, rtol=1e-10, atol=1e-9)
        assert np.isclose(s, std, rtol=1e-8)


def test_normalize_edge_cases():
    const = np.full((8, 9, 10), 5.0, dtype=np.float32)
    assert not normalize_volume(const).any()
    bad = np.ones((8, 9, 10), dtype=np.float32)
    bad[2, 3, 4] = np.nan
    with pytest.raises(ValueError):
        normalize_volume(bad)
    with pytest.raises(ValueError):
        normalize_volume(np.zeros((4, 4)))
    # bool / small dtypes go through the NumPy path
    b = np.random.default_rng(3).integers(0, 2, size=(8, 9, 10)).astype(bool)
    out = normalize_volume(b)
    assert np.isclose(float(out.mean()), 0.0, atol=1e-6) and np.isclose(float(out.std()), 1.0, atol=1e-5)
