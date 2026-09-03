import numpy as np
import pytest

from al_dvc.io.volume_ops import normalize_volume
from al_dvc.mesh.grid_mesh import mesh_setup
from al_dvc.solver.integer_search import (
    auto_pyramid_levels,
    block_downsample,
    ncc_search,
    phase_correlation_shift,
    pyramid_search,
)
from al_dvc.synthetic import (
    affine_displacement,
    generate_speckle_volume,
    warp_volume_lagrangian,
)


@pytest.fixture(scope="module")
def shifted_pair():
    vol = generate_speckle_volume((64, 64, 72), sigma=2.0, seed=11)
    shift = (3.4, -2.6, 1.2)
    g = warp_volume_lagrangian(vol, affine_displacement(None, shift))
    return normalize_volume(vol), normalize_volume(g), shift


def test_block_downsample_shape():
    v = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    d = block_downsample(v, 2)
    assert d.shape == (2, 3, 4)
    assert np.isclose(d[0, 0, 0], v[:2, :2, :2].mean())


def test_phase_correlation_shift(shifted_pair):
    f, g, shift = shifted_pair
    s = phase_correlation_shift(f, g)
    assert np.allclose(s, np.round(shift), atol=1.0)


def test_ncc_search_recovers_subvoxel_shift(shifted_pair):
    f, g, shift = shifted_pair
    ax = np.arange(16.0, 50.0, 8.0)
    mesh = mesh_setup(ax, ax, ax)
    coords = mesh.coordinates.astype(np.int64)
    res = ncc_search(f, g, coords, (16, 16, 16), (6, 6, 6))
    assert res["ok"].all() and not res["clipped"].any()
    err = res["disp"] - np.array(shift)
    assert np.all(np.abs(err) < 0.25)
    assert np.all(res["cc"] > 0.9)
    assert np.all(res["pce"] > 1.0)


def test_ncc_search_flags_clipped_peaks(shifted_pair):
    f, g, shift = shifted_pair
    ax = np.arange(20.0, 46.0, 8.0)
    mesh = mesh_setup(ax, ax, ax)
    coords = mesh.coordinates.astype(np.int64)
    res = ncc_search(f, g, coords, (16, 16, 16), (2, 2, 2))
    assert res["clipped"].mean() > 0.5


def test_ncc_search_with_prior_shift(shifted_pair):
    f, g, shift = shifted_pair
    ax = np.arange(20.0, 46.0, 8.0)
    mesh = mesh_setup(ax, ax, ax)
    coords = mesh.coordinates.astype(np.int64)
    prior = np.tile(np.round(shift).astype(np.int64), (coords.shape[0], 1))
    res = ncc_search(f, g, coords, (16, 16, 16), (2, 2, 2), prior)
    assert res["ok"].all() and not res["clipped"].any()
    assert np.all(np.abs(res["disp"] - np.array(shift)) < 0.25)


def test_pyramid_recovers_large_shift():
    vol = generate_speckle_volume((72, 72, 80), sigma=2.0, seed=12)
    shift = (11.3, -9.6, 6.4)
    g = warp_volume_lagrangian(vol, affine_displacement(None, shift))
    f, g = normalize_volume(vol), normalize_volume(g)
    ax = np.arange(24.0, 50.0, 8.0)
    mesh = mesh_setup(ax, ax, ax)
    coords = mesh.coordinates.astype(np.int64)
    levels = auto_pyramid_levels(f.shape, (16, 16, 16), (4, 4, 4))
    assert levels >= 1
    info = pyramid_search(f, g, coords, mesh.grid_shape, (16, 16, 16), (4, 4, 4), levels=levels, global_shift=None)
    assert np.all(np.abs(info["disp"] - np.array(shift)) < 0.3)
    # with the global pre-shift the coarse level is unnecessary but harmless
    info2 = pyramid_search(
        f, g, coords, mesh.grid_shape, (16, 16, 16), (4, 4, 4), levels=0, global_shift=phase_correlation_shift(f, g)
    )
    assert np.all(np.abs(info2["disp"] - np.array(shift)) < 0.3)
