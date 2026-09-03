import numpy as np
import pytest

from al_dvc.core.data_structures import VOIRange
from al_dvc.io.volume_io import (
    FileVolumeProvider,
    load_volume,
    load_volumes,
    save_volume,
    volume_info,
)
from al_dvc.io.volume_ops import (
    ListVolumeProvider,
    build_reference_bundle,
    compute_gradients,
    normalize_volume,
    prefilter_bspline,
)


def test_normalize_volume_stats():
    rng = np.random.default_rng(0)
    vol = (rng.uniform(0, 1000, (20, 22, 24))).astype(np.uint16)
    n = normalize_volume(vol)
    assert n.dtype == np.float32
    assert abs(float(n.mean())) < 1e-3
    assert abs(float(n.std()) - 1.0) < 1e-3


def test_normalize_uses_voi():
    vol = np.zeros((20, 20, 20), dtype=np.float32)
    vol[5:10, 5:10, 5:10] = 3.0
    voi = VOIRange(x=(5, 9), y=(5, 9), z=(5, 9))
    n = normalize_volume(vol, voi)
    # constant inside the VOI -> std clamps to 1, values become 0 inside
    assert np.allclose(n[5:10, 5:10, 5:10], 0.0)


def test_gradients_of_polynomial_are_exact():
    z, y, x = np.mgrid[0:24, 0:26, 0:28].astype(np.float64)
    vol = (0.5 * x**2 + 2.0 * y - 0.25 * z**2 + x * y).astype(np.float32)
    gx, gy, gz = compute_gradients(vol)
    sl = (slice(4, -4), slice(4, -4), slice(4, -4))
    assert np.allclose(gx[sl], (x + y)[sl], atol=1e-3)
    assert np.allclose(gy[sl], (2.0 + x)[sl], atol=1e-3)
    assert np.allclose(gz[sl], (-0.5 * z)[sl], atol=1e-3)
    assert np.all(gx[:3] == 0) and np.all(gx[:, :, -3:] == 0)


def test_prefilter_bspline_matches_scipy():
    from scipy.ndimage import spline_filter

    vol = np.random.default_rng(1).standard_normal((12, 13, 14)).astype(np.float32)
    c = prefilter_bspline(vol)
    ref = spline_filter(vol.astype(np.float64), order=3, mode="mirror")
    assert np.allclose(c, ref, atol=1e-4)


def test_reference_bundle_mask_shape_check():
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        build_reference_bundle(vol, np.ones((10, 10, 9), dtype=bool))
    b = build_reference_bundle(vol, None)
    assert b.mask.dtype == np.uint8 and b.mask.all()


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    vol = rng.integers(0, 65535, (8, 9, 10), dtype=np.uint16)
    for ext in (".tif", ".npy", ".mat"):
        p = tmp_path / f"v{ext}"
        save_volume(p, vol, matlab_order=(ext == ".mat"))
        back = load_volume(p)
        assert back.shape == vol.shape, ext
        assert np.array_equal(back, vol), ext
    info = volume_info(tmp_path / "v.tif")
    assert info["shape_zyx"] == (8, 9, 10)


def test_load_mat_transposes_matlab_order(tmp_path):
    from scipy.io import savemat

    vol_zyx = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    savemat(str(tmp_path / "m.mat"), {"vol": np.transpose(vol_zyx, (2, 1, 0))})  # MATLAB (x, y, z)
    back = load_volume(tmp_path / "m.mat")
    assert np.array_equal(back, vol_zyx)
    raw = load_volume(tmp_path / "m.mat", matlab_order=False)
    assert raw.shape == (4, 3, 2)


def test_load_slice_folder(tmp_path):
    import tifffile

    d = tmp_path / "slices"
    d.mkdir()
    vol = np.random.default_rng(3).integers(0, 255, (5, 6, 7), dtype=np.uint8)
    for k in range(5):
        tifffile.imwrite(str(d / f"s_{k:03d}.tif"), vol[k])
    back = load_volume(d)
    assert np.array_equal(back, vol)


def test_file_provider_streams_and_caches(tmp_path):
    rng = np.random.default_rng(4)
    paths = []
    for k in range(3):
        vol = rng.uniform(0, 1, (10, 11, 12)).astype(np.float32)
        p = tmp_path / f"f{k}.npy"
        np.save(p, vol)
        paths.append(p)
    prov = FileVolumeProvider(paths, cache_size=2)
    assert len(prov) == 3 and prov.shape == (10, 11, 12)
    a = prov.get_normalized(0)
    b = prov.get_normalized(2)
    c = prov.get_normalized(1)
    assert a.dtype == np.float32 and b.shape == (10, 11, 12) and c.shape == (10, 11, 12)
    assert prov.get_mask(1) is None
    vols = load_volumes(str(tmp_path / "f*.npy"))
    assert len(vols) == 3


def test_list_provider_checks_shapes():
    with pytest.raises(ValueError):
        ListVolumeProvider([np.zeros((5, 5, 5)), np.zeros((5, 5, 6))])
    prov = ListVolumeProvider([np.zeros((5, 5, 5)), np.ones((5, 5, 5))], masks=[None, np.ones((5, 5, 5), bool)])
    assert prov.get_mask(0) is None and prov.get_mask(1).all()
