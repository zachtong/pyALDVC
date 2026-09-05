"""Texture analysis: the autocorrelation estimator, the profiles, the crossings and the Boolean model."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate

from al_dvc.texture import (
    THRESHOLDS,
    analyse_texture,
    analysis_window,
    analytic_length,
    autocorrelation,
    boolean_correlation,
    boolean_spheres,
    correlation_length,
    directional_profiles,
    lengths,
    radial_profile,
)
from al_dvc.texture.acf import _overlap_counts

ONE_OVER_E = THRESHOLDS[0]


def _smooth_noise(shape, sigma=1.2, seed=20260905):
    return gaussian_filter(np.random.default_rng(seed).normal(size=shape), sigma=sigma)


def _reference_window_acf(vol, lag_xyz):
    """``S(h) / S(0)`` from scipy's full correlation, cut to ``[-L, L]`` around the zero lag."""
    u = np.asarray(vol, dtype=np.float64)
    u = u - u.mean()
    full = correlate(u, u, mode="full", method="fft")
    c = tuple(n - 1 for n in u.shape)
    lz, ly, lx = lag_xyz[2], lag_xyz[1], lag_xyz[0]
    cut = full[c[0] - lz : c[0] + lz + 1, c[1] - ly : c[1] + ly + 1, c[2] - lx : c[2] + lx + 1]
    return cut / full[c]


@pytest.mark.parametrize("shape", [(12, 12, 12), (13, 13, 13), (7, 12, 10), (64, 64, 64)])
def test_window_estimator_matches_scipy_full_correlation(shape):
    """Sizes 12 and 64 are the ones whose fast FFT length exceeds 2N-1, where the old scripts went wrong."""
    vol = _smooth_noise(shape)
    lag = tuple(n - 1 for n in shape[::-1])  # every lag; min_overlap=1/N keeps all of them
    ac = autocorrelation(vol, max_lag=lag, estimator="window", min_overlap=1.0 / max(shape) ** 3)
    ref = _reference_window_acf(vol, lag)
    assert ac.acf.shape == ref.shape
    assert np.nanmax(np.abs(ac.acf - ref)) < 2e-5
    assert np.nanmax(np.abs(ac.acf - ac.acf[::-1, ::-1, ::-1])) < 2e-5  # C(h) = C(-h)
    assert ac.acf[ac.centre] == 1.0 and ac.status == "ok"


def test_overlap_estimator_removes_the_finite_window_factor():
    shape = (24, 26, 28)
    vol = _smooth_noise(shape)
    lag = (10, 10, 10)
    window = autocorrelation(vol, max_lag=lag, estimator="window", min_overlap=0.05)
    overlap = autocorrelation(vol, max_lag=lag, estimator="overlap", min_overlap=0.05)
    counts = _overlap_counts(shape, lag)
    factor = counts / counts[overlap.centre]
    assert np.allclose(window.acf, overlap.acf * factor, atol=2e-5, equal_nan=True)
    # the overlap estimate along an axis does not depend on how long the box is along that axis
    a = autocorrelation(vol, max_lag=(6, 0, 0), min_overlap=0.05)
    b = autocorrelation(vol[:, :, :20], max_lag=(6, 0, 0), min_overlap=0.05)
    assert np.allclose(a.line("x"), b.line("x"), atol=0.15)


def test_masked_region_matches_the_cropped_box():
    """A rectangular mask must give the same estimate as cropping the volume to that box."""
    vol = _smooth_noise((20, 22, 24))
    mask = np.zeros(vol.shape, dtype=bool)
    mask[3:17, 4:19, 5:21] = True
    masked = autocorrelation(vol, max_lag=(5, 5, 5), mask=mask, min_overlap=0.3)
    cropped = autocorrelation(vol[3:17, 4:19, 5:21], max_lag=(5, 5, 5), min_overlap=0.3)
    assert masked.n_voxels == cropped.n_voxels
    assert np.allclose(masked.acf, cropped.acf, atol=1e-4, equal_nan=True)


def test_input_validation_and_no_texture():
    with pytest.raises(ValueError):
        autocorrelation(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        autocorrelation(np.full((6, 6, 6), np.nan))
    with pytest.raises(ValueError):
        autocorrelation(np.zeros((6, 6, 6)), estimator="magic")
    flat = autocorrelation(np.full((8, 8, 8), 7.0))
    assert flat.status == "no_texture" and np.isnan(flat.acf).all()
    res = analyse_texture(np.full((8, 8, 8), 7.0))
    assert res.status == "no_texture" and res.lengths["radial"][ONE_OVER_E].status == "invalid"
    with pytest.raises(ValueError):
        autocorrelation(np.zeros((6, 6, 6)), mask=np.zeros((6, 6, 6), dtype=bool))


def test_boolean_model_matches_its_analytic_correlation():
    radius, phi = 7.0, 0.35
    vol, centres = boolean_spheres((96, 96, 96), radius, phi, seed=3)
    assert 0.25 < vol.mean() < 0.45  # the realised solid fraction is near the target
    res = analyse_texture(vol, max_lag=24)
    radial = res.profiles["radial"]
    ok = np.isfinite(radial.mean) & (radial.distance < 2 * radius)
    expected = boolean_correlation(radial.distance[ok], radius, phi)
    assert np.max(np.abs(radial.mean[ok] - expected)) < 0.04
    for t in (ONE_OVER_E, 0.1):
        measured = res.length("radial", t)
        truth = analytic_length(radius, phi, t)
        assert measured is not None and abs(measured - truth) / truth < 0.03, (t, measured, truth)
    # the three axes agree with each other on an isotropic texture
    axes = [res.length(a) for a in ("x", "y", "z")]
    assert max(axes) - min(axes) < 0.3


def test_overlap_estimator_is_size_independent_where_the_window_one_is_not():
    """The finite-window factor shortens the 1/e length more the smaller the volume; the overlap estimator does not."""
    radius, phi = 6.0, 0.3
    vol, _ = boolean_spheres((160, 160, 160), radius, phi, seed=5)
    truth = analytic_length(radius, phi, ONE_OVER_E)
    err = {}
    for est in ("overlap", "window"):
        err[est] = []
        for n in (20, 32, 64, 160):
            res = analyse_texture(vol[:n, :n, :n], max_lag=11, estimator=est, min_overlap=0.2)
            err[est].append(res.length("radial", ONE_OVER_E) - truth)
    assert max(abs(e) for e in err["overlap"]) < 0.5  # sampling noise only, at every size
    assert err["window"][0] < -0.9 and err["window"][0] < err["window"][2] - 0.5 < 0  # a bias that grows as the box shrinks
    assert abs(err["window"][-1]) < 0.3  # and vanishes in a large volume


def test_directional_profiles_see_anisotropy_and_spacing():
    rng = np.random.default_rng(7)
    aniso = gaussian_filter(rng.normal(size=(48, 48, 48)), sigma=(4.0, 1.0, 1.0))  # long along z
    res = analyse_texture(aniso, max_lag=16)
    lz, ly, lx = (res.length(a) for a in ("z", "y", "x"))
    assert lz > 2.5 * lx and abs(lx - ly) < 0.4
    # voxel spacing: the physical length along z scales with dz, the voxel length does not
    res2 = analyse_texture(aniso, spacing=(1.0, 1.0, 2.5), max_lag=16)
    assert abs(res2.length("z") - lz) < 1e-6
    assert abs(res2.physical_lengths["z"][ONE_OVER_E].value - 2.5 * lz) < 1e-6
    assert res2.profiles["radial"].distance[1] >= 1.0  # the first shell is at the physical distance


def test_radial_profile_reports_actual_shell_radii():
    ac = autocorrelation(_smooth_noise((20, 20, 20)), max_lag=6, min_overlap=0.2)
    radial = radial_profile(ac)
    assert radial.lag[0] == 0.0 and radial.coverage[0] == 1.0
    assert abs(radial.lag[1] - 1.4164) < 1e-3  # 6 at 1, 12 at sqrt 2, 8 at sqrt 3
    assert radial.count[1] == 26
    assert np.all(np.diff(radial.lag) > 0) and np.all(radial.coverage <= 1.0)
    prof = directional_profiles(ac)
    assert set(prof) == {"x", "y", "z"} and prof["x"].mean[0] == 1.0 and len(prof["z"]) == 7


@pytest.mark.parametrize(
    ("y", "threshold", "value", "status"),
    [
        ([1, 0.8, 0.6], ONE_OVER_E, None, "not_crossed"),
        ([1, 0.3, 0.2], 0.1, None, "not_crossed"),
        ([1, 0.3, 0.2, 0.4, 0.05], 0.1, 3 + (0.4 - 0.1) / (0.4 - 0.05), "crossed"),
        ([1, 0.1, 0.1, 0.05], 0.1, 2.0, "plateau"),
        ([1, 0.2, -0.2], 0.1, 1.25, "crossed"),
        ([1, 0.3, 0.05, 0.005, 0.03], 0.01, 2 + (0.05 - 0.01) / (0.05 - 0.005), "crossed"),
        ([1, 0.5, np.nan, 0.05], 0.1, None, "not_crossed"),
        ([0.2, 0.1], 0.3, None, "invalid"),
    ],
)
def test_correlation_length_definition(y, threshold, value, status):
    c = correlation_length(np.arange(len(y)), y, threshold)
    assert c.status == status
    if value is None:
        assert c.value is None and c.reason
    else:
        assert c.value == pytest.approx(value)
    with pytest.raises(ValueError):
        correlation_length([0, 1], [1, 0], 1.5)


def test_lengths_helper_and_analysis_window():
    res = analyse_texture(_smooth_noise((30, 32, 34)), max_lag=10)
    table = lengths(res.profiles["x"], thresholds=(0.5, 0.2))
    assert set(table) == {0.5, 0.2} and table[0.5].value < table[0.2].value
    assert analysis_window((40, 50, 60)) == (slice(0, 40), slice(0, 50), slice(0, 60))
    mask = np.zeros((40, 50, 60), dtype=bool)
    mask[10:30, 5:45, 20:50] = True
    assert analysis_window((40, 50, 60), mask) == (slice(10, 30), slice(5, 45), slice(20, 50))
    small = analysis_window((40, 50, 60), max_voxels=8000)
    assert all(s.stop - s.start >= 2 for s in small) and np.prod([s.stop - s.start for s in small]) <= 8000
    with pytest.raises(ValueError):
        analysis_window((4, 4, 4), np.zeros((4, 4, 4), dtype=bool))
    res = analyse_texture(_smooth_noise((40, 50, 60)), mask=mask, max_lag=8)
    assert res.window == (slice(10, 30), slice(5, 45), slice(20, 50)) and res.status == "ok"


def test_periodicity_and_noise_floor():
    z, y, x = np.mgrid[0:48, 0:48, 0:48]
    wave = np.cos(2 * np.pi * x / 12.0) + 0.05 * np.random.default_rng(1).normal(size=(48, 48, 48))
    res = analyse_texture(wave, max_lag=20)
    assert res.periodicity is not None
    axis, dist, height = res.periodicity
    assert axis == "x" and abs(dist - 12.0) < 0.6 and height > 0.8  # the period shows on the x profile
    noise = analyse_texture(np.random.default_rng(2).normal(size=(40, 40, 40)), max_lag=15)
    assert noise.periodicity is None
    assert noise.length("radial") < 1.0 and noise.noise_floor < 0.05
