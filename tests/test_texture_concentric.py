"""Concentric cubes: the geometry, the overlap-corrected analysis of one cube, and the size sweep."""

import numpy as np
import pytest

from al_dvc.synthetic import generate_speckle_volume
from al_dvc.texture import (
    THRESHOLDS,
    analyse_cube,
    autocorrelation,
    box_centre,
    box_of_mask,
    concentric_boxes,
    concentric_sizes,
    cube_box,
    cube_limits,
    lengths,
    max_lag_for,
    normalise_box,
    radial_profile,
    sweep_concentric,
    whole_box,
)
from al_dvc.texture.concentric import LAG_FRACTION, MIN_OVERLAP


def test_boxes():
    assert whole_box((10, 20, 30)) == ((0, 30), (0, 20), (0, 10))
    assert normalise_box(((25, 5), (-3, 8), (0, 99)), (10, 20, 30)) == ((5, 25), (0, 8), (0, 10))
    with pytest.raises(ValueError):
        normalise_box(((5, 6), (0, 8), (0, 10)), (10, 20, 30))
    mask = np.zeros((10, 20, 30), dtype=bool)
    mask[2:5, 3:9, 4:14] = True
    assert box_of_mask(mask) == ((4, 14), (3, 9), (2, 5))
    assert box_centre(((4, 14), (3, 9), (2, 5))) == (9, 6, 3)


def test_cube_geometry_stays_inside_the_region():
    bounds = ((0, 64), (0, 48), (0, 40))
    assert cube_limits((32, 24, 20), bounds) == (64, 48, 40)  # the middle: the whole region fits
    assert cube_limits((10, 24, 20), bounds) == (20, 48, 40)  # near the -x face: twice the distance to it
    with pytest.raises(ValueError, match="outside the region"):
        cube_limits((70, 24, 20), bounds)
    assert cube_box((32, 24, 20), 16) == ((24, 40), (16, 32), (12, 28))
    assert cube_box((32, 24, 20), (16, 8, 40)) == ((24, 40), (20, 28), (0, 40))
    # every box of a schedule is inside the bounds, and the sizes stop growing where the region does
    sizes = concentric_sizes((32, 24, 20), bounds, start=16, step=16, count=8)
    assert sizes[0] == (16, 16, 16) and sizes[-1] == (64, 48, 40)
    assert len(sizes) == 4 and len({s for s in sizes}) == 4  # 16, 32, 48, 64 with the per-axis clipping
    for box in concentric_boxes((32, 24, 20), bounds, start=16, step=16, count=8):
        assert all(b0 <= v0 and v1 <= b1 for (v0, v1), (b0, b1) in zip(box, bounds))
    with pytest.raises(ValueError, match="no cube fits"):
        concentric_sizes((0, 24, 20), bounds, start=16, step=16, count=4)


def test_one_cube_is_the_overlap_corrected_estimator_of_its_own_voxels():
    vol = generate_speckle_volume((64, 64, 64), sigma=2.0, seed=3)
    box = cube_box((32, 32, 32), 48)
    res = analyse_cube(vol, box)
    assert res.status == "ok" and res.settings["estimator"] == "overlap"
    assert res.settings["size"] == (48, 48, 48) and res.settings["centre"] == (32, 32, 32)
    assert res.acf.max_lag == max_lag_for((48, 48, 48)) == (48 // LAG_FRACTION,) * 3
    assert res.settings["fill"] == 1.0 and res.acf.n_voxels == 48**3
    # the reported lag cube is complete: the corner still keeps 0.75 ** 3 = 0.42 of the pairs
    assert np.isfinite(res.acf.acf).all() and MIN_OVERLAP < 0.75**3
    # and it is exactly the estimator run on the cropped volume, nothing borrowed from outside
    (x0, x1), (y0, y1), (z0, z1) = box
    direct = autocorrelation(vol[z0:z1, y0:y1, x0:x1], max_lag=12, estimator="overlap", min_overlap=MIN_OVERLAP)
    assert np.allclose(res.acf.acf, direct.acf, equal_nan=True)


def test_the_correction_is_exactly_the_geometric_factor():
    """rho_raw / rho_corrected is the share of voxel pairs the lag keeps -- nothing else."""
    vol = generate_speckle_volume((32, 32, 32), sigma=3.0, seed=11)
    kw = {"max_lag": 8, "min_overlap": MIN_OVERLAP}
    raw = autocorrelation(vol, estimator="window", **kw).line("x")
    corrected = autocorrelation(vol, estimator="overlap", **kw).line("x")
    lags = np.arange(9)
    assert np.allclose(raw / corrected, 1.0 - lags / 32.0, atol=1e-5)


def test_the_correction_removes_the_size_dependence_that_the_raw_estimator_has():
    """The same texture must give the same length whatever the cube; uncorrected, a small cube reads short."""
    vol = generate_speckle_volume((96, 96, 96), sigma=3.0, seed=11)
    small, large = cube_box((48, 48, 48), 24), cube_box((48, 48, 48), 96)
    corrected = [analyse_cube(vol, box).length("radial") for box in (small, large)]
    assert abs(corrected[0] - corrected[1]) < 0.15 * corrected[1]
    raw = []
    for box in (small, large):
        (x0, x1), (y0, y1), (z0, z1) = box
        sub = vol[z0:z1, y0:y1, x0:x1]
        ac = autocorrelation(sub, max_lag=max_lag_for((x1 - x0,) * 3), estimator="window", min_overlap=MIN_OVERLAP)
        raw.append(lengths(radial_profile(ac), THRESHOLDS)[float(THRESHOLDS[0])].value)
    assert raw[0] < 0.9 * raw[1]  # the 24 voxel cube reads about 15 % short without the correction


def test_a_mask_only_removes_the_voxels_it_excludes():
    vol = generate_speckle_volume((48, 48, 48), sigma=2.0, seed=7)
    mask = np.ones(vol.shape, dtype=bool)
    mask[:, :, :20] = False  # a face of the cube is outside the region
    res = analyse_cube(vol, cube_box((24, 24, 24), 32), mask=mask)
    assert res.status == "ok" and 0.3 < res.settings["fill"] < 0.9
    assert res.acf.n_voxels < 32**3 and res.length("radial") is not None
    mask[:] = False
    with pytest.raises(ValueError, match="no voxel of the region"):
        analyse_cube(vol, cube_box((24, 24, 24), 32), mask=mask)


def test_the_sweep_analyses_one_cube_per_size_and_decides():
    vol = generate_speckle_volume((64, 72, 80), sigma=1.5, seed=5)
    bounds = whole_box(vol.shape)
    centre = box_centre(bounds)
    sizes = concentric_sizes(centre, bounds, start=16, step=16, count=5)
    sweep = sweep_concentric(vol, centre, bounds, start=16, step=16, count=5)
    assert [lvl.size for lvl in sweep.levels] == sizes
    assert all(len(lvl.samples) == 1 and lvl.radial is not None for lvl in sweep.levels)
    assert set(sweep.decisions) == {float(t) for t in THRESHOLDS}
    assert sweep.settings["estimator"] == "overlap" and sweep.settings["centre"] == centre
    assert sweep.settings["bounds"] == bounds and sweep.settings["count"] == 5
    # a stationary texture: the 1/e length is the same at every size, so it settles at the first
    means = sweep.means(THRESHOLDS[0])
    assert np.all(np.isfinite(means)) and float(np.std(means)) < 0.3
    d = sweep.decisions[float(THRESHOLDS[0])]
    assert d.converged and d.start_index == 0


def test_the_sweep_reports_progress_and_can_be_stopped():
    vol = generate_speckle_volume((48, 48, 48), sigma=2.0, seed=2)
    seen: list[tuple[float, str]] = []
    sweep = sweep_concentric(
        vol, (24, 24, 24), start=16, step=8, count=4, progress=lambda f, m: seen.append((f, m)), stop=lambda: len(seen) > 2
    )
    assert seen and seen[-1][1] == "done" and len(sweep.levels) < 4
