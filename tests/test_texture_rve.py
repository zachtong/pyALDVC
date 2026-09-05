"""Size sweep (schedule, sampling, plateau decision) and the parameter recommendation."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from al_dvc.texture import (
    THRESHOLDS,
    analyse_texture,
    analytic_length,
    boolean_spheres,
    decide_plateau,
    recommend_parameters,
    sample_positions,
    size_schedule,
    sweep_sizes,
)

ONE_OVER_E = THRESHOLDS[0]


def test_size_schedule_clips_and_stops_at_the_box():
    assert size_schedule((100, 100, 100), 16, 16, 4) == [(16, 16, 16), (32, 32, 32), (48, 48, 48), (64, 64, 64)]
    # clipped sizes are not repeated: the sweep ends when the box is reached
    assert size_schedule((40, 60, 80), 16, 16, 8) == [(16, 16, 16), (32, 32, 32), (40, 48, 48), (40, 60, 64), (40, 60, 80)]
    assert size_schedule((20, 20, 20), 16, 16, 8) == [(16, 16, 16), (20, 20, 20)]
    assert size_schedule((50, 50, 50), (8, 16, 24), (4, 0, 8), 3) == [(8, 16, 24), (12, 16, 32), (16, 16, 40)]
    with pytest.raises(ValueError):
        size_schedule((50, 50, 50), 16, 0, 3)
    with pytest.raises(ValueError):
        size_schedule((50, 50, 50), 16, -4, 3)


def test_sample_positions_tile_first_then_random():
    rng = np.random.default_rng(0)
    tiles = sample_positions((64, 64, 64), (32, 32, 32), 8, rng)
    assert len(tiles) == 8 and len(set(tiles)) == 8
    for z0, y0, x0 in tiles:  # non-overlapping tiles
        assert z0 % 32 == 0 and y0 % 32 == 0 and x0 % 32 == 0
    spread = sample_positions((64, 64, 64), (16, 16, 16), 4, rng)  # 64 tiles, 4 spread over them
    assert len(spread) == 4 and spread[0] == (0, 0, 0) and spread[-1] == (48, 48, 48)
    extra = sample_positions((40, 40, 40), (32, 32, 32), 3, rng)  # one tile, two random overlapping
    assert len(extra) == 3 and extra[0] == (0, 0, 0)
    assert all(0 <= v <= 8 for o in extra for v in o)
    whole = sample_positions((30, 30, 30), (30, 30, 30), 5, rng)
    assert whole == [(0, 0, 0)] + [(0, 0, 0)] * 4  # the whole box: every sample is the same voxels
    with pytest.raises(ValueError):
        sample_positions((20, 20, 20), (24, 20, 20), 1, rng)


def test_decide_plateau_accepts_plateaus_and_rejects_drifts():
    sizes = np.array([16, 24, 32, 48, 64, 96, 128], dtype=float)
    flat = np.array([9.0, 10.2, 10.0, 10.1, 9.9, 10.0, 10.05])
    spread = np.full(sizes.size, 0.2)
    d = decide_plateau(sizes, flat, spread, 0.1)
    assert d.converged and d.start_index == 1 and abs(d.reference - 10.025) < 1e-9
    assert d.tolerance == pytest.approx(0.25 + 0.05 * d.reference)
    # a monotone drift is never a plateau, whatever the local variation
    drift = np.linspace(10.0, 20.0, sizes.size)
    assert not decide_plateau(sizes, drift, spread, 0.1).converged
    slow = np.linspace(10.0, 11.5, sizes.size)  # 15 % over the range, 2.5 % between neighbours
    loose = decide_plateau(sizes, slow, spread, 0.1)  # inside the default band (5 % + 0.25) from size 48 on
    assert loose.converged and loose.start_index == 3 and loose.deviations[3] <= loose.tolerance
    tight = decide_plateau(sizes, slow, spread, 0.1, tolerance_rel=0.02, tolerance_abs=0.1)
    assert not tight.converged and "span" in tight.reason  # only the last sizes fit, too little span
    # a plateau that starts too late has no size span to back it
    late = np.array([6.0, 7.0, 8.0, 9.0, 9.9, 10.0, 10.05])
    d = decide_plateau(sizes, late, spread, 0.1)
    assert d.converged and d.start_index == 4  # from 64 to 128 the sizes span 2x: enough by default
    d = decide_plateau(sizes, late, spread, 0.1, min_span=3.0)
    assert not d.converged and "span" in d.reason
    # a large spread across positions blocks the size even when the means agree
    wide = np.where(sizes < 48, 2.0, 0.1)
    d = decide_plateau(sizes, flat, wide, 0.1)
    assert d.converged and d.start_index == 3
    # missing crossings
    holes = flat.copy()
    holes[-1] = np.nan
    d = decide_plateau(sizes, holes, spread, 0.1)  # a larger size without a crossing is no evidence of stability
    assert not d.converged and "no crossing" in d.reason
    assert not decide_plateau(sizes, np.full(sizes.size, np.nan), spread, 0.1).converged


def test_sweep_on_the_boolean_model_settles_near_the_analytic_length():
    radius, phi = 6.0, 0.3
    vol, _ = boolean_spheres((128, 128, 128), radius, phi, seed=11)
    sweep = sweep_sizes(vol, sizes=size_schedule((128, 128, 128), 16, 16, 8), samples_per_size=4, seed=1)
    assert [lvl.size for lvl in sweep.levels][-1] == (128, 128, 128)
    assert all(len(lvl.samples) >= 1 for lvl in sweep.levels)
    assert len(sweep.levels[0].samples) == 4 and len(sweep.levels[-1].samples) == 1  # the whole box once
    d = sweep.decisions[ONE_OVER_E]
    truth = analytic_length(radius, phi, ONE_OVER_E)
    assert d.converged, d.reason
    assert abs(d.reference - truth) < 0.4
    assert sweep.sizes[d.start_index] <= 48  # a 1/e length of 5 voxels settles in a few subsets' worth of volume
    means = sweep.means(ONE_OVER_E)
    assert np.isfinite(means).all() and np.all(sweep.stds(ONE_OVER_E)[:-1] >= 0)
    assert sweep.settings["axis"] == "radial" and sweep.settings["region"] == ((0, 128), (0, 128), (0, 128))


def test_sweep_respects_mask_progress_and_stop():
    vol, _ = boolean_spheres((64, 64, 64), 5.0, 0.3, seed=2)
    mask = np.zeros(vol.shape, dtype=bool)
    mask[8:56, 4:60, 10:58] = True
    calls = []
    sweep = sweep_sizes(
        vol, mask=mask, sizes=[(16, 16, 16), (32, 32, 32)], samples_per_size=2, progress=lambda f, m: calls.append(f)
    )
    assert sweep.settings["region"] == ((8, 56), (4, 60), (10, 58))
    assert calls and calls[-1] == pytest.approx(1.0) and len(sweep.levels) == 2
    stopped = sweep_sizes(vol, sizes=[(16, 16, 16), (32, 32, 32), (64, 64, 64)], samples_per_size=2, stop=lambda: True)
    assert len(stopped.levels) <= 1
    with pytest.raises(ValueError):
        sweep_sizes(vol, axis="diagonal")


def test_recommendation_follows_the_directional_lengths():
    rng = np.random.default_rng(3)
    aniso = gaussian_filter(rng.normal(size=(64, 64, 64)), sigma=(4.0, 1.5, 1.5))  # long along z
    res = analyse_texture(aniso, max_lag=20)
    rec = recommend_parameters(res)
    assert rec.factor == 2.5 and all(e % 2 == 0 for e in rec.subset) and all(s % 2 == 0 for s in rec.step)
    assert rec.subset[2] > rec.subset[0] and rec.subset[0] == rec.subset[1]
    assert rec.subset[2] >= 2.5 * res.length("z") - 2 and rec.step[2] == max(2, rec.subset[2] // 2 + rec.subset[2] // 2 % 2)
    assert rec.basis["z"] == res.length("z")
    big = recommend_parameters(res, factor=40.0)
    assert big.subset[2] == 128 and all(8 <= e <= 128 for e in big.subset)  # clamped to the largest edge
    z, y, x = np.mgrid[0:48, 0:48, 0:48]
    wave = np.cos(2 * np.pi * x / 10.0) + 0.05 * rng.normal(size=(48, 48, 48))
    periodic = recommend_parameters(analyse_texture(wave, max_lag=20))
    assert periodic.subset[0] >= 16 and any("periodic" in n for n in periodic.notes)
    flat = recommend_parameters(analyse_texture(np.full((16, 16, 16), 3.0)))
    assert flat.subset == (16, 16, 16) and any("no correlation length" in n for n in flat.notes)
    with pytest.raises(ValueError):
        recommend_parameters(res, factor=0)
