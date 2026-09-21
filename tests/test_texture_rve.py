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
    assert rec.factor == 4.0 and all(e % 2 == 0 for e in rec.subset) and all(s % 2 == 0 for s in rec.step)
    assert rec.subset[2] > rec.subset[0] and rec.subset[0] == rec.subset[1]
    assert rec.subset[2] >= 4.0 * res.length("z") - 2 and rec.step[2] == max(2, rec.subset[2] // 2 + rec.subset[2] // 2 % 2)
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


# ----------------------------------------------------------------------------- the DVC Challenge 2.0 criterion
def _reference_decision(sizes, lengths, window_size, cv_threshold, abs_tolerance=0.5):
    """determine_target_decisions from the DVC Challenge 2.0 analysis scripts (rve_analysis.py),
    reduced to one target: the same loops, the same skips, the same np.std. Returns the ORIGINAL index
    of the first ROI of the earliest persistent stable window, or None."""
    eps = 1e-12
    indexed = sorted(enumerate(zip(sizes, lengths)), key=lambda item: (item[1][0], item[0]))
    order = [i for i, _ in indexed]
    vals_sorted = [v for _, (_, v) in indexed]
    window_size = max(2, window_size)
    if len(vals_sorted) < window_size:
        return None

    def values_for_window(end_idx):
        start_idx = end_idx - window_size + 1
        if start_idx < 0:
            return None
        out = []
        for idx in range(start_idx, end_idx + 1):
            v = vals_sorted[idx]
            if v is None:
                return None
            out.append(float(v))
        return out

    for end_idx in range(window_size - 1, len(vals_sorted)):
        wv = values_for_window(end_idx)
        if wv is None:
            continue
        mean_val = float(np.mean(wv))
        if mean_val <= eps:
            continue
        std_val = float(np.std(wv))
        if not (std_val / mean_val <= cv_threshold or std_val <= abs_tolerance):
            continue
        future_ok = True
        for future_end in range(end_idx + 1, len(vals_sorted)):
            fv = values_for_window(future_end)
            if fv is None:
                future_ok = False
                break
            fm = float(np.mean(fv))
            if fm <= eps:
                continue
            fs = float(np.std(fv))
            if not (fs / fm <= cv_threshold or fs <= abs_tolerance):
                future_ok = False
                break
        if future_ok:
            return order[end_idx - window_size + 1]
    return None


def test_cv_window_matches_the_reference_implementation():
    """Fuzzed against a verbatim port of the paper's loop: converged and the chosen level agree everywhere."""
    from al_dvc.texture import decide_cv_window

    rng = np.random.default_rng(7)
    checked = 0
    for trial in range(600):
        n = int(rng.integers(3, 11))
        sizes = np.sort(rng.uniform(10, 300, n)) if rng.random() < 0.8 else rng.uniform(10, 300, n)
        if rng.random() < 0.3:  # ties in size, so the index tiebreak matters
            sizes[1:3] = sizes[1]
        base = rng.uniform(2, 30)
        drift = rng.choice([0.0, 0.02, 0.1, 0.5])
        noise = rng.choice([0.0, 0.05, 0.3, 1.5])
        lengths = base + drift * base * np.exp(-np.arange(n) / 2.0) + noise * rng.normal(size=n)
        lengths = np.abs(lengths)
        if rng.random() < 0.3:
            lengths[rng.integers(0, n)] = np.nan
        w = int(rng.choice([2, 3, 4]))
        tol = float(rng.choice([0.05, 0.10, 0.20]))
        expected = _reference_decision(list(sizes), [None if np.isnan(v) else float(v) for v in lengths], w, tol)
        got = decide_cv_window(sizes, lengths, np.full(n, np.nan), 0.1, window=w, cv_tolerance=tol)
        assert got.converged == (expected is not None), (trial, sizes, lengths, w, tol, got.reason)
        if expected is not None:
            assert got.start_index == expected, (trial, sizes, lengths, w, tol)
            checked += 1
    assert checked > 100  # the fuzz did exercise the converged branch


def test_cv_window_reports_the_first_level_of_the_earliest_persistent_window():
    from al_dvc.texture import decide_cv_window

    sizes = np.array([20, 40, 60, 80, 100, 120.0])
    lengths = np.array([20.0, 12.0, 12.2, 12.1, 12.0, 12.1])
    d = decide_cv_window(sizes, lengths, np.full(6, np.nan), 0.1, window=4, cv_tolerance=0.05)
    assert d.converged and d.start_index == 1 and d.criterion == "cv_window"
    assert d.reference == pytest.approx(np.mean(lengths[1:5]))
    assert d.tolerance == pytest.approx(0.05 * d.reference)
    assert np.allclose(d.deviations, np.abs(lengths - d.reference))


def test_cv_window_persistence_rejects_a_transient_plateau():
    from al_dvc.texture import decide_cv_window

    sizes = np.arange(1, 8, dtype=float) * 20
    lengths = np.array([12.0, 12.0, 12.0, 12.0, 30.0, 12.0, 12.0])
    d = decide_cv_window(sizes, lengths, np.full(7, np.nan), 0.1, window=4, cv_tolerance=0.05)
    assert not d.converged and d.start_index is None
    # the reason is the last window's, as in the reference implementation: the transient plateau
    # at the start was rejected by the persistence check, and every later window failed on its own
    assert "exceeds" in d.reason


def test_cv_window_uses_the_population_standard_deviation():
    """The reference implementation's np.std is ddof=0; the paper's text says "sample", and the
    numbers were computed with ddof=0. On this window the two disagree about the 0.5-voxel fallback."""
    from al_dvc.texture import decide_cv_window

    lengths = np.array([3.0, 3.0, 4.0, 4.0])
    assert np.std(lengths) == pytest.approx(0.5) and np.std(lengths, ddof=1) > 0.5
    d = decide_cv_window(np.arange(4.0), lengths, np.full(4, np.nan), 0.1, window=4, cv_tolerance=0.05)
    assert d.converged and d.start_index == 0


def test_cv_window_skips_windows_with_a_missing_length():
    from al_dvc.texture import decide_cv_window

    sizes = np.arange(1, 7, dtype=float)
    lengths = np.array([12.0, np.nan, 12.0, 12.0, 12.0, 12.0])
    d = decide_cv_window(sizes, lengths, np.full(6, np.nan), 0.1, window=4, cv_tolerance=0.05)
    assert d.converged and d.start_index == 2  # the first window without the gap
    lengths = np.array([12.0, 12.0, 12.0, 12.0, np.nan, 12.0])
    d = decide_cv_window(sizes, lengths, np.full(6, np.nan), 0.1, window=4, cv_tolerance=0.05)
    assert not d.converged  # a gap in a later window breaks persistence for every earlier window
    assert "missing" in d.reason


def test_cv_window_orders_the_levels_by_size_and_returns_the_original_index():
    from al_dvc.texture import decide_cv_window

    sizes = np.array([100.0, 20.0, 60.0, 80.0, 40.0])  # analysed out of order
    lengths = np.array([12.0, 30.0, 12.1, 12.0, 12.2])  # only the 20-voxel level is off
    d = decide_cv_window(sizes, lengths, np.full(5, np.nan), 0.1, window=4, cv_tolerance=0.05)
    assert d.converged and d.start_index == 4  # the 40-voxel level, i.e. original index 4


def test_cv_window_refuses_too_few_sizes_and_reports_why():
    from al_dvc.texture import decide_cv_window

    d = decide_cv_window(np.arange(3.0), np.ones(3), np.full(3, np.nan), 0.1, window=4)
    assert not d.converged and "fewer sizes" in d.reason
    assert np.isnan(d.reference) and np.isnan(d.tolerance)


def test_cv_tolerance_defaults_follow_the_paper_per_threshold():
    from al_dvc.texture import DEFAULT_CV_TOLERANCE, cv_tolerance_for

    assert cv_tolerance_for(ONE_OVER_E) == 0.05 and cv_tolerance_for(0.1) == 0.10 and cv_tolerance_for(0.01) == 0.20
    assert cv_tolerance_for(0.1, 0.07) == 0.07  # one number for every threshold
    assert cv_tolerance_for(0.1, {0.1: 0.3}) == 0.3 and cv_tolerance_for(0.01, {0.1: 0.3}) == DEFAULT_CV_TOLERANCE[0.01]


def test_decide_dispatches_on_the_criterion_and_refuses_unknown_ones():
    from al_dvc.texture import decide

    sizes = np.arange(1, 7, dtype=float) * 20
    lengths = np.array([20.0, 12.0, 12.2, 12.1, 12.0, 12.1])
    assert decide(sizes, lengths, np.zeros(6), 0.1, "cv_window").criterion == "cv_window"
    assert decide(sizes, lengths, np.zeros(6), 0.1, "plateau").criterion == "plateau"
    with pytest.raises(ValueError):
        decide(sizes, lengths, np.zeros(6), 0.1, "median")


def test_sweep_sizes_records_the_criterion_and_its_parameters():
    from al_dvc.texture import sweep_sizes

    vol, _ = boolean_spheres((64, 64, 64), 4.0, 0.3, seed=2)
    sweep = sweep_sizes(vol, sizes=size_schedule((64, 64, 64), 16, 16, 4), samples_per_size=2, criterion="cv_window")
    assert sweep.settings["criterion"] == "cv_window" and sweep.settings["cv_window"] == 4
    assert sweep.settings["cv_tolerance"] == {ONE_OVER_E: 0.05, 0.1: 0.10, 0.01: 0.20}
    assert sweep.settings["cv_abs_tolerance"] == 0.5
    assert all(d.criterion == "cv_window" for d in sweep.decisions.values())
    with pytest.raises(ValueError):
        sweep_sizes(vol, sizes=size_schedule((64, 64, 64), 16, 16, 4), criterion="nope")
