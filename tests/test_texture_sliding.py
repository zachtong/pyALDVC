"""The sliding-window estimator and the concentric window size analysis."""

import numpy as np
import pytest

from al_dvc.synthetic import generate_speckle_volume
from al_dvc.texture import (
    analyse_range,
    analyse_texture,
    box_of_mask,
    centred_window,
    lag_reach,
    normalise_box,
    sliding_autocorrelation,
    sweep_concentric,
    sweep_sizes_concentric,
    whole_box,
)


def test_boxes():
    assert whole_box((10, 20, 30)) == ((0, 30), (0, 20), (0, 10))
    assert normalise_box(((25, 5), (-3, 8), (0, 99)), (10, 20, 30)) == ((5, 25), (0, 8), (0, 10))
    with pytest.raises(ValueError):
        normalise_box(((5, 6), (0, 8), (0, 10)), (10, 20, 30))
    mask = np.zeros((10, 20, 30), dtype=bool)
    mask[2:5, 3:9, 4:14] = True
    assert box_of_mask(mask) == ((4, 14), (3, 9), (2, 5))
    box = ((0, 64), (0, 48), (0, 40))
    win = centred_window(box, 16)
    assert win == ((24, 40), (16, 32), (12, 28)) and lag_reach(box, win) == (24, 16, 12)
    assert centred_window(box, 100) == box and lag_reach(box, box) == (0, 0, 0)


def test_sliding_agrees_with_the_overlap_estimator_and_keeps_the_pair_count():
    vol = generate_speckle_volume((48, 56, 64), sigma=2.0, seed=3)
    box = whole_box(vol.shape)
    ac, window = sliding_autocorrelation(vol, box, 32)
    assert ac.ok and ac.estimator == "sliding" and ac.n_voxels == 32**3 and ac.max_lag == (16, 12, 8)
    assert ac.acf[ac.centre] == 1.0 and window == ((16, 48), (12, 44), (8, 40))
    line = ac.line("x")
    assert line[0] == pytest.approx(1.0) and np.all(np.isfinite(line))
    a = analyse_range(vol, box, 32)
    b = analyse_texture(vol, max_lag=8)  # overlap-corrected, the whole volume
    for axis in ("x", "y", "z", "radial"):
        la, lb = a.length(axis), b.length(axis)
        assert la is not None and lb is not None and abs(la - lb) < 0.35 * lb + 0.3  # same texture, different pairs


def test_no_texture_and_window_size_analysis():
    flat = np.full((24, 24, 24), 2.0, dtype=np.float32)
    ac, _ = sliding_autocorrelation(flat, whole_box(flat.shape), 8)
    assert ac.status == "no_texture"
    vol = generate_speckle_volume((40, 48, 56), sigma=1.5, seed=5)
    box = whole_box(vol.shape)
    sizes = sweep_sizes_concentric(box, start=8, step=8, min_lag=8)
    assert sizes[0] == (8, 8, 8) and sizes[-1] == (40, 32, 24) and all(s[2] <= 24 for s in sizes)
    with pytest.raises(ValueError):
        sweep_sizes_concentric(box, start=32, step=8, min_lag=16)
    sweep = sweep_concentric(vol, box, start=8, step=8, min_lag=8)
    assert len(sweep.levels) == len(sizes) and all(len(lvl.samples) == 1 for lvl in sweep.levels)
    assert set(sweep.decisions) == {float(t) for t in (np.exp(-1.0), 0.1, 0.01)}
    assert sweep.settings["estimator"] == "sliding" and sweep.settings["range"] == box
