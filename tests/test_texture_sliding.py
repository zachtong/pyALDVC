"""The sliding-window estimator: kept as the reference the report compares the shipped one against."""

import numpy as np
import pytest

from al_dvc.synthetic import generate_speckle_volume
from al_dvc.texture import (
    analyse_range,
    analyse_texture,
    centred_window,
    lag_reach,
    sliding_autocorrelation,
    whole_box,
)


def test_the_window_sits_in_the_middle_and_the_shifts_follow():
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


def test_no_texture():
    flat = np.full((24, 24, 24), 2.0, dtype=np.float32)
    ac, _ = sliding_autocorrelation(flat, whole_box(flat.shape), 8)
    assert ac.status == "no_texture"
    res = analyse_range(flat, whole_box(flat.shape), 8)
    assert res.status == "no_texture" and res.length("radial") is None
