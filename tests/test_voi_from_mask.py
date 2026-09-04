"""The analysed box derived from a region of interest, and validation with an unset VOI."""

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.data_structures import VOI_EXTRA_MARGIN, VOIRange, voi_from_mask
from al_dvc.utils.validation import validate_para_against_volume

SHAPE = (60, 70, 80)  # (nz, ny, nx)


def _mask(z, y, x):
    m = np.zeros(SHAPE, dtype=bool)
    m[z[0] : z[1], y[0] : y[1], x[0] : x[1]] = True
    return m


def test_box_is_bounding_box_plus_margins():
    m = _mask((20, 30), (25, 45), (30, 60))
    voi = voi_from_mask(m, winsize=(16, 16, 16), search_radius=(3, 3, 3))
    margin = 8 + 3 + VOI_EXTRA_MARGIN
    assert voi == VOIRange(x=(30 - margin, 59 + margin), y=(25 - margin, 44 + margin), z=(20 - margin, 29 + margin))
    box = voi.clamp(SHAPE)
    assert box.extent == tuple(hi - lo + 1 for lo, hi in (box.z, box.y, box.x))


def test_box_is_clamped_to_the_volume():
    m = _mask((0, 5), (0, 5), (75, 80))
    voi = voi_from_mask(m, winsize=(32, 32, 32), search_radius=(5, 5, 5))
    assert voi.z[0] == 0 and voi.y[0] == 0 and voi.x[1] == SHAPE[2] - 1
    assert voi.x[0] == 75 - (16 + 5 + VOI_EXTRA_MARGIN)


def test_empty_or_covering_mask_means_whole_volume():
    assert voi_from_mask(np.zeros(SHAPE, dtype=bool), 16, 3) is None
    assert voi_from_mask(np.ones(SHAPE, dtype=bool), 16, 3) is None
    nearly = _mask((2, 58), (2, 68), (2, 78))  # margins reach every edge
    assert voi_from_mask(nearly, 16, 3) is None


def test_per_axis_parameters_and_bad_input():
    m = _mask((30, 40), (25, 45), (30, 60))
    voi = voi_from_mask(m, winsize=(8, 16, 32), search_radius=(1, 2, 3))
    assert voi.x[0] == 30 - (4 + 1 + VOI_EXTRA_MARGIN)
    assert voi.y[0] == 25 - (8 + 2 + VOI_EXTRA_MARGIN)
    assert voi.z[0] == 30 - (16 + 3 + VOI_EXTRA_MARGIN)
    with pytest.raises(ValueError):
        voi_from_mask(np.ones((4, 4), dtype=bool), 8, 1)


def test_whole_volume_sentinel():
    assert VOIRange().is_whole
    assert not VOIRange(x=(0, 10)).is_whole


def test_validation_accepts_unset_voi():
    para = dvcpara_default()
    validate_para_against_volume(para.__class__(**{**para.__dict__, "voi": None}), SHAPE)
    small = VOIRange(x=(0, 20), y=(0, 69), z=(0, 59))
    with pytest.raises(ValueError, match="too large for the VOI extent"):
        validate_para_against_volume(para.__class__(**{**para.__dict__, "voi": small, "winsize": (32, 32, 32)}), SHAPE)
