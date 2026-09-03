"""Outlier detection on node-grid fields.

Universal median test (Westerweel & Scarano, Exp. Fluids 2005), port of
MATLAB ``funRemoveOutliers3.m``: for every node the residual to the median
of its 26 neighbours is normalised by the median absolute residual of the
neighbourhood (plus a noise floor ``eps``); nodes whose normalised
fluctuation magnitude exceeds ``threshold`` are outliers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from .inpaint import fill_nan_grid


def _neighbour_footprint(size: int = 3) -> NDArray[np.bool_]:
    fp = np.ones((size, size, size), dtype=bool)
    c = size // 2
    fp[c, c, c] = False  # exclude the centre (MATLAB skipIdx)
    return fp


def normalized_fluctuation(
    field: NDArray[np.float64],
    valid: NDArray[np.bool_] | None = None,
    eps: float = 0.1,
    size: int = 3,
) -> NDArray[np.float64]:
    """Normalised median fluctuation of a ``(nz, ny, nx)`` scalar grid field.

    Invalid nodes (``valid == False`` or NaN) are first inpainted so they do
    not poison the neighbourhood medians; their own fluctuation is reported
    as 0.
    """
    arr = np.array(field, dtype=np.float64, copy=True)
    if valid is not None:
        arr[~valid] = np.nan
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.zeros_like(arr)
    if nan_mask.any():
        arr = fill_nan_grid(arr)
    fp = _neighbour_footprint(size)
    med = median_filter(arr, footprint=fp, mode="nearest")
    fluct = arr - med
    med_res = median_filter(np.abs(fluct), footprint=fp, mode="nearest")
    nf = np.abs(fluct) / (med_res + eps)
    nf[nan_mask] = 0.0
    return nf


def universal_median_test(
    field: NDArray[np.float64],
    valid: NDArray[np.bool_] | None,
    threshold: float,
    eps: float = 0.1,
    size: int = 3,
) -> NDArray[np.bool_]:
    """Flag outliers of a vector grid field ``(nz, ny, nx, C)`` (or scalar).

    Returns a boolean ``(nz, ny, nx)`` array, True = outlier. Only nodes that
    are ``valid`` can be flagged.
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]
    if threshold <= 0:
        return np.zeros(arr.shape[:3], dtype=bool)
    mag2 = np.zeros(arr.shape[:3], dtype=np.float64)
    for c in range(arr.shape[3]):
        nf = normalized_fluctuation(arr[..., c], valid, eps=eps, size=size)
        mag2 += nf * nf
    flag = np.sqrt(mag2) > threshold
    if valid is not None:
        flag &= valid
    return flag


def convergence_outliers(
    n_iter: NDArray[np.int32],
    good: NDArray[np.bool_],
    sigma_factor: float = 1.0,
    min_threshold: int = 6,
) -> NDArray[np.bool_]:
    """Flag nodes that converged abnormally slowly (pyALDIC ``detect_bad_points``)."""
    flag = np.zeros(n_iter.shape, dtype=bool)
    if good.sum() < 2:
        return flag
    it = n_iter[good].astype(np.float64)
    thr = max(float(np.mean(it) + sigma_factor * np.std(it, ddof=1)), float(min_threshold))
    flag[good] = n_iter[good] > thr
    return flag
