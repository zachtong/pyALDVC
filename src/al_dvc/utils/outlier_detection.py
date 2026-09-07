"""Outlier detection on node-grid fields.

Universal median test (Westerweel & Scarano, Exp. Fluids 2005), port of
MATLAB ``funRemoveOutliers3.m``: for every node the residual to the median
of its 26 neighbours is normalised by the median absolute residual of the
neighbourhood (plus a noise floor ``eps``); nodes whose normalised
fluctuation magnitude exceeds ``threshold`` are outliers.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from .inpaint import fill_nan_grid

MIN_NEIGHBOURS = 6  # fewer reachable neighbours than this and a node cannot be judged (cut mesh)


def _at_offset(arr: NDArray, off: tuple[int, int, int], fill) -> NDArray:
    """``out[n] = arr[n + off]`` on a 3-D grid, ``fill`` where the neighbour is outside the grid."""
    out = np.full(arr.shape, fill, dtype=arr.dtype)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    for ax, o in enumerate(off):
        n = arr.shape[ax]
        if o > 0:
            dst[ax] = slice(0, n - o)
            src[ax] = slice(o, n)
        elif o < 0:
            dst[ax] = slice(-o, n)
            src[ax] = slice(0, n + o)
    out[tuple(dst)] = arr[tuple(src)]
    return out


def reachable_offsets(edge_ok: NDArray[np.bool_]) -> dict[tuple[int, int, int], NDArray[np.bool_]]:
    """For each of the 26 neighbour offsets ``(dz, dy, dx)``: ``(nz, ny, nx)`` bool, True where the neighbour is
    reached from the node through ok edges inside the 3 x 3 x 3 window (some order of the axis steps).

    ``edge_ok`` is ``(nz, ny, nx, 3)`` with the +x, +y, +z edge of every node in its last axis.
    """
    E = np.asarray(edge_ok, dtype=bool)
    step_ok = {}
    for ax in range(3):  # array axis: 0 = z, 1 = y, 2 = x
        plus = E[..., 2 - ax].copy()
        last = [slice(None)] * 3
        last[ax] = slice(E.shape[ax] - 1, E.shape[ax])
        plus[tuple(last)] = False
        step_ok[(ax, 1)] = plus
        step_ok[(ax, -1)] = _at_offset(plus, tuple(-1 if a == ax else 0 for a in range(3)), False)
    out = {}
    for off in itertools.product((-1, 0, 1), repeat=3):
        if off == (0, 0, 0):
            continue
        steps = [(ax, off[ax]) for ax in range(3) if off[ax] != 0]
        reach = np.zeros(E.shape[:3], dtype=bool)
        for order in itertools.permutations(steps):
            r = np.ones(E.shape[:3], dtype=bool)
            pos = [0, 0, 0]
            for ax, d in order:
                r &= _at_offset(step_ok[(ax, d)], tuple(pos), False)
                pos[ax] += d
            reach |= r
        out[off] = reach
    return out


def _fluctuation_across_edges(arr: NDArray[np.float64], edge_ok: NDArray[np.bool_], eps: float) -> NDArray[np.float64]:
    """The normalised median fluctuation with the neighbourhood restricted to the node's own side of a cut mesh."""
    nan_mask = np.isnan(arr)
    reach = reachable_offsets(edge_ok)
    stack = np.stack([np.where(r, _at_offset(arr, off, np.nan), np.nan) for off, r in reach.items()])
    count = np.sum(np.isfinite(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN neighbourhoods
        med = np.nanmedian(stack, axis=0)
        fluct = arr - med
        absf = np.abs(fluct)
        res_stack = np.stack([np.where(r, _at_offset(absf, off, np.nan), np.nan) for off, r in reach.items()])
        med_res = np.nanmedian(res_stack, axis=0)
    nf = absf / (med_res + eps)
    nf[nan_mask | ~np.isfinite(nf) | (count < MIN_NEIGHBOURS)] = 0.0
    return nf


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
    edge_ok: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Normalised median fluctuation of a ``(nz, ny, nx)`` scalar grid field.

    Invalid nodes (``valid == False`` or NaN) are first inpainted so they do
    not poison the neighbourhood medians; their own fluctuation is reported
    as 0. With ``edge_ok`` (``(nz, ny, nx, 3)``, a cut mesh) the neighbourhood
    holds only the nodes reachable through ok edges inside the 3 x 3 x 3 window,
    invalid nodes are left out instead of inpainted, and nodes with fewer than
    ``MIN_NEIGHBOURS`` reachable neighbours are not judged.
    """
    arr = np.array(field, dtype=np.float64, copy=True)
    if valid is not None:
        arr[~valid] = np.nan
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.zeros_like(arr)
    if edge_ok is not None:
        if size != 3:
            raise ValueError("the cut-mesh median test supports the 3 x 3 x 3 neighbourhood only")
        return _fluctuation_across_edges(arr, edge_ok, eps)
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
    edge_ok: NDArray[np.bool_] | None = None,
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
        nf = normalized_fluctuation(arr[..., c], valid, eps=eps, size=size, edge_ok=edge_ok)
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
