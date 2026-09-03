"""Interpolation of node-grid fields at arbitrary points.

The node grid is regular (axes ``x0``, ``y0``, ``z0`` with constant spacing)
so ``scipy.ndimage.map_coordinates`` on index coordinates is exact and fast.
Replaces MATLAB ``interp3(..., 'makima')`` in the cumulative-displacement
composition.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates

from .inpaint import fill_nan_grid


def interp_grid_field(
    field: NDArray[np.float64],
    axes: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    query_xyz: NDArray[np.float64],
    order: int = 1,
) -> NDArray[np.float64]:
    """Sample a ``(nz, ny, nx)`` grid field at ``query_xyz`` (M, 3) points.

    Coordinates are in voxels; ``axes = (x0, y0, z0)``. Points outside the
    grid are extrapolated with the nearest edge value. NaN nodes are
    inpainted before sampling.
    """
    x0, y0, z0 = (np.asarray(a, dtype=np.float64) for a in axes)
    arr = np.asarray(field, dtype=np.float64)
    if np.isnan(arr).any():
        arr = fill_nan_grid(arr)
    q = np.asarray(query_xyz, dtype=np.float64)
    hx = x0[1] - x0[0] if len(x0) > 1 else 1.0
    hy = y0[1] - y0[0] if len(y0) > 1 else 1.0
    hz = z0[1] - z0[0] if len(z0) > 1 else 1.0
    ix = (q[:, 0] - x0[0]) / hx
    iy = (q[:, 1] - y0[0]) / hy
    iz = (q[:, 2] - z0[0]) / hz
    if order > 1:
        # Odd reflection continues the field linearly past the grid edge, so the
        # cubic spline (and its prefilter) reproduce linear fields exactly up to
        # the boundary instead of bending towards a constant extension.
        pad = tuple((10, 10) if n > 1 else (0, 0) for n in arr.shape)
        arr = np.pad(arr, pad, mode="reflect", reflect_type="odd")
        iz = iz + pad[0][0]
        iy = iy + pad[1][0]
        ix = ix + pad[2][0]
    coords = np.vstack([iz, iy, ix])
    return map_coordinates(arr, coords, order=order, mode="nearest", prefilter=(order > 1))


def smooth_grid_field(
    field: NDArray[np.float64],
    sigma: float | tuple[float, float, float],
    valid: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """NaN/mask-aware Gaussian smoothing of a grid field (sigma in nodes).

    Uses normalised convolution so masked/NaN nodes neither contribute nor
    receive weight; they are returned as NaN.
    """
    arr = np.array(field, dtype=np.float64, copy=True)
    if sigma is None or (np.isscalar(sigma) and sigma <= 0):
        return arr
    w = np.isfinite(arr)
    if valid is not None:
        w &= valid
    if not w.any():
        return arr
    a = np.where(w, arr, 0.0)
    num = gaussian_filter(a, sigma=sigma, mode="nearest")
    den = gaussian_filter(w.astype(np.float64), sigma=sigma, mode="nearest")
    out = np.full_like(arr, np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    out[~w] = np.nan
    return out
