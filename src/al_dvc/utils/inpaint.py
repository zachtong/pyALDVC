"""NaN inpainting on regular grids.

``fill_nan_grid`` reproduces MATLAB ``inpaint_nans3`` (John D'Errico) methods
0/1 -- the "spring" model -- by solving the discrete Laplace equation on the
unknown nodes with the known nodes as Dirichlet data. For very large gaps
it falls back to nearest-neighbour filling, which is O(N) and never fails.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import cg, spsolve

SPRING_MAX_UNKNOWN = 300_000


def fill_nan_nearest(grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Replace NaNs by the value of the nearest finite node (Euclidean)."""
    arr = np.array(grid, dtype=np.float64, copy=True)
    nan = np.isnan(arr)
    if not nan.any():
        return arr
    if nan.all():
        warnings.warn("fill_nan_nearest: every node is NaN, returning zeros.", stacklevel=2)
        return np.zeros_like(arr)
    idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def fill_nan_grid(grid: NDArray[np.float64], method: str = "spring") -> NDArray[np.float64]:
    """Fill NaN nodes of a ``(nz, ny, nx)`` (or 2-D / 1-D) grid.

    Args:
        grid: array with NaN at unknown nodes.
        method: ``"spring"`` (harmonic extension, default) or ``"nearest"``.
    """
    arr = np.array(grid, dtype=np.float64, copy=True)
    nan = np.isnan(arr)
    n_unknown = int(nan.sum())
    if n_unknown == 0:
        return arr
    if nan.all():
        warnings.warn("fill_nan_grid: every node is NaN, returning zeros.", stacklevel=2)
        return np.zeros_like(arr)
    if method == "nearest" or n_unknown > SPRING_MAX_UNKNOWN:
        return fill_nan_nearest(arr)

    shape = arr.shape
    flat = arr.ravel()
    n = flat.size
    unknown_idx = np.flatnonzero(nan.ravel())
    pos = -np.ones(n, dtype=np.int64)
    pos[unknown_idx] = np.arange(n_unknown)

    # neighbour pairs along every axis (6-connectivity in 3-D)
    rows: list[NDArray] = []
    cols: list[NDArray] = []
    vals: list[NDArray] = []
    rhs = np.zeros(n_unknown, dtype=np.float64)
    diag = np.zeros(n_unknown, dtype=np.float64)
    strides = np.array(arr.strides) // arr.itemsize
    idx_grid = np.arange(n).reshape(shape)
    for ax in range(arr.ndim):
        if shape[ax] < 2:
            continue
        sl_a = [slice(None)] * arr.ndim
        sl_b = [slice(None)] * arr.ndim
        sl_a[ax] = slice(0, -1)
        sl_b[ax] = slice(1, None)
        a = idx_grid[tuple(sl_a)].ravel()
        b = idx_grid[tuple(sl_b)].ravel()
        for p, q in ((a, b), (b, a)):
            # equation of unknown p gets contribution from neighbour q
            pu = pos[p]
            sel = pu >= 0
            pu = pu[sel]
            qn = q[sel]
            np.add.at(diag, pu, 1.0)
            qu = pos[qn]
            known = qu < 0
            np.add.at(rhs, pu[known], flat[qn[known]])
            rows.append(pu[~known])
            cols.append(qu[~known])
            vals.append(-np.ones(int((~known).sum()), dtype=np.float64))
    del strides
    rows.append(np.arange(n_unknown))
    cols.append(np.arange(n_unknown))
    vals.append(diag)
    A = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_unknown, n_unknown),
    ).tocsr()
    # components with no known neighbour are singular -> tiny Tikhonov term
    A = A + sparse.eye(n_unknown, format="csr") * 1e-10
    try:
        if n_unknown <= 60_000:
            x = spsolve(A.tocsc(), rhs)
        else:
            x, info = cg(A, rhs, rtol=1e-10, maxiter=2000)
            if info != 0:
                x = spsolve(A.tocsc(), rhs)
    except Exception:  # pragma: no cover - numerical safety net
        return fill_nan_nearest(arr)
    if not np.all(np.isfinite(x)):
        return fill_nan_nearest(arr)
    flat[unknown_idx] = x
    return flat.reshape(shape)


def fill_nan_nodes(values: NDArray[np.float64], grid_shape: tuple[int, int, int], method: str = "spring") -> NDArray[np.float64]:
    """Fill NaNs of a per-node array ``(N,)`` or ``(N, C)`` laid out on ``grid_shape``."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        return fill_nan_grid(v.reshape(grid_shape), method).ravel()
    out = v.copy()
    for c in range(v.shape[1]):
        out[:, c] = fill_nan_grid(v[:, c].reshape(grid_shape), method).ravel()
    return out
