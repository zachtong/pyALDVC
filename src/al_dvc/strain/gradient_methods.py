"""Displacement-gradient estimation on the node grid.

Three estimators (MATLAB ``ComputeStrain3.m`` methods 1-3):

* :func:`gradient_plane_fit` -- weighted local plane fit over a
  ``(2r+1)^3`` node window (MATLAB ``funPlaneFit3``). On the regular grid the
  fit is a set of correlations (3-D Savitzky-Golay); invalid nodes carry
  zero weight so masks and NaNs are honoured exactly.
* :func:`gradient_fd` -- central finite differences with one-sided
  differences next to invalid nodes (MATLAB ``funDerivativeOp3``).
* FEM nodal gradient -- provided by :func:`al_dvc.solver.global_operators.nodal_gradient`.

All return ``F[..., i, j] = du_i/dx_j`` in the units of ``spacing`` and a
``complete`` flag marking nodes whose stencil was fully inside the valid set.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import correlate


def _window_offsets(halfwidth: tuple[int, int, int], spacing: tuple[float, float, float]):
    rx, ry, rz = halfwidth
    hx, hy, hz = spacing
    Z, Y, X = np.meshgrid(
        np.arange(-rz, rz + 1) * hz,
        np.arange(-ry, ry + 1) * hy,
        np.arange(-rx, rx + 1) * hx,
        indexing="ij",
    )
    return X.astype(np.float64), Y.astype(np.float64), Z.astype(np.float64)


def gradient_plane_fit(
    U_grid: NDArray[np.float64],
    spacing: tuple[float, float, float],
    halfwidth: tuple[int, int, int],
    valid: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Weighted least-squares plane fit of ``U_grid`` ``(nz, ny, nx, 3)``.

    Returns ``(F_grid (nz, ny, nx, 3, 3), complete (nz, ny, nx))``. Nodes with
    fewer than 4 valid neighbours (or a singular fit) get NaN.
    """
    U = np.asarray(U_grid, dtype=np.float64)
    nz, ny, nx, nc = U.shape
    if valid is None:
        valid = np.all(np.isfinite(U), axis=-1)
    else:
        valid = np.asarray(valid, dtype=bool) & np.all(np.isfinite(U), axis=-1)
    w = valid.astype(np.float64)
    Uw = np.where(valid[..., None], U, 0.0)
    X, Y, Z = _window_offsets(halfwidth, spacing)
    ones = np.ones_like(X)

    def corr(field: NDArray, kern: NDArray) -> NDArray:
        return correlate(field, kern, mode="constant", cval=0.0)

    # normal-equation sums (correlation of weights with polynomial kernels)
    S = {}
    kernels = {"1": ones, "x": X, "y": Y, "z": Z, "xx": X * X, "yy": Y * Y, "zz": Z * Z, "xy": X * Y, "xz": X * Z, "yz": Y * Z}
    for k, kern in kernels.items():
        S[k] = corr(w, kern)
    AtA = np.empty((nz, ny, nx, 4, 4))
    AtA[..., 0, 0] = S["1"]
    AtA[..., 0, 1] = AtA[..., 1, 0] = S["x"]
    AtA[..., 0, 2] = AtA[..., 2, 0] = S["y"]
    AtA[..., 0, 3] = AtA[..., 3, 0] = S["z"]
    AtA[..., 1, 1] = S["xx"]
    AtA[..., 2, 2] = S["yy"]
    AtA[..., 3, 3] = S["zz"]
    AtA[..., 1, 2] = AtA[..., 2, 1] = S["xy"]
    AtA[..., 1, 3] = AtA[..., 3, 1] = S["xz"]
    AtA[..., 2, 3] = AtA[..., 3, 2] = S["yz"]
    AtB = np.empty((nz, ny, nx, 4, nc))
    for c in range(nc):
        AtB[..., 0, c] = corr(Uw[..., c], ones)
        AtB[..., 1, c] = corr(Uw[..., c], X)
        AtB[..., 2, c] = corr(Uw[..., c], Y)
        AtB[..., 3, c] = corr(Uw[..., c], Z)

    n_win = float(ones.size)
    count = S["1"]
    solvable = count >= 4.5  # at least 5 points to pin a plane robustly
    F = np.full((nz, ny, nx, nc, 3), np.nan)
    if solvable.any():
        A = AtA[solvable]
        B = AtB[solvable]
        # regularise degenerate windows (coplanar points) instead of raising
        A = A + 1e-12 * np.eye(4) * np.maximum(np.abs(A[..., 0, 0])[..., None, None], 1.0)
        try:
            coef = np.linalg.solve(A, B)  # (n, 4, nc)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A.reshape(-1, 4, 4), B.reshape(-1, 4, nc), rcond=None)[0]
        grad = np.transpose(coef[:, 1:4, :], (0, 2, 1))  # (n, nc, 3): du_c/dx_j
        F[solvable] = grad
    F[~valid] = np.nan
    complete = valid & (np.abs(count - n_win) < 0.5)
    return F, complete


def gradient_fd(
    U_grid: NDArray[np.float64],
    spacing: tuple[float, float, float],
    valid: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Central / one-sided finite differences of ``U_grid`` ``(nz, ny, nx, 3)``."""
    U = np.asarray(U_grid, dtype=np.float64)
    nz, ny, nx, nc = U.shape
    if valid is None:
        valid = np.all(np.isfinite(U), axis=-1)
    else:
        valid = np.asarray(valid, dtype=bool) & np.all(np.isfinite(U), axis=-1)
    F = np.full((nz, ny, nx, nc, 3), np.nan)
    complete = valid.copy()
    for j, (ax, h) in enumerate(zip((2, 1, 0), spacing)):
        v_prev = np.zeros_like(valid)
        v_next = np.zeros_like(valid)
        sl_c = [slice(None)] * 3
        sl_p = [slice(None)] * 3
        sl_c[ax] = slice(1, None)
        sl_p[ax] = slice(0, -1)
        v_prev[tuple(sl_c)] = valid[tuple(sl_p)]
        v_next[tuple(sl_p)] = valid[tuple(sl_c)]
        Up = np.zeros_like(U)
        Un = np.zeros_like(U)
        Up[tuple(sl_c)] = U[tuple(sl_p)]
        Un[tuple(sl_p)] = U[tuple(sl_c)]
        both = valid & v_prev & v_next
        only_n = valid & v_next & ~v_prev
        only_p = valid & v_prev & ~v_next
        d = np.full((nz, ny, nx, nc), np.nan)
        d[both] = (Un[both] - Up[both]) / (2.0 * h)
        d[only_n] = (Un[only_n] - U[only_n]) / h
        d[only_p] = (U[only_p] - Up[only_p]) / h
        F[..., :, j] = d
        complete &= both
    return F, complete
