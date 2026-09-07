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

from .._numba_compat import JIT_CACHE, njit, prange


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


@njit(parallel=True, cache=JIT_CACHE)
def _plane_fit_cut_jit(U, valid, edge_ok, rx, ry, rz, hx, hy, hz, F, complete):
    """Per-node plane fit over the nodes of the window reachable through ok edges (cut mesh)."""
    nz = valid.shape[0]
    ny = valid.shape[1]
    nx = valid.shape[2]
    nc = U.shape[3]
    Wz = 2 * rz + 1
    Wy = 2 * ry + 1
    Wx = 2 * rx + 1
    W = Wz * Wy * Wx
    for n in prange(nz * ny * nx):
        iz = n // (ny * nx)
        iy = (n // nx) % ny
        ix = n % nx
        if not valid[iz, iy, ix]:
            continue
        reach = np.zeros((Wz, Wy, Wx), dtype=np.uint8)
        queue = np.empty(W, dtype=np.int64)
        reach[rz, ry, rx] = 1
        queue[0] = (rz * Wy + ry) * Wx + rx
        head = 0
        tail = 1
        while head < tail:
            lin = queue[head]
            head += 1
            wz = lin // (Wy * Wx)
            rem = lin - wz * (Wy * Wx)
            wy = rem // Wx
            wx = rem - wy * Wx
            gz = iz + wz - rz
            gy = iy + wy - ry
            gx = ix + wx - rx
            for k in range(6):
                nwz = wz
                nwy = wy
                nwx = wx
                ok = False
                if k == 0 and wx + 1 < Wx and gx + 1 < nx:
                    ok = edge_ok[gz, gy, gx, 0]
                    nwx = wx + 1
                elif k == 1 and wx - 1 >= 0 and gx - 1 >= 0:
                    ok = edge_ok[gz, gy, gx - 1, 0]
                    nwx = wx - 1
                elif k == 2 and wy + 1 < Wy and gy + 1 < ny:
                    ok = edge_ok[gz, gy, gx, 1]
                    nwy = wy + 1
                elif k == 3 and wy - 1 >= 0 and gy - 1 >= 0:
                    ok = edge_ok[gz, gy - 1, gx, 1]
                    nwy = wy - 1
                elif k == 4 and wz + 1 < Wz and gz + 1 < nz:
                    ok = edge_ok[gz, gy, gx, 2]
                    nwz = wz + 1
                elif k == 5 and wz - 1 >= 0 and gz - 1 >= 0:
                    ok = edge_ok[gz - 1, gy, gx, 2]
                    nwz = wz - 1
                if not ok or reach[nwz, nwy, nwx] != 0:
                    continue
                reach[nwz, nwy, nwx] = 1
                queue[tail] = (nwz * Wy + nwy) * Wx + nwx
                tail += 1
        A = np.zeros((4, 4))
        B = np.zeros((4, nc))
        count = 0
        for wz in range(Wz):
            gz = iz + wz - rz
            if gz < 0 or gz >= nz:
                continue
            Z = (wz - rz) * hz
            for wy in range(Wy):
                gy = iy + wy - ry
                if gy < 0 or gy >= ny:
                    continue
                Y = (wy - ry) * hy
                for wx in range(Wx):
                    gx = ix + wx - rx
                    if gx < 0 or gx >= nx or reach[wz, wy, wx] == 0 or not valid[gz, gy, gx]:
                        continue
                    X = (wx - rx) * hx
                    count += 1
                    A[0, 0] += 1.0
                    A[0, 1] += X
                    A[0, 2] += Y
                    A[0, 3] += Z
                    A[1, 1] += X * X
                    A[2, 2] += Y * Y
                    A[3, 3] += Z * Z
                    A[1, 2] += X * Y
                    A[1, 3] += X * Z
                    A[2, 3] += Y * Z
                    for c in range(nc):
                        u = U[gz, gy, gx, c]
                        B[0, c] += u
                        B[1, c] += u * X
                        B[2, c] += u * Y
                        B[3, c] += u * Z
        if count < 5:
            continue
        A[1, 0] = A[0, 1]
        A[2, 0] = A[0, 2]
        A[3, 0] = A[0, 3]
        A[2, 1] = A[1, 2]
        A[3, 1] = A[1, 3]
        A[3, 2] = A[2, 3]
        reg = 1e-12 * max(abs(A[0, 0]), 1.0)
        for i in range(4):
            A[i, i] += reg
        coef = np.linalg.solve(A, B)
        for c in range(nc):
            for j in range(3):
                F[iz, iy, ix, c, j] = coef[1 + j, c]
        complete[iz, iy, ix] = count == W


def gradient_plane_fit(
    U_grid: NDArray[np.float64],
    spacing: tuple[float, float, float],
    halfwidth: tuple[int, int, int],
    valid: NDArray[np.bool_] | None = None,
    edge_ok: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Weighted least-squares plane fit of ``U_grid`` ``(nz, ny, nx, 3)``.

    Returns ``(F_grid (nz, ny, nx, 3, 3), complete (nz, ny, nx))``. Nodes with
    fewer than 4 valid neighbours (or a singular fit) get NaN. With ``edge_ok``
    (``(nz, ny, nx, 3)``, a cut mesh) every node fits only the nodes of its
    window that it reaches through ok edges, so the fit never spans a boundary.
    """
    U = np.asarray(U_grid, dtype=np.float64)
    nz, ny, nx, nc = U.shape
    if valid is None:
        valid = np.all(np.isfinite(U), axis=-1)
    else:
        valid = np.asarray(valid, dtype=bool) & np.all(np.isfinite(U), axis=-1)
    if edge_ok is not None:
        F = np.full((nz, ny, nx, nc, 3), np.nan)
        complete = np.zeros((nz, ny, nx), dtype=np.bool_)
        rx, ry, rz = (int(r) for r in halfwidth)
        hx, hy, hz = (float(h) for h in spacing)
        _plane_fit_cut_jit(
            np.ascontiguousarray(np.where(valid[..., None], U, 0.0)),
            np.ascontiguousarray(valid, dtype=np.bool_),
            np.ascontiguousarray(edge_ok, dtype=np.bool_),
            rx,
            ry,
            rz,
            hx,
            hy,
            hz,
            F,
            complete,
        )
        return F, complete
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
    edge_ok: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Central / one-sided finite differences of ``U_grid`` ``(nz, ny, nx, 3)``; no difference across a cut edge."""
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
        if edge_ok is not None:
            ok = np.asarray(edge_ok, dtype=bool)[..., j][tuple(sl_p)]
            v_prev[tuple(sl_c)] &= ok
            v_next[tuple(sl_p)] &= ok
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
