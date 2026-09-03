"""Pure-NumPy reference implementations of the IC-GN kernels.

These are slow but transparent and are used (a) by the test-suite to check
the Numba kernels and (b) as the ``backend="numpy"`` fallback. They share
the exact same conventions as :mod:`al_dvc.solver.numba_kernels`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import (
    STATUS_CONVERGED,
    STATUS_INVALID_SUBSET,
    STATUS_MAX_ITER,
    STATUS_NAN,
    STATUS_OUT_OF_BOUNDS,
    STATUS_SINGULAR,
)
from .interp_kernels import INTERP_BSPLINE, INTERP_CUBIC, INTERP_LINEAR, SAMPLE_HI_MARGIN, SAMPLE_LO

ABS_TOL = 1e-5
LM_DAMPING_3DOF = 1e-3


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def keys_weights(t: NDArray) -> NDArray:
    t = np.asarray(t, dtype=np.float64)
    return np.stack([
        ((-0.5 * t + 1.0) * t - 0.5) * t,
        (1.5 * t - 2.5) * t * t + 1.0,
        ((-1.5 * t + 2.0) * t + 0.5) * t,
        (0.5 * t - 0.5) * t * t,
    ], axis=-1)


def bspline_weights(t: NDArray) -> NDArray:
    t = np.asarray(t, dtype=np.float64)
    t2, t3 = t * t, t * t * t
    return np.stack([
        (1 - 3 * t + 3 * t2 - t3) / 6, (4 - 6 * t2 + 3 * t3) / 6,
        (1 + 3 * t + 3 * t2 - 3 * t3) / 6, t3 / 6,
    ], axis=-1)


def sample_volume_np(vol: NDArray, z: NDArray, y: NDArray, x: NDArray, mode: int) -> NDArray:
    """Vectorised reference sampler (NaN outside the admissible domain)."""
    vol = np.asarray(vol, dtype=np.float64)
    nz, ny, nx = vol.shape
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    inside = (
        (x >= SAMPLE_LO) & (x <= nx - 1 - SAMPLE_HI_MARGIN)
        & (y >= SAMPLE_LO) & (y <= ny - 1 - SAMPLE_HI_MARGIN)
        & (z >= SAMPLE_LO) & (z <= nz - 1 - SAMPLE_HI_MARGIN)
    )
    out = np.full(x.shape, np.nan)
    if not inside.any():
        return out
    xs, ys, zs = x[inside], y[inside], z[inside]
    ix = np.minimum(np.floor(xs).astype(int), nx - 3)
    iy = np.minimum(np.floor(ys).astype(int), ny - 3)
    iz = np.minimum(np.floor(zs).astype(int), nz - 3)
    fx, fy, fz = xs - ix, ys - iy, zs - iz
    if mode == INTERP_LINEAR:
        val = np.zeros_like(xs)
        for dz, wz in ((0, 1 - fz), (1, fz)):
            for dy, wy in ((0, 1 - fy), (1, fy)):
                for dx, wx in ((0, 1 - fx), (1, fx)):
                    val += wz * wy * wx * vol[iz + dz, iy + dy, ix + dx]
    else:
        wfun = keys_weights if mode == INTERP_CUBIC else bspline_weights
        wx, wy, wz = wfun(fx), wfun(fy), wfun(fz)
        val = np.zeros_like(xs)
        for k in range(4):
            for j in range(4):
                for i in range(4):
                    val += wz[:, k] * wy[:, j] * wx[:, i] * vol[iz - 1 + k, iy - 1 + j, ix - 1 + i]
    out[inside] = val
    return out


# ---------------------------------------------------------------------------
# Subset geometry helpers
# ---------------------------------------------------------------------------


def subset_offsets(half: tuple[int, int, int]) -> tuple[NDArray, NDArray, NDArray]:
    hx, hy, hz = half
    dz, dy, dx = np.meshgrid(
        np.arange(-hz, hz + 1), np.arange(-hy, hy + 1), np.arange(-hx, hx + 1), indexing="ij",
    )
    return dx.ravel().astype(np.float64), dy.ravel().astype(np.float64), dz.ravel().astype(np.float64)


def steepest_descent(gxv, gyv, gzv, X, Y, Z) -> NDArray:
    """``(S, 12)`` steepest-descent images."""
    return np.column_stack([
        gxv * X, gxv * Y, gxv * Z, gyv * X, gyv * Y, gyv * Z, gzv * X, gzv * Y, gzv * Z, gxv, gyv, gzv,
    ])


def warp_points(P: NDArray, x0: int, y0: int, z0: int, X, Y, Z):
    A = np.eye(3) + P[:9].reshape(3, 3)
    pts = np.stack([X, Y, Z], axis=1) @ A.T
    return pts[:, 0] + x0 + P[9], pts[:, 1] + y0 + P[10], pts[:, 2] + z0 + P[11]


def compose_warp_np(P: NDArray, dP: NDArray) -> NDArray | None:
    A = np.eye(3) + P[:9].reshape(3, 3)
    dA = np.eye(3) + dP[:9].reshape(3, 3)
    if abs(np.linalg.det(dA)) < 1e-300:
        return None
    Anew = A @ np.linalg.inv(dA)
    tnew = P[9:12] - Anew @ dP[9:12]
    out = np.empty(12)
    out[:9] = (Anew - np.eye(3)).ravel()
    out[9:12] = tnew
    return out


def precompute_node_np(coord, half, f, gx, gy, gz, mask, min_valid_ratio=0.5, cond_max=1e12):
    """Reference of ``_precompute_one``: returns ``(H, meanf, bottomf, n_valid, ok)``."""
    x0, y0, z0 = (int(c) for c in coord)
    hx, hy, hz = half
    nz, ny, nx = f.shape
    if x0 - hx < 0 or x0 + hx >= nx or y0 - hy < 0 or y0 + hy >= ny or z0 - hz < 0 or z0 + hz >= nz:
        return np.zeros((12, 12)), 0.0, 1.0, 0, False
    sl = (slice(z0 - hz, z0 + hz + 1), slice(y0 - hy, y0 + hy + 1), slice(x0 - hx, x0 + hx + 1))
    m = np.asarray(mask[sl]).ravel() > 0
    X, Y, Z = subset_offsets(half)
    fv = np.asarray(f[sl], dtype=np.float64).ravel()[m]
    SD = steepest_descent(
        np.asarray(gx[sl], dtype=np.float64).ravel()[m], np.asarray(gy[sl], dtype=np.float64).ravel()[m],
        np.asarray(gz[sl], dtype=np.float64).ravel()[m], X[m], Y[m], Z[m],
    )
    H = SD.T @ SD
    n_valid = int(m.sum())
    total = m.size
    if n_valid < 27 or n_valid < min_valid_ratio * total:
        return H, 0.0, 1.0, n_valid, False
    meanf = float(fv.mean())
    bottomf = float(np.sqrt(max(np.sum((fv - meanf) ** 2), 1e-30)))
    if bottomf < 1e-10:
        return H, meanf, bottomf, n_valid, False
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return H, meanf, bottomf, n_valid, False
    d = np.diag(L)
    if d.min() <= 0 or (d.max() / d.min()) ** 2 > cond_max:
        return H, meanf, bottomf, n_valid, False
    return H, meanf, bottomf, n_valid, True


def icgn_12dof_np(P0, coord, half, f, gx, gy, gz, mask, g, mode, H, meanf, bottomf, tol, max_iter):
    """Reference 12-DOF IC-GN for one node. Returns ``(P, n_iter, status, zncc)``."""
    x0, y0, z0 = (int(c) for c in coord)
    hx, hy, hz = half
    sl = (slice(z0 - hz, z0 + hz + 1), slice(y0 - hy, y0 + hy + 1), slice(x0 - hx, x0 + hx + 1))
    m = np.asarray(mask[sl]).ravel() > 0
    X, Y, Z = subset_offsets(half)
    X, Y, Z = X[m], Y[m], Z[m]
    fv = np.asarray(f[sl], dtype=np.float64).ravel()[m]
    SD = steepest_descent(
        np.asarray(gx[sl], dtype=np.float64).ravel()[m], np.asarray(gy[sl], dtype=np.float64).ravel()[m],
        np.asarray(gz[sl], dtype=np.float64).ravel()[m], X, Y, Z,
    )
    P = np.array(P0, dtype=np.float64).copy()
    norm_init = None
    half_scale = max(half)
    zncc = np.nan
    fn = (fv - meanf) / bottomf
    for it in range(1, max_iter + 1):
        xw, yw, zw = warp_points(P, x0, y0, z0, X, Y, Z)
        gv = sample_volume_np(g, zw, yw, xw, mode)
        if np.isnan(gv).any():
            return P, it, STATUS_OUT_OF_BOUNDS, np.nan
        n_valid = gv.size
        meang = gv.mean()
        bottomg = np.sqrt(max(np.sum((gv - meang) ** 2), 1e-30))
        res = fn - (gv - meang) / bottomg
        b = bottomf * (SD.T @ res)
        norm_abs = float(np.linalg.norm(b))
        if not np.isfinite(norm_abs):
            return P, it, STATUS_NAN, np.nan
        if norm_init is None:
            norm_init = norm_abs
        norm_rel = norm_abs / norm_init if norm_init > 1e-300 else 0.0
        zncc = float(np.sum((fv - meanf) * (gv - meang)) / (bottomf * bottomg))
        if norm_rel < tol or norm_abs < ABS_TOL:
            return P, it, STATUS_CONVERGED, zncc
        dP = -np.linalg.solve(H, b)
        scaled = dP.copy()
        scaled[:9] *= half_scale
        if np.linalg.norm(scaled) < tol:
            return P, it, STATUS_CONVERGED, zncc
        Pn = compose_warp_np(P, dP)
        if Pn is None:
            return P, it, STATUS_SINGULAR, zncc
        P = Pn
    return P, max_iter, STATUS_MAX_ITER, zncc


def icgn_3dof_np(U_old, F_fixed, vdual, coord, half, f, gx, gy, gz, mask, g, mode, H, meanf, bottomf,
                 mu, tol, max_iter):
    """Reference 3-DOF IC-GN (ADMM subpb1) for one node."""
    x0, y0, z0 = (int(c) for c in coord)
    hx, hy, hz = half
    sl = (slice(z0 - hz, z0 + hz + 1), slice(y0 - hy, y0 + hy + 1), slice(x0 - hx, x0 + hx + 1))
    m = np.asarray(mask[sl]).ravel() > 0
    X, Y, Z = subset_offsets(half)
    X, Y, Z = X[m], Y[m], Z[m]
    fv = np.asarray(f[sl], dtype=np.float64).ravel()[m]
    G = np.column_stack([
        np.asarray(gx[sl], dtype=np.float64).ravel()[m], np.asarray(gy[sl], dtype=np.float64).ravel()[m],
        np.asarray(gz[sl], dtype=np.float64).ravel()[m],
    ])
    P = np.empty(12)
    P[:9] = np.asarray(F_fixed).ravel()
    P[9:] = U_old
    A = np.eye(3) + P[:9].reshape(3, 3)
    bf2 = max(bottomf**2, 1e-30)
    H3 = H[9:, 9:] * 2.0 / bf2 + mu * np.eye(3)
    Hd = H3 + LM_DAMPING_3DOF * np.max(np.diag(H3)) * np.eye(3)
    norm_init = None
    zncc = np.nan
    fn = (fv - meanf) / bottomf
    for it in range(1, max_iter + 1):
        xw, yw, zw = warp_points(P, x0, y0, z0, X, Y, Z)
        gv = sample_volume_np(g, zw, yw, xw, mode)
        if np.isnan(gv).any():
            return P[9:].copy(), it, STATUS_OUT_OF_BOUNDS, np.nan
        n_valid = gv.size
        meang = gv.mean()
        bottomg = np.sqrt(max(np.sum((gv - meang) ** 2), 1e-30))
        res = fn - (gv - meang) / bottomg
        b = bottomf * (G.T @ res)
        tb = b * 2.0 / bf2 + mu * (P[9:] - U_old - vdual)
        norm_abs = float(np.linalg.norm(tb))
        if norm_init is None:
            norm_init = norm_abs
        norm_rel = norm_abs / norm_init if norm_init > 1e-300 else 0.0
        zncc = float(np.sum((fv - meanf) * (gv - meang)) / (bottomf * bottomg))
        if norm_rel < tol or norm_abs < mu * 1e-4:
            return P[9:].copy(), it, STATUS_CONVERGED, zncc
        dt = -np.linalg.solve(Hd, tb)
        if np.linalg.norm(dt) < tol:
            return P[9:].copy(), it, STATUS_CONVERGED, zncc
        P[9:] -= A @ dt
    return P[9:].copy(), max_iter, STATUS_MAX_ITER, zncc


# ---------------------------------------------------------------------------
# Batch wrappers with the same signatures as the Numba kernels
# ---------------------------------------------------------------------------


def precompute_nodes_np(coords, hx, hy, hz, f, gx, gy, gz, mask, min_valid_ratio, cond_max):
    N = coords.shape[0]
    H_all = np.zeros((N, 12, 12))
    L_all = np.zeros((N, 12, 12))
    meanf = np.zeros(N)
    bottomf = np.ones(N)
    nvalid = np.zeros(N, dtype=np.int64)
    valid = np.zeros(N, dtype=bool)
    for n in range(N):
        H, mf, bf, nv, ok = precompute_node_np(coords[n], (hx, hy, hz), f, gx, gy, gz, mask, min_valid_ratio, cond_max)
        H_all[n] = H
        meanf[n] = mf
        bottomf[n] = bf
        nvalid[n] = nv
        valid[n] = ok
        if ok:
            L_all[n] = np.linalg.cholesky(H)
    return H_all, L_all, meanf, bottomf, nvalid, valid


def icgn_12dof_batch_np(coords, P0, hx, hy, hz, f, gx, gy, gz, mask, g, mode, H_all, meanf_all, bottomf_all,
                        valid, tol, max_iter):
    N = coords.shape[0]
    P_out = np.array(P0, dtype=np.float64).copy()
    n_iter = np.zeros(N, dtype=np.int32)
    status = np.full(N, STATUS_INVALID_SUBSET, dtype=np.int8)
    zncc = np.full(N, np.nan)
    for n in range(N):
        if not valid[n]:
            continue
        if not np.all(np.isfinite(P0[n])):
            status[n] = STATUS_NAN
            continue
        P, it, st, zc = icgn_12dof_np(P0[n], coords[n], (hx, hy, hz), f, gx, gy, gz, mask, g, mode,
                                      H_all[n], meanf_all[n], bottomf_all[n], tol, max_iter)
        P_out[n] = P
        n_iter[n] = it
        status[n] = st
        zncc[n] = zc
    return P_out, n_iter, status, zncc


def icgn_3dof_batch_np(coords, U_old, F_fixed, vdual, hx, hy, hz, f, gx, gy, gz, mask, g, mode, H_all,
                       meanf_all, bottomf_all, valid, mu, tol, max_iter):
    N = coords.shape[0]
    U_out = np.array(U_old, dtype=np.float64).copy()
    n_iter = np.zeros(N, dtype=np.int32)
    status = np.full(N, STATUS_INVALID_SUBSET, dtype=np.int8)
    zncc = np.full(N, np.nan)
    for n in range(N):
        if not valid[n]:
            continue
        if not (np.all(np.isfinite(U_old[n])) and np.all(np.isfinite(F_fixed[n]))):
            status[n] = STATUS_NAN
            continue
        U, it, st, zc = icgn_3dof_np(U_old[n], F_fixed[n], vdual[n], coords[n], (hx, hy, hz), f, gx, gy, gz,
                                     mask, g, mode, H_all[n], meanf_all[n], bottomf_all[n], mu, tol, max_iter)
        U_out[n] = U
        n_iter[n] = it
        status[n] = st
        zncc[n] = zc
    return U_out, n_iter, status, zncc
