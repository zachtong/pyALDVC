"""Numba-compiled IC-GN kernels (12-DOF local, 3-DOF ADMM subproblem 1).

Port of MATLAB ``funICGN3.m`` (12-DOF) and ``funICGN_Subpb13.m`` (3-DOF),
restructured so that nothing per-node is pre-extracted except the 12x12
Hessian (stored as its Cholesky factor), the reference-subset mean and the
ZNSSD normalisation. Subset voxels are read in place from the reference
volume, its three gradient volumes and the mask.

Warp parameter layout (see docs/design.md):
    P[0:9]  = F.ravel()  row-major, F[i, j] = du_i/dx_j
    P[9:12] = (u, v, w)
    x_def = x0 + P[9:12] + (I + F) @ (x - x0)

Steepest-descent images for the ZNSSD inverse-compositional update:
    SD[0:3] = gx * (X, Y, Z)   SD[3:6] = gy * (X, Y, Z)
    SD[6:9] = gz * (X, Y, Z)   SD[9:12] = (gx, gy, gz)

Status codes are the constants in ``al_dvc.core.data_structures``.
"""

from __future__ import annotations

import numpy as np

from .._numba_compat import JIT_CACHE, njit, prange
from .interp_kernels import interp_margin_ok, sample_volume

STATUS_CONVERGED = 0
STATUS_MAX_ITER = 1
STATUS_OUT_OF_BOUNDS = 2
STATUS_INVALID_SUBSET = 3
STATUS_SINGULAR = 4
STATUS_NAN = 5
STATUS_SKIPPED = 6
STATUS_STALLED = 7

ABS_TOL = 1e-5  # MATLAB: normOfWNew*normOfWNewInit < 1e-5
STALL_ZNCC_EPS = 1e-4  # a 12-DOF iteration must raise the ZNCC by this much to count as progress
STALL_STEP_DECAY = 0.999  # a 3-DOF iteration must shrink the step norm by this factor to count as progress
SCHEDULE_LANES = 64  # stripes of the block-cyclic node schedule in the parallel IC-GN wrappers
MIN_SUBSET_VOXELS = 27  # fewer valid voxels than this and a subset is not correlated
MIN_VALID_FRACTION = 0.5  # a subset must keep this fraction of its reference-valid voxels under the deformed mask
LM_DAMPING_3DOF = 1e-3


# ---------------------------------------------------------------------------
# Small dense linear algebra helpers (no allocations in the hot loops)
# ---------------------------------------------------------------------------


@njit(cache=JIT_CACHE)
def _cholesky12(H, L):
    """Cholesky ``H = L L^T`` for a 12x12 SPD matrix. Returns False if not SPD."""
    n = 12
    for i in range(n):
        for j in range(n):
            L[i, j] = 0.0
    for j in range(n):
        s = H[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        if s <= 0.0 or not np.isfinite(s):
            return False
        ljj = np.sqrt(s)
        L[j, j] = ljj
        for i in range(j + 1, n):
            t = H[i, j]
            for k in range(j):
                t -= L[i, k] * L[j, k]
            L[i, j] = t / ljj
    return True


@njit(cache=JIT_CACHE, inline="always")
def _chol_solve12(L, b, x):
    """Solve ``L L^T x = b`` in place (b is not modified)."""
    y = np.empty(12)
    for i in range(12):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s / L[i, i]
    for i in range(11, -1, -1):
        s = y[i]
        for k in range(i + 1, 12):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]


@njit(cache=JIT_CACHE, inline="always")
def _inv3(a, out):
    """Inverse of a 3x3 matrix via cofactors. Returns the determinant."""
    det = (
        a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
        - a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
        + a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0])
    )
    if abs(det) < 1e-300:
        return 0.0
    inv = 1.0 / det
    out[0, 0] = (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) * inv
    out[0, 1] = (a[0, 2] * a[2, 1] - a[0, 1] * a[2, 2]) * inv
    out[0, 2] = (a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1]) * inv
    out[1, 0] = (a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2]) * inv
    out[1, 1] = (a[0, 0] * a[2, 2] - a[0, 2] * a[2, 0]) * inv
    out[1, 2] = (a[0, 2] * a[1, 0] - a[0, 0] * a[1, 2]) * inv
    out[2, 0] = (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]) * inv
    out[2, 1] = (a[0, 1] * a[2, 0] - a[0, 0] * a[2, 1]) * inv
    out[2, 2] = (a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]) * inv
    return det


@njit(cache=JIT_CACHE)
def compose_warp_inplace(P, dP):
    """``W(P) <- W(P) W(dP)^-1`` on 12-vectors. Returns False if singular.

    With ``A = I + F``, ``t = u`` and ``dA = I + dF``, ``dt = du``:
        A_new = A dA^-1,   t_new = t - A_new dt
    """
    A = np.empty((3, 3))
    dA = np.empty((3, 3))
    dAinv = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            A[i, j] = P[3 * i + j]
            dA[i, j] = dP[3 * i + j]
        A[i, i] += 1.0
        dA[i, i] += 1.0
    det = _inv3(dA, dAinv)
    if det == 0.0:
        return False
    Anew = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += A[i, k] * dAinv[k, j]
            Anew[i, j] = s
    for i in range(3):
        s = 0.0
        for k in range(3):
            s += Anew[i, k] * dP[9 + k]
        P[9 + i] = P[9 + i] - s
    for i in range(3):
        for j in range(3):
            P[3 * i + j] = Anew[i, j]
        P[3 * i + i] -= 1.0
    return True


# ---------------------------------------------------------------------------
# Per-node precomputation: Hessian (as Cholesky factor), ZNSSD statistics
# ---------------------------------------------------------------------------


@njit(cache=JIT_CACHE, inline="always")
def _grad_at(f, gx, gy, gz, zz, yy, xx):
    """Reference gradient at a voxel: the stored arrays, or the 7-point stencil on ``f``
    when the gradient volumes are the 1x1x1 placeholders of ``gradient_mode="on_the_fly"``."""
    if gx.shape[0] == f.shape[0]:
        return float(gx[zz, yy, xx]), float(gy[zz, yy, xx]), float(gz[zz, yy, xx])
    c1 = 0.75
    c2 = -0.15
    c3 = 1.0 / 60.0
    gxv = (
        c1 * (float(f[zz, yy, xx + 1]) - float(f[zz, yy, xx - 1]))
        + c2 * (float(f[zz, yy, xx + 2]) - float(f[zz, yy, xx - 2]))
        + c3 * (float(f[zz, yy, xx + 3]) - float(f[zz, yy, xx - 3]))
    )
    gyv = (
        c1 * (float(f[zz, yy + 1, xx]) - float(f[zz, yy - 1, xx]))
        + c2 * (float(f[zz, yy + 2, xx]) - float(f[zz, yy - 2, xx]))
        + c3 * (float(f[zz, yy + 3, xx]) - float(f[zz, yy - 3, xx]))
    )
    gzv = (
        c1 * (float(f[zz + 1, yy, xx]) - float(f[zz - 1, yy, xx]))
        + c2 * (float(f[zz + 2, yy, xx]) - float(f[zz - 2, yy, xx]))
        + c3 * (float(f[zz + 3, yy, xx]) - float(f[zz - 3, yy, xx]))
    )
    return gxv, gyv, gzv


@njit(cache=JIT_CACHE)
def _precompute_one(x0, y0, z0, hx, hy, hz, f, gx, gy, gz, mask, min_valid_ratio, cond_max, H, L):
    """Fill ``H`` (12x12) and ``L`` for one node.

    Returns ``(meanf, bottomf, n_valid, ok)``.
    """
    nz = f.shape[0]
    ny = f.shape[1]
    nx = f.shape[2]
    for i in range(12):
        for j in range(12):
            H[i, j] = 0.0

    if x0 - hx < 0 or x0 + hx >= nx or y0 - hy < 0 or y0 + hy >= ny or z0 - hz < 0 or z0 + hz >= nz:
        return 0.0, 1.0, 0, False

    sd = np.empty(12)
    n_valid = 0
    sum_f = 0.0
    sum_f2 = 0.0
    for dz in range(-hz, hz + 1):
        zz = z0 + dz
        Z = float(dz)
        for dy in range(-hy, hy + 1):
            yy = y0 + dy
            Y = float(dy)
            for dx in range(-hx, hx + 1):
                xx = x0 + dx
                if mask[zz, yy, xx] == 0:
                    continue
                X = float(dx)
                fv = float(f[zz, yy, xx])
                gxv, gyv, gzv = _grad_at(f, gx, gy, gz, zz, yy, xx)
                n_valid += 1
                sum_f += fv
                sum_f2 += fv * fv
                sd[0] = gxv * X
                sd[1] = gxv * Y
                sd[2] = gxv * Z
                sd[3] = gyv * X
                sd[4] = gyv * Y
                sd[5] = gyv * Z
                sd[6] = gzv * X
                sd[7] = gzv * Y
                sd[8] = gzv * Z
                sd[9] = gxv
                sd[10] = gyv
                sd[11] = gzv
                for a in range(12):
                    sa = sd[a]
                    for b in range(a, 12):
                        H[a, b] += sa * sd[b]
    for a in range(12):
        for b in range(a):
            H[a, b] = H[b, a]

    total = (2 * hx + 1) * (2 * hy + 1) * (2 * hz + 1)
    if n_valid < 27 or n_valid < min_valid_ratio * total:
        return 0.0, 1.0, n_valid, False

    meanf = sum_f / n_valid
    # MATLAB: bottomf = sqrt((n-1) * var(f)) with the sample variance
    #       = sqrt(sum (f - meanf)^2)
    ssf = sum_f2 - n_valid * meanf * meanf
    if ssf < 0.0:
        ssf = 0.0
    bottomf = np.sqrt(max(ssf, 1e-30))
    if bottomf < 1e-10:
        return meanf, bottomf, n_valid, False

    ok = _cholesky12(H, L)
    if not ok:
        return meanf, bottomf, n_valid, False
    # conditioning proxy from the Cholesky diagonal: (max L_ii / min L_ii)^2
    dmin = L[0, 0]
    dmax = L[0, 0]
    for i in range(1, 12):
        if L[i, i] < dmin:
            dmin = L[i, i]
        if L[i, i] > dmax:
            dmax = L[i, i]
    if dmin <= 0.0 or (dmax / dmin) * (dmax / dmin) > cond_max:
        return meanf, bottomf, n_valid, False
    return meanf, bottomf, n_valid, True


@njit(parallel=True, cache=JIT_CACHE)
def precompute_nodes(coords, hx, hy, hz, f, gx, gy, gz, mask, min_valid_ratio, cond_max):
    """Parallel per-node precomputation.

    Args:
        coords: ``(N, 3)`` int64 node centres ``[x, y, z]``.
        hx, hy, hz: subset half-widths (``winsize // 2``).
        f, gx, gy, gz: ``(nz, ny, nx)`` float32 reference volume and gradients.
        mask: ``(nz, ny, nx)`` uint8.
    Returns:
        ``(H_all (N,12,12), L_all (N,12,12), meanf (N,), bottomf (N,), n_valid (N,), valid (N,))``
    """
    N = coords.shape[0]
    H_all = np.zeros((N, 12, 12))
    L_all = np.zeros((N, 12, 12))
    meanf_all = np.zeros(N)
    bottomf_all = np.ones(N)
    nvalid_all = np.zeros(N, dtype=np.int64)
    valid = np.zeros(N, dtype=np.bool_)
    for n in prange(N):
        meanf, bottomf, nv, ok = _precompute_one(
            coords[n, 0],
            coords[n, 1],
            coords[n, 2],
            hx,
            hy,
            hz,
            f,
            gx,
            gy,
            gz,
            mask,
            min_valid_ratio,
            cond_max,
            H_all[n],
            L_all[n],
        )
        meanf_all[n] = meanf
        bottomf_all[n] = bottomf
        nvalid_all[n] = nv
        valid[n] = ok
    return H_all, L_all, meanf_all, bottomf_all, nvalid_all, valid


# ---------------------------------------------------------------------------
# Shared per-iteration work: warp + sample + ZNSSD statistics
# ---------------------------------------------------------------------------


@njit(cache=JIT_CACHE)
def _warp_and_sample(P, x0, y0, z0, hx, hy, hz, mask, f, g, mode, gbuf):
    """Sample the deformed volume over the warped subset into ``gbuf``.

    Voxels masked out in the reference (``mask == 0``) or whose sample hits a
    NaN voxel of ``g`` (deformed-frame mask) are stored as NaN in ``gbuf`` and
    excluded from the statistics; the reference statistics are recomputed on
    the same dynamic set. Returns
    ``(ok, n_valid, meanf, bottomf, meang, bottomg)``; ``ok`` is False when a
    warped voxel leaves the admissible sampling domain (``n_valid == 0``) or
    fewer than ``MIN_SUBSET_VOXELS`` voxels remain.
    """
    nz = g.shape[0]
    ny = g.shape[1]
    nx = g.shape[2]
    a00 = 1.0 + P[0]
    a01 = P[1]
    a02 = P[2]
    a10 = P[3]
    a11 = 1.0 + P[4]
    a12 = P[5]
    a20 = P[6]
    a21 = P[7]
    a22 = 1.0 + P[8]
    tx = x0 + P[9]
    ty = y0 + P[10]
    tz = z0 + P[11]

    n_valid = 0
    n_ref = 0
    s1 = 0.0
    s2 = 0.0
    s1f = 0.0
    s2f = 0.0
    idx = 0
    for dz in range(-hz, hz + 1):
        Z = float(dz)
        zz = z0 + dz
        for dy in range(-hy, hy + 1):
            Y = float(dy)
            yy = y0 + dy
            for dx in range(-hx, hx + 1):
                if mask[zz, yy, x0 + dx] == 0:
                    gbuf[idx] = np.nan
                    idx += 1
                    continue
                n_ref += 1
                X = float(dx)
                xw = a00 * X + a01 * Y + a02 * Z + tx
                yw = a10 * X + a11 * Y + a12 * Z + ty
                zw = a20 * X + a21 * Y + a22 * Z + tz
                if not interp_margin_ok(zw, yw, xw, nz, ny, nx):
                    return False, 0, 0.0, 1.0, 0.0, 1.0
                val = sample_volume(g, zw, yw, xw, mode)
                gbuf[idx] = val
                idx += 1
                if val != val:  # NaN: masked out in the deformed volume
                    continue
                fv = float(f[zz, yy, x0 + dx])
                n_valid += 1
                s1 += val
                s2 += val * val
                s1f += fv
                s2f += fv * fv
    if n_valid < MIN_SUBSET_VOXELS or n_valid < MIN_VALID_FRACTION * n_ref:
        return False, n_valid, 0.0, 1.0, 0.0, 1.0
    meang = s1 / n_valid
    ssg = s2 - n_valid * meang * meang
    if ssg < 0.0:
        ssg = 0.0
    bottomg = np.sqrt(max(ssg, 1e-30))
    meanf = s1f / n_valid
    ssf = s2f - n_valid * meanf * meanf
    if ssf < 0.0:
        ssf = 0.0
    bottomf = np.sqrt(max(ssf, 1e-30))
    return True, n_valid, meanf, bottomf, meang, bottomg


@njit(cache=JIT_CACHE)
def _zncc_from_buffer(x0, y0, z0, hx, hy, hz, f, mask, gbuf, meanf, bottomf, meang, bottomg):
    """Zero-normalised cross-correlation between the reference subset and ``gbuf``."""
    s = 0.0
    idx = 0
    for dz in range(-hz, hz + 1):
        zz = z0 + dz
        for dy in range(-hy, hy + 1):
            yy = y0 + dy
            for dx in range(-hx, hx + 1):
                xx = x0 + dx
                gv = gbuf[idx]
                if gv == gv:
                    s += (float(f[zz, yy, xx]) - meanf) * (gv - meang)
                idx += 1
    return s / (bottomf * bottomg)


# ---------------------------------------------------------------------------
# 12-DOF IC-GN (local subset DVC, MATLAB funICGN3)
# ---------------------------------------------------------------------------


@njit(cache=JIT_CACHE)
def _icgn_12dof_single(
    P, x0, y0, z0, hx, hy, hz, f, gx, gy, gz, mask, g, mode, L, meanf, bottomf, tol, dp_tol, max_iter, patience, gbuf
):
    """Iterate one node in place. Returns ``(n_iter, status, zncc)``."""
    b = np.empty(12)
    dP = np.empty(12)
    negb = np.empty(12)
    norm_init = -1.0
    half_scale = max(hx, max(hy, hz))
    zncc = np.nan
    best_zncc = -2.0
    best_dp = 1e300
    stall = 0
    P_best = np.empty(12)

    for it in range(1, max_iter + 1):
        ok, n_valid, meanf_d, bottomf_d, meang, bottomg = _warp_and_sample(P, x0, y0, z0, hx, hy, hz, mask, f, g, mode, gbuf)
        if not ok:
            return it, (STATUS_OUT_OF_BOUNDS if n_valid == 0 else STATUS_INVALID_SUBSET), np.nan

        for k in range(12):
            b[k] = 0.0
        idx = 0
        for dz in range(-hz, hz + 1):
            Z = float(dz)
            zz = z0 + dz
            for dy in range(-hy, hy + 1):
                Y = float(dy)
                yy = y0 + dy
                for dx in range(-hx, hx + 1):
                    xx = x0 + dx
                    gv = gbuf[idx]
                    if gv != gv:
                        idx += 1
                        continue
                    X = float(dx)
                    res = (float(f[zz, yy, xx]) - meanf_d) / bottomf_d - (gv - meang) / bottomg
                    idx += 1
                    dfx, dfy, dfz = _grad_at(f, gx, gy, gz, zz, yy, xx)
                    gxv = dfx * res
                    gyv = dfy * res
                    gzv = dfz * res
                    b[0] += gxv * X
                    b[1] += gxv * Y
                    b[2] += gxv * Z
                    b[3] += gyv * X
                    b[4] += gyv * Y
                    b[5] += gyv * Z
                    b[6] += gzv * X
                    b[7] += gzv * Y
                    b[8] += gzv * Z
                    b[9] += gxv
                    b[10] += gyv
                    b[11] += gzv
        norm_abs = 0.0
        for k in range(12):
            b[k] *= bottomf_d
            norm_abs += b[k] * b[k]
        norm_abs = np.sqrt(norm_abs)
        if not np.isfinite(norm_abs):
            return it, STATUS_NAN, np.nan
        if norm_init < 0.0:
            norm_init = norm_abs
        norm_rel = norm_abs / norm_init if norm_init > 1e-300 else 0.0

        zncc = _zncc_from_buffer(x0, y0, z0, hx, hy, hz, f, mask, gbuf, meanf_d, bottomf_d, meang, bottomg)
        if norm_rel < tol or norm_abs < ABS_TOL:
            return it, STATUS_CONVERGED, zncc

        for k in range(12):
            negb[k] = -b[k]
        _chol_solve12(L, negb, dP)

        # parameter-update norm with gradient terms scaled to voxels at the subset edge
        dp_norm = 0.0
        for k in range(9):
            v = dP[k] * half_scale
            dp_norm += v * v
        for k in range(9, 12):
            dp_norm += dP[k] * dP[k]
        dp_norm = np.sqrt(dp_norm)
        if not np.isfinite(dp_norm):
            return it, STATUS_NAN, np.nan
        if dp_norm < dp_tol:
            return it, STATUS_CONVERGED, zncc
        improved = zncc > best_zncc + STALL_ZNCC_EPS
        if improved:
            best_zncc = zncc
            for k in range(12):
                P_best[k] = P[k]
        if dp_norm < best_dp * STALL_STEP_DECAY:
            best_dp = dp_norm
            improved = True
        if improved:
            stall = 0
        else:
            stall += 1
            if patience > 0 and stall >= patience:
                for k in range(12):
                    P[k] = P_best[k]
                return it, STATUS_STALLED, best_zncc

        if not compose_warp_inplace(P, dP):
            return it, STATUS_SINGULAR, zncc

    return max_iter, STATUS_MAX_ITER, zncc


@njit(parallel=True, cache=JIT_CACHE)
def icgn_12dof_parallel(
    coords, P0, hx, hy, hz, f, gx, gy, gz, mask, g, mode, L_all, meanf_all, bottomf_all, valid, tol, dp_tol, max_iter, patience
):
    """Parallel 12-DOF IC-GN over all nodes.

    Returns ``(P_out (N,12), n_iter (N,), status (N,), zncc (N,))``.
    """
    N = coords.shape[0]
    P_out = P0.copy()
    n_iter = np.zeros(N, dtype=np.int32)
    status = np.full(N, STATUS_SKIPPED, dtype=np.int8)
    zncc = np.full(N, np.nan)
    S = (2 * hx + 1) * (2 * hy + 1) * (2 * hz + 1)
    for n in range(N):
        if not valid[n]:
            status[n] = STATUS_INVALID_SUBSET
    idx = np.flatnonzero(valid)
    M = idx.size
    L = (M + SCHEDULE_LANES - 1) // SCHEDULE_LANES
    # block-cyclic schedule over SCHEDULE_LANES stripes of the active-node list: every static
    # prange chunk then touches all regions of the grid, so spatial clusters of skipped or
    # hard nodes cannot idle most threads, while runs of consecutive nodes keep cache locality
    for kk in prange(SCHEDULE_LANES * L):
        p = (kk % SCHEDULE_LANES) * L + kk // SCHEDULE_LANES
        if p >= M:
            continue
        n = idx[p]
        gbuf = np.empty(S)
        P = P_out[n]
        finite = True
        for k in range(12):
            if not np.isfinite(P[k]):
                finite = False
        if not finite:
            status[n] = STATUS_NAN
            continue
        it, st, zc = _icgn_12dof_single(
            P,
            coords[n, 0],
            coords[n, 1],
            coords[n, 2],
            hx,
            hy,
            hz,
            f,
            gx,
            gy,
            gz,
            mask,
            g,
            mode,
            L_all[n],
            meanf_all[n],
            bottomf_all[n],
            tol,
            dp_tol,
            max_iter,
            patience,
            gbuf,
        )
        n_iter[n] = it
        status[n] = st
        zncc[n] = zc
    return P_out, n_iter, status, zncc


# ---------------------------------------------------------------------------
# 3-DOF IC-GN with fixed F and proximal penalty (ADMM subproblem 1,
# MATLAB funICGN_Subpb13)
# ---------------------------------------------------------------------------


@njit(cache=JIT_CACHE)
def _icgn_3dof_single(
    P,
    x0,
    y0,
    z0,
    hx,
    hy,
    hz,
    f,
    gx,
    gy,
    gz,
    mask,
    g,
    mode,
    H,
    meanf,
    bottomf,
    u_old,
    vdual,
    mu,
    tol,
    dp_tol,
    max_iter,
    patience,
    gbuf,
):
    """Translation-only IC-GN at one node. Returns ``(n_iter, status, zncc)``."""
    best_dn = 1e300
    stall = 0
    bf2 = bottomf * bottomf
    if bf2 < 1e-30:
        bf2 = 1e-30
    # H3 = 2 * H_tt / bottomf^2 + mu I   (H_tt = translation block of the 12x12 Hessian)
    H3 = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            H3[i, j] = H[9 + i, 9 + j] * 2.0 / bf2
        H3[i, i] += mu
    max_diag = max(H3[0, 0], max(H3[1, 1], H3[2, 2]))
    Hd = np.empty((3, 3))
    Hinv = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            Hd[i, j] = H3[i, j]
        Hd[i, i] += LM_DAMPING_3DOF * max_diag
    det = _inv3(Hd, Hinv)
    if det == 0.0:
        return 0, STATUS_SINGULAR, np.nan

    b = np.empty(3)
    tb = np.empty(3)
    dt = np.empty(3)
    norm_init = -1.0
    zncc = np.nan
    a00 = 1.0 + P[0]
    a01 = P[1]
    a02 = P[2]
    a10 = P[3]
    a11 = 1.0 + P[4]
    a12 = P[5]
    a20 = P[6]
    a21 = P[7]
    a22 = 1.0 + P[8]

    for it in range(1, max_iter + 1):
        ok, n_valid, meanf_d, bottomf_d, meang, bottomg = _warp_and_sample(P, x0, y0, z0, hx, hy, hz, mask, f, g, mode, gbuf)
        if not ok:
            return it, (STATUS_OUT_OF_BOUNDS if n_valid == 0 else STATUS_INVALID_SUBSET), np.nan
        b[0] = 0.0
        b[1] = 0.0
        b[2] = 0.0
        idx = 0
        for dz in range(-hz, hz + 1):
            zz = z0 + dz
            for dy in range(-hy, hy + 1):
                yy = y0 + dy
                for dx in range(-hx, hx + 1):
                    xx = x0 + dx
                    gv = gbuf[idx]
                    if gv != gv:
                        idx += 1
                        continue
                    res = (float(f[zz, yy, xx]) - meanf_d) / bottomf_d - (gv - meang) / bottomg
                    idx += 1
                    dfx, dfy, dfz = _grad_at(f, gx, gy, gz, zz, yy, xx)
                    b[0] += dfx * res
                    b[1] += dfy * res
                    b[2] += dfz * res
        norm_abs = 0.0
        for k in range(3):
            b[k] *= bottomf_d
            tb[k] = b[k] * 2.0 / bf2 + mu * (P[9 + k] - u_old[k] - vdual[k])
            norm_abs += tb[k] * tb[k]
        norm_abs = np.sqrt(norm_abs)
        if not np.isfinite(norm_abs):
            return it, STATUS_NAN, np.nan
        if norm_init < 0.0:
            norm_init = norm_abs
        norm_rel = norm_abs / norm_init if norm_init > 1e-300 else 0.0
        zncc = _zncc_from_buffer(x0, y0, z0, hx, hy, hz, f, mask, gbuf, meanf_d, bottomf_d, meang, bottomg)
        # MATLAB: break when normOfWNew < tol or normOfWNew*normOfWNewInit < mu*1e-4
        if norm_rel < tol or norm_abs < mu * 1e-4:
            return it, STATUS_CONVERGED, zncc

        for i in range(3):
            s = 0.0
            for j in range(3):
                s += Hinv[i, j] * tb[j]
            dt[i] = -s
        dn = np.sqrt(dt[0] * dt[0] + dt[1] * dt[1] + dt[2] * dt[2])
        if not np.isfinite(dn):
            return it, STATUS_NAN, np.nan
        if dn < dp_tol:
            return it, STATUS_CONVERGED, zncc
        if dn < best_dn * STALL_STEP_DECAY:
            best_dn = dn
            stall = 0
        else:
            stall += 1
            if patience > 0 and stall >= patience:
                return it, STATUS_STALLED, zncc
        # translation-only inverse compositional update: t <- t - A dt
        P[9] -= a00 * dt[0] + a01 * dt[1] + a02 * dt[2]
        P[10] -= a10 * dt[0] + a11 * dt[1] + a12 * dt[2]
        P[11] -= a20 * dt[0] + a21 * dt[1] + a22 * dt[2]

    return max_iter, STATUS_MAX_ITER, zncc


@njit(parallel=True, cache=JIT_CACHE)
def icgn_3dof_parallel(
    coords,
    U_old,
    F_fixed,
    vdual,
    hx,
    hy,
    hz,
    f,
    gx,
    gy,
    gz,
    mask,
    g,
    mode,
    H_all,
    meanf_all,
    bottomf_all,
    valid,
    mu,
    tol,
    dp_tol,
    max_iter,
    patience,
):
    """Parallel 3-DOF IC-GN (ADMM subproblem 1) over all nodes.

    Args:
        U_old: ``(N, 3)`` global-step displacement ``u_hat`` (start point and prox centre).
        F_fixed: ``(N, 3, 3)`` global-step gradient (held fixed).
        vdual: ``(N, 3)`` displacement dual variable.
    Returns:
        ``(U_out (N,3), n_iter, status, zncc)``.
    """
    N = coords.shape[0]
    U_out = U_old.copy()
    n_iter = np.zeros(N, dtype=np.int32)
    status = np.full(N, STATUS_SKIPPED, dtype=np.int8)
    zncc = np.full(N, np.nan)
    S = (2 * hx + 1) * (2 * hy + 1) * (2 * hz + 1)
    for n in range(N):
        if not valid[n]:
            status[n] = STATUS_INVALID_SUBSET
    idx = np.flatnonzero(valid)
    M = idx.size
    L = (M + SCHEDULE_LANES - 1) // SCHEDULE_LANES
    # block-cyclic schedule over SCHEDULE_LANES stripes of the active-node list: every static
    # prange chunk then touches all regions of the grid, so spatial clusters of skipped or
    # hard nodes cannot idle most threads, while runs of consecutive nodes keep cache locality
    for kk in prange(SCHEDULE_LANES * L):
        p = (kk % SCHEDULE_LANES) * L + kk // SCHEDULE_LANES
        if p >= M:
            continue
        n = idx[p]
        gbuf = np.empty(S)
        P = np.empty(12)
        finite = True
        for i in range(3):
            for j in range(3):
                P[3 * i + j] = F_fixed[n, i, j]
                if not np.isfinite(F_fixed[n, i, j]):
                    finite = False
        for k in range(3):
            P[9 + k] = U_old[n, k]
            if not np.isfinite(U_old[n, k]):
                finite = False
        if not finite:
            status[n] = STATUS_NAN
            continue
        it, st, zc = _icgn_3dof_single(
            P,
            coords[n, 0],
            coords[n, 1],
            coords[n, 2],
            hx,
            hy,
            hz,
            f,
            gx,
            gy,
            gz,
            mask,
            g,
            mode,
            H_all[n],
            meanf_all[n],
            bottomf_all[n],
            U_old[n],
            vdual[n],
            mu,
            tol,
            dp_tol,
            max_iter,
            patience,
            gbuf,
        )
        for k in range(3):
            U_out[n, k] = P[9 + k]
        n_iter[n] = it
        status[n] = st
        zncc[n] = zc
    return U_out, n_iter, status, zncc


# ---------------------------------------------------------------------------
# ZNCC evaluation of a given warp field (diagnostics)
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=JIT_CACHE)
def evaluate_zncc_parallel(coords, P_all, hx, hy, hz, f, mask, g, mode, meanf_all, bottomf_all, valid):
    """ZNCC of each node's subset under the warp ``P_all[n]`` (NaN if invalid)."""
    N = coords.shape[0]
    out = np.full(N, np.nan)
    S = (2 * hx + 1) * (2 * hy + 1) * (2 * hz + 1)
    for n in prange(N):
        if not valid[n]:
            continue
        gbuf = np.empty(S)
        ok, nv, mf, bf, meang, bottomg = _warp_and_sample(
            P_all[n],
            coords[n, 0],
            coords[n, 1],
            coords[n, 2],
            hx,
            hy,
            hz,
            mask,
            f,
            g,
            mode,
            gbuf,
        )
        if not ok:
            continue
        out[n] = _zncc_from_buffer(coords[n, 0], coords[n, 1], coords[n, 2], hx, hy, hz, f, mask, gbuf, mf, bf, meang, bottomg)
    return out
