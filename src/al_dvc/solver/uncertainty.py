"""Per-node displacement uncertainty from the IC-GN normal equations.

At a converged node the ZNSSD residual ``r_i = f~_i - g~_i(P)`` has
``n_valid`` terms built from unit-norm normalised intensities, so
``sum r_i^2 = 2 (1 - ZNCC)``; with equal intensity noise in both volumes the
per-voxel noise variance in the node's normalisation is
``s^2 = (1 - ZNCC) bottomf^2 / n_valid``.

IC-GN solves ``sum_i J_i r_i = 0`` with the *noisy* reference-gradient
Jacobian ``J_i = J0_i + grad(n_f)_i``. Linearising around the true warp with
the deformed-volume gradient ``K_i ~ J0_i + grad(n_g)_i`` gives the error
``dP = (sum J_i K_i^T)^-1 sum J_i r*_i`` whose expectation of the system
matrix is the noise-free Hessian ``H0``, while the stored Hessian
``H = sum J_i J_i^T`` is inflated: ``E[H] = H0 + c s^2 (I_3 (x) M)``, where
``c = sum w_k^2 = 1.17`` is the variance gain of the 7-point stencil and
``M = sum_i [X_i, Y_i, Z_i, 1][X_i, Y_i, Z_i, 1]^T`` the moment matrix of the
subset coordinates. The right-hand side has two independent parts: the
noise-times-clean-gradient term, covariance ``2 s^2 H0``, and the
reference-gradient-noise-times-deformed-noise term, covariance
``c s^4 (I_3 (x) M)`` (the analogous reference-noise product vanishes for a
central-difference stencil). Hence

    Cov(P) = 2 s^2 H0^-1 + c s^4 H0^-1 (I_3 (x) M) H0^-1,   H0 = H - c s^2 (I_3 (x) M).

The translation block yields the standard deviation of ``u, v, w`` in voxels
(for translation-only subsets and low noise this is the classic
``var(u) = 2 s^2 / sum f_x^2`` estimate of Wang et al. 2007).

Calibration (``scripts/make_uncertainty_report.py``): 20-35 % below the
empirical error for SNR >= 3 (ZNCC >= 0.9), about 2x below it at SNR ~ 1.5
where the linearised theory no longer holds. The estimate describes the
random error of the *local* solution at a converged node. It does not include
interpolation bias, the subset shape-function bias, or the error reduction of
the global (ADMM) step, which lowers the actual error below this value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import STATUS_CONVERGED
from ..io.volume_ops import STENCIL7

N_DOF = 12
MIN_EXTRA_DOF = 2  # a node needs n_valid > N_DOF + MIN_EXTRA_DOF voxels for a variance estimate
STENCIL_NOISE_GAIN = float(np.sum(STENCIL7**2))  # variance gain of the gradient stencil on white noise
MIN_CORRECTED_FRACTION = 0.1  # keep at least this fraction of the translation diagonal after correction


def subset_moment_matrix(half: tuple[int, int, int], stride: int = 1) -> NDArray[np.float64]:
    """``M = sum [X, Y, Z, 1][X, Y, Z, 1]^T`` over the sampled subset (every ``stride``-th voxel per axis)."""
    hx, hy, hz = (int(h) for h in half)
    ax = np.arange(-hx, hx + 1, int(stride))
    ay = np.arange(-hy, hy + 1, int(stride))
    az = np.arange(-hz, hz + 1, int(stride))
    nx, ny, nz = ax.size, ay.size, az.size
    n = nx * ny * nz
    sx2 = float(np.sum(ax**2)) * ny * nz
    sy2 = float(np.sum(ay**2)) * nx * nz
    sz2 = float(np.sum(az**2)) * nx * ny
    return np.diag([sx2, sy2, sz2, float(n)])


def noise_hessian_pattern(half: tuple[int, int, int], stride: int = 1) -> NDArray[np.float64]:
    """``(12, 12)`` expectation of ``sum J_i J_i^T`` for unit-variance white gradient noise.

    Parameter ``p`` of ``P = [F.ravel(), u, v, w]`` is gradient component
    ``a_p`` times coordinate ``b_p`` (``b = 3`` for the translations):
    ``E[J J^T]_{pq} = delta(a_p, a_q) * M[b_p, b_q]``.
    """
    M = subset_moment_matrix(half, stride)
    comp = np.array([p // 3 for p in range(9)] + [0, 1, 2])
    coord = np.array([p % 3 for p in range(9)] + [3, 3, 3])
    pattern = np.zeros((N_DOF, N_DOF))
    for p in range(N_DOF):
        for q in range(N_DOF):
            if comp[p] == comp[q]:
                pattern[p, q] = M[coord[p], coord[q]]
    return pattern


def parameter_covariance_factor(ctx, zncc: NDArray, status: NDArray | None = None) -> tuple[NDArray, NDArray]:
    """``(idx, sigma_r^2 * bottomf^2)`` for the nodes where a covariance can be formed."""
    zncc = np.asarray(zncc, dtype=np.float64)
    ok = ctx.valid & np.isfinite(zncc) & (ctx.n_valid > N_DOF + MIN_EXTRA_DOF)
    if status is not None:
        ok &= np.asarray(status) == STATUS_CONVERGED
    idx = np.flatnonzero(ok)
    sigma_r2 = 2.0 * np.clip(1.0 - zncc[idx], 0.0, None) / (ctx.n_valid[idx] - N_DOF)
    return idx, sigma_r2 * ctx.bottomf[idx] ** 2


def displacement_uncertainty(
    ctx, zncc: NDArray, status: NDArray | None = None, noise_correction: bool = True
) -> NDArray[np.float64]:
    """Standard deviation ``(N, 3)`` of ``u, v, w`` in voxels; NaN where undefined.

    Args:
        ctx: ``LocalContext`` of the reference (Hessians, ``bottomf``, ``n_valid``).
        zncc: final ZNCC per node.
        status: node status codes; when given, only converged nodes get a value.
        noise_correction: use the two-term model of the module docstring
            (``False``: plain ``2 s^2 H^-1`` with the stored Hessian).
    """
    n = ctx.n_nodes
    out = np.full((n, 3), np.nan)
    idx, factor = parameter_covariance_factor(ctx, zncc, status)
    if idx.size == 0:
        return out
    H = np.array(ctx.H_all[idx], dtype=np.float64)
    zncc = np.asarray(zncc, dtype=np.float64)
    s2 = np.clip(1.0 - zncc[idx], 0.0, None) * ctx.bottomf[idx] ** 2 / ctx.n_valid[idx]
    pattern = noise_hessian_pattern(ctx.half, int(getattr(ctx, "stride", 1)))
    # scale the pattern to the node's actual voxel count (masked subsets have fewer voxels)
    scale = ctx.n_valid[idx] / pattern[9, 9]
    if noise_correction:
        correction = (STENCIL_NOISE_GAIN * s2 * scale)[:, None, None] * pattern[None, :, :]
        # never remove more than (1 - MIN_CORRECTED_FRACTION) of the translation diagonal
        tdiag = np.einsum("nii->ni", H[:, 9:, 9:])
        cdiag = np.einsum("nii->ni", correction[:, 9:, 9:])
        limit = np.min(np.where(cdiag > 0, (1.0 - MIN_CORRECTED_FRACTION) * tdiag / np.maximum(cdiag, 1e-300), 1.0), axis=1)
        H = H - correction * np.clip(limit, 0.0, 1.0)[:, None, None]
    try:
        inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(H)
    var = factor[:, None] * np.diagonal(inv, axis1=1, axis2=2)[:, 9:12]
    if noise_correction:
        cross = inv @ (pattern[None, :, :] * scale[:, None, None]) @ inv
        var = var + (STENCIL_NOISE_GAIN * s2**2)[:, None] * np.diagonal(cross, axis1=1, axis2=2)[:, 9:12]
    out[idx] = np.sqrt(np.clip(var, 0.0, None))
    return out
