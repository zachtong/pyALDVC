"""Strain measures from the displacement gradient (MATLAB ``ComputeStrain3.m``).

All functions are vectorised over ``(N, 3, 3)`` tensors with NaN-safe
behaviour (NaN in -> NaN out).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_EYE = np.eye(3)


def deformation_gradient(F_grad: NDArray[np.float64]) -> NDArray[np.float64]:
    """``F = I + grad(u)``."""
    return np.asarray(F_grad, dtype=np.float64) + _EYE


def strain_tensor(F_grad: NDArray[np.float64], strain_type: str) -> NDArray[np.float64]:
    """Strain tensor ``(N, 3, 3)`` of the requested type.

    * ``infinitesimal``: ``0.5 (H + H^T)``, ``H = grad(u)``
    * ``green_lagrange``: ``0.5 (F^T F - I)``
    * ``euler_almansi``:  ``0.5 (I - (F F^T)^-1)``
    * ``hencky``:         ``0.5 log(F F^T)`` (via eigendecomposition)
    """
    H = np.asarray(F_grad, dtype=np.float64)
    if strain_type == "infinitesimal":
        return 0.5 * (H + np.swapaxes(H, 1, 2))
    Fm = H + _EYE
    finite = np.all(np.isfinite(Fm), axis=(1, 2))
    out = np.full_like(H, np.nan)
    Ff = Fm[finite]
    if Ff.shape[0] == 0:
        return out
    if strain_type == "green_lagrange":
        C = np.einsum("nki,nkj->nij", Ff, Ff)
        out[finite] = 0.5 * (C - _EYE)
    elif strain_type == "euler_almansi":
        B = np.einsum("nik,njk->nij", Ff, Ff)
        try:
            Binv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            Binv = np.linalg.pinv(B)
        out[finite] = 0.5 * (_EYE - Binv)
    elif strain_type == "hencky":
        B = np.einsum("nik,njk->nij", Ff, Ff)
        vals, vecs = np.linalg.eigh(B)
        vals = np.maximum(vals, 1e-30)
        logv = 0.5 * np.log(vals)
        out[finite] = np.einsum("nik,nk,njk->nij", vecs, logv, vecs)
    else:
        raise ValueError(f"unknown strain_type {strain_type!r}")
    return out


def principal_strains(E: NDArray[np.float64]) -> NDArray[np.float64]:
    """Eigenvalues of the symmetric strain tensors, sorted descending ``(N, 3)``."""
    E = np.asarray(E, dtype=np.float64)
    out = np.full((E.shape[0], 3), np.nan)
    finite = np.all(np.isfinite(E), axis=(1, 2))
    if finite.any():
        vals = np.linalg.eigvalsh(0.5 * (E[finite] + np.swapaxes(E[finite], 1, 2)))
        out[finite] = vals[:, ::-1]
    return out


def von_mises_strain(E: NDArray[np.float64]) -> NDArray[np.float64]:
    """Equivalent (von Mises) strain ``sqrt(2/3 e':e')`` with ``e'`` deviatoric."""
    E = np.asarray(E, dtype=np.float64)
    tr = np.trace(E, axis1=1, axis2=2)
    dev = E - (tr / 3.0)[:, None, None] * _EYE
    return np.sqrt(2.0 / 3.0 * np.einsum("nij,nij->n", dev, dev))


def max_shear_strain(principal: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (principal[:, 0] - principal[:, 2])


def volumetric_strain(E: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.trace(np.asarray(E, dtype=np.float64), axis1=1, axis2=2)


def det_deformation_gradient(F_grad: NDArray[np.float64]) -> NDArray[np.float64]:
    Fm = deformation_gradient(F_grad)
    out = np.full(Fm.shape[0], np.nan)
    finite = np.all(np.isfinite(Fm), axis=(1, 2))
    out[finite] = np.linalg.det(Fm[finite])
    return out


def polar_rotation_deg(F_grad: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation angle (degrees) of ``R`` from the polar decomposition ``F = R U``."""
    Fm = deformation_gradient(F_grad)
    out = np.full(Fm.shape[0], np.nan)
    finite = np.all(np.isfinite(Fm), axis=(1, 2))
    if not finite.any():
        return out
    Uu, _, Vt = np.linalg.svd(Fm[finite])
    R = Uu @ Vt
    neg = np.linalg.det(R) < 0
    if neg.any():
        Uu[neg, :, -1] *= -1
        R = Uu @ Vt
    cos_t = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    out[finite] = np.degrees(np.arccos(cos_t))
    return out


def scale_to_physical(
    U: NDArray[np.float64],
    F_grad: NDArray[np.float64],
    voxel_size: tuple[float, float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert voxel-unit displacement/gradient to physical units.

    ``u_i^phys = s_i u_i``; ``F_ij^phys = (s_i / s_j) F_ij`` (MATLAB Section 8).
    """
    s = np.asarray(voxel_size, dtype=np.float64)
    U_p = np.asarray(U, dtype=np.float64) * s[None, :]
    F_p = np.asarray(F_grad, dtype=np.float64) * (s[:, None] / s[None, :])[None, :, :]
    return U_p, F_p
