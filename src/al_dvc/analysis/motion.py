"""Rigid-body motion of a displacement field: fit it, remove it, and correct the gradient consistently.

All positions and displacements are in physical units: with anisotropic voxels a rigid rotation in
micrometres is not a rotation of voxel indices.

The rotation is fitted exactly (weighted Kabsch/Umeyama with the reflection guard), not linearised: a
linearised 2-degree rotation leaves an error of about 6e-4 in the infinitesimal strain, as large as a DVC
noise floor.

Removal is a view, never an edit of the stored result:

* ``translation`` subtracts the mean displacement (the only correction of the DVC Challenge 2.0 noise floor);
* ``rigid`` maps every deformed position back through the inverse of the fitted motion,
  ``u' = R^T (X + u - Xc - t) + Xc - X``, so the gradient becomes ``R^T (I + H) - I`` -- the objective strains
  (Green-Lagrange, principal stretches) do not change, the infinitesimal strain loses the rotation's
  ``cos(theta) - 1``;
* ``affine`` subtracts the fitted homogeneous field ``t + (A - I)(X - Xc)``, leaving the non-affine part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MOTION_KINDS = ("none", "translation", "rigid", "affine")
_MIN_NODES = {"none": 0, "translation": 1, "rigid": 3, "affine": 4}
_EYE = np.eye(3)


@dataclass(frozen=True, eq=False)
class MotionFit:
    """A fitted motion ``x = Xc + t + A (X - Xc)`` of the reference positions ``X``.

    ``matrix`` is ``A``: the identity for a translation, the rotation for a rigid fit, the homogeneous
    deformation gradient ``F`` for an affine one. ``residual_rms`` is the root mean square distance between
    the displacement and the fitted motion over the nodes used, in the length unit of the fit.
    """

    kind: str
    n: int
    centre: NDArray[np.float64]
    translation: NDArray[np.float64]
    matrix: NDArray[np.float64]
    residual_rms: float

    @property
    def rotation(self) -> NDArray[np.float64]:
        """The rotation part: ``matrix`` itself, or the ``R`` of its polar decomposition for an affine fit."""
        if self.kind != "affine":
            return self.matrix
        from scipy.linalg import polar

        R, _stretch = polar(self.matrix, side="right")
        return R

    @property
    def rotation_vector_deg(self) -> NDArray[np.float64]:
        """Axis times angle, in degrees."""
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(self.rotation).as_rotvec(degrees=True)

    @property
    def rotation_deg(self) -> float:
        return float(np.linalg.norm(self.rotation_vector_deg))

    @property
    def euler_xyz_deg(self) -> NDArray[np.float64]:
        """Rotations about x, then y, then z (extrinsic), in degrees."""
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(self.rotation).as_euler("xyz", degrees=True)

    def model(self, X) -> NDArray[np.float64]:
        """The fitted displacement at positions ``X`` ``(N, 3)``."""
        X = np.asarray(X, dtype=np.float64)
        return self.translation + (X - self.centre) @ (self.matrix - _EYE).T

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "nodes": self.n,
            "centre": self.centre.tolist(),
            "translation": self.translation.tolist(),
            "matrix": self.matrix.tolist(),
            "rotation_deg": self.rotation_deg,
            "rotation_vector_deg": self.rotation_vector_deg.tolist(),
            "euler_xyz_deg": self.euler_xyz_deg.tolist(),
            "residual_rms": self.residual_rms,
        }


def fit_motion(X, U, kind: str, weights=None) -> MotionFit:
    """Fit a motion of ``kind`` (see :data:`MOTION_KINDS`) to displacements ``U`` at positions ``X`` ``(n, 3)``.

    Rows with a non-finite value or a zero weight are left out. Raises ``ValueError`` when fewer nodes remain
    than the fit needs (1 for a translation, 3 for a rigid motion, 4 for an affine one).
    """
    if kind not in MOTION_KINDS:
        raise ValueError(f"unknown motion {kind!r}; use one of {MOTION_KINDS}")
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    U = np.asarray(U, dtype=np.float64).reshape(-1, 3)
    w = np.ones(X.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    keep = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(U), axis=1) & np.isfinite(w) & (w > 0)
    X, U, w = X[keep], U[keep], w[keep]
    n = int(X.shape[0])
    if n < max(1, _MIN_NODES[kind]):
        raise ValueError(f"a {kind} fit needs at least {max(1, _MIN_NODES[kind])} nodes, {n} are usable")
    w = w / w.sum()
    Xc = w @ X
    x = X + U
    xc = w @ x
    P, Q = X - Xc, x - xc
    if kind == "none":
        return MotionFit("none", n, Xc, np.zeros(3), _EYE.copy(), float(np.sqrt(w @ np.sum(U * U, axis=1))))
    if kind == "translation":
        A = _EYE.copy()
    elif kind == "rigid":
        M = (Q * w[:, None]).T @ P  # sum w Q P^T; the best R maximises tr(R M^T)
        Um, _s, Vt = np.linalg.svd(M)
        d = 1.0 if np.linalg.det(Um @ Vt) >= 0 else -1.0  # a reflection is never a motion (planar node sets)
        A = Um @ np.diag([1.0, 1.0, d]) @ Vt
    else:  # affine: weighted least squares of Q = P A^T
        sw = np.sqrt(w)[:, None]
        At, *_ = np.linalg.lstsq(P * sw, Q * sw, rcond=None)
        A = At.T
    res = Q - P @ A.T
    return MotionFit(kind, n, Xc, xc - Xc, A, float(np.sqrt(w @ np.sum(res * res, axis=1))))


def remove_motion(X, U, fit: MotionFit) -> NDArray[np.float64]:
    """``U`` ``(N, 3)`` at positions ``X`` without the fitted motion (NaN stays NaN)."""
    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if fit.kind == "none":
        return U.copy()
    if fit.kind == "translation":
        return U - fit.translation
    if fit.kind == "rigid":  # the inverse motion applied to the deformed positions
        return (X + U - fit.centre - fit.translation) @ fit.matrix + fit.centre - X
    return U - fit.model(X)


def corrected_gradient(H, fit: MotionFit) -> NDArray[np.float64]:
    """The displacement gradient ``(N, 3, 3)`` of the corrected field (physical units)."""
    H = np.asarray(H, dtype=np.float64)
    if fit.kind == "rigid":
        return np.einsum("ji,njk->nik", fit.matrix, H + _EYE) - _EYE
    if fit.kind == "affine":
        return H - (fit.matrix - _EYE)
    return H.copy()
