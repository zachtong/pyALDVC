"""Hex8 (8-node trilinear hexahedron) shape functions and quadrature.

Node ordering (matches MATLAB ``Subpb23.m`` / ``MeshSetUp3.m``)::

    n0 = (-1,-1,-1)  n1 = (+1,-1,-1)  n2 = (+1,+1,-1)  n3 = (-1,+1,-1)
    n4 = (-1,-1,+1)  n5 = (+1,-1,+1)  n6 = (+1,+1,+1)  n7 = (-1,+1,+1)

in natural coordinates ``(ksi, eta, zeta)`` mapped to ``(x, y, z)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Signs of each node in natural coordinates, shape (8, 3)
HEX8_SIGNS = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ],
    dtype=np.float64,
)


def gauss_points_1d(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1-D Gauss-Legendre points and weights on [-1, 1]."""
    if order == 1:
        return np.array([0.0]), np.array([2.0])
    if order == 2:
        g = 1.0 / np.sqrt(3.0)
        return np.array([-g, g]), np.array([1.0, 1.0])
    if order == 3:
        g = np.sqrt(3.0 / 5.0)
        return np.array([-g, 0.0, g]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    raise ValueError(f"Gauss order {order} not supported (use 1, 2 or 3)")


def hex8_gauss_points(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Tensor-product Gauss points ``(n_gp, 3)`` and weights ``(n_gp,)``."""
    p1, w1 = gauss_points_1d(order)
    pts = []
    wts = []
    for k, wk in zip(p1, w1):
        for j, wj in zip(p1, w1):
            for i, wi in zip(p1, w1):
                pts.append((i, j, k))
                wts.append(wi * wj * wk)
    return np.array(pts, dtype=np.float64), np.array(wts, dtype=np.float64)


def hex8_shape(ksi: float, eta: float, zeta: float) -> NDArray[np.float64]:
    """Shape function values ``N_a`` (8,) at a natural point."""
    s = HEX8_SIGNS
    return 0.125 * (1.0 + s[:, 0] * ksi) * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)


def hex8_dshape(ksi: float, eta: float, zeta: float) -> NDArray[np.float64]:
    """Natural derivatives ``dN_a/d(ksi, eta, zeta)`` as an (8, 3) array."""
    s = HEX8_SIGNS
    dN = np.empty((8, 3), dtype=np.float64)
    dN[:, 0] = 0.125 * s[:, 0] * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)
    dN[:, 1] = 0.125 * (1.0 + s[:, 0] * ksi) * s[:, 1] * (1.0 + s[:, 2] * zeta)
    dN[:, 2] = 0.125 * (1.0 + s[:, 0] * ksi) * (1.0 + s[:, 1] * eta) * s[:, 2]
    return dN


def hex8_box_matrices(
    spacing: tuple[float, float, float],
    order: int = 2,
) -> dict[str, NDArray[np.float64]]:
    """Element matrices of an axis-aligned box with edge lengths ``spacing``.

    Every element of a uniform grid mesh is the same box, so the element
    stiffness ``Ke = int grad(N)^T grad(N)``, mass ``Me = int N^T N`` and the
    three gradient-mass matrices ``Ge_j[a, b] = int (dN_a/dx_j) N_b`` are
    computed once and scattered into the global operators.

    Returns:
        dict with ``K`` (8, 8), ``M`` (8, 8), ``G`` (3, 8, 8), ``volume``.
    """
    hx, hy, hz = (float(s) for s in spacing)
    jac = np.diag([hx / 2.0, hy / 2.0, hz / 2.0])
    inv_jac = np.diag([2.0 / hx, 2.0 / hy, 2.0 / hz])
    det_j = float(np.prod(np.diag(jac)))
    pts, wts = hex8_gauss_points(order)

    K = np.zeros((8, 8), dtype=np.float64)
    M = np.zeros((8, 8), dtype=np.float64)
    G = np.zeros((3, 8, 8), dtype=np.float64)
    for (ksi, eta, zeta), w in zip(pts, wts):
        N = hex8_shape(ksi, eta, zeta)
        dN = hex8_dshape(ksi, eta, zeta) @ inv_jac  # (8, 3) physical derivatives
        wj = w * det_j
        K += wj * (dN @ dN.T)
        M += wj * np.outer(N, N)
        for j in range(3):
            G[j] += wj * np.outer(dN[:, j], N)
    return {"K": K, "M": M, "G": G, "volume": hx * hy * hz}
