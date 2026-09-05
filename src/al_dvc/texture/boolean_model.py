"""The Boolean model of spheres: a random texture with a closed-form correlation function.

Sphere centres form a Poisson point process of intensity ``lam`` (per voxel^3); the union of
equal spheres of radius ``R`` is the "solid" phase. The void phase has two-point probability
``S2(r) = exp(-lam [2 V - V_int(r)])`` with ``V`` the sphere volume and ``V_int(r)`` the volume
common to two spheres at distance ``r``, so the normalised autocorrelation of the indicator is

    rho(r) = exp(-lam V) (exp(lam V_int(r)) - 1) / (1 - exp(-lam V)),   rho = 0 for r >= 2 R.

It equals 1 at ``r = 0`` and is the same for both phases; it gives the correlation lengths an
exact reference (the scripts this replaces only compared sphere volumes by eye).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sphere_volume(radius: float) -> float:
    return 4.0 / 3.0 * np.pi * float(radius) ** 3


def intersection_volume(r, radius: float) -> NDArray[np.float64]:
    """Volume common to two spheres of ``radius`` whose centres are ``r`` apart (0 beyond ``2 R``)."""
    R = float(radius)
    r = np.asarray(r, dtype=np.float64)
    inside = np.clip(r, 0.0, 2.0 * R)
    v = np.pi / 12.0 * (4.0 * R + inside) * (2.0 * R - inside) ** 2
    return np.where(r < 2.0 * R, v, 0.0)


def intensity_for_fraction(radius: float, volume_fraction: float) -> float:
    """Poisson intensity (centres per voxel^3) giving the solid volume fraction ``1 - exp(-lam V)``."""
    phi = float(volume_fraction)
    if not 0.0 < phi < 1.0:
        raise ValueError(f"volume_fraction must be in (0, 1), got {volume_fraction!r}")
    return -np.log(1.0 - phi) / sphere_volume(radius)


def boolean_correlation(r, radius: float, volume_fraction: float) -> NDArray[np.float64]:
    """Normalised autocorrelation ``rho(r)`` of the Boolean model's indicator (either phase)."""
    lam = intensity_for_fraction(radius, volume_fraction)
    V = sphere_volume(radius)
    e = np.exp(-lam * V)
    return e * np.expm1(lam * intersection_volume(r, radius)) / (1.0 - e)


def analytic_length(radius: float, volume_fraction: float, threshold: float, tol: float = 1e-9) -> float:
    """Distance at which ``rho(r)`` falls through ``threshold`` (bisection on ``[0, 2 R]``)."""
    lo, hi = 0.0, 2.0 * float(radius)
    t = float(threshold)
    if not 0.0 < t < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if boolean_correlation(mid, radius, volume_fraction) > t:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def boolean_spheres(
    shape: tuple[int, int, int],
    radius: float,
    volume_fraction: float,
    seed: int = 0,
    spacing=1.0,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """A binary ``(nz, ny, nx)`` volume of the Boolean model and its sphere centres ``(n, 3)`` as ``(x, y, z)``.

    Centres are drawn in the box grown by ``R`` on every side, so spheres cut by the faces are as
    frequent as in an infinite medium. ``spacing`` (``(dx, dy, dz)`` or one number, physical
    units per voxel) makes the spheres round in physical space on anisotropic voxels; ``radius``
    is then in physical units too.
    """
    nz, ny, nx = (int(s) for s in shape)
    R = float(radius)
    sp = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (3,))
    dx, dy, dz = (float(v) for v in sp)
    lam = intensity_for_fraction(R, volume_fraction) / (dx * dy * dz)  # centres per voxel
    rng = np.random.default_rng(seed)
    grow = np.ceil(R / sp).astype(int)  # voxels of margin per axis (x, y, z)
    ext = (nx + 2 * grow[0], ny + 2 * grow[1], nz + 2 * grow[2])
    n = rng.poisson(lam * ext[0] * ext[1] * ext[2])
    centres = rng.uniform(-grow, np.array([nx, ny, nz]) + grow, size=(n, 3))  # (x, y, z) in voxels
    vol = np.zeros((nz, ny, nx), dtype=bool)
    rx, ry, rz = (int(np.ceil(R / d)) for d in (dx, dy, dz))
    for cx, cy, cz in centres:
        x0, x1 = max(0, int(np.floor(cx - rx))), min(nx - 1, int(np.ceil(cx + rx)))
        y0, y1 = max(0, int(np.floor(cy - ry))), min(ny - 1, int(np.ceil(cy + ry)))
        z0, z1 = max(0, int(np.floor(cz - rz))), min(nz - 1, int(np.ceil(cz + rz)))
        if x1 < x0 or y1 < y0 or z1 < z0:
            continue
        gx = (np.arange(x0, x1 + 1) - cx) * dx
        gy = (np.arange(y0, y1 + 1) - cy) * dy
        gz = (np.arange(z0, z1 + 1) - cz) * dz
        d2 = gz[:, None, None] ** 2 + gy[None, :, None] ** 2 + gx[None, None, :] ** 2
        vol[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] |= d2 <= R * R
    return vol.astype(np.float32), centres
