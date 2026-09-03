"""Synthetic volumetric speckle data with exact ground-truth deformation.

Used by the test-suite, the CLI ``synth`` command and the validation
reports. Deformed volumes are generated in the *Lagrangian* convention the
solver uses (``x = X + u(X)``) by inverting the mapping with a fixed-point
iteration and sampling the reference with a quintic B-spline, so the only
error left is the interpolation of the DVC itself.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates

DispFunc = Callable[[NDArray, NDArray, NDArray], tuple[NDArray, NDArray, NDArray]]


def generate_speckle_volume(
    shape: tuple[int, int, int] = (96, 96, 96),
    sigma: float = 2.0,
    seed: int = 0,
    contrast: float = 1.0,
) -> NDArray[np.float32]:
    """Gaussian-filtered random noise, rescaled to ``[0, 1]`` (``(nz, ny, nx)``)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape)
    vol = gaussian_filter(noise, sigma=sigma, mode="nearest")
    vol -= vol.min()
    vol /= max(vol.max(), 1e-12)
    vol = 0.5 + contrast * (vol - 0.5)
    return np.ascontiguousarray(vol, dtype=np.float32)


def generate_bead_volume(
    shape: tuple[int, int, int] = (96, 96, 96),
    n_beads: int = 4000,
    radius: float = 2.0,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Random Gaussian beads on a dark background (micro-CT-like tracer field)."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    vol = np.zeros(shape, dtype=np.float32)
    centres = rng.uniform(0, [nz, ny, nx], size=(n_beads, 3))
    r = int(np.ceil(3 * radius))
    for cz, cy, cx in centres:
        z0, y0, x0 = int(round(cz)), int(round(cy)), int(round(cx))
        zs = slice(max(0, z0 - r), min(nz, z0 + r + 1))
        ys = slice(max(0, y0 - r), min(ny, y0 + r + 1))
        xs = slice(max(0, x0 - r), min(nx, x0 + r + 1))
        Z, Y, X = np.mgrid[zs, ys, xs]
        vol[zs, ys, xs] += np.exp(-((Z - cz) ** 2 + (Y - cy) ** 2 + (X - cx) ** 2) / (2 * radius**2)).astype(np.float32)
    vol /= max(vol.max(), 1e-12)
    return np.ascontiguousarray(vol, dtype=np.float32)


def affine_displacement(
    F: NDArray[np.float64] | None = None,
    t: tuple[float, float, float] = (0.0, 0.0, 0.0),
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> DispFunc:
    """``u(X) = t + F (X - c)`` with ``F`` the displacement gradient (3x3)."""
    Fm = np.zeros((3, 3)) if F is None else np.asarray(F, dtype=np.float64)
    cx, cy, cz = centre
    tx, ty, tz = t

    def fn(x, y, z):
        dx, dy, dz = x - cx, y - cy, z - cz
        u = tx + Fm[0, 0] * dx + Fm[0, 1] * dy + Fm[0, 2] * dz
        v = ty + Fm[1, 0] * dx + Fm[1, 1] * dy + Fm[1, 2] * dz
        w = tz + Fm[2, 0] * dx + Fm[2, 1] * dy + Fm[2, 2] * dz
        return u, v, w

    return fn


def sinusoidal_displacement(amplitude: float, wavelength: float, centre=(0.0, 0.0, 0.0)) -> DispFunc:
    """``u = A sin(2 pi (y-c)/L)``, ``v = A sin(2 pi (z-c)/L)``, ``w = A sin(2 pi (x-c)/L)``."""
    k = 2.0 * np.pi / wavelength
    cx, cy, cz = centre

    def fn(x, y, z):
        return amplitude * np.sin(k * (y - cy)), amplitude * np.sin(k * (z - cz)), amplitude * np.sin(k * (x - cx))

    return fn


def rotation_displacement(angle_deg: float, axis: str = "z", centre=(0.0, 0.0, 0.0)) -> DispFunc:
    """Rigid rotation about a coordinate axis through ``centre``."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "z":
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return affine_displacement(R - np.eye(3), (0.0, 0.0, 0.0), centre)


def warp_volume_lagrangian(
    ref: NDArray,
    disp: DispFunc,
    n_iter: int = 25,
    order: int = 5,
) -> NDArray[np.float32]:
    """Deformed volume ``g(x) = f(X)`` with ``x = X + u(X)`` (fixed-point inversion)."""
    ref64 = np.asarray(ref, dtype=np.float64)
    nz, ny, nx = ref64.shape
    Z, Y, X = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float64)
    Xr, Yr, Zr = X.copy(), Y.copy(), Z.copy()
    for _ in range(n_iter):
        u, v, w = disp(Xr, Yr, Zr)
        Xr, Yr, Zr = X - u, Y - v, Z - w
    coords = np.vstack([Zr.ravel(), Yr.ravel(), Xr.ravel()])
    g = map_coordinates(ref64, coords, order=order, mode="nearest").reshape(ref64.shape)
    return np.ascontiguousarray(g, dtype=np.float32)


def add_noise(vol: NDArray, sigma: float, seed: int = 1) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(np.asarray(vol, dtype=np.float32) + sigma * rng.standard_normal(vol.shape).astype(np.float32))


def evaluate_at_nodes(disp: DispFunc, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ground-truth ``(N, 3)`` displacement at node coordinates ``[x, y, z]``."""
    u, v, w = disp(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    return np.column_stack([u, v, w]).astype(np.float64)


def gradient_at_nodes(disp: DispFunc, coordinates: NDArray[np.float64], h: float = 1e-3) -> NDArray[np.float64]:
    """Ground-truth ``(N, 3, 3)`` displacement gradient by central differences."""
    F = np.empty((coordinates.shape[0], 3, 3))
    for j in range(3):
        cp = coordinates.copy()
        cm = coordinates.copy()
        cp[:, j] += h
        cm[:, j] -= h
        up = evaluate_at_nodes(disp, cp)
        um = evaluate_at_nodes(disp, cm)
        F[:, :, j] = (up - um) / (2 * h)
    return F
