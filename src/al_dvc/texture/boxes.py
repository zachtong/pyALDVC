"""Boxes of a volume: the shared vocabulary of the texture analysis.

A box is ``((x0, x1), (y0, y1), (z0, z1))``, half-open and in voxels, like every coordinate triple
in the package (x, y, z) while the arrays themselves run ``(nz, ny, nx)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Box = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

MAX_ANALYSIS_VOXELS = 400**3  # one FFT of the analysed box must fit in memory

__all__ = [
    "MAX_ANALYSIS_VOXELS",
    "Box",
    "box_centre",
    "box_of_mask",
    "box_size",
    "centred_window",
    "normalise_box",
    "whole_box",
]


def whole_box(shape: tuple[int, int, int]) -> Box:
    """The whole volume ``(nz, ny, nx)`` as a box ``((0, nx), (0, ny), (0, nz))``."""
    nz, ny, nx = (int(s) for s in shape)
    return ((0, nx), (0, ny), (0, nz))


def normalise_box(box, shape: tuple[int, int, int]) -> Box:
    """Integer, ordered, clipped box; raises ``ValueError`` when it is empty."""
    nz, ny, nx = (int(s) for s in shape)
    out = []
    for (a, b), n in zip(box, (nx, ny, nz)):
        lo, hi = sorted((int(a), int(b)))
        lo, hi = max(0, lo), min(n, hi)
        if hi - lo < 2:
            raise ValueError(f"the analysis region must span at least 2 voxels on every axis, got {box}")
        out.append((lo, hi))
    return tuple(out)  # type: ignore[return-value]


def box_size(box: Box) -> tuple[int, int, int]:
    """``(nx, ny, nz)`` of a box."""
    return tuple(int(b - a) for a, b in box)  # type: ignore[return-value]


def box_centre(box: Box) -> tuple[int, int, int]:
    """``(cx, cy, cz)`` of a box: the voxel a cube of the same parity would be built around."""
    return tuple(int(a + (b - a) // 2) for a, b in box)  # type: ignore[return-value]


def box_of_mask(mask: NDArray) -> Box:
    """Bounding box of the True voxels of ``mask`` ``(nz, ny, nx)``."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3 or not m.any():
        raise ValueError("mask must be a 3-D array with at least one True voxel")
    zs, ys, xs = (np.flatnonzero(m.any(axis=ax)) for ax in ((1, 2), (0, 2), (0, 1)))
    return ((int(xs[0]), int(xs[-1]) + 1), (int(ys[0]), int(ys[-1]) + 1), (int(zs[0]), int(zs[-1]) + 1))


def centred_window(box: Box, size) -> Box:
    """A window of ``size`` ``(wx, wy, wz)`` (or one edge) centred in ``box``, clipped to it."""
    w = np.broadcast_to(np.asarray(size, dtype=int), (3,))
    if np.any(w < 1):
        raise ValueError(f"window size must be positive, got {size!r}")
    out = []
    for (lo, hi), wj in zip(box, w):
        n = hi - lo
        wj = int(min(max(2, int(wj)), n))
        start = lo + (n - wj) // 2
        out.append((start, start + wj))
    return tuple(out)  # type: ignore[return-value]
