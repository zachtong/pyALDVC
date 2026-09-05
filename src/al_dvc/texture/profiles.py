"""Correlation profiles: along the three axes and over spherical shells.

The radial profile bins the lags by their physical distance ``sqrt((hx dx)^2 + (hy dy)^2 +
(hz dz)^2)``; every bin reports the actual mean radius of its samples (the first shell of a
unit lattice has mean radius 1.416, not 1 and not 1.5), the number of samples, and how much of
the full spherical shell they cover (shells that run out of the lag box are incomplete). The
statistics are accumulated slab by slab with ``bincount``, so no coordinate grid of the size
of the autocorrelation is ever built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .acf import AXES, Autocorrelation


@dataclass(frozen=True)
class Profile:
    """A correlation curve against distance.

    ``lag`` is the mean index distance of the samples (voxels), ``distance`` their mean physical
    distance; ``std`` is the spread inside a bin (the direction dependence of the texture plus
    the estimate's noise, not a confidence interval); ``coverage`` is the fraction of the ideal
    shell present in the lag box (1 for the axis profiles).
    """

    axis: str
    lag: NDArray[np.float64]
    distance: NDArray[np.float64]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    count: NDArray[np.int64]
    coverage: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.lag.size)


def directional_profiles(ac: Autocorrelation) -> dict[str, Profile]:
    """The half-lines along ``x``, ``y`` and ``z`` from the zero lag (NaN beyond the reliable lag)."""
    out: dict[str, Profile] = {}
    for i, axis in enumerate(AXES):
        values = ac.line(axis)
        lag = ac.lags(axis)
        out[axis] = Profile(
            axis=axis,
            lag=lag,
            distance=lag * ac.spacing[i],
            mean=values,
            std=np.zeros_like(values),
            count=np.where(np.isfinite(values), 2, 0).astype(np.int64),
            coverage=np.ones_like(values),
        )
    return out


def radial_profile(ac: Autocorrelation, bin_width: float | None = None) -> Profile:
    """Shell statistics of the autocorrelation against physical distance.

    ``bin_width`` is in physical units and defaults to the smallest voxel edge, so on a unit
    lattice bin ``k`` holds the lags with ``k <= r < k + 1``.
    """
    dx, dy, dz = ac.spacing
    width = float(bin_width) if bin_width is not None else float(min(ac.spacing))
    if width <= 0:
        raise ValueError("bin_width must be positive")
    lx, ly, lz = ac.max_lag
    hx = np.arange(-lx, lx + 1, dtype=np.float64)
    hy = np.arange(-ly, ly + 1, dtype=np.float64)
    r2_xy = (hx[None, :] * dx) ** 2 + (hy[:, None] * dy) ** 2  # (2ly+1, 2lx+1)
    i2_xy = hx[None, :] ** 2 + hy[:, None] ** 2
    r_max = np.sqrt((lx * dx) ** 2 + (ly * dy) ** 2 + (lz * dz) ** 2)
    n_bins = int(np.floor(r_max / width)) + 1
    count = np.zeros(n_bins, dtype=np.int64)
    s1 = np.zeros(n_bins)
    s2 = np.zeros(n_bins)
    s_r = np.zeros(n_bins)
    s_i = np.zeros(n_bins)
    for k, hz in enumerate(range(-lz, lz + 1)):
        plane = np.asarray(ac.acf[k], dtype=np.float64)
        ok = np.isfinite(plane)
        if not ok.any():
            continue
        r = np.sqrt(r2_xy + (hz * dz) ** 2)[ok]
        ri = np.sqrt(i2_xy + float(hz) ** 2)[ok]
        v = plane[ok]
        b = np.minimum((r / width).astype(np.int64), n_bins - 1)
        count += np.bincount(b, minlength=n_bins)
        s1 += np.bincount(b, weights=v, minlength=n_bins)
        s2 += np.bincount(b, weights=v * v, minlength=n_bins)
        s_r += np.bincount(b, weights=r, minlength=n_bins)
        s_i += np.bincount(b, weights=ri, minlength=n_bins)
    keep = count > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(keep, s1 / count, np.nan)
        var = np.where(keep, s2 / count - mean * mean, np.nan)
        std = np.sqrt(np.clip(var, 0.0, None))
        distance = np.where(keep, s_r / count, np.nan)
        lag = np.where(keep, s_i / count, np.nan)
    edges = np.arange(n_bins + 1) * width
    ideal = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3) / (dx * dy * dz)
    coverage = np.clip(count / np.maximum(ideal, 1e-300), 0.0, 1.0)
    coverage[0] = 1.0 if count[0] else 0.0  # the zero-lag bin is a single sample, not a shell
    return Profile(
        axis="radial",
        lag=lag[keep],
        distance=distance[keep],
        mean=mean[keep],
        std=std[keep],
        count=count[keep],
        coverage=coverage[keep],
    )
