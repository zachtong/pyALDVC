"""One call from a volume to its texture summary: autocorrelation, profiles, lengths, noise, periodicity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .acf import AXES, DEFAULT_MIN_OVERLAP, Autocorrelation, autocorrelation
from .crossing import THRESHOLDS, Crossing, lengths
from .profiles import Profile, directional_profiles, radial_profile

DEFAULT_MAX_VOXELS = 256**3  # the analysis window is cropped to this many voxels (FFT memory)
PERIODICITY_MIN_HEIGHT = 0.1  # a secondary peak of the radial profile above this is reported
NOISE_FLOOR_LENGTHS = 3.0  # the noise floor is measured beyond this many 1/e lengths


@dataclass(frozen=True)
class TextureResult:
    """Everything the texture window and the report show.

    ``lengths[axis][threshold]`` holds a :class:`Crossing` in voxels for ``axis`` in
    ``("x", "y", "z", "radial")``; ``physical_lengths`` the same in physical units.
    """

    acf: Autocorrelation
    profiles: dict[str, Profile]
    lengths: dict[str, dict[float, Crossing]]
    physical_lengths: dict[str, dict[float, Crossing]]
    noise_floor: float
    periodicity: tuple[str, float, float] | None  # (axis, distance, height) of the strongest secondary peak
    window: tuple[slice, slice, slice]  # the part of the volume that was analysed (z, y, x)
    settings: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.acf.status

    def length(self, axis: str, threshold: float = THRESHOLDS[0]) -> float | None:
        return self.lengths[axis][float(threshold)].value


def analysis_window(
    shape: tuple[int, int, int], mask: NDArray | None = None, max_voxels: int = DEFAULT_MAX_VOXELS
) -> tuple[slice, slice, slice]:
    """The box to analyse: the mask's bounding box (or the whole volume), shrunk about its centre
    to at most ``max_voxels`` voxels while keeping its aspect ratio."""
    nz, ny, nx = (int(s) for s in shape)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (nz, ny, nx):
            raise ValueError(f"mask shape {m.shape} does not match the volume shape {(nz, ny, nx)}")
        if not m.any():
            raise ValueError("mask is empty")
        zs, ys, xs = (np.flatnonzero(m.any(axis=ax)) for ax in ((1, 2), (0, 2), (0, 1)))
        lo = np.array([zs[0], ys[0], xs[0]])
        hi = np.array([zs[-1], ys[-1], xs[-1]]) + 1
    else:
        lo = np.zeros(3, dtype=int)
        hi = np.array([nz, ny, nx])
    size = hi - lo
    total = int(np.prod(size))
    if total > max_voxels:
        f = (max_voxels / total) ** (1.0 / 3.0)
        new = np.maximum(2, np.floor(size * f).astype(int))
        centre = (lo + hi) // 2
        lo = np.maximum(0, centre - new // 2)
        hi = np.minimum([nz, ny, nx], lo + new)
        lo = hi - new
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _noise_floor(radial: Profile, one_over_e: float | None) -> float:
    """Standard deviation of the radial profile beyond three 1/e lengths (or beyond half its range)."""
    finite = np.isfinite(radial.mean)
    if not finite.any():
        return float("nan")
    start = NOISE_FLOOR_LENGTHS * one_over_e if one_over_e else 0.5 * float(np.nanmax(radial.lag))
    tail = finite & (radial.lag >= start)
    if tail.sum() < 3:
        return float("nan")
    return float(np.std(radial.mean[tail]))


def _secondary_peak(profile: Profile) -> tuple[float, float] | None:
    """The first local maximum of a profile after its first minimum, if it is a real peak.

    The first maximum is the period; the global maximum of the tail could be the last sample
    of a still-rising curve, which is no peak at all.
    """
    y = profile.mean
    finite = np.isfinite(y)
    if finite.sum() < 5:
        return None
    y = y[finite]
    d = profile.distance[finite]
    i_min = None
    for i in range(1, y.size - 1):
        if y[i] <= y[i - 1] and y[i] < y[i + 1]:
            i_min = i
            break
    if i_min is None:
        return None
    for j in range(i_min + 1, y.size - 1):
        if y[j] >= y[j - 1] and y[j] > y[j + 1]:
            if y[j] >= PERIODICITY_MIN_HEIGHT and y[j] > y[i_min] + 0.02:
                return float(d[j]), float(y[j])
            return None  # the first bump after the minimum is too small to be a period
    return None


def _periodicity(profiles: dict[str, Profile]) -> tuple[str, float, float] | None:
    """``(axis, distance, height)`` of the strongest secondary peak over the axis and radial profiles.

    A periodic texture along one axis shows its period on that axis profile; the shell average
    smears it (a cosine along x averages to a sinc over the spheres), so every profile is tried.
    """
    best = None
    for axis, profile in profiles.items():
        peak = _secondary_peak(profile)
        if peak is not None and (best is None or peak[1] > best[2]):
            best = (axis, peak[0], peak[1])
    return best


def analyse_texture(
    vol: NDArray,
    spacing=1.0,
    mask: NDArray | None = None,
    max_lag=None,
    estimator: str = "overlap",
    thresholds=THRESHOLDS,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    radial_bin: float | None = None,
) -> TextureResult:
    """Autocorrelation, profiles and correlation lengths of the region of ``vol`` selected by ``mask``."""
    a = np.asarray(vol)
    if a.ndim != 3:
        raise ValueError(f"vol must be 3-D (nz, ny, nx), got shape {a.shape}")
    window = analysis_window(a.shape, mask, max_voxels)
    sub = a[window]
    sub_mask = np.asarray(mask, dtype=bool)[window] if mask is not None else None
    if sub_mask is not None and sub_mask.all():
        sub_mask = None  # a full box needs no mask correction
    ac = autocorrelation(sub, spacing, max_lag, estimator, sub_mask, min_overlap)
    settings = {
        "estimator": estimator,
        "max_lag": ac.max_lag,
        "min_overlap": min_overlap,
        "thresholds": tuple(float(t) for t in thresholds),
        "window": tuple((s.start, s.stop) for s in window),
        "n_voxels": ac.n_voxels,
    }
    if not ac.ok:
        empty = {
            axis: {float(t): Crossing(float(t), None, "invalid", None, "no texture") for t in thresholds}
            for axis in (*AXES, "radial")
        }
        nan = np.array([np.nan])
        empty_profile = Profile("radial", nan, nan, nan, nan, np.array([0]), nan)
        profiles = {axis: Profile(axis, nan, nan, nan, nan, np.array([0]), nan) for axis in AXES}
        profiles["radial"] = empty_profile
        return TextureResult(ac, profiles, empty, empty, float("nan"), None, window, settings)
    profiles = directional_profiles(ac)
    profiles["radial"] = radial_profile(ac, radial_bin)
    voxel = {axis: lengths(p, thresholds, physical=False) for axis, p in profiles.items()}
    physical = {axis: lengths(p, thresholds, physical=True) for axis, p in profiles.items()}
    one_over_e = voxel["radial"][float(thresholds[0])].value
    return TextureResult(
        acf=ac,
        profiles=profiles,
        lengths=voxel,
        physical_lengths=physical,
        noise_floor=_noise_floor(profiles["radial"], one_over_e),
        periodicity=_periodicity(profiles),
        window=window,
        settings=settings,
    )
