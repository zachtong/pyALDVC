"""Sliding-window autocorrelation: a window moves inside an analysis range.

The user picks an *analysis range* ``R`` (a box of the volume) and a *window* ``W`` (a box centred
in ``R``). For every shift ``h`` the window is compared with the volume shifted by ``h``::

    C(h) = sum_{x in W} f~(x) f~(x + h),      rho(h) = C(h) / C(0),

with ``f~`` the grey values minus their mean over the range. The shifted window ``W + h`` must stay
inside ``R``, so the largest shift per axis is ``L_j = (R_j - w_j) // 2`` and every ``C(h)`` sums the
same number of voxel pairs (the window's). No overlap correction is needed and the estimate does not
depend on the range beyond the shifts it allows.

The window size analysis (``sweep_concentric``) repeats the estimate for windows of growing size,
all centred in the range like the concentric regions of the DVC Challenge 2.0 paper; the size from
which the correlation lengths stop changing is the representative volume for the texture.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfftn, next_fast_len, rfftn

from .acf import AXES, NO_TEXTURE_RELATIVE_VARIANCE, Autocorrelation, _as_triple, _centred_lags
from .analysis import TextureResult, _noise_floor, _periodicity
from .crossing import THRESHOLDS, Crossing, lengths
from .profiles import Profile, directional_profiles, radial_profile
from .rve import (
    DEFAULT_MIN_SPAN,
    DEFAULT_TOLERANCE_ABS,
    DEFAULT_TOLERANCE_REL,
    SizeLevel,
    SizeSweep,
    SubVolume,
    decide_plateau,
)

Box = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]  # ((x0, x1), (y0, y1), (z0, z1)), half-open
DEFAULT_MIN_LAG = 16  # the window size analysis keeps at least this many voxels of shift on every axis
MAX_RANGE_VOXELS = 400**3  # one FFT of the range must fit in memory

__all__ = [
    "Box",
    "DEFAULT_MIN_LAG",
    "MAX_RANGE_VOXELS",
    "analyse_range",
    "box_of_mask",
    "box_size",
    "centred_window",
    "lag_reach",
    "normalise_box",
    "sliding_autocorrelation",
    "sweep_concentric",
    "sweep_sizes_concentric",
    "whole_box",
]


# ----------------------------------------------------------------------------- boxes
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
            raise ValueError(f"the analysis range must span at least 2 voxels on every axis, got {box}")
        out.append((lo, hi))
    return tuple(out)  # type: ignore[return-value]


def box_size(box: Box) -> tuple[int, int, int]:
    """``(nx, ny, nz)`` of a box."""
    return tuple(int(b - a) for a, b in box)  # type: ignore[return-value]


def box_of_mask(mask: NDArray) -> Box:
    """Bounding box of the True voxels of ``mask`` ``(nz, ny, nx)``."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3 or not m.any():
        raise ValueError("mask must be a 3-D array with at least one True voxel")
    zs, ys, xs = (np.flatnonzero(m.any(axis=ax)) for ax in ((1, 2), (0, 2), (0, 1)))
    return ((int(xs[0]), int(xs[-1]) + 1), (int(ys[0]), int(ys[-1]) + 1), (int(zs[0]), int(zs[-1]) + 1))


def centred_window(box: Box, size) -> Box:
    """A window of ``size`` ``(wx, wy, wz)`` (or one edge) centred in ``box``, clipped to it."""
    w = _as_triple(size, "window size", int, minimum=1)
    out = []
    for (lo, hi), wj in zip(box, w):
        n = hi - lo
        wj = int(min(max(2, wj), n))
        start = lo + (n - wj) // 2
        out.append((start, start + wj))
    return tuple(out)  # type: ignore[return-value]


def lag_reach(box: Box, window: Box) -> tuple[int, int, int]:
    """Largest shift ``(Lx, Ly, Lz)`` that keeps ``window`` inside ``box``."""
    return tuple(int(min(w0 - b0, b1 - w1)) for (b0, b1), (w0, w1) in zip(box, window))  # type: ignore[return-value]


# ----------------------------------------------------------------------------- estimator
def sliding_autocorrelation(vol: NDArray, box, window_size, spacing=1.0, dtype=np.float32) -> tuple[Autocorrelation, Box]:
    """Autocorrelation of the window centred in ``box`` against the volume inside ``box``.

    Returns the :class:`Autocorrelation` (lags ``(-L_j .. L_j)`` per axis, ``estimator="sliding"``) and
    the window box that was used. ``status`` is ``no_texture`` when the window has no grey-value variation.
    """
    a = np.asarray(vol)
    if a.ndim != 3:
        raise ValueError(f"vol must be 3-D (nz, ny, nx), got shape {a.shape}")
    spacing_xyz = _as_triple(spacing, "spacing")
    box = normalise_box(box, a.shape)
    window = centred_window(box, window_size)
    lag_xyz = lag_reach(box, window)
    (x0, x1), (y0, y1), (z0, z1) = box
    (wx0, wx1), (wy0, wy1), (wz0, wz1) = window
    region = a[z0:z1, y0:y1, x0:x1]
    if not np.all(np.isfinite(region)):
        raise ValueError("the analysis range has non-finite values")
    rel = (slice(wz0 - z0, wz1 - z0), slice(wy0 - y0, wy1 - y0), slice(wx0 - x0, wx1 - x0))
    mean = float(np.mean(region, dtype=np.float64))
    win_values = region[rel].astype(np.float64)
    n_w = int(win_values.size)
    variance = float(np.var(win_values))

    def _empty(status: str) -> Autocorrelation:
        acf = np.full(tuple(2 * L + 1 for L in lag_xyz[::-1]), np.nan, dtype=dtype)
        return Autocorrelation(acf, lag_xyz, "sliding", spacing_xyz, variance, n_w, region.shape, status, 1.0)

    if n_w < 8 or variance <= NO_TEXTURE_RELATIVE_VARIANCE * max(1.0, mean * mean):
        return _empty("no_texture"), window
    u = (region.astype(np.float64) - mean).astype(dtype)
    w = np.zeros_like(u)
    w[rel] = u[rel]
    fft_shape = tuple(next_fast_len(int(n)) for n in region.shape)  # W + h stays inside R: no wrap for |h| <= L
    FR = rfftn(u, s=fft_shape, workers=-1)
    FW = rfftn(w, s=fft_shape, workers=-1)
    del u, w
    FW = np.conj(FW)
    FW *= FR
    del FR
    full = irfftn(FW, s=fft_shape, workers=-1)  # sum_x W(x) R(x + h) at index h
    del FW
    C = _centred_lags(full, lag_xyz).astype(np.float64)
    del full
    c = tuple(L for L in lag_xyz[::-1])
    c0 = float(C[c])
    if c0 <= 0.0:
        return _empty("no_texture"), window
    rho = (C / c0).astype(dtype)
    rho[c] = 1.0
    return Autocorrelation(rho, lag_xyz, "sliding", spacing_xyz, c0 / n_w, n_w, region.shape, "ok", 1.0), window


def analyse_range(
    vol: NDArray,
    box,
    window_size,
    spacing=1.0,
    thresholds=THRESHOLDS,
    radial_bin: float | None = None,
) -> TextureResult:
    """Profiles, correlation lengths, noise floor and periodicity of the window centred in ``box``."""
    a = np.asarray(vol)
    ac, window = sliding_autocorrelation(a, box, window_size, spacing)
    box = normalise_box(box, a.shape)
    window_zyx = tuple(slice(int(lo), int(hi)) for lo, hi in window[::-1])
    settings = {
        "estimator": "sliding",
        "range": tuple(tuple(int(v) for v in pair) for pair in box),
        "window": tuple((s.start, s.stop) for s in window_zyx),  # (z, y, x) like the old field
        "window_xyz": tuple(tuple(int(v) for v in pair) for pair in window),
        "max_lag": ac.max_lag,
        "thresholds": tuple(float(t) for t in thresholds),
        "n_voxels": ac.n_voxels,
    }
    if not ac.ok:
        empty = {
            axis: {float(t): Crossing(float(t), None, "invalid", None, "no texture") for t in thresholds}
            for axis in (*AXES, "radial")
        }
        nan = np.array([np.nan])
        profiles = {axis: Profile(axis, nan, nan, nan, nan, np.array([0]), nan) for axis in (*AXES, "radial")}
        return TextureResult(ac, profiles, empty, empty, float("nan"), None, window_zyx, settings)
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
        window=window_zyx,
        settings=settings,
    )


# ----------------------------------------------------------------------------- window size analysis
def sweep_sizes_concentric(
    box: Box, start: int = 16, step: int = 16, min_lag: int = DEFAULT_MIN_LAG
) -> list[tuple[int, int, int]]:
    """Window sizes ``(wx, wy, wz)`` centred in ``box``: ``start + i * step`` per axis, each axis capped so
    that ``min_lag`` voxels of shift remain on both sides; the list ends when no axis grows any more."""
    start, step, min_lag = int(start), int(step), int(min_lag)
    if start < 2 or step < 1 or min_lag < 0:
        raise ValueError("start >= 2, step >= 1 and min_lag >= 0 are required")
    caps = tuple(max(2, n - 2 * min_lag) for n in box_size(box))
    if min(caps) < start:
        raise ValueError(
            f"the range {box_size(box)} leaves no room for a {start} voxel window with {min_lag} voxels of shift on every axis"
        )
    sizes: list[tuple[int, int, int]] = []
    i = 0
    while True:
        s = tuple(int(min(start + i * step, cap)) for cap in caps)
        if sizes and s == sizes[-1]:
            break
        sizes.append(s)  # type: ignore[arg-type]
        i += 1
    return sizes


def sweep_concentric(
    vol: NDArray,
    box,
    start: int = 16,
    step: int = 16,
    min_lag: int = DEFAULT_MIN_LAG,
    spacing=1.0,
    thresholds=THRESHOLDS,
    axis: str = "radial",
    tolerance_rel: float = DEFAULT_TOLERANCE_REL,
    tolerance_abs: float = DEFAULT_TOLERANCE_ABS,
    min_span: float = DEFAULT_MIN_SPAN,
    progress: Callable[[float, str], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> SizeSweep:
    """Correlation lengths against the size of a window centred in ``box`` (concentric windows).

    Every window is analysed with :func:`analyse_range` against the same range, so all of them see
    the same shifts (at least ``min_lag`` voxels). One window per size: the spread across positions
    is not measured (the plateau test then rests on the size trend alone).
    """
    a = np.asarray(vol)
    box = normalise_box(box, a.shape)
    sizes = sweep_sizes_concentric(box, start, step, min_lag)
    levels: list[SizeLevel] = []
    for i, size in enumerate(sizes):
        if stop is not None and stop():
            break
        if progress is not None:
            progress(i / max(1, len(sizes)), " x ".join(str(v) for v in size))
        res = analyse_range(a, box, size, spacing, thresholds)
        (wx0, wx1), (wy0, wy1), (wz0, wz1) = res.settings["window_xyz"]
        sub = SubVolume((wz0, wz1, wy0, wy1, wx0, wx1), {float(t): res.length(axis, t) for t in thresholds})
        mean = {float(t): (float(v) if (v := sub.lengths[float(t)]) is not None else float("nan")) for t in thresholds}
        std = {float(t): float("nan") for t in thresholds}
        n_valid = {float(t): int(sub.lengths[float(t)] is not None) for t in thresholds}
        levels.append(SizeLevel(sub.size, [sub], mean, std, n_valid, res.profiles.get(axis)))
    if progress is not None:
        progress(1.0, "done")
    eff = np.array([lvl.effective for lvl in levels], dtype=np.float64)
    decisions = {
        float(t): decide_plateau(
            eff,
            [lvl.mean[float(t)] for lvl in levels],
            [lvl.std[float(t)] for lvl in levels],
            float(t),
            tolerance_rel,
            tolerance_abs,
            min_span,
        )
        for t in thresholds
    }
    settings = {
        "axis": axis,
        "estimator": "sliding",
        "range": tuple(tuple(int(v) for v in pair) for pair in box),
        "start": int(start),
        "step": int(step),
        "min_lag": int(min_lag),
        "tolerance_rel": tolerance_rel,
        "tolerance_abs": tolerance_abs,
        "min_span": min_span,
    }
    return SizeSweep(levels, decisions, axis, settings)
