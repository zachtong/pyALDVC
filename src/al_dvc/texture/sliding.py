"""Sliding-window autocorrelation: a window is compared with the volume around it.

The window ``W`` sits inside a larger range ``R``; for every shift ``h`` that keeps ``W + h`` inside
``R``::

    C(h) = sum_{x in W} f~(x) f~(x + h),      rho(h) = C(h) / C(0),

with ``f~`` the grey values minus their mean over the range. Every ``C(h)`` sums the same number of
voxel pairs -- the window's -- so no overlap correction is needed and the finite-window decay of
:mod:`~al_dvc.texture.acf` cannot appear.

This is *not* the estimator the application uses. It reads grey values from outside the window, so
the answer of a small window rests partly on data the user did not select, which is exactly what a
size sweep must not do; :mod:`~al_dvc.texture.concentric` does the analysis instead. It is kept as
the reference for the comparison in ``reports/texture.pdf``: at a fixed size, and where the data
around the window really is material, it is the lowest-variance estimator of the three.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfftn, next_fast_len, rfftn

from .acf import NO_TEXTURE_RELATIVE_VARIANCE, Autocorrelation, _as_triple, _centred_lags
from .analysis import TextureResult, result_from_acf
from .boxes import Box, centred_window, normalise_box
from .crossing import THRESHOLDS

__all__ = ["analyse_range", "lag_reach", "sliding_autocorrelation"]


def lag_reach(box: Box, window: Box) -> tuple[int, int, int]:
    """Largest shift ``(Lx, Ly, Lz)`` that keeps ``window`` inside ``box``."""
    return tuple(int(min(w0 - b0, b1 - w1)) for (b0, b1), (w0, w1) in zip(box, window))  # type: ignore[return-value]


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
    return result_from_acf(ac, thresholds, radial_bin, window_zyx, settings)  # type: ignore[arg-type]
