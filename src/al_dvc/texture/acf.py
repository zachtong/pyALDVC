"""Linear 3-D autocorrelation of a volume by FFT, correctly centred, with an explicit estimator.

The autocorrelation of the mean-subtracted volume ``u`` is ``S(h) = sum_x u(x) u(x + h)`` over
the voxels where both ``x`` and ``x + h`` lie in the region. Two normalisations are offered:

``overlap``
    ``rho(h) = [S(h) / M(h)] / [S(0) / M(0)]`` with ``M(h)`` the number of voxel pairs that
    contribute to lag ``h`` (the overlap count). Without a mask ``M(h) = prod_j (N_j - |h_j|)``;
    with a mask ``M`` is the autocorrelation of the mask itself. This removes the finite-window
    factor ``prod_j (1 - |h_j| / N_j)`` that otherwise makes the curve decay faster in smaller
    volumes (a sub-volume comparison would then see a size effect that is not in the texture).
``window``
    ``rho(h) = S(h) / S(0)``: the finite-window estimator of the original DVC Challenge scripts,
    kept for comparison.

Both are evaluated only where the overlap is at least ``min_overlap`` of the region (NaN
elsewhere); beyond that the estimate rests on too few pairs to be reliable.

The inverse FFT is shifted before the lags ``[-L_j, L_j]`` are cut around the true zero lag.
Cutting first and shifting afterwards, as the original scripts did, mislabels the negative
lags whenever the FFT length exceeds ``2 N - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfftn, next_fast_len, rfftn

ESTIMATORS = ("overlap", "window")
DEFAULT_MIN_OVERLAP = 0.5  # fraction of the region's voxel pairs a lag must keep to be reported
NO_TEXTURE_RELATIVE_VARIANCE = 1e-12  # variance below this times max(1, mean^2) is "no texture"
AXES = ("x", "y", "z")  # index 0, 1, 2 of every (x, y, z) tuple; array axes run (z, y, x)


@dataclass(frozen=True)
class Autocorrelation:
    """The centred autocorrelation and how it was estimated.

    ``acf`` has shape ``(2 Lz + 1, 2 Ly + 1, 2 Lx + 1)`` with the zero lag at ``centre``; entries
    whose overlap is below ``min_overlap`` are NaN. ``spacing`` and ``max_lag`` are ``(x, y, z)``
    tuples like every coordinate triple in the package.
    """

    acf: NDArray[np.float32]
    max_lag: tuple[int, int, int]
    estimator: str
    spacing: tuple[float, float, float]
    variance: float
    n_voxels: int
    shape: tuple[int, int, int]
    status: str
    min_overlap: float

    @property
    def centre(self) -> tuple[int, int, int]:
        """Array index ``(iz, iy, ix)`` of the zero lag."""
        lx, ly, lz = self.max_lag
        return lz, ly, lx

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def lags(self, axis: str) -> NDArray[np.float64]:
        """Lags ``0 .. L`` (voxels) along ``axis``."""
        return np.arange(self.max_lag[AXES.index(axis)] + 1, dtype=np.float64)

    def line(self, axis: str) -> NDArray[np.float64]:
        """The half-line of the autocorrelation from the zero lag along ``+axis`` (the ``-axis`` half
        is averaged in: the two are equal up to floating-point round-off)."""
        cz, cy, cx = self.centre
        if axis == "x":
            plus, minus = self.acf[cz, cy, cx:], self.acf[cz, cy, cx::-1]
        elif axis == "y":
            plus, minus = self.acf[cz, cy:, cx], self.acf[cz, cy::-1, cx]
        elif axis == "z":
            plus, minus = self.acf[cz:, cy, cx], self.acf[cz::-1, cy, cx]
        else:
            raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
        return 0.5 * (np.asarray(plus, dtype=np.float64) + np.asarray(minus, dtype=np.float64))


def _as_triple(value, name: str, kind=float, minimum=0.0) -> tuple:
    arr = np.broadcast_to(np.asarray(value, dtype=kind), (3,))
    out = tuple(kind(v) for v in arr)
    if any(v < minimum for v in out) or (minimum == 0.0 and kind is float and any(v <= 0 for v in out)):
        raise ValueError(f"{name} must be positive, got {value!r}")
    return out


def _centred_lags(full: NDArray, lag_xyz: tuple[int, int, int]) -> NDArray:
    """Cut lags ``[-L_j, L_j]`` out of a circular correlation whose zero lag sits at index 0."""
    lx, ly, lz = lag_xyz
    idx = []
    for m, L in zip(full.shape, (lz, ly, lx)):
        idx.append(np.r_[m - L : m, 0 : L + 1])
    return full[np.ix_(*idx)]


def _fft_autocorrelation(u: NDArray, fft_shape: tuple[int, int, int]) -> NDArray:
    """``sum_x u(x) u(x + h)`` for every circular lag of the zero-padded ``u`` (real FFT)."""
    F = rfftn(u, s=fft_shape, workers=-1)
    F *= np.conj(F)
    return irfftn(F, s=fft_shape, workers=-1)


def _overlap_counts(shape_zyx: tuple[int, int, int], lag_xyz: tuple[int, int, int]) -> NDArray[np.float64]:
    """``prod_j (N_j - |h_j|)`` on the centred lag grid, for a region that is the whole box."""
    lx, ly, lz = lag_xyz
    nz, ny, nx = shape_zyx
    cz = nz - np.abs(np.arange(-lz, lz + 1))
    cy = ny - np.abs(np.arange(-ly, ly + 1))
    cx = nx - np.abs(np.arange(-lx, lx + 1))
    return cz[:, None, None].astype(np.float64) * cy[None, :, None] * cx[None, None, :]


def autocorrelation(
    vol: NDArray,
    spacing=1.0,
    max_lag=None,
    estimator: str = "overlap",
    mask: NDArray | None = None,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    dtype=np.float32,
) -> Autocorrelation:
    """Centred, normalised autocorrelation of ``vol`` (``(nz, ny, nx)``).

    Args:
        vol: the volume; finite values.
        spacing: voxel size ``(dx, dy, dz)`` or one number; only carried along for the profiles.
        max_lag: largest lag per axis ``(Lx, Ly, Lz)`` or one number; default ``N_j // 2``.
        estimator: ``"overlap"`` (default) or ``"window"``, see the module docstring.
        mask: optional boolean region; voxels outside it are excluded from every pair.
        min_overlap: lags whose overlap count is below this fraction of the region are NaN.
        dtype: floating type of the FFT (``float32`` halves the memory of a ``256^3`` analysis).
    """
    a = np.asarray(vol)
    if a.ndim != 3:
        raise ValueError(f"vol must be 3-D (nz, ny, nx), got shape {a.shape}")
    if a.size == 0:
        raise ValueError("vol is empty")
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    if not 0.0 < float(min_overlap) <= 1.0:
        raise ValueError(f"min_overlap must be in (0, 1], got {min_overlap!r}")
    spacing_xyz = _as_triple(spacing, "spacing")
    nz, ny, nx = a.shape
    if max_lag is None:
        lag_xyz = (nx // 2, ny // 2, nz // 2)
    else:
        lag_xyz = _as_triple(max_lag, "max_lag", int, minimum=0)  # a zero lag along an axis is allowed
        lag_xyz = tuple(min(int(L), n - 1) for L, n in zip(lag_xyz, (nx, ny, nz)))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            raise ValueError(f"mask shape {m.shape} does not match the volume shape {a.shape}")
        if not m.any():
            raise ValueError("mask is empty")
    else:
        m = None
    values = a[m] if m is not None else a.ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError("vol has non-finite values inside the region")
    n_voxels = int(values.size)
    mean = float(np.mean(values, dtype=np.float64))
    variance = float(np.var(values, dtype=np.float64))

    def _empty(status: str) -> Autocorrelation:
        acf = np.full(tuple(2 * L + 1 for L in lag_xyz[::-1]), np.nan, dtype=dtype)
        return Autocorrelation(acf, lag_xyz, estimator, spacing_xyz, variance, n_voxels, (nz, ny, nx), status, min_overlap)

    if n_voxels < 8 or variance <= NO_TEXTURE_RELATIVE_VARIANCE * max(1.0, mean * mean):
        return _empty("no_texture")

    u = (a.astype(np.float64) - mean).astype(dtype)
    if m is not None:
        u[~m] = 0
    # a linear correlation up to lag L needs at least N + L samples per axis before it wraps
    fft_shape = tuple(next_fast_len(n + L) for n, L in zip((nz, ny, nx), lag_xyz[::-1]))
    S = _centred_lags(_fft_autocorrelation(u, fft_shape), lag_xyz).astype(np.float64)
    del u
    if m is None:
        M = _overlap_counts((nz, ny, nx), lag_xyz)
    else:
        M = _centred_lags(_fft_autocorrelation(m.astype(dtype), fft_shape), lag_xyz).astype(np.float64)
        M = np.clip(np.rint(M), 0.0, None)
    c = tuple(L for L in lag_xyz[::-1])
    m0 = float(M[c])
    s0 = float(S[c])
    reliable = M >= min_overlap * m0
    with np.errstate(divide="ignore", invalid="ignore"):
        if estimator == "overlap":
            rho = (S / M) / (s0 / m0)
        else:
            rho = S / s0
    rho = np.where(reliable, rho, np.nan).astype(dtype)
    rho[c] = 1.0
    return Autocorrelation(rho, lag_xyz, estimator, spacing_xyz, s0 / m0, n_voxels, (nz, ny, nx), "ok", min_overlap)
