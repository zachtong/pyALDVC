"""Summary statistics of result fields, with the definitions of the DVC Challenge 2.0 paper.

Standard deviations are population ones (divided by ``n``), as the paper's noise floor (Tong et al., Eq. 2),
so the numbers reproduce its tables. Every function ignores non-finite values and returns NaN, never a
warning, for an empty input: an empty region or frame is a normal case in a statistics table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

STAT_NAMES = ("n", "mean", "std", "median", "robust_std", "p05", "p95", "min", "max", "rms")
CI_NAMES = ("n_eff", "ci95")  # present when the confidence interval of the mean was asked for
RHO_CUTOFF = 0.05  # the autocorrelation is summed out to the first shell where it falls below this
TAIL_Q = (0.5, 2.5)  # the exponent range of the fitted tail beyond it: stretched exponential to super-Gaussian
MAD_TO_STD = 1.4826  # median absolute deviation -> standard deviation of a normal distribution
_NAN = float("nan")


@dataclass(frozen=True)
class FieldStats:
    """Statistics of one scalar field over the nodes used."""

    n: int
    mean: float = _NAN
    std: float = _NAN  # population (ddof 0)
    median: float = _NAN
    robust_std: float = _NAN  # 1.4826 x median absolute deviation: blind to the few wild nodes that dominate a std
    p05: float = _NAN
    p95: float = _NAN
    min: float = _NAN
    max: float = _NAN
    rms: float = _NAN
    n_eff: float = _NAN  # effective number of independent nodes (correlated node values)
    ci95: float = _NAN  # half width of the 95 % confidence interval of the mean

    def as_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name in STAT_NAMES + CI_NAMES}


def summarize(values) -> FieldStats:
    """:class:`FieldStats` of the finite entries of ``values``."""
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return FieldStats(n=0)
    median = float(np.median(x))
    p05, p95 = np.percentile(x, (5.0, 95.0))
    return FieldStats(
        n=int(x.size),
        mean=float(x.mean()),
        std=float(x.std()),
        median=median,
        robust_std=MAD_TO_STD * float(np.median(np.abs(x - median))),
        p05=float(p05),
        p95=float(p95),
        min=float(x.min()),
        max=float(x.max()),
        rms=float(np.sqrt(np.mean(x * x))),
    )


@dataclass(frozen=True, eq=False)
class VectorStats:
    """Bias and noise floor of a displacement field against a nominal one (DVC Challenge 2.0, section 2.3).

    ``bias`` is the mean error per component (Eq. 1), ``noise`` the population standard deviation of the
    error after that mean -- the rigid translation -- is removed (Eq. 2), ``u_rms`` the root mean square of
    the error vector, ``sqrt(|bias|^2 + |noise|^2)`` (supplementary Eq. S10).
    """

    n: int
    bias: NDArray[np.float64]
    noise: NDArray[np.float64]
    u_rms: float

    def as_dict(self) -> dict:
        return {"n": self.n, "bias": self.bias.tolist(), "noise": self.noise.tolist(), "u_rms": self.u_rms}


def vector_stats(U, nominal=None) -> VectorStats:
    """:class:`VectorStats` of ``U`` ``(n, 3)`` against ``nominal`` (``(3,)`` or ``(n, 3)``; zero by default).
    Rows with a non-finite component are left out."""
    U = np.asarray(U, dtype=np.float64).reshape(-1, 3)
    err = U - (0.0 if nominal is None else np.asarray(nominal, dtype=np.float64))
    err = err[np.all(np.isfinite(err), axis=1)]
    if err.shape[0] == 0:
        nan3 = np.full(3, np.nan)
        return VectorStats(n=0, bias=nan3, noise=nan3.copy(), u_rms=_NAN)
    bias = err.mean(axis=0)
    noise = err.std(axis=0)
    return VectorStats(n=int(err.shape[0]), bias=bias, noise=noise, u_rms=float(np.sqrt(np.sum(bias**2) + np.sum(noise**2))))


def strain_precision(E6) -> tuple[float, float]:
    """``(MAER, SDER)`` of a zero-strain test (Liu & Morgan 2007): the mean and the standard deviation over the
    nodes of the average absolute strain component, ``e_k = (1/6) sum_c |eps_c,k|`` over the six tensor
    components (tensorial shear). Rows with a non-finite component are left out."""
    E = np.asarray(E6, dtype=np.float64).reshape(-1, 6)
    E = E[np.all(np.isfinite(E), axis=1)]
    if E.shape[0] == 0:
        return _NAN, _NAN
    e = np.abs(E).mean(axis=1)
    return float(e.mean()), float(e.std())


def _tail_model(shell_means: dict[int, float]):
    """``(L, q)`` of ``rho(r) = exp(-(r / L)^q)`` fitted to the positive shell means (shell ``r`` holds the lags of
    radius ``[r, r + 1)``); ``None`` with fewer than two -- no correlation was seen beyond the nearest lags, and
    extrapolating one point would invent a tail."""
    rs = np.array([r for r, v in shell_means.items() if v > 0.0], dtype=np.float64)
    if rs.size < 2:
        return None
    ys = np.array([shell_means[int(r)] for r in rs])
    rc = rs + 0.5
    from scipy.optimize import least_squares

    def residual(p):
        return -((rc / p[0]) ** p[1]) - np.log(ys)

    fit = least_squares(residual, x0=[2.0, 1.5], bounds=([0.1, TAIL_Q[0]], [1e4, TAIL_Q[1]]))
    return float(fit.x[0]), float(fit.x[1])


def effective_sample_size(grid_values, grid_mask, removed: int = 1) -> float:
    """How many independent values the masked nodes of a correlated field are worth, ``n^2 / sum_ij rho_ij``.

    Node values are correlated: subsets overlap, the global step and the strain fit smooth. The autocorrelation
    ``rho`` of the mean-removed field is estimated by FFT on the node grid (zero-padded, divided by the number of
    node pairs at each lag, which takes the region's own shape out) and summed out to the first shell of lags where
    its mean falls below :data:`RHO_CUTOFF`. Two corrections make the sum honest in 3-D, where many lags are far:

    * removing the region's mean pulls every estimated ``rho`` down by about ``1 / n_eff`` -- ``removed / n_eff``
      when ``removed`` smooth components were taken out (4 for a plane, see :func:`mean_confidence`); that is
      undone to first order, ``T = T_hat / (1 - removed (P - T_hat) / n^2)`` with ``P`` the node pairs summed over;
    * the lags beyond the cut still carry correlation (a quarter of the sum for Gaussian-smoothed noise): they are
      summed with ``rho = exp(-(r / L)^q)`` fitted to the shell means, ``q`` between 0.5 and 2.5.

    Monte Carlo on smoothed noise (``reports/statistics.pdf``): the 95 % interval covers the mean 91-94 % of the
    time when the region spans several correlation lengths, 85-88 % when it spans only two or three. The result
    lies in ``[1, n]``.
    """
    from scipy import fft

    x = np.asarray(grid_values, dtype=np.float64)
    m = np.asarray(grid_mask, dtype=bool) & np.isfinite(x)
    n = int(m.sum())
    if n < 2:
        return float(n)
    xc = np.where(m, x - x[m].mean(), 0.0)
    var = float(np.sum(xc * xc)) / n
    if var <= 0.0:
        return float(n)
    shape = tuple(2 * s for s in x.shape)
    fx = fft.rfftn(xc, s=shape, workers=-1)
    fm = fft.rfftn(m.astype(np.float64), s=shape, workers=-1)
    acov = fft.irfftn(fx * np.conj(fx), s=shape, workers=-1)
    pairs = np.rint(fft.irfftn(fm * np.conj(fm), s=shape, workers=-1))
    lag2 = [np.where(np.arange(s) < s // 2, np.arange(s), np.arange(s) - s).astype(np.float64) ** 2 for s in shape]
    radius = np.sqrt(lag2[0][:, None, None] + lag2[1][None, :, None] + lag2[2][None, None, :])  # broadcast, one array
    valid = pairs > 0.5
    rho = np.where(valid, acov / np.where(valid, pairs, 1.0) / var, 0.0)
    shell = np.floor(radius).astype(np.int64)
    reliable = valid & (pairs >= max(1.0, 0.25 * n))
    r_max = int(shell[reliable].max()) if reliable.any() else 0
    cut = r_max + 1
    means: dict[int, float] = {}
    for r in range(1, r_max + 1):
        sel = reliable & (shell == r)
        if not sel.any():
            cut = r
            break
        means[r] = float(rho[sel].mean())
        if means[r] < RHO_CUTOFF:
            cut = r
            break
    inside = valid & (radius < cut)
    t_hat = float(np.sum((pairs * rho)[inside]))
    if t_hat <= 0.0:
        return float(n)
    den = 1.0 - max(1, int(removed)) * (float(np.sum(pairs[inside])) - t_hat) / (float(n) * n)
    if den <= 0.0:  # the region is about one correlation length: one independent value
        return 1.0
    total = t_hat / den
    model = _tail_model(means)
    if model is not None:
        L, q = model
        beyond = valid & (radius >= cut)
        total += float(np.sum(pairs[beyond] * np.exp(-((radius[beyond] / L) ** q))))
    return float(np.clip(n * n / total, 1.0, n))


def detrended(grid_values, grid_mask) -> NDArray[np.float64]:
    """The masked values minus their least-squares plane ``a + b.(z, y, x)`` on the node grid (0 elsewhere)."""
    x = np.asarray(grid_values, dtype=np.float64)
    m = np.asarray(grid_mask, dtype=bool) & np.isfinite(x)
    idx = np.nonzero(m)
    A = np.column_stack([np.ones(idx[0].size), *(i.astype(np.float64) for i in idx)])
    coef, *_ = np.linalg.lstsq(A, x[m], rcond=None)
    out = np.zeros(x.shape)
    out[m] = x[m] - A @ coef
    return out


def mean_confidence(grid_values, grid_mask, level: float = 0.95) -> tuple[float, float]:
    """``(n_eff, half width)`` of the ``level`` confidence interval of the mean of the masked nodes.

    The interval is that of the measurement's fluctuations, not of the field's own variation across the region: a
    linear trend in space (a real gradient) is fitted and removed first, then ``n_eff`` and the std are those of
    the residual, and the half width is ``t(n_eff - 4) std / sqrt(n_eff - 4)`` (the population std of the residual
    of ``n_eff`` independent values after a four-parameter fit underestimates theirs by ``sqrt((n_eff - 4) / n_eff)``).
    A curved field leaves its curvature in the residual, which widens the interval: it errs on the safe side.
    """
    from scipy.stats import t

    x = np.asarray(grid_values, dtype=np.float64)
    m = np.asarray(grid_mask, dtype=bool) & np.isfinite(x)
    n = int(m.sum())
    if n < 2:
        return float(n), _NAN
    removed = 4 if n > 8 else 1  # a plane, or just the mean of a handful of nodes
    r = detrended(x, m) if removed == 4 else np.where(m, x - x[m].mean(), 0.0)
    n_eff = effective_sample_size(r, m, removed)
    dof = max(n_eff - removed, 1.0)
    q = float(t.ppf(0.5 + level / 2.0, dof))
    return n_eff, q * float(r[m].std()) / float(np.sqrt(dof))


def spatial_temporal_std(U_frames) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """The two noise-floor measures of the iDICs Good Practices Guide for several static frames ``(K, n, 3)``.

    Spatial: the standard deviation over the nodes, averaged over the frames. Temporal: the standard deviation
    over the frames at each node, averaged over the nodes (``None`` with a single frame). Only nodes finite in
    every frame count, so both measures use the same nodes.
    """
    U = np.asarray(U_frames, dtype=np.float64)
    if U.ndim != 3 or U.shape[2] != 3:
        raise ValueError(f"expected frames of shape (K, n, 3), got {U.shape}")
    keep = np.all(np.isfinite(U), axis=(0, 2))
    U = U[:, keep, :]
    if U.shape[1] == 0:
        return np.full(3, np.nan), None if U.shape[0] < 2 else np.full(3, np.nan)
    spatial = U.std(axis=1).mean(axis=0)
    temporal = U.std(axis=0).mean(axis=0) if U.shape[0] >= 2 else None
    return spatial, temporal
