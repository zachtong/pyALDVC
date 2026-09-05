"""Where a correlation profile falls through a threshold: one definition for every use.

The correlation length at threshold ``t`` is the distance of the first strict downward crossing:
the first pair of neighbouring samples with ``y[i] >= t > y[i + 1]``, linearly interpolated. The
samples keep their sign (a negative lobe is information, not an error), the search stops at the
first non-finite sample (the reliable range of the estimate), and nothing is extrapolated: a
curve that never drops through ``t`` reports ``not_crossed`` instead of a number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATUSES = ("crossed", "plateau", "not_crossed", "invalid")
THRESHOLDS = (float(np.exp(-1.0)), 0.1, 0.01)  # feature scale, mesoscale, tail (diagnostic)
THRESHOLD_LABELS = {THRESHOLDS[0]: "1/e", 0.1: "0.1", 0.01: "0.01"}


@dataclass(frozen=True)
class Crossing:
    """The result of one threshold search.

    ``value`` is in the units of the ``x`` passed in (voxels or physical), ``None`` unless the
    status is ``crossed`` or ``plateau``; ``index`` is the sample before the crossing; ``reason``
    explains a ``not_crossed`` / ``invalid`` status.
    """

    threshold: float
    value: float | None
    status: str
    index: int | None = None
    reason: str = ""

    @property
    def found(self) -> bool:
        return self.value is not None


def correlation_length(x, y, threshold: float) -> Crossing:
    """First strict downward crossing of ``y(x)`` through ``threshold``."""
    t = float(threshold)
    if not 0.0 < t < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold!r}")
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    if xs.size != ys.size:
        raise ValueError("x and y must have the same length")
    finite = np.isfinite(ys) & np.isfinite(xs)
    n = int(np.argmin(finite)) if not finite.all() else xs.size  # samples before the first gap
    if n < 2:
        return Crossing(t, None, "invalid", None, "fewer than two reliable samples")
    if ys[0] < t:
        return Crossing(t, None, "invalid", None, "the profile starts below the threshold")
    for i in range(n - 1):
        if ys[i] >= t > ys[i + 1]:
            if ys[i] == t:
                status = "plateau" if i > 0 and ys[i - 1] == t else "crossed"
                return Crossing(t, float(xs[i]), status, i)
            frac = (ys[i] - t) / (ys[i] - ys[i + 1])
            return Crossing(t, float(xs[i] + frac * (xs[i + 1] - xs[i])), "crossed", i)
    reason = "the reliable range ends above the threshold" if n < xs.size else "the profile stays above the threshold"
    return Crossing(t, None, "not_crossed", None, reason)


def lengths(profile, thresholds=THRESHOLDS, physical: bool = False) -> dict[float, Crossing]:
    """Crossings of a :class:`~al_dvc.texture.profiles.Profile` at every threshold, in voxels or physical units."""
    x = profile.distance if physical else profile.lag
    return {float(t): correlation_length(x, profile.mean, t) for t in thresholds}
