"""Subset and step suggestions from the correlation lengths: a starting point, not a rule.

The subset along an axis is set to ``factor`` times the 1/e correlation length along that
axis (2.5 by default), so it spans a few independent features; the step is a fraction of the
subset. A periodic texture pushes the subset above one period, a high noise floor is reported.
The report validates the factor on synthetic pairs; the user sees the lengths the suggestion
came from and can change every number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .acf import AXES
from .analysis import TextureResult
from .crossing import THRESHOLDS

DEFAULT_FACTOR = 2.5
MIN_EDGE = 8
MAX_EDGE = 128
STEP_FRACTION = 0.5
PERIOD_FACTOR = 1.5  # a periodic texture needs more than one period in the subset
NOISE_FLOOR_WARNING = 0.1


@dataclass(frozen=True)
class Recommendation:
    subset: tuple[int, int, int]  # even edges (x, y, z) in voxels, the solver's winsize
    step: tuple[int, int, int]
    basis: dict[str, float | None]  # the 1/e length per axis the subset came from (voxels)
    factor: float
    notes: list[str] = field(default_factory=list)


def _even(value: float, lo: int = MIN_EDGE, hi: int = MAX_EDGE) -> int:
    n = int(np.ceil(value))
    n += n % 2
    return int(min(max(n, lo), hi))


def recommend_parameters(
    result: TextureResult,
    factor: float = DEFAULT_FACTOR,
    step_fraction: float = STEP_FRACTION,
    min_edge: int = MIN_EDGE,
    max_edge: int = MAX_EDGE,
) -> Recommendation:
    """Subset and step per axis from the 1/e lengths of ``result`` (radial length where an axis has none)."""
    if factor <= 0 or not 0 < step_fraction <= 1:
        raise ValueError("factor must be positive and step_fraction in (0, 1]")
    notes: list[str] = []
    t0 = float(THRESHOLDS[0])
    radial = result.length("radial", t0)
    basis: dict[str, float | None] = {}
    edges = []
    for axis in AXES:
        L = result.length(axis, t0)
        if L is None:
            L = radial
            if L is not None:
                notes.append(f"no 1/e crossing along {axis}; the radial length is used")
        basis[axis] = L
        if L is None:
            notes.append(f"no correlation length along {axis}; the default edge is kept")
            edges.append(_even(min_edge * 2, min_edge, max_edge))
        else:
            edges.append(_even(factor * L, min_edge, max_edge))
    if result.periodicity is not None:
        axis, period, height = result.periodicity
        notes.append(f"periodic texture along {axis}: period {period:.1f} voxels, secondary peak {height:.2f}")
        if axis in AXES:
            i = AXES.index(axis)
            edges[i] = max(edges[i], _even(PERIOD_FACTOR * period, min_edge, max_edge))
        else:
            edges = [max(e, _even(PERIOD_FACTOR * period, min_edge, max_edge)) for e in edges]
    if np.isfinite(result.noise_floor) and result.noise_floor > NOISE_FLOOR_WARNING:
        notes.append(f"noise floor {result.noise_floor:.2f}: little texture contrast, expect a noisy correlation")
    steps = tuple(max(2, _even(step_fraction * e, 2, max_edge)) for e in edges)
    return Recommendation(tuple(edges), steps, basis, float(factor), notes)
