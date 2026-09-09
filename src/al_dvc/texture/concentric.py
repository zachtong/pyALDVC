"""Concentric cubes about one centre point: the size sweep and the single region it settles on.

The user picks a centre and a schedule of cube edges; every cube is analysed on its own with the
overlap-corrected estimator of :mod:`~al_dvc.texture.acf`, so the answer of one size rests only on
the voxels of that size. That is what makes the sweep readable as a convergence study: a small cube
that borrowed grey values from outside would look representative before it is. The price is the
usual one of the corrected estimator -- the pair count falls with the lag -- which is why the lag is
kept to a quarter of the edge (:data:`LAG_FRACTION`) and lags below :data:`MIN_OVERLAP` of the pairs
are dropped.

A cube of edge ``e`` about ``c`` is ``[c - e // 2, c - e // 2 + e)``; it never leaves the bounds, so
an edge that would stick out is clipped per axis and the schedule ends when no axis grows any more.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .acf import DEFAULT_MIN_OVERLAP, autocorrelation
from .analysis import TextureResult, result_from_acf
from .boxes import Box, box_centre, box_size, normalise_box, whole_box
from .crossing import THRESHOLDS
from .rve import (
    DEFAULT_MIN_SPAN,
    DEFAULT_TOLERANCE_ABS,
    DEFAULT_TOLERANCE_REL,
    SizeLevel,
    SizeSweep,
    SubVolume,
    decide_plateau,
)

DEFAULT_START = 32  # edge of the smallest cube of the sweep
DEFAULT_STEP = 16  # growth of the edge from one size to the next
DEFAULT_COUNT = 8  # number of sizes asked for (fewer when the bounds run out)
LAG_FRACTION = 4  # largest lag analysed = edge / this
MIN_OVERLAP = DEFAULT_MIN_OVERLAP  # lags keeping fewer than this share of the cube's voxel pairs are not reported
ESTIMATOR = "overlap"  # B against B', divided by the number of pairs that actually contribute

__all__ = [
    "DEFAULT_COUNT",
    "DEFAULT_START",
    "DEFAULT_STEP",
    "ESTIMATOR",
    "LAG_FRACTION",
    "MIN_OVERLAP",
    "analyse_cube",
    "concentric_boxes",
    "concentric_sizes",
    "cube_box",
    "cube_limits",
    "max_lag_for",
    "sweep_concentric",
]


# ----------------------------------------------------------------------------- geometry
def max_lag_for(size) -> tuple[int, int, int]:
    """Largest lag ``(Lx, Ly, Lz)`` reported for a cube of ``size`` ``(nx, ny, nz)``."""
    return tuple(max(1, int(n) // LAG_FRACTION) for n in size)  # type: ignore[return-value]


def cube_limits(centre, bounds: Box) -> tuple[int, int, int]:
    """Largest edge ``(ex, ey, ez)`` per axis of a cube centred at ``centre`` that stays inside ``bounds``."""
    c = tuple(int(v) for v in centre)
    if len(c) != 3:
        raise ValueError(f"centre must be (x, y, z), got {centre!r}")
    if any(not (lo <= cj < hi) for cj, (lo, hi) in zip(c, bounds)):
        raise ValueError(f"the centre {c} is outside the region {tuple(bounds)}")
    return tuple(2 * min(cj - lo, hi - cj) for cj, (lo, hi) in zip(c, bounds))  # type: ignore[return-value]


def cube_box(centre, size) -> Box:
    """The box of the cube of ``size`` ``(nx, ny, nz)`` (or one edge) centred at ``centre``."""
    c = tuple(int(v) for v in centre)
    e = np.broadcast_to(np.asarray(size, dtype=int), (3,))
    return tuple((int(cj - ej // 2), int(cj - ej // 2 + ej)) for cj, ej in zip(c, e))  # type: ignore[return-value]


def concentric_sizes(
    centre, bounds: Box, start: int = DEFAULT_START, step: int = DEFAULT_STEP, count: int = DEFAULT_COUNT
) -> list[tuple[int, int, int]]:
    """Cube sizes ``(nx, ny, nz)``: ``start + i * step``, clipped per axis to what fits about ``centre``.

    The list ends at ``count`` sizes or when the clipping stops the growth, whichever comes first.
    """
    start, step, count = int(start), int(step), int(count)
    if start < 2 or step < 1 or count < 1:
        raise ValueError(f"start >= 2, step >= 1 and count >= 1 are required, got {start}, {step}, {count}")
    limits = cube_limits(centre, bounds)
    if min(limits) < 2:
        raise ValueError(f"no cube fits around the centre {tuple(int(v) for v in centre)}: it is on the edge of the region")
    sizes: list[tuple[int, int, int]] = []
    for i in range(count):
        s = tuple(int(min(start + i * step, limit)) for limit in limits)
        if sizes and s == sizes[-1]:
            break  # clipped on every axis: a larger nominal edge would analyse the same voxels
        sizes.append(s)  # type: ignore[arg-type]
    return sizes


def concentric_boxes(
    centre, bounds: Box, start: int = DEFAULT_START, step: int = DEFAULT_STEP, count: int = DEFAULT_COUNT
) -> list[Box]:
    """The boxes of :func:`concentric_sizes`, ready to draw or to analyse."""
    return [cube_box(centre, size) for size in concentric_sizes(centre, bounds, start, step, count)]


# ----------------------------------------------------------------------------- one cube
def analyse_cube(
    vol: NDArray,
    box,
    spacing=1.0,
    mask: NDArray | None = None,
    thresholds=THRESHOLDS,
    max_lag=None,
    min_overlap: float = MIN_OVERLAP,
    radial_bin: float | None = None,
) -> TextureResult:
    """Profiles, correlation lengths, noise floor and periodicity of one box of ``vol``.

    ``mask`` (the analysis region, over the whole volume) only matters where it cuts into the box:
    the voxels it excludes take no part in any pair and the overlap correction counts the pairs that
    remain. ``settings["fill"]`` reports how much of the box survived.
    """
    a = np.asarray(vol)
    if a.ndim != 3:
        raise ValueError(f"vol must be 3-D (nz, ny, nx), got shape {a.shape}")
    box = normalise_box(box, a.shape)
    (x0, x1), (y0, y1), (z0, z1) = box
    sub = a[z0:z1, y0:y1, x0:x1]
    sub_mask = None
    fill = 1.0
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            raise ValueError(f"mask shape {m.shape} does not match the volume shape {a.shape}")
        m = m[z0:z1, y0:y1, x0:x1]
        fill = float(m.mean())
        if not m.any():
            raise ValueError("the analysed box holds no voxel of the region")
        if not m.all():
            sub_mask = m
    size = box_size(box)
    lag = max_lag_for(size) if max_lag is None else max_lag
    ac = autocorrelation(sub, spacing, lag, ESTIMATOR, sub_mask, min_overlap)
    settings = {
        "estimator": ESTIMATOR,
        "box": tuple(tuple(int(v) for v in pair) for pair in box),
        "centre": box_centre(box),
        "size": size,
        "max_lag": ac.max_lag,
        "min_overlap": float(min_overlap),
        "fill": fill,
        "thresholds": tuple(float(t) for t in thresholds),
        "n_voxels": ac.n_voxels,
    }
    window = tuple(slice(int(lo), int(hi)) for lo, hi in box[::-1])  # (z, y, x)
    return result_from_acf(ac, thresholds, radial_bin, window, settings)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- the sweep
def sweep_concentric(
    vol: NDArray,
    centre,
    bounds=None,
    start: int = DEFAULT_START,
    step: int = DEFAULT_STEP,
    count: int = DEFAULT_COUNT,
    spacing=1.0,
    mask: NDArray | None = None,
    thresholds=THRESHOLDS,
    axis: str = "radial",
    tolerance_rel: float = DEFAULT_TOLERANCE_REL,
    tolerance_abs: float = DEFAULT_TOLERANCE_ABS,
    min_span: float = DEFAULT_MIN_SPAN,
    progress: Callable[[float, str], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> SizeSweep:
    """Correlation lengths against the edge of concentric cubes about ``centre``.

    One cube per size, each analysed on its own data alone, so the size from which the lengths stop
    moving is the size the texture actually needs. The spread across positions is not measured (one
    cube per size), so the plateau test rests on the size trend alone.
    """
    a = np.asarray(vol)
    bounds = whole_box(a.shape) if bounds is None else normalise_box(bounds, a.shape)
    sizes = concentric_sizes(centre, bounds, start, step, count)
    levels: list[SizeLevel] = []
    for i, size in enumerate(sizes):
        if stop is not None and stop():
            break
        if progress is not None:
            progress(i / max(1, len(sizes)), " x ".join(str(v) for v in size))
        box = cube_box(centre, size)
        res = analyse_cube(a, box, spacing, mask, thresholds)
        (x0, x1), (y0, y1), (z0, z1) = box
        sub = SubVolume((z0, z1, y0, y1, x0, x1), {float(t): res.length(axis, t) for t in thresholds})
        mean = {float(t): (float(v) if (v := sub.lengths[float(t)]) is not None else float("nan")) for t in thresholds}
        std = {float(t): float("nan") for t in thresholds}
        n_valid = {float(t): int(sub.lengths[float(t)] is not None) for t in thresholds}
        levels.append(SizeLevel(size, [sub], mean, std, n_valid, res.profiles.get(axis)))
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
        "estimator": ESTIMATOR,
        "centre": tuple(int(v) for v in centre),
        "bounds": tuple(tuple(int(v) for v in pair) for pair in bounds),
        "start": int(start),
        "step": int(step),
        "count": int(count),
        "min_overlap": MIN_OVERLAP,
        "tolerance_rel": tolerance_rel,
        "tolerance_abs": tolerance_abs,
        "min_span": min_span,
    }
    return SizeSweep(levels, decisions, axis, settings)
