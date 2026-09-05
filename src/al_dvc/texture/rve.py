"""Size sweep: is the analysed region large enough for the correlation lengths to be stable?

For every sub-volume size several sub-volumes are analysed at different positions (tiles of
the region first, random positions when the tiles run out), so the spread across positions is
measured rather than assumed. A threshold length is *converged* from the first size whose mean
stays within a tolerance band of the reference (the mean over the largest sizes) for every
larger size, provided the sizes from there to the largest span a minimum ratio and the spread
across positions at that size is inside the band as well. Sub-volumes that the region clips to
identical bounds are analysed once; the schedule stops when the sizes stop growing.

This measures the stability of an image statistic (the correlation length of the grey values)
against the sampled volume. It is not a material RVE: that needs the property of interest and
its own convergence study.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .acf import AXES, DEFAULT_MIN_OVERLAP
from .analysis import analyse_texture, analysis_window
from .crossing import THRESHOLDS

DEFAULT_SAMPLES = 4
DEFAULT_TOLERANCE_REL = 0.05
DEFAULT_TOLERANCE_ABS = 0.25  # voxels
DEFAULT_MIN_SPAN = 1.5  # largest size / plateau start size
DEFAULT_REFERENCE_SIZES = 2  # the reference is the mean over this many of the largest sizes


@dataclass(frozen=True)
class SubVolume:
    """One analysed sub-volume: its bounds ``(z0, z1, y0, y1, x0, x1)`` and its lengths per threshold."""

    bounds: tuple[int, int, int, int, int, int]
    lengths: dict[float, float | None]

    @property
    def size(self) -> tuple[int, int, int]:
        z0, z1, y0, y1, x0, x1 = self.bounds
        return x1 - x0, y1 - y0, z1 - z0


@dataclass(frozen=True)
class SizeLevel:
    """All sub-volumes of one nominal size, with the mean and spread of every threshold length."""

    size: tuple[int, int, int]  # (nx, ny, nz) actually analysed
    samples: list[SubVolume]
    mean: dict[float, float]  # NaN when no sample crossed
    std: dict[float, float]
    n_valid: dict[float, int]

    @property
    def effective(self) -> float:
        """Geometric mean edge (voxels)."""
        return float(np.prod([float(s) for s in self.size]) ** (1.0 / 3.0))


@dataclass(frozen=True)
class PlateauDecision:
    """Whether and where one threshold length settled, with the numbers the decision rests on."""

    threshold: float
    converged: bool
    start_index: int | None  # index into the sweep's levels
    reference: float
    tolerance: float  # the band half-width applied, in voxels
    deviations: NDArray[np.float64]  # |mean_i - reference| per level
    spreads: NDArray[np.float64]  # std across positions per level
    reason: str = ""


@dataclass(frozen=True)
class SizeSweep:
    levels: list[SizeLevel]
    decisions: dict[float, PlateauDecision]
    axis: str
    settings: dict = field(default_factory=dict)

    @property
    def sizes(self) -> NDArray[np.float64]:
        return np.array([lvl.effective for lvl in self.levels], dtype=np.float64)

    def means(self, threshold: float) -> NDArray[np.float64]:
        return np.array([lvl.mean[float(threshold)] for lvl in self.levels], dtype=np.float64)

    def stds(self, threshold: float) -> NDArray[np.float64]:
        return np.array([lvl.std[float(threshold)] for lvl in self.levels], dtype=np.float64)


def size_schedule(box: tuple[int, int, int], start=16, step=16, count: int = 8) -> list[tuple[int, int, int]]:
    """Sub-volume sizes ``(nx, ny, nz)``: ``start + i * step`` per axis, clipped to ``box`` (``(nx, ny, nz)``),
    without repeats, ending when the clipped size stops growing."""
    start_xyz = np.broadcast_to(np.asarray(start, dtype=int), (3,))
    step_xyz = np.broadcast_to(np.asarray(step, dtype=int), (3,))
    box_xyz = np.asarray(box, dtype=int)
    if count < 1 or np.any(start_xyz < 2) or np.any(step_xyz < 0) or np.all(step_xyz == 0) or np.any(box_xyz < 2):
        raise ValueError("size_schedule needs count >= 1, start >= 2, step >= 0 with one axis growing, box >= 2")
    sizes: list[tuple[int, int, int]] = []
    for i in range(int(count)):
        s = tuple(int(v) for v in np.minimum(start_xyz + i * step_xyz, box_xyz))
        if sizes and s == sizes[-1]:
            break  # clipped to the box: no new size evidence beyond this point
        sizes.append(s)
    return sizes


def sample_positions(
    box: tuple[int, int, int], size: tuple[int, int, int], n_samples: int, rng: np.random.Generator
) -> list[tuple[int, int, int]]:
    """Origins ``(z0, y0, x0)`` of ``n_samples`` sub-volumes of ``size`` (``(nx, ny, nz)``) inside ``box``.

    Tiles come first (non-overlapping, spread over the box); when they run out, random origins
    are added, so the spread across positions never rests on one sub-volume unless the size is
    the whole box.
    """
    bx, by, bz = (int(v) for v in box)
    sx, sy, sz = (int(v) for v in size)
    if sx > bx or sy > by or sz > bz:
        raise ValueError(f"size {size} does not fit in the box {box}")
    n_samples = max(1, int(n_samples))
    tiles = [(bz // sz, by // sy, bx // sx)]
    nt = int(np.prod(tiles[0]))
    origins: list[tuple[int, int, int]] = []
    if nt >= 1:
        all_tiles = [
            (iz * sz, iy * sy, ix * sx) for iz in range(tiles[0][0]) for iy in range(tiles[0][1]) for ix in range(tiles[0][2])
        ]
        if nt <= n_samples:
            origins = all_tiles
        else:
            pick = np.unique(np.linspace(0, nt - 1, n_samples).round().astype(int))
            origins = [all_tiles[i] for i in pick]
    while len(origins) < n_samples:
        origins.append((int(rng.integers(0, bz - sz + 1)), int(rng.integers(0, by - sy + 1)), int(rng.integers(0, bx - sx + 1))))
    return origins


def decide_plateau(
    sizes: NDArray,
    means: NDArray,
    spreads: NDArray,
    threshold: float,
    tolerance_rel: float = DEFAULT_TOLERANCE_REL,
    tolerance_abs: float = DEFAULT_TOLERANCE_ABS,
    min_span: float = DEFAULT_MIN_SPAN,
    reference_sizes: int = DEFAULT_REFERENCE_SIZES,
) -> PlateauDecision:
    """The plateau test on one threshold's means (NaN where a size gave no crossing)."""
    sizes = np.asarray(sizes, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    spreads = np.asarray(spreads, dtype=np.float64)
    n = sizes.size
    finite = np.isfinite(means)
    if finite.sum() < 2:
        return PlateauDecision(
            threshold,
            False,
            None,
            float("nan"),
            float("nan"),
            np.full(n, np.nan),
            spreads,
            "fewer than two sizes crossed the threshold",
        )
    ref_idx = np.flatnonzero(finite)[-max(1, int(reference_sizes)) :]
    reference = float(np.mean(means[ref_idx]))
    tol = float(tolerance_abs + tolerance_rel * abs(reference))
    dev = np.abs(means - reference)
    reason = "no size stays within the band up to the largest size"
    for i in range(n):
        if not finite[i]:
            continue
        later = np.arange(i, n)
        if not finite[later].all():
            reason = "a larger size gave no crossing"
            continue
        if np.any(dev[later] > tol):
            continue
        if sizes[-1] / sizes[i] < min_span:
            reason = f"the sizes from {sizes[i]:.0f} to {sizes[-1]:.0f} span less than {min_span:g}x"
            break
        if np.isfinite(spreads[i]) and spreads[i] > tol:
            reason = "the spread across positions exceeds the band"
            continue
        return PlateauDecision(threshold, True, int(i), reference, tol, dev, spreads, "")
    return PlateauDecision(threshold, False, None, reference, tol, dev, spreads, reason)


def sweep_sizes(
    vol: NDArray,
    mask: NDArray | None = None,
    sizes=None,
    samples_per_size: int = DEFAULT_SAMPLES,
    thresholds=THRESHOLDS,
    axis: str = "radial",
    spacing=1.0,
    estimator: str = "overlap",
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    tolerance_rel: float = DEFAULT_TOLERANCE_REL,
    tolerance_abs: float = DEFAULT_TOLERANCE_ABS,
    min_span: float = DEFAULT_MIN_SPAN,
    seed: int = 0,
    progress: Callable[[float, str], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> SizeSweep:
    """Correlation lengths against sub-volume size inside the region of ``vol``."""
    if axis not in (*AXES, "radial"):
        raise ValueError(f"axis must be x, y, z or radial, got {axis!r}")
    a = np.asarray(vol)
    window = analysis_window(a.shape, mask, max_voxels=a.size)  # the region's box, uncropped
    region = a[window]
    region_mask = np.asarray(mask, dtype=bool)[window] if mask is not None else None
    bz, by, bx = region.shape
    if sizes is None:
        sizes = size_schedule((bx, by, bz), start=16, step=max(8, min(bx, by, bz) // 8), count=8)
    rng = np.random.default_rng(seed)
    levels: list[SizeLevel] = []
    seen: set[tuple[int, ...]] = set()
    n_total = sum(samples_per_size for _ in sizes)
    done = 0
    for size in sizes:
        sx, sy, sz = (int(min(v, b)) for v, b in zip(size, (bx, by, bz)))
        samples: list[SubVolume] = []
        for z0, y0, x0 in sample_positions((bx, by, bz), (sx, sy, sz), samples_per_size, rng):
            if stop is not None and stop():
                break
            bounds = (z0, z0 + sz, y0, y0 + sy, x0, x0 + sx)
            done += 1
            if progress is not None:
                progress(done / max(1, n_total), f"{sx} x {sy} x {sz}")
            if bounds in seen:
                continue  # the same voxels again would only repeat the previous number
            seen.add(bounds)
            sub = region[bounds[0] : bounds[1], bounds[2] : bounds[3], bounds[4] : bounds[5]]
            sub_mask = (
                region_mask[bounds[0] : bounds[1], bounds[2] : bounds[3], bounds[4] : bounds[5]]
                if region_mask is not None
                else None
            )
            if sub_mask is not None and not sub_mask.any():
                continue
            res = analyse_texture(sub, spacing, sub_mask, None, estimator, thresholds, min_overlap, max_voxels=sub.size)
            samples.append(SubVolume(bounds, {float(t): res.length(axis, t) for t in thresholds}))
        if not samples:
            continue
        mean, std, n_valid = {}, {}, {}
        for t in thresholds:
            vals = np.array([s.lengths[float(t)] for s in samples if s.lengths[float(t)] is not None], dtype=np.float64)
            n_valid[float(t)] = int(vals.size)
            mean[float(t)] = float(vals.mean()) if vals.size else float("nan")
            std[float(t)] = float(vals.std()) if vals.size > 1 else (0.0 if vals.size == 1 else float("nan"))
        levels.append(SizeLevel((sx, sy, sz), samples, mean, std, n_valid))
        if stop is not None and stop():
            break
    sweep_sizes_arr = np.array([lvl.effective for lvl in levels], dtype=np.float64)
    decisions = {
        float(t): decide_plateau(
            sweep_sizes_arr,
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
        "samples_per_size": samples_per_size,
        "estimator": estimator,
        "min_overlap": min_overlap,
        "tolerance_rel": tolerance_rel,
        "tolerance_abs": tolerance_abs,
        "min_span": min_span,
        "seed": seed,
        "region": tuple((s.start, s.stop) for s in window),
    }
    return SizeSweep(levels, decisions, axis, settings)
