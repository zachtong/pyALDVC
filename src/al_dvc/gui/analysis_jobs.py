"""What the Statistics tab computes off the UI thread, and the thread that runs it.

:func:`compute_current` is the job of the frame on screen (statistics with their confidence intervals,
histograms, the homogeneous fit, the comparison of the regions, the profile, the line and the canvas values);
the ``*_job`` functions are the long jobs run on request (series over frames, noise floor, profiles over frames,
the line over frames with its end points and the extensometer). Every job works on the result object and
immutable controls it was handed, never on widgets.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, replace

import numpy as np
from PySide6.QtCore import QThread, Signal

from al_dvc.analysis import (
    Extensometer,
    FieldStats,
    NodeFilter,
    Region,
    axis_profile,
    extensometer,
    frame_series,
    frame_view,
    homogeneous,
    mean_confidence,
    noise_floor,
    sample_line,
    select_nodes,
    summarize,
)

logger = logging.getLogger(__name__)

NODES_USED = "nodes_used"  # the canvas shows the node selection instead of a field
MAX_BINS = 80


class AnalysisCancelled(Exception):
    pass


class Worker(QThread):
    """Runs ``fn(stop, progress)`` off the UI thread; the payload goes back with the key it was made for."""

    done = Signal(object, object)  # (key, payload)
    failed = Signal(object, str)
    progress = Signal(float)

    def __init__(self, key: tuple, fn, parent=None) -> None:
        super().__init__(parent)
        self.key = key
        self._fn = fn
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            payload = self._fn(lambda: self._stop, self.progress.emit)
        except AnalysisCancelled:
            self.failed.emit(self.key, "")
            return
        except Exception as exc:  # surface to the UI
            logger.error("statistics failed:\n%s", traceback.format_exc())
            self.failed.emit(self.key, f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(self.key, payload)


def same_source(a: tuple | None, b: tuple | None) -> bool:
    """Two job keys describe the same computation: the same result object (by identity -- comparing two
    results field by field would compare their arrays) and equal controls."""
    if a is None or b is None or len(a) != len(b):
        return False
    return a[0] is b[0] and a[1:] == b[1:]


def histogram(values: np.ndarray):
    """``(counts, edges)`` of the finite values (Freedman-Diaconis bins, at most :data:`MAX_BINS`), ``None`` if none."""
    x = values[np.isfinite(values)]
    if x.size == 0:
        return None
    lo, hi = float(x.min()), float(x.max())
    if not hi > lo:  # a constant field (a pure translation, a zero component): bins around the value
        pad = max(abs(lo), 1.0) * 1e-6
        lo, hi = lo - pad, hi + pad
    try:
        edges = np.histogram_bin_edges(x, bins="fd", range=(lo, hi))
    except ValueError:  # a range too narrow for the data's float spacing
        edges = None
    if edges is None or edges.size - 1 > MAX_BINS or edges.size < 3:
        edges = np.linspace(lo, hi, min(MAX_BINS, max(10, int(np.sqrt(x.size)))) + 1)
    counts, edges = np.histogram(x, bins=edges)
    return counts, edges


def stats_with_ci(result, values: np.ndarray, mask: np.ndarray) -> FieldStats:
    """:func:`summarize` of ``values[mask]`` with the 95 % confidence interval of the mean (correlated nodes)."""
    st = summarize(values[mask])
    if st.n >= 2:
        mesh = result.dvc_mesh
        n_eff, half = mean_confidence(mesh.to_grid(values), mesh.to_grid(mask))
        st = replace(st, n_eff=n_eff, ci95=half)
    return st


def _mask(region: Region | None, result):
    return None if region is None else region.node_mask(result)


@dataclass(frozen=True, eq=False)
class CurrentFrame:
    """What the current frame's job produced."""

    key: tuple
    selection: object
    fit: object
    stats: dict
    histograms: dict  # field -> (counts, edges)
    homogeneous: object
    canvas: np.ndarray | None
    compare_field: str = ""
    region_rows: tuple = ()  # (Region | None, FieldStats) of the compared field: every node kept, then each region
    profile: object = None  # AxisProfile of the compared field
    line: tuple | None = None  # (distance, values) of the compared field along the line
    error: str = ""


def compute_current(
    result,
    frame: int,
    fields,
    node_filter: NodeFilter,
    motion: str,
    shown: str,
    key: tuple,
    stats_region: Region | None = None,
    fit_region: Region | None = None,
    regions=(),
    profile_axis: str = "z",
    line=None,
) -> CurrentFrame:
    """The current frame's statistics, histograms, homogeneous fit, region comparison, profile, line and canvas
    values (worker thread). The motion is fitted over ``fit_region``; the statistics are taken over
    ``stats_region``; the canvas shows every node the filter keeps."""
    fit_mask = _mask(fit_region, result)
    region_mask = _mask(stats_region, result)
    try:
        view = frame_view(result, frame, motion, node_filter, None, fit_mask)
    except ValueError as exc:  # too few nodes for the motion fit
        return CurrentFrame(key, None, None, {}, {}, None, None, error=str(exc))
    kept = view.selection.mask
    selection = view.selection if region_mask is None else select_nodes(result, frame, node_filter, region_mask)
    mask = selection.mask
    values = {f: view.values(f) for f in fields}
    stats = {f: stats_with_ci(result, v, mask) for f, v in values.items()}
    hists = {f: histogram(v[mask]) for f, v in values.items()}
    try:
        hom = homogeneous(result, frame, node_filter, region=region_mask)
    except ValueError:
        hom = None
    compare = shown if shown != NODES_USED else fields[0]
    cmp_values = values[compare] if compare in values else view.values(compare)
    rows = [(None, stats_with_ci(result, cmp_values, kept))]
    for region in regions:
        rows.append((region, stats_with_ci(result, cmp_values, kept & region.node_mask(result))))
    profile = axis_profile(view, compare, profile_axis, region=region_mask)
    line_data = None if line is None else sample_line(view, compare, line[0], line[1])
    if shown == NODES_USED:
        canvas = np.where(mask, 1.0, 0.0)
        valid = np.asarray(result.dvc_mesh.node_valid, dtype=bool)
        if valid.size == canvas.size:
            canvas[~valid] = np.nan
    else:
        canvas = np.where(kept, cmp_values, np.nan)
    return CurrentFrame(
        key,
        selection,
        view.fit,
        stats,
        hists,
        hom,
        canvas,
        compare_field=compare,
        region_rows=tuple(rows),
        profile=profile,
        line=line_data,
    )


def _check(stop) -> None:
    if stop():
        raise AnalysisCancelled()


def series_job(
    result, fields, node_filter, motion, stats_region, fit_region, compare: bool, regions, min_share: float, stop, progress
):
    """``[(Region | None, [FrameStats])]``: the series of ``stats_region``, or of every node and each region; a frame
    using less than ``min_share`` of a region's nodes is flagged (a gap)."""
    targets = [None, *regions] if compare else [stats_region]
    fit_mask = _mask(fit_region, result)
    n = max(1, len(result.result_disp))
    out = []
    for i, region in enumerate(targets):
        series = frame_series(
            result,
            fields,
            node_filter,
            motion,
            region=_mask(region, result),
            min_valid_fraction=min_share,
            stop=stop,
            progress=lambda k, _n, i=i: progress((i * n + k + 1) / (len(targets) * n)),
            fit_region=fit_mask,
            with_ci=True,
        )
        _check(stop)
        out.append((region, series))
    return out


def noise_job(result, frame, node_filter, nominal, stats_region, stop, progress):
    return noise_floor(result, frame, node_filter, nominal=nominal, region=_mask(stats_region, result))


def profiles_job(result, field, node_filter, motion, fit_region, stats_region, axis, stop, progress):
    """``[(frame, AxisProfile)]`` over every frame; a frame whose motion cannot be fitted is left out."""
    fit_mask = _mask(fit_region, result)
    region_mask = _mask(stats_region, result)
    n = len(result.result_disp)
    out = []
    for k in range(n):
        _check(stop)
        try:
            view = frame_view(result, k, motion, node_filter, None, fit_mask)
        except ValueError:
            continue
        out.append((k, axis_profile(view, field, axis, region=region_mask)))
        progress((k + 1) / max(1, n))
    return out


@dataclass(frozen=True, eq=False)
class LineSeries:
    """A line over every frame: ``field`` along it (``values`` ``(frames, samples)``, NaN where a frame could not be
    corrected), at its two ends (``ends`` ``(frames, 2)``), and the extensometer between the ends."""

    field: str
    frames: np.ndarray
    distance: np.ndarray
    values: np.ndarray
    extensometer: Extensometer

    @property
    def ends(self) -> np.ndarray:
        return self.values[:, [0, -1]]

    def as_dict(self) -> dict:
        return {
            "field": self.field,
            "frames": (self.frames + 1).tolist(),
            "distance": self.distance.tolist(),
            "values": self.values.tolist(),
            "extensometer": self.extensometer.as_dict(),
        }


def line_job(result, p0, p1, field, node_filter, motion, fit_region, stop, progress) -> LineSeries:
    """``field`` along the segment in every frame (the motion removed as on screen) and the extensometer between
    its ends (a length: nothing to remove)."""
    fit_mask = _mask(fit_region, result)
    n = len(result.result_disp)
    rows = []
    distance = None
    for k in range(n):
        _check(stop)
        try:
            view = frame_view(result, k, motion, node_filter, None, fit_mask)
        except ValueError:
            rows.append(None)
            continue
        distance, values = sample_line(view, field, p0, p1)
        rows.append(values)
        progress((k + 1) / max(1, n + 1))
    if distance is None:
        raise ValueError("the motion could not be fitted in any frame")
    values = np.vstack([np.full(distance.size, np.nan) if r is None else r for r in rows])
    ext = extensometer(result, p0, p1, node_filter=node_filter)
    progress(1.0)
    return LineSeries(field=field, frames=np.arange(n), distance=distance, values=values, extensometer=ext)
