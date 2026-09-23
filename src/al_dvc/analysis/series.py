"""Statistics of one frame, and of a sequence frame by frame."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import PipelineResult
from .motion import MotionFit
from .selection import NodeFilter, Selection
from .stats import FieldStats, mean_confidence, summarize
from .view import frame_view


@dataclass(frozen=True, eq=False)
class FrameStats:
    """The statistics of a set of fields for one frame, with the nodes and the motion fit they stand on.

    ``flag`` is ``"few_nodes"`` when the share of the region's nodes used fell below the threshold the
    statistics were asked with (a mean of a handful of nodes is not the mean of the region), ``"no_fit"``
    when too few nodes were left to fit the motion to remove.
    """

    frame: int
    selection: Selection
    fit: MotionFit | None
    stats: dict[str, FieldStats]
    flag: str = "ok"

    def as_dict(self) -> dict:
        return {
            "frame": self.frame + 1,
            "flag": self.flag,
            **self.selection.as_dict(),
            "motion": None if self.fit is None else self.fit.as_dict(),
            "stats": {name: s.as_dict() for name, s in self.stats.items()},
        }


def frame_stats(
    result: PipelineResult,
    frame: int,
    fields: Sequence[str],
    node_filter: NodeFilter | None = None,
    motion: str = "none",
    region: NDArray[np.bool_] | None = None,
    min_valid_fraction: float = 0.0,
    fit_region: NDArray[np.bool_] | None = None,
    with_ci: bool = False,
) -> FrameStats:
    """:class:`FrameStats` of ``fields`` for frame ``frame``; flagged ``"no_fit"`` (and empty) when too few nodes
    are left to fit ``motion``."""
    try:
        view = frame_view(result, frame, motion, node_filter, region, fit_region)
    except ValueError:  # too few nodes for the motion fit: say so rather than report an uncorrected field
        selection = frame_view(result, frame, "none", node_filter, region).selection
        return FrameStats(
            frame=frame, selection=selection, fit=None, stats={name: FieldStats(n=0) for name in fields}, flag="no_fit"
        )
    mask = view.selection.mask
    stats = {}
    for name in fields:
        values = view.values(name)
        st = summarize(values[mask])
        if with_ci and st.n >= 2:  # the nodes are correlated: the interval of the mean needs their effective number
            mesh = result.dvc_mesh
            n_eff, half = mean_confidence(mesh.to_grid(values), mesh.to_grid(mask))
            st = replace(st, n_eff=n_eff, ci95=half)
        stats[name] = st
    flag = "few_nodes" if view.selection.valid_fraction < min_valid_fraction else "ok"
    return FrameStats(frame=frame, selection=view.selection, fit=view.fit, stats=stats, flag=flag)


def frame_series(
    result: PipelineResult,
    fields: Sequence[str],
    node_filter: NodeFilter | None = None,
    motion: str = "none",
    frames: Iterable[int] | None = None,
    region: NDArray[np.bool_] | None = None,
    min_valid_fraction: float = 0.0,
    stop: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    fit_region: NDArray[np.bool_] | None = None,
    with_ci: bool = False,
) -> list[FrameStats]:
    """:func:`frame_stats` frame after frame (all frames by default); ``stop`` is asked before each frame and
    ends the series early, ``progress(i, n)`` is told after each."""
    frames = list(range(len(result.result_disp)) if frames is None else frames)
    out: list[FrameStats] = []
    for i, frame in enumerate(frames):
        if stop is not None and stop():
            break
        out.append(frame_stats(result, frame, fields, node_filter, motion, region, min_valid_fraction, fit_region, with_ci))
        if progress is not None:
            progress(i, len(frames))
    return out
