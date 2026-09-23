"""Which nodes a statistic stands on.

Not every value on the node grid was measured: a node whose local solve did not converge carries the global
step's value or an inpainted one, and a node the median test rejected was replaced. By default a statistic
uses measured nodes only. Every filter removes its nodes in a fixed order and reports how many, so a table
can always say what its numbers stand on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import STATUS_CONVERGED, PipelineResult

FILTER_REASONS = ("invalid", "not_converged", "outlier", "low_zncc", "edge", "cut")


@dataclass(frozen=True)
class NodeFilter:
    """Which nodes count. Nodes invalid on the reference (outside the region of interest, ill-conditioned)
    never count: nothing was measured there."""

    converged_only: bool = True  # drop nodes whose final local solve did not converge
    drop_outliers: bool = True  # drop nodes the median test rejected and replaced
    min_zncc: float = 0.0  # drop nodes below this correlation (0 = off)
    edge_layers: int = 0  # drop this many node layers next to the grid border or an invalid node
    drop_cut: bool = False  # drop nodes whose subset was cut at a boundary (subset splitting)


@dataclass(frozen=True, eq=False)
class Selection:
    """The nodes used, out of ``candidates`` (the region), and how many each filter removed."""

    mask: NDArray[np.bool_]
    candidates: int
    removed: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.mask.sum())

    @property
    def valid_fraction(self) -> float:
        return self.n / self.candidates if self.candidates else 0.0

    def as_dict(self) -> dict:
        return {
            "nodes_used": self.n,
            "candidates": self.candidates,
            "valid_fraction": self.valid_fraction,
            "removed": dict(self.removed),
        }


def chain_pairs(result: PipelineResult, frame: int) -> list[int]:
    """Indices into ``result.result_disp`` of the pairs whose displacements compose frame ``frame``.

    One pair in accumulative mode; in incremental mode (or a custom schedule) every pair back to the
    reference, since the cumulative displacement passes through each of them.
    """
    schedule = getattr(result, "frame_schedule", None)
    if schedule is None:  # no schedule to follow: the pair itself
        return [frame]
    path = schedule.path_to_root(frame + 1)  # a frame outside the schedule is the caller's bug: IndexError
    pairs = [k - 1 for k in path[:-1] if 0 <= k - 1 < len(result.result_disp)]
    return pairs or [frame]


def select_nodes(
    result: PipelineResult, frame: int, node_filter: NodeFilter | None = None, region: NDArray[np.bool_] | None = None
) -> Selection:
    """The nodes of ``region`` (all nodes by default) that pass ``node_filter`` for frame ``frame``."""
    nf = node_filter or NodeFilter()
    mesh = result.dvc_mesh
    n = mesh.n_nodes
    inside = np.ones(n, dtype=bool) if region is None else np.asarray(region, dtype=bool).ravel().copy()
    if inside.shape != (n,):
        raise ValueError(f"region has {inside.size} entries for {n} nodes")
    mask = inside.copy()
    removed: dict[str, int] = {}

    def drop(reason: str, bad: NDArray[np.bool_]) -> None:
        removed[reason] = int(np.sum(mask & bad))
        mask[bad] = False

    valid = np.asarray(mesh.node_valid, dtype=bool) if np.size(mesh.node_valid) == n else np.ones(n, dtype=bool)
    drop("invalid", ~valid)
    not_converged = np.zeros(n, dtype=bool)
    outlier = np.zeros(n, dtype=bool)
    low = np.zeros(n, dtype=bool)
    cut = np.zeros(n, dtype=bool)
    for p in chain_pairs(result, frame):
        fr = result.result_disp[p]
        if nf.converged_only and fr.status is not None:
            not_converged |= np.asarray(fr.status) != STATUS_CONVERGED
        if nf.drop_outliers and fr.outlier is not None:
            outlier |= np.asarray(fr.outlier, dtype=bool)
        if nf.min_zncc > 0 and fr.zncc is not None:
            low |= ~(np.asarray(fr.zncc, dtype=np.float64) >= nf.min_zncc)  # NaN counts as low
        if nf.drop_cut and fr.split_fraction is not None:
            cut |= ~(np.asarray(fr.split_fraction, dtype=np.float64) >= 1.0)
    drop("not_converged", not_converged)
    drop("outlier", outlier)
    drop("low_zncc", low)
    edge = np.zeros(n, dtype=bool)
    if nf.edge_layers > 0:
        from scipy.ndimage import binary_erosion

        body = (valid & inside).reshape(mesh.grid_shape)
        core = binary_erosion(body, structure=np.ones((3, 3, 3), dtype=bool), iterations=int(nf.edge_layers), border_value=0)
        edge = ~core.ravel()
    drop("edge", edge)
    drop("cut", cut)
    return Selection(mask=mask, candidates=int(inside.sum()), removed=removed)
