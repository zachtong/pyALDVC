"""Profiles of a field: the mean of every node layer along an axis, the values along a line, and a virtual
extensometer between two points.

Sampling off the nodes is trilinear on the node grid and never invents a value: a sample whose neighbouring
nodes include one without a value (outside the grid, invalid, not used) is NaN -- a gap, not a clamped edge value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import PipelineResult
from .selection import NodeFilter
from .view import FrameView, frame_view

AXES = {"x": 0, "y": 1, "z": 2}
SAMPLES_PER_STEP = 2  # line samples per node spacing
MAX_SAMPLES = 512


@dataclass(frozen=True, eq=False)
class AxisProfile:
    """The mean, standard deviation and count of a field over each node layer along ``axis``."""

    axis: str
    positions: NDArray[np.float64]  # physical units
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    n: NDArray[np.int64]

    def as_dict(self) -> dict:
        return {
            "axis": self.axis,
            "positions": self.positions.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n": self.n.tolist(),
        }


def axis_profile(view: FrameView, field: str, axis: str, region: NDArray[np.bool_] | None = None) -> AxisProfile:
    """Layer means of ``field`` along ``axis`` over the nodes the view uses (and ``region``)."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {tuple(AXES)}, got {axis!r}")
    mesh = view.result.dvc_mesh
    values = view.values(field)
    use = view.selection.mask & np.isfinite(values)
    if region is not None:
        use &= np.asarray(region, dtype=bool)
    k = AXES[axis]
    grid_axis = 2 - k  # the node grid is (nz, ny, nx)
    g = np.moveaxis(mesh.to_grid(np.where(use, values, 0.0)), grid_axis, 0)
    c = np.moveaxis(mesh.to_grid(use), grid_axis, 0)
    layers = g.shape[0]
    g = g.reshape(layers, -1)
    c = c.reshape(layers, -1)
    n = c.sum(axis=1).astype(np.int64)
    safe = np.maximum(n, 1)
    mean = g.sum(axis=1) / safe
    var = np.where(c, (g - mean[:, None]) ** 2, 0.0).sum(axis=1) / safe
    empty = n == 0
    mean[empty] = np.nan
    std = np.sqrt(var)
    std[empty] = np.nan
    coord = (mesh.x0, mesh.y0, mesh.z0)[k]
    scale = float(np.asarray(view.result.dvc_para.voxel_size, dtype=np.float64)[k])
    return AxisProfile(axis=axis, positions=np.asarray(coord, dtype=np.float64) * scale, mean=mean, std=std, n=n)


def _trilinear(result: PipelineResult, values: NDArray, use: NDArray[np.bool_], points) -> NDArray[np.float64]:
    """``values`` (per node, used where ``use``) at voxel positions ``points`` ``(M, 3)``; NaN where any node of
    the stencil with a non-zero weight has no value, or the point is off the grid."""
    from scipy.ndimage import map_coordinates

    mesh = result.dvc_mesh
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    hx, hy, hz = (float(h) for h in mesh.spacing)
    idx = np.vstack([(P[:, 2] - float(mesh.z0[0])) / hz, (P[:, 1] - float(mesh.y0[0])) / hy, (P[:, 0] - float(mesh.x0[0])) / hx])
    ok = np.asarray(use, dtype=bool) & np.isfinite(values)
    grid = mesh.to_grid(np.where(ok, values, 0.0))
    weight = mesh.to_grid(ok.astype(np.float64))
    v = map_coordinates(grid, idx, order=1, mode="constant", cval=0.0)
    w = map_coordinates(weight, idx, order=1, mode="constant", cval=0.0)
    return np.where(w > 1.0 - 1e-9, v, np.nan)


def sample_line(view: FrameView, field: str, p0, p1, n: int | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(distance, values)`` of ``field`` along the segment ``p0``-``p1`` (voxel positions ``[x, y, z]``);
    distances in physical units from ``p0``. ``n`` samples, by default two per node spacing."""
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    mesh = view.result.dvc_mesh
    if n is None:
        steps = float(np.max(np.abs(p1 - p0) / np.asarray(mesh.spacing, dtype=np.float64)))
        n = int(np.clip(np.ceil(SAMPLES_PER_STEP * steps) + 1, 2, MAX_SAMPLES))
    t = np.linspace(0.0, 1.0, int(n))
    P = p0 + t[:, None] * (p1 - p0)
    vs = np.asarray(view.result.dvc_para.voxel_size, dtype=np.float64)
    distance = np.linalg.norm((P - p0) * vs, axis=1)
    values = view.values(field)
    return distance, _trilinear(view.result, values, view.selection.mask, P)


@dataclass(frozen=True, eq=False)
class Extensometer:
    """A virtual extensometer: the length between two material points over the frames, ``strain = L / L0 - 1``.
    Rotation-invariant (a length), so no motion needs removing; NaN in a frame where an end point has no value."""

    p0: NDArray[np.float64]
    p1: NDArray[np.float64]
    L0: float  # physical units
    frames: NDArray[np.int64]
    length: NDArray[np.float64]
    strain: NDArray[np.float64]

    def as_dict(self) -> dict:
        return {
            "p0_voxel": self.p0.tolist(),
            "p1_voxel": self.p1.tolist(),
            "L0": self.L0,
            "frames": (self.frames + 1).tolist(),
            "length": self.length.tolist(),
            "strain": self.strain.tolist(),
        }


def extensometer(result: PipelineResult, p0, p1, frames=None, node_filter: NodeFilter | None = None) -> Extensometer:
    """The extensometer between the reference positions ``p0`` and ``p1`` (voxels) over ``frames`` (all)."""
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    vs = np.asarray(result.dvc_para.voxel_size, dtype=np.float64)
    L0 = float(np.linalg.norm((p1 - p0) * vs))
    if L0 <= 0.0:
        raise ValueError("the two points of an extensometer must differ")
    frames = np.asarray(list(range(len(result.result_disp)) if frames is None else frames), dtype=np.int64)
    length = np.full(frames.size, np.nan)
    ends = np.vstack([p0, p1])
    for i, k in enumerate(frames):
        view = frame_view(result, int(k), "none", node_filter)
        U = view.displacement()
        u = np.column_stack([_trilinear(result, U[:, c], view.selection.mask, ends) for c in range(3)])
        if np.all(np.isfinite(u)):
            length[i] = float(np.linalg.norm((p1 * vs + u[1]) - (p0 * vs + u[0])))
    return Extensometer(p0=p0, p1=p1, L0=L0, frames=frames, length=length, strain=length / L0 - 1.0)
