"""The planned node lattice, before any run: where the nodes will be and how big the subsets are.

The slice viewer draws this so the user can judge the subset size and the step against the
texture of the volume (pyALDIC shows the same thing on its image). Everything here is pure
NumPy on the lattice axes, so it stays cheap for large scans: the only per-voxel work is one
lookup of the mask at the node centres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from al_dvc.core.data_structures import VOIRange
from al_dvc.mesh.grid_mesh import build_grid_axes

PLANES = ("xy", "xz", "yz")
# (horizontal axis, vertical axis, normal axis) of every plane, as indices into (x, y, z)
PLANE_AXES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
HOVER_RADIUS = 1.5  # in steps: a pointer further than this from every node highlights nothing


@dataclass(frozen=True)
class LatticePlan:
    """Node axes ``(x0, y0, z0)`` in voxels, the subset span and the mask validity of the node centres."""

    x0: NDArray[np.float64]
    y0: NDArray[np.float64]
    z0: NDArray[np.float64]
    winsize: tuple[int, int, int]  # even; the subset spans winsize + 1 voxels (2h + 1)
    winstepsize: tuple[int, int, int]
    centre_valid: NDArray[np.bool_] | None = None  # (nz, ny, nx) over the lattice; None without a mask

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return len(self.z0), len(self.y0), len(self.x0)

    @property
    def n_nodes(self) -> int:
        nz, ny, nx = self.grid_shape
        return nz * ny * nx

    @property
    def n_valid(self) -> int:
        return self.n_nodes if self.centre_valid is None else int(self.centre_valid.sum())

    @property
    def half(self) -> tuple[int, int, int]:
        return tuple(int(w) // 2 for w in self.winsize)  # type: ignore[return-value]

    @property
    def overlap(self) -> tuple[float, float, float]:
        """Fraction of the subset edge shared with the next node along every axis (0 = touching, < 0 = gap)."""
        return tuple(1.0 - float(s) / float(w + 1) for s, w in zip(self.winstepsize, self.winsize))  # type: ignore[return-value]

    def axes(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return self.x0, self.y0, self.z0


def plan_lattice(
    shape: tuple[int, int, int],
    winsize: tuple[int, int, int],
    winstepsize: tuple[int, int, int],
    voi: VOIRange | None = None,
    mask: NDArray | None = None,
) -> LatticePlan:
    """The lattice the pipeline would build for ``shape`` (``(nz, ny, nx)``) with these parameters.

    Raises ``ValueError`` with the pipeline's own message when the subset does not fit.
    """
    nz, ny, nx = (int(s) for s in shape)
    box = voi if voi is not None else VOIRange(x=(0, nx - 1), y=(0, ny - 1), z=(0, nz - 1))
    ws = tuple(int(w) for w in winsize)
    st = tuple(int(s) for s in winstepsize)
    x0, y0, z0 = build_grid_axes(box, (nz, ny, nx), ws, st)  # type: ignore[arg-type]
    centre_valid = None
    if mask is not None:
        m = np.asarray(mask)
        if m.shape != (nz, ny, nx):
            raise ValueError(f"mask shape {m.shape} does not match the volume shape {(nz, ny, nx)}")
        cz = np.clip(np.round(z0).astype(int), 0, nz - 1)
        cy = np.clip(np.round(y0).astype(int), 0, ny - 1)
        cx = np.clip(np.round(x0).astype(int), 0, nx - 1)
        centre_valid = m[np.ix_(cz, cy, cx)] > 0
    return LatticePlan(x0, y0, z0, ws, st, centre_valid)  # type: ignore[arg-type]


def layer_index(plan: LatticePlan, plane: str, slice_index: int) -> int:
    """Index of the lattice layer nearest to ``slice_index`` along the plane's normal axis."""
    normal = plan.axes()[PLANE_AXES[plane][2]]
    return int(np.argmin(np.abs(normal - float(slice_index))))


def layer_nodes(
    plan: LatticePlan, plane: str, slice_index: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], float]:
    """Nodes of the layer nearest to the slice: ``(h, v, valid, distance)``.

    ``h`` / ``v`` are the in-plane node coordinates (flattened), ``valid`` the mask validity of
    the centres and ``distance`` how far the layer is from the slice along the normal (voxels).
    """
    ih, iv, inorm = PLANE_AXES[plane]
    ax = plan.axes()
    k = layer_index(plan, plane, slice_index)
    H, V = np.meshgrid(ax[ih], ax[iv], indexing="xy")  # V varies along rows
    if plan.centre_valid is None:
        valid = np.ones(H.shape, dtype=bool)
    else:
        cv = plan.centre_valid  # (nz, ny, nx)
        valid = {"xy": cv[k], "xz": cv[:, k, :], "yz": cv[:, :, k]}[plane]
    return H.ravel(), V.ravel(), np.asarray(valid).ravel(), float(abs(ax[inorm][k] - slice_index))


def nearest_node(plan: LatticePlan, plane: str, h: float, v: float, slice_index: int) -> tuple[float, float] | None:
    """The node of the layer nearest to the pointer, within :data:`HOVER_RADIUS` steps, else ``None``."""
    ih, iv, _ = PLANE_AXES[plane]
    H, V, _valid, _d = layer_nodes(plan, plane, slice_index)
    if H.size == 0:
        return None
    sh, sv = plan.winstepsize[ih], plan.winstepsize[iv]
    d2 = ((H - h) / sh) ** 2 + ((V - v) / sv) ** 2
    i = int(np.argmin(d2))
    if d2[i] > HOVER_RADIUS**2:
        return None
    return float(H[i]), float(V[i])


def subset_rect(plan: LatticePlan, plane: str, centre: tuple[float, float]) -> tuple[float, float, float, float]:
    """``(left, bottom, width, height)`` of the subset around ``centre`` on the plane, in voxel units.

    The subset spans ``2h + 1`` voxels per axis; the rectangle covers those voxels' full extent,
    matching an image drawn with voxel centres at integer coordinates.
    """
    ih, iv, _ = PLANE_AXES[plane]
    hh, hv = plan.half[ih], plan.half[iv]
    return centre[0] - hh - 0.5, centre[1] - hv - 0.5, 2 * hh + 1, 2 * hv + 1


def describe(plan: LatticePlan) -> str:
    """One line for the viewer: grid size, node count and subset overlap."""
    nz, ny, nx = plan.grid_shape
    ws = " x ".join(str(w + 1) for w in plan.winsize)
    ov = plan.overlap
    same = all(abs(o - ov[0]) < 1e-9 for o in ov)
    overlap = f"{ov[0] * 100:.0f} %" if same else " / ".join(f"{o * 100:.0f} %" for o in ov)
    text = f"{nx} x {ny} x {nz} = {plan.n_nodes:,} nodes, subset {ws}, overlap {overlap}"
    if plan.centre_valid is not None and plan.n_valid != plan.n_nodes:
        text += f", {plan.n_valid:,} in the region of interest"
    return text
