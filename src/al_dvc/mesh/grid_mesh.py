"""Uniform hex8 node grid: construction, mask trimming, neighbour helpers.

Port of MATLAB ``MeshSetUp3.m`` with 0-based indices and the layout contract
``node n = iz*ny*nx + iy*nx + ix`` (grid fields are ``(nz, ny, nx)``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DVCMesh, VOIRange
from ..io.volume_ops import GRADIENT_BORDER

INTERP_MARGIN = 2  # voxels needed by the tricubic kernel on each side


def build_grid_axes(
    voi: VOIRange,
    volume_shape: tuple[int, int, int],
    winsize: tuple[int, int, int],
    winstepsize: tuple[int, int, int],
    extra_margin: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Node coordinate axes ``(x0, y0, z0)`` covering the VOI.

    Nodes are placed so that every reference subset lies inside the VOI and
    at least ``GRADIENT_BORDER + INTERP_MARGIN`` voxels away from the volume
    border (where gradients are zero and interpolation is undefined).
    Nodes are centred in the available span so the grid is symmetric.
    """
    voi = voi.clamp(volume_shape)
    nz, ny, nx = volume_shape
    axes = []
    for axis_len, (lo, hi), w, h in zip(
        (nx, ny, nz), (voi.x, voi.y, voi.z), winsize, winstepsize,
    ):
        half = w // 2
        border = GRADIENT_BORDER + INTERP_MARGIN + int(extra_margin)
        first = max(lo, border) + half
        last = min(hi, axis_len - 1 - border) - half
        if last < first:
            raise ValueError(
                f"Subset winsize={w} does not fit in VOI range ({lo}, {hi}) of a "
                f"{axis_len}-voxel axis (need >= {2 * (half + border) + 1} voxels)."
            )
        n = int((last - first) // h) + 1
        if n < 2:
            raise ValueError(
                f"Fewer than 2 nodes along an axis: VOI range ({lo}, {hi}), winsize={w}, "
                f"step={h}. Reduce winsize/winstepsize or enlarge the VOI."
            )
        span = (n - 1) * h
        start = first + (last - first - span) // 2
        axes.append(np.arange(n, dtype=np.float64) * h + start)
    return axes[0], axes[1], axes[2]


def mesh_setup(
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    z0: NDArray[np.float64],
) -> DVCMesh:
    """Create the uniform hex8 mesh on the tensor grid ``x0 x y0 x z0``."""
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    z0 = np.asarray(z0, dtype=np.float64)
    nx, ny, nz = len(x0), len(y0), len(z0)
    if min(nx, ny, nz) < 2:
        raise ValueError(f"Need >= 2 nodes per axis (got nx={nx}, ny={ny}, nz={nz}).")

    Z, Y, X = np.meshgrid(z0, y0, x0, indexing="ij")  # each (nz, ny, nx)
    coordinates = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    iz, iy, ix = np.meshgrid(
        np.arange(nz - 1), np.arange(ny - 1), np.arange(nx - 1), indexing="ij",
    )
    ix = ix.ravel()
    iy = iy.ravel()
    iz = iz.ravel()

    def nid(a: NDArray, b: NDArray, c: NDArray) -> NDArray:
        return c * ny * nx + b * nx + a

    elements = np.column_stack([
        nid(ix, iy, iz), nid(ix + 1, iy, iz), nid(ix + 1, iy + 1, iz), nid(ix, iy + 1, iz),
        nid(ix, iy, iz + 1), nid(ix + 1, iy, iz + 1), nid(ix + 1, iy + 1, iz + 1), nid(ix, iy + 1, iz + 1),
    ]).astype(np.int64)

    spacing = (
        float(x0[1] - x0[0]) if nx > 1 else 1.0,
        float(y0[1] - y0[0]) if ny > 1 else 1.0,
        float(z0[1] - z0[0]) if nz > 1 else 1.0,
    )
    n_nodes = coordinates.shape[0]
    mesh = DVCMesh(
        coordinates=coordinates,
        elements=elements,
        grid_shape=(nz, ny, nx),
        x0=x0.copy(),
        y0=y0.copy(),
        z0=z0.copy(),
        spacing=spacing,
        node_valid=np.ones(n_nodes, dtype=bool),
    )
    mesh.boundary_nodes = grid_surface_nodes(mesh.grid_shape)
    return mesh


def grid_surface_nodes(grid_shape: tuple[int, int, int]) -> NDArray[np.int64]:
    """Indices of nodes on the outer surface of the node grid."""
    nz, ny, nx = grid_shape
    on = np.zeros(grid_shape, dtype=bool)
    on[0, :, :] = on[-1, :, :] = True
    on[:, 0, :] = on[:, -1, :] = True
    on[:, :, 0] = on[:, :, -1] = True
    return np.flatnonzero(on.ravel()).astype(np.int64)


def subset_valid_fraction(
    mask: NDArray[np.uint8] | NDArray[np.bool_],
    coordinates: NDArray[np.float64],
    winsize: tuple[int, int, int],
) -> NDArray[np.float64]:
    """Fraction of mask-valid voxels inside each node's subset (integral image)."""
    m = np.asarray(mask)
    nz, ny, nx = m.shape
    # 3-D summed-area table with a leading zero plane along every axis
    sat = np.zeros((nz + 1, ny + 1, nx + 1), dtype=np.int64)
    sat[1:, 1:, 1:] = np.cumsum(np.cumsum(np.cumsum(m.astype(np.int64), axis=0), axis=1), axis=2)

    hx, hy, hz = (w // 2 for w in winsize)
    cx = np.round(coordinates[:, 0]).astype(np.int64)
    cy = np.round(coordinates[:, 1]).astype(np.int64)
    cz = np.round(coordinates[:, 2]).astype(np.int64)
    x_lo = np.clip(cx - hx, 0, nx)
    x_hi = np.clip(cx + hx + 1, 0, nx)
    y_lo = np.clip(cy - hy, 0, ny)
    y_hi = np.clip(cy + hy + 1, 0, ny)
    z_lo = np.clip(cz - hz, 0, nz)
    z_hi = np.clip(cz + hz + 1, 0, nz)

    def S(z, y, x):
        return sat[z, y, x]

    count = (
        S(z_hi, y_hi, x_hi) - S(z_lo, y_hi, x_hi) - S(z_hi, y_lo, x_hi) - S(z_hi, y_hi, x_lo)
        + S(z_lo, y_lo, x_hi) + S(z_lo, y_hi, x_lo) + S(z_hi, y_lo, x_lo) - S(z_lo, y_lo, x_lo)
    )
    total = float((2 * hx + 1) * (2 * hy + 1) * (2 * hz + 1))
    return count.astype(np.float64) / total


def apply_mask_to_mesh(
    mesh: DVCMesh,
    mask: NDArray[np.uint8] | NDArray[np.bool_] | None,
    winsize: tuple[int, int, int],
    min_valid_ratio: float,
) -> DVCMesh:
    """Mark nodes invalid (outside the mask / poor coverage) and drop elements.

    A node is valid when its centre voxel is inside the mask and at least
    ``min_valid_ratio`` of its subset voxels are. Elements with any invalid
    corner are removed (row set to -1). Returns a new ``DVCMesh``.
    """
    n = mesh.n_nodes
    if mask is None:
        node_valid = np.ones(n, dtype=bool)
        elements = mesh.elements.copy()
    else:
        m = np.asarray(mask)
        nz, ny, nx = m.shape
        cx = np.clip(np.round(mesh.coordinates[:, 0]).astype(int), 0, nx - 1)
        cy = np.clip(np.round(mesh.coordinates[:, 1]).astype(int), 0, ny - 1)
        cz = np.clip(np.round(mesh.coordinates[:, 2]).astype(int), 0, nz - 1)
        centre_ok = m[cz, cy, cx] > 0
        frac = subset_valid_fraction(m, mesh.coordinates, winsize)
        node_valid = centre_ok & (frac >= min_valid_ratio)
        elements = mesh.elements.copy()
        if elements.size:
            bad_elem = ~node_valid[elements].all(axis=1)
            elements[bad_elem, :] = -1

    boundary = set(grid_surface_nodes(mesh.grid_shape).tolist())
    if not node_valid.all():
        # valid nodes adjacent (6-connectivity) to an invalid node are boundary
        nz, ny, nx = mesh.grid_shape
        v = node_valid.reshape(nz, ny, nx)
        near_invalid = np.zeros_like(v)
        pad = np.pad(v, 1, mode="constant", constant_values=True)
        for shift in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            sl = tuple(slice(1 + s, 1 + s + d) for s, d in zip(shift, (nz, ny, nx)))
            near_invalid |= ~pad[sl]
        boundary |= set(np.flatnonzero((near_invalid & v).ravel()).tolist())

    return DVCMesh(
        coordinates=mesh.coordinates,
        elements=elements,
        grid_shape=mesh.grid_shape,
        x0=mesh.x0,
        y0=mesh.y0,
        z0=mesh.z0,
        spacing=mesh.spacing,
        node_valid=node_valid,
        boundary_nodes=np.array(sorted(boundary), dtype=np.int64),
    )


def active_elements(mesh: DVCMesh) -> NDArray[np.int64]:
    """Element rows that were not dropped."""
    if mesh.elements.size == 0:
        return np.empty((0, 8), dtype=np.int64)
    return mesh.elements[mesh.elements[:, 0] >= 0]


def nodes_in_active_elements(mesh: DVCMesh) -> NDArray[np.bool_]:
    """Boolean ``(N,)``: node belongs to at least one active element."""
    used = np.zeros(mesh.n_nodes, dtype=bool)
    act = active_elements(mesh)
    if act.size:
        used[np.unique(act)] = True
    return used
