"""Cut the hex8 mesh where a masked boundary separates the corners of an element.

The 3-D port of pyALDIC's ``mark_bridging``: the material (in-mask) voxels inside an element's
bounding box are labelled with 6-connectivity, and the element is *bridging* when its eight
corners do not all fall in the component of the first corner. A bridging element spans a masked
boundary, however thin, and is removed so that no global operator, inpainting step, median test
or strain fit reaches across it. The test is local on purpose: at a crack front the material wraps
around inside the box, the corners stay connected and the cut stops there by itself, exactly like
the subset splitting of the local kernels.

An edge of the node grid is *cut* when every element that contains it (among the elements whose
corners are all valid) is bridging; ``edge_ok`` records the +x, +y and +z edge of every node.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._numba_compat import JIT_CACHE, njit, prange
from ..utils.flood_fill import flood_fill_from

# hex8 corner order of ``mesh_setup``: 0 (0,0,0) 1 (1,0,0) 2 (1,1,0) 3 (0,1,0) 4 (0,0,1) 5 (1,0,1) 6 (1,1,1) 7 (0,1,1)
EDGES = {  # axis (x, y, z) -> (lower corner, upper corner) of the four element edges along it
    0: ((0, 1), (3, 2), (4, 5), (7, 6)),
    1: ((0, 3), (1, 2), (4, 7), (5, 6)),
    2: ((0, 4), (1, 5), (2, 6), (3, 7)),
}


@njit(parallel=True, cache=JIT_CACHE)
def _bridging_jit(mask, corners, out):
    """``corners``: ``(E, 8, 3)`` int64 voxel positions ``[x, y, z]``; ``out``: ``(E,)`` bool."""
    E = corners.shape[0]
    nz = mask.shape[0]
    ny = mask.shape[1]
    nx = mask.shape[2]
    for e in prange(E):
        x_lo = corners[e, 0, 0]
        x_hi = x_lo
        y_lo = corners[e, 0, 1]
        y_hi = y_lo
        z_lo = corners[e, 0, 2]
        z_hi = z_lo
        for c in range(1, 8):
            x_lo = min(x_lo, corners[e, c, 0])
            x_hi = max(x_hi, corners[e, c, 0])
            y_lo = min(y_lo, corners[e, c, 1])
            y_hi = max(y_hi, corners[e, c, 1])
            z_lo = min(z_lo, corners[e, c, 2])
            z_hi = max(z_hi, corners[e, c, 2])
        x_lo = max(x_lo, 0)
        y_lo = max(y_lo, 0)
        z_lo = max(z_lo, 0)
        x_hi = min(x_hi, nx - 1)
        y_hi = min(y_hi, ny - 1)
        z_hi = min(z_hi, nz - 1)
        Sx = x_hi - x_lo + 1
        Sy = y_hi - y_lo + 1
        Sz = z_hi - z_lo + 1
        sub = np.zeros((Sz, Sy, Sx), dtype=np.uint8)
        comp = np.zeros((Sz, Sy, Sx), dtype=np.uint8)
        queue = np.empty(Sz * Sy * Sx, dtype=np.int64)
        for iz in range(Sz):
            for iy in range(Sy):
                for ix in range(Sx):
                    if mask[z_lo + iz, y_lo + iy, x_lo + ix] != 0:
                        sub[iz, iy, ix] = 1
        cz = min(max(corners[e, 0, 2] - z_lo, 0), Sz - 1)
        cy = min(max(corners[e, 0, 1] - y_lo, 0), Sy - 1)
        cx = min(max(corners[e, 0, 0] - x_lo, 0), Sx - 1)
        if flood_fill_from(sub, comp, queue, cz, cy, cx) == 0:
            out[e] = True  # the first corner sits on masked material: the element spans a cut
            continue
        bridging = False
        for c in range(1, 8):
            rz = min(max(corners[e, c, 2] - z_lo, 0), Sz - 1)
            ry = min(max(corners[e, c, 1] - y_lo, 0), Sy - 1)
            rx = min(max(corners[e, c, 0] - x_lo, 0), Sx - 1)
            if comp[rz, ry, rx] == 0:
                bridging = True
        out[e] = bridging


def bridging_elements(mask, coordinates: NDArray[np.float64], elements: NDArray[np.int64]) -> NDArray[np.bool_]:
    """``(E,)`` True for every element of ``elements`` (rows of 8 node indices) that spans a masked boundary."""
    elements = np.asarray(elements, dtype=np.int64)
    out = np.zeros(elements.shape[0], dtype=np.bool_)
    if elements.shape[0] == 0:
        return out
    corners = np.round(np.asarray(coordinates, dtype=np.float64)[elements]).astype(np.int64)  # (E, 8, 3)
    _bridging_jit(np.ascontiguousarray(mask, dtype=np.uint8), np.ascontiguousarray(corners), out)
    return out


def edge_ok_from_elements(elements: NDArray[np.int64], bridging: NDArray[np.bool_], n_nodes: int) -> NDArray[np.bool_]:
    """``(N, 3)`` edge flags: the +x, +y, +z edge of a node is cut when every element containing it is bridging."""
    n_total = np.zeros((n_nodes, 3), dtype=np.int64)
    n_bridge = np.zeros((n_nodes, 3), dtype=np.int64)
    b = np.asarray(bridging, dtype=np.int64)
    for axis, pairs in EDGES.items():
        for lo, _hi in pairs:
            nodes = elements[:, lo]
            np.add.at(n_total[:, axis], nodes, 1)
            np.add.at(n_bridge[:, axis], nodes, b)
    return ~((n_total > 0) & (n_bridge == n_total))


def cut_mesh(
    elements: NDArray[np.int64], coordinates: NDArray[np.float64], mask, n_nodes: int
) -> tuple[NDArray[np.int64], NDArray[np.bool_], int]:
    """Drop the bridging elements (rows set to -1) and return ``(elements, edge_ok, n_dropped)``.

    Only the surviving rows (``elements[:, 0] >= 0``) are tested; rows already dropped for an invalid
    corner stay dropped and do not count against an edge.
    """
    elements = np.array(elements, dtype=np.int64, copy=True)
    rows = np.flatnonzero(elements[:, 0] >= 0) if elements.size else np.empty(0, dtype=np.int64)
    if rows.size == 0:
        return elements, np.ones((n_nodes, 3), dtype=bool), 0
    live = elements[rows]
    bridging = bridging_elements(mask, coordinates, live)
    edge_ok = edge_ok_from_elements(live, bridging, n_nodes)
    elements[rows[bridging]] = -1
    return elements, edge_ok, int(bridging.sum())


def lattice_edge_ok(x0, y0, z0, mask) -> NDArray[np.bool_]:
    """``(N, 3)`` edge flags of the lattice ``x0 x y0 x z0`` under ``mask`` (no mesh object needed).

    Used by the previews, which show the grid before a run: the same bridging test as
    :func:`cut_mesh`, so the lines they draw stop where the solver's elements do.
    """
    from .grid_mesh import mesh_setup

    mesh = mesh_setup(np.asarray(x0, float), np.asarray(y0, float), np.asarray(z0, float))
    _elements, edge_ok, _n = cut_mesh(mesh.elements, mesh.coordinates, np.asarray(mask), mesh.n_nodes)
    return edge_ok


def cut_edge_nodes(edge_ok: NDArray[np.bool_], grid_shape: tuple[int, int, int]) -> NDArray[np.int64]:
    """Indices of the nodes at either end of a cut edge."""
    nz, ny, nx = grid_shape
    strides = (1, nx, nx * ny)  # +x, +y, +z neighbour of node n
    nodes = []
    for axis in range(3):
        lo = np.flatnonzero(~edge_ok[:, axis])
        nodes.append(lo)
        nodes.append(lo + strides[axis])
    if not nodes:
        return np.empty(0, dtype=np.int64)
    out = np.unique(np.concatenate(nodes))
    return out[out < nz * ny * nx]
