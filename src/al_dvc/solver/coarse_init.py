"""Initial guess from a coarse node lattice: NCC + IC-GN on every k-th node, interpolated to all nodes.

The NCC pyramid search is the second most expensive stage after the local
IC-GN, and its cost is proportional to the number of nodes. Solving the
displacement *and* the deformation gradient on a lattice with ``k^3`` fewer
nodes and interpolating both to the full grid (the idea behind pyALDIC's
seed propagation, without the sequential wave) gives the fine IC-GN a start
that is typically within 0.1 voxel and already carries the local gradient,
so it also needs fewer iterations. The fine nodes still run the full 12-DOF
IC-GN: the interpolated field is only where they start.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from ..core.config import DVCPara
from ..core.data_structures import DVCMesh, ReferenceBundle
from ..mesh.grid_mesh import mesh_setup
from .init_disp import compute_initial_guess
from .local_icgn import local_icgn, precompute_local_context

logger = logging.getLogger(__name__)

MIN_COARSE_NODES_PER_AXIS = 2


def coarse_lattice(mesh: DVCMesh, factor: int) -> tuple[DVCMesh, NDArray[np.int64]] | None:
    """Every ``factor``-th node per axis as a mesh of its own (``None`` when too few nodes remain)."""
    k = int(factor)
    if k <= 1:
        return None
    axes = [np.arange(0, n, k) for n in mesh.grid_shape]  # (z, y, x) index sets
    if min(a.size for a in axes) < MIN_COARSE_NODES_PER_AXIS:
        return None
    coarse = mesh_setup(mesh.x0[axes[2]], mesh.y0[axes[1]], mesh.z0[axes[0]])
    iz, iy, ix = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    nz, ny, nx = mesh.grid_shape
    fine_index = (iz * ny * nx + iy * nx + ix).ravel()
    if mesh.node_valid.size == mesh.n_nodes:
        coarse.node_valid = np.asarray(mesh.node_valid)[fine_index].copy()
    return coarse, fine_index


def interpolate_to_mesh(coarse: DVCMesh, values: NDArray, mesh: DVCMesh) -> NDArray[np.float64]:
    """Trilinear interpolation of per-node values ``(Nc, ...)`` from the coarse lattice to every node of ``mesh``.

    Nodes outside the coarse lattice (the boundary layers dropped by the
    stride) are extrapolated linearly; NaN entries are filled from the
    nearest finite coarse node first so the interpolation is always defined.
    """
    v = np.asarray(values, dtype=np.float64)
    grid = coarse.to_grid(v)
    flat = grid.reshape(grid.shape[:3] + (-1,))
    if not np.all(np.isfinite(flat)):
        flat = _fill_nan_nearest(flat)
    interp = RegularGridInterpolator(
        (coarse.z0, coarse.y0, coarse.x0), flat, method="linear", bounds_error=False, fill_value=None
    )
    pts = np.column_stack([mesh.coordinates[:, 2], mesh.coordinates[:, 1], mesh.coordinates[:, 0]])
    out = interp(pts)
    return out.reshape((mesh.n_nodes,) + v.shape[1:])


def _fill_nan_nearest(flat: NDArray[np.float64]) -> NDArray[np.float64]:
    from scipy.ndimage import distance_transform_edt

    out = flat.copy()
    bad = ~np.all(np.isfinite(out), axis=-1)
    if bad.all():
        return np.nan_to_num(out)
    _, idx = distance_transform_edt(bad, return_distances=True, return_indices=True)
    return out[tuple(idx)]


def coarse_initial_guess(
    bundle: ReferenceBundle,
    g_norm: NDArray[np.float32],
    g_prep: NDArray[np.float32],
    mesh: DVCMesh,
    para: DVCPara,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, dict]:
    """``(U0, F0, info)`` for every node of ``mesh`` from the coarse-lattice solve.

    Falls back to the plain NCC initial guess (``F0 = None``) when the coarse
    lattice would have fewer than two nodes per axis.
    """
    t0 = time.perf_counter()
    lattice = coarse_lattice(mesh, int(para.init_coarse_factor))
    if lattice is None:
        U0, info = compute_initial_guess(bundle.f, g_norm, mesh, para)
        info["coarse_factor"] = 1
        return U0, None, info
    coarse, fine_index = lattice
    U0c, info = compute_initial_guess(bundle.f, g_norm, coarse, para)
    t_ncc = time.perf_counter() - t0
    ctx_c = precompute_local_context(coarse, bundle, para)
    Uc, Fc, info_c, bad_c = local_icgn(ctx_c, bundle, g_prep, U0c, para, coarse)
    U0 = interpolate_to_mesh(coarse, Uc, mesh)
    F0 = interpolate_to_mesh(coarse, Fc.reshape(coarse.n_nodes, 9), mesh).reshape(mesh.n_nodes, 3, 3)
    dt = time.perf_counter() - t0
    info.update(
        {
            "method": f"coarse x{int(para.init_coarse_factor)} + {info.get('method', 'ncc')}",
            "coarse_factor": int(para.init_coarse_factor),
            "coarse_nodes": int(coarse.n_nodes),
            "coarse_converged": float(np.mean(info_c.status == 0)),
            "coarse_bad": int(bad_c.sum()),
            "coarse_ncc_time": t_ncc,
            "coarse_icgn_time": float(info_c.solve_time),
            "coarse_total_time": dt,
        }
    )
    logger.info(
        "Coarse initial guess: %d of %d nodes solved (%.1f%% converged, %d filled), %.2fs NCC + %.2fs IC-GN",
        coarse.n_nodes,
        mesh.n_nodes,
        100.0 * info["coarse_converged"],
        int(bad_c.sum()),
        t_ncc,
        info_c.solve_time,
    )
    return U0, F0, info
