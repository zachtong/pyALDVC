"""Strain computation router (MATLAB Section 8 / ``ComputeStrain3.m``)."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ..core.config import DVCPara
from ..core.data_structures import DVCMesh, StrainResult
from ..utils.grid_interp import smooth_grid_field
from .gradient_methods import gradient_fd, gradient_plane_fit
from .strain_types import (
    det_deformation_gradient,
    max_shear_strain,
    polar_rotation_deg,
    principal_strains,
    scale_to_physical,
    strain_tensor,
    volumetric_strain,
    von_mises_strain,
)

logger = logging.getLogger(__name__)


def compute_strain(
    mesh: DVCMesh,
    para: DVCPara,
    U: NDArray[np.float64],
    F_direct: NDArray[np.float64] | None = None,
    valid: NDArray[np.bool_] | None = None,
    ops=None,
) -> StrainResult:
    """Strain of one frame from nodal displacement ``U`` (N, 3) in voxels.

    Args:
        F_direct: ``(N, 3, 3)`` ADMM gradient, used by ``strain_method="direct"``.
        valid: ``(N,)`` node validity (defaults to ``mesh.node_valid``).
        ops: ``GlobalOperators`` for ``strain_method="fem"`` (built on demand
            otherwise).
    """
    N = mesh.n_nodes
    grid = mesh.grid_shape
    U = np.asarray(U, dtype=np.float64).reshape(N, 3)
    if valid is None:
        valid = mesh.node_valid if mesh.node_valid.size == N else np.ones(N, dtype=bool)
    valid = np.asarray(valid, dtype=bool) & np.all(np.isfinite(U), axis=1)
    valid_grid = valid.reshape(grid)
    U_grid = U.reshape(grid + (3,)).copy()
    U_grid[~valid_grid] = np.nan

    # optional displacement smoothing (sigma in nodes)
    if para.disp_smoothing > 0:
        for c in range(3):
            U_grid[..., c] = smooth_grid_field(U_grid[..., c], para.disp_smoothing, valid_grid)
        U = np.where(valid[:, None], U_grid.reshape(N, 3), U)

    method = para.strain_method
    complete = valid_grid.copy()
    if method == "direct" and F_direct is None:
        logger.warning("strain_method='direct' requires the ADMM gradient; falling back to plane_fit.")
        method = "plane_fit"
    if method == "fem" and ops is None:
        from ..solver.global_operators import build_global_operators

        try:
            ops = build_global_operators(mesh, "fem", para.gauss_pt_order)
        except ValueError:
            logger.warning("No active elements for FEM strain; falling back to plane_fit.")
            method = "plane_fit"

    if method == "plane_fit":
        F_grid, complete = gradient_plane_fit(U_grid, mesh.spacing, para.strain_plane_fit_halfwidth, valid_grid)
        F = F_grid.reshape(N, 3, 3)
    elif method == "fd":
        F_grid, complete = gradient_fd(U_grid, mesh.spacing, valid_grid)
        F = F_grid.reshape(N, 3, 3)
    elif method == "fem":
        from ..solver.global_operators import nodal_gradient

        U_fill = U.copy()
        U_fill[~valid] = 0.0
        F = nodal_gradient(ops, U_fill)
        F[~valid] = np.nan
        complete = valid_grid & ~np.isin(np.arange(N), mesh.boundary_nodes).reshape(grid)
    else:  # direct
        F = np.asarray(F_direct, dtype=np.float64).reshape(N, 3, 3).copy()
        F[~valid] = np.nan
        complete = valid_grid & ~np.isin(np.arange(N), mesh.boundary_nodes).reshape(grid)

    # physical units
    U_phys, F_phys = scale_to_physical(U, F, para.voxel_size)

    # optional gradient smoothing
    if para.strain_smoothing > 0:
        Fg = F_phys.reshape(grid + (3, 3))
        for i in range(3):
            for j in range(3):
                Fg[..., i, j] = smooth_grid_field(Fg[..., i, j], para.strain_smoothing, valid_grid)
        F_phys = Fg.reshape(N, 3, 3)

    E = strain_tensor(F_phys, para.strain_type)
    princ = principal_strains(E)
    strain_valid = valid & np.all(np.isfinite(E), axis=(1, 2))
    if para.strain_edge_trim:
        strain_valid &= complete.ravel()

    return StrainResult(
        disp_u=U_phys[:, 0],
        disp_v=U_phys[:, 1],
        disp_w=U_phys[:, 2],
        F=F_phys,
        exx=E[:, 0, 0],
        eyy=E[:, 1, 1],
        ezz=E[:, 2, 2],
        exy=E[:, 0, 1],
        exz=E[:, 0, 2],
        eyz=E[:, 1, 2],
        principal=princ,
        max_shear=max_shear_strain(princ),
        von_mises=von_mises_strain(E),
        volumetric=volumetric_strain(E),
        det_F=det_deformation_gradient(F_phys),
        rotation_deg=polar_rotation_deg(F_phys),
        strain_valid=strain_valid,
        strain_type=para.strain_type,
        method=method,
    )
