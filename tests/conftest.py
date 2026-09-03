"""Shared fixtures: synthetic speckle volumes with exact ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.io.volume_ops import compute_gradients, normalize_volume
from al_dvc.mesh.grid_mesh import build_grid_axes, mesh_setup
from al_dvc.synthetic import (
    affine_displacement,
    evaluate_at_nodes,
    generate_speckle_volume,
    gradient_at_nodes,
    warp_volume_lagrangian,
)

SHAPE = (64, 72, 80)  # (nz, ny, nx), deliberately non-cubic
CENTRE = ((SHAPE[2] - 1) / 2, (SHAPE[1] - 1) / 2, (SHAPE[0] - 1) / 2)
F_AFFINE = np.array([[0.02, 0.01, 0.0], [0.0, -0.015, 0.005], [0.003, 0.0, 0.01]])
T_AFFINE = (1.7, -0.4, 2.3)


@pytest.fixture(scope="session")
def speckle() -> np.ndarray:
    return generate_speckle_volume(SHAPE, sigma=2.0, seed=7)


@pytest.fixture(scope="session")
def affine_pair(speckle):
    """(reference, deformed, displacement function) for an affine deformation."""
    disp = affine_displacement(F_AFFINE, T_AFFINE, CENTRE)
    g = warp_volume_lagrangian(speckle, disp)
    return speckle, g, disp


@pytest.fixture(scope="session")
def normalized_pair(affine_pair):
    f, g, disp = affine_pair
    fn = normalize_volume(f)
    gn = normalize_volume(g)
    gx, gy, gz = compute_gradients(fn)
    mask = np.ones(fn.shape, dtype=np.uint8)
    return {"f": fn, "g": gn, "gx": gx, "gy": gy, "gz": gz, "mask": mask, "disp": disp}


@pytest.fixture
def small_para():
    return dvcpara_default(winsize=16, winstepsize=8, search_radius=5, verbose=False)


@pytest.fixture
def small_mesh(small_para):
    x0, y0, z0 = build_grid_axes(small_para.voi, SHAPE, small_para.winsize, small_para.winstepsize)
    return mesh_setup(x0, y0, z0)


def gt_at(mesh, disp):
    return evaluate_at_nodes(disp, mesh.coordinates), gradient_at_nodes(disp, mesh.coordinates)


def interior_mask(mesh):
    return ~np.isin(np.arange(mesh.n_nodes), mesh.boundary_nodes)
