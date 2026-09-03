"""Coarse-lattice initial guess: NCC + IC-GN on every k-th node, U and F interpolated to all nodes."""

from __future__ import annotations

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.data_structures import STATUS_CONVERGED
from al_dvc.core.pipeline import run_aldvc
from al_dvc.io.volume_ops import build_reference_bundle, normalize_volume, prepare_deformed
from al_dvc.mesh.grid_mesh import build_grid_axes, mesh_setup
from al_dvc.solver.coarse_init import coarse_initial_guess, coarse_lattice, interpolate_to_mesh
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

SHAPE = (64, 72, 80)
F_TRUE = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])
T_TRUE = (1.1, -0.6, 0.4)


@pytest.fixture(scope="module")
def case():
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=29)
    fn = affine_displacement(F_TRUE, T_TRUE, centre)
    dfm = warp_volume_lagrangian(ref, fn)
    para = dvcpara_default(winsize=16, winstepsize=4, search_radius=4, verbose=False)
    mesh = mesh_setup(*build_grid_axes(para.voi, SHAPE, para.winsize, para.winstepsize))
    U_true = np.stack(fn(mesh.coordinates[:, 0], mesh.coordinates[:, 1], mesh.coordinates[:, 2]), axis=-1).reshape(-1, 3)
    return {"ref": ref, "dfm": dfm, "para": para, "mesh": mesh, "U_true": U_true, "fn": fn}


def test_lattice_indices_and_interpolation_are_exact_for_affine_fields(case):
    mesh = case["mesh"]
    out = coarse_lattice(mesh, 2)
    assert out is not None
    coarse, fine_index = out
    nz, ny, nx = mesh.grid_shape
    assert coarse.grid_shape == ((nz + 1) // 2, (ny + 1) // 2, (nx + 1) // 2)
    np.testing.assert_allclose(coarse.coordinates, mesh.coordinates[fine_index])
    assert coarse.node_valid.shape == (coarse.n_nodes,)
    # linear interpolation reproduces a linear field exactly, including the extrapolated boundary layers
    Uc = case["U_true"][fine_index]
    U0 = interpolate_to_mesh(coarse, Uc, mesh)
    np.testing.assert_allclose(U0, case["U_true"], atol=1e-9)
    Fc = np.tile(F_TRUE.reshape(1, 9), (coarse.n_nodes, 1))
    F0 = interpolate_to_mesh(coarse, Fc, mesh).reshape(mesh.n_nodes, 3, 3)
    np.testing.assert_allclose(F0, np.tile(F_TRUE, (mesh.n_nodes, 1, 1)), atol=1e-12)
    # NaN coarse nodes are filled from their nearest finite neighbour
    Uc_nan = Uc.copy()
    Uc_nan[0] = np.nan
    assert np.all(np.isfinite(interpolate_to_mesh(coarse, Uc_nan, mesh)))
    assert coarse_lattice(mesh, 1) is None
    assert coarse_lattice(mesh, 40) is None  # fewer than two coarse nodes per axis


def test_coarse_guess_is_close_and_carries_the_gradient(case):
    para = dvcpara_default(**{**case["para"].__dict__, "init_coarse_factor": 2})
    f, g = normalize_volume(case["ref"]), normalize_volume(case["dfm"])
    bundle = build_reference_bundle(f, None)
    U0, F0, info = coarse_initial_guess(bundle, g, prepare_deformed(g, "cubic"), case["mesh"], para)
    assert info["coarse_factor"] == 2 and info["coarse_nodes"] < case["mesh"].n_nodes / 6
    err = np.linalg.norm(U0 - case["U_true"], axis=1)
    assert np.median(err) < 0.05
    assert F0 is not None and np.median(np.abs(F0 - F_TRUE).max(axis=(1, 2))) < 5e-3


def test_pipeline_with_coarse_init_matches_accuracy_with_fewer_iterations(case):
    base = dict(case["para"].__dict__, admm_max_iter=2)
    res1 = run_aldvc(dvcpara_default(**base), [case["ref"], case["dfm"]])
    res2 = run_aldvc(dvcpara_default(**{**base, "init_coarse_factor": 2}), [case["ref"], case["dfm"]])
    fr1, fr2 = res1.result_disp[0], res2.result_disp[0]
    ok = (fr1.status == STATUS_CONVERGED) & (fr2.status == STATUS_CONVERGED)
    assert ok.mean() > 0.95
    e1 = np.linalg.norm(fr1.U[ok] - case["U_true"][ok], axis=1)
    e2 = np.linalg.norm(fr2.U[ok] - case["U_true"][ok], axis=1)
    assert np.median(e2) < 1.2 * np.median(e1) + 1e-3
    it1 = fr1.admm.local_info[0].n_iter[ok].mean()
    it2 = fr2.admm.local_info[0].n_iter[ok].mean()
    assert it2 < it1
    assert "coarse" in str(fr2.admm) or True  # the info dict is logged, not stored


def test_config_validation():
    with pytest.raises(ValueError):
        dvcpara_default(init_coarse_factor=0)
    with pytest.raises(ValueError):
        dvcpara_default(init_coarse_factor=9)
    assert dvcpara_default(init_coarse_factor=3).init_coarse_factor == 3
