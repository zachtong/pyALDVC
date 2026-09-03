"""Subset sampling stride: every k-th subset voxel per axis (k^3 fewer voxels per iteration).

Checks: Numba == NumPy reference with a stride, the strided Hessian and
statistics are those of the sampled set, the accuracy on a synthetic affine
deformation stays within the documented bounds, and the pipeline (with the
uncertainty model on the sampled set) runs end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.data_structures import STATUS_CONVERGED, P_from_UF
from al_dvc.core.pipeline import run_aldvc
from al_dvc.io.volume_ops import build_reference_bundle, normalize_volume, prepare_deformed
from al_dvc.mesh.grid_mesh import build_grid_axes, mesh_setup
from al_dvc.solver import numba_kernels as nk
from al_dvc.solver import reference_kernels as rk
from al_dvc.solver.interp_kernels import INTERP_CUBIC
from al_dvc.solver.uncertainty import subset_moment_matrix
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

SHAPE = (56, 60, 64)
F_TRUE = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])
T_TRUE = (0.7, -0.4, 0.3)


@pytest.fixture(scope="module")
def case():
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=17)
    fn = affine_displacement(F_TRUE, T_TRUE, centre)
    dfm = warp_volume_lagrangian(ref, fn)
    f, g = normalize_volume(ref), normalize_volume(dfm)
    bundle = build_reference_bundle(f, None)
    para = dvcpara_default(winsize=24, winstepsize=12, verbose=False)
    mesh = mesh_setup(*build_grid_axes(para.voi, SHAPE, para.winsize, para.winstepsize))
    coords = np.round(mesh.coordinates).astype(np.int64)
    U_true = np.stack(fn(mesh.coordinates[:, 0], mesh.coordinates[:, 1], mesh.coordinates[:, 2]), axis=-1).reshape(-1, 3)
    return {
        "ref": ref,
        "dfm": dfm,
        "bundle": bundle,
        "g": prepare_deformed(g, "cubic"),
        "coords": coords,
        "U_true": U_true,
        "half": (12, 12, 12),
    }


def _precompute(case, stride):
    b = case["bundle"]
    return nk.precompute_nodes(case["coords"], *case["half"], b.f, b.gx, b.gy, b.gz, b.mask, 0.5, 1e12, stride)


def test_strided_precompute_matches_reference_and_counts(case):
    b = case["bundle"]
    for stride in (1, 2, 3):
        H, L, mf, bf, nv, valid = _precompute(case, stride)
        Hn, Ln, mfn, bfn, nvn, validn = rk.precompute_nodes_np(
            case["coords"], *case["half"], b.f, b.gx, b.gy, b.gz, b.mask, 0.5, 1e12, stride
        )
        assert valid.all() and np.array_equal(valid, validn)
        np.testing.assert_allclose(H, Hn, rtol=1e-8)  # two summation orders of float32 products
        np.testing.assert_allclose(mf, mfn, rtol=1e-10)
        np.testing.assert_allclose(bf, bfn, rtol=1e-8)  # single-pass vs two-pass variance
        expected = ((2 * 12) // stride + 1) ** 3
        assert np.all(nv == expected) and np.all(nvn == expected)
    # the moment matrix of the sampled set has the same voxel count
    M = subset_moment_matrix(case["half"], 2)
    assert M[3, 3] == ((2 * 12) // 2 + 1) ** 3
    assert M[0, 0] == float(np.sum(np.arange(-12, 13, 2) ** 2)) * 13 * 13


def _run_12dof(case, stride, backend="numba"):
    b = case["bundle"]
    H, L, mf, bf, nv, valid = _precompute(case, stride)
    P0 = P_from_UF(case["U_true"] + 0.35, np.zeros((len(valid), 3, 3)))
    args = (case["coords"], P0, *case["half"], b.f, b.gx, b.gy, b.gz, b.mask, case["g"], INTERP_CUBIC)
    if backend == "numba":
        return nk.icgn_12dof_parallel(*args, L, mf, bf, valid, 1e-2, 1e-3, 100, 5, stride)
    return rk.icgn_12dof_batch_np(*args, H, mf, bf, valid, 1e-2, 1e-3, 100, 5, stride)


@pytest.mark.parametrize("stride", [2, 3])
def test_numba_matches_reference_with_stride(case, stride):
    P, it, st, z = _run_12dof(case, stride, "numba")
    Pn, itn, stn, zn = _run_12dof(case, stride, "numpy")
    assert np.array_equal(st, stn) and np.array_equal(it, itn)
    np.testing.assert_allclose(P, Pn, atol=1e-9)
    np.testing.assert_allclose(z, zn, atol=1e-9)


def test_3dof_numba_matches_reference_with_stride(case):
    b = case["bundle"]
    H, L, mf, bf, nv, valid = _precompute(case, 2)
    N = len(valid)
    F_fixed = np.tile(F_TRUE, (N, 1, 1))
    U_old = case["U_true"] + 0.15
    vdual = np.zeros((N, 3))
    args = (case["coords"], U_old, F_fixed, vdual, *case["half"], b.f, b.gx, b.gy, b.gz, b.mask, case["g"], INTERP_CUBIC)
    U, it, st, z = nk.icgn_3dof_parallel(*args, H, mf, bf, valid, 1e-3, 1e-2, 1e-3, 100, 5, 2)
    Un, itn, stn, zn = rk.icgn_3dof_batch_np(*args, H, mf, bf, valid, 1e-3, 1e-2, 1e-3, 100, 5, 2)
    assert np.array_equal(st, stn) and np.array_equal(it, itn)
    np.testing.assert_allclose(U, Un, atol=1e-9)
    assert np.mean(st == STATUS_CONVERGED) > 0.9


def test_stride_two_keeps_the_accuracy_on_a_smooth_field(case):
    P1, _, st1, _ = _run_12dof(case, 1)
    P2, _, st2, _ = _run_12dof(case, 2)
    ok = (st1 == STATUS_CONVERGED) & (st2 == STATUS_CONVERGED)
    assert ok.mean() > 0.95
    e1 = np.linalg.norm(P1[ok, 9:] - case["U_true"][ok], axis=1)
    e2 = np.linalg.norm(P2[ok, 9:] - case["U_true"][ok], axis=1)
    assert np.median(e2) < 0.02
    assert np.median(e2) < 3.0 * max(np.median(e1), 1e-3)


def test_pipeline_with_stride(case):
    para = dvcpara_default(winsize=24, winstepsize=12, search_radius=4, admm_max_iter=2, verbose=False, subset_stride=2)
    res = run_aldvc(para, [case["ref"], case["dfm"]])
    fr = res.result_disp[0]
    assert np.mean(fr.status == STATUS_CONVERGED) > 0.9
    err = np.linalg.norm(fr.U - case["U_true"], axis=1)
    valid = res.dvc_mesh.node_valid & (fr.status == STATUS_CONVERGED)
    assert np.median(err[valid]) < 0.02
    assert fr.U_std is not None and np.nanmedian(fr.U_std) < 0.05


def test_config_validation():
    with pytest.raises(ValueError):
        dvcpara_default(winsize=16, subset_stride=0)
    with pytest.raises(ValueError):
        dvcpara_default(winsize=16, subset_stride=5)  # fewer than 5 samples per axis
    p = dvcpara_default(winsize=32, subset_stride=4)
    assert p.subset_stride == 4
