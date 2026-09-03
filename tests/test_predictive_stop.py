"""One-step look-ahead stopping rule of the IC-GN kernels.

The increment criterion discards the step that is already below ``dp_tol``, so
a node's last iteration only confirms convergence. When the steps contract and
the predicted next step ``r dp_k`` (``r = dp_k / dp_{k-1}``) is below ``dp_tol``,
the kernels apply ``dp_k`` and stop. Checks: Numba == NumPy reference, fewer
iterations on a smooth field, the same solution within a few ``dp_tol``, and
the pipeline switch.
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
from al_dvc.solver.local_icgn import precompute_local_context
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

SHAPE = (56, 60, 64)
F_TRUE = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])
T_TRUE = (0.7, -0.4, 0.3)
DP_TOL = 1e-3


@pytest.fixture(scope="module")
def case():
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=31)
    fn = affine_displacement(F_TRUE, T_TRUE, centre)
    dfm = warp_volume_lagrangian(ref, fn)
    para = dvcpara_default(winsize=24, winstepsize=12, verbose=False)
    f, g = normalize_volume(ref), normalize_volume(dfm)
    bundle = build_reference_bundle(f, None)
    mesh = mesh_setup(*build_grid_axes(para.voi, SHAPE, para.winsize, para.winstepsize))
    ctx = precompute_local_context(mesh, bundle, para)
    U_true = np.stack(fn(mesh.coordinates[:, 0], mesh.coordinates[:, 1], mesh.coordinates[:, 2]), axis=-1).reshape(-1, 3)
    return {
        "ref": ref,
        "dfm": dfm,
        "bundle": bundle,
        "g": prepare_deformed(g, "cubic"),
        "ctx": ctx,
        "para": para,
        "U_true": U_true,
    }


def _run12(case, predictive, backend="numba"):
    b, ctx = case["bundle"], case["ctx"]
    pattern, gain = ctx.noise_args(case["para"])
    P0 = P_from_UF(case["U_true"] + 0.3, np.zeros((ctx.n_nodes, 3, 3)))
    hx, hy, hz = ctx.half
    args = (ctx.coords_int, P0, hx, hy, hz, b.f, b.gx, b.gy, b.gz, b.mask, case["g"], INTERP_CUBIC)
    if backend == "numba":
        return nk.icgn_12dof_parallel(
            *args, ctx.L_all, ctx.meanf, ctx.bottomf, ctx.valid, 1e-2, DP_TOL, 100, 5, 1, ctx.H_all, pattern, gain, predictive
        )
    return rk.icgn_12dof_batch_np(
        *args, ctx.H_all, ctx.meanf, ctx.bottomf, ctx.valid, 1e-2, DP_TOL, 100, 5, 1, pattern, gain, predictive
    )


def _run3(case, predictive, backend="numba"):
    b, ctx = case["bundle"], case["ctx"]
    pattern, gain = ctx.noise_args(case["para"])
    n_full = float(pattern[9, 9])
    N = ctx.n_nodes
    F_fixed = np.tile(F_TRUE, (N, 1, 1))
    U_old = case["U_true"] + 0.15
    vdual = np.zeros((N, 3))
    hx, hy, hz = ctx.half
    args = (ctx.coords_int, U_old, F_fixed, vdual, hx, hy, hz, b.f, b.gx, b.gy, b.gz, b.mask, case["g"], INTERP_CUBIC)
    if backend == "numba":
        return nk.icgn_3dof_parallel(
            *args, ctx.H_all, ctx.meanf, ctx.bottomf, ctx.valid, 1e-3, 1e-2, DP_TOL, 100, 5, 1, n_full, gain, predictive
        )
    return rk.icgn_3dof_batch_np(
        *args, ctx.H_all, ctx.meanf, ctx.bottomf, ctx.valid, 1e-3, 1e-2, DP_TOL, 100, 5, 1, n_full, gain, predictive
    )


@pytest.mark.parametrize("predictive", [True, False])
def test_numba_matches_reference(case, predictive):
    P, it, st, z = _run12(case, predictive, "numba")
    Pn, itn, stn, zn = _run12(case, predictive, "numpy")
    assert np.array_equal(st, stn) and np.array_equal(it, itn)
    np.testing.assert_allclose(P, Pn, atol=1e-8)
    U, it3, st3, _ = _run3(case, predictive, "numba")
    Un, it3n, st3n, _ = _run3(case, predictive, "numpy")
    assert np.array_equal(st3, st3n) and np.array_equal(it3, it3n)
    np.testing.assert_allclose(U, Un, atol=1e-8)


def test_fewer_iterations_same_solution(case):
    P_on, it_on, st_on, _ = _run12(case, True)
    P_off, it_off, st_off, _ = _run12(case, False)
    ok = (st_on == STATUS_CONVERGED) & (st_off == STATUS_CONVERGED)
    assert ok.mean() > 0.95
    assert it_on[ok].mean() < it_off[ok].mean()
    assert np.max(np.abs(P_on[ok, 9:] - P_off[ok, 9:])) < 5 * DP_TOL
    e_on = np.linalg.norm(P_on[ok, 9:] - case["U_true"][ok], axis=1)
    e_off = np.linalg.norm(P_off[ok, 9:] - case["U_true"][ok], axis=1)
    assert np.median(e_on) <= np.median(e_off) + DP_TOL
    U_on, it3_on, st3_on, _ = _run3(case, True)
    U_off, it3_off, st3_off, _ = _run3(case, False)
    ok3 = (st3_on == STATUS_CONVERGED) & (st3_off == STATUS_CONVERGED)
    assert it3_on[ok3].mean() < it3_off[ok3].mean()
    assert np.max(np.abs(U_on[ok3] - U_off[ok3])) < 5 * DP_TOL


def test_pipeline_switch(case):
    base = dict(case["para"].__dict__, search_radius=4, admm_max_iter=2)
    res_on = run_aldvc(dvcpara_default(**base), [case["ref"], case["dfm"]])
    res_off = run_aldvc(dvcpara_default(**{**base, "icgn_predictive_stop": False}), [case["ref"], case["dfm"]])
    fr_on, fr_off = res_on.result_disp[0], res_off.result_disp[0]
    assert dvcpara_default().icgn_predictive_stop is True
    ok = (fr_on.status == STATUS_CONVERGED) & (fr_off.status == STATUS_CONVERGED)
    assert ok.mean() > 0.9
    assert fr_on.admm.local_info[0].n_iter[ok].mean() <= fr_off.admm.local_info[0].n_iter[ok].mean()
    assert np.median(np.linalg.norm(fr_on.U[ok] - fr_off.U[ok], axis=1)) < 5 * DP_TOL
