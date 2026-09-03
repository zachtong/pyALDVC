"""Noise-corrected Hessian in the IC-GN kernels.

The stored Hessian is inflated by the reference-gradient noise, which makes the
Gauss-Newton steps too short on noisy data (linear convergence, ~16 iterations
per node at SNR ~ 5). Subtracting the expected inflation once the node is in its
fine-convergence phase keeps the fixed point and shortens the path. Checks:
Numba == NumPy reference with the correction, fewer iterations and the same
solution on noisy data, no change at all on clean data, pipeline default on.
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


def _case(noise: float):
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=23)
    fn = affine_displacement(F_TRUE, T_TRUE, centre)
    dfm = warp_volume_lagrangian(ref, fn)
    if noise > 0:
        rng = np.random.default_rng(5)
        ref = ref + rng.normal(0, noise, ref.shape)
        dfm = dfm + rng.normal(0, noise, dfm.shape)
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


@pytest.fixture(scope="module")
def noisy():
    return _case(0.03)


@pytest.fixture(scope="module")
def clean():
    return _case(0.0)


def _run(case, gain: float, backend="numba", start=0.3):
    b, ctx = case["bundle"], case["ctx"]
    pattern, full_gain = ctx.noise_args(case["para"])
    gain = full_gain if gain > 0 else 0.0
    P0 = P_from_UF(case["U_true"] + start, np.zeros((ctx.n_nodes, 3, 3)))
    hx, hy, hz = ctx.half
    args = (ctx.coords_int, P0, hx, hy, hz, b.f, b.gx, b.gy, b.gz, b.mask, case["g"], INTERP_CUBIC)
    if backend == "numba":
        return nk.icgn_12dof_parallel(
            *args, ctx.L_all, ctx.meanf, ctx.bottomf, ctx.valid, 1e-2, 1e-3, 100, 5, 1, ctx.H_all, pattern, gain
        )
    return rk.icgn_12dof_batch_np(*args, ctx.H_all, ctx.meanf, ctx.bottomf, ctx.valid, 1e-2, 1e-3, 100, 5, 1, pattern, gain)


def test_numba_matches_reference_with_the_correction(noisy):
    P, it, st, z = _run(noisy, 1.0, "numba")
    Pn, itn, stn, zn = _run(noisy, 1.0, "numpy")
    assert np.array_equal(st, stn) and np.array_equal(it, itn)
    np.testing.assert_allclose(P, Pn, atol=1e-8)


def test_fewer_iterations_same_solution_on_noisy_data(noisy):
    P_plain, it_plain, st_plain, _ = _run(noisy, 0.0)
    P_corr, it_corr, st_corr, _ = _run(noisy, 1.0)
    ok = (st_plain == STATUS_CONVERGED) & (st_corr == STATUS_CONVERGED)
    assert ok.mean() > 0.95
    assert it_corr[ok].mean() < 0.75 * it_plain[ok].mean()
    assert np.max(np.abs(P_corr[ok, 9:] - P_plain[ok, 9:])) < 2e-2  # same fixed point, different stopping point
    e_plain = np.linalg.norm(P_plain[ok, 9:] - noisy["U_true"][ok], axis=1)
    e_corr = np.linalg.norm(P_corr[ok, 9:] - noisy["U_true"][ok], axis=1)
    assert np.median(e_corr) < 1.1 * np.median(e_plain) + 1e-3


def test_no_effect_on_clean_data(clean):
    P_plain, it_plain, st_plain, _ = _run(clean, 0.0)
    P_corr, it_corr, st_corr, _ = _run(clean, 1.0)
    assert np.array_equal(it_plain, it_corr) and np.array_equal(st_plain, st_corr)
    np.testing.assert_allclose(P_corr, P_plain, atol=1e-6)


def test_3dof_numba_matches_reference_with_the_correction(noisy):
    b, ctx = noisy["bundle"], noisy["ctx"]
    pattern, gain = ctx.noise_args(noisy["para"])
    n_full = float(pattern[9, 9])
    N = ctx.n_nodes
    F_fixed = np.tile(F_TRUE, (N, 1, 1))
    U_old = noisy["U_true"] + 0.1
    vdual = np.zeros((N, 3))
    hx, hy, hz = ctx.half
    args = (ctx.coords_int, U_old, F_fixed, vdual, hx, hy, hz, b.f, b.gx, b.gy, b.gz, b.mask, noisy["g"], INTERP_CUBIC)
    U, it, st, z = nk.icgn_3dof_parallel(
        *args, ctx.H_all, ctx.meanf, ctx.bottomf, ctx.valid, 1e-3, 1e-2, 1e-3, 100, 5, 1, n_full, gain
    )
    Un, itn, stn, zn = rk.icgn_3dof_batch_np(
        *args, ctx.H_all, ctx.meanf, ctx.bottomf, ctx.valid, 1e-3, 1e-2, 1e-3, 100, 5, 1, n_full, gain
    )
    assert np.array_equal(st, stn) and np.array_equal(it, itn)
    np.testing.assert_allclose(U, Un, atol=1e-8)
    U0, it0, st0, _ = nk.icgn_3dof_parallel(
        *args, ctx.H_all, ctx.meanf, ctx.bottomf, ctx.valid, 1e-3, 1e-2, 1e-3, 100, 5, 1, n_full, 0.0
    )
    ok = (st == STATUS_CONVERGED) & (st0 == STATUS_CONVERGED)
    assert it[ok].mean() <= it0[ok].mean()
    assert np.max(np.abs(U[ok] - U0[ok])) < 2e-2


def test_pipeline_default_and_switch(noisy):
    para_on = dvcpara_default(winsize=24, winstepsize=12, search_radius=4, admm_max_iter=2, verbose=False)
    assert para_on.icgn_noise_hessian is True
    res_on = run_aldvc(para_on, [noisy["ref"], noisy["dfm"]])
    res_off = run_aldvc(dvcpara_default(**{**para_on.__dict__, "icgn_noise_hessian": False}), [noisy["ref"], noisy["dfm"]])
    fr_on, fr_off = res_on.result_disp[0], res_off.result_disp[0]
    assert np.mean(fr_on.status == STATUS_CONVERGED) > 0.9
    it_on = fr_on.admm.local_info[0].n_iter
    it_off = fr_off.admm.local_info[0].n_iter
    assert it_on.mean() < it_off.mean()
    ok = (fr_on.status == STATUS_CONVERGED) & (fr_off.status == STATUS_CONVERGED)
    assert np.median(np.linalg.norm(fr_on.U[ok] - fr_off.U[ok], axis=1)) < 5e-3
