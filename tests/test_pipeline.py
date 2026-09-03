"""Integration tests: run_aldvc against synthetic ground truth."""

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.data_structures import FrameSchedule
from al_dvc.core.pipeline import run_aldvc
from al_dvc.synthetic import (
    add_noise,
    affine_displacement,
    generate_speckle_volume,
    rotation_displacement,
    sinusoidal_displacement,
    warp_volume_lagrangian,
)

from tests.conftest import CENTRE, F_AFFINE, SHAPE, gt_at, interior_mask


def _rmse(a, b, sel):
    e = (a - b)[sel]
    return np.sqrt(np.mean(e**2, axis=0))


def test_affine_local_only(affine_pair, small_para):
    f, g, disp = affine_pair
    from dataclasses import replace

    para = replace(small_para, use_global_step=False)
    res = run_aldvc(para, [f, g])
    mesh = res.dvc_mesh
    fr = res.result_disp[0]
    U_gt, F_gt = gt_at(mesh, disp)
    sel = interior_mask(mesh)
    assert np.all(_rmse(fr.U, U_gt, sel) < 0.02)
    assert np.sqrt(np.mean((fr.F - F_gt)[sel] ** 2)) < 4e-3
    assert np.mean(fr.status == 0) > 0.98
    assert fr.admm is None and fr.U_accum is not None
    assert np.allclose(fr.U_accum, fr.U)


@pytest.mark.parametrize("method", ["fem", "fd"])
def test_affine_aldvc(affine_pair, small_para, method):
    f, g, disp = affine_pair
    from dataclasses import replace

    para = replace(small_para, subpb2_method=method)
    res = run_aldvc(para, [f, g])
    mesh = res.dvc_mesh
    fr = res.result_disp[0]
    U_gt, F_gt = gt_at(mesh, disp)
    sel = interior_mask(mesh)
    assert np.all(_rmse(fr.U, U_gt, sel) < 0.02)
    assert np.sqrt(np.mean((fr.F - F_gt)[sel] ** 2)) < 2e-3
    assert fr.admm is not None and fr.admm.n_steps >= 2
    assert fr.admm.beta > 0
    # ADMM must not be worse than the local result on the gradient
    assert np.sqrt(np.mean((fr.F - F_gt)[sel] ** 2)) <= np.sqrt(np.mean((fr.F_local - F_gt)[sel] ** 2)) + 1e-4
    sr = res.result_strain[0]
    v = sr.strain_valid
    assert abs(np.mean(sr.exx[v]) - F_AFFINE[0, 0]) < 5e-4
    assert np.std(sr.exx[v]) < 3e-3
    assert "total" in res.timings


def test_noise_robustness(affine_pair, small_para):
    """SNR ~ 3 (speckle std 0.06, noise 0.02): sub-0.05 voxel accuracy, and Gaussian
    pre-smoothing recovers most of the noise-free accuracy."""
    from dataclasses import replace

    f, g, disp = affine_pair
    fn = add_noise(f, 0.02, 1)
    gn = add_noise(g, 0.02, 2)
    res = run_aldvc(small_para, [fn, gn])
    mesh = res.dvc_mesh
    U_gt, _ = gt_at(mesh, disp)
    sel = interior_mask(mesh)
    rmse_raw = _rmse(res.result_disp[0].U, U_gt, sel)
    assert np.all(rmse_raw < 0.06)
    assert np.nanmedian(res.result_disp[0].zncc) > 0.8
    res2 = run_aldvc(replace(small_para, prefilter_sigma=0.8), [fn, gn])
    rmse_pf = _rmse(res2.result_disp[0].U, U_gt, sel)
    assert np.all(rmse_pf < 0.04)
    assert np.nanmedian(res2.result_disp[0].zncc) > np.nanmedian(res.result_disp[0].zncc)


def test_large_translation_pyramid(speckle):
    shift = (9.4, -7.6, 5.2)
    g = warp_volume_lagrangian(speckle, affine_displacement(None, shift))
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=3, verbose=False, use_global_step=False)
    res = run_aldvc(para, [speckle, g])
    U = res.result_disp[0].U
    sel = interior_mask(res.dvc_mesh)
    assert np.all(np.abs(U[sel] - np.array(shift)) < 0.05)


def test_rotation(speckle):
    disp = rotation_displacement(4.0, "z", CENTRE)
    g = warp_volume_lagrangian(speckle, disp)
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=5, verbose=False)
    res = run_aldvc(para, [speckle, g])
    mesh = res.dvc_mesh
    U_gt, _ = gt_at(mesh, disp)
    sel = interior_mask(mesh)
    assert np.all(_rmse(res.result_disp[0].U, U_gt, sel) < 0.03)
    rot = res.result_strain[0].rotation_deg
    assert abs(np.nanmedian(rot[res.result_strain[0].strain_valid]) - 4.0) < 0.2


@pytest.mark.slow
def test_sinusoidal_field(speckle):
    """Heterogeneous field: wavelength 80 voxels vs subset 16 (first-order subset
    shape functions carry a curvature bias ~0.05 voxel here; see the validation report)."""
    disp = sinusoidal_displacement(1.0, 80.0, CENTRE)
    g = warp_volume_lagrangian(speckle, disp)
    para = dvcpara_default(winsize=16, winstepsize=4, search_radius=4, verbose=False)
    res = run_aldvc(para, [speckle, g])
    mesh = res.dvc_mesh
    U_gt, _ = gt_at(mesh, disp)
    sel = interior_mask(mesh)
    assert np.all(_rmse(res.result_disp[0].U, U_gt, sel) < 0.08)
    sr = res.result_strain[0]
    assert sr.strain_valid.sum() > 100 and np.nanmax(np.abs(sr.exy)) > 0.02


def test_mask_cylinder(affine_pair, small_para):
    f, g, disp = affine_pair
    nz, ny, nx = f.shape
    Z, Y, X = np.mgrid[0:nz, 0:ny, 0:nx]
    mask = (X - (nx - 1) / 2) ** 2 + (Y - (ny - 1) / 2) ** 2 < (min(nx, ny) * 0.4) ** 2
    res = run_aldvc(small_para, [f, g], masks=[mask, mask])
    mesh = res.dvc_mesh
    assert 0 < mesh.node_valid.sum() < mesh.n_nodes
    U_gt, _ = gt_at(mesh, disp)
    valid = mesh.node_valid & interior_mask(mesh)
    assert valid.sum() > 20
    assert np.all(_rmse(res.result_disp[0].U, U_gt, valid) < 0.03)
    sr = res.result_strain[0]
    assert sr.strain_valid.sum() > 0 and not sr.strain_valid[~mesh.node_valid].any()


def test_incremental_vs_accumulative_three_frames(speckle, small_para):
    from dataclasses import replace

    disp1 = affine_displacement(0.5 * F_AFFINE, (0.8, -0.2, 1.1), CENTRE)
    disp2 = affine_displacement(F_AFFINE, (1.6, -0.4, 2.2), CENTRE)
    g1 = warp_volume_lagrangian(speckle, disp1)
    g2 = warp_volume_lagrangian(speckle, disp2)
    acc = run_aldvc(replace(small_para, reference_mode="accumulative"), [speckle, g1, g2], compute_strain=False)
    inc = run_aldvc(replace(small_para, reference_mode="incremental"), [speckle, g1, g2], compute_strain=False)
    assert acc.frame_schedule.ref_indices == (0, 0)
    assert inc.frame_schedule.ref_indices == (0, 1)
    mesh = acc.dvc_mesh
    U_gt2, _ = gt_at(mesh, disp2)
    sel = interior_mask(mesh)
    assert np.all(_rmse(acc.result_disp[1].U_accum, U_gt2, sel) < 0.03)
    assert np.all(_rmse(inc.result_disp[1].U_accum, U_gt2, sel) < 0.05)
    # incremental pair 2 is the displacement between frames 1 and 2 (smaller than cumulative)
    assert np.linalg.norm(inc.result_disp[1].U) < np.linalg.norm(inc.result_disp[1].U_accum)
    sched = run_aldvc(replace(small_para, frame_schedule=FrameSchedule((0, 0))), [speckle, g1, g2], compute_strain=False)
    assert sched.frame_schedule.ref_indices == (0, 0)


@pytest.mark.parametrize("method", ["zero", "ncc", "previous"])
def test_init_guess_methods(affine_pair, small_para, method):
    from dataclasses import replace

    f, g, disp = affine_pair
    para = replace(small_para, init_guess_method=method, use_global_step=False)
    res = run_aldvc(para, [f, g])
    mesh = res.dvc_mesh
    U_gt, _ = gt_at(mesh, disp)
    sel = interior_mask(mesh)
    assert np.all(_rmse(res.result_disp[0].U, U_gt, sel) < 0.03)


def test_interp_methods_and_dual_reset(affine_pair, small_para):
    from dataclasses import replace

    f, g, disp = affine_pair
    for kw in (dict(interp_method="bspline"), dict(interp_method="linear"), dict(dual_update="reset")):
        res = run_aldvc(replace(small_para, **kw), [f, g], compute_strain=False)
        mesh = res.dvc_mesh
        U_gt, _ = gt_at(mesh, disp)
        sel = interior_mask(mesh)
        tol = 0.06 if kw.get("interp_method") == "linear" else 0.02
        assert np.all(_rmse(res.result_disp[0].U, U_gt, sel) < tol), kw


def test_numpy_backend_matches_numba(affine_pair):
    from dataclasses import replace

    f, g, disp = affine_pair
    para = dvcpara_default(winsize=16, winstepsize=16, search_radius=4, verbose=False, admm_max_iter=2)
    a = run_aldvc(para, [f, g], compute_strain=False)
    b = run_aldvc(replace(para, backend="numpy"), [f, g], compute_strain=False)
    assert np.allclose(a.result_disp[0].U, b.result_disp[0].U, atol=1e-7)
    assert np.allclose(a.result_disp[0].F, b.result_disp[0].F, atol=1e-7)


def test_stop_fn_returns_partial_results(speckle, small_para):
    g1 = warp_volume_lagrangian(speckle, affine_displacement(None, (0.5, 0.5, 0.5)))
    g2 = warp_volume_lagrangian(speckle, affine_displacement(None, (1.0, 1.0, 1.0)))
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3  # cancel during the second frame

    res = run_aldvc(small_para, [speckle, g1, g2], stop_fn=stop, compute_strain=False)
    assert res.stopped_early and res.n_frames == 1


def test_input_validation(speckle):
    para = dvcpara_default(winsize=16, winstepsize=8)
    with pytest.raises(ValueError):
        run_aldvc(para, [speckle])
    with pytest.raises(ValueError):
        run_aldvc(dvcpara_default(winsize=60), [speckle[:40, :40, :40], speckle[:40, :40, :40]])
    with pytest.raises(ValueError):
        run_aldvc(para, [speckle, speckle[:-1]])
