"""Non-cubic subsets: per-axis half-widths through the kernels, the pipeline and the GUI."""

import os
from dataclasses import replace

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.data_structures import STATUS_CONVERGED
from al_dvc.core.pipeline import run_aldvc
from al_dvc.solver import numba_kernels as nk
from al_dvc.solver import reference_kernels as rk
from al_dvc.solver.interp_kernels import INTERP_CUBIC
from tests.conftest import F_AFFINE, gt_at, interior_mask

HALF = (5, 8, 11)  # (hx, hy, hz): a subset of 11 x 17 x 23 voxels
WINSIZE = (12, 16, 24)  # x, y, z: 13 x 17 x 25 voxels


def _rmse(a, b, sel):
    e = (a - b)[sel]
    return np.sqrt(np.mean(e**2, axis=0))


def test_icgn_kernels_accept_per_axis_half_widths(normalized_pair):
    d = normalized_pair
    coords = np.array([[30, 32, 28], [45, 20, 40], [24, 44, 36]], dtype=np.int64)
    H_all, L_all, mf, bf, nv, valid = nk.precompute_nodes(coords, *HALF, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], 0.5, 1e12)
    assert np.all(valid) and np.all(nv == (2 * HALF[0] + 1) * (2 * HALF[1] + 1) * (2 * HALF[2] + 1))
    u_gt = np.column_stack(d["disp"](coords[:, 0].astype(float), coords[:, 1].astype(float), coords[:, 2].astype(float)))
    P0 = np.zeros((3, 12))
    P0[:, 9:] = np.round(u_gt)
    P, it, st, zc = nk.icgn_12dof_parallel(
        coords,
        P0.copy(),
        *HALF,
        d["f"],
        d["gx"],
        d["gy"],
        d["gz"],
        d["mask"],
        d["g"],
        INTERP_CUBIC,
        L_all,
        mf,
        bf,
        valid,
        1e-2,
        1e-3,
        100,
        5,
    )
    assert np.all(st == STATUS_CONVERGED)
    assert np.all(np.abs(P[:, 9:] - u_gt) < 0.03)
    assert np.all(np.abs(P[:, :9] - F_AFFINE.ravel()) < 0.01)
    for n in range(3):  # the NumPy reference takes the same per-axis half-widths and agrees to round-off
        Pr, itr, str_, zcr = rk.icgn_12dof_np(
            P0[n],
            coords[n],
            HALF,
            d["f"],
            d["gx"],
            d["gy"],
            d["gz"],
            d["mask"],
            d["g"],
            INTERP_CUBIC,
            H_all[n],
            mf[n],
            bf[n],
            1e-2,
            1e-3,
            100,
            5,
        )
        assert itr == it[n] and str_ == st[n]
        assert np.allclose(P[n], Pr, atol=1e-10)
        assert np.isclose(zc[n], zcr, atol=1e-10)


@pytest.mark.parametrize("winsize", [WINSIZE, WINSIZE[::-1]])
def test_pipeline_recovers_the_affine_field_with_a_flat_subset(affine_pair, winsize):
    f, g, disp = affine_pair
    para = dvcpara_default(winsize=winsize, winstepsize=8, search_radius=5, verbose=False)
    assert para.winsize == tuple(winsize)
    res = run_aldvc(para, [f, g], compute_strain=False)
    mesh = res.dvc_mesh
    fr = res.result_disp[0]
    U_gt, F_gt = gt_at(mesh, disp)
    sel = interior_mask(mesh)
    assert np.all(_rmse(fr.U, U_gt, sel) < 0.03)
    assert np.sqrt(np.mean((fr.F - F_gt)[sel] ** 2)) < 5e-3
    assert np.mean(fr.status == STATUS_CONVERGED) > 0.98
    # the lattice keeps every subset inside the volume: the first node sits at least a half-width from the border
    assert mesh.x0.min() >= winsize[0] // 2 and mesh.z0.min() >= winsize[2] // 2


def test_config_warns_when_the_step_exceeds_the_smallest_axis():
    with pytest.warns(UserWarning, match="exceeds winsize"):
        dvcpara_default(winsize=(12, 16, 24), winstepsize=14)
    para = dvcpara_default(winsize=(12, 16, 24), winstepsize=(6, 8, 12))
    assert replace(para, winsize=(24, 24, 24)).winsize == (24, 24, 24)


# --------------------------------------------------------------------------- GUI
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from al_dvc.gui.app import create_application

    return create_application(["pytest"])


def test_param_panel_locks_and_unlocks_the_subset_axes(qapp):
    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow

    window = MainWindow()
    panel = window.param_panel
    assert panel.winsize_lock.isChecked() and len(panel.winsize_axes) == 3
    panel.winsize.setValue(25)  # locked: x drives y and z
    QApplication.processEvents()
    assert [w.value() for w in panel.winsize_axes] == [25, 25, 25]
    assert window.state.para.winsize == (24, 24, 24)
    panel.winsize_lock.setChecked(False)
    panel.winsize_axes[2].setValue(33)  # z alone
    QApplication.processEvents()
    assert window.state.para.winsize == (24, 24, 32)
    panel.winsize_axes[1].setValue(16)  # even values round up to the next odd span
    QApplication.processEvents()
    assert panel.winsize_axes[1].value() == 17 and window.state.para.winsize == (24, 16, 32)
    panel.winsize_lock.setChecked(True)  # locking makes it cubic again from x
    QApplication.processEvents()
    assert window.state.para.winsize == (24, 24, 24)
    # a non-cubic subset arriving from outside (a session, a script) unlocks the axes on refresh
    window.state.set_params(winsize=(16, 24, 32))
    QApplication.processEvents()
    assert not panel.winsize_lock.isChecked()
    assert [w.value() for w in panel.winsize_axes] == [17, 25, 33]
    window.close()
