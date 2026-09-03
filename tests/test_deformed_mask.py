"""Deformed-frame masks: voxels masked out in the deformed volume drop out of the correlation."""

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.data_structures import STATUS_CONVERGED, STATUS_INVALID_SUBSET, STATUS_OUT_OF_BOUNDS
from al_dvc.core.pipeline import run_aldvc
from al_dvc.io.volume_ops import prefilter_bspline, prepare_deformed
from al_dvc.solver import numba_kernels as nk
from al_dvc.solver import reference_kernels as rk
from al_dvc.solver.interp_kernels import INTERP_CUBIC
from tests.conftest import F_AFFINE, gt_at, interior_mask

SLAB_X = 56  # the deformed volume is garbage for x >= SLAB_X


def _corrupt(g, x_from=SLAB_X, seed=3):
    """Replace the slab ``x >= x_from`` of ``g`` by noise of the same statistics; return ``(g_bad, mask)``."""
    gb = np.array(g, copy=True)
    rng = np.random.default_rng(seed)
    part = gb[:, :, x_from:]
    gb[:, :, x_from:] = rng.normal(loc=float(gb.mean()), scale=float(gb.std()), size=part.shape).astype(gb.dtype)
    mask = np.ones(g.shape, dtype=bool)
    mask[:, :, x_from:] = False
    return gb, mask


def test_prepare_deformed_masks_voxels(normalized_pair):
    g = normalized_pair["g"]
    mask = np.ones(g.shape, dtype=bool)
    mask[:, :, 50:] = False
    out = prepare_deformed(g, "cubic", mask)
    assert out is not g
    assert np.isnan(out[:, :, 50:]).all() and not np.isnan(out[:, :, :50]).any()
    assert not np.isnan(g).any()
    outb = prepare_deformed(g, "bspline", mask)
    assert np.isnan(outb[:, :, 50:]).all()
    assert np.allclose(outb[:, :, :50], prefilter_bspline(g)[:, :, :50])
    with pytest.raises(ValueError):
        prepare_deformed(g, "cubic", mask[1:])


def _setup(d, coords):
    half = (8, 8, 8)
    H_all, L_all, mf, bf, nv, valid = nk.precompute_nodes(coords, *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], 0.5, 1e12)
    u_gt = np.column_stack(d["disp"](coords[:, 0].astype(float), coords[:, 1].astype(float), coords[:, 2].astype(float)))
    P0 = np.zeros((coords.shape[0], 12))
    P0[:, 9:] = np.round(u_gt)
    return half, H_all, L_all, mf, bf, valid, u_gt, P0


def test_kernels_exclude_masked_deformed_voxels(normalized_pair):
    d = normalized_pair
    coords = np.array([[50, 36, 32], [30, 36, 32]], dtype=np.int64)  # first node overlaps the slab, second is far
    half, H_all, L_all, mf, bf, valid, u_gt, P0 = _setup(d, coords)
    g_bad, mask_g = _corrupt(d["g"])
    g_nan = prepare_deformed(g_bad, "cubic", mask_g)

    def run12(g):
        return nk.icgn_12dof_parallel(
            coords,
            P0.copy(),
            *half,
            d["f"],
            d["gx"],
            d["gy"],
            d["gz"],
            d["mask"],
            g,
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

    P_nan, it, st, zc = run12(g_nan)
    P_bad, it2, st2, zc2 = run12(g_bad)
    P_ok, it3, st3, zc3 = run12(d["g"])
    assert st[0] == STATUS_CONVERGED
    err_nan = np.abs(P_nan[0, 9:] - u_gt[0]).max()
    err_bad = np.abs(P_bad[0, 9:] - u_gt[0]).max()
    assert err_nan < 0.05
    assert st2[0] != STATUS_CONVERGED or err_bad > 3 * err_nan
    assert zc[0] > 0.9
    # the far node does not see the mask at all
    assert np.allclose(P_nan[1], P_ok[1], atol=1e-9) and it[1] == it3[1]
    # NumPy reference agrees with the kernel on the masked volume
    for n in range(2):
        Pr, itr, str_, zcr = rk.icgn_12dof_np(
            P0[n],
            coords[n],
            half,
            d["f"],
            d["gx"],
            d["gy"],
            d["gz"],
            d["mask"],
            g_nan,
            INTERP_CUBIC,
            H_all[n],
            mf[n],
            bf[n],
            1e-2,
            1e-3,
            100,
            5,
        )
        assert str_ == st[n] and itr == it[n]
        assert np.allclose(Pr, P_nan[n], atol=1e-9)
        assert np.isclose(zcr, zc[n], atol=1e-9)
    # 3-DOF pass on the masked volume: kernel and reference agree, node converges near the truth
    U, it4, st4, zc4 = nk.icgn_3dof_parallel(
        coords,
        np.round(u_gt),
        np.broadcast_to(F_AFFINE, (2, 3, 3)).copy(),
        np.zeros((2, 3)),
        *half,
        d["f"],
        d["gx"],
        d["gy"],
        d["gz"],
        d["mask"],
        g_nan,
        INTERP_CUBIC,
        H_all,
        mf,
        bf,
        valid,
        1e-3,
        1e-2,
        1e-3,
        100,
        5,
    )
    assert st4[0] == STATUS_CONVERGED and np.abs(U[0] - u_gt[0]).max() < 0.06
    Ur, itr, str_, zcr = rk.icgn_3dof_np(
        np.round(u_gt[0]),
        np.array(F_AFFINE, dtype=np.float64),
        np.zeros(3),
        coords[0],
        half,
        d["f"],
        d["gx"],
        d["gy"],
        d["gz"],
        d["mask"],
        g_nan,
        INTERP_CUBIC,
        H_all[0],
        mf[0],
        bf[0],
        1e-3,
        1e-2,
        1e-3,
        100,
        5,
    )
    assert str_ == st4[0] and itr == it4[0] and np.allclose(Ur, U[0], atol=1e-9)
    # a subset lying entirely in the masked slab is reported, not solved
    coords_in = np.array([[70, 36, 32]], dtype=np.int64)
    half, H_all, L_all, mf, bf, valid, u_gt_in, P0_in = _setup(d, coords_in)
    P_in, it_in, st_in, zc_in = nk.icgn_12dof_parallel(
        coords_in,
        P0_in.copy(),
        *half,
        d["f"],
        d["gx"],
        d["gy"],
        d["gz"],
        d["mask"],
        g_nan,
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
    assert st_in[0] in (STATUS_INVALID_SUBSET, STATUS_OUT_OF_BOUNDS) and np.isnan(zc_in[0])


def test_pipeline_uses_deformed_mask(affine_pair):
    f, g, disp = affine_pair
    g_bad, mask_g = _corrupt(g)
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=5, verbose=False, use_global_step=False)
    r_mask = run_aldvc(para, [f, g_bad], masks=[np.ones(f.shape, dtype=bool), mask_g], compute_strain=False)
    r_none = run_aldvc(para, [f, g_bad], compute_strain=False)
    r_clean = run_aldvc(para, [f, g], compute_strain=False)
    mesh = r_mask.dvc_mesh
    U_gt, _ = gt_at(mesh, disp)
    x0 = mesh.coordinates[:, 0]
    inner = interior_mask(mesh)
    near = inner & (x0 >= SLAB_X - 12) & (x0 < SLAB_X - 4)  # x0 = 47: subsets lose less than half their voxels
    boundary = inner & (x0 >= SLAB_X - 4) & (x0 < SLAB_X)  # x0 = 55: more than half masked -> reported, not solved
    far = inner & (x0 < SLAB_X - 24)
    assert near.sum() >= 10 and boundary.sum() >= 10 and far.sum() >= 10
    fm, fn_, fc = r_mask.result_disp[0], r_none.result_disp[0], r_clean.result_disp[0]
    assert np.all(fm.status[boundary] == STATUS_INVALID_SUBSET)
    conv_m = near & (fm.status == STATUS_CONVERGED)
    assert conv_m.sum() >= 0.8 * near.sum()
    rmse_mask = np.sqrt(np.mean((fm.U - U_gt)[conv_m] ** 2))
    rmse_clean = np.sqrt(np.mean((fc.U - U_gt)[near] ** 2))
    assert rmse_mask < 0.03
    assert rmse_mask < 2.5 * rmse_clean + 0.01
    # without the mask the boundary layer is silently wrong
    rmse_none_boundary = np.sqrt(np.mean((fn_.U - U_gt)[boundary] ** 2))
    assert rmse_none_boundary > 0.1
    # far from the slab the mask changes nothing
    assert np.allclose(fm.U[far], fc.U[far], atol=1e-6)
    assert np.all(np.isfinite(fm.U_std[conv_m])) and np.all(np.isnan(fm.U_std[boundary]))
