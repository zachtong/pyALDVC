"""The parameter-increment tolerance ``icgn_dp_tol`` controls the IC-GN precision."""

import numpy as np

from al_dvc.core.data_structures import STATUS_CONVERGED
from al_dvc.solver import numba_kernels as nk
from al_dvc.solver.interp_kernels import INTERP_CUBIC

DP_TOLS = (1e-1, 1e-2, 1e-3, 1e-4)


def _setup(d):
    coords = np.array([[40, 36, 32], [24, 48, 30], [56, 20, 34]], dtype=np.int64)
    half = (8, 8, 8)
    H_all, L_all, mf, bf, nv, valid = nk.precompute_nodes(coords, *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], 0.5, 1e12)
    u_gt = np.column_stack(d["disp"](coords[:, 0].astype(float), coords[:, 1].astype(float), coords[:, 2].astype(float)))
    return coords, half, H_all, L_all, mf, bf, valid, u_gt


def test_dp_tol_controls_precision_12dof(normalized_pair):
    d = normalized_pair
    coords, half, H_all, L_all, mf, bf, valid, u_gt = _setup(d)
    P0 = np.zeros((3, 12))
    P0[:, 9:] = np.round(u_gt)
    out = {}
    for dp in DP_TOLS:
        P, it, st, zc = nk.icgn_12dof_parallel(
            coords,
            P0.copy(),
            *half,
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
            dp,
            100,
            5,
        )
        assert np.all(st == STATUS_CONVERGED)
        out[dp] = (P, it)
    # a tighter increment tolerance never needs fewer iterations
    for a, b in zip(DP_TOLS[:-1], DP_TOLS[1:]):
        assert np.all(out[b][1] >= out[a][1])
    assert np.any(out[1e-4][1] > out[1e-1][1])
    err = {dp: np.abs(out[dp][0][:, 9:] - u_gt).max() for dp in DP_TOLS}
    assert err[1e-4] <= err[1e-1] + 1e-9
    assert err[1e-4] < 0.01
    # the two tightest settings agree to the tolerance itself
    assert np.allclose(out[1e-3][0][:, 9:], out[1e-4][0][:, 9:], atol=2e-3)


def test_dp_tol_controls_precision_3dof(normalized_pair):
    d = normalized_pair
    coords, half, H_all, L_all, mf, bf, valid, u_gt = _setup(d)
    F_fixed = np.zeros((3, 3, 3))
    U_old = np.round(u_gt)
    vd = np.zeros((3, 3))
    out = {}
    for dp in DP_TOLS:
        U, it, st, zc = nk.icgn_3dof_parallel(
            coords,
            U_old.copy(),
            F_fixed,
            vd,
            *half,
            d["f"],
            d["gx"],
            d["gy"],
            d["gz"],
            d["mask"],
            d["g"],
            INTERP_CUBIC,
            H_all,
            mf,
            bf,
            valid,
            1e-3,
            1e-2,
            dp,
            100,
            5,
        )
        assert np.all(st == STATUS_CONVERGED)
        out[dp] = (U, it)
    for a, b in zip(DP_TOLS[:-1], DP_TOLS[1:]):
        assert np.all(out[b][1] >= out[a][1])
    assert np.allclose(out[1e-3][0], out[1e-4][0], atol=2e-3)
