"""Stall detection: hopeless nodes stop after ``icgn_patience`` non-improving iterations."""

import numpy as np

from al_dvc.core.data_structures import STATUS_CONVERGED, STATUS_MAX_ITER, STATUS_OUT_OF_BOUNDS, STATUS_STALLED
from al_dvc.solver import numba_kernels as nk
from al_dvc.solver import reference_kernels as rk
from al_dvc.solver.interp_kernels import INTERP_CUBIC

PATIENCE = 5


def _setup(d, offset):
    coords = np.array([[40, 36, 32], [24, 48, 30], [56, 20, 34]], dtype=np.int64)
    half = (8, 8, 8)
    H_all, L_all, mf, bf, nv, valid = nk.precompute_nodes(coords, *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], 0.5, 1e12)
    u_gt = np.column_stack(d["disp"](coords[:, 0].astype(float), coords[:, 1].astype(float), coords[:, 2].astype(float)))
    P0 = np.zeros((3, 12))
    P0[:, 9:] = np.round(u_gt) + offset
    return coords, half, H_all, L_all, mf, bf, valid, u_gt, P0


def test_hopeless_12dof_stalls_quickly(normalized_pair):
    d = normalized_pair
    coords, half, H_all, L_all, mf, bf, valid, u_gt, P0 = _setup(d, np.array([11.0, -9.0, 0.0]))
    args = (coords, P0.copy(), *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], d["g"], INTERP_CUBIC, L_all, mf, bf, valid)
    P, it, st, zc = nk.icgn_12dof_parallel(*args, 1e-2, 1e-3, 100, PATIENCE)
    # a wrong basin never converges to the truth; with patience it is abandoned early
    assert np.all(np.isin(st, [STATUS_STALLED, STATUS_OUT_OF_BOUNDS, STATUS_MAX_ITER, STATUS_CONVERGED]))
    stalled = st == STATUS_STALLED
    assert stalled.any()
    assert np.all(it[stalled] <= 5 * PATIENCE)
    # the returned parameters are the best-ZNCC iterate, never NaN
    assert np.all(np.isfinite(P[stalled]))
    assert np.all(zc[stalled] <= 1.0)
    # without patience the same nodes run to the iteration cap or leave the volume
    P2, it2, st2, zc2 = nk.icgn_12dof_parallel(*args, 1e-2, 1e-3, 100, 0)
    assert not np.any(st2 == STATUS_STALLED)
    assert np.all(it2[stalled] >= it[stalled])
    # reference implementation agrees on status and iteration count
    for n in np.flatnonzero(stalled):
        Pr, itr, str_, zcr = rk.icgn_12dof_np(
            P0[n],
            coords[n],
            half,
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
            PATIENCE,
        )
        assert str_ == STATUS_STALLED and itr == it[n]
        assert np.allclose(Pr, P[n], atol=1e-10)


def test_good_nodes_never_stall(normalized_pair):
    d = normalized_pair
    coords, half, H_all, L_all, mf, bf, valid, u_gt, P0 = _setup(d, 0.0)
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
        1e-3,
        100,
        PATIENCE,
    )
    assert np.all(st == STATUS_CONVERGED)
    assert np.all(np.abs(P[:, 9:] - u_gt) < 0.03)
    U_old = np.round(u_gt)
    U, it3, st3, zc3 = nk.icgn_3dof_parallel(
        coords,
        U_old.copy(),
        np.zeros((3, 3, 3)),
        np.zeros((3, 3)),
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
        1e-3,
        100,
        PATIENCE,
    )
    assert np.all(st3 == STATUS_CONVERGED)


def test_hopeless_3dof_stalls(normalized_pair):
    d = normalized_pair
    coords, half, H_all, L_all, mf, bf, valid, u_gt, P0 = _setup(d, 0.0)
    U_old = np.round(u_gt) + np.array([11.0, -9.0, 0.0])
    args = (
        coords,
        U_old.copy(),
        np.zeros((3, 3, 3)),
        np.zeros((3, 3)),
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
    )
    U, it, st, zc = nk.icgn_3dof_parallel(*args, 1e-2, 1e-3, 100, PATIENCE)
    U2, it2, st2, zc2 = nk.icgn_3dof_parallel(*args, 1e-2, 1e-3, 100, 0)
    assert np.all(np.isin(st, [STATUS_STALLED, STATUS_OUT_OF_BOUNDS, STATUS_MAX_ITER, STATUS_CONVERGED]))
    assert not np.any(st2 == STATUS_STALLED)
    assert np.all(it <= it2)
    for n in np.flatnonzero(st == STATUS_STALLED):
        Ur, itr, str_, zcr = rk.icgn_3dof_np(
            U_old[n],
            np.zeros((3, 3)),
            np.zeros(3),
            coords[n],
            half,
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
            1e-3,
            1e-2,
            1e-3,
            100,
            PATIENCE,
        )
        assert str_ == STATUS_STALLED and itr == it[n]
        assert np.allclose(Ur, U[n], atol=1e-10)


def test_noise_volume_stalls(normalized_pair):
    """A deformed volume without correlation (pure noise) is abandoned within a few patience windows."""
    d = normalized_pair
    coords, half, H_all, L_all, mf, bf, valid, u_gt, P0 = _setup(d, 0.0)
    noise = np.random.default_rng(5).normal(size=d["g"].shape).astype(np.float32)
    args = (coords, P0.copy(), *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], noise, INTERP_CUBIC, L_all, mf, bf, valid)
    P, it, st, zc = nk.icgn_12dof_parallel(*args, 1e-2, 1e-3, 100, PATIENCE)
    P0_, it0, st0, zc0 = nk.icgn_12dof_parallel(*args, 1e-2, 1e-3, 100, 0)
    assert not np.any(st == STATUS_CONVERGED)
    assert np.all(np.isin(st, [STATUS_STALLED, STATUS_OUT_OF_BOUNDS]))
    assert np.all(it <= 6 * PATIENCE)
    assert it.sum() < 0.5 * it0.sum()
    assert np.all(np.isfinite(P))
