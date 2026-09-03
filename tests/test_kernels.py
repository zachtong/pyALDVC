"""Numba kernels vs NumPy reference implementations."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from al_dvc.core.data_structures import (
    STATUS_CONVERGED,
    STATUS_INVALID_SUBSET,
    STATUS_OUT_OF_BOUNDS,
)
from al_dvc.io.volume_ops import prefilter_bspline
from al_dvc.solver import numba_kernels as nk
from al_dvc.solver import reference_kernels as rk
from al_dvc.solver.interp_kernels import (
    INTERP_BSPLINE,
    INTERP_CUBIC,
    INTERP_LINEAR,
    sample_points,
)
from tests.conftest import F_AFFINE


@pytest.fixture(scope="module")
def rand_points():
    rng = np.random.default_rng(0)
    n = 400
    return rng.uniform(3, 60, n), rng.uniform(3, 68, n), rng.uniform(3, 76, n)


@pytest.mark.parametrize("mode", [INTERP_LINEAR, INTERP_CUBIC, INTERP_BSPLINE])
def test_interp_numba_matches_reference(normalized_pair, rand_points, mode):
    zs, ys, xs = rand_points
    vol = normalized_pair["f"]
    src = prefilter_bspline(vol) if mode == INTERP_BSPLINE else vol
    a = sample_points(src, zs, ys, xs, mode)
    b = rk.sample_volume_np(src, zs, ys, xs, mode)
    assert np.allclose(a, b, atol=1e-12)


def test_interp_matches_scipy(normalized_pair, rand_points):
    zs, ys, xs = rand_points
    vol = normalized_pair["f"]
    c3 = map_coordinates(vol.astype(np.float64), np.vstack([zs, ys, xs]), order=3, mode="mirror")
    a3 = sample_points(prefilter_bspline(vol), zs, ys, xs, INTERP_BSPLINE)
    assert np.allclose(a3, c3, atol=1e-5)
    c1 = map_coordinates(vol.astype(np.float64), np.vstack([zs, ys, xs]), order=1, mode="nearest")
    a1 = sample_points(vol, zs, ys, xs, INTERP_LINEAR)
    assert np.allclose(a1, c1, atol=1e-12)


def test_cubic_reproduces_quadratic():
    z, y, x = np.mgrid[0:20, 0:22, 0:24].astype(np.float64)
    vol = (0.1 * x**2 - 0.2 * y * z + 0.3 * y + 0.05 * z**2 + 1.0).astype(np.float32)
    rng = np.random.default_rng(1)
    zs, ys, xs = rng.uniform(2, 17, 200), rng.uniform(2, 19, 200), rng.uniform(2, 21, 200)
    exact = 0.1 * xs**2 - 0.2 * ys * zs + 0.3 * ys + 0.05 * zs**2 + 1.0
    got = sample_points(vol, zs, ys, xs, INTERP_CUBIC)
    assert np.allclose(got, exact, atol=1e-3)  # float32 storage limits the accuracy
    out = sample_points(vol, np.array([0.5]), np.array([5.0]), np.array([5.0]), INTERP_CUBIC)
    assert np.isnan(out[0])  # outside the admissible domain


def test_compose_warp_matches_reference():
    rng = np.random.default_rng(2)
    P = rng.normal(0, 0.05, 12)
    dP = rng.normal(0, 0.02, 12)
    P1 = P.copy()
    ok = nk.compose_warp_inplace(P1, dP)
    P2 = rk.compose_warp_np(P, dP)
    assert ok and np.allclose(P1, P2, atol=1e-13)

    # W(P_new) = W(P) W(dP)^-1  -> W(P_new) W(dP) = W(P)
    def W(p):
        M = np.eye(4)
        M[:3, :3] = np.eye(3) + p[:9].reshape(3, 3)
        M[:3, 3] = p[9:]
        return M

    assert np.allclose(W(P1) @ W(dP), W(P), atol=1e-12)


def test_precompute_matches_reference(normalized_pair):
    d = normalized_pair
    coords = np.array([[30, 32, 28], [45, 20, 40]], dtype=np.int64)
    half = (6, 6, 6)
    H_all, L_all, mf, bf, nv, valid = nk.precompute_nodes(coords, *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], 0.5, 1e12)
    assert valid.all()
    for n in range(2):
        H, m, b, nvr, ok = rk.precompute_node_np(coords[n], half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"])
        assert ok and nvr == nv[n] == 13**3
        assert np.allclose(H_all[n], H, rtol=1e-10, atol=1e-8)
        assert np.isclose(mf[n], m) and np.isclose(bf[n], b)
        assert np.allclose(L_all[n] @ L_all[n].T, H, rtol=1e-9, atol=1e-6)


def test_precompute_rejects_masked_and_border_subsets(normalized_pair):
    d = normalized_pair
    mask = d["mask"].copy()
    mask[:, :, 20:] = 0
    coords = np.array([[30, 32, 28], [3, 32, 28], [10, 30, 30]], dtype=np.int64)
    _, _, _, _, nv, valid = nk.precompute_nodes(coords, 6, 6, 6, d["f"], d["gx"], d["gy"], d["gz"], mask, 0.5, 1e12)
    assert not valid[0]  # fully masked subset
    assert not valid[1]  # subset leaves the volume
    assert valid[2] and nv[2] == 13**3


@pytest.mark.parametrize("mode", [INTERP_CUBIC, INTERP_BSPLINE, INTERP_LINEAR])
def test_icgn_12dof_numba_matches_reference_and_truth(normalized_pair, mode):
    d = normalized_pair
    g = prefilter_bspline(d["g"]) if mode == INTERP_BSPLINE else d["g"]
    coords = np.array([[30, 32, 28], [45, 20, 40], [24, 44, 36]], dtype=np.int64)
    half = (7, 7, 7)
    H_all, L_all, mf, bf, nv, valid = nk.precompute_nodes(coords, *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], 0.5, 1e12)
    u_gt = np.column_stack(d["disp"](coords[:, 0].astype(float), coords[:, 1].astype(float), coords[:, 2].astype(float)))
    P0 = np.zeros((3, 12))
    P0[:, 9:] = np.round(u_gt)
    P, it, st, zc = nk.icgn_12dof_parallel(
        coords, P0.copy(), *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], g, mode, L_all, mf, bf, valid, 1e-2, 1e-3, 100, 5
    )
    assert np.all(st == STATUS_CONVERGED) and np.all(it <= 10)
    tol_u = 0.03 if mode != INTERP_LINEAR else 0.06
    assert np.all(np.abs(P[:, 9:] - u_gt) < tol_u)
    assert np.all(np.abs(P[:, :9] - F_AFFINE.ravel()) < 0.01)
    assert np.all(zc > 0.95) and np.all(zc <= 1.0 + 1e-9)
    for n in range(3):
        Pr, itr, str_, zcr = rk.icgn_12dof_np(
            P0[n],
            coords[n],
            half,
            d["f"],
            d["gx"],
            d["gy"],
            d["gz"],
            d["mask"],
            g,
            mode,
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


def test_icgn_12dof_status_codes(normalized_pair):
    d = normalized_pair
    coords = np.array([[30, 32, 28], [30, 32, 28]], dtype=np.int64)
    half = (6, 6, 6)
    H_all, L_all, mf, bf, nv, valid = nk.precompute_nodes(coords, *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], 0.5, 1e12)
    P0 = np.zeros((2, 12))
    P0[0, 9] = 60.0  # warps outside the volume
    valid[1] = False
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
        5,
    )
    assert st[0] == STATUS_OUT_OF_BOUNDS and np.isnan(zc[0])
    assert st[1] == STATUS_INVALID_SUBSET


def test_icgn_3dof_numba_matches_reference(normalized_pair):
    d = normalized_pair
    coords = np.array([[30, 32, 28], [45, 20, 40]], dtype=np.int64)
    half = (7, 7, 7)
    H_all, L_all, mf, bf, nv, valid = nk.precompute_nodes(coords, *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], 0.5, 1e12)
    u_gt = np.column_stack(d["disp"](coords[:, 0].astype(float), coords[:, 1].astype(float), coords[:, 2].astype(float)))
    U_old = u_gt + np.array([0.3, -0.2, 0.25])
    F_fixed = np.tile(F_AFFINE, (2, 1, 1))
    vd = np.array([[0.05, 0.0, -0.02], [0.0, 0.01, 0.0]])
    mu = 1e-3
    U, it, st, zc = nk.icgn_3dof_parallel(
        coords,
        U_old,
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
        mu,
        1e-2,
        1e-3,
        100,
        5,
    )
    assert np.all(st == STATUS_CONVERGED)
    assert np.all(np.abs(U - u_gt) < 0.05)
    for n in range(2):
        Ur, itr, str_, zcr = rk.icgn_3dof_np(
            U_old[n],
            F_fixed[n],
            vd[n],
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
            mu,
            1e-2,
            1e-3,
            100,
            5,
        )
        assert itr == it[n] and np.allclose(U[n], Ur, atol=1e-10)


def test_masked_subset_solves(normalized_pair):
    """Half-masked subsets still converge (masked ZNSSD)."""
    d = normalized_pair
    mask = d["mask"].copy()
    mask[:, :, 34:] = 0
    coords = np.array([[30, 32, 28]], dtype=np.int64)
    half = (7, 7, 7)
    H_all, L_all, mf, bf, nv, valid = nk.precompute_nodes(coords, *half, d["f"], d["gx"], d["gy"], d["gz"], mask, 0.5, 1e12)
    assert valid[0] and 0.5 * 15**3 < nv[0] < 15**3
    u_gt = np.column_stack(d["disp"](np.array([30.0]), np.array([32.0]), np.array([28.0])))
    P0 = np.zeros((1, 12))
    P0[:, 9:] = np.round(u_gt)
    P, it, st, zc = nk.icgn_12dof_parallel(
        coords,
        P0.copy(),
        *half,
        d["f"],
        d["gx"],
        d["gy"],
        d["gz"],
        mask,
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
    assert st[0] == STATUS_CONVERGED
    assert np.all(np.abs(P[0, 9:] - u_gt[0]) < 0.05)
    assert 0.9 < zc[0] <= 1.0 + 1e-9


def test_evaluate_zncc(normalized_pair):
    d = normalized_pair
    coords = np.array([[30, 32, 28]], dtype=np.int64)
    half = (6, 6, 6)
    H_all, L_all, mf, bf, nv, valid = nk.precompute_nodes(coords, *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], 0.5, 1e12)
    u_gt = np.column_stack(d["disp"](np.array([30.0]), np.array([32.0]), np.array([28.0])))
    P = np.zeros((1, 12))
    P[0, :9] = F_AFFINE.ravel()
    P[0, 9:] = u_gt[0]
    z_true = nk.evaluate_zncc_parallel(coords, P, *half, d["f"], d["mask"], d["g"], INTERP_CUBIC, mf, bf, valid)
    z_zero = nk.evaluate_zncc_parallel(coords, np.zeros((1, 12)), *half, d["f"], d["mask"], d["g"], INTERP_CUBIC, mf, bf, valid)
    assert z_true[0] > 0.98 and z_zero[0] < z_true[0]
