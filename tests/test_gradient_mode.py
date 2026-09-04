"""gradient_mode='on_the_fly': no gradient volumes, same solution."""

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.pipeline import run_aldvc
from al_dvc.io.volume_ops import build_reference_bundle, memory_model
from al_dvc.solver import numba_kernels as nk
from al_dvc.solver.interp_kernels import INTERP_CUBIC


def test_kernels_match_with_and_without_gradient_volumes(normalized_pair):
    d = normalized_pair
    coords = np.array([[40, 36, 32], [24, 48, 30], [56, 20, 34]], dtype=np.int64)
    half = (8, 8, 8)
    dummy = np.zeros((1, 1, 1), dtype=np.float32)
    H_s, L_s, mf_s, bf_s, nv_s, valid_s = nk.precompute_nodes(
        coords, *half, d["f"], d["gx"], d["gy"], d["gz"], d["mask"], 0.5, 1e12
    )
    H_f, L_f, mf_f, bf_f, nv_f, valid_f = nk.precompute_nodes(coords, *half, d["f"], dummy, dummy, dummy, d["mask"], 0.5, 1e12)
    assert np.array_equal(valid_s, valid_f) and np.array_equal(nv_s, nv_f)
    assert np.allclose(H_s, H_f, rtol=1e-5, atol=1e-6 * np.abs(H_s).max())
    assert np.allclose(mf_s, mf_f) and np.allclose(bf_s, bf_f)
    u_gt = np.column_stack(d["disp"](coords[:, 0].astype(float), coords[:, 1].astype(float), coords[:, 2].astype(float)))
    P0 = np.zeros((3, 12))
    P0[:, 9:] = np.round(u_gt)
    common = (*half, d["f"])
    P_s, it_s, st_s, zc_s = nk.icgn_12dof_parallel(
        coords,
        P0.copy(),
        *common,
        d["gx"],
        d["gy"],
        d["gz"],
        d["mask"],
        d["g"],
        INTERP_CUBIC,
        L_s,
        mf_s,
        bf_s,
        valid_s,
        1e-2,
        1e-3,
        100,
        5,
    )
    P_f, it_f, st_f, zc_f = nk.icgn_12dof_parallel(
        coords,
        P0.copy(),
        *common,
        dummy,
        dummy,
        dummy,
        d["mask"],
        d["g"],
        INTERP_CUBIC,
        L_f,
        mf_f,
        bf_f,
        valid_f,
        1e-2,
        1e-3,
        100,
        5,
    )
    assert np.array_equal(st_s, st_f)
    assert np.abs(P_s - P_f).max() < 1e-5
    assert np.abs(zc_s - zc_f).max() < 1e-7
    U_s, *_ = nk.icgn_3dof_parallel(
        coords,
        np.round(u_gt),
        np.zeros((3, 3, 3)),
        np.zeros((3, 3)),
        *common,
        d["gx"],
        d["gy"],
        d["gz"],
        d["mask"],
        d["g"],
        INTERP_CUBIC,
        H_s,
        mf_s,
        bf_s,
        valid_s,
        1e-3,
        1e-2,
        1e-3,
        100,
        5,
    )
    U_f, *_ = nk.icgn_3dof_parallel(
        coords,
        np.round(u_gt),
        np.zeros((3, 3, 3)),
        np.zeros((3, 3)),
        *common,
        dummy,
        dummy,
        dummy,
        d["mask"],
        d["g"],
        INTERP_CUBIC,
        H_f,
        mf_f,
        bf_f,
        valid_f,
        1e-3,
        1e-2,
        1e-3,
        100,
        5,
    )
    assert np.abs(U_s - U_f).max() < 1e-5


def test_pipeline_on_the_fly_matches_stored(affine_pair):
    f, g, _ = affine_pair
    common = dict(winsize=16, winstepsize=8, search_radius=5, verbose=False, admm_max_iter=2, backend="numba")  # CPU semantics
    r_s = run_aldvc(dvcpara_default(**common), [f, g], compute_strain=False)
    r_f = run_aldvc(dvcpara_default(gradient_mode="on_the_fly", **common), [f, g], compute_strain=False)
    a, b = r_s.result_disp[0], r_f.result_disp[0]
    assert np.array_equal(a.status, b.status)
    assert np.abs(a.U - b.U).max() < 1e-4
    assert np.abs(a.U_local - b.U_local).max() < 1e-4
    assert np.allclose(a.U_std, b.U_std, rtol=1e-3, equal_nan=True)


def test_bundle_and_memory_model(affine_pair):
    f, _, _ = affine_pair
    fn = np.asarray(f, dtype=np.float32)
    stored = build_reference_bundle(fn, None, "stored")
    fly = build_reference_bundle(fn, None, "on_the_fly")
    assert stored.gx.shape == fn.shape and fly.gx.shape == (1, 1, 1)
    with pytest.raises(ValueError):
        build_reference_bundle(fn, None, "sometimes")
    m_s = memory_model((1000, 1000, 1000), "stored")
    m_f = memory_model((1000, 1000, 1000), "on_the_fly")
    assert m_s["bytes_per_voxel"] == 21.0 and m_f["bytes_per_voxel"] == 9.0
    assert m_s["total_gb"] == pytest.approx(21.0)
    assert memory_model((10, 10, 10), "stored", "bspline", masked=True)["bytes_per_voxel"] == 29.0
    with pytest.raises(ValueError):
        dvcpara_default(gradient_mode="fast")
    with pytest.raises(ValueError):
        dvcpara_default(gradient_mode="on_the_fly", backend="numpy")
