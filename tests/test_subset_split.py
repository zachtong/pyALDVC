"""Subset splitting: the connected component around the subset centre, the kernels' gate, and the effect."""

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.pipeline import run_aldvc
from al_dvc.solver import numba_kernels as nk
from al_dvc.solver import reference_kernels as rk
from al_dvc.solver.interp_kernels import INTERP_CUBIC
from al_dvc.synthetic import (
    affine_displacement,
    evaluate_at_nodes,
    generate_speckle_volume,
    two_body_displacement,
    warp_volume_lagrangian,
)

SHAPE = (64, 72, 96)
WALL = 45.5  # the masked wall occupies x = 45, 46
WIN = 24


def test_flood_fill_matches_scipy_and_respects_a_diagonal_crack():
    rng = np.random.default_rng(0)
    for trial in range(20):
        sub = (rng.random((9, 11, 13)) > 0.4).astype(np.uint8)
        if trial % 3 == 0:
            sub[4, 5, 6] = 1  # centre in the mask
        out = np.zeros_like(sub)
        count = nk._flood_fill_centre(sub, out, np.empty(sub.size, np.int64))
        ref = rk.centre_component_np(sub)
        assert np.array_equal(out, ref) and count == int(ref.sum())
    sub = np.ones((9, 9, 9), np.uint8)
    for i in range(8):
        sub[:, i + 1, i] = 0  # the plane y = x + 1, one voxel thick and diagonal
    out = np.zeros_like(sub)
    nk._flood_fill_centre(sub, out, np.empty(sub.size, np.int64))
    _, yy, xx = np.nonzero(out)
    assert out.sum() > 0 and np.all(yy < xx + 1)  # 6-connectivity does not leak through it


@pytest.fixture(scope="module")
def wall_pair():
    f = generate_speckle_volume(SHAPE, sigma=2.0, seed=7)
    disp = two_body_displacement("x", WALL, affine_displacement(t=(0.6, -0.3, 0.2)), affine_displacement(t=(-0.5, 0.4, -0.3)))
    g = warp_volume_lagrangian(f, disp)
    mask_ref = np.ones(SHAPE, np.uint8)
    mask_ref[:, :, 45:47] = 0
    mask_def = np.ones(SHAPE, np.uint8)
    mask_def[:, :, 44:48] = 0
    return f, g, disp, mask_ref, mask_def


def test_numba_matches_reference_on_split_nodes(wall_pair):
    from al_dvc.io.volume_ops import compute_gradients, normalize_volume

    f, g, disp, mask, _ = wall_pair
    fn, gn = normalize_volume(f), normalize_volume(g)
    gx, gy, gz = compute_gradients(fn)
    half = (WIN // 2,) * 3
    coords = np.array([[40, 36, 32], [48, 36, 32], [20, 36, 32]], dtype=np.int64)  # two straddle the wall
    rows, n_keep, n_inmask = nk.build_split_rows(coords, np.array([0, 1]), *half, 2, mask, 0)
    split_index = np.array([0, 1, -1])
    S = rk.subset_count(half, 2)
    assert np.all(n_keep < n_inmask) and rows.shape == (2, (S + 7) // 8)
    H_all, L_all, mf, bf, nv, valid = nk.precompute_nodes(coords, *half, fn, gx, gy, gz, mask, 0.5, 1e12, 2, split_index, rows)
    assert valid.all() and nv[0] == n_keep[0] and nv[1] == n_keep[1] and nv[2] == S
    keeps = [rk.keep_from_row(rows[r], S) if r >= 0 else None for r in split_index]
    for n in range(3):
        H, m, b, nvr, ok = rk.precompute_node_np(coords[n], half, fn, gx, gy, gz, mask, 0.5, 1e12, 2, keeps[n])
        assert ok and nvr == nv[n] and np.allclose(H_all[n], H, rtol=1e-10, atol=1e-8)
        assert np.isclose(mf[n], m) and np.isclose(bf[n], b)
    u_gt = evaluate_at_nodes(disp, coords.astype(float))
    P0 = np.zeros((3, 12))
    P0[:, 9:] = np.round(u_gt)
    P, it, st, zc = nk.icgn_12dof_parallel(
        coords,
        P0.copy(),
        *half,
        fn,
        gx,
        gy,
        gz,
        mask,
        gn,
        INTERP_CUBIC,
        L_all,
        mf,
        bf,
        valid,
        1e-2,
        1e-3,
        100,
        5,
        stride=2,
        split_index=split_index,
        split_keep=rows,
    )
    assert np.all(st == 0) and np.all(np.abs(P[:, 9:] - u_gt) < 0.03)  # each side recovers its own translation
    for n in range(3):
        Pr, itr, sr, zr = rk.icgn_12dof_np(
            P0[n],
            coords[n],
            half,
            fn,
            gx,
            gy,
            gz,
            mask,
            gn,
            INTERP_CUBIC,
            H_all[n],
            mf[n],
            bf[n],
            1e-2,
            1e-3,
            100,
            5,
            stride=2,
            keep=keeps[n],
        )
        assert itr == it[n] and sr == st[n] and np.allclose(P[n], Pr, atol=1e-10) and np.isclose(zc[n], zr, atol=1e-10)


def _run(wall_pair, **overrides):
    f, g, disp, mask_ref, mask_def = wall_pair
    para = dvcpara_default(
        winsize=WIN,
        winstepsize=8,
        search_radius=5,
        verbose=False,
        backend="numba",
        use_global_step=False,
        local_outlier_threshold=0.0,
        **overrides,
    )
    res = run_aldvc(para, [f, g], masks=[mask_ref, mask_def], compute_strain=False, resume=False)
    fr, mesh = res.result_disp[0], res.dvc_mesh
    err = np.linalg.norm(fr.U - evaluate_at_nodes(disp, mesh.coordinates), axis=1)
    dist = np.abs(mesh.coordinates[:, 0] - WALL)
    near = (dist <= WIN / 2) & (dist > 1.0) & np.asarray(mesh.node_valid, dtype=bool)
    return fr, float(np.sqrt(np.mean(err[near] ** 2))), near


def test_two_bodies_keep_their_own_translation(wall_pair):
    fr_off, near_off, near = _run(wall_pair, subset_split=False)
    fr_on, near_on, _ = _run(wall_pair, subset_split=True)
    assert fr_off.split_fraction is None and fr_on.split_fraction is not None
    assert np.all(fr_on.split_fraction[near] < 1.0) and np.all(fr_on.split_fraction[near] > 0.4)
    assert near_off > 0.2 and near_on < 0.05  # the subsets across the wall averaged the jump away; split ones do not


def test_split_without_boundaries_changes_nothing(wall_pair):
    f, g, _, _, _ = wall_pair
    outs = []
    for split in (False, True):
        para = dvcpara_default(
            winsize=WIN, winstepsize=8, search_radius=5, verbose=False, backend="numba", use_global_step=False, subset_split=split
        )
        res = run_aldvc(para, [f, g], masks=[np.ones(SHAPE, np.uint8)] * 2, compute_strain=False, resume=False)
        outs.append(res.result_disp[0])
    assert np.array_equal(outs[0].U, outs[1].U) and np.array_equal(outs[0].zncc, outs[1].zncc)
    assert np.all(outs[1].split_fraction[np.asarray(outs[1].status) == 0] == 1.0)


def test_mesh_is_cut_at_the_wall_and_the_admm_result_keeps_the_jump(wall_pair):
    from al_dvc.mesh.mesh_cut import cut_edge_nodes

    f, g, disp, mask_ref, mask_def = wall_pair
    para = dvcpara_default(winsize=WIN, winstepsize=8, search_radius=5, verbose=False, backend="numba", subset_split=True)
    res = run_aldvc(para, [f, g], masks=[mask_ref, mask_def], compute_strain=True, resume=False)
    mesh = res.dvc_mesh
    x = mesh.coordinates[:, 0]
    nx = mesh.grid_shape[2]
    assert mesh.is_cut
    straddle = (x < WALL) & (x + mesh.spacing[0] > WALL) & np.asarray(mesh.node_valid)
    assert straddle.any() and not mesh.edge_ok[straddle, 0].any()  # every +x edge across the wall is cut
    assert mesh.edge_ok[~straddle, 0].all() and mesh.edge_ok[:, 1].all() and mesh.edge_ok[:, 2].all()
    assert set(cut_edge_nodes(mesh.edge_ok, mesh.grid_shape)) <= set(mesh.boundary_nodes.tolist())
    live = mesh.elements[mesh.elements[:, 0] >= 0]
    assert not np.any((x[live[:, 0]] < WALL) & (x[live[:, 1]] > WALL))  # no surviving element spans the wall
    fr = res.result_disp[0]
    err = np.linalg.norm(fr.U - evaluate_at_nodes(disp, mesh.coordinates), axis=1)
    dist = np.abs(x - WALL)
    near = (dist <= WIN / 2) & (dist > 1.0) & np.asarray(mesh.node_valid, dtype=bool)
    assert fr.admm is not None and float(np.sqrt(np.mean(err[near] ** 2))) < 0.05  # the global step no longer smooths it
    # rigid bodies: the strain next to the wall stays small because the plane fit does not reach across
    sr = res.result_strain[0]
    exx = np.asarray(sr.exx)
    assert np.nanmax(np.abs(exx[near & np.isfinite(exx)])) < 0.01
    assert nx > 0
