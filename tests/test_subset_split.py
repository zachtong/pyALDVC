"""Subset splitting: the connected component around the subset centre, the kernels' gate, and the effect."""

import sys

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


@pytest.mark.skipif(not __import__("al_dvc.solver.cuda_kernels", fromlist=["x"]).cuda_available(), reason="no CUDA device")
def test_cuda_matches_the_cpu_kernels_on_split_subsets(wall_pair):
    f, g, disp, mask_ref, mask_def = wall_pair
    out = {}
    for backend in ("numba", "cuda"):
        para = dvcpara_default(
            winsize=WIN,
            winstepsize=8,
            search_radius=5,
            verbose=False,
            backend=backend,
            subset_split=True,
            local_outlier_threshold=0.0,
        )
        res = run_aldvc(para, [f, g], masks=[mask_ref, mask_def], compute_strain=False, resume=False)
        out[backend] = res.result_disp[0]
    a, b = out["numba"], out["cuda"]
    assert np.allclose(a.split_fraction, b.split_fraction, equal_nan=True)
    assert np.array_equal(a.status, b.status)
    assert np.max(np.abs(a.U - b.U)) < 5e-3  # float32 accumulation on the GPU, float64 on the CPU
    assert np.nanmax(np.abs(a.zncc - b.zncc)) < 1e-4


def test_split_nodes_take_their_initial_guess_from_their_own_side():
    """The integer search still spans the boundary; a cut node is filled from its own side instead."""
    from al_dvc.mesh.grid_mesh import mesh_setup
    from al_dvc.solver.init_disp import clean_initial_guess

    axis = np.arange(5.0)
    mesh = mesh_setup(axis, axis, axis)
    n = mesh.n_nodes
    grid = mesh.grid_shape
    mesh.node_valid = np.ones(n, dtype=bool)
    ix = (np.arange(n) % grid[2]).reshape(grid)
    edge_ok = np.ones(grid + (3,), dtype=bool)
    edge_ok[:, :, 1, 0] = False  # the +x edge between ix = 1 and ix = 2 is cut everywhere
    mesh.edge_ok = edge_ok.reshape(n, 3)
    disp = np.zeros((n, 3))
    disp[(ix <= 1).ravel(), 0] = 1.0
    disp[(ix >= 2).ravel(), 0] = -1.0
    disp[(ix == 1).ravel(), 0] = 0.0  # the contaminated peak: halfway between the two sides
    split_fraction = np.ones(n, dtype=np.float32)
    split_fraction[(ix == 1).ravel()] = 0.6
    para = dvcpara_default(winsize=WIN, winstepsize=8, init_outlier_threshold=0.0, verbose=False)
    ok = np.ones(n, dtype=bool)
    out, bad = clean_initial_guess(disp.copy(), ok, None, mesh, para, split_fraction)
    cut = (ix == 1).ravel()
    assert bad[cut].all() and not bad[~cut].any()
    assert np.allclose(out[cut, 0], 1.0, atol=1e-6)  # filled from ix = 0, not from the far side
    assert np.allclose(out[(ix >= 2).ravel(), 0], -1.0)
    out2, bad2 = clean_initial_guess(disp.copy(), ok, None, mesh, para, None)
    assert not bad2.any() and np.allclose(out2[cut, 0], 0.0)  # without the split info the guess stays contaminated


def test_the_drawn_grid_stops_at_the_wall():
    """The lattice preview and the 3-D lattice must not join nodes the mask separates."""
    from al_dvc.gui.lattice_preview import layer_segments, plan_lattice
    from al_dvc.mesh.grid_mesh import build_grid_axes, mesh_setup
    from al_dvc.mesh.mesh_cut import cut_mesh

    shape = (48, 64, 96)
    mask = np.ones(shape, np.uint8)
    mask[:, :, 46:49] = 0  # a wall three voxels wide, right through the volume
    para = dvcpara_default(winsize=16, winstepsize=6, verbose=False)
    plan = plan_lattice(shape, para.winsize, para.winstepsize, None, mask)
    assert plan.edge_ok is not None and not plan.edge_ok.all()
    for plane, index in (("xy", 24), ("xz", 32), ("yz", 20)):
        seg, _ = layer_segments(plan, plane, index)
        if plane == "yz":  # the wall is normal to x: this plane never crosses it
            continue
        h = seg[:, :, 0]  # the horizontal axis of xy and xz is x
        assert not np.any((h.min(axis=1) < 45) & (h.max(axis=1) > 49)), plane
    # the mesh the solver builds drops the same connections
    x0, y0, z0 = build_grid_axes(para.voi, shape, para.winsize, para.winstepsize)
    mesh = mesh_setup(x0, y0, z0)
    elements, edge_ok, dropped = cut_mesh(mesh.elements, mesh.coordinates, mask, mesh.n_nodes)
    assert dropped > 0 and np.array_equal(edge_ok.reshape(mesh.grid_shape + (3,)), plan.edge_ok)
    live = elements[elements[:, 0] >= 0]
    x = mesh.coordinates[:, 0]
    assert not np.any((x[live[:, 0]] < 45) & (x[live[:, 1]] > 49))


def test_the_split_budget_is_decided_for_the_reference_not_per_tile(monkeypatch):
    """A budget that runs out mid-plan must leave no tile split, not only the late ones.

    Deciding per tile gave the tiles that fitted a Hessian, a mean and a ZNCC denominator built over
    their kept component alone, and then dropped the keep rows for the whole reference -- so those
    nodes went on to correlate the *full* window against a normalisation built from a subset of it,
    while the nodes of the later tiles used the full window throughout. Nothing raised; the numbers
    were simply wrong, and inconsistently so between tiles.
    """
    from dataclasses import replace

    from al_dvc.core.data_structures import VOIRange
    from al_dvc.io.volume_ops import normalize_volume
    from al_dvc.mesh.grid_mesh import apply_mask_to_mesh, build_grid_axes, mesh_setup
    from al_dvc.solver.local_icgn import plan_split, precompute_local_context, split_keep_bytes
    from al_dvc.solver.tiling import ReferenceSource, plan_tiles

    shape = (64, 72, 192)  # long enough in x that plan_tiles gives more than one box
    para = replace(dvcpara_default(), winsize=(WIN, WIN, WIN), winstepsize=(8, 8, 8), subset_split=True, tile_local=112)
    mask = np.ones(shape, np.uint8)
    mask[:, :, 45:47] = 0  # one wall per half, so more than one tile holds subsets that are really cut
    mask[:, :, 141:143] = 0
    fn = normalize_volume(generate_speckle_volume(shape, sigma=2.0, seed=3))
    voi = VOIRange(x=(0, shape[2] - 1), y=(0, shape[1] - 1), z=(0, shape[0] - 1))
    mesh = mesh_setup(*build_grid_axes(voi, shape, para.winsize, para.winstepsize))
    mesh = apply_mask_to_mesh(mesh, mask, para.winsize, para.min_valid_ratio, cut_bridging=True)

    def source():
        return ReferenceSource(f=fn, mask=mask > 0, gradient_mode="stored")

    half = tuple(w // 2 for w in para.winsize)
    stride = int(getattr(para, "subset_stride", 1))
    coords_int = np.round(mesh.coordinates).astype(np.int64)
    tiles = list(plan_tiles(mesh.grid_shape, coords_int, shape, para, half))
    assert len(tiles) > 1, "the plan must be tiled for this test to mean anything"

    frac, enabled = plan_split(mesh, source(), para, mesh.node_valid, half, stride)
    assert enabled and frac is not None
    node_valid = np.asarray(mesh.node_valid, dtype=bool)
    whole_need = split_keep_bytes(int(np.count_nonzero(node_valid & (frac < 1.0))), half, stride)
    first = tiles[0][0]
    first_need = split_keep_bytes(int(np.count_nonzero(node_valid[first] & (frac[first] < 1.0))), half, stride)
    assert 0 < first_need < whole_need, (first_need, whole_need)
    # fits one tile but not the reference: the old code split that tile, then dropped the keep rows.
    # al_dvc.solver re-exports the local_icgn *function* under the submodule's name, so the module has
    # to come from sys.modules -- attribute lookup finds the function.
    module = sys.modules["al_dvc.solver.local_icgn"]
    monkeypatch.setattr(module, "MAX_SPLIT_BYTES", (first_need + whole_need) // 2)

    squeezed = precompute_local_context(mesh, source(), para)
    never = precompute_local_context(mesh, source(), replace(para, subset_split=False))

    assert squeezed.split_index is None and squeezed.split_fraction is None
    for field in ("H_all", "L_all", "meanf", "bottomf", "n_valid", "valid"):
        assert np.array_equal(getattr(squeezed, field), getattr(never, field)), field

    # The geometry has to be one where splitting actually moves those numbers, or the equality above
    # would hold for any implementation and this test would assert nothing.
    monkeypatch.setattr(module, "MAX_SPLIT_BYTES", 512 * 1024 * 1024)
    full = precompute_local_context(mesh, source(), para)
    assert int(np.count_nonzero(np.nan_to_num(full.split_fraction, nan=1.0) < 1.0)) > 0
    assert not np.array_equal(full.H_all, never.H_all)
