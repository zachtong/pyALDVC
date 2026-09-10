"""Tiled local steps: the plan's geometry, and that tiling changes the memory and not the answer."""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.data_structures import STATUS_CONVERGED, VOIRange
from al_dvc.core.pipeline import run_aldvc
from al_dvc.io.volume_ops import normalize_volume, prepare_deformed
from al_dvc.mesh.grid_mesh import apply_mask_to_mesh, build_grid_axes, mesh_setup
from al_dvc.solver.local_icgn import local_icgn, precompute_local_context
from al_dvc.solver.tiling import ReferenceSource, TileBox, plan_tiles, tile_pad, whole_box_tile
from al_dvc.synthetic import (
    affine_displacement,
    evaluate_at_nodes,
    generate_speckle_volume,
    warp_volume_lagrangian,
)

SHAPE = (96, 112, 128)
DP_TOL = 1e-3  # the solver's own convergence tolerance: a tiling difference must be far below it


@pytest.fixture(scope="module")
def pair():
    """A speckle pair with a known affine field, and a mask that excludes a border."""
    f = generate_speckle_volume(SHAPE, sigma=2.0, seed=1)
    disp = affine_displacement(F=np.diag([0.004, -0.002, 0.001]), t=(0.7, -0.5, 0.3), centre=tuple(s / 2 for s in SHAPE[::-1]))
    g = warp_volume_lagrangian(f, disp)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[6:-6, 8:-8, 10:-10] = True
    return {"f": f, "g": g, "disp": disp, "mask": mask, "fn": normalize_volume(f), "gn": normalize_volume(g)}


def _mesh(para, mask=None, cut=False):
    voi = VOIRange(x=(0, SHAPE[2] - 1), y=(0, SHAPE[1] - 1), z=(0, SHAPE[0] - 1))
    mesh = mesh_setup(*build_grid_axes(voi, SHAPE, para.winsize, para.winstepsize))
    m = None if mask is None else mask.astype(np.uint8)
    return apply_mask_to_mesh(mesh, m, para.winsize, para.min_valid_ratio, cut_bridging=cut)


# ----------------------------------------------------------------------------- geometry
def test_a_box_crops_and_shifts_consistently():
    box = TileBox(origin=(4, 8, 12), shape=(20, 24, 28), open_faces=(True, True, False, True, True, False))
    vol = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    crop = box.crop(vol)
    assert crop.shape == (20, 24, 28) and crop.flags.c_contiguous
    assert crop[0, 0, 0] == vol[4, 8, 12] and crop[-1, -1, -1] == vol[23, 31, 39]
    coords = np.array([[20, 15, 10], [30, 20, 12]], dtype=np.int64)  # [x, y, z]
    assert np.array_equal(box.shift(coords), coords - np.array([12, 8, 4]))
    # a voxel read at the shifted coordinate is the voxel the global coordinate would have read
    for (x, y, z), (xs, ys, zs) in zip(coords, box.shift(coords)):
        assert crop[zs, ys, xs] == vol[z, y, x]
    assert box.has_open_face and not box.is_whole
    whole = whole_box_tile(SHAPE)
    assert whole.is_whole and not whole.has_open_face and whole.crop(vol) is vol
    assert whole.crop(None) is None
    placeholder = np.ones((1, 1, 1), dtype=np.uint8)  # an absent mask keeps the shape the kernels know it by
    assert box.crop(placeholder) is placeholder


def test_the_plan_covers_every_node_once_and_holds_its_window():
    para = dvcpara_default(winsize=16, winstepsize=8, verbose=False)
    mesh = _mesh(para)
    coords = np.round(mesh.coordinates).astype(np.int64)
    ref_pad = tile_pad(para, (8, 8, 8))
    assert ref_pad == (11, 11, 11)  # the subset half-width plus the 3-voxel gradient stencil
    assert min(tile_pad(para, (8, 8, 8), deformed=True)) > max(ref_pad)  # the deformed side needs more
    for edge in (96, 112, 128):
        plan = plan_tiles(mesh.grid_shape, coords, SHAPE, para, (8, 8, 8), edge=edge)
        assert len(plan) > 1 and not plan.fallback
        covered = np.concatenate([nodes for nodes, _box in plan])
        assert np.array_equal(np.sort(covered), np.arange(mesh.n_nodes))
        for nodes, box in plan:
            assert box.contains_windows(coords[nodes], plan.ref_pad)
            assert all(o >= 0 for o in box.origin)
            assert all(o + n <= s for o, n, s in zip(box.origin, box.shape, SHAPE))


def test_a_target_too_small_for_one_subset_falls_back_to_the_whole_volume():
    para = dvcpara_default(winsize=16, winstepsize=8, verbose=False)
    mesh = _mesh(para)
    coords = np.round(mesh.coordinates).astype(np.int64)
    plan = plan_tiles(mesh.grid_shape, coords, SHAPE, para, (8, 8, 8), edge=24)
    assert plan.is_whole and "smaller than one subset" in plan.fallback
    off = plan_tiles(mesh.grid_shape, coords, SHAPE, para, (8, 8, 8), edge=0)
    assert off.is_whole and not off.fallback  # off is not a fallback, it is the untiled path


# ----------------------------------------------------------------------------- equivalence
@pytest.mark.parametrize("gradient_mode", ["stored", "on_the_fly"])
@pytest.mark.parametrize("use_mask", [False, True])
def test_the_precompute_is_bit_identical_tiled(pair, gradient_mode, use_mask, caplog):
    para = dvcpara_default(
        winsize=16, winstepsize=8, verbose=False, backend="numba", gradient_mode=gradient_mode, subset_split=use_mask
    )
    mask = pair["mask"] if use_mask else None
    mesh = _mesh(para, mask, cut=use_mask)

    def source():
        return ReferenceSource(f=pair["fn"], mask=mask, gradient_mode=gradient_mode)

    with caplog.at_level(logging.INFO, logger="al_dvc.solver.tiling"):
        whole = precompute_local_context(mesh, source(), para)
        tiled = precompute_local_context(mesh, source(), replace(para, tile_local=96))
    assert "Local steps tiled" in caplog.text
    for field in ("H_all", "L_all", "meanf", "bottomf", "n_valid", "valid"):
        assert np.array_equal(getattr(whole, field), getattr(tiled, field)), field
    if whole.split_fraction is None:
        assert tiled.split_fraction is None
    else:
        assert np.array_equal(np.nan_to_num(whole.split_fraction, nan=-1.0), np.nan_to_num(tiled.split_fraction, nan=-1.0))
        # the rows are renumbered per tile, so compare which nodes are split, not the row indices
        assert np.array_equal(whole.split_index >= 0, tiled.split_index >= 0)


@pytest.mark.parametrize("use_mask", [False, True])
def test_the_12dof_solve_matches_far_below_its_own_tolerance(pair, use_mask):
    para = dvcpara_default(winsize=16, winstepsize=8, verbose=False, backend="numba", subset_split=use_mask)
    mask = pair["mask"] if use_mask else None
    mesh = _mesh(para, mask, cut=use_mask)
    g_prep = prepare_deformed(pair["gn"], para.interp_method)

    def source():
        return ReferenceSource(f=pair["fn"], mask=mask, gradient_mode="stored")

    ctx = precompute_local_context(mesh, source(), para)
    U0 = np.round(evaluate_at_nodes(pair["disp"], mesh.coordinates))
    U_a, F_a, info_a, bad_a = local_icgn(ctx, source(), g_prep, U0, para, mesh)
    U_b, F_b, info_b, bad_b = local_icgn(ctx, source(), g_prep, U0, replace(para, tile_local=96), mesh)
    assert (info_a.status == STATUS_CONVERGED).all(), "the untiled solve must converge for the comparison to mean anything"
    assert np.array_equal(info_a.status, info_b.status)
    assert np.array_equal(info_a.n_iter, info_b.n_iter)
    assert np.array_equal(bad_a, bad_b)
    assert np.allclose(info_a.zncc, info_b.zncc, equal_nan=True)
    # not bit-identical: a tile-local x0 rounds `x0 + P[9]` differently. It must stay negligible.
    assert np.abs(U_a - U_b).max() < 1e-9 * DP_TOL / 1e-3
    assert np.abs(F_a - F_b).max() < 1e-12


@pytest.mark.parametrize("use_mask", [False, True])
def test_a_whole_run_gives_the_same_field_tiled(pair, use_mask):
    masks = [pair["mask"], pair["mask"]] if use_mask else None
    out = {}
    for edge in (0, 112):
        para = dvcpara_default(winsize=16, winstepsize=8, verbose=False, backend="numba", tile_local=edge)
        out[edge] = run_aldvc(para, [pair["f"], pair["g"]], masks=masks, compute_strain=False)
    a, b = out[0].result_disp[0], out[112].result_disp[0]
    assert np.abs(a.U - b.U).max() < 1e-9
    assert np.array_equal(a.status, b.status) and np.allclose(a.zncc, b.zncc, equal_nan=True)
    truth = evaluate_at_nodes(pair["disp"], out[0].dvc_mesh.coordinates)
    good = a.status == STATUS_CONVERGED
    assert np.sqrt((((a.U - truth)[good]) ** 2).mean()) < 0.02  # and it is still the right answer


def test_tiling_never_hands_the_gradient_kernel_a_whole_volume(pair, monkeypatch):
    """The point of the exercise: the gradients of a box, not of the scan.

    Structural rather than a memory threshold -- it holds whatever the volume size and whatever the
    allocator does: every array the gradient kernel is given must fit inside the largest tile box.
    """
    import al_dvc.io.volume_ops as volume_ops

    para = dvcpara_default(winsize=16, winstepsize=8, verbose=False, backend="numba", gradient_mode="stored")
    mesh = _mesh(para)
    coords = np.round(mesh.coordinates).astype(np.int64)
    real = volume_ops.compute_gradients
    seen: list[tuple[int, int, int]] = []

    def spy(f):
        seen.append(tuple(int(s) for s in f.shape))
        return real(f)

    monkeypatch.setattr(volume_ops, "compute_gradients", spy)

    precompute_local_context(mesh, ReferenceSource(f=pair["fn"], mask=None, gradient_mode="stored"), para)
    assert seen == [SHAPE]  # untiled: one call, the whole volume

    seen.clear()
    tiled = replace(para, tile_local=96)
    plan = plan_tiles(mesh.grid_shape, coords, SHAPE, tiled, (8, 8, 8))
    precompute_local_context(mesh, ReferenceSource(f=pair["fn"], mask=None, gradient_mode="stored"), tiled)
    assert len(seen) == len(plan) > 1
    assert all(int(np.prod(shape)) <= plan.max_voxels for shape in seen), seen
    assert max(int(np.prod(shape)) for shape in seen) < int(np.prod(SHAPE))
