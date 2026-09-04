"""CUDA backend: parity with the CPU kernels, masks, gradient modes, stride, and the backend switch.

Skipped when numba-cuda or a usable NVIDIA GPU is missing (CPU-only installs
and the CI runners); the backend-selection tests at the end run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.data_structures import STATUS_CONVERGED, STATUS_INVALID_SUBSET, STATUS_OUT_OF_BOUNDS, P_from_UF
from al_dvc.core.pipeline import run_aldvc
from al_dvc.io.volume_ops import build_reference_bundle, normalize_volume, prepare_deformed
from al_dvc.mesh.grid_mesh import build_grid_axes, mesh_setup
from al_dvc.solver import cuda_kernels as ck
from al_dvc.solver import numba_kernels as nk
from al_dvc.solver.interp_kernels import INTERP_MODE_BY_NAME
from al_dvc.solver.local_icgn import precompute_local_context, resolve_backend
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

SHAPE = (56, 60, 64)
F_TRUE = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])
T_TRUE = (0.7, -0.4, 0.3)
gpu = pytest.mark.skipif(not ck.cuda_available(), reason=f"CUDA backend unavailable ({ck.unavailable_reason()})")


def _case(noise=0.0, ref_mask=False, def_mask=False, gradient_mode="stored", interp="cubic", stride=1, winsize=24):
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=37)
    fn = affine_displacement(F_TRUE, T_TRUE, centre)
    dfm = warp_volume_lagrangian(ref, fn)
    if noise > 0:
        rng = np.random.default_rng(7)
        ref = ref + rng.normal(0, noise, ref.shape)
        dfm = dfm + rng.normal(0, noise, dfm.shape)
    para = dvcpara_default(
        winsize=winsize, winstepsize=12, verbose=False, gradient_mode=gradient_mode, interp_method=interp, subset_stride=stride
    )
    f, g = normalize_volume(ref), normalize_volume(dfm)
    rmask = None
    if ref_mask:
        rmask = np.ones(SHAPE, dtype=bool)
        rmask[:, :, :20] = False  # a slab of the reference is not material
    dmask = None
    if def_mask:
        dmask = np.ones(SHAPE, dtype=bool)
        dmask[:22, :, :] = False
    bundle = build_reference_bundle(f, rmask, gradient_mode)
    mesh = mesh_setup(*build_grid_axes(para.voi, SHAPE, para.winsize, para.winstepsize))
    ctx = precompute_local_context(mesh, bundle, para)
    U_true = np.stack(fn(mesh.coordinates[:, 0], mesh.coordinates[:, 1], mesh.coordinates[:, 2]), axis=-1).reshape(-1, 3)
    return {
        "ref": ref,
        "dfm": dfm,
        "bundle": bundle,
        "g": prepare_deformed(g, interp, dmask),
        "ctx": ctx,
        "para": para,
        "U_true": U_true,
        "mesh": mesh,
    }


def _both_12(case, start=0.3):
    b, ctx, para = case["bundle"], case["ctx"], case["para"]
    pattern, gain = ctx.noise_args(para)
    mode = INTERP_MODE_BY_NAME[para.interp_method]
    P0 = P_from_UF(case["U_true"] + start, np.zeros((ctx.n_nodes, 3, 3)))
    hx, hy, hz = ctx.half
    args = (
        ctx.coords_int,
        P0,
        hx,
        hy,
        hz,
        b.f,
        b.gx,
        b.gy,
        b.gz,
        b.mask,
        case["g"],
        mode,
        ctx.L_all,
        ctx.meanf,
        ctx.bottomf,
        ctx.valid,
        1e-2,
        1e-3,
        100,
        5,
        ctx.stride,
        ctx.H_all,
        pattern,
        gain,
        True,
    )
    return nk.icgn_12dof_parallel(*args), ck.icgn_12dof_cuda(*args)


def _both_3(case):
    b, ctx, para = case["bundle"], case["ctx"], case["para"]
    pattern, gain = ctx.noise_args(para)
    mode = INTERP_MODE_BY_NAME[para.interp_method]
    N = ctx.n_nodes
    F_fixed = np.tile(F_TRUE, (N, 1, 1))
    U_old = case["U_true"] + 0.15
    vdual = np.zeros((N, 3))
    hx, hy, hz = ctx.half
    args = (
        ctx.coords_int,
        U_old,
        F_fixed,
        vdual,
        hx,
        hy,
        hz,
        b.f,
        b.gx,
        b.gy,
        b.gz,
        b.mask,
        case["g"],
        mode,
        ctx.H_all,
        ctx.meanf,
        ctx.bottomf,
        ctx.valid,
        1e-3,
        1e-2,
        1e-3,
        100,
        5,
        ctx.stride,
        float(pattern[9, 9]),
        gain,
        True,
    )
    return nk.icgn_3dof_parallel(*args), ck.icgn_3dof_cuda(*args)


def _assert_parity(cpu, gpu, tol=2e-3):
    Pc, itc, stc, zc = cpu
    Pg, itg, stg, zg = gpu
    assert np.array_equal(stc, stg), "status codes differ"
    assert np.mean(itc == itg) > 0.95, "iteration counts differ at more than 5 % of the nodes"
    ok = stc == STATUS_CONVERGED
    if Pc.ndim == 2 and Pc.shape[1] == 12:
        assert np.max(np.abs(Pg[ok, 9:] - Pc[ok, 9:])) < tol
        assert np.max(np.abs(Pg[ok, :9] - Pc[ok, :9])) < tol
    else:
        assert np.max(np.abs(Pg[ok] - Pc[ok])) < tol
    finite = np.isfinite(zc) & np.isfinite(zg)
    assert np.max(np.abs(zg[finite] - zc[finite])) < 1e-3


@gpu
@pytest.mark.parametrize(
    "kw", [{}, {"noise": 0.03}, {"stride": 2, "winsize": 32}, {"interp": "bspline"}, {"gradient_mode": "on_the_fly"}]
)
def test_kernels_match_cpu(kw):
    case = _case(**kw)
    cpu12, gpu12 = _both_12(case)
    _assert_parity(cpu12, gpu12)
    assert np.mean(gpu12[2] == STATUS_CONVERGED) > 0.9
    cpu3, gpu3 = _both_3(case)
    _assert_parity(cpu3, gpu3)


@gpu
def test_masks_match_cpu():
    case = _case(ref_mask=True, def_mask=True)
    cpu12, gpu12 = _both_12(case)
    _assert_parity(cpu12, gpu12)
    st = gpu12[2]
    assert (st == STATUS_INVALID_SUBSET).any() or (st == STATUS_OUT_OF_BOUNDS).any() or (st == STATUS_CONVERGED).all()
    assert np.mean(st == STATUS_CONVERGED) > 0.25  # the slab masks invalidate many subsets on purpose
    cpu3, gpu3 = _both_3(case)
    _assert_parity(cpu3, gpu3)


@gpu
def test_out_of_bounds_and_nan_start():
    case = _case()
    ctx = case["ctx"]
    P0 = P_from_UF(case["U_true"] + 0.3, np.zeros((ctx.n_nodes, 3, 3)))
    P0[0, 9] = np.nan
    P0[1, 9] = 500.0  # warps far outside the volume
    cpu12, gpu12 = _both_12(case)
    b, para = case["bundle"], case["para"]
    pattern, gain = ctx.noise_args(para)
    hx, hy, hz = ctx.half
    args = (
        ctx.coords_int,
        P0,
        hx,
        hy,
        hz,
        b.f,
        b.gx,
        b.gy,
        b.gz,
        b.mask,
        case["g"],
        1,
        ctx.L_all,
        ctx.meanf,
        ctx.bottomf,
        ctx.valid,
        1e-2,
        1e-3,
        100,
        5,
        1,
        ctx.H_all,
        pattern,
        gain,
        True,
    )
    Pg, itg, stg, zg = ck.icgn_12dof_cuda(*args)
    Pc, itc, stc, zc = nk.icgn_12dof_parallel(*args)
    assert stg[0] == stc[0] and stg[1] == stc[1] == STATUS_OUT_OF_BOUNDS
    assert np.array_equal(stc, stg)


@gpu
def test_pipeline_cuda_matches_numba():
    case = _case()
    base = dict(case["para"].__dict__, search_radius=4, admm_max_iter=3)
    res_c = run_aldvc(dvcpara_default(**{**base, "backend": "numba"}), [case["ref"], case["dfm"]])
    res_g = run_aldvc(dvcpara_default(**{**base, "backend": "cuda"}), [case["ref"], case["dfm"]])
    fc, fg = res_c.result_disp[0], res_g.result_disp[0]
    ok = (fc.status == STATUS_CONVERGED) & (fg.status == STATUS_CONVERGED)
    assert ok.mean() > 0.9
    assert np.max(np.abs(fg.U[ok] - fc.U[ok])) < 2e-3
    assert np.nanmax(np.abs(fg.U_std[ok] - fc.U_std[ok])) < 1e-3
    assert resolve_backend(dvcpara_default(backend="auto")) == "cuda"
    assert resolve_backend(dvcpara_default(backend="numba")) == "numba"


@gpu
def test_chunked_launches_give_the_same_result():
    case = _case()
    cpu12, gpu_full = _both_12(case)
    b, ctx, para = case["bundle"], case["ctx"], case["para"]
    pattern, gain = ctx.noise_args(para)
    P0 = P_from_UF(case["U_true"] + 0.3, np.zeros((ctx.n_nodes, 3, 3)))
    hx, hy, hz = ctx.half
    seen = []
    Pg, itg, stg, zg = ck.icgn_12dof_cuda(
        ctx.coords_int,
        P0,
        hx,
        hy,
        hz,
        b.f,
        b.gx,
        b.gy,
        b.gz,
        b.mask,
        case["g"],
        1,
        ctx.L_all,
        ctx.meanf,
        ctx.bottomf,
        ctx.valid,
        1e-2,
        1e-3,
        100,
        5,
        1,
        ctx.H_all,
        pattern,
        gain,
        True,
        chunk=7,
        progress_fn=seen.append,
    )
    np.testing.assert_allclose(Pg, gpu_full[0], atol=1e-12)
    assert seen and seen[-1] == 1.0 and len(seen) >= 2


def test_backend_resolution_without_cuda(monkeypatch):
    monkeypatch.setattr(ck, "_available", False)
    monkeypatch.setattr(ck, "_unavailable_reason", "forced off")
    assert resolve_backend(dvcpara_default(backend="auto")) == "numba"
    with pytest.raises(RuntimeError, match="forced off"):
        resolve_backend(dvcpara_default(backend="cuda"))
    assert resolve_backend(dvcpara_default(backend="numpy")) == "numpy"
    with pytest.raises(ValueError):
        dvcpara_default(backend="opencl")


@gpu
def test_precompute_matches_cpu():
    case = _case(ref_mask=True)
    b, ctx = case["bundle"], case["ctx"]
    Hc, Lc, mfc, bfc, nvc, validc = nk.precompute_nodes(ctx.coords_int, *ctx.half, b.f, b.gx, b.gy, b.gz, b.mask, 0.5, 1e12, 1)
    Hg, Lg, mfg, bfg, nvg, validg = ck.precompute_nodes_cuda(
        ctx.coords_int, *ctx.half, b.f, b.gx, b.gy, b.gz, b.mask, 0.5, 1e12, 1
    )
    assert np.array_equal(validc, validg) and np.array_equal(nvc, nvg)
    ok = validc
    np.testing.assert_allclose(Hg[ok], Hc[ok], rtol=1e-4, atol=1e-3 * np.abs(Hc[ok]).max())
    np.testing.assert_allclose(mfg[ok], mfc[ok], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(bfg[ok], bfc[ok], rtol=1e-5)
    np.testing.assert_allclose(Lg[ok], Lc[ok], rtol=1e-3, atol=1e-3 * np.abs(Lc[ok]).max())
    # the pipeline context on the GPU backend uses it
    para = dvcpara_default(**{**case["para"].__dict__, "backend": "cuda"})
    ctx_g = precompute_local_context(case["mesh"], b, para)
    assert np.array_equal(ctx_g.valid, ctx.valid)
    np.testing.assert_allclose(ctx_g.bottomf[ok], ctx.bottomf[ok], rtol=1e-5)
