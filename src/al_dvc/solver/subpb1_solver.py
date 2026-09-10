"""ADMM subproblem 1 dispatcher: 3-DOF IC-GN with fixed F (MATLAB ``Subpb13.m``)."""

from __future__ import annotations

import logging
import time

import numpy as np
from numpy.typing import NDArray

from .._numba_compat import set_num_threads
from ..core.config import DVCPara
from ..core.data_structures import (
    STATUS_CONVERGED,
    STATUS_INVALID_SUBSET,
    STATUS_OUT_OF_BOUNDS,
    STATUS_SKIPPED,
    DVCMesh,
    LocalSolveInfo,
    ReferenceBundle,
)
from ..utils.outlier_detection import universal_median_test
from .interp_kernels import INTERP_MODE_BY_NAME
from .local_icgn import LocalContext, resolve_backend
from .tiling import as_source, plan_tiles, whole_box_tile

logger = logging.getLogger(__name__)


def _icgn_3dof_tile(backend, ctx, bundle, box, nodes, U_hat, F_hat, vdual, g, mode, mu, para, n_full, gain):
    """One block of nodes against one box: ``(U, n_iter, status, zncc)`` for those nodes only."""
    hx, hy, hz = ctx.half
    coords_t = ctx.coords_int[nodes] if box.is_whole else box.shift(ctx.coords_int[nodes])
    common = (
        coords_t,
        U_hat[nodes],
        F_hat[nodes],
        vdual[nodes],
        hx,
        hy,
        hz,
        bundle.f,
        bundle.gx,
        bundle.gy,
        bundle.gz,
        bundle.mask,
        box.crop(g),
        mode,
        ctx.H_all[nodes],
        ctx.meanf[nodes],
        ctx.bottomf[nodes],
        ctx.valid[nodes],
        float(mu),
        float(para.icgn_tol),
        float(para.icgn_dp_tol),
        int(para.icgn_max_iter),
        int(para.icgn_patience),
        ctx.stride,
        n_full,
        gain,
        bool(para.icgn_predictive_stop),
    )
    split_kw = ctx.split_args(nodes)
    if backend == "cuda":
        from .cuda_kernels import icgn_3dof_cuda

        return icgn_3dof_cuda(*common, **split_kw)
    if backend == "numba":
        from .numba_kernels import icgn_3dof_parallel

        return icgn_3dof_parallel(*common, **split_kw)
    from .reference_kernels import icgn_3dof_batch_np

    return icgn_3dof_batch_np(*common, **split_kw)


def subpb1_solver(
    ctx: LocalContext,
    ref: ReferenceBundle,
    g: NDArray[np.float32],
    U_hat: NDArray[np.float64],
    F_hat: NDArray[np.float64],
    vdual: NDArray[np.float64],
    mu: float,
    para: DVCPara,
    mesh: DVCMesh,
) -> tuple[NDArray[np.float64], LocalSolveInfo, NDArray[np.bool_]]:
    """Solve the local ADMM step for every node.

    Minimises ``ZNSSD(u; F_hat) + mu/2 |u - (U_hat + vdual)|^2`` w.r.t. the
    3 translations, starting from ``U_hat``.

    Returns ``(U, info, bad)``.
    """
    if para.n_threads > 0:
        set_num_threads(para.n_threads)
    N = ctx.n_nodes
    U_hat = np.ascontiguousarray(U_hat, dtype=np.float64).reshape(N, 3)
    F_hat = np.ascontiguousarray(F_hat, dtype=np.float64).reshape(N, 3, 3)
    vdual = np.ascontiguousarray(vdual, dtype=np.float64).reshape(N, 3)
    mode = INTERP_MODE_BY_NAME[para.interp_method]
    hx, hy, hz = ctx.half
    g = np.ascontiguousarray(g, dtype=np.float32)
    pattern, gain = ctx.noise_args(para)
    n_full = float(pattern[9, 9])

    t0 = time.perf_counter()
    backend = resolve_backend(para)
    source = as_source(ref, para.gradient_mode)
    plan = plan_tiles(mesh.grid_shape, ctx.coords_int, source.shape, para, ctx.half, disp=U_hat + vdual)
    U = np.zeros((N, 3), dtype=np.float64)
    n_iter = np.zeros(N, dtype=np.int32)
    status = np.zeros(N, dtype=np.int8)
    zncc = np.zeros(N, dtype=np.float64)
    open_face = np.zeros(N, dtype=bool)
    for nodes, box in plan:
        bundle = source.bundle_for(box)
        open_face[nodes] = box.has_open_face
        U[nodes], n_iter[nodes], status[nodes], zncc[nodes] = _icgn_3dof_tile(
            backend, ctx, bundle, box, nodes, U_hat, F_hat, vdual, g, mode, mu, para, n_full, gain
        )
        if backend == "cuda" and not plan.is_whole:
            from .cuda_kernels import clear_device_cache

            clear_device_cache()
    escaped = np.flatnonzero((status == STATUS_OUT_OF_BOUNDS) & open_face)
    if escaped.size:  # the halo was short for these, not the volume: retry them whole
        logger.warning("%d node(s) left their tile in the ADMM step; solving them on the whole volume", int(escaped.size))
        whole = whole_box_tile(source.shape)
        bundle = source.bundle_for(whole)
        U[escaped], n_iter[escaped], status[escaped], zncc[escaped] = _icgn_3dof_tile(
            backend, ctx, bundle, whole, escaped, U_hat, F_hat, vdual, g, mode, mu, para, n_full, gain
        )
    solve_time = time.perf_counter() - t0
    U = np.asarray(U, dtype=np.float64)
    status = np.asarray(status, dtype=np.int8)
    n_iter = np.asarray(n_iter, dtype=np.int32)
    zncc = np.asarray(zncc, dtype=np.float64)

    bad = status != STATUS_CONVERGED
    if para.local_outlier_threshold > 0:
        good = ~bad
        if good.sum() > 27:
            flag = universal_median_test(
                U.reshape(mesh.grid_shape + (3,)),
                good.reshape(mesh.grid_shape),
                para.local_outlier_threshold,
                edge_ok=mesh.edges_ok_grid(),
            )
            bad |= flag.ravel()
    n_bad = int(np.sum(bad & (status != STATUS_INVALID_SUBSET) & (status != STATUS_SKIPPED)))
    logger.info(
        "Subpb1 (3-DOF): %d/%d converged, %d bad, median ZNCC=%.4f, %.2fs",
        int(np.sum(status == STATUS_CONVERGED)),
        N,
        int(bad.sum()),
        float(np.nanmedian(zncc)) if np.isfinite(zncc).any() else float("nan"),
        solve_time,
    )
    # nodes the local solver could not handle keep the global estimate
    U_filled = U.copy()
    U_filled[bad] = U_hat[bad]
    info = LocalSolveInfo(n_iter=n_iter, status=status, zncc=zncc, solve_time=solve_time, n_bad=n_bad)
    return U_filled, info, bad
