"""Local subset DVC dispatcher (Section 4, MATLAB ``LocalICGN3.m``).

Builds the per-node context once per (reference frame, mesh) and runs the
12-DOF IC-GN over all nodes, then cleans failed / outlying nodes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._numba_compat import HAS_NUMBA, set_num_threads
from ..core.config import DVCPara
from ..core.data_structures import (
    STATUS_CONVERGED,
    STATUS_INVALID_SUBSET,
    STATUS_SKIPPED,
    DVCMesh,
    LocalSolveInfo,
    P_from_UF,
    ReferenceBundle,
    UF_from_P,
)
from ..utils.inpaint import fill_nan_grid
from ..utils.outlier_detection import universal_median_test
from .interp_kernels import INTERP_MODE_BY_NAME

logger = logging.getLogger(__name__)


@dataclass
class LocalContext:
    """Per-(reference, mesh) precomputed data for the local solvers."""

    coords_int: NDArray[np.int64]  # (N, 3) [x, y, z]
    half: tuple[int, int, int]  # (hx, hy, hz)
    H_all: NDArray[np.float64]  # (N, 12, 12)
    L_all: NDArray[np.float64]  # (N, 12, 12) Cholesky factors
    meanf: NDArray[np.float64]
    bottomf: NDArray[np.float64]
    n_valid: NDArray[np.int64]
    valid: NDArray[np.bool_]  # solver-valid nodes
    precompute_time: float
    stride: int = 1  # subset sampling stride used for H, meanf, bottomf and n_valid
    noise_pattern: NDArray[np.float64] | None = None  # (12, 12) noise Hessian pattern of the sampled subset

    def noise_args(self, para) -> tuple[NDArray[np.float64], float]:
        """``(pattern, gain)`` for the kernels: gain 0 keeps the stored Hessian."""
        from .uncertainty import STENCIL_NOISE_GAIN, noise_hessian_pattern

        pattern = self.noise_pattern if self.noise_pattern is not None else noise_hessian_pattern(self.half, self.stride)
        gain = float(STENCIL_NOISE_GAIN) if getattr(para, "icgn_noise_hessian", True) else 0.0
        return np.ascontiguousarray(pattern, dtype=np.float64), gain

    @property
    def n_nodes(self) -> int:
        return int(self.coords_int.shape[0])


def resolve_backend(para: DVCPara) -> str:
    """``"cuda"``, ``"numba"`` or ``"numpy"`` for this parameter set on this machine."""
    backend = getattr(para, "backend", "auto")
    if backend in ("auto", "cuda"):
        from .cuda_kernels import cuda_available, unavailable_reason

        if cuda_available():
            return "cuda"
        if backend == "cuda":
            raise RuntimeError(f"backend='cuda' requested but CUDA is not usable ({unavailable_reason()})")
        backend = "numba"
    if backend == "numba" and not HAS_NUMBA:
        return "numpy"
    return backend


def _use_numba(para: DVCPara) -> bool:
    return resolve_backend(para) in ("numba", "cuda")


def _configure_threads(para: DVCPara) -> None:
    if para.n_threads > 0:
        set_num_threads(para.n_threads)


def _noise_pattern(hx: int, hy: int, hz: int, stride: int) -> NDArray[np.float64]:
    from .uncertainty import noise_hessian_pattern

    return np.ascontiguousarray(noise_hessian_pattern((hx, hy, hz), stride), dtype=np.float64)


def precompute_local_context(mesh: DVCMesh, ref: ReferenceBundle, para: DVCPara) -> LocalContext:
    """Hessians, normalisation and validity for every node of ``mesh``."""
    _configure_threads(para)
    t0 = time.perf_counter()
    coords_int = np.round(mesh.coordinates).astype(np.int64)
    hx, hy, hz = (int(w) // 2 for w in para.winsize)
    stride = int(getattr(para, "subset_stride", 1))
    backend = resolve_backend(para)
    if backend == "cuda":
        from .cuda_kernels import precompute_nodes_cuda

        H_all, L_all, meanf, bottomf, n_valid, valid = precompute_nodes_cuda(
            coords_int,
            hx,
            hy,
            hz,
            ref.f,
            ref.gx,
            ref.gy,
            ref.gz,
            ref.mask,
            float(para.min_valid_ratio),
            float(para.hessian_cond_max),
            stride,
        )
    elif backend == "numba":
        from .numba_kernels import precompute_nodes

        H_all, L_all, meanf, bottomf, n_valid, valid = precompute_nodes(
            coords_int,
            hx,
            hy,
            hz,
            ref.f,
            ref.gx,
            ref.gy,
            ref.gz,
            ref.mask,
            float(para.min_valid_ratio),
            float(para.hessian_cond_max),
            stride,
        )
    else:
        from .reference_kernels import precompute_nodes_np

        H_all, L_all, meanf, bottomf, n_valid, valid = precompute_nodes_np(
            coords_int,
            hx,
            hy,
            hz,
            ref.f,
            ref.gx,
            ref.gy,
            ref.gz,
            ref.mask,
            float(para.min_valid_ratio),
            float(para.hessian_cond_max),
            stride,
        )
    valid = np.asarray(valid, dtype=bool) & np.asarray(mesh.node_valid, dtype=bool)
    dt = time.perf_counter() - t0
    logger.info(
        "Local precompute: %d nodes, %d valid (%.1f%%), %.2fs",
        coords_int.shape[0],
        int(valid.sum()),
        100.0 * valid.mean() if valid.size else 0.0,
        dt,
    )
    return LocalContext(
        coords_int=coords_int,
        half=(hx, hy, hz),
        H_all=H_all,
        L_all=L_all,
        meanf=np.asarray(meanf),
        bottomf=np.asarray(bottomf),
        n_valid=np.asarray(n_valid),
        valid=valid,
        precompute_time=dt,
        stride=stride,
        noise_pattern=_noise_pattern(hx, hy, hz, stride),
    )


def local_icgn(
    ctx: LocalContext,
    ref: ReferenceBundle,
    g: NDArray[np.float32],
    U0: NDArray[np.float64],
    para: DVCPara,
    mesh: DVCMesh,
    F0: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], LocalSolveInfo, NDArray[np.bool_]]:
    """12-DOF IC-GN over all nodes starting from ``U0`` (and optional ``F0``).

    Args:
        g: deformed volume prepared for ``para.interp_method`` (see
            :func:`al_dvc.io.volume_ops.prepare_deformed`).
    Returns:
        ``(U, F, info, bad)`` -- ``U`` (N,3), ``F`` (N,3,3) with failed nodes
        filled by inpainting; ``bad`` marks the nodes that were filled.
    """
    _configure_threads(para)
    N = ctx.n_nodes
    U0 = np.asarray(U0, dtype=np.float64).reshape(N, 3)
    if F0 is None:
        F0 = np.zeros((N, 3, 3), dtype=np.float64)
    P0 = P_from_UF(U0, np.asarray(F0, dtype=np.float64).reshape(N, 3, 3))
    mode = INTERP_MODE_BY_NAME[para.interp_method]
    hx, hy, hz = ctx.half
    g = np.ascontiguousarray(g, dtype=np.float32)
    pattern, gain = ctx.noise_args(para)

    t0 = time.perf_counter()
    backend = resolve_backend(para)
    if backend == "cuda":
        from .cuda_kernels import icgn_12dof_cuda

        P, n_iter, status, zncc = icgn_12dof_cuda(
            ctx.coords_int,
            P0,
            hx,
            hy,
            hz,
            ref.f,
            ref.gx,
            ref.gy,
            ref.gz,
            ref.mask,
            g,
            mode,
            ctx.L_all,
            ctx.meanf,
            ctx.bottomf,
            ctx.valid,
            float(para.icgn_tol),
            float(para.icgn_dp_tol),
            int(para.icgn_max_iter),
            int(para.icgn_patience),
            ctx.stride,
            ctx.H_all,
            pattern,
            gain,
            bool(para.icgn_predictive_stop),
        )
    elif backend == "numba":
        from .numba_kernels import icgn_12dof_parallel

        P, n_iter, status, zncc = icgn_12dof_parallel(
            ctx.coords_int,
            P0,
            hx,
            hy,
            hz,
            ref.f,
            ref.gx,
            ref.gy,
            ref.gz,
            ref.mask,
            g,
            mode,
            ctx.L_all,
            ctx.meanf,
            ctx.bottomf,
            ctx.valid,
            float(para.icgn_tol),
            float(para.icgn_dp_tol),
            int(para.icgn_max_iter),
            int(para.icgn_patience),
            ctx.stride,
            ctx.H_all,
            pattern,
            gain,
            bool(para.icgn_predictive_stop),
        )
    else:
        from .reference_kernels import icgn_12dof_batch_np

        P, n_iter, status, zncc = icgn_12dof_batch_np(
            ctx.coords_int,
            P0,
            hx,
            hy,
            hz,
            ref.f,
            ref.gx,
            ref.gy,
            ref.gz,
            ref.mask,
            g,
            mode,
            ctx.H_all,
            ctx.meanf,
            ctx.bottomf,
            ctx.valid,
            float(para.icgn_tol),
            float(para.icgn_dp_tol),
            int(para.icgn_max_iter),
            int(para.icgn_patience),
            ctx.stride,
            pattern,
            gain,
            bool(para.icgn_predictive_stop),
        )
    solve_time = time.perf_counter() - t0

    U, F = UF_from_P(np.asarray(P))
    status = np.asarray(status, dtype=np.int8)
    n_iter = np.asarray(n_iter, dtype=np.int32)
    zncc = np.asarray(zncc, dtype=np.float64)

    bad = status != STATUS_CONVERGED
    # median test on the converged nodes (universal outlier detection)
    if para.local_outlier_threshold > 0:
        good = ~bad
        if good.sum() > 27:
            U_grid = U.reshape(mesh.grid_shape + (3,))
            flag = universal_median_test(U_grid, good.reshape(mesh.grid_shape), para.local_outlier_threshold)
            bad |= flag.ravel()
    n_bad = int(np.sum(bad & (status != STATUS_INVALID_SUBSET) & (status != STATUS_SKIPPED)))
    logger.info(
        "Local IC-GN: %d/%d converged, %d bad (%.1f%%), median ZNCC=%.4f, %.2fs",
        int(np.sum(status == STATUS_CONVERGED)),
        N,
        int(bad.sum()),
        100.0 * bad.mean(),
        float(np.nanmedian(zncc)) if np.isfinite(zncc).any() else float("nan"),
        solve_time,
    )

    U, F = fill_bad_nodes(U, F, bad, mesh)
    info = LocalSolveInfo(n_iter=n_iter, status=status, zncc=zncc, solve_time=solve_time, n_bad=n_bad)
    return U, F, info, bad


def fill_bad_nodes(
    U: NDArray[np.float64],
    F: NDArray[np.float64] | None,
    bad: NDArray[np.bool_],
    mesh: DVCMesh,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Replace ``bad`` nodes by spring-model inpainting on the node grid."""
    if not bad.any():
        return U, F
    if bad.all():
        logger.warning("All nodes are bad; filling with zeros.")
        U = np.zeros_like(U)
        if F is not None:
            F = np.zeros_like(F)
        return U, F
    shape = mesh.grid_shape
    U_out = U.copy()
    for c in range(3):
        comp = U[:, c].reshape(shape).copy()
        comp[bad.reshape(shape)] = np.nan
        U_out[:, c] = fill_nan_grid(comp).ravel()
    F_out = None
    if F is not None:
        F_out = F.copy()
        for i in range(3):
            for j in range(3):
                comp = F[:, i, j].reshape(shape).copy()
                comp[bad.reshape(shape)] = np.nan
                F_out[:, i, j] = fill_nan_grid(comp).ravel()
    return U_out, F_out
