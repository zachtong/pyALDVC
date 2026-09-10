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
    STATUS_OUT_OF_BOUNDS,
    STATUS_SKIPPED,
    DVCMesh,
    LocalSolveInfo,
    P_from_UF,
    ReferenceBundle,
    UF_from_P,
)
from ..mesh.grid_mesh import subset_valid_fraction
from ..utils.inpaint import fill_nan_grid
from ..utils.outlier_detection import universal_median_test
from .interp_kernels import INTERP_MODE_BY_NAME
from .tiling import as_source, merge_split, plan_tiles, whole_box_tile

logger = logging.getLogger(__name__)

MAX_SPLIT_BYTES = 512 * 1024 * 1024  # packed keep rows: beyond this the split is skipped for the run


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
    # subset splitting (``para.subset_split``): ``split_index[n]`` is the row of ``split_keep`` holding the
    # packed keep bits of node n (-1: the whole in-mask subset); ``split_fraction`` is kept / in-mask
    # sampled voxels (1.0 when nothing was cut, NaN for nodes the precompute rejected)
    split_index: NDArray[np.int64] | None = None
    split_keep: NDArray[np.uint8] | None = None
    split_fraction: NDArray[np.float32] | None = None

    def noise_args(self, para) -> tuple[NDArray[np.float64], float]:
        """``(pattern, gain)`` for the kernels: gain 0 keeps the stored Hessian."""
        from .uncertainty import STENCIL_NOISE_GAIN, noise_hessian_pattern

        pattern = self.noise_pattern if self.noise_pattern is not None else noise_hessian_pattern(self.half, self.stride)
        gain = float(STENCIL_NOISE_GAIN) if getattr(para, "icgn_noise_hessian", True) else 0.0
        return np.ascontiguousarray(pattern, dtype=np.float64), gain

    @property
    def n_nodes(self) -> int:
        return int(self.coords_int.shape[0])

    @property
    def n_split(self) -> int:
        """Nodes whose subset was cut to the component around its centre."""
        return 0 if self.split_index is None else int(np.count_nonzero(self.split_index >= 0))

    def split_args(self, nodes=None) -> dict:
        """Keyword arguments for the kernels (empty without subset splitting).

        ``nodes`` asks for one tile's rows, renumbered from zero: a tile's kernel indexes its own
        contiguous ``split_keep``, and the plan a tile came from is not the plan the rows were built
        with, so they are gathered rather than sliced.
        """
        if self.split_index is None:
            return {}
        if nodes is None:
            return {"split_index": self.split_index, "split_keep": self.split_keep}
        from .tiling import local_split

        index, keep = local_split(self.split_index, self.split_keep, nodes)
        return {"split_index": index, "split_keep": keep}


def describe_backend(para) -> str:
    """Human-readable backend line for the log: ``cuda (NVIDIA ...)`` or ``numba (CPU, 24 threads)``."""
    backend = resolve_backend(para)
    if backend == "cuda":
        from .cuda_kernels import device_name

        return f"cuda ({device_name()})"
    if backend == "numba":
        from al_dvc._numba_compat import get_num_threads

        return f"numba (CPU, {get_num_threads()} threads)"
    return "numpy (CPU reference kernels)"


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


def split_rows(ref: ReferenceBundle, para: DVCPara, coords_int, node_valid, half, stride, budget=None):
    """``(split_index, split_keep, n_keep, n_inmask)`` of subset splitting, or ``None`` when it is off.

    Candidates are the valid nodes whose full subset window contains a masked voxel (or leaves the
    box); each gets the 6-connected in-mask component around its centre as a packed keep row. The
    coordinates are expressed in ``ref``'s own frame, so a tile passes its cropped bundle and its
    shifted coordinates and gets rows numbered from zero.
    """
    if not bool(getattr(para, "subset_split", False)) or not ref.has_mask:
        return None  # nothing to split against: no mask means no boundary inside the volume
    from .numba_kernels import build_split_rows

    coords_int = np.asarray(coords_int, dtype=np.int64)
    frac = subset_valid_fraction(ref.mask, coords_int.astype(np.float64), para.winsize)
    cand = np.flatnonzero(np.asarray(node_valid, dtype=bool) & (frac < 1.0))
    N = coords_int.shape[0]
    split_index = np.full(N, -1, dtype=np.int64)
    if cand.size == 0:
        return split_index, np.zeros((1, 1), dtype=np.uint8), np.zeros(0, np.int64), np.zeros(0, np.int64)
    n_sampled = int(np.prod([(2 * h) // stride + 1 for h in half]))
    need = cand.size * ((n_sampled + 7) // 8)
    limit = MAX_SPLIT_BYTES if budget is None else budget
    if need > limit:
        logger.warning(
            "Subset splitting off for this reference: %d subsets touch a boundary and their keep rows would need "
            "%.1f GB (limit %.1f GB). Use a larger step, a smaller subset or subset_stride.",
            cand.size,
            need / 1e9,
            limit / 1e9,
        )
        return None
    # on-the-fly gradients read a 3-voxel stencil around every kept voxel: keep those away from the faces
    margin = 0 if ref.stored_gradients else 3
    rows, n_keep, n_inmask = build_split_rows(coords_int, cand, *half, stride, ref.mask, margin)
    split_index[cand] = np.arange(cand.size)
    return split_index, rows, n_keep, n_inmask


def precompute_local_context(mesh: DVCMesh, ref, para: DVCPara) -> LocalContext:
    """Hessians, normalisation and validity for every node of ``mesh``.

    ``ref`` is a :class:`~al_dvc.core.data_structures.ReferenceBundle` or a
    :class:`~al_dvc.solver.tiling.ReferenceSource`. With ``para.tile_local`` set, the nodes are
    solved in blocks against a box of the volume each, so the gradients of a box never exist at full
    size; with it at 0 the plan is a single whole-volume box and this is the code it always was.
    """
    _configure_threads(para)
    t0 = time.perf_counter()
    coords_int = np.round(mesh.coordinates).astype(np.int64)
    hx, hy, hz = (int(w) // 2 for w in para.winsize)
    half = (hx, hy, hz)
    stride = int(getattr(para, "subset_stride", 1))
    backend = resolve_backend(para)
    source = as_source(ref, para.gradient_mode)
    plan = plan_tiles(mesh.grid_shape, coords_int, source.shape, para, half)
    if plan.fallback:
        logger.warning("%s", plan.fallback)
    N = coords_int.shape[0]
    H_all = np.zeros((N, 12, 12), dtype=np.float64)
    L_all = np.zeros((N, 12, 12), dtype=np.float64)
    meanf = np.zeros(N, dtype=np.float64)
    bottomf = np.ones(N, dtype=np.float64)
    n_valid = np.zeros(N, dtype=np.int64)
    valid = np.zeros(N, dtype=bool)
    node_valid = np.asarray(mesh.node_valid, dtype=bool)
    split_parts: list[tuple] = []
    split_off = 0
    budget = MAX_SPLIT_BYTES
    split_used = True
    for nodes, box in plan:
        bundle = source.bundle_for(box)
        local = coords_int[nodes] if box.is_whole else box.shift(coords_int[nodes])
        split = split_rows(bundle, para, local, node_valid[nodes], half, stride, budget=budget)
        if split is None:
            split_used = False
            split_kw: dict = {}
        else:
            split_kw = {"split_index": split[0], "split_keep": split[1]}
            budget -= int(np.asarray(split[1]).nbytes)
        args = (
            local,
            hx,
            hy,
            hz,
            bundle.f,
            bundle.gx,
            bundle.gy,
            bundle.gz,
            bundle.mask,
            float(para.min_valid_ratio),
            float(para.hessian_cond_max),
            stride,
        )
        if backend == "cuda":
            from .cuda_kernels import clear_device_cache, precompute_nodes_cuda

            out = precompute_nodes_cuda(*args, **split_kw)
            if not plan.is_whole:
                clear_device_cache()  # the next box is a different array: do not hold this one
        elif backend == "numba":
            from .numba_kernels import precompute_nodes

            out = precompute_nodes(*args, **split_kw)
        else:
            from .reference_kernels import precompute_nodes_np

            out = precompute_nodes_np(*args, **split_kw)
        H_t, L_t, mf_t, bf_t, nv_t, vd_t = out
        H_all[nodes] = H_t
        L_all[nodes] = L_t
        meanf[nodes] = mf_t
        bottomf[nodes] = bf_t
        n_valid[nodes] = nv_t
        valid[nodes] = np.asarray(vd_t, dtype=bool)
        if split is not None:
            split_parts.append((nodes, split[0], split[1], split[2], split[3]))
        split_off += 1
    valid = np.asarray(valid, dtype=bool) & node_valid
    dt = time.perf_counter() - t0
    split_index = split_keep = split_fraction = None
    if split_used and split_parts:
        split_index, split_keep = merge_split([(n, li, keep) for n, li, keep, _k, _m in split_parts], N)
        split_fraction = np.ones(N, dtype=np.float32)
        for nodes, li, _keep, n_keep, n_inmask in split_parts:
            taken = np.flatnonzero(np.asarray(li) >= 0)
            if taken.size == 0:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = (np.asarray(n_keep) / np.maximum(np.asarray(n_inmask), 1)).astype(np.float32)
            split_fraction[np.asarray(nodes)[taken]] = frac
        split_fraction[~valid] = np.nan
        rows = np.flatnonzero(split_index >= 0)
        cut = rows[split_fraction[rows] < 1.0]
        logger.info(
            "Subset splitting: %d subsets touch a boundary, %d were cut (median %.0f%% of their voxels kept), %d rejected",
            rows.size,
            cut.size,
            100.0 * float(np.median(split_fraction[cut])) if cut.size else 100.0,
            int(np.count_nonzero(~valid[rows])),
        )
    logger.info(
        "Local precompute: %d nodes, %d valid (%.1f%%), %d tile(s), %.2fs",
        N,
        int(valid.sum()),
        100.0 * valid.mean() if valid.size else 0.0,
        len(plan),
        dt,
    )
    return LocalContext(
        coords_int=coords_int,
        half=half,
        H_all=H_all,
        L_all=L_all,
        meanf=meanf,
        bottomf=bottomf,
        n_valid=n_valid,
        valid=valid,
        precompute_time=dt,
        stride=stride,
        noise_pattern=_noise_pattern(hx, hy, hz, stride),
        split_index=split_index,
        split_keep=split_keep,
        split_fraction=split_fraction,
    )


def _icgn_12dof_tile(backend, ctx, bundle, box, nodes, P0, g, mode, para, pattern, gain):
    """One block of nodes against one box: ``(P, n_iter, status, zncc)`` for those nodes only.

    The kernels address the volumes relative to a node centre, so a box needs nothing but the crop
    and the same coordinates minus its origin -- the displacement in ``P[9:12]`` is untouched,
    because the reference and the deformed frame are cut from the same box.
    """
    hx, hy, hz = ctx.half
    coords_t = ctx.coords_int[nodes] if box.is_whole else box.shift(ctx.coords_int[nodes])
    common = (
        coords_t,
        P0[nodes],
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
    )
    tail = (
        ctx.meanf[nodes],
        ctx.bottomf[nodes],
        ctx.valid[nodes],
        float(para.icgn_tol),
        float(para.icgn_dp_tol),
        int(para.icgn_max_iter),
        int(para.icgn_patience),
        ctx.stride,
    )
    predictive = bool(para.icgn_predictive_stop)
    split_kw = ctx.split_args(nodes)
    if backend == "cuda":
        from .cuda_kernels import icgn_12dof_cuda

        return icgn_12dof_cuda(*common, ctx.L_all[nodes], *tail, ctx.H_all[nodes], pattern, gain, predictive, **split_kw)
    if backend == "numba":
        from .numba_kernels import icgn_12dof_parallel

        return icgn_12dof_parallel(*common, ctx.L_all[nodes], *tail, ctx.H_all[nodes], pattern, gain, predictive, **split_kw)
    from .reference_kernels import icgn_12dof_batch_np

    return icgn_12dof_batch_np(*common, ctx.H_all[nodes], *tail, pattern, gain, predictive, **split_kw)


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
    source = as_source(ref, para.gradient_mode)
    plan = plan_tiles(mesh.grid_shape, ctx.coords_int, source.shape, para, ctx.half, disp=U0)
    P = np.zeros((N, 12), dtype=np.float64)
    n_iter = np.zeros(N, dtype=np.int32)
    status = np.zeros(N, dtype=np.int8)
    zncc = np.zeros(N, dtype=np.float64)
    open_face = np.zeros(N, dtype=bool)  # this node's box was cut by the tile, not by the volume
    for nodes, box in plan:
        bundle = source.bundle_for(box)
        open_face[nodes] = box.has_open_face
        P[nodes], n_iter[nodes], status[nodes], zncc[nodes] = _icgn_12dof_tile(
            backend, ctx, bundle, box, nodes, P0, g, mode, para, pattern, gain
        )
        if backend == "cuda" and not plan.is_whole:
            from .cuda_kernels import clear_device_cache

            clear_device_cache()  # the next box is a different array; do not pin this one
    # a subset that left its BOX rather than the volume means the halo was too small: retry it whole,
    # so a halo that is one voxel short costs time instead of a wrong answer
    escaped = np.flatnonzero((status == STATUS_OUT_OF_BOUNDS) & open_face)
    if escaped.size:
        logger.warning(
            "%d node(s) left their tile instead of the volume (the halo was short); solving them on the whole volume",
            int(escaped.size),
        )
        whole = whole_box_tile(source.shape)
        bundle = source.bundle_for(whole)
        P[escaped], n_iter[escaped], status[escaped], zncc[escaped] = _icgn_12dof_tile(
            backend, ctx, bundle, whole, escaped, P0, g, mode, para, pattern, gain
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
            flag = universal_median_test(
                U_grid, good.reshape(mesh.grid_shape), para.local_outlier_threshold, edge_ok=mesh.edges_ok_grid()
            )
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
    edge_ok = mesh.edges_ok_grid()  # a cut mesh is filled from each side separately
    U_out = U.copy()
    for c in range(3):
        comp = U[:, c].reshape(shape).copy()
        comp[bad.reshape(shape)] = np.nan
        U_out[:, c] = fill_nan_grid(comp, edge_ok=edge_ok).ravel()
    F_out = None
    if F is not None:
        F_out = F.copy()
        for i in range(3):
            for j in range(3):
                comp = F[:, i, j].reshape(shape).copy()
                comp[bad.reshape(shape)] = np.nan
                F_out[:, i, j] = fill_nan_grid(comp, edge_ok=edge_ok).ravel()
    return U_out, F_out
