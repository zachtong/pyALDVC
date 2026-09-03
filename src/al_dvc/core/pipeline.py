"""Full AL-DVC pipeline (MATLAB ``main_ALDVC.m`` Sections 2-8).

Section 2  normalise volumes, build the node grid
Section 3  initial guess (global shift + pyramid NCC), outlier cleaning
Section 4  local 12-DOF IC-GN (subproblem 1, first pass)
Section 5  global step (subproblem 2) with beta auto-tuning
Section 6  ADMM iterations (3-DOF local / global / dual update)
Section 7  cumulative displacement composition (incremental tracking)
Section 8  strain
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .._numba_compat import set_num_threads
from ..io.volume_ops import (
    ListVolumeProvider,
    build_reference_bundle,
    prepare_deformed,
    presmooth_volume,
)
from ..mesh.grid_mesh import apply_mask_to_mesh, build_grid_axes, mesh_setup
from ..solver.beta_tuning import auto_tune_beta
from ..solver.global_operators import build_global_operators, nodal_gradient
from ..solver.init_disp import compute_initial_guess
from ..solver.local_icgn import local_icgn, precompute_local_context
from ..solver.subpb1_solver import subpb1_solver
from ..solver.subpb2_solver import build_global_system, solve_subpb2
from ..solver.uncertainty import displacement_uncertainty
from ..strain.compute_strain import compute_strain as _compute_strain
from ..utils.grid_interp import interp_grid_field
from ..utils.validation import validate_para_against_volume
from .config import DVCPara
from .data_structures import (
    ADMMInfo,
    DVCMesh,
    FrameResult,
    FrameSchedule,
    PipelineResult,
    StrainResult,
    VolumeProvider,
)

logger = logging.getLogger(__name__)

_REF_CACHE_SIZE = 2


class RunCancelled(RuntimeError):
    """Raised internally when ``stop_fn`` returns True."""


def _default_progress(frac: float, msg: str) -> None:
    logger.info("[%3.0f%%] %s", 100.0 * frac, msg)


def _rms(a: NDArray[np.float64]) -> float:
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def _resolve_provider(volumes, para: DVCPara, masks) -> VolumeProvider:
    if hasattr(volumes, "get_normalized"):
        return volumes  # type: ignore[return-value]
    return ListVolumeProvider(list(volumes), para.voi, masks)


def run_aldvc(
    para: DVCPara,
    volumes,
    masks=None,
    progress_fn: Callable[[float, str], None] | None = None,
    stop_fn: Callable[[], bool] | None = None,
    compute_strain: bool = True,
) -> PipelineResult:
    """Execute the AL-DVC pipeline on a sequence of volumes.

    Args:
        para: validated parameters (``dvcpara_default``).
        volumes: list of ``(nz, ny, nx)`` arrays (any dtype) or a
            ``VolumeProvider``. ``volumes[0]`` is the first reference.
        masks: optional list of boolean ``(nz, ny, nx)`` arrays (one per
            frame, entries may be ``None``); ignored when ``volumes`` is a
            provider (the provider supplies masks).
        progress_fn: ``(fraction, message)`` callback.
        stop_fn: returns True to cancel; completed frames are returned.
        compute_strain: run Section 8.

    Returns:
        :class:`PipelineResult`.
    """
    t_start = time.perf_counter()
    progress = progress_fn or _default_progress
    should_stop = stop_fn or (lambda: False)
    timings: dict[str, float] = {}

    provider = _resolve_provider(volumes, para, masks)
    n_frames = len(provider)
    if n_frames < 2:
        raise ValueError(f"At least 2 volumes are required (got {n_frames}).")
    shape = tuple(int(s) for s in provider.shape)
    validate_para_against_volume(para, shape)  # type: ignore[arg-type]
    para = replace(para, voi=provider.clamped_voi, volume_shape=shape)
    if para.n_threads > 0:
        set_num_threads(para.n_threads)

    schedule = para.frame_schedule
    if schedule is None:
        schedule = FrameSchedule.from_mode(para.reference_mode, n_frames)
    elif len(schedule) != n_frames - 1:
        raise ValueError(f"frame_schedule length {len(schedule)} != n_frames-1 = {n_frames - 1}")

    # ------------------------------------------------------------------
    # Section 2: node grid (geometry is shared by every frame)
    # ------------------------------------------------------------------
    progress(0.0, "Section 2: building node grid")
    x0, y0, z0 = build_grid_axes(para.voi, shape, para.winsize, para.winstepsize)  # type: ignore[arg-type]
    base_mesh = mesh_setup(x0, y0, z0)
    logger.info(
        "Node grid %s (%d nodes), spacing %s, winsize %s, volume %s",
        base_mesh.grid_shape,
        base_mesh.n_nodes,
        base_mesh.spacing,
        para.winsize,
        shape,
    )

    ref_cache: OrderedDict[int, dict] = OrderedDict()
    beta_by_ref: dict[int, float] = {}
    results: list[FrameResult | None] = [None] * (n_frames - 1)
    mesh_by_frame: list[DVCMesh | None] = [None] * (n_frames - 1)
    stopped_early = False
    stopped_at = None
    stop_reason = ""

    def get_reference(ref_idx: int) -> dict:
        if ref_idx in ref_cache:
            ref_cache.move_to_end(ref_idx)
            return ref_cache[ref_idx]
        t0 = time.perf_counter()
        f = presmooth_volume(provider.get_normalized(ref_idx), para.prefilter_sigma)
        mask = provider.get_mask(ref_idx)
        bundle = build_reference_bundle(f, mask)
        mesh = apply_mask_to_mesh(base_mesh, bundle.mask if mask is not None else None, para.winsize, para.min_valid_ratio)
        ctx = precompute_local_context(mesh, bundle, para)
        mesh.node_valid = ctx.valid.copy()
        ops = None
        if para.use_global_step:
            try:
                ops = build_global_operators(mesh, para.subpb2_method, para.gauss_pt_order)
            except ValueError as exc:
                logger.warning("Global step disabled for reference %d: %s", ref_idx, exc)
        entry = {"bundle": bundle, "mesh": mesh, "ctx": ctx, "ops": ops, "time": time.perf_counter() - t0}
        ref_cache[ref_idx] = entry
        while len(ref_cache) > _REF_CACHE_SIZE:
            ref_cache.popitem(last=False)
        timings["reference_precompute"] = timings.get("reference_precompute", 0.0) + entry["time"]
        return entry

    # ------------------------------------------------------------------
    # Sections 3-6 per frame pair
    # ------------------------------------------------------------------
    n_pairs = n_frames - 1
    prev_ref: int | None = None
    prev_U: NDArray[np.float64] | None = None
    try:
        for k in range(1, n_frames):
            if should_stop():
                raise RunCancelled("Computation cancelled by user.")
            base = (k - 1) / n_pairs * 0.9
            span = 0.9 / n_pairs
            ref_idx = schedule.parent(k)
            progress(base, f"Frame {k}/{n_pairs}: reference {ref_idx}")
            t_frame = time.perf_counter()

            ref = get_reference(ref_idx)
            bundle, mesh, ctx, ops = ref["bundle"], ref["mesh"], ref["ctx"], ref["ops"]
            g_norm = presmooth_volume(provider.get_normalized(k), para.prefilter_sigma)
            g_mask = provider.get_mask(k)
            g_prep = prepare_deformed(g_norm, para.interp_method, mask=g_mask)
            if g_mask is not None:  # the NCC search sees the masked voxels as featureless (0 = mean after normalisation)
                g_norm = np.where(g_mask, g_norm, np.float32(0.0)).astype(np.float32)

            # --- Section 3: initial guess ---
            t0 = time.perf_counter()
            previous = prev_U if (prev_ref == ref_idx and prev_U is not None) else None
            U0, init_info = compute_initial_guess(bundle.f, g_norm, mesh, para, previous=previous)
            timings["init_guess"] = timings.get("init_guess", 0.0) + time.perf_counter() - t0
            progress(base + 0.15 * span, f"Frame {k}: initial guess ({init_info.get('method')})")
            if should_stop():
                raise RunCancelled("Computation cancelled by user.")

            # --- Section 4: local 12-DOF IC-GN ---
            U1, F1, info_local, bad_local = local_icgn(ctx, bundle, g_prep, U0, para, mesh)
            timings["local_icgn"] = timings.get("local_icgn", 0.0) + info_local.solve_time
            U_local, F_local = U1.copy(), F1.copy()
            progress(base + 0.5 * span, f"Frame {k}: local IC-GN ({info_local.solve_time:.1f}s)")

            admm_info = None
            if para.use_global_step and ops is not None:
                # --- Section 5: global step + beta ---
                t5 = time.perf_counter()
                exclude = np.zeros(mesh.n_nodes, dtype=bool)
                exclude[mesh.boundary_nodes] = True
                exclude |= bad_local
                beta_sweep = None
                if para.beta is not None:
                    beta = float(para.beta)
                elif ref_idx in beta_by_ref:
                    beta = beta_by_ref[ref_idx]
                else:
                    beta, beta_sweep = auto_tune_beta(ops, para, para.mu, U1, F1, exclude)
                    beta_by_ref[ref_idx] = beta
                mu = float(para.mu)
                system = build_global_system(ops, beta, mu, para.alpha, para)
                W = np.zeros((mesh.n_nodes, 3, 3))
                v = np.zeros((mesh.n_nodes, 3))
                U2, s2info = solve_subpb2(system, U1, F1, W, v)
                F2 = nodal_gradient(ops, U2)
                F2 = np.where(np.isfinite(F2), F2, F1)
                W = F2 - F1
                v = U2 - U1
                t_sub2 = time.perf_counter() - t5
                upd_g: list[float] = []
                upd_l: list[float] = []
                res_u: list[float] = [_rms(U2 - U1)]
                res_f: list[float] = [_rms(F2 - F1)]
                local_infos = [info_local]
                progress(base + 0.6 * span, f"Frame {k}: global step (beta={beta:.3e})")

                # --- Section 6: ADMM iterations ---
                n_steps = 1
                for step in range(2, para.admm_max_iter + 1):
                    if should_stop():
                        raise RunCancelled("Computation cancelled by user.")
                    n_steps = step
                    U1_prev = U1
                    U1, info_s1, _ = subpb1_solver(ctx, bundle, g_prep, U2, F2, v, mu, para, mesh)
                    F1 = F2
                    local_infos.append(info_s1)
                    timings["subpb1"] = timings.get("subpb1", 0.0) + info_s1.solve_time
                    t6 = time.perf_counter()
                    U2_new, _ = solve_subpb2(system, U1, F1, W, v)
                    F2_new = nodal_gradient(ops, U2_new)
                    F2_new = np.where(np.isfinite(F2_new), F2_new, F1)
                    t_sub2 += time.perf_counter() - t6
                    du_g = _rms(U2_new - U2)
                    du_l = _rms(U1 - U1_prev)
                    upd_g.append(du_g)
                    upd_l.append(du_l)
                    if para.dual_update == "accumulate":
                        W = W + (F2_new - F1)
                        v = v + (U2_new - U1)
                    else:
                        W = F2_new - F1
                        v = U2_new - U1
                    U2, F2 = U2_new, F2_new
                    res_u.append(_rms(U2 - U1))
                    res_f.append(_rms(F2 - F1))
                    logger.info("ADMM step %d: |dU_global|=%.3e |dU_local|=%.3e (tol %.1e)", step, du_g, du_l, para.admm_tol)
                    progress(
                        base + (0.6 + 0.3 * (step - 1) / max(1, para.admm_max_iter - 1)) * span, f"Frame {k}: ADMM step {step}"
                    )
                    if du_g < para.admm_tol or du_l < para.admm_tol:
                        break
                timings["subpb2"] = timings.get("subpb2", 0.0) + t_sub2
                admm_info = ADMMInfo(
                    beta=beta,
                    mu=mu,
                    n_steps=n_steps,
                    update_global=tuple(upd_g),
                    update_local=tuple(upd_l),
                    primal_residual_u=tuple(res_u),
                    primal_residual_f=tuple(res_f),
                    local_info=tuple(local_infos),
                    subpb2_time=t_sub2,
                    beta_sweep=beta_sweep,
                )
                U_final, F_final = U2, F2
                zncc = local_infos[-1].zncc
                status = local_infos[-1].status
            else:
                U_final, F_final = U1, F1
                zncc = info_local.zncc
                status = info_local.status

            U_std = displacement_uncertainty(ctx, zncc, status)
            results[k - 1] = FrameResult(
                U=U_final,
                F=F_final,
                ref_frame=ref_idx,
                U_local=U_local if para.store_local_result else None,
                F_local=F_local if para.store_local_result else None,
                U0=U0 if para.store_local_result else None,
                zncc=zncc,
                U_std=U_std,
                status=status,
                admm=admm_info,
            )
            mesh_by_frame[k - 1] = mesh
            prev_ref, prev_U = ref_idx, U_final
            timings[f"frame_{k}"] = time.perf_counter() - t_frame
            logger.info("Frame %d/%d done in %.1fs", k, n_pairs, timings[f"frame_{k}"])
    except RunCancelled as exc:
        stopped_early = True
        stopped_at = next((i + 1 for i, r in enumerate(results) if r is None), None)
        stop_reason = str(exc)
        logger.warning("%s Keeping %d completed frame(s).", stop_reason, sum(r is not None for r in results))

    done = [r for r in results if r is not None]
    done_meshes = [m for m in mesh_by_frame if m is not None]
    if not done:
        raise RuntimeError(stop_reason or "No frame was solved.")

    # ------------------------------------------------------------------
    # Section 7: cumulative displacements
    # ------------------------------------------------------------------
    progress(0.9, "Section 7: composing cumulative displacements")
    t7 = time.perf_counter()
    done = _compose_cumulative(done, done_meshes, schedule, base_mesh, para)
    timings["cumulative"] = time.perf_counter() - t7

    # ------------------------------------------------------------------
    # Section 8: strain
    # ------------------------------------------------------------------
    strains: list[StrainResult] = []
    if compute_strain:
        t8 = time.perf_counter()
        mesh0 = done_meshes[0]
        ops0 = None
        if para.strain_method == "fem":
            entry = ref_cache.get(0)
            ops0 = entry["ops"] if entry is not None else None
        for i, fr in enumerate(done):
            progress(0.92 + 0.07 * (i + 1) / len(done), f"Section 8: strain frame {i + 1}/{len(done)}")
            U_acc = fr.U_accum if fr.U_accum is not None else fr.U
            F_direct = fr.F if fr.ref_frame == 0 else None
            strains.append(_compute_strain(mesh0, para, U_acc, F_direct=F_direct, valid=mesh0.node_valid, ops=ops0))
        timings["strain"] = time.perf_counter() - t8

    timings["total"] = time.perf_counter() - t_start
    progress(1.0, "Pipeline complete")
    return PipelineResult(
        dvc_para=para,
        dvc_mesh=done_meshes[0],
        result_disp=done,
        result_strain=strains,
        frame_schedule=schedule,
        volume_shape=shape,  # type: ignore[arg-type]
        timings=timings,
        stopped_early=stopped_early,
        stopped_at_frame=stopped_at,
        stop_reason=stop_reason,
    )


def _compose_cumulative(
    results: list[FrameResult],
    meshes: list[DVCMesh],
    schedule: FrameSchedule,
    base_mesh: DVCMesh,
    para: DVCPara,
) -> list[FrameResult]:
    """Fill ``U_accum`` by walking each frame's reference chain to frame 0.

    Every pair is solved on the same node grid placed in its own reference
    volume, so the material points that started at the frame-0 nodes are
    tracked by interpolating each pair's displacement grid at their current
    positions (MATLAB Section 7, ``interp3(...,'makima')`` -> cubic spline).
    """
    coords0 = base_mesh.coordinates
    axes = (base_mesh.x0, base_mesh.y0, base_mesh.z0)
    order = 3 if para.cumulative_interp == "cubic" else 1
    positions: dict[int, NDArray[np.float64]] = {0: coords0.copy()}
    out: list[FrameResult] = []
    for i, fr in enumerate(results):
        frame = i + 1
        parent = schedule.parent(frame)
        if parent == 0:
            U_acc = fr.U.copy()
            positions[frame] = coords0 + U_acc
        else:
            pos_parent = positions.get(parent)
            if pos_parent is None:
                # parent was not solved (cancelled run); fall back to raw U
                U_acc = fr.U.copy()
                positions[frame] = coords0 + U_acc
            else:
                grid = meshes[i].grid_shape
                disp = np.empty_like(pos_parent)
                for c in range(3):
                    disp[:, c] = interp_grid_field(fr.U[:, c].reshape(grid), axes, pos_parent, order=order)
                pos = pos_parent + disp
                positions[frame] = pos
                U_acc = pos - coords0
        out.append(replace(fr, U_accum=U_acc))
    return out
