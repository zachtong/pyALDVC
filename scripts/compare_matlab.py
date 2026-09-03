#!/usr/bin/env python
"""Cross-validate pyALDVC against the MATLAB ALDVC results shipped with the reference code.

The MATLAB repository contains ``results_ws32_st8.mat`` (data set
``DVC_images/20190504_cut_01/02.mat``) and ``results_ws30_st30.mat``
(``eyes_0/1.mat``). This script runs pyALDVC with the same subset size, node
spacing, interpolation, global-step discretisation, ADMM tolerances and penalty
parameters on the same node positions, then compares node by node the initial
guess, the local IC-GN solution, the final AL-DVC displacement and the
displacement gradient.

Two comparisons do not depend on either code's convergence tolerance or
outlier handling and are therefore the decisive ones:

* **solver equivalence** -- both codes' local solutions are refined by the
  pyALDVC IC-GN kernel to a tight increment tolerance; if the two
  implementations minimise the same functional they end at the same optimum;
* **objective value** -- the ZNCC of every stored solution is evaluated with
  the pyALDVC kernel, so "which code found the better optimum" is measurable
  without ground truth.

MATLAB nodes whose subsets touch the volume border (closer than
``winsize/2 + GRADIENT_BORDER + INTERP_MARGIN`` voxels) cannot be placed by
pyALDVC and are excluded; the remaining node grids coincide exactly. The
volumes are cropped to the VOI plus a margin covering the MATLAB displacement
range, which changes nothing in the correlation but keeps runs fast.

Usage::

    python scripts/compare_matlab.py                       # results_ws32_st8.mat
    python scripts/compare_matlab.py --results results_ws30_st30.mat
    python scripts/compare_matlab.py --quick               # central sub-region only

Writes ``reports/matlab_crossval_<tag>.pdf`` and a metrics CSV next to it.
Reports are generated, never hand-edited.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__  # noqa: E402
from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.data_structures import STATUS_NAMES, P_from_UF, VOIRange  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.io.matlab_results import (  # noqa: E402
    MatlabResults,
    load_matlab_results,
    match_nodes,
)
from al_dvc.io.volume_io import load_volume  # noqa: E402
from al_dvc.io.volume_ops import (  # noqa: E402
    GRADIENT_BORDER,
    build_reference_bundle,
    normalize_volume,
    prepare_deformed,
    presmooth_volume,
)
from al_dvc.mesh.grid_mesh import INTERP_MARGIN  # noqa: E402
from al_dvc.solver.interp_kernels import (  # noqa: E402
    INTERP_BSPLINE,
    INTERP_CUBIC,
    INTERP_LINEAR,
)
from al_dvc.solver.local_icgn import precompute_local_context  # noqa: E402
from al_dvc.solver.numba_kernels import (  # noqa: E402
    evaluate_zncc_parallel,
    icgn_12dof_parallel,
)

DEFAULT_ALDVC_DIR = ROOT.parent / "ALDVC" / "ALDVC"
QUICK_NODES_XY = 12
QUICK_NODES_Z = 6
DISP_OUTLIER_VOXEL = 0.1
GRAD_OUTLIER = 1e-2
CROP_EXTRA_MARGIN = 16  # voxels beyond the MATLAB displacement range
MAX_CROP_PAD = 96  # cap for diverged MATLAB results (e.g. the eyes data set)
REFINE_DP_TOL = 1e-4  # increment tolerance of the solver-equivalence refinement
REFINE_MAX_ITER = 200
REFINE_MAX_NODES = 20_000  # refine at most this many (randomly sampled) nodes
MIN_CONVERGED_NODES = 10  # below this the 'converged' masks fall back to all valid nodes
INTERP_MODES = {"cubic": INTERP_CUBIC, "linear": INTERP_LINEAR, "bspline": INTERP_BSPLINE}


# --------------------------------------------------------------------------- grid alignment / cropping
def aligned_voi(
    res: MatlabResults, volume_shape: tuple[int, int, int], nodes_keep: tuple[int, int, int] | None
) -> tuple[VOIRange, dict]:
    """VOI for which pyALDVC places its nodes exactly on the MATLAB nodes.

    Returns the VOI and a dict with the number of MATLAB node layers dropped
    per axis (those closer to the border than the pyALDVC margin rule allows;
    with ``nodes_keep`` also the layers outside the central sub-region of that
    many node layers per axis).
    """
    nz, ny, nx = volume_shape
    lengths = (nx, ny, nz)
    border = GRADIENT_BORDER + INTERP_MARGIN
    ranges = []
    dropped = {}
    for ax, name in enumerate("xyz"):
        nodes = np.unique(res.coordinates[:, ax])
        half = res.winsize[ax] // 2
        h = res.winstepsize[ax]
        steps = np.diff(nodes)
        if nodes.size > 1 and not np.allclose(steps, h):
            raise ValueError(f"MATLAB nodes along {name} are not uniformly spaced by {h}: {np.unique(steps)}")
        ok = nodes[(nodes >= half + border) & (nodes <= lengths[ax] - 1 - half - border)]
        if nodes_keep is not None:
            n_keep = int(nodes_keep[ax])
            mid = ok.size // 2
            ok = ok[max(0, mid - n_keep // 2) : mid - n_keep // 2 + n_keep]
        if ok.size < 2:
            raise ValueError(f"fewer than 2 MATLAB nodes along {name} satisfy the pyALDVC margin rule")
        dropped[name] = int(nodes.size - ok.size)
        ranges.append((int(ok[0] - half), int(ok[-1] + half)))
    return VOIRange(x=ranges[0], y=ranges[1], z=ranges[2]), dropped


def crop_to_voi(f, g, voi: VOIRange, res: MatlabResults, pad: int):
    """Crop both volumes to ``voi`` plus ``pad`` voxels; shift the VOI and the MATLAB coordinates accordingly."""
    nz, ny, nx = f.shape
    lo = np.array([max(voi.x[0] - pad, 0), max(voi.y[0] - pad, 0), max(voi.z[0] - pad, 0)], dtype=np.int64)
    hi = np.array([min(voi.x[1] + pad, nx - 1), min(voi.y[1] + pad, ny - 1), min(voi.z[1] + pad, nz - 1)], dtype=np.int64)
    sl = (slice(lo[2], hi[2] + 1), slice(lo[1], hi[1] + 1), slice(lo[0], hi[0] + 1))
    fc = np.ascontiguousarray(f[sl])
    gc = np.ascontiguousarray(g[sl])
    voi_c = VOIRange(
        x=(voi.x[0] - lo[0], voi.x[1] - lo[0]), y=(voi.y[0] - lo[1], voi.y[1] - lo[1]), z=(voi.z[0] - lo[2], voi.z[1] - lo[2])
    )
    res_c = dataclasses.replace(res, coordinates=res.coordinates - lo[None, :])
    return fc, gc, voi_c, res_c, lo


# --------------------------------------------------------------------------- metrics
def _grid_index(mesh):
    n = np.arange(mesh.n_nodes)
    gz, gy, gx = mesh.grid_shape
    return n // (gy * gx), (n // gx) % gy, n % gx


def interior_mask(mesh, layers: int = 1) -> np.ndarray:
    iz, iy, ix = _grid_index(mesh)
    gz, gy, gx = mesh.grid_shape
    return (ix >= layers) & (ix < gx - layers) & (iy >= layers) & (iy < gy - layers) & (iz >= layers) & (iz < gz - layers)


def diff_stats(ours: np.ndarray, theirs: np.ndarray, mask: np.ndarray, outlier: float) -> dict:
    """Per-component statistics of ``ours - theirs`` on ``mask`` (any trailing shape)."""
    d = (ours - theirs).reshape(ours.shape[0], -1)[mask]
    d = d[np.all(np.isfinite(d), axis=1)]
    out = {"n": int(d.shape[0])}
    if d.shape[0] == 0:
        return out
    ad = np.abs(d)
    out["median_abs"] = np.median(ad, axis=0)
    out["rms"] = np.sqrt(np.mean(d**2, axis=0))
    out["p95"] = np.percentile(ad, 95, axis=0)
    out["p99"] = np.percentile(ad, 99, axis=0)
    out["max"] = ad.max(axis=0)
    out["bias"] = d.mean(axis=0)
    out["frac_outlier"] = float(np.mean(np.any(ad > outlier, axis=1)))
    return out


def _fmt_vec(v, fmt="{:.4f}") -> str:
    return "[" + ", ".join(fmt.format(float(x)) for x in np.atleast_1d(v)) + "]"


# --------------------------------------------------------------------------- pyALDVC run
def make_para(res: MatlabResults, voi: VOIRange, beta, n_threads: int, verbose: bool, init_coarse: int = 1):
    fr0 = res.frames[0]
    para_m = res.para
    subpb2 = "fd" if str(para_m.get("Subpb2FDOrFEM", "finiteDifference")).lower().startswith("finited") else "fem"
    interp = "cubic" if str(para_m.get("interpMethod", "cubic")).lower().startswith("cubic") else "linear"
    return dvcpara_default(
        voi=voi,
        winsize=res.winsize,
        winstepsize=res.winstepsize,
        interp_method=interp,
        subpb2_method=subpb2,
        icgn_tol=float(para_m.get("ICGNtol", 1e-2)),
        icgn_max_iter=int(para_m.get("Subpb1ICGNMaxIterNum", 100)),
        admm_tol=float(para_m.get("ADMMtol", 1e-2)),
        admm_max_iter=4,
        mu=float(fr0.mu) if fr0.mu and np.isfinite(fr0.mu) else 1e-3,
        beta=beta,
        dual_update="reset",
        init_outlier_threshold=float(para_m.get("medianFilterThreshold", 2.0)),
        store_local_result=True,
        n_threads=n_threads,
        verbose=verbose,
        init_coarse_factor=int(init_coarse),
    )


def compare(result, res: MatlabResults) -> dict:
    """All node-wise comparisons between a pyALDVC result and the MATLAB frame 1."""
    mesh = result.dvc_mesh
    fr = result.result_disp[0]
    m = res.frames[0]
    idx = match_nodes(mesh.coordinates, res.coordinates)
    if np.any(idx < 0):
        raise RuntimeError(f"{int(np.sum(idx < 0))} pyALDVC nodes have no MATLAB counterpart; grids are not aligned")
    interior = interior_mask(mesh)
    local_info = fr.admm.local_info if fr.admm is not None else []
    ours_conv = (local_info[0].status == 0) if local_info else np.ones(mesh.n_nodes, bool)
    max_iter = int(res.para.get("Subpb1ICGNMaxIterNum", 100))
    if m.conv_iter is not None and m.conv_iter.shape[1]:
        theirs_conv = m.conv_iter[idx, 0] < max_iter
    else:
        theirs_conv = np.ones(mesh.n_nodes, bool)
    both = ours_conv & theirs_conv & mesh.node_valid
    fallback = int(both.sum()) < MIN_CONVERGED_NODES
    if fallback:  # e.g. a diverged MATLAB run: keep the plots meaningful on all valid nodes
        both = mesh.node_valid.copy()
    masks = {"all": mesh.node_valid.copy(), "converged": both, "converged interior": both & interior}

    cmp = {"idx": idx, "masks": masks, "rows": [], "interior": interior, "converged_fallback": fallback}
    pairs = [
        ("U0 (initial guess)", fr.U0, m.U0, DISP_OUTLIER_VOXEL * 10),
        ("U_local (12-DOF IC-GN)", fr.U_local, m.U_local, DISP_OUTLIER_VOXEL),
        ("U (final AL-DVC)", fr.U, m.U, DISP_OUTLIER_VOXEL),
        ("F_local (12-DOF IC-GN)", fr.F_local, m.F_local, GRAD_OUTLIER),
        ("F (final AL-DVC)", fr.F, m.F, GRAD_OUTLIER),
    ]
    for name, ours, theirs, outlier in pairs:
        if ours is None or theirs is None:
            continue
        for mname, mask in masks.items():
            st = diff_stats(np.asarray(ours, float), np.asarray(theirs, float)[idx], mask, outlier)
            cmp["rows"].append({"quantity": name, "nodes": mname, **st})
    cmp["fields"] = {
        name: (np.asarray(ours, float), None if theirs is None else np.asarray(theirs, float)[idx])
        for name, ours, theirs, _ in pairs
        if ours is not None
    }
    cmp["ours_conv"] = ours_conv
    cmp["theirs_conv"] = theirs_conv
    return cmp


def solver_equivalence(result, res: MatlabResults, fc, gc, para, cmp: dict, max_refine: int = REFINE_MAX_NODES) -> dict:
    """Refine both local solutions with the pyALDVC kernel; evaluate the ZNCC of every stored solution.

    Also returns the texture anisotropy of the reference VOI and the
    Hessian-based displacement uncertainty proxy ``sqrt(diag(H^-1))``.
    """
    mesh = result.dvc_mesh
    fr = result.result_disp[0]
    m = res.frames[0]
    idx = cmp["idx"]
    t0 = time.perf_counter()
    fn = presmooth_volume(normalize_volume(fc, para.voi), para.prefilter_sigma)
    gn = presmooth_volume(normalize_volume(gc, para.voi), para.prefilter_sigma)
    bundle = build_reference_bundle(fn, None)
    ctx = precompute_local_context(mesh, bundle, para)
    gp = prepare_deformed(gn, para.interp_method)
    mode = INTERP_MODES[para.interp_method]
    hx, hy, hz = ctx.half

    def zncc_of(U, F):
        P = P_from_UF(np.asarray(U, float), np.asarray(F, float))
        return evaluate_zncc_parallel(
            ctx.coords_int, P, hx, hy, hz, bundle.f, bundle.mask, gp, mode, ctx.meanf, ctx.bottomf, ctx.valid
        )

    valid_ref = ctx.valid.copy()
    n_valid = int(valid_ref.sum())
    if max_refine and n_valid > max_refine:
        keep = np.zeros(n_valid, dtype=bool)
        keep[np.random.default_rng(0).choice(n_valid, int(max_refine), replace=False)] = True
        valid_ref[np.flatnonzero(valid_ref)] = keep

    def refine(U, F):
        P0 = P_from_UF(np.asarray(U, float), np.asarray(F, float))
        return icgn_12dof_parallel(
            ctx.coords_int,
            P0,
            hx,
            hy,
            hz,
            bundle.f,
            bundle.gx,
            bundle.gy,
            bundle.gz,
            bundle.mask,
            gp,
            mode,
            ctx.L_all,
            ctx.meanf,
            ctx.bottomf,
            valid_ref,
            float(para.icgn_tol),
            REFINE_DP_TOL,
            REFINE_MAX_ITER,
            int(para.icgn_patience),
        )

    out = {"zncc": {}}
    if fr.U_local is not None and m.U_local is not None and m.F_local is not None:
        Pm, itm, stm, zm = refine(m.U_local[idx], m.F_local[idx])
        Po, ito, sto, zo = refine(fr.U_local, fr.F_local)
        ok = (stm == 0) & (sto == 0) & mesh.node_valid
        out.update(
            ok=ok,
            Um=Pm[:, 9:12],
            Uo=Po[:, 9:12],
            Fm=Pm[:, :9].reshape(-1, 3, 3),
            Fo=Po[:, :9].reshape(-1, 3, 3),
            iters_matlab=itm,
            iters_ours=ito,
        )
        out["zncc"]["U_local pyALDVC"] = zncc_of(fr.U_local, fr.F_local)
        out["zncc"]["U_local MATLAB"] = zncc_of(m.U_local[idx], m.F_local[idx])
        out["zncc"]["U_local refined from pyALDVC"] = zo
        out["zncc"]["U_local refined from MATLAB"] = zm
        masks = {"both refined": ok, "both refined, interior": ok & cmp["interior"]}
        for mname, mask in masks.items():
            st = diff_stats(out["Uo"], out["Um"], mask, DISP_OUTLIER_VOXEL)
            cmp["rows"].append({"quantity": "U_local refined (both)", "nodes": mname, **st})
            st = diff_stats(out["Fo"], out["Fm"], mask, GRAD_OUTLIER)
            cmp["rows"].append({"quantity": "F_local refined (both)", "nodes": mname, **st})
    if m.F is not None:
        out["zncc"]["U final pyALDVC"] = zncc_of(fr.U, fr.F)
        out["zncc"]["U final MATLAB"] = zncc_of(m.U[idx], m.F[idx])

    s = para.voi.clamp(fn.shape).slices
    out["texture_energy"] = [float(np.mean(a[s].astype(np.float64) ** 2)) for a in (bundle.gx, bundle.gy, bundle.gz)]
    hinv = np.full((mesh.n_nodes, 3), np.nan)
    valid = np.flatnonzero(ctx.valid)
    if valid.size:
        try:
            inv = np.linalg.inv(ctx.H_all[valid])
            hinv[valid] = np.sqrt(np.maximum(np.diagonal(inv, axis1=1, axis2=2)[:, 9:12], 0.0))
        except np.linalg.LinAlgError:
            pass
    out["hinv_diag"] = hinv
    out["time"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------- report
def _text_page(pdf, title, lines, width=105):
    import textwrap

    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width, subsequent_indent="    ") if len(line) > width else [line])
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.96, title, fontsize=15, weight="bold", va="top")
    y = 0.925
    for line in wrapped:
        fig.text(0.05, y, line, fontsize=7.6, family="monospace", va="top")
        y -= 0.0148
        if y < 0.04:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            y = 0.96
    pdf.savefig(fig)
    plt.close(fig)


def _hist_diff(ax, d, title, unit, clip):
    d = d[np.isfinite(d)]
    if d.size == 0:
        ax.axis("off")
        return
    inside = np.abs(d) <= clip
    ax.hist(np.clip(d, -clip, clip), bins=80, color="#4c72b0")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(
        f"{title}\nmedian |d|={np.median(np.abs(d)):.4f}, rms={np.sqrt(np.mean(d**2)):.4f}, "
        f"{100 * (1 - inside.mean()):.2f}% beyond +-{clip:g}",
        fontsize=8,
    )
    ax.set_xlabel(f"pyALDVC - MATLAB [{unit}]", fontsize=8)
    ax.tick_params(labelsize=7)


def _slice_rows(fig, axes, mesh, ours, theirs, mask, comp_names, unit, iz, offset):
    x0, y0 = mesh.x0 + offset[0], mesh.y0 + offset[1]
    extent = [x0[0], x0[-1], y0[0], y0[-1]]
    for r, name in enumerate(comp_names):
        a = mesh.to_grid(np.where(mask, ours[:, r], np.nan))[iz]
        b = mesh.to_grid(np.where(mask, theirs[:, r], np.nan))[iz]
        d = a - b
        both = np.concatenate([a.ravel(), b.ravel()])
        finite = both[np.isfinite(both)]
        if finite.size == 0:
            for c in range(3):
                axes[r, c].axis("off")
            continue
        vmin, vmax = np.nanpercentile(finite, 1), np.nanpercentile(finite, 99)
        dd = np.abs(d[np.isfinite(d)])
        dl = max(float(np.nanpercentile(dd, 99)) if dd.size else 0.0, 1e-3)
        for c, (img, ttl, cmap, lo, hi) in enumerate(
            [
                (b, f"MATLAB {name}", "viridis", vmin, vmax),
                (a, f"pyALDVC {name}", "viridis", vmin, vmax),
                (d, f"difference {name}", "coolwarm", -dl, dl),
            ]
        ):
            ax = axes[r, c]
            im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
            ax.set_title(f"{ttl} [{unit}]", fontsize=8)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)


def _row_line(row: dict) -> str:
    if row["n"] == 0:
        return ""
    if row["quantity"].startswith("F"):
        med = f"{np.mean(row['median_abs']):.2e} (mean of 9)"
        rms = f"{np.mean(row['rms']):.2e}"
        p99 = f"{np.max(row['p99']):.2e} (max of 9)"
    else:
        med, rms, p99 = _fmt_vec(row["median_abs"]), _fmt_vec(row["rms"]), _fmt_vec(row["p99"])
    return (
        f"  {row['quantity']:<26} {row['nodes']:<24} {row['n']:>6}  {med:<28} {rms:<28} {p99:<28} "
        f"{100 * row['frac_outlier']:.2f}%"
    )


def write_report(path: Path, ctx: dict) -> None:
    res: MatlabResults = ctx["res"]
    runs = ctx["runs"]
    main_key = ctx["main_key"]
    result = runs[main_key]["result"]
    cmp = runs[main_key]["cmp"]
    eq = runs[main_key]["eq"]
    mesh = result.dvc_mesh
    fr = result.result_disp[0]
    m = res.frames[0]
    offset = ctx["offset"]
    with PdfPages(str(path)) as pdf:
        # ---- summary
        lines = [f"pyALDVC {__version__} vs MATLAB ALDVC   ({res.path.name})", ""]
        lines.append(
            f"volumes: {', '.join(res.file_names)}   shape (z,y,x): {ctx['shape']}   dtype: {ctx['dtype']}   "
            f"load {ctx['t_load']:.1f} s"
        )
        lines.append(
            f"cropped to (z,y,x) {ctx['crop_shape']} at offset (x,y,z) {tuple(int(v) for v in offset)} "
            f"(VOI + {ctx['pad']} voxels; MATLAB |U| 99.9th percentile {ctx['umax']:.2f} voxel)"
        )
        lines.append("")
        lines.append("MATLAB parameters:")
        for k in (
            "winsize",
            "winstepsize",
            "Subpb2FDOrFEM",
            "interpMethod",
            "ICGNtol",
            "ADMMtol",
            "Subpb1ICGNMaxIterNum",
            "medianFilterThreshold",
            "initFFTMethod",
            "trackingMode",
        ):
            if k in res.para:
                lines.append(f"  {k:<26} {res.para[k]}")
        lines.append(f"  gridRange (0-based)        {res.grid_range}")
        lines.append(f"  nodes                      {res.n_nodes}   beta = {m.beta:.5g}   mu = {m.mu:.3g}")
        lines.append("")
        lines.append("pyALDVC parameters (original-volume coordinates; differences from MATLAB: pyramid NCC initial guess,")
        lines.append("node-wise search-radius expansion, PCG global solver, increment tolerance icgn_dp_tol;")
        lines.append("dual_update='reset' reproduces the MATLAB dual update):")
        para = runs[main_key]["para"]
        lines.append(f"  {'voi':<26} x={ctx['voi_full'].x} y={ctx['voi_full'].y} z={ctx['voi_full'].z}")
        for k in (
            "winsize",
            "winstepsize",
            "interp_method",
            "subpb2_method",
            "icgn_tol",
            "icgn_dp_tol",
            "icgn_max_iter",
            "admm_tol",
            "admm_max_iter",
            "mu",
            "beta",
            "beta_criterion",
            "dual_update",
            "init_guess_method",
            "search_radius",
            "init_outlier_threshold",
            "local_outlier_threshold",
            "n_threads",
        ):
            lines.append(f"  {k:<26} {getattr(para, k)}")
        lines.append(f"  nodes                      {mesh.n_nodes}  grid (z,y,x) {mesh.grid_shape}")
        lines.append(
            f"  MATLAB node layers dropped (x, y, z): {ctx['dropped']}"
            + ("   [--quick: central sub-region]" if ctx["quick"] else "")
        )
        lines.append("")
        te = eq["texture_energy"]
        lines.append(
            f"Reference texture in the VOI: mean squared gradient x, y, z = {te[0]:.3f}, {te[1]:.3f}, {te[2]:.3f} "
            f"(z/x = {te[2] / max(te[0], 1e-12):.2f})"
        )
        hv = np.nanmedian(eq["hinv_diag"], axis=0)
        lines.append(
            f"Hessian uncertainty proxy, median sqrt(diag(H^-1)) for u, v, w = {hv[0]:.4f}, {hv[1]:.4f}, {hv[2]:.4f} "
            f"(w/u = {hv[2] / max(hv[0], 1e-12):.2f})"
        )
        lines.append("")
        for key, run in runs.items():
            a = run["result"].result_disp[0].admm
            lines.append(
                f"run '{key}': beta = {a.beta:.5g} (MATLAB {m.beta:.5g}), ADMM steps = {a.n_steps}, "
                f"wall time = {run['time']:.1f} s (+ {run['eq']['time']:.1f} s equivalence check)"
            )
            lines.append("  timings [s]: " + ", ".join(f"{k}={v:.1f}" for k, v in run["result"].timings.items()))
            if a.beta_sweep is not None:
                sw = a.beta_sweep
                lines.append(
                    "  L-curve sweep: betas = " + _fmt_vec(sw["betas"], "{:.3g}") + "  score = " + _fmt_vec(sw["score"], "{:.3g}")
                )
                lines.append(
                    "    err1 |u-u_hat| = "
                    + _fmt_vec(sw["err1"], "{:.4g}")
                    + "  err2 |F-grad u_hat| = "
                    + _fmt_vec(sw["err2"], "{:.4g}")
                    + f"  criterion = {sw.get('criterion', '?')}"
                )
        lines.append("")
        lines.append("Convergence (frame 1):")
        if cmp.get("converged_fallback"):
            lines.append("  NOTE: fewer than 10 nodes converged in both codes; the 'converged' masks below use all valid nodes.")
        li = fr.admm.local_info
        ours_iter = [f"{np.mean(x.n_iter[x.status == 0]):.1f}" for x in li]
        theirs_iter = (
            [f"{np.mean(m.conv_iter[:, k]):.1f}" for k in range(m.conv_iter.shape[1]) if np.any(m.conv_iter[:, k] > 0)]
            if m.conv_iter is not None
            else []
        )
        lines.append(f"  mean IC-GN iterations per ADMM pass   pyALDVC: {ours_iter}   MATLAB: {theirs_iter}")
        lines.append(
            f"  pyALDVC converged (pass 1): {int(cmp['ours_conv'].sum())}/{mesh.n_nodes}   "
            f"MATLAB converged (pass 1, < max iter): {int(cmp['theirs_conv'].sum())}/{mesh.n_nodes}"
        )
        st = np.asarray(li[0].status)
        codes, counts = np.unique(st, return_counts=True)
        lines.append(
            "  pyALDVC status (pass 1): "
            + ", ".join(f"{STATUS_NAMES.get(int(c), c)}={n}" for c, n in zip(codes, counts))
            + f", median-test inpainted = {li[0].n_bad}"
        )
        if "iters_matlab" in eq:
            ok = eq["ok"]
            lines.append(
                f"  refinement to dp_tol={REFINE_DP_TOL:g}: from MATLAB {np.mean(eq['iters_matlab'][ok]):.1f} iterations, "
                f"from pyALDVC {np.mean(eq['iters_ours'][ok]):.1f}; both converged at {int(ok.sum())} nodes"
            )
        lines.append("")
        lines.append("ZNCC of each stored solution evaluated with the pyALDVC kernel (nodes converged in both codes):")
        mask = cmp["masks"]["converged"]
        z = eq["zncc"]
        for name, val in z.items():
            lines.append(f"  {name:<32} mean {np.nanmean(val[mask]):.5f}   median {np.nanmedian(val[mask]):.5f}")
        for a_, b_ in (("U_local pyALDVC", "U_local MATLAB"), ("U final pyALDVC", "U final MATLAB")):
            if a_ in z and b_ in z:
                d = (z[a_] - z[b_])[mask]
                d = d[np.isfinite(d)]
                lines.append(
                    f"  {a_} higher than {b_} at {100 * np.mean(d > 0):.1f}% of nodes (mean difference {np.mean(d):+.5f})"
                )
        lines.append("")
        for key, run in runs.items():
            lines.append(f"Node-wise differences pyALDVC - MATLAB, run '{key}'  (voxel for U, 1 for F):")
            lines.append(
                f"  {'quantity':<26} {'nodes':<24} {'n':>6}  {'median|d| (x,y,z)':<28} {'rms (x,y,z)':<28} "
                f"{'p99 (x,y,z)':<28} outliers"
            )
            for row in run["cmp"]["rows"]:
                line = _row_line(row)
                if line:
                    lines.append(line)
            lines.append("")
        lines.append("Outliers: nodes where any component differs by more than 0.1 voxel (U), 1 voxel (U0) or 1e-2 (F).")
        lines.append("'refined (both)': both codes' local solutions re-optimised by the pyALDVC IC-GN kernel to")
        lines.append(f"dp_tol={REFINE_DP_TOL:g}; residual differences there are distinct local optima, not solver differences.")
        lines.append("Expected sources of the other differences: convergence tolerances (both codes stop early in the")
        lines.append("weakly textured direction), different outlier/inpainting rules, MATLAB's integer FFT initial guess,")
        lines.append("MATLAB clamping interpolation at the border and zeroing the dual on Neumann boundary nodes.")
        _text_page(pdf, "pyALDVC vs MATLAB ALDVC cross-validation", lines)

        # ---- difference histograms (local, final)
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for r, (name, clip) in enumerate([("U_local (12-DOF IC-GN)", 0.2), ("U (final AL-DVC)", 0.2)]):
            ours, theirs = cmp["fields"][name]
            for c, comp in enumerate("uvw"):
                _hist_diff(axes[r, c], (ours[:, c] - theirs[:, c])[mask], f"{name}: {comp}", "voxel", clip)
        fig.suptitle(f"Displacement differences on {int(mask.sum())} nodes converged in both codes")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- solver equivalence page
        if "Um" in eq:
            ok = eq["ok"]
            fig, axes = plt.subplots(2, 3, figsize=(13, 8))
            for c, comp in enumerate("uvw"):
                _hist_diff(
                    axes[0, c],
                    (eq["Uo"][:, c] - eq["Um"][:, c])[ok],
                    f"refined from pyALDVC - refined from MATLAB: {comp}",
                    "voxel",
                    0.05,
                )
            zl = eq["zncc"]
            axes[1, 0].hist(
                [zl["U_local pyALDVC"][mask], zl["U_local MATLAB"][mask]],
                bins=60,
                label=["pyALDVC", "MATLAB"],
                histtype="step",
                lw=1.5,
            )
            axes[1, 0].set_title("ZNCC of the local solutions", fontsize=9)
            axes[1, 0].legend(fontsize=8)
            if "U final pyALDVC" in zl:
                axes[1, 1].hist(
                    [zl["U final pyALDVC"][mask], zl["U final MATLAB"][mask]],
                    bins=60,
                    label=["pyALDVC", "MATLAB"],
                    histtype="step",
                    lw=1.5,
                )
                axes[1, 1].set_title("ZNCC of the final solutions", fontsize=9)
                axes[1, 1].legend(fontsize=8)
            hv = eq["hinv_diag"]
            axes[1, 2].hist([hv[mask, 0], hv[mask, 1], hv[mask, 2]], bins=60, label=["u", "v", "w"], histtype="step", lw=1.5)
            axes[1, 2].set_title("sqrt(diag(H^-1)) per node (uncertainty proxy)", fontsize=9)
            axes[1, 2].legend(fontsize=8)
            fig.suptitle(
                "Solver equivalence: both local solutions refined by the pyALDVC IC-GN kernel to "
                f"dp_tol={REFINE_DP_TOL:g} ({int(ok.sum())} nodes)"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ---- slice maps
        iz = mesh.grid_shape[0] // 2
        for name in ("U (final AL-DVC)", "U_local (12-DOF IC-GN)"):
            ours, theirs = cmp["fields"][name]
            fig, axes = plt.subplots(3, 3, figsize=(13, 12))
            _slice_rows(fig, axes, mesh, ours, theirs, cmp["masks"]["all"], list("uvw"), "voxel", iz, offset)
            fig.suptitle(f"{name}: node layer z = {mesh.z0[iz] + offset[2]:.0f} (all valid nodes)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ---- gradient differences
        if "F (final AL-DVC)" in cmp["fields"]:
            ours, theirs = cmp["fields"]["F (final AL-DVC)"]
            fig, axes = plt.subplots(3, 3, figsize=(13, 10))
            for i in range(3):
                for j in range(3):
                    _hist_diff(axes[i, j], (ours[:, i, j] - theirs[:, i, j])[mask], f"F{i + 1}{j + 1}", "1", 0.01)
            fig.suptitle("Final displacement-gradient differences (converged nodes)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ---- convergence and beta
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        data, labels = [], []
        for k, x in enumerate(li):
            data.append(x.n_iter[x.status == 0])
            labels.append(f"py {k + 1}")
        if m.conv_iter is not None:
            for k in range(m.conv_iter.shape[1]):
                col = m.conv_iter[cmp["idx"], k]
                if np.any(col > 0):
                    data.append(col)
                    labels.append(f"MATLAB {k + 1}")
        axes[0].boxplot([d if d.size else [0] for d in data], showfliers=False)
        axes[0].set_xticks(range(1, len(labels) + 1))
        axes[0].set_xticklabels(labels, rotation=45, fontsize=7)
        axes[0].set_title("IC-GN iterations per ADMM pass", fontsize=9)
        axes[1].bar([STATUS_NAMES.get(int(c), str(c)) for c in codes], counts, color="#dd8452")
        axes[1].set_title("pyALDVC node status (pass 1)", fontsize=9)
        axes[1].tick_params(axis="x", rotation=30, labelsize=7)
        a = fr.admm
        if a.beta_sweep is not None:
            sw = a.beta_sweep
            axes[2].semilogx(sw["betas"], sw["score"], "o-", label="pyALDVC L-curve score")
            axes[2].axvline(a.beta, color="k", ls="--", lw=1, label=f"pyALDVC beta {a.beta:.3g}")
        axes[2].axvline(m.beta, color="r", ls=":", lw=1.5, label=f"MATLAB beta {m.beta:.3g}")
        axes[2].set_xlabel("beta")
        axes[2].legend(fontsize=7)
        axes[2].set_title("beta selection", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- scatter
        ours, theirs = cmp["fields"]["U (final AL-DVC)"]
        sel = np.flatnonzero(mask)
        if sel.size > 6000:
            sel = np.random.default_rng(0).choice(sel, 6000, replace=False)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
        for c, comp in enumerate("uvw"):
            x, y = theirs[sel, c], ours[sel, c]
            ok_ = np.isfinite(x) & np.isfinite(y)
            axes[c].plot(x[ok_], y[ok_], ".", ms=2, alpha=0.4)
            if ok_.sum() < 2:
                axes[c].axis("off")
                continue
            lo, hi = np.nanmin(x[ok_]), np.nanmax(x[ok_])
            axes[c].plot([lo, hi], [lo, hi], "k-", lw=0.8)
            r = np.corrcoef(x[ok_], y[ok_])[0, 1] if ok_.sum() > 2 else np.nan
            axes[c].set_title(f"final {comp}: r = {r:.5f}", fontsize=9)
            axes[c].set_xlabel("MATLAB [voxel]", fontsize=8)
            axes[c].set_ylabel("pyALDVC [voxel]", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- other runs (MATLAB beta)
        for key, run in runs.items():
            if key == main_key:
                continue
            c2 = run["cmp"]
            ours, theirs = c2["fields"]["U (final AL-DVC)"]
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
            for c, comp in enumerate("uvw"):
                _hist_diff(
                    axes[c], (ours[:, c] - theirs[:, c])[c2["masks"]["converged"]], f"U final, run '{key}': {comp}", "voxel", 0.2
                )
            fig.suptitle(f"Run '{key}' (beta = {run['result'].result_disp[0].admm.beta:.4g})")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def write_csv(path: Path, runs: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("run,quantity,nodes,n,median_abs,rms,p99,max,bias,frac_outlier\n")
        for key, run in runs.items():
            for row in run["cmp"]["rows"]:
                if row["n"] == 0:
                    continue
                fh.write(
                    ",".join(
                        [
                            key,
                            row["quantity"],
                            row["nodes"],
                            str(row["n"]),
                            _fmt_vec(row["median_abs"], "{:.6g}").replace(",", ";"),
                            _fmt_vec(row["rms"], "{:.6g}").replace(",", ";"),
                            _fmt_vec(row["p99"], "{:.6g}").replace(",", ";"),
                            _fmt_vec(row["max"], "{:.6g}").replace(",", ";"),
                            _fmt_vec(row["bias"], "{:.6g}").replace(",", ";"),
                            f"{row['frac_outlier']:.6g}",
                        ]
                    )
                    + "\n"
                )


# --------------------------------------------------------------------------- main
def _progress(fraction: float, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {100 * fraction:5.1f}%  {message}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aldvc-dir", default=str(DEFAULT_ALDVC_DIR), help="MATLAB ALDVC code directory")
    ap.add_argument("--results", default="results_ws32_st8.mat", help="result file inside --aldvc-dir")
    ap.add_argument("--out", default=None, help="output PDF (default reports/matlab_crossval_<tag>.pdf)")
    ap.add_argument(
        "--quick",
        action="store_true",
        help=f"central sub-region of {QUICK_NODES_XY}x{QUICK_NODES_XY}x{QUICK_NODES_Z} nodes (fast check)",
    )
    ap.add_argument(
        "--nodes",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="central sub-region with this many node layers per axis (scale-up steps between --quick and full)",
    )
    ap.add_argument(
        "--beta",
        choices=["auto", "matlab", "both"],
        default="both",
        help="use the L-curve beta, the MATLAB beta, or both (second run only if they differ)",
    )
    ap.add_argument("--no-crop", action="store_true", help="keep the full volumes instead of cropping to the VOI")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--init-coarse", type=int, default=1, help="init_coarse_factor: NCC + IC-GN on every k-th node, interpolated")
    ap.add_argument(
        "--refine-max", type=int, default=REFINE_MAX_NODES, help="nodes refined in the solver-equivalence check (random sample)"
    )
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    aldvc = Path(args.aldvc_dir)
    res = load_matlab_results(aldvc / args.results)
    if len(res.file_names) < 2:
        raise SystemExit("result file does not name two volumes")
    paths = [aldvc / "DVC_images" / n for n in res.file_names[:2]]
    print(f"MATLAB result: {res.path.name}  nodes={res.n_nodes}  winsize={res.winsize}  step={res.winstepsize}")
    t0 = time.perf_counter()
    f = load_volume(paths[0])
    g = load_volume(paths[1])
    t_load = time.perf_counter() - t0
    shape, dtype = f.shape, str(f.dtype)
    print(f"loaded {paths[0].name}, {paths[1].name}: shape {shape} {dtype} in {t_load:.1f} s")

    m = res.frames[0]
    nodes_keep = tuple(args.nodes) if args.nodes else ((QUICK_NODES_XY, QUICK_NODES_XY, QUICK_NODES_Z) if args.quick else None)
    voi_full, dropped = aligned_voi(res, f.shape, nodes_keep)
    umax = float(np.nanpercentile(np.abs(m.U), 99.9))  # robust to isolated MATLAB outliers
    pad = 0 if args.no_crop else min(int(np.ceil(umax)) + CROP_EXTRA_MARGIN, MAX_CROP_PAD)
    if args.no_crop:
        fc, gc, voi, res_c, offset = f, g, voi_full, res, np.zeros(3, dtype=np.int64)
    else:
        fc, gc, voi, res_c, offset = crop_to_voi(f, g, voi_full, res, pad)
        del f, g
    print(f"aligned VOI: {voi_full}   dropped MATLAB layers {dropped}   crop {fc.shape} at offset {tuple(offset)}")

    runs = {}

    def do_run(key, beta):
        para = make_para(res_c, voi, beta, args.threads, not args.quiet, args.init_coarse)
        t1 = time.perf_counter()
        result = run_aldvc(para, [fc, gc], compute_strain=False, progress_fn=_progress)
        dt = time.perf_counter() - t1
        cmp = compare(result, res_c)
        eq = solver_equivalence(result, res_c, fc, gc, para, cmp, max_refine=args.refine_max)
        runs[key] = {"para": para, "result": result, "time": dt, "cmp": cmp, "eq": eq}
        print(
            f"run '{key}': beta={result.result_disp[0].admm.beta:.4g} (MATLAB {m.beta:.4g}) in {dt:.1f} s "
            f"(+ {eq['time']:.1f} s equivalence check)"
        )

    if args.beta in ("auto", "both"):
        do_run("auto beta", None)
    need_matlab = args.beta == "matlab" or (
        args.beta == "both" and abs(np.log(runs["auto beta"]["result"].result_disp[0].admm.beta / m.beta)) > 1e-3
    )
    if need_matlab and m.beta and np.isfinite(m.beta):
        do_run("matlab beta", float(m.beta))
    main_key = "matlab beta" if ("matlab beta" in runs and args.beta == "matlab") else next(iter(runs))

    for key, run in runs.items():
        print(f"--- run '{key}'")
        for row in run["cmp"]["rows"]:
            line = _row_line(row)
            if line:
                print(line)
        for name, val in run["eq"]["zncc"].items():
            mask = run["cmp"]["masks"]["converged"]
            print(f"  ZNCC {name:<32} mean {np.nanmean(val[mask]):.5f}")

    tag = (
        res.path.stem.replace("results_", "")
        + ("_quick" if args.quick else "")
        + ("_{}x{}x{}".format(*args.nodes) if args.nodes else "")
    )
    out = Path(args.out) if args.out else ROOT / "reports" / f"matlab_crossval_{tag}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx = {
        "res": res_c,
        "runs": runs,
        "main_key": main_key,
        "shape": shape,
        "dtype": dtype,
        "t_load": t_load,
        "dropped": dropped,
        "quick": args.quick,
        "offset": offset,
        "crop_shape": fc.shape,
        "pad": pad,
        "umax": umax,
        "voi_full": voi_full,
    }
    write_report(out, ctx)
    write_csv(out.with_suffix(".csv"), runs)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
