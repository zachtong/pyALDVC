"""Synthetic ground-truth validation report for pyALDVC.

Generates ``reports/validation_synthetic.pdf`` with:
    1. Case table: translation, affine, rotation, sinusoid, cylinder mask,
       3-frame incremental vs accumulative -- RMSE of u/v/w and gradient.
    2. Noise study: displacement RMSE vs noise level, with/without pre-smoothing.
    3. Spatial-resolution study: sinusoid RMSE vs wavelength for two subsets.
    4. Local-only vs AL-DVC gradient error; dual update accumulate vs reset.
    5. Interpolation scheme comparison (linear / cubic / bspline).
    6. Field slices and error maps of the affine and sinusoid cases.

Run:  python scripts/make_validation_report.py  [--quick]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from al_dvc import __version__, dvcpara_default, run_aldvc
from al_dvc.synthetic import (
    add_noise,
    affine_displacement,
    evaluate_at_nodes,
    generate_speckle_volume,
    gradient_at_nodes,
    rotation_displacement,
    sinusoidal_displacement,
    warp_volume_lagrangian,
)
from al_dvc.viz.slices import histogram_panel, plot_field_slices

logging.basicConfig(level=logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "validation_synthetic.pdf"


def interior(mesh):
    return ~np.isin(np.arange(mesh.n_nodes), mesh.boundary_nodes)


def rmse(a, b, sel):
    e = (np.asarray(a) - np.asarray(b))[sel]
    return np.sqrt(np.mean(e**2, axis=0))


def run_case(name, f, g, disp, para, masks=None, frames=None):
    t0 = time.perf_counter()
    vols = [f, g] if frames is None else frames
    res = run_aldvc(para, vols, masks=masks)
    dt = time.perf_counter() - t0
    mesh = res.dvc_mesh
    fr = res.result_disp[-1]
    U_gt = evaluate_at_nodes(disp, mesh.coordinates)
    F_gt = gradient_at_nodes(disp, mesh.coordinates)
    sel = interior(mesh) & mesh.node_valid
    U = fr.U_accum if fr.U_accum is not None else fr.U
    row = {
        "case": name, "nodes": mesh.n_nodes, "time_s": dt,
        "rmse_u": rmse(U, U_gt, sel), "rmse_F": float(np.sqrt(np.mean((fr.F - F_gt)[sel] ** 2))),
        "rmse_u_local": rmse(fr.U_local, U_gt, sel) if fr.U_local is not None else None,
        "rmse_F_local": float(np.sqrt(np.mean((fr.F_local - F_gt)[sel] ** 2))) if fr.F_local is not None else None,
        "max_err": float(np.max(np.abs((U - U_gt)[sel]))),
        "zncc": float(np.nanmedian(fr.zncc)), "conv": float(np.mean(fr.status == 0)),
        "beta": fr.admm.beta if fr.admm else None, "admm_steps": fr.admm.n_steps if fr.admm else None,
        "res": res, "U_gt": U_gt, "F_gt": F_gt, "sel": sel,
    }
    return row


def text_page(pdf, title, lines, size=9):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.04, 0.95, title, fontsize=15, weight="bold", va="top")
    y = 0.90
    for line in lines:
        fig.text(0.04, y, line, fontsize=size, family="monospace", va="top")
        y -= 0.024
        if y < 0.04:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(11, 8.5))
            y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def main(quick: bool) -> None:
    shape = (80, 88, 96) if quick else (112, 120, 128)
    c = tuple((s - 1) / 2 for s in shape[::-1])
    vol = generate_speckle_volume(shape, sigma=2.0, seed=7)
    F_aff = np.array([[0.02, 0.01, 0.0], [0.0, -0.015, 0.005], [0.003, 0.0, 0.01]])
    base = dvcpara_default(winsize=16, winstepsize=8, search_radius=6, verbose=False)
    rows = []
    lines_meta = [f"pyALDVC {__version__}   volume {shape} (z,y,x)   speckle sigma=2   winsize=16 step=8 (unless noted)",
                  "RMSE in voxels over interior valid nodes; 'F' = displacement-gradient RMSE over all 9 components.", ""]

    # --- 1. canonical cases ---
    d_tr = affine_displacement(None, (3.4, -2.6, 1.2), c)
    rows.append(run_case("translation (3.4,-2.6,1.2)", vol, warp_volume_lagrangian(vol, d_tr), d_tr, base))
    d_af = affine_displacement(F_aff, (1.7, -0.4, 2.3), c)
    g_af = warp_volume_lagrangian(vol, d_af)
    rows.append(run_case("affine 2% (fem)", vol, g_af, d_af, base))
    rows.append(run_case("affine 2% (fd)", vol, g_af, d_af, replace(base, subpb2_method="fd")))
    rows.append(run_case("affine 2% local only", vol, g_af, d_af, replace(base, use_global_step=False)))
    rows.append(run_case("affine 2% dual=reset", vol, g_af, d_af, replace(base, dual_update="reset")))
    d_rot = rotation_displacement(5.0, "z", c)
    rows.append(run_case("rotation 5 deg about z", vol, warp_volume_lagrangian(vol, d_rot), d_rot, base))
    d_big = affine_displacement(0.5 * F_aff, (12.3, -9.6, 7.4), c)
    rows.append(run_case("large motion (12,-10,7)+1%", vol, warp_volume_lagrangian(vol, d_big), d_big,
                         replace(base, search_radius=4)))
    d_sin = sinusoidal_displacement(1.0, 80.0, c)
    g_sin = warp_volume_lagrangian(vol, d_sin)
    rows.append(run_case("sinusoid A=1 L=80", vol, g_sin, d_sin, replace(base, winstepsize=4)))
    nz, ny, nx = shape
    Z, Y, X = np.mgrid[0:nz, 0:ny, 0:nx]
    cyl = (X - c[0]) ** 2 + (Y - c[1]) ** 2 < (min(nx, ny) * 0.38) ** 2
    rows.append(run_case("affine 2% + cylinder mask", vol, g_af, d_af, base, masks=[cyl, cyl]))
    d1 = affine_displacement(0.5 * F_aff, (0.8, -0.2, 1.1), c)
    g1 = warp_volume_lagrangian(vol, d1)
    rows.append(run_case("3 frames accumulative", vol, g_af, d_af, base, frames=[vol, g1, g_af]))
    rows.append(run_case("3 frames incremental", vol, g_af, d_af, replace(base, reference_mode="incremental"),
                         frames=[vol, g1, g_af]))
    for interp in ("linear", "bspline"):
        rows.append(run_case(f"affine 2% interp={interp}", vol, g_af, d_af, replace(base, interp_method=interp)))

    # --- 2. noise study ---
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03] if not quick else [0.0, 0.01, 0.03]
    noise_rows = []
    for nl in noise_levels:
        fn, gn = add_noise(vol, nl, 1), add_noise(g_af, nl, 2)
        for pf in (0.0, 0.8):
            r = run_case(f"noise {nl} pf {pf}", fn, gn, d_af, replace(base, prefilter_sigma=pf))
            noise_rows.append((nl, pf, r))

    # --- 3. spatial resolution study ---
    wl = [40.0, 60.0, 80.0, 120.0, 160.0] if not quick else [40.0, 80.0, 160.0]
    res_rows = []
    for L in wl:
        d = sinusoidal_displacement(1.0, L, c)
        gs = warp_volume_lagrangian(vol, d)
        for ws in (12, 16, 24):
            r = run_case(f"L={L} ws={ws}", vol, gs, d, replace(base, winsize=ws, winstepsize=4, search_radius=4))
            res_rows.append((L, ws, r))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(OUT)) as pdf:
        lines = list(lines_meta)
        lines.append(f"{'case':<32}{'nodes':>6}{'time':>7}  {'RMSE u':>8}{'RMSE v':>8}{'RMSE w':>8}  {'RMSE F':>9}  {'F local':>9}  {'max|e|':>7}  {'ZNCC':>6}  {'beta':>9}")
        for r in rows:
            fl = "" if r["rmse_F_local"] is None else f"{r['rmse_F_local']:9.5f}"
            b = "" if r["beta"] is None else f"{r['beta']:9.2e}"
            lines.append(f"{r['case']:<32}{r['nodes']:>6}{r['time_s']:>7.1f}  {r['rmse_u'][0]:8.4f}{r['rmse_u'][1]:8.4f}{r['rmse_u'][2]:8.4f}  {r['rmse_F']:9.5f}  {fl:>9}  {r['max_err']:7.3f}  {r['zncc']:6.3f}  {b:>9}")
        lines += ["", "Noise study (affine 2%, noise sd relative to a [0,1] speckle with sd 0.06):",
                  f"{'noise':>7}{'presmooth':>10}  {'RMSE u':>8}{'RMSE v':>8}{'RMSE w':>8}  {'RMSE F':>9}  {'ZNCC':>6}"]
        for nl, pf, r in noise_rows:
            lines.append(f"{nl:7.3f}{pf:10.1f}  {r['rmse_u'][0]:8.4f}{r['rmse_u'][1]:8.4f}{r['rmse_u'][2]:8.4f}  {r['rmse_F']:9.5f}  {r['zncc']:6.3f}")
        lines += ["", "Spatial resolution (sinusoid amplitude 1 voxel; first-order subsets carry a curvature bias):",
                  f"{'L':>6}{'winsize':>8}  {'RMSE u':>8}{'RMSE v':>8}{'RMSE w':>8}  {'max|e|':>7}"]
        for L, ws, r in res_rows:
            lines.append(f"{L:6.0f}{ws:8d}  {r['rmse_u'][0]:8.4f}{r['rmse_u'][1]:8.4f}{r['rmse_u'][2]:8.4f}  {r['max_err']:7.3f}")
        text_page(pdf, "pyALDVC synthetic validation", lines, size=7.5)

        # noise figure
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for pf, style in ((0.0, "o-"), (0.8, "s--")):
            xs = [nl for nl, p, _ in noise_rows if p == pf]
            ys = [np.mean(r["rmse_u"]) for nl, p, r in noise_rows if p == pf]
            yF = [r["rmse_F"] for nl, p, r in noise_rows if p == pf]
            axes[0].plot(xs, ys, style, label=f"presmooth sigma={pf}")
            axes[1].plot(xs, yF, style, label=f"presmooth sigma={pf}")
        axes[0].set_xlabel("noise sd"); axes[0].set_ylabel("mean RMSE of u,v,w [voxel]"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("noise sd"); axes[1].set_ylabel("RMSE of displacement gradient"); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.suptitle("Noise robustness (affine 2%, winsize 16, step 8)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # resolution figure
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for ws, style in ((12, "o-"), (16, "s-"), (24, "^-")):
            xs = [L for L, w, _ in res_rows if w == ws]
            ys = [np.mean(r["rmse_u"]) for L, w, r in res_rows if w == ws]
            ax.plot(xs, ys, style, label=f"winsize {ws}")
        ax.set_xlabel("wavelength of the sinusoidal field [voxel]"); ax.set_ylabel("mean RMSE [voxel]")
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend()
        ax.set_title("Spatial resolution: amplitude 1 voxel sinusoid")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # local vs ALDVC gradient, dual update
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        r_fem = rows[1]; r_loc = rows[3]; r_reset = rows[4]
        sel = r_fem["sel"]
        histogram_panel(axes[0], (r_loc["res"].result_disp[0].F - r_loc["F_gt"])[sel][:, 0, 0], "local IC-GN: F11 error")
        histogram_panel(axes[1], (r_fem["res"].result_disp[0].F - r_fem["F_gt"])[sel][:, 0, 0], "AL-DVC (accumulate): F11 error")
        histogram_panel(axes[2], (r_reset["res"].result_disp[0].F - r_reset["F_gt"])[sel][:, 0, 0], "AL-DVC (dual reset): F11 error")
        for ax in axes:
            ax.set_xlim(-0.01, 0.01)
        fig.suptitle("Global step reduces gradient noise: F11 error distributions (affine 2%)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ADMM convergence
        a = r_fem["res"].result_disp[0].admm
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].semilogy(range(1, len(a.primal_residual_u) + 1), a.primal_residual_u, "o-", label="rms|u-u_hat|")
        axes[0].semilogy(range(1, len(a.primal_residual_f) + 1), a.primal_residual_f, "s-", label="rms|F-grad u_hat|")
        axes[0].set_xlabel("ADMM step"); axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_title(f"primal residuals (beta={a.beta:.2e})")
        axes[1].semilogy(range(2, 2 + len(a.update_global)), a.update_global, "o-", label="|dU| global")
        axes[1].semilogy(range(2, 2 + len(a.update_local)), a.update_local, "s-", label="|dU| local")
        axes[1].axhline(base.admm_tol, color="k", ls="--", label="admm_tol"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[1].set_xlabel("ADMM step"); axes[1].set_title("ADMM updates")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # field pages: affine + sinusoid
        for r in (rows[1], rows[7]):
            res = r["res"]; mesh = res.dvc_mesh; fr = res.result_disp[0]
            U = fr.U_accum if fr.U_accum is not None else fr.U
            for comp, nm in enumerate("uvw"):
                fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
                plot_field_slices(mesh.to_grid(U[:, comp]), mesh, title=f"{r['case']}: {nm}", fig=fig, axes=axes[0])
                err = np.where(r["sel"], U[:, comp] - r["U_gt"][:, comp], np.nan)
                lim = np.nanpercentile(np.abs(err), 99)
                plot_field_slices(mesh.to_grid(err), mesh, title=f"error {nm}", fig=fig, axes=axes[1], cmap="coolwarm", vmin=-lim, vmax=lim)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
            sr = res.result_strain[0]
            fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
            plot_field_slices(mesh.to_grid(sr.field("exx")), mesh, title=f"{r['case']}: exx", fig=fig, axes=axes[0])
            plot_field_slices(mesh.to_grid(sr.field("exy")), mesh, title=f"{r['case']}: exy", fig=fig, axes=axes[1])
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # mask case
        r = rows[8]
        res = r["res"]; mesh = res.dvc_mesh
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
        vals = np.where(mesh.node_valid, res.result_disp[0].U[:, 0], np.nan)
        plot_field_slices(mesh.to_grid(vals), mesh, title="cylinder mask: u", fig=fig, axes=axes[0])
        plot_field_slices(mesh.to_grid(res.result_strain[0].field("exx")), mesh, title="cylinder mask: exx (edge-trimmed)", fig=fig, axes=axes[1])
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(args.quick)
