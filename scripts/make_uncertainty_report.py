#!/usr/bin/env python
"""Report on the per-node displacement uncertainty estimate (``FrameResult.U_std``).

Synthetic speckle volumes with an exact affine deformation and additive
Gaussian noise: the predicted standard deviation (from the IC-GN normal
equations, see ``al_dvc.solver.uncertainty``) is compared with the empirical
error against the ground truth, per noise level, subset size and texture
anisotropy. Writes ``reports/uncertainty.pdf``. Reports are generated, never
hand-edited.

Usage::

    python scripts/make_uncertainty_report.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__  # noqa: E402
from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.data_structures import STATUS_CONVERGED  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.synthetic import (  # noqa: E402
    add_noise,
    affine_displacement,
    evaluate_at_nodes,
    generate_speckle_volume,
    warp_volume_lagrangian,
)

SHAPE = (96, 104, 112)
F_AFFINE = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])  # displacement gradient
T_AFFINE = np.array([1.3, -0.7, 0.4])
NOISES = (0.005, 0.01, 0.02, 0.04)
SUBSETS = (16, 24)


def interior(mesh, layers=1):
    n = np.arange(mesh.n_nodes)
    gz, gy, gx = mesh.grid_shape
    iz, iy, ix = n // (gy * gx), (n // gx) % gy, n % gx
    return (ix >= layers) & (ix < gx - layers) & (iy >= layers) & (iy < gy - layers) & (iz >= layers) & (iz < gz - layers)


def run_case(f, g, disp, noise, winsize, seed=1):
    fn = add_noise(f, noise, seed=seed) if noise > 0 else f
    gn = add_noise(g, noise, seed=seed + 1) if noise > 0 else g
    para = dvcpara_default(winsize=winsize, winstepsize=8, search_radius=5, verbose=False, use_global_step=False)
    t0 = time.perf_counter()
    r = run_aldvc(para, [fn, gn], compute_strain=False)
    fr = r.result_disp[0]
    mesh = r.dvc_mesh
    U_gt = evaluate_at_nodes(disp, mesh.coordinates)
    ok = interior(mesh) & (fr.status == STATUS_CONVERGED) & np.all(np.isfinite(fr.U_std), axis=1)
    err = (fr.U - U_gt)[ok]
    std = fr.U_std[ok]
    return {
        "noise": noise,
        "winsize": winsize,
        "n": int(ok.sum()),
        "emp": np.sqrt(np.mean(err**2, axis=0)),
        "pred": np.sqrt(np.mean(std**2, axis=0)),
        "err": err,
        "std": std,
        "time": time.perf_counter() - t0,
        "mesh": mesh,
        "U_std": fr.U_std,
    }


def text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.96, title, fontsize=15, weight="bold", va="top")
    y = 0.92
    for line in lines:
        fig.text(0.05, y, line, fontsize=8.5, family="monospace", va="top")
        y -= 0.017
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer cases")
    ap.add_argument("--out", default=str(ROOT / "reports" / "uncertainty.pdf"))
    args = ap.parse_args(argv)
    noises = NOISES[1:3] if args.quick else NOISES
    subsets = SUBSETS[:1] if args.quick else SUBSETS

    disp = affine_displacement(F_AFFINE, T_AFFINE, tuple(s / 2 for s in SHAPE[::-1]))
    f = generate_speckle_volume(SHAPE, sigma=2.0, seed=11)
    g = warp_volume_lagrangian(f, disp)
    rows = [run_case(f, g, disp, nz, ws) for ws in subsets for nz in noises]
    f_b = gaussian_filter(f, sigma=(2.5, 0.0, 0.0), mode="nearest")
    g_b = warp_volume_lagrangian(f_b, disp)
    aniso = run_case(f_b, g_b, disp, 0.02, 16)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        lines = [f"pyALDVC {__version__} -- per-node displacement uncertainty (FrameResult.U_std)", ""]
        lines.append(
            "Cov(P) = 2 s^2 H0^-1 + c s^4 H0^-1 (I3 (x) M) H0^-1, s^2 = (1 - ZNCC) bottomf^2 / n, H0 = H - c s^2 (I3 (x) M);"
        )
        lines.append("converged nodes of the local (subset) solution (see al_dvc.solver.uncertainty),")
        lines.append(f"speckle {SHAPE[::-1]} (x,y,z), affine 2 % deformation, additive Gaussian noise on both volumes,")
        lines.append("step 8, interior converged nodes. 'emp' = RMSE against the ground truth, 'pred' = rms of U_std.")
        lines.append("")
        lines.append(
            f"{'subset':>6} {'noise':>6} {'nodes':>6}  {'emp u':>7} {'emp v':>7} {'emp w':>7}  "
            f"{'pred u':>7} {'pred v':>7} {'pred w':>7}  {'ratio u':>7} {'ratio v':>7} {'ratio w':>7}"
        )
        for r in rows:
            e, p = r["emp"], r["pred"]
            lines.append(
                f"{r['winsize']:>6} {r['noise']:>6.3f} {r['n']:>6}  {e[0]:7.4f} {e[1]:7.4f} {e[2]:7.4f}  "
                f"{p[0]:7.4f} {p[1]:7.4f} {p[2]:7.4f}  {p[0] / e[0]:7.2f} {p[1] / e[1]:7.2f} {p[2] / e[2]:7.2f}"
            )
        e, p = aniso["emp"], aniso["pred"]
        lines.append("")
        lines.append("Texture blurred along z (sigma 2.5 voxel), noise 0.02, subset 16:")
        lines.append(
            f"{'':>6} {'':>6} {aniso['n']:>6}  {e[0]:7.4f} {e[1]:7.4f} {e[2]:7.4f}  {p[0]:7.4f} {p[1]:7.4f} {p[2]:7.4f}  "
            f"{p[0] / e[0]:7.2f} {p[1] / e[1]:7.2f} {p[2] / e[2]:7.2f}"
        )
        lines.append("")
        lines.append("Reading: ratios near 1 mean the estimate is calibrated. At the lowest noise the empirical error")
        lines.append("contains the interpolation and shape-function bias that the estimate does not model, so the ratio")
        lines.append("drops below 1. The estimate refers to the local solution; the AL-DVC global step lowers the actual")
        lines.append("error further. Exported as disp_std_u/v/w and disp_std (npz: U_std, vti: displacement_std).")
        text_page(pdf, "Displacement uncertainty: predicted vs empirical", lines)

        # calibration scatter: |error| binned by predicted std
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
        for c, comp in enumerate("uvw"):
            ax = axes[c]
            for r in rows:
                if r["winsize"] != subsets[0]:
                    continue
                s = r["std"][:, c]
                a = np.abs(r["err"][:, c])
                q = np.quantile(s, np.linspace(0, 1, 9))
                xs, ys = [], []
                for lo, hi in zip(q[:-1], q[1:]):
                    sel = (s >= lo) & (s <= hi)
                    if sel.sum() > 5:
                        xs.append(s[sel].mean())
                        ys.append(np.sqrt(np.mean(a[sel] ** 2)))
                ax.plot(xs, ys, "o-", ms=4, label=f"noise {r['noise']:.3f}")
            lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="rms error = predicted std")
            ax.set_xlabel(f"predicted std {comp} [voxel]")
            ax.set_ylabel(f"rms error {comp} [voxel]")
            ax.set_title(f"calibration, subset {subsets[0]}", fontsize=9)
            ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # anisotropy maps
        mesh = aniso["mesh"]
        std = aniso["U_std"]
        iz = mesh.grid_shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
        for c, comp in enumerate("uvw"):
            grid = mesh.to_grid(std[:, c])[iz]
            im = axes[c].imshow(grid, origin="lower", cmap="magma", vmin=0, vmax=np.nanpercentile(std, 99))
            axes[c].set_title(f"predicted std {comp} [voxel], z-blurred texture", fontsize=9)
            fig.colorbar(im, ax=axes[c], fraction=0.046)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # histograms per noise level
        fig, axes = plt.subplots(1, len(noises), figsize=(3.3 * len(noises), 3.6))
        for ax, r in zip(np.atleast_1d(axes), [x for x in rows if x["winsize"] == subsets[0]]):
            ax.hist(np.linalg.norm(r["std"], axis=1), bins=40, color="#4c72b0", alpha=0.7, label="predicted |std|")
            ax.hist(np.linalg.norm(r["err"], axis=1), bins=40, color="#dd8452", alpha=0.6, label="|error|")
            ax.set_title(f"noise {r['noise']:.3f}", fontsize=9)
            ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"report: {out}")
    for r in rows:
        print(
            f"subset {r['winsize']} noise {r['noise']:.3f}: emp {r['emp'].round(4)} pred {r['pred'].round(4)} ({r['time']:.1f} s)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
