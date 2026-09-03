#!/usr/bin/env python
"""Report on deformed-frame masks.

Two synthetic cases where part of the deformed volume is not the deformed
reference (a corrupted slab; a void that appears): with a deformed-frame mask
the kernels drop the affected subset voxels and the nodes next to the region
converge to the truth, without it they are biased or lost. Writes
``reports/deformed_mask.pdf``. Reports are generated, never hand-edited.

Usage::

    python scripts/make_mask_report.py [--quick]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__  # noqa: E402
from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.data_structures import STATUS_CONVERGED, STATUS_NAMES  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.synthetic import (  # noqa: E402
    affine_displacement,
    evaluate_at_nodes,
    generate_speckle_volume,
    warp_volume_lagrangian,
)

SHAPE = (96, 104, 112)
F_AFFINE = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])  # displacement gradient
T_AFFINE = np.array([1.3, -0.7, 0.4])
SLAB_X = 80
VOID_CENTRE = (56, 52, 48)  # (x, y, z)
VOID_RADIUS = 14


def interior(mesh, layers=1):
    n = np.arange(mesh.n_nodes)
    gz, gy, gx = mesh.grid_shape
    iz, iy, ix = n // (gy * gx), (n // gx) % gy, n % gx
    return (ix >= layers) & (ix < gx - layers) & (iy >= layers) & (iy < gy - layers) & (iz >= layers) & (iz < gz - layers)


def corrupt(g, region, seed=3):
    gb = np.array(g, copy=True)
    rng = np.random.default_rng(seed)
    gb[region] = rng.normal(float(g.mean()), float(g.std()), size=int(region.sum())).astype(gb.dtype)
    return gb, ~region


def run(f, g, disp, masks, use_global, winsize):
    para = dvcpara_default(winsize=winsize, winstepsize=8, search_radius=5, verbose=False, use_global_step=use_global)
    t0 = time.perf_counter()
    r = run_aldvc(para, [f, g], masks=masks, compute_strain=False)
    fr = r.result_disp[0]
    mesh = r.dvc_mesh
    U_gt = evaluate_at_nodes(disp, mesh.coordinates)
    return {"r": r, "mesh": mesh, "fr": fr, "err": fr.U - U_gt, "time": time.perf_counter() - t0}


def dist_to_region(mesh, region_mask):
    """Distance (voxels) of each node to the nearest masked voxel, along the grid (approximate)."""
    from scipy.ndimage import distance_transform_edt

    dt = distance_transform_edt(region_mask)  # distance to the nearest False (= masked) voxel
    c = mesh.coordinates.astype(int)
    return dt[c[:, 2], c[:, 1], c[:, 0]]


def text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.96, title, fontsize=15, weight="bold", va="top")
    y = 0.92
    for line in lines:
        fig.text(0.05, y, line, fontsize=8.3, family="monospace", va="top")
        y -= 0.016
    pdf.savefig(fig)
    plt.close(fig)


def case_rows(name, runs, mesh, dist, half):
    lines = [
        f"{name}:",
        f"  {'run':<28} {'nodes':>6} {'conv':>6}  {'rmse touching':>14} {'rmse far':>9}  {'status':<40} {'t [s]':>6}",
    ]
    inner = interior(mesh)
    touching = inner & (dist <= half + 10)  # the two node layers next to the region (step 8)
    far = inner & (dist > half + 14)
    for label, res in runs.items():
        fr, err = res["fr"], res["err"]
        conv = fr.status == STATUS_CONVERGED
        rm_t = np.sqrt(np.mean(np.sum(err[touching & conv] ** 2, axis=1))) if (touching & conv).any() else np.nan
        rm_f = np.sqrt(np.mean(np.sum(err[far & conv] ** 2, axis=1))) if (far & conv).any() else np.nan
        codes, counts = np.unique(fr.status[touching], return_counts=True)
        st = ", ".join(f"{STATUS_NAMES.get(int(c), c)}={n}" for c, n in zip(codes, counts))
        lines.append(
            f"  {label:<28} {int(touching.sum()):>6} {int((touching & conv).sum()):>6}  "
            f"{rm_t:14.4f} {rm_f:9.4f}  {st:<40} {res['time']:6.1f}"
        )
    lines.append("")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports" / "deformed_mask.pdf"))
    args = ap.parse_args(argv)
    winsize = 16
    half = winsize // 2

    disp = affine_displacement(F_AFFINE, T_AFFINE, tuple(s / 2 for s in SHAPE[::-1]))
    f = generate_speckle_volume(SHAPE, sigma=2.0, seed=11)
    g = warp_volume_lagrangian(f, disp)
    ones = np.ones(f.shape, dtype=bool)
    zz, yy, xx = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), np.arange(SHAPE[2]), indexing="ij")
    cases = {
        "corrupted slab x >= 80": xx >= SLAB_X,
        "void of radius 14": (xx - VOID_CENTRE[0]) ** 2 + (yy - VOID_CENTRE[1]) ** 2 + (zz - VOID_CENTRE[2]) ** 2
        <= VOID_RADIUS**2,
    }
    if args.quick:
        cases = dict(list(cases.items())[:1])
    results = {}
    for name, region in cases.items():
        g_bad, mask_g = corrupt(g, region)
        runs = {
            "clean g, no mask": run(f, g, disp, None, False, winsize),
            "corrupted g, no mask": run(f, g_bad, disp, None, False, winsize),
            "corrupted g, deformed mask": run(f, g_bad, disp, [ones, mask_g], False, winsize),
            "corrupted g, mask + AL-DVC": run(f, g_bad, disp, [ones, mask_g], True, winsize),
        }
        results[name] = (runs, mask_g, g_bad)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        lines = [f"pyALDVC {__version__} -- deformed-frame masks", ""]
        lines.append("Mechanism: masked voxels of the deformed volume are NaN in the sampled array; a subset voxel whose")
        lines.append("interpolation stencil touches them is dropped from that node's ZNSSD (reference statistics and")
        lines.append("Jacobian sums are recomputed on the remaining voxels each iteration). Nodes with fewer than 27")
        lines.append("valid voxels get status invalid_subset. The reference-frame mask works as before.")
        lines.append(f"Speckle {SHAPE[::-1]} (x,y,z), affine 2 %, subset {winsize}, step 8, local solution unless noted.")
        lines.append("'touching' = interior nodes within winsize/2 + 10 voxels of the masked region (two node layers;")
        lines.append("the layer whose subsets lose more than half their voxels is reported invalid_subset and inpainted),")
        lines.append("'far' = more than winsize/2 + 14 voxels away. rmse over converged nodes, in voxels.")
        lines.append("")
        for name, (runs, mask_g, g_bad) in results.items():
            mesh = runs["clean g, no mask"]["mesh"]
            dist = dist_to_region(mesh, mask_g)
            lines += case_rows(name, runs, mesh, dist, half)
        text_page(pdf, "Deformed-frame masks: effect on the nodes next to a masked region", lines)

        for name, (runs, mask_g, g_bad) in results.items():
            mesh = runs["clean g, no mask"]["mesh"]
            iz = mesh.grid_shape[0] // 2
            zc = int(mesh.z0[iz])
            fig, axes = plt.subplots(2, 3, figsize=(13, 8))
            axes[0, 0].imshow(g_bad[zc], cmap="gray", origin="lower")
            axes[0, 0].contour(mask_g[zc], levels=[0.5], colors="r", linewidths=0.8)
            axes[0, 0].set_title(f"deformed volume, z = {zc} (red: mask boundary)", fontsize=9)
            for ax, key in zip(
                [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]],
                ["clean g, no mask", "corrupted g, no mask", "corrupted g, deformed mask", "corrupted g, mask + AL-DVC"],
            ):
                res = runs[key]
                emag = np.linalg.norm(res["err"], axis=1)
                emag = np.where(res["fr"].status == STATUS_CONVERGED, emag, np.nan)
                im = ax.imshow(
                    mesh.to_grid(emag)[iz],
                    origin="lower",
                    cmap="magma",
                    vmin=0,
                    vmax=0.2,
                    extent=[mesh.x0[0], mesh.x0[-1], mesh.y0[0], mesh.y0[-1]],
                )
                ax.set_title(f"|error| [voxel], {key}", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046)
            st = runs["corrupted g, deformed mask"]["fr"].status
            codes, counts = np.unique(st, return_counts=True)
            axes[1, 2].bar([STATUS_NAMES.get(int(c), str(c)) for c in codes], counts, color="#dd8452")
            axes[1, 2].set_title("node status with the deformed mask", fontsize=9)
            axes[1, 2].tick_params(axis="x", rotation=30, labelsize=7)
            fig.suptitle(name)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
