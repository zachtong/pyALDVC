#!/usr/bin/env python
"""Report on subset splitting at boundaries (``subset_split``).

Synthetic cases with a known deformation: two rigid bodies separated by a masked wall, a crack
that stops at a front, a porous phantom. Shows what a split subset keeps, the displacement error
next to the wall with and without splitting (local pass and full ADMM), where the split cannot
help, the effect on a porous material, and the cost. Writes ``reports/subset_split.pdf``.
Reports are generated, never hand-edited.

Usage::

    python scripts/make_subset_split_report.py [--quick]
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
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.io.volume_ops import build_reference_bundle, normalize_volume  # noqa: E402
from al_dvc.mesh.grid_mesh import apply_mask_to_mesh, build_grid_axes, mesh_setup  # noqa: E402
from al_dvc.solver.local_icgn import split_rows  # noqa: E402
from al_dvc.solver.numba_kernels import build_split_rows  # noqa: E402
from al_dvc.solver.reference_kernels import keep_from_row, subset_count  # noqa: E402
from al_dvc.synthetic import (  # noqa: E402
    affine_displacement,
    evaluate_at_nodes,
    generate_speckle_volume,
    two_body_displacement,
    warp_volume_lagrangian,
)

SHAPE = (64, 72, 96)  # (nz, ny, nx)
WALL = 45.5  # the wall occupies x = 45, 46
WIN, STEP = 24, 8
T_BELOW = (0.6, -0.3, 0.2)
T_ABOVE = (-0.5, 0.4, -0.3)
F_AFFINE = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])


def two_bodies(front_y: int | None = None):
    """Reference, deformed, displacement and masks of two bodies separated by a wall (optionally with a front)."""
    f = generate_speckle_volume(SHAPE, sigma=2.0, seed=7)
    disp = two_body_displacement("x", WALL, affine_displacement(t=T_BELOW), affine_displacement(t=T_ABOVE))
    g = warp_volume_lagrangian(f, disp)
    mask_ref = np.ones(SHAPE, np.uint8)
    mask_def = np.ones(SHAPE, np.uint8)
    y_end = SHAPE[1] if front_y is None else front_y
    mask_ref[:, :y_end, 45:47] = 0
    mask_def[:, :y_end, 44:48] = 0
    return f, g, disp, mask_ref, mask_def


def run(f, g, disp, masks, split: bool, global_step: bool, winsize=WIN, step=STEP):
    para = dvcpara_default(
        winsize=winsize,
        winstepsize=step,
        search_radius=5,
        verbose=False,
        backend="numba",
        subset_split=split,
        use_global_step=global_step,
        local_outlier_threshold=0.0,
    )
    t0 = time.perf_counter()
    res = run_aldvc(para, [f, g], masks=masks, compute_strain=True, resume=False)
    dt = time.perf_counter() - t0
    fr, mesh = res.result_disp[0], res.dvc_mesh
    err = np.linalg.norm(fr.U - evaluate_at_nodes(disp, mesh.coordinates), axis=1)
    return res, fr, mesh, err, dt


# ---------------------------------------------------------------------- pages
def page_subset(pdf, f, mask_ref):
    """What a split subset keeps: the XY slice through the centre of a subset next to the wall."""
    half = (WIN // 2,) * 3
    coords = np.array([[40, 36, 32]], dtype=np.int64)
    rows, n_keep, n_inmask = build_split_rows(coords, np.array([0]), *half, 1, mask_ref, 0)
    S = subset_count(half, 1)
    keep = keep_from_row(rows[0], S).reshape(2 * half[2] + 1, 2 * half[1] + 1, 2 * half[0] + 1)
    x0, y0, z0 = coords[0]
    sl = (z0, slice(y0 - half[1], y0 + half[1] + 1), slice(x0 - half[0], x0 + half[0] + 1))
    img = f[sl]
    m = mask_ref[sl] > 0
    k = keep[half[2]]
    rgb = np.repeat(((img - img.min()) / (np.ptp(img) + 1e-12))[..., None], 3, axis=-1)
    rgb[~m] = (0.25, 0.25, 0.25)
    rgb[m & ~k] = rgb[m & ~k] * 0.4 + np.array([0.6, 0.1, 0.1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(rgb, origin="lower", extent=[x0 - half[0] - 0.5, x0 + half[0] + 0.5, y0 - half[1] - 0.5, y0 + half[1] + 0.5])
    axes[0].plot([x0], [y0], "c+", ms=14, mew=2)
    axes[0].set_title(f"XY slice, subset centre x = {x0}\ngrey kept, red dropped, dark mask", fontsize=10)
    axes[0].set_xlabel("x [voxel]")
    axes[0].set_ylabel("y [voxel]")
    prof = keep.sum(axis=(0, 1))
    axes[1].bar(np.arange(-half[0], half[0] + 1) + x0, prof, color="steelblue")
    axes[1].axvline(WALL, color="k", ls="--")
    axes[1].set_xlabel("x [voxel]")
    axes[1].set_ylabel("kept voxels per x layer")
    axes[1].set_title(
        f"kept {int(n_keep[0])} of {int(n_inmask[0])} in-mask voxels\n({100 * n_keep[0] / n_inmask[0]:.0f} % of the subset)",
        fontsize=10,
    )
    fig.suptitle(f"pyALDVC {__version__}: a subset next to a masked wall keeps the side of its centre", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_two_bodies(pdf):
    f, g, disp, mask_ref, mask_def = two_bodies()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    rows = []
    for split, global_step, style in [(False, False, "r--"), (True, False, "b--"), (False, True, "r-"), (True, True, "b-")]:
        res, fr, mesh, err, dt = run(f, g, disp, [mask_ref, mask_def], split, global_step)
        valid = np.asarray(mesh.node_valid, dtype=bool)
        dist = np.abs(mesh.coordinates[:, 0] - WALL)
        bins = np.unique(np.round(dist[valid]))
        rmse = [float(np.sqrt(np.mean(err[valid & (np.round(dist) == b)] ** 2))) for b in bins]
        label = f"split {'on' if split else 'off'}, {'full ADMM' if global_step else 'local only'}"
        axes[0].plot(bins, rmse, style, marker="o", ms=4, label=label)
        near = valid & (dist <= WIN / 2) & (dist > 1)
        far = valid & (dist > WIN)
        rows.append((label, np.sqrt(np.mean(err[near] ** 2)), np.sqrt(np.mean(err[far] ** 2)), dt))
        if split and global_step:
            sr = res.result_strain[0]
            exx = np.asarray(sr.exx)
            axes[1].plot(dist[valid], np.abs(exx[valid]), "b.", ms=4, label="split on, |exx|")
        if (not split) and global_step:
            sr = res.result_strain[0]
            exx = np.asarray(sr.exx)
            axes[1].plot(dist[valid], np.abs(exx[valid]), "r.", ms=4, label="split off, |exx|")
    axes[0].axvline(WIN / 2, color="k", ls=":", lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("distance of the node from the wall [voxel]")
    axes[0].set_ylabel("displacement RMSE [voxel]")
    axes[0].set_title(
        f"two rigid bodies, jump {np.linalg.norm(np.subtract(T_ABOVE, T_BELOW)):.2f} voxel, subset {WIN}, step {STEP}"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("distance from the wall [voxel]")
    axes[1].set_ylabel("|exx| (rigid bodies: truth 0)")
    axes[1].set_title("strain from the plane fit, full ADMM")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    text = "\n".join(f"{r[0]:<26} near-wall RMSE {r[1]:.4f}   far RMSE {r[2]:.4f}   {r[3]:.1f} s" for r in rows)
    fig.text(0.05, 0.01, text, fontsize=8, family="monospace", va="bottom")
    fig.suptitle(f"pyALDVC {__version__}: displacement error next to a masked wall", fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)
    return rows


def page_front(pdf):
    """A crack that stops halfway: the split changes only the nodes whose window the crack really separates."""
    front = SHAPE[1] // 2
    f, g, disp, mask_ref, mask_def = two_bodies(front_y=front)
    res, fr, mesh, err, dt = run(f, g, disp, [mask_ref, mask_def], True, True)
    gz, gy, gx = mesh.grid_shape
    x = mesh.coordinates[:, 0]
    layer = np.flatnonzero(np.isclose(x, np.unique(x[x < WALL])[-1]))  # the node column just below the wall
    sf = np.asarray(fr.split_fraction)[layer].reshape(gz, gy)
    e = err[layer].reshape(gz, gy)
    ys = mesh.coordinates[layer, 1].reshape(gz, gy)[0]
    zs = mesh.coordinates[layer, 2].reshape(gz, gy)[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    im0 = axes[0].imshow(
        sf, origin="lower", extent=[ys[0], ys[-1], zs[0], zs[-1]], vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto"
    )
    axes[0].axvline(front, color="w", ls="--")
    axes[0].set_title("split_fraction of the node column next to the wall", fontsize=10)
    axes[0].set_xlabel("y [voxel]  (crack for y < front, dashed)")
    axes[0].set_ylabel("z [voxel]")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(e, origin="lower", extent=[ys[0], ys[-1], zs[0], zs[-1]], cmap="magma", aspect="auto")
    axes[1].axvline(front, color="w", ls="--")
    axes[1].set_title("displacement error [voxel]", fontsize=10)
    axes[1].set_xlabel("y [voxel]")
    fig.colorbar(im1, ax=axes[1])
    fig.suptitle(
        f"pyALDVC {__version__}: at a crack front the window stays connected and the subset is not split\n"
        "the mask ends at the dashed line; within half a subset of it the two sides are still joined inside the window,\n"
        "so split_fraction returns to 1 and the error stays as large as without the split",
        fontsize=11,
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def porous_mask(shape, porosity: float, seed: int = 11):
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.normal(size=shape), 4.0)
    thr = np.quantile(field, porosity)
    return (field > thr).astype(np.uint8)


def page_porous(pdf, quick: bool):
    """Affine deformation of a porous body: the share of subsets that touch a pore and the error with and without the split."""
    shape = (64, 64, 64)
    f = generate_speckle_volume(shape, sigma=2.0, seed=3)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    disp = affine_displacement(F_AFFINE, (0.8, -0.4, 0.3), centre)
    g = warp_volume_lagrangian(f, disp)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    porosities = (0.1, 0.2, 0.3) if quick else (0.05, 0.1, 0.2, 0.3, 0.4)
    table = []
    for por in porosities:
        mask = porous_mask(shape, por)
        r = {}
        for split in (False, True):
            res, fr, mesh, err, dt = run(f, g, disp, [mask, mask], split, True, winsize=16, step=8)
            valid = np.asarray(mesh.node_valid, dtype=bool)
            touch = valid & (np.asarray(fr.split_fraction) < 1.0) if fr.split_fraction is not None else valid
            r[split] = (float(np.sqrt(np.mean(err[valid] ** 2))), float(touch.mean()), int(valid.sum()))
        table.append((por, r))
    ps = [t[0] for t in table]
    axes[0].plot(ps, [t[1][False][0] for t in table], "r-o", lw=3, alpha=0.5, label="split off")
    axes[0].plot(ps, [t[1][True][0] for t in table], "b--s", ms=5, label="split on")
    axes[0].set_xlabel("porosity")
    axes[0].set_ylabel("displacement RMSE over the valid nodes [voxel]")
    axes[0].set_title("affine deformation of a porous body, subset 16, step 8")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(ps, [100 * t[1][True][1] for t in table], "k-o")
    axes[1].set_xlabel("porosity")
    axes[1].set_ylabel("subsets cut by the split [% of the valid nodes]")
    axes[1].grid(alpha=0.3)
    axes[1].set_title("share of subsets that touch a pore")
    fig.suptitle(
        f"pyALDVC {__version__}: porous phantom, smooth (affine) deformation\n"
        "the two curves coincide: with a continuous field the split costs nothing, and it is the discontinuous\n"
        "case above where it pays",
        fontsize=11,
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    return table


def page_cost(pdf, quick: bool):
    """Time and memory of the split rows against the number of subsets that touch a boundary."""
    shape = (96, 96, 96) if quick else (128, 128, 128)
    f = normalize_volume(generate_speckle_volume(shape, sigma=2.0, seed=5))
    para = dvcpara_default(winsize=32, winstepsize=8, verbose=False, subset_split=True)
    x0, y0, z0 = build_grid_axes(para.voi, shape, para.winsize, para.winstepsize)
    base = mesh_setup(x0, y0, z0)
    counts, times, mem = [], [], []
    warm = np.ones((8, 8, 8), np.uint8)
    build_split_rows(np.array([[4, 4, 4]], np.int64), np.array([0]), 2, 2, 2, 1, warm, 0)  # JIT before timing
    for por in (0.0, 0.02, 0.05, 0.1, 0.2):
        mask = porous_mask(shape, por, seed=2) if por > 0 else np.ones(shape, np.uint8)
        ref = build_reference_bundle(f, mask, para.gradient_mode)
        mesh = apply_mask_to_mesh(base, ref.mask, para.winsize, para.min_valid_ratio)
        coords = np.round(mesh.coordinates).astype(np.int64)
        t0 = time.perf_counter()
        out = split_rows(mesh, ref, para, coords, tuple(int(w) // 2 for w in para.winsize), 1)
        dt = time.perf_counter() - t0
        n = 0 if out is None else int(np.count_nonzero(out[0] >= 0))
        counts.append(n)
        times.append(dt)
        mem.append(0 if out is None else out[1].nbytes / 1e6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(counts, times, "k-o")
    axes[0].set_xlabel("subsets that touch a boundary")
    axes[0].set_ylabel("time of the flood fills and packing [s]")
    axes[0].set_title(f"{shape[2]} x {shape[1]} x {shape[0]} voxels, subset 32, {base.n_nodes} nodes", fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[1].plot(counts, mem, "k-o")
    axes[1].set_xlabel("subsets that touch a boundary")
    axes[1].set_ylabel("packed keep rows [MB]")
    axes[1].set_title(f"4.5 kB per split subset\ndense over all nodes: {base.n_nodes * 33**3 / 1e9:.2f} GB", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"pyALDVC {__version__}: cost of the split rows", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = ROOT / "reports" / "subset_split.pdf"
    out.parent.mkdir(exist_ok=True)
    f, g, disp, mask_ref, mask_def = two_bodies()
    with PdfPages(out) as pdf:
        page_subset(pdf, f, mask_ref)
        rows = page_two_bodies(pdf)
        page_front(pdf)
        page_porous(pdf, args.quick)
        page_cost(pdf, args.quick)
    for r in rows:
        print(f"{r[0]:<26} near {r[1]:.4f}  far {r[2]:.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
