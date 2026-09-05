#!/usr/bin/env python
"""Report on the texture analysis. Writes ``reports/texture.pdf``.

Pages
    1. Validation on the Boolean sphere model: measured radial profile against the closed-form
       correlation, and the 1/e and 0.1 lengths against their analytic values.
    2. The crop-before-shift defect of the DVC Challenge scripts reproduced on 64^3 (their
       ordering re-implemented here) next to the corrected estimator.
    3. Finite-window against overlap-corrected estimator across sub-volume sizes.
    4. Directional profiles on an anisotropic texture, with and without anisotropic voxels.
    5. Size sweep on the Boolean model with the plateau decision.
    6. The subset heuristic against the measured displacement error of the DVC pipeline.
    7. Timings and limitations.

Reports are generated, never hand-edited.
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
from scipy.fft import irfftn, next_fast_len, rfftn  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__  # noqa: E402
from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.synthetic import affine_displacement, evaluate_at_nodes, warp_volume_lagrangian  # noqa: E402
from al_dvc.texture import (  # noqa: E402
    THRESHOLDS,
    analyse_texture,
    analytic_length,
    boolean_correlation,
    boolean_spheres,
    recommend_parameters,
    size_schedule,
    sweep_sizes,
)

ONE_OVER_E = THRESHOLDS[0]


def old_script_radial_profile(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The DVC Challenge scripts' estimator: crop to 2N-1 first, then fftshift (wrong when the FFT is longer)."""
    u = np.asarray(vol, dtype=np.float32)
    u = u - u.mean()
    pad = tuple(2 * s - 1 for s in u.shape)
    fft_shape = tuple(next_fast_len(s) for s in pad)
    F = rfftn(u, s=fft_shape)
    F *= np.conj(F)
    acf = irfftn(F, s=fft_shape)[: pad[0], : pad[1], : pad[2]]
    acf /= acf[0, 0, 0]
    acf = np.fft.fftshift(acf)
    c = [s // 2 for s in acf.shape]
    z, y, x = np.ogrid[-c[0] : acf.shape[0] - c[0], -c[1] : acf.shape[1] - c[1], -c[2] : acf.shape[2] - c[2]]
    r = np.sqrt(x**2 + y**2 + z**2)
    b = np.fix(r).astype(np.int64).ravel()
    counts = np.bincount(b)
    sums = np.bincount(b, weights=acf.ravel())
    radii = np.bincount(b, weights=r.ravel())  # the actual mean radius per shell, as the new profiles report it
    return radii / np.maximum(counts, 1), sums / np.maximum(counts, 1)


def first_crossing(x, y, t):
    for i in range(len(y) - 1):
        if y[i] >= t > y[i + 1]:
            return x[i] + (y[i] - t) / (y[i] - y[i + 1]) * (x[i + 1] - x[i])
    return np.nan


def page_validation(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    rows = []
    for ax, (radius, phi, seed) in zip(axes, ((5.0, 0.25, 1), (8.0, 0.4, 2))):
        vol, _ = boolean_spheres((128, 128, 128), radius, phi, seed=seed)
        res = analyse_texture(vol, max_lag=int(3 * radius))
        radial = res.profiles["radial"]
        ok = np.isfinite(radial.mean)
        rr = np.linspace(0, 2.5 * radius, 200)
        ax.plot(rr, boolean_correlation(rr, radius, phi), "k-", lw=2, label="analytic")
        ax.plot(radial.distance[ok], radial.mean[ok], "o", ms=3, label="measured (radial)")
        for axis, style in (("x", "-"), ("y", "--"), ("z", ":")):
            p = res.profiles[axis]
            ax.plot(p.distance, p.mean, style, lw=1, alpha=0.7, label=f"measured ({axis})")
        for t in (ONE_OVER_E, 0.1):
            ax.axhline(t, color="gray", lw=0.6)
            rows.append((radius, phi, t, analytic_length(radius, phi, t), res.length("radial", t), vol.mean()))
        ax.set_title(f"Boolean spheres R = {radius:g}, fraction {phi:g} (realised {vol.mean():.2f})", fontsize=10)
        ax.set_xlabel("distance [voxel]")
        ax.set_ylabel("autocorrelation")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"pyALDVC {__version__}: texture analysis validated on the Boolean sphere model", fontsize=12)
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    text = "\n".join(
        f"R = {r:g}, fraction {p:g}: threshold {t:.3f}  analytic {a:.3f}  measured {m:.3f}  error {100 * (m - a) / a:+.1f} %"
        for r, p, t, a, m, _f in rows
    )
    fig.text(0.08, 0.02, text, fontsize=8, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)
    return rows


def page_crop_bug(pdf):
    vol = gaussian_filter(np.random.default_rng(20260905).normal(size=(64, 64, 64)), sigma=1.2)
    d_old, y_old = old_script_radial_profile(vol)
    res = analyse_texture(vol, estimator="window", max_lag=63, min_overlap=1e-6)
    radial = res.profiles["radial"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, xmax in zip(axes, (12, 40)):
        ax.plot(d_old, y_old, "r.-", ms=3, lw=1, label="scripts: crop, then shift (128-point FFT)")
        ax.plot(radial.lag, radial.mean, "k.-", ms=3, lw=1, label="corrected: shift, then cut")
        ax.axhline(ONE_OVER_E, color="gray", lw=0.6)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("lag [voxel]")
        ax.set_ylabel("autocorrelation (finite-window estimator)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    l_old = first_crossing(d_old, y_old, ONE_OVER_E)
    l_new = res.length("radial", ONE_OVER_E)
    fig.suptitle(
        f"The crop-before-shift defect on a 64^3 smoothed-noise volume: 1/e length {l_old:.3f} (scripts) "
        f"against {l_new:.3f} (corrected)",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "With a 64-voxel edge the full correlation needs 127 points; the fast FFT length is 128. Cutting the first 127 "
        "samples before the shift drops lag -1 and keeps a zero gap,\nso the negative half is shifted by one voxel; "
        "the shells at small radii then mix wrong lags (both curves use the same shell binning). "
        "Sizes whose fast length equals 2N-1 (32, 13, ...) were unaffected.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    pdf.savefig(fig)
    plt.close(fig)
    return l_old, l_new


def page_estimators(pdf):
    radius, phi = 6.0, 0.3
    vol, _ = boolean_spheres((160, 160, 160), radius, phi, seed=5)
    sizes = (20, 24, 32, 48, 64, 96, 160)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for t, marker in ((ONE_OVER_E, "o"), (0.1, "s")):
        truth = analytic_length(radius, phi, t)
        for est, color in (("overlap", "C0"), ("window", "C3")):
            vals = []
            for n in sizes:
                r = analyse_texture(vol[:n, :n, :n], max_lag=11, estimator=est, min_overlap=0.2)
                v = r.length("radial", t)
                vals.append(np.nan if v is None else v - truth)
            ax.plot(sizes, vals, marker=marker, color=color, ls="-" if t > 0.2 else "--", label=f"{est}, threshold {t:.2f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("sub-volume edge [voxel]")
    ax.set_ylabel("measured - analytic length [voxel]")
    ax.set_title("Finite-window estimator against overlap-corrected estimator, Boolean spheres R = 6, fraction 0.3")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_directional(pdf):
    rng = np.random.default_rng(7)
    aniso = gaussian_filter(rng.normal(size=(64, 64, 64)), sigma=(4.0, 1.0, 1.0))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, spacing, title in zip(axes, (1.0, (1.0, 1.0, 2.5)), ("isotropic voxels", "voxel spacing (1, 1, 2.5)")):
        res = analyse_texture(aniso, spacing=spacing, max_lag=20)
        for axis, style in (("x", "-"), ("y", "--"), ("z", ":"), ("radial", "-.")):
            p = res.profiles[axis]
            L = res.physical_lengths[axis][ONE_OVER_E].value
            ax.plot(p.distance, p.mean, style, lw=1.5, label=f"{axis}: 1/e at {L:.2f}" if L else axis)
        ax.axhline(ONE_OVER_E, color="gray", lw=0.6)
        ax.set_title(f"Gaussian texture, sigma (z, y, x) = (4, 1, 1), {title}", fontsize=10)
        ax.set_xlabel("physical distance")
        ax.set_ylabel("autocorrelation")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Directional profiles separate structural anisotropy from voxel anisotropy", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def page_sweep(pdf):
    radius, phi = 6.0, 0.3
    vol, _ = boolean_spheres((128, 128, 128), radius, phi, seed=11)
    t0 = time.perf_counter()
    sweep = sweep_sizes(vol, sizes=size_schedule((128, 128, 128), 16, 16, 8), samples_per_size=4, seed=1)
    dt = time.perf_counter() - t0
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for t, color in ((ONE_OVER_E, "C0"), (0.1, "C1")):
        d = sweep.decisions[t]
        ax.errorbar(
            sweep.sizes, sweep.means(t), yerr=sweep.stds(t), marker="o", color=color, capsize=3, label=f"threshold {t:.2f}"
        )
        ax.axhline(analytic_length(radius, phi, t), color=color, lw=0.8, ls=":")
        if d.converged:
            ax.axvline(sweep.sizes[d.start_index], color=color, ls="--", lw=1)
            ax.axhspan(d.reference - d.tolerance, d.reference + d.tolerance, color=color, alpha=0.08)
    ax.set_xlabel("sub-volume edge (geometric mean) [voxel]")
    ax.set_ylabel("correlation length [voxel]")
    ax.set_title(f"Size sweep, Boolean spheres R = 6 in 128^3, four positions per size ({dt:.1f} s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    lines = []
    for t in (ONE_OVER_E, 0.1):
        d = sweep.decisions[t]
        where = f"from edge {sweep.sizes[d.start_index]:.0f}" if d.converged else f"not converged: {d.reason}"
        lines.append(f"threshold {t:.2f}: reference {d.reference:.2f}, band +/- {d.tolerance:.2f} voxel, {where}")
    fig.text(0.08, 0.01, "\n".join(lines), fontsize=8, family="monospace")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)
    return sweep


def page_heuristic(pdf):
    """The subset factor against the displacement error of the pipeline on a Boolean-sphere pair."""
    radius, phi = 5.0, 0.3
    shape = (96, 104, 112)
    ref, _ = boolean_spheres(shape, radius, phi, seed=21)
    ref = gaussian_filter(ref, 0.8) + 0.02 * np.random.default_rng(3).normal(size=shape).astype(np.float32)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    F = np.array([[0.008, 0.002, 0.0], [0.0, -0.005, 0.001], [0.001, 0.0, 0.006]])
    disp = affine_displacement(F, (0.35, -0.25, 0.4), centre)
    dfm = warp_volume_lagrangian(ref, disp)
    res = analyse_texture(ref)
    L = res.length("radial")
    factors = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    rows = []
    for f in factors:
        rec = recommend_parameters(res, factor=f)
        para = dvcpara_default(winsize=rec.subset, winstepsize=8, search_radius=4, verbose=False)
        t0 = time.perf_counter()
        out = run_aldvc(para, [ref, dfm], compute_strain=False)
        dt = time.perf_counter() - t0
        mesh = out.dvc_mesh
        fr = out.result_disp[0]
        interior = ~np.isin(np.arange(mesh.n_nodes), mesh.boundary_nodes)
        err = np.linalg.norm(fr.U - evaluate_at_nodes(disp, mesh.coordinates), axis=1)
        ok = interior & np.isfinite(err)
        rows.append((f, rec.subset[0], float(np.sqrt(np.mean(err[ok] ** 2))), float(np.mean(fr.status == 0)), dt))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-")
    for r in rows:
        ax.annotate(f"{r[1]}", (r[0], r[2]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("subset factor (subset edge / 1/e length)")
    ax.set_ylabel("displacement RMSE [voxel]")
    ax.set_title(f"Subset size against error on a Boolean-sphere pair (1/e length {L:.2f} voxel, labels: subset edge)")
    ax.grid(alpha=0.3)
    fig.text(
        0.08,
        0.01,
        "\n".join(
            f"factor {r[0]:.1f}: subset {r[1]:3d}  RMSE {r[2]:.4f}  converged {100 * r[3]:.0f} %  {r[4]:.1f} s" for r in rows
        ),
        fontsize=8,
        family="monospace",
    )
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)
    return rows


def page_timings_and_limits(pdf, rows_valid, crop, heuristic):
    timings = []
    for n in (64, 128, 192, 256):
        vol = gaussian_filter(np.random.default_rng(n).normal(size=(n, n, n)).astype(np.float32), 1.5)
        t0 = time.perf_counter()
        analyse_texture(vol)
        timings.append((n, time.perf_counter() - t0))
    fig = plt.figure(figsize=(11, 8.5))
    best = min(heuristic, key=lambda r: r[2])
    text = (
        "Timings (analyse_texture, default lags N/2, overlap estimator, one thread of FFT work per axis):\n"
        + "\n".join(f"  {n}^3: {t:.1f} s" for n, t in timings)
        + "\n\nLimitations\n\n"
        "- The correlation length describes the grey-value texture of the image: it changes with blur, noise and\n"
        "  contrast, and it is not a particle size or a material length without a model of the image formation.\n"
        "- The size sweep measures the stability of that statistic against the sampled volume; a material RVE\n"
        "  needs the property of interest and its own convergence study.\n"
        "- The overlap estimator removes the finite-window factor but not every finite-sample effect (the mean and\n"
        "  the variance are estimated from the same voxels); lags with less than half the region overlapping\n"
        "  are not reported by default.\n"
        "- The shell statistics carry the actual mean radius of every shell; a bin still spans one voxel, so\n"
        "  lengths below about two voxels are resolution limited.\n"
        f"- The subset heuristic (factor {best[0]:.1f} gave the lowest error here, {best[2]:.4f} voxel) is a starting\n"
        "  point: the error also depends on noise, on the deformation and on the step; page 6 shows one texture.\n"
        f"- The crop-before-shift defect of the DVC Challenge scripts changed the 1/e length of page 2 from\n"
        f"  {crop[1]:.3f} to {crop[0]:.3f} voxel; their published numbers depend on the volume sizes they used.\n"
        "- Validation: analytic errors on page 1 were "
        + ", ".join(f"{100 * (m - a) / a:+.1f} %" for _r, _p, _t, a, m, _f in rows_valid)
        + "."
    )
    fig.text(0.06, 0.95, text, fontsize=9, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "texture.pdf"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        rows_valid = page_validation(pdf)
        crop = page_crop_bug(pdf)
        page_estimators(pdf)
        page_directional(pdf)
        page_sweep(pdf)
        heuristic = page_heuristic(pdf)
        page_timings_and_limits(pdf, rows_valid, crop, heuristic)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
