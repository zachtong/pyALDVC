"""Visual report of the statistics of results (al_dvc.analysis and the Analysis tab).

    python scripts/make_statistics_report.py [--out reports/statistics.pdf] [--quick]

Pages:

1. the definitions, and a noise floor in the DVC Challenge 2.0 layout: a synthetic static pair (8-bit speckle
   plus 5 % Gaussian noise, as the paper's synthetic second volumes) run through the pipeline;
2. rigid-body motion: rotation recovered against the angle and the noise, and the residual after removal;
3. what a rotation does to strain: the spurious infinitesimal strain ``cos(theta) - 1`` before and after the
   rigid motion is removed, and the Green-Lagrange strain that never changed;
4. the mean strain of a region: the mean of nodal strains against the affine fit, under noise -- the noise bias
   of a nonlinear strain;
5. the Analysis tab (offscreen capture);
6. regions, and the rigid motion fitted over one of them (a grip) and removed everywhere;
7. the confidence interval of a mean: coverage with the effective sample size against the naive one;
8. profiles, the line and the virtual extensometer (a stretch under a growing rotation, a hole in the region);
9. performance against the node count;
10. the Statistics tab's Regions and Line pages and the main window's badge (offscreen captures);
11. boundary conditions and limitations.

Pages 6-10 are in ``statistics_report_phase2.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from scipy.spatial.transform import Rotation  # noqa: E402

from al_dvc import __version__  # noqa: E402
from al_dvc.analysis import fit_motion, frame_view, homogeneous, noise_floor, summarize  # noqa: E402
from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.synthetic import generate_speckle_volume  # noqa: E402

SHAPE_FULL = (96, 96, 96)
SHAPE_QUICK = (56, 56, 56)


def _text_page(pdf, title: str, lines: list[str], size: float = 8.5) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.95, title, fontsize=15, weight="bold", va="top")
    fig.text(0.05, 0.90, "\n".join(lines), fontsize=size, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def static_pair(shape, noise_fraction=0.05, seed=11):
    """8-bit speckle and the same volume with Gaussian noise of ``noise_fraction`` x 255 (DVC Challenge 2.0 Eq. 3)."""
    ref = generate_speckle_volume(shape, sigma=2.0, seed=seed)
    ref8 = np.round(255.0 * (ref - ref.min()) / max(float(np.ptp(ref)), 1e-12))
    rng = np.random.default_rng(seed + 1)
    noisy = np.clip(np.round(ref8 + rng.normal(0.0, noise_fraction * 255.0, ref8.shape)), 0, 255)
    return ref8.astype(np.float32), noisy.astype(np.float32)


def page_noise_floor(pdf, shape) -> dict:
    f, g = static_pair(shape)
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=4, verbose=False)
    t0 = time.perf_counter()
    res = run_aldvc(para, [f, g])
    run_s = time.perf_counter() - t0
    nf = noise_floor(res, 0)
    view = frame_view(res, 0)
    m = view.selection.mask
    U = view.displacement()[m]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2))
    for k, (ax, name) in enumerate(zip(axes, ("u", "v", "w"))):
        ax.hist(U[:, k], bins=40, color="#4c72b0")
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        ax.set_title(f"{name}: mean {nf.displacement.bias[k]:.4f}, std {nf.displacement.noise[k]:.4f} vx", fontsize=8)
        ax.set_xlabel("measured displacement [voxel]", fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle("Static pair, 5 % noise: per-component histograms (DVC Challenge 2.0 Fig. 2 layout)", fontsize=9)
    fig.tight_layout()
    lines = [
        f"pyALDVC {__version__} -- statistics of results (al_dvc.analysis, Analysis tab, al-dvc stats)",
        "",
        "Definitions (every export repeats them):",
        "  nodes used   node_valid and converged and not rejected by the median test (defaults);",
        "               options: ZNCC floor, k edge layers, cut subsets; each removal is counted",
        "  std          population, divided by N (DVC Challenge 2.0 Eq. 2)",
        "  robust std   1.4826 x median absolute deviation",
        "  bias         mean(u - u_nominal) per component (Eq. 1); noise floor = its std (Eq. 2),",
        "               i.e. after removing the rigid translation only; u_rms = sqrt(|bias|^2+|noise|^2)",
        "  MAER / SDER  mean / std over nodes of (1/6) sum |eps_c| (six tensor components)",
        "  VSG          (window - 1) x step + subset span (iDICs GPG Eq. 7.3)",
        "  rigid        weighted Kabsch fit in physical units, removed through the inverse motion;",
        "               strain recomputed from R^T (I + H) - I (exact: the plane fit is linear)",
        "",
        f"Worked example: {shape[2]} x {shape[1]} x {shape[0]} speckle, second volume = the same + N(0, 0.05 x 255),",
        f"subset 17, step 8; run {run_s:.1f} s, {nf.displacement.n} nodes used.",
        "",
        f"  bias  x, y, z [vx]   {'  '.join(f'{v:+.4f}' for v in nf.displacement.bias)}",
        f"  noise x, y, z [vx]   {'  '.join(f'{v:.4f}' for v in nf.displacement.noise)}",
        f"  u_rms [vx]           {nf.displacement.u_rms:.4f}",
        f"  after a rigid fit    {'  '.join(f'{v:.4f}' for v in nf.rigid.noise)}",
        f"  rigid rotation       {nf.rigid_fit.rotation_deg:.4f} deg between the two volumes",
    ]
    if nf.strain is not None:
        lines += [
            f"  strain std exx..eyz  {'  '.join(f'{s.std:.2e}' for s in nf.strain.values())}",
            f"  MAER / SDER          {nf.maer:.2e} / {nf.sder:.2e}",
            f"  VSG [vx]             {', '.join(f'{v:g}' for v in nf.vsg)}",
        ]
    _text_page_with_figure(pdf, "Statistics of results: definitions and a noise floor", lines, fig)
    return {"noise": nf.displacement.noise, "result": res}


def _text_page_with_figure(pdf, title, lines, fig_small) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.95, title, fontsize=15, weight="bold", va="top")
    fig.text(0.05, 0.90, "\n".join(lines), fontsize=8.2, va="top", family="monospace")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        name = fh.name
    fig_small.savefig(name, dpi=130)
    plt.close(fig_small)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.33])
    ax.imshow(mpimg.imread(name))
    ax.axis("off")
    os.unlink(name)
    pdf.savefig(fig)
    plt.close(fig)


def _grid_points(n_side=12, step=8.0):
    ax = step * np.arange(n_side)
    Z, Y, X = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def page_rigid(pdf, quick: bool) -> None:
    X = _grid_points(8 if quick else 12)
    c = X.mean(axis=0)
    angles = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 30])
    noises = (0.0, 0.02, 0.1)
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for s in noises:
        err_angle, resid = [], []
        for a in angles:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            R = Rotation.from_rotvec(np.radians(a) * axis).as_matrix()
            U = (X - c) @ R.T + c + np.array([3.0, -1.0, 2.0]) - X + rng.normal(0, s, X.shape)
            fit = fit_motion(X, U, "rigid")
            err_angle.append(abs(fit.rotation_deg - a))
            resid.append(fit.residual_rms)
        axes[0].semilogy(angles, np.maximum(err_angle, 1e-16), "o-", label=f"noise {s} vx")
        axes[1].plot(angles, resid, "o-", label=f"noise {s} vx")
    axes[0].set_xlabel("imposed rotation [deg]")
    axes[0].set_ylabel("|fitted - imposed| [deg]")
    axes[0].set_title("rotation recovered (exact Kabsch, no linearisation)", fontsize=9)
    axes[1].set_xlabel("imposed rotation [deg]")
    axes[1].set_ylabel("residual RMS after removal [vx]")
    axes[1].set_title("what is left is the noise, at any angle", fontsize=9)
    for ax in axes:
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_rotation_strain(pdf) -> None:
    from tests.test_analysis import make_result, rigid_in_physical

    angles = np.array([0.5, 1, 2, 3, 5, 8, 10])
    raw_inf, fixed_inf, raw_gl, fixed_gl = [], [], [], []
    for a in angles:
        fn, _R = rigid_in_physical([0.0, 0.0, a], [2.0, 1.0, 0.0], (60.0, 60.0, 70.0), (1.0, 1.0, 1.0))
        for measure, raw_list, fix_list in (("infinitesimal", raw_inf, fixed_inf), ("green_lagrange", raw_gl, fixed_gl)):
            res = make_result([fn], strain_type=measure)
            raw_list.append(float(np.nanmedian(frame_view(res, 0).values("eyy"))))
            fix_list.append(float(np.nanmedian(frame_view(res, 0, "rigid").values("eyy"))))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(angles, raw_inf, "o-", label="infinitesimal eyy, as measured")
    ax.plot(angles, np.cos(np.radians(angles)) - 1.0, "k:", label="cos(theta) - 1")
    ax.plot(angles, fixed_inf, "s-", label="infinitesimal eyy, rigid motion removed")
    ax.plot(angles, raw_gl, "^-", label="Green-Lagrange eyy, as measured (objective)")
    ax.axhspan(-3e-4, 3e-4, color="0.85", label="typical DVC strain noise floor (3e-4)")
    ax.set_xlabel("rigid rotation about z [deg] (no deformation)")
    ax.set_ylabel("median eyy")
    ax.set_title("A rigid rotation reads as strain in the infinitesimal measure until it is removed", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_mean_strain(pdf, quick: bool) -> None:
    """(a) an undeformed region: the mean of a nonlinear quantity (von Mises) grows with the noise, the affine fit
    stays at zero; (b) a 1 % stretch: the spread over repeated noise of the region's mean exx, nodal mean against
    the affine fit."""
    from al_dvc.strain.strain_types import von_mises_strain
    from tests.test_analysis import make_result

    c = np.array([60.0, 60.0, 70.0])
    n_axis = (9, 10, 11)
    noise_levels = np.array([0.0, 0.05, 0.1, 0.2, 0.4])
    trials = 4 if quick else 12
    rng = np.random.default_rng(5)
    vm_nodal, vm_fit, spread_nodal, spread_fit = [], [], [], []
    for s in noise_levels:
        a_vm_n, a_vm_f, a_n, a_f = [], [], [], []
        for _ in range(trials):
            for H, sink in ((np.zeros((3, 3)), "vm"), (np.diag([0.01, 0.0, 0.0]), "mean")):
                noise = rng.normal(0, s, size=(int(np.prod(n_axis)), 3))

                def fn(x, y, z, noise=noise, H=H):
                    u = (np.column_stack([x, y, z]) - c) @ H.T + noise
                    return u[:, 0], u[:, 1], u[:, 2]

                res = make_result([fn], n_axis=n_axis)
                hom = homogeneous(res, 0)
                if sink == "vm":
                    a_vm_n.append(summarize(frame_view(res, 0).values("von_mises")).mean)
                    a_vm_f.append(float(von_mises_strain(hom.infinitesimal[None])[0]))
                else:
                    a_n.append(summarize(frame_view(res, 0).values("exx")).mean)
                    a_f.append(hom.infinitesimal[0, 0])
        vm_nodal.append(np.mean(a_vm_n))
        vm_fit.append(np.mean(a_vm_f))
        spread_nodal.append(np.std(a_n))
        spread_fit.append(np.std(a_f))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].plot(noise_levels, vm_nodal, "o-", label="mean of nodal von Mises strain")
    axes[0].plot(noise_levels, vm_fit, "s-", label="von Mises of the affine-fit strain")
    axes[0].set_title("(a) undeformed: a nonlinear mean is biased by noise", fontsize=9)
    axes[0].set_ylabel("von Mises strain")
    axes[1].plot(noise_levels, spread_nodal, "o-", label="mean of nodal exx")
    axes[1].plot(noise_levels, spread_fit, "s-", label="affine fit exx")
    axes[1].set_title("(b) 1 % stretch: a linear component -- both scatter alike", fontsize=9)
    axes[1].set_ylabel(f"std over {trials} noise draws")
    for ax in axes:
        ax.set_xlabel("displacement noise [voxel]")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_screenshot(pdf, result) -> None:
    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow, create_application

    create_application([sys.argv[0]])
    window = MainWindow()
    window.state.set_results(result)
    post = window.open_statistics()
    post.resize(1400, 900)
    tab = post.analysis
    idx = tab.motion.findData("rigid")
    tab.motion.setCurrentIndex(idx)
    tab.wait(120_000)
    tab.results_tabs.setCurrentIndex(1)
    for _ in range(20):
        QApplication.processEvents()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        name = fh.name
    post.grab().save(name)
    shot = mpimg.imread(name)
    os.unlink(name)
    window.settle_workers(10_000)
    window.state.dirty = False
    post.close()
    window.close()
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.imshow(shot)
    ax.axis("off")
    ax.set_title("Analysis tab: rigid motion removed, histograms of u, v, w (offscreen capture)", fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


LIMITS = [
    "Boundary conditions:",
    "  * a region too small for the motion (fewer than 3 nodes for rigid, 4 for affine) reports",
    "    'too few nodes' instead of an uncorrected field; an empty selection gives '-', not a warning;",
    "  * incremental tracking: a node counts only if every pair of its chain converged and was not",
    "    rejected; the displacement uncertainty (U_std) is not rotated with a rigid correction;",
    "  * anisotropic voxels: every fit is made in physical units (tested with 1 x 1 x 2.5 voxels);",
    "  * a constant field (pure translation) has a histogram with a single bin around its value;",
    "  * a region with no node inside is named in the Regions list; its statistics are '-';",
    "  * a drawn outline goes through the whole node grid along the plane's normal (its depth is",
    "    editable); regions live in the reference configuration, in voxels, bounds included;",
    "  * a frame whose motion cannot be fitted is left as measured in the main window and the",
    "    exports and listed in <basename>_correction.json ('uncorrected_frames').",
    "",
    "Limitations (docs/plans/2026-09-23-result-statistics.md):",
    "  * the confidence interval assumes a stationary field over the region; a region only a",
    "    few correlation lengths wide gives a noisy estimate (coverage a few points low);",
    "  * the extensometer and the line need valid nodes around their samples; a gap is NaN,",
    "    never an extrapolated value;",
    "  * the series' gap threshold counts the region's nodes, including those outside the",
    "    region of interest; it is off (0 %) by default;",
    "  * no imported load or time axis: series are plotted against the frame number;",
    "  * affine removal changes the displacement only; the fitted homogeneous strain is reported",
    "    beside the mean of the nodal strains (Homogeneous tab);",
    "  * robust (IRLS/RANSAC) and uncertainty-weighted fits, anchors on several regions and",
    "    per-region rotations are phase 3.",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "statistics.pdf"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    shape = SHAPE_QUICK if args.quick else SHAPE_FULL
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        info = page_noise_floor(pdf, shape)
        page_rigid(pdf, args.quick)
        page_rotation_strain(pdf)
        page_mean_strain(pdf, args.quick)
        from tests.test_analysis import make_result, rigid_in_physical

        rigid, _R = rigid_in_physical([2.0, -1.0, 6.0], [4.0, -2.0, 7.0], (60.0, 60.0, 70.0), (1.0, 1.0, 1.0))
        rng = np.random.default_rng(9)

        def deformed(x, y, z):  # a rigid motion on top of a stretch and a smooth bump, plus noise
            u, v, w = rigid(x, y, z)
            return (
                u + 0.01 * (x - 60.0) + 0.3 * np.sin(y / 12.0) + rng.normal(0, 0.03, x.shape),
                v + rng.normal(0, 0.03, x.shape),
                w + 0.2 * np.cos(x / 15.0) + rng.normal(0, 0.03, x.shape),
            )

        page_screenshot(pdf, make_result([deformed, deformed], n_axis=(14, 14, 16)))
        import statistics_report_phase2 as p2

        p2.page_regions(pdf, make_result, rigid_in_physical)
        p2.page_confidence(pdf, args.quick)
        p2.page_line(pdf, make_result, rigid_in_physical)
        p2.page_performance(pdf, make_result, args.quick)
        p2.page_screens(pdf, make_result, rigid_in_physical)
        _text_page(pdf, "Boundary conditions and limitations", LIMITS)
        del info
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
