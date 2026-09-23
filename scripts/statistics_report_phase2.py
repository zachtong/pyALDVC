"""Phase-2 pages of ``reports/statistics.pdf`` (called by ``make_statistics_report.py``).

* regions and the motion fitted over a region: a grip that only moves, a gauge section that stretches;
* the confidence interval of a mean: coverage against the correlation length, with the effective sample size and
  with the naive ``std / sqrt(N)``;
* profiles, the line and the virtual extensometer: a stretch under a growing rigid rotation, and a hole in the
  region (a line never invents values);
* performance: effective sample size, region masks and the corrected result against the node count;
* the Statistics tab (offscreen captures of the Regions, Profile and Line pages and the main window's badge).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from al_dvc.analysis import (
    Correction,
    NodeFilter,
    Region,
    axis_profile,
    corrected_result,
    effective_sample_size,
    extensometer,
    frame_view,
    mean_confidence,
    sample_line,
)

C_RAW, C_ALL, C_GRIP, C_OK = "#64748b", "#f59e0b", "#0ea5e9", "#16a34a"


def _grip_result(make_result, rigid_in_physical, n_axis=(14, 10, 12)):
    """A bar along x: the grip (x <= 44) moves rigidly (translation + 3 deg about z), the gauge section stretches by
    1 % from the grip's end, plus 0.02-voxel noise."""
    rigid, _R = rigid_in_physical([0.0, 0.0, 3.0], [1.5, -0.8, 0.4], (40.0, 60.0, 70.0), (1.0, 1.0, 1.0))
    rng = np.random.default_rng(5)

    def fn(x, y, z):
        u, v, w = rigid(x, y, z)
        return (u + 0.01 * np.maximum(x - 44.0, 0.0) + rng.normal(0, 0.02, x.shape), v + rng.normal(0, 0.02, x.shape), w)

    return make_result([fn], n_axis=n_axis)


def page_regions(pdf, make_result, rigid_in_physical) -> None:
    from al_dvc.gui.analysis_regions import region_cut

    res = _grip_result(make_result, rigid_in_physical)
    grip = Region(1, "grip", "box", {"lo": [0, 0, 0], "hi": [44, 200, 200]}, color=C_GRIP)
    gauge = Region(2, "gauge", "box", {"lo": [60, 30, 40], "hi": [108, 90, 100]}, color=C_OK)
    ball = Region(3, "ball", "sphere", {"centre": [84, 60, 70], "radius": 18.0}, color="#c026d3")
    grip_mask = grip.node_mask(res)
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Regions, and the rigid motion fitted over one of them", fontsize=13, weight="bold", y=0.97)
    # the field on a slice with the regions outlined, as on the tab's canvas
    ax = fig.add_axes([0.07, 0.62, 0.40, 0.28])
    mesh = res.dvc_mesh
    view_grip = frame_view(res, 0, "rigid", fit_region=grip_mask)
    grid = mesh.to_grid(view_grip.values("disp_u"))
    kz = grid.shape[0] // 2
    x0, y0 = mesh.x0, mesh.y0
    hx, hy = mesh.spacing[0], mesh.spacing[1]
    extent = [x0[0] - hx / 2, x0[-1] + hx / 2, y0[0] - hy / 2, y0[-1] + hy / 2]
    im = ax.imshow(grid[kz], origin="lower", cmap="turbo", extent=extent)
    for r in (grip, gauge, ball):  # the outlines the tab draws on its canvas
        a, b, inside = region_cut(r, "xy", float(mesh.z0[kz]), res.volume_shape)
        if inside.any():
            ax.contour(a, b, inside.astype(float), levels=[0.5], colors=[r.color], linewidths=2.0 if r is gauge else 1.2)
            rr, cc = np.nonzero(inside)
            top = rr.max()
            ax.text(a[cc[rr == top].min()], b[top], " " + r.name, color=r.color, fontsize=7, va="bottom")
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 110)
    ax.set_title("u, rigid motion of the grip removed (z = %d)" % int(mesh.z0[kz]), fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    # the layer means along x: as measured, rigid removed over every node, over the grip
    bx = fig.add_axes([0.58, 0.62, 0.38, 0.28])
    for label, color, view in (
        ("as measured", C_RAW, frame_view(res, 0, "none")),
        ("rigid fitted over every node", C_ALL, frame_view(res, 0, "rigid")),
        ("rigid fitted over the grip", C_GRIP, view_grip),
    ):
        prof = axis_profile(view, "disp_u", "x")
        bx.plot(prof.positions, prof.mean, "o-", ms=3, color=color, label=label)
    xs = np.linspace(20, 124, 100)
    bx.plot(xs, 0.01 * np.maximum(xs - 44.0, 0.0), "k--", lw=0.8, label="imposed stretch")
    bx.axvspan(0, 44, color=C_GRIP, alpha=0.08)
    bx.set_xlabel("x [voxel]", fontsize=8)
    bx.set_ylabel("mean u of each node layer [voxel]", fontsize=8)
    bx.legend(fontsize=7)
    bx.tick_params(labelsize=7)
    # the regions compared, mean with its 95 % CI
    cx = fig.add_axes([0.07, 0.30, 0.40, 0.24])
    rows = []
    values = view_grip.values("exx")
    kept = view_grip.selection.mask
    for r in (grip, gauge, ball):
        m = kept & r.node_mask(res)
        n_eff, half = mean_confidence(mesh.to_grid(values), mesh.to_grid(m))
        rows.append((r.name, r.color, float(np.nanmean(values[m])), half, float(np.nanstd(values[m])), int(m.sum()), n_eff))
    for i, (_n, color, mean, half, std, _c, _e) in enumerate(rows):
        cx.bar(i, mean, color=color, alpha=0.55, width=0.6)
        cx.errorbar(i, mean, yerr=std, color="#475569", lw=0.8, capsize=2)
        cx.errorbar(i, mean, yerr=half, color=color, lw=2.2, capsize=5)
    cx.axhline(0.01, color="k", ls="--", lw=0.8)
    cx.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    cx.set_ylabel("exx", fontsize=8)
    cx.set_title("mean exx per region; thick: 95 % CI, thin: std; dashed: imposed 1 %", fontsize=8)
    cx.tick_params(labelsize=7)
    fits = {
        "every node": frame_view(res, 0, "rigid").fit,
        "the grip": view_grip.fit,
    }
    lines = [
        "Setup: a bar along x on a 14 x 10 x 12 node grid; the grip (x <= 44) moves rigidly by t = (1.5, -0.8, 0.4)",
        "voxel and 3 deg about z; the gauge section stretches by 1 % from the grip's end; 0.02-voxel noise on u, v.",
        "",
        "exx in the grip is not 0 near its end: the strain fit (a plane over a window of nodes) reaches",
        "across the kink at x = 44. Choose a region a strain window away from such a boundary.",
        "",
        "Fitting the rigid motion over every node mixes the stretch into the 'rigid' motion: the fit tilts and",
        "translates to absorb part of the stretch, so the grip no longer sits at zero (orange). Fitting over the",
        "grip alone (the region a user would pick: a fixture) removes exactly the grip's motion (blue): the grip",
        "is at rest and the gauge section shows the imposed 1 % stretch. The fit is made once and removed from",
        "the whole field; the statistics can then be taken over any other region.",
        "",
        "Rotation found: %.3f deg over every node, %.3f deg over the grip (imposed 3.000)."
        % (fits["every node"].rotation_deg, fits["the grip"].rotation_deg),
    ]
    table = ["Regions: name, nodes, mean exx,", "95 % CI half width, effective nodes"]
    for name, _c, mean, half, _s, n, n_eff in rows:
        table.append(f"  {name:6s} {n:5d}  {mean: .5f} +- {half:.5f}  {n_eff:7.1f}")
    fig.text(0.53, 0.52, "\n".join(table), fontsize=7.5, family="monospace", va="top")
    fig.text(0.07, 0.24, "\n".join(lines), fontsize=7.8, family="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def page_confidence(pdf, quick: bool) -> None:
    rng = np.random.default_rng(12)
    shape = (20, 20, 20) if quick else (26, 26, 26)
    mask = np.ones(shape, dtype=bool)
    mask[:, :, : shape[2] // 5] = False  # an irregular region
    sigmas = (0.0, 1.0, 2.0, 3.0)
    trials = 50 if quick else 150
    cover, naive, ratio, exact = [], [], [], []
    n = int(mask.sum())
    s2 = tuple(2 * k for k in shape)
    fm = np.fft.rfftn(mask.astype(float), s=s2)
    pairs = np.rint(np.fft.irfftn(fm * np.conj(fm), s=s2))
    lag2 = [np.where(np.arange(k) < k // 2, np.arange(k), np.arange(k) - k).astype(float) ** 2 for k in s2]
    r2 = lag2[0][:, None, None] + lag2[1][None, :, None] + lag2[2][None, None, :]
    for s in sigmas:
        hits = nhits = 0
        neffs = []
        for _ in range(trials):
            f = rng.normal(size=shape)
            if s > 0:
                f = gaussian_filter(f, s, mode="wrap")
            f = 2.0 + f / f.std()
            n_eff, half = mean_confidence(f, mask)
            m = f[mask].mean()
            hits += abs(m - 2.0) <= half
            nhits += abs(m - 2.0) <= 1.96 * f[mask].std() / np.sqrt(mask.sum())
            neffs.append(n_eff / mask.sum())
        cover.append(hits / trials)
        naive.append(nhits / trials)
        ratio.append(float(np.median(neffs)))
        # exact n_eff of Gaussian-smoothed noise: rho(r) = exp(-r^2 / (4 sigma^2)) (white noise: n)
        exact.append(1.0 if s == 0 else float(n / np.sum(pairs * np.exp(-r2 / (4 * s * s)))))
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("The confidence interval of a mean: correlated nodes", fontsize=13, weight="bold", y=0.97)
    ax = fig.add_axes([0.1, 0.60, 0.38, 0.28])
    ax.plot(sigmas, cover, "o-", color=C_OK, label="effective sample size (pyALDVC)")
    ax.plot(sigmas, naive, "s-", color="#dc2626", label="naive std / sqrt(N)")
    ax.axhspan(0.93, 0.97, color=C_OK, alpha=0.12)
    ax.axhline(0.95, color="k", lw=0.6, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("correlation length (Gaussian sigma, nodes)", fontsize=8)
    ax.set_ylabel("share of intervals covering the true mean", fontsize=8)
    ax.legend(fontsize=7, loc="lower left")
    ax.tick_params(labelsize=7)
    bx = fig.add_axes([0.58, 0.60, 0.36, 0.28])
    bx.semilogy(sigmas, ratio, "o-", color=C_GRIP, label="estimated (median)")
    bx.semilogy(sigmas, exact, "k--", lw=0.9, label="exact")
    bx.set_xlabel("correlation length (Gaussian sigma, nodes)", fontsize=8)
    bx.set_ylabel("n_eff / N", fontsize=8)
    bx.legend(fontsize=7)
    bx.tick_params(labelsize=7)
    lines = [
        f"Monte Carlo: {trials} fields per point on a {shape[0]}^3 node grid, one fifth masked out (an irregular region),",
        "Gaussian noise smoothed with sigma nodes and normalised to unit std; the true mean is 2.",
        "",
        "Node values of a DVC result are correlated: subsets overlap, the global step couples neighbours, the",
        "strain fit and the smoothing average them. N correlated nodes carry the information of n_eff < N",
        "independent ones, and the interval std / sqrt(N) is too narrow by sqrt(N / n_eff).",
        "",
        "First a least-squares plane is removed: a real gradient across the region is the field, not the",
        "uncertainty of its mean (without this a stretched region read 0.03 +- 1.1 voxel). Then",
        "n_eff = N^2 / sum_ij rho_ij, rho the autocorrelation of the residual on the node grid (FFT, divided",
        "by the node pairs at each lag so the region's own shape drops out), summed out to the first lag shell",
        "where it falls below 0.05, with two corrections that matter in 3-D, where most lags are far: removing",
        "the plane pulls every rho down by about 4/n_eff (undone to first order), and the lags beyond the cut",
        "still hold about a quarter of the sum (a tail exp(-(r/L)^q) fitted to the shells). Interval:",
        "t(n_eff - 4) std / sqrt(n_eff - 4). Without the corrections the estimate was a third too high at",
        "sigma = 2 and three times too high at sigma = 3, and the coverage fell to 0.76.",
        "",
        "sigma                 " + "  ".join(f"{s:4.1f}" for s in sigmas),
        "coverage (effective)  " + "  ".join(f"{c:4.2f}" for c in cover),
        "coverage (naive)      " + "  ".join(f"{c:4.2f}" for c in naive),
        "n_eff / N estimated   " + "  ".join(f"{r:.4f}" for r in ratio),
        "n_eff / N exact       " + "  ".join(f"{r:.4f}" for r in exact),
        "",
        "Limits: the estimate assumes a stationary field (the same correlation everywhere in the region). When",
        "the region spans only two or three correlation lengths, n_eff is a few tens and itself uncertain:",
        "the coverage is then 85-90 %, not 95 % -- read the interval as a lower bound of the uncertainty. A",
        "smooth trend (a real strain gradient) counts as correlation and widens the interval: it is the",
        "interval of the region's mean, not of a local value.",
    ]
    fig.text(0.07, 0.52, "\n".join(lines), fontsize=8, family="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def page_line(pdf, make_result, rigid_in_physical) -> None:
    strains = np.linspace(0.0, 0.03, 7)
    angles = np.linspace(0.0, 12.0, 7)  # the specimen also turns about z as it is loaded
    fns = []
    for e, a in zip(strains, angles):
        rigid, _R = rigid_in_physical([0.0, 0.0, a], [0.0, 0.0, 0.0], (60.0, 60.0, 70.0), (1.0, 1.0, 1.0))

        def fn(x, y, z, e=e, rigid=rigid):
            # stretch first (u = e (x - 60)), then the rigid rotation of the stretched body
            xs = x + e * (x - 60.0)
            u, v, w = rigid(xs, y, z)
            return (xs - x + u, v, w)

        fns.append(fn)
    n = 14 * 12 * 12
    valid = np.ones(n, dtype=bool)
    res0 = make_result(fns[:1], n_axis=(14, 12, 12), strain=False)
    X = res0.dvc_mesh.coordinates
    valid[np.linalg.norm(X - [84, 60, 68], axis=1) < 13] = False  # a hole in the region of interest
    res = make_result(fns, n_axis=(14, 12, 12), node_valid=valid, strain=False)
    p0, p1 = (28.0, 60.0, 68.0), (108.0, 60.0, 68.0)
    ext = extensometer(res, p0, p1)
    naive = []
    for k in range(len(fns)):
        v = frame_view(res, k)
        U = v.displacement()
        # a naive reading: the x displacement difference of the nearest nodes at the ends, over L0
        i0 = int(np.argmin(np.linalg.norm(X - p0, axis=1)))
        i1 = int(np.argmin(np.linalg.norm(X - p1, axis=1)))
        naive.append((U[i1, 0] - U[i0, 0]) / (p1[0] - p0[0]))
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Profiles, the line and the virtual extensometer", fontsize=13, weight="bold", y=0.97)
    ax = fig.add_axes([0.1, 0.63, 0.36, 0.26])
    ax.plot(strains * 100, ext.strain * 100, "o-", color=C_OK, label="extensometer L/L0 - 1")
    ax.plot(strains * 100, np.asarray(naive) * 100, "s-", color="#dc2626", label="naive du / L0")
    ax.plot(strains * 100, strains * 100, "k--", lw=0.8, label="imposed")
    ax.set_xlabel("imposed strain [%] (rotation 0 to 12 deg)", fontsize=8)
    ax.set_ylabel("measured strain [%]", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    bx = fig.add_axes([0.58, 0.63, 0.36, 0.26])
    for k in range(len(fns)):
        d, vals = sample_line(frame_view(res, k), "disp_magnitude", p0, p1)
        bx.plot(d, vals, "-", color=plt.cm.viridis(k / (len(fns) - 1)), lw=1.0)
    bx.axvspan(84 - 13 - 28, 84 + 13 - 28, color="#94a3b8", alpha=0.2)
    bx.set_xlabel("distance along the line [voxel]", fontsize=8)
    bx.set_ylabel("|u| along the line, every frame", fontsize=8)
    bx.set_title("grey: the hole in the region -- a gap, not an invented value", fontsize=7.5)
    bx.tick_params(labelsize=7)
    cx = fig.add_axes([0.1, 0.30, 0.36, 0.24])
    for k in (0, 3, 6):
        prof = axis_profile(frame_view(res, k, "rigid"), "disp_u", "x")
        cx.plot(prof.positions, prof.mean, "o-", ms=3, label=f"frame {k + 1}")
    cx.set_xlabel("x [voxel]", fontsize=8)
    cx.set_ylabel("layer mean of u, rigid removed [voxel]", fontsize=8)
    cx.legend(fontsize=7)
    cx.tick_params(labelsize=7)
    err = float(np.nanmax(np.abs(ext.strain - strains)))
    lines = [
        "Setup: 14 x 12 x 12 nodes, 7 frames; the body stretches along x by 0 to 3 % and turns by 0 to 12 deg",
        "about z at the same time; a spherical hole (radius 13 voxels) is cut out of the region of interest.",
        "",
        "Extensometer: the two points move with the trilinearly interpolated displacement; L is their",
        "distance, strain = L / L0 - 1. A length ignores rigid motion, so no correction is needed;",
        f"largest error against the imposed strain: {err:.1e}. The naive difference of the x displacements",
        "over L0 reads the rotation as strain (red).",
        "",
        "Line: the field is sampled trilinearly at two points per node spacing; a sample whose stencil",
        "touches a node without a value is NaN, so the hole is a gap in every frame (grey band).",
        "",
        "Profile: the mean of each node layer along an axis, here u along x with the rigid motion removed;",
        "the slope of each profile is the frame's stretch.",
        "",
        "Limits: an extensometer end must lie inside the node grid with valid neighbours (else that frame",
        "is NaN); the line is sampled in the reference configuration (material points), not along a fixed",
        "line in space.",
    ]
    fig.text(0.53, 0.54, "\n".join(lines), fontsize=7.4, family="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def page_performance(pdf, make_result, quick: bool) -> None:
    sides = (10, 20, 30, 40) if quick else (10, 20, 30, 40, 60, 80)
    t_neff, t_mask, t_corr, nodes = [], [], [], []
    triangle = {"plane": "xy", "outline": "polygon", "points": [[0, 0], [900, 0], [0, 900]], "lo": 0, "hi": 900}
    poly = Region(1, "p", "prism", triangle)
    for s in sides:
        rng = np.random.default_rng(s)
        g = gaussian_filter(rng.normal(size=(s, s, s)), 1.5)
        t0 = time.perf_counter()
        effective_sample_size(g, np.ones(g.shape, dtype=bool))
        t_neff.append(time.perf_counter() - t0)
        res = make_result([lambda x, y, z: (0.01 * x, 0.002 * y, 0 * z)], n_axis=(s, s, s), strain=s <= 40)
        nodes.append(res.dvc_mesh.n_nodes)
        t0 = time.perf_counter()
        poly.node_mask(res)
        t_mask.append(time.perf_counter() - t0)
        shown = corrected_result(res, Correction("rigid", NodeFilter()))
        t0 = time.perf_counter()
        shown.result_disp[0]
        t_corr.append(time.perf_counter() - t0)
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Performance (this machine, one thread per job)", fontsize=13, weight="bold", y=0.97)
    ax = fig.add_axes([0.12, 0.55, 0.8, 0.33])
    ax.loglog(nodes, t_neff, "o-", label="effective sample size (one field)")
    ax.loglog(nodes, t_mask, "s-", label="polygon region: which nodes are inside")
    ax.loglog(nodes, t_corr, "^-", label="corrected frame (rigid fit + removal)")
    ax.set_xlabel("nodes", fontsize=8)
    ax.set_ylabel("seconds", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)
    lines = ["nodes      n_eff [s]   region [s]   corrected frame [s]"]
    for n, a, b, c in zip(nodes, t_neff, t_mask, t_corr):
        lines.append(f"{n:9d}   {a:9.3f}   {b:10.3f}   {c:14.3f}")
    lines += [
        "",
        "The current frame's statistics are recomputed after every change of the controls, with the confidence",
        "interval of each field: at a million nodes that is about a second per field, on a worker thread (the",
        "window stays responsive; a newer request waits for the running one). Series over frames, profiles over",
        "frames and the line over frames run on request, with progress and Cancel.",
        "",
        "The corrected result the main window shows is computed frame by frame on first use and kept; the",
        "strain of a rigid correction is recomputed from the corrected gradient (exact for the plane fit).",
    ]
    fig.text(0.07, 0.47, "\n".join(lines), fontsize=8, family="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _grab(widget) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        name = fh.name
    widget.grab().save(name)
    img = mpimg.imread(name)
    os.unlink(name)
    return img


def page_screens(pdf, make_result, rigid_in_physical) -> None:
    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow, create_application

    create_application([sys.argv[0]])
    window = MainWindow()
    window.state.set_results(_grip_result(make_result, rigid_in_physical, n_axis=(16, 12, 14)))
    post = window.open_statistics()
    post.resize(1400, 950)
    tab = post.analysis
    grip = tab.regions_panel.add("box")
    tab.regions_panel._edits["lo0"].setValue(0.0)
    tab.regions_panel._edits["hi0"].setValue(44.0)
    for i in (1, 2):
        tab.regions_panel._edits[f"lo{i}"].setValue(0.0)
        tab.regions_panel._edits[f"hi{i}"].setValue(200.0)
    tab.regions_panel.apply_editor()
    tab.regions_panel.add("sphere")
    tab.motion.setCurrentIndex(tab.motion.findData("rigid"))
    tab.fit_region_combo.setCurrentIndex(tab.fit_region_combo.findData(grip.id))
    tab.shown.setCurrentIndex(tab.shown.findData("disp_u"))
    tab.apply_main.setChecked(True)
    tab.wait(120_000)
    shots = []
    for index, title in ((3, "Regions: the field on the slices with the regions outlined, the comparison below"), (5, "Line")):
        tab.results_tabs.setCurrentIndex(index)
        if index == 5:
            tab.compute_line()
        tab.wait(120_000)
        for _ in range(20):
            QApplication.processEvents()
        shots.append((title, _grab(post)))
    window.results_panel.resize(360, 620)
    for _ in range(10):
        QApplication.processEvents()
    main = _grab(window.results_panel)
    window.settle_workers(10_000)
    window.state.dirty = False
    post.close()
    window.close()
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("The Statistics tab and the main window (offscreen captures)", fontsize=12, weight="bold", y=0.985)
    for k, (title, img) in enumerate(shots):
        ax = fig.add_axes([0.03, 0.655 - 0.3 * k, 0.94, 0.28])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=8)
    ax = fig.add_axes([0.03, 0.01, 0.34, 0.31])
    ax.imshow(main)
    ax.axis("off")
    ax.set_title("Results panel: the badge and 'As measured'", fontsize=8)
    fig.text(
        0.42,
        0.28,
        "The switch 'Also in the main window and\n"
        "exports' carries the correction to the slices,\n"
        "the 3-D view and the CSV, ParaView, image and\n"
        "report exports. The badge says what was\n"
        "removed and where it was fitted; 'As measured'\n"
        "turns it off. The npz and mat archives always\n"
        "hold the result as measured, and\n"
        "<basename>_correction.json is written next to\n"
        "corrected exports.",
        fontsize=8,
        family="monospace",
        va="top",
    )
    pdf.savefig(fig)
    plt.close(fig)
