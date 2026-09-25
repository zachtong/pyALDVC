#!/usr/bin/env python
"""Figures of the project website's "How it works" section, computed with pyALDVC.

Every figure comes from synthetic volumes with a known deformation, run through the public ``al_dvc`` API
on the Numba CPU backend (bit-reproducible; the GPU agrees to about 1e-5 voxel but not bit for bit) with
fixed seeds. Dark theme in the application's colours (``gui/theme.py``). The numbers the captions quote
are written to ``numbers.json`` next to the figures, so the page copy is regenerated, never hand-edited.
Run from anywhere::

    python scripts/make_site_figures.py [--out site/figures] [--only NAME ...]

Figures (PNG, about 1800 px wide: display them at half their pixel width)

    aldvc_local_vs_global.png   du/dx of the local subsets alone, of AL-DVC and the imposed one
    tracking_modes.png          accumulative against incremental tracking of a growing rotation
    noise_floor.png             a static pair, DVC Challenge 2.0 definitions: bias, noise floor, CI of the mean
    uncertainty_map.png         the per-node U_std of a run against its actual error where the texture fades
    rigid_removal.png           a 4 degree rotation on top of 0.4 % compression, before and after removal
    texture_subset.png          subset size against the correlation length of the texture

Texture-guide animations (``guide``): the in-app guide of ``scripts/make_guide_animations.py`` at twice its
resolution on the site's background, in the site figures' font, with its one-line notes in the text colour
(the amber stays on the thresholds it marks); the GIFs converted to H.264 MP4 plus a poster frame when that
is smaller (``guide_*.mp4`` / ``guide_*.png``). The conversion needs ``imageio-ffmpeg``; without it the GIFs
are kept. Step 3 (``guide_subset.png``) is drawn here, with the subset and step of ``recommend_parameters()``
(four lengths rounded up to an even edge, half of it rounded up to an even step), as the application
suggests them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats as sps  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__, dvcpara_default, run_aldvc  # noqa: E402
from al_dvc.analysis import NodeFilter, frame_stats, frame_view, noise_floor  # noqa: E402
from al_dvc.core.data_structures import STATUS_CONVERGED  # noqa: E402
from al_dvc.synthetic import (  # noqa: E402
    add_noise,
    affine_displacement,
    evaluate_at_nodes,
    generate_speckle_volume,
    gradient_at_nodes,
    rotation_displacement,
    warp_volume_lagrangian,
)
from al_dvc.texture import analyse_texture, boolean_spheres, recommend_parameters  # noqa: E402

DEFAULT_OUT = ROOT / "site" / "figures"
NUMBERS = "numbers.json"

# --- app colours (src/al_dvc/gui/theme.py DARK; the guide and branding scripts use the same) ---------------
BG = "#141929"  # BG_PANEL: figure face, the colour of a card on the site
AX_BG = "#0b0f1a"  # BG_DARKEST
BORDER = "#1e293b"
GRID = "#243044"
TEXT = "#e2e8f0"
TEXT_2 = "#94a3b8"
MUTED = "#64748b"
ACCENT = "#818cf8"  # pyALDVC / AL-DVC / corrected
PINK = "#f472b6"  # raw / local / as measured (as in the guide animations)
TEAL = "#2dd4bf"  # truth
AMBER = "#fbbf24"  # reference lines and thresholds

WIDTH = 9.0  # inches: the site shows a figure at about 900 px, so 1 pt of text is about 1.4 px there
DPI = 200  # 2x: about 1800 px wide
GUIDE_DPI = 180  # 2x the in-app guide's 90
BACKEND = "numba"
PNG_BUDGET = 150 * 1024  # bytes; larger PNGs are quantised to 256 colours when that helps

log = logging.getLogger("make_site_figures")


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": AX_BG,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "text.color": TEXT,
            "legend.facecolor": BG,
            "legend.edgecolor": BORDER,
            "legend.fontsize": 9,
            "legend.labelcolor": TEXT_2,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.unicode_minus": True,
        }
    )


def sig(x, n: int = 4) -> float | list | None:
    """``x`` (a number or a nested list of numbers) rounded to ``n`` significant digits, for numbers.json."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [sig(v, n) for v in x]
    x = float(x)
    if not np.isfinite(x):
        return None
    return float(f"{x:.{n}g}")


def save(fig, out: Path, name: str) -> dict:
    """Write ``name`` at ``DPI``; quantise to 256 colours when that brings a large PNG under the budget."""
    path = out / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    rgb.save(path, optimize=True)
    quantised = False
    if path.stat().st_size > PNG_BUDGET:
        q = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        tmp = path.with_suffix(".q.png")
        q.save(tmp, optimize=True)
        if tmp.stat().st_size < path.stat().st_size:
            tmp.replace(path)
            quantised = True
        else:
            tmp.unlink()
    size = path.stat().st_size
    print(f"  wrote {name}: {rgb.width} x {rgb.height} px, {size / 1024:.0f} kB{' (256 colours)' if quantised else ''}")
    return {"file": name, "width_px": rgb.width, "height_px": rgb.height, "bytes": size}


def colorbar(fig, im, ax, label: str, extend: str = "neither") -> None:
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, extend=extend)
    cb.outline.set_edgecolor(BORDER)
    cb.ax.tick_params(colors=TEXT_2, labelsize=8.5)
    cb.set_label(label, color=TEXT_2, fontsize=9.5)


def note(ax, text: str, loc: str = "lower left") -> None:
    """A boxed annotation inside ``ax``."""
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.04, "bottom") if "lower" in loc else (0.96, "top")
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9.5,
        color=TEXT,
        bbox={"boxstyle": "round,pad=0.3", "fc": BG, "ec": BORDER, "alpha": 0.9},
    )


def layer(mesh, values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """The middle node layer (z) of a per-node array, NaN where invalid, and its image extent in voxels."""
    grid = mesh.to_grid(np.where(valid, values, np.nan))
    k = grid.shape[0] // 2
    x0, y0 = np.asarray(mesh.x0, float), np.asarray(mesh.y0, float)
    hx = (x0[-1] - x0[0]) / max(len(x0) - 1, 1) / 2
    hy = (y0[-1] - y0[0]) / max(len(y0) - 1, 1) / 2
    return grid[k], (x0[0] - hx, x0[-1] + hx, y0[-1] + hy, y0[0] - hy)


def para(**kw) -> object:
    return dvcpara_default(backend=BACKEND, verbose=False, **kw)


# ------------------------------------------------------------------ 1. local subsets vs AL-DVC
ALDVC_SEEDS = ((2, 1, 2), (3, 3, 4), (4, 5, 6))  # (texture, noise of f, noise of g); the first one is drawn


def _aldvc_case(n: int, noise: float, ws: int, seeds: tuple[int, int, int]) -> dict:
    c = np.full(3, (n - 1) / 2)
    amp, s = 1.5, n / 7.0

    def disp(x, y, z):  # a smooth bump: a localised stretch around the centre
        g = np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / (2 * s * s))
        return amp * g * np.tanh((x - c[0]) / s), 0.5 * amp * g, 0.0 * x

    ref = generate_speckle_volume((n, n, n), sigma=2.0, seed=seeds[0])
    dfm = warp_volume_lagrangian(ref, disp)
    p = para(winsize=ws, winstepsize=4, search_radius=4)
    res = run_aldvc(p, [add_noise(ref, noise, seed=seeds[1]), add_noise(dfm, noise, seed=seeds[2])], compute_strain=False)
    mesh, fr = res.dvc_mesh, res.result_disp[0]
    X = mesh.coordinates
    T, G = evaluate_at_nodes(disp, X), gradient_at_nodes(disp, X)
    v = mesh.node_valid
    return {
        "mesh": mesh,
        "fr": fr,
        "G": G,
        "valid": v,
        "snr": float(ref.std()) / noise,
        "rms_dudx": [float(np.sqrt(np.mean((F[:, 0, 0] - G[:, 0, 0])[v] ** 2))) for F in (fr.F_local, fr.F)],
        "rms_F": [float(np.sqrt(np.mean((F - G)[v] ** 2))) for F in (fr.F_local, fr.F)],
        "rms_U": [float(np.sqrt(np.mean(np.sum((U - T)[v] ** 2, axis=1)))) for U in (fr.U_local, fr.U)],
        "max_dudx": float(np.max(np.abs(G[:, 0, 0]))),
        "peak_uvw": [float(np.max(np.abs(T[:, j]))) for j in range(3)],
    }


def fig_aldvc(out: Path) -> dict:
    n, noise, ws = 96, 0.03, 12
    cases = [_aldvc_case(n, noise, ws, s) for s in ALDVC_SEEDS]
    c0 = cases[0]
    mesh, fr, G, v = c0["mesh"], c0["fr"], c0["G"], c0["valid"]
    lim = c0["max_dudx"]
    over = max(float(np.nanmax(np.abs(fr.F_local[v, 0, 0]))), float(np.nanmax(np.abs(fr.F[v, 0, 0])))) > lim
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.2), constrained_layout=True)
    panels = (
        (G[:, 0, 0], "(a) Imposed", None),
        (fr.F_local[:, 0, 0], "(b) Local subsets only", c0["rms_dudx"][0]),
        (fr.F[:, 0, 0], "(c) AL-DVC: local + global", c0["rms_dudx"][1]),
    )
    for ax, (vals, title, err) in zip(axes, panels):
        img, ext = layer(mesh, vals, v)
        im = ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x [voxel]")
        if err is not None:
            note(ax, f"RMS error {err:.3f}")
    axes[0].set_ylabel("y [voxel]")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    colorbar(fig, im, axes[-1], r"$\partial u / \partial x$  [–]", extend="both" if over else "neither")
    info = save(fig, out, "aldvc_local_vs_global.png")

    def mean_sd(key: str, i: int) -> list:
        vals = np.array([cc[key][i] for cc in cases])
        return [sig(vals.mean()), sig(vals.std(ddof=1))]

    ratio_g = np.array([cc["rms_dudx"][0] / cc["rms_dudx"][1] for cc in cases])
    ratio_F = np.array([cc["rms_F"][0] / cc["rms_F"][1] for cc in cases])
    drop_u = np.array([1 - cc["rms_U"][1] / cc["rms_U"][0] for cc in cases])
    numbers = {
        **info,
        "setup": {
            "volume_voxels": [n, n, n],
            "texture": "Gaussian-filtered speckle (sigma 2 voxel)",
            "displacement": "smooth Gaussian bump about the centre (u: a local stretch along x; v: a bulge)",
            "peak_abs_displacement_uvw_voxel": sig(c0["peak_uvw"], 3),
            "noise_std_both_volumes": noise,
            "snr": sig(c0["snr"], 3),
            "subset_voxel": ws,
            "step_voxel": 4,
            "search_radius_voxel": 4,
            "n_nodes": int(mesh.n_nodes),
            "n_valid_nodes": int(np.sum(v)),
            "shown": "middle node layer (z), seed set 1",
            "seed_sets": [list(s) for s in ALDVC_SEEDS],
        },
        "drawn_case": {
            "rms_dudx_local": sig(c0["rms_dudx"][0]),
            "rms_dudx_aldvc": sig(c0["rms_dudx"][1]),
            "rms_gradient_all9_local": sig(c0["rms_F"][0]),
            "rms_gradient_all9_aldvc": sig(c0["rms_F"][1]),
            "rms_displacement_local_voxel": sig(c0["rms_U"][0]),
            "rms_displacement_aldvc_voxel": sig(c0["rms_U"][1]),
            "max_abs_dudx_imposed": sig(lim),
            "beta": sig(fr.admm.beta),
            "admm_steps": int(fr.admm.n_steps),
        },
        "over_seed_sets_mean_sd": {
            "rms_dudx_local": mean_sd("rms_dudx", 0),
            "rms_dudx_aldvc": mean_sd("rms_dudx", 1),
            "rms_gradient_all9_local": mean_sd("rms_F", 0),
            "rms_gradient_all9_aldvc": mean_sd("rms_F", 1),
            "rms_displacement_local_voxel": mean_sd("rms_U", 0),
            "rms_displacement_aldvc_voxel": mean_sd("rms_U", 1),
            "dudx_error_ratio_local_over_aldvc_range": sig([ratio_g.min(), ratio_g.max()], 3),
            "gradient_error_ratio_local_over_aldvc_range": sig([ratio_F.min(), ratio_F.max()], 3),
            "displacement_error_reduction_percent_range": sig([100 * drop_u.min(), 100 * drop_u.max()], 3),
        },
    }
    print(f"    du/dx rms {c0['rms_dudx']}, all-9 rms {c0['rms_F']}, U rms {c0['rms_U']}, ratios {ratio_F}, U drop {drop_u}")
    return numbers


# ------------------------------------------------------------------ 2. accumulative vs incremental
def fig_tracking(out: Path) -> dict:
    n, step_deg, n_steps = 112, 5.0, 9
    shape = (56, n, n)
    c = ((n - 1) / 2, (n - 1) / 2, 0.0)
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c[0], yy - c[1])
    specimen = np.broadcast_to(rr < 0.46 * n, shape)  # a speckled cylinder turning about its axis
    ref = (generate_speckle_volume(shape, sigma=2.0, seed=4) * specimen).astype(np.float32)
    angles = [step_deg * k for k in range(1, n_steps + 1)]
    # n_iter: the fixed-point inversion of warp_volume_lagrangian contracts its error by 2 sin(theta / 2) per
    # iteration, so the default 25 is not enough above about 40 degrees (and nothing converges from 60 up)
    vols = [ref] + [warp_volume_lagrangian(ref, rotation_displacement(a, "z", c), n_iter=80) for a in angles]
    mask = np.ascontiguousarray(np.broadcast_to(rr < 0.44 * n, shape))
    curves = {}
    n_inner = 0
    for mode in ("accumulative", "incremental"):
        p = para(winsize=16, winstepsize=8, search_radius=12, reference_mode=mode)
        res = run_aldvc(p, vols, masks=[mask] * len(vols), compute_strain=False)
        X = res.dvc_mesh.coordinates
        inner = (np.hypot(X[:, 0] - c[0], X[:, 1] - c[1]) < 0.36 * n) & res.dvc_mesh.node_valid
        n_inner = int(inner.sum())
        med, q25, q75, conv = [], [], [], []
        for a, fr in zip(angles, res.result_disp):
            e = np.linalg.norm(fr.U_accum - evaluate_at_nodes(rotation_displacement(a, "z", c), X), axis=1)[inner]
            q = np.percentile(e, [25, 50, 75])
            q25.append(float(q[0]))
            med.append(float(q[1]))
            q75.append(float(q[2]))
            conv.append(100.0 * float(np.mean(fr.status[inner] == STATUS_CONVERGED)))
        curves[mode] = (med, q25, q75, conv)
        print(f"    {mode}: median error {np.round(med, 4)} converged {np.round(conv, 1)}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 3.3), constrained_layout=True)
    styles = (
        ("accumulative", PINK, "o", "-", "accumulative: every scan vs. the first"),
        ("incremental", ACCENT, "s", "--", "incremental: every scan vs. the previous one"),
    )
    for mode, color, marker, ls, label in styles:
        med, q25, q75, conv = curves[mode]
        a1.fill_between(angles, q25, q75, color=color, alpha=0.18, lw=0)
        a1.semilogy(angles, med, marker=marker, ls=ls, color=color, ms=5, lw=1.8, label=label)
        a2.plot(angles, conv, marker=marker, ls=ls, color=color, ms=5, lw=1.8)
    a1.set_xlabel("total rotation [°]  (5° between scans)")
    a1.set_ylabel("error of the cumulative\ndisplacement [voxel]")
    a1.set_title("(a) Median error, interquartile band")
    a2.set_xlabel("total rotation [°]")
    a2.set_ylabel("converged nodes [%]")
    a2.set_ylim(0, 105)
    a2.set_title("(b) Nodes that converged")
    for ax in (a1, a2):
        ax.grid(True)
        ax.set_xticks(angles)
    a1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    fig.legend(loc="outside upper center", ncol=2, frameon=False, fontsize=9.5)
    info = save(fig, out, "tracking_modes.png")
    acc, inc = curves["accumulative"], curves["incremental"]
    return {
        **info,
        "setup": {
            "volume_voxels": list(shape[::-1]),
            "specimen": "speckled cylinder, radius 0.46 x 112 voxel, masked",
            "rotation_about": "z (cylinder axis)",
            "step_deg": step_deg,
            "n_scans": n_steps + 1,
            "subset_voxel": 16,
            "step_voxel": 8,
            "search_radius_voxel": 12,
            "nodes_evaluated": n_inner,
            "evaluated": "nodes within 0.36 x 112 voxel of the axis",
        },
        "angles_deg": angles,
        "accumulative": {"median_error_voxel": sig(acc[0], 3), "converged_percent": sig(acc[3], 3)},
        "incremental": {"median_error_voxel": sig(inc[0], 3), "converged_percent": sig(inc[3], 3)},
    }


# ------------------------------------------------------------------ 3. noise floor (DVC Challenge 2.0 protocol)
def fig_noise_floor(out: Path) -> dict:
    # Two repeat scans of a static specimen: the same 8-bit volume with independent Gaussian noise of k x 255 in
    # each, as two real scans would have (the DVC Challenge 2.0 repeat-scan protocol). The run uses the default
    # parameters, so the plain Gauss-Newton step (icgn_noise_hessian=False).
    n, k = 96, 0.02
    ref = generate_speckle_volume((n, n, n), sigma=2.0, seed=11)
    ref8 = np.round(255.0 * (ref - ref.min()) / max(float(np.ptp(ref)), 1e-12))
    rng = np.random.default_rng(12)
    f, g = (np.clip(np.round(ref8 + rng.normal(0.0, k * 255.0, ref8.shape)), 0, 255).astype(np.float32) for _ in range(2))
    res = run_aldvc(para(winsize=16, winstepsize=8, search_radius=4), [f, g])
    nf = noise_floor(res, 0)
    fs = frame_stats(res, 0, ("disp_u", "disp_v", "disp_w"), NodeFilter(), with_ci=True)
    U = frame_view(res, 0).displacement()[fs.selection.mask]
    lim = float(np.percentile(np.abs(U), 99.7))
    bins = np.linspace(-lim, lim, 37)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.0), constrained_layout=True, sharey=True)
    naive = []
    top = 0.0
    for j, (ax, name) in enumerate(zip(axes, ("u", "v", "w"))):
        st = fs.stats[f"disp_{name}"]
        b, s = nf.displacement.bias[j], nf.displacement.noise[j]
        counts, _, _ = ax.hist(U[:, j], bins=bins, color=ACCENT, alpha=0.9, lw=0)
        top = max(top, float(counts.max()))
        # the reference lines stop just above the tallest bar, below the text
        ax.vlines(b, 0, 1.08 * counts.max(), color=AMBER, lw=1.6, label="bias (mean)" if j == 0 else None)
        ax.vlines(
            [b - s, b + s],
            0,
            1.08 * counts.max(),
            color=AMBER,
            lw=1.2,
            ls="--",
            label="bias ± noise floor" if j == 0 else None,
        )
        hw = float(sps.t.ppf(0.975, st.n - 1) * np.std(U[:, j], ddof=1) / np.sqrt(st.n))
        naive.append(hw)
        ax.set_title(f"({'abc'[j]}) {name}")
        ax.set_xlabel(f"measured {name} [voxel]")
        ax.text(
            0.04,
            0.96,
            f"bias {b:+.4f}, noise floor {s:.4f} voxel\n"
            f"95 % CI of the mean ±{st.ci95:.4f}\n"
            f"{st.n} nodes ≈ {st.n_eff:.0f} independent",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color=TEXT,
            linespacing=1.35,
        )
    axes[0].set_ylim(0, top * 1.72)
    axes[0].set_ylabel("nodes")
    fig.legend(loc="outside upper center", ncol=2, frameon=False, fontsize=9.5)
    info = save(fig, out, "noise_floor.png")
    ci = [fs.stats[f"disp_{c}"].ci95 for c in "uvw"]
    neff = [fs.stats[f"disp_{c}"].n_eff for c in "uvw"]
    snr = float(ref8.std()) / (k * 255.0)
    print(
        f"    bias {nf.displacement.bias}, noise {nf.displacement.noise}, u_rms {nf.displacement.u_rms:.4f}, "
        f"rot {nf.rigid_fit.rotation_deg:.4f}, n {nf.displacement.n}, ci {ci}, naive {naive}, n_eff {neff}"
    )
    return {
        **info,
        "setup": {
            "volume_voxels": [n, n, n],
            "scans": "two 8-bit scans of one static speckle volume",
            "noise": f"independent Gaussian, std {k:.0%} of 255, in each scan",
            "snr": sig(snr, 3),
            "subset_voxel": 16,
            "step_voxel": 8,
            "search_radius_voxel": 4,
        },
        "n_nodes": int(nf.displacement.n),
        "n_converged": int(np.sum(res.result_disp[0].status == STATUS_CONVERGED)),
        "bias_voxel": sig(nf.displacement.bias, 3),
        "noise_floor_voxel": sig(nf.displacement.noise, 3),
        "u_rms_voxel": sig(nf.displacement.u_rms, 3),
        "ci95_half_width_voxel": sig(ci, 3),
        "naive_ci95_half_width_voxel": sig(naive, 3),
        "ci_widening_factor": sig(np.array(ci) / np.array(naive), 3),
        "n_eff": sig(neff, 3),
        "rigid_fit_rotation_deg": sig(nf.rigid_fit.rotation_deg, 3),
        "maer_strain": sig(nf.maer, 3) if nf.maer is not None else None,
        "sder_strain": sig(nf.sder, 3) if nf.sder is not None else None,
        "vsg_voxel": sig(nf.vsg, 3) if nf.vsg is not None else None,
    }


# ------------------------------------------------------------------ 4. per-node uncertainty where the texture fades
def fig_uncertainty(out: Path) -> dict:
    n, noise = 96, 0.01
    shape = (n, n, n)
    c = ((n - 1) / 2,) * 3
    base = generate_speckle_volume(shape, sigma=2.0, seed=8)
    ramp = np.linspace(1.0, 0.25, n, dtype=np.float32)[None, None, :]  # the texture contrast fades along x
    ref = (0.5 + (base - 0.5) * ramp).astype(np.float32)
    disp = affine_displacement(np.diag([0.01, -0.004, 0.0]), (0.4, -0.3, 0.2), c)
    dfm = warp_volume_lagrangian(ref, disp)
    res = run_aldvc(
        para(winsize=16, winstepsize=4, search_radius=4),
        [add_noise(ref, noise, seed=1), add_noise(dfm, noise, seed=2)],
        compute_strain=False,
    )
    mesh, fr = res.dvc_mesh, res.result_disp[0]
    v = mesh.node_valid & np.all(np.isfinite(fr.U_std), axis=1)
    err = np.linalg.norm(fr.U_local - evaluate_at_nodes(disp, mesh.coordinates), axis=1)
    std = np.linalg.norm(fr.U_std, axis=1)
    hi = float(np.nanpercentile(np.concatenate([err[v], std[v]]), 99))
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.1), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, vals, title in ((axes[0], std, r"(a) Predicted $|U_\mathrm{std}|$"), (axes[1], err, "(b) Actual |error|")):
        img, ext = layer(mesh, vals, v)
        im = ax.imshow(img, cmap="turbo", vmin=0, vmax=hi, extent=ext, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x [voxel]  (contrast 1 → 0.25)")
    axes[1].set_yticklabels([])
    axes[0].set_ylabel("y [voxel]")
    colorbar(fig, im, axes[1], "voxel", extend="max")
    # calibration: nodes binned by predicted std (deciles); rms predicted against rms actual error in each bin
    nb = 10
    edges = np.quantile(std[v], np.linspace(0, 1, nb + 1))
    idx = np.clip(np.digitize(std[v], edges) - 1, 0, nb - 1)
    pred = np.array([np.sqrt(np.mean(std[v][idx == b] ** 2)) for b in range(nb)])
    emp = np.array([np.sqrt(np.mean(err[v][idx == b] ** 2)) for b in range(nb)])
    ax = axes[2]
    top = float(max(pred.max(), emp.max()) * 1.12)
    ax.plot([0, top], [0, top], ":", color=TEXT_2, lw=1.2, label="error = prediction")
    ax.plot(pred, emp, "o-", color=ACCENT, ms=5, lw=1.8, label="nodes in deciles of $U_\\mathrm{std}$")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel(r"predicted RMS $|U_\mathrm{std}|$ [voxel]")
    ax.set_ylabel("actual RMS |error| [voxel]")
    ax.set_title("(c) Calibration")
    ax.grid(True)
    ax.legend(loc="lower right")
    info = save(fig, out, "uncertainty_map.png")
    ratio = pred / emp
    overall = float(np.sqrt(np.mean(std[v] ** 2)) / np.sqrt(np.mean(err[v] ** 2)))
    rho = float(sps.spearmanr(std[v], err[v]).statistic)
    xs = mesh.coordinates[v, 0]
    third = (xs.max() - xs.min()) / 3
    lo_band, hi_band = xs < xs.min() + third, xs > xs.max() - third
    print(f"    pred {np.round(pred, 4)} emp {np.round(emp, 4)} ratio {np.round(ratio, 3)} overall {overall:.3f} rho {rho:.3f}")
    return {
        **info,
        "setup": {
            "volume_voxels": [n, n, n],
            "texture": "speckle whose contrast falls linearly from 1 to 0.25 along x",
            "displacement": "affine: 1 % stretch in x, -0.4 % in y, translation (0.4, -0.3, 0.2) voxel",
            "noise_std_both_volumes": noise,
            "subset_voxel": 16,
            "step_voxel": 4,
            "search_radius_voxel": 4,
            "compared": "|U_std| against the error of the local (IC-GN) displacement",
            "n_nodes": int(v.sum()),
        },
        "rms_prediction_over_rms_error": sig(overall, 3),
        "per_decile_prediction_over_error_range": sig([ratio.min(), ratio.max()], 3),
        "spearman_rank_correlation": sig(rho, 3),
        "rms_error_high_contrast_third_voxel": sig(np.sqrt(np.mean(err[v][lo_band] ** 2)), 3),
        "rms_error_low_contrast_third_voxel": sig(np.sqrt(np.mean(err[v][hi_band] ** 2)), 3),
        "rms_ustd_high_contrast_third_voxel": sig(np.sqrt(np.mean(std[v][lo_band] ** 2)), 3),
        "rms_ustd_low_contrast_third_voxel": sig(np.sqrt(np.mean(std[v][hi_band] ** 2)), 3),
    }


# ------------------------------------------------------------------ 5. rigid-body motion removed
def fig_rigid(out: Path) -> dict:
    n = 96
    c = ((n - 1) / 2,) * 3
    axis = np.array([0.3, 0.2, 0.93])
    axis /= np.linalg.norm(axis)
    angle = 4.0
    R = Rotation.from_rotvec(np.radians(angle) * axis).as_matrix()
    E = np.diag([-0.004, 0.0012, 0.0012])  # 0.4 % compression along x with a lateral expansion
    shift = (2.5, -1.5, 3.0)
    disp = affine_displacement(R @ (np.eye(3) + E) - np.eye(3), shift, c)
    ref = generate_speckle_volume((n, n, n), sigma=2.0, seed=5)
    dfm = warp_volume_lagrangian(ref, disp)
    res = run_aldvc(
        para(winsize=16, winstepsize=8, search_radius=6), [add_noise(ref, 0.005, seed=1), add_noise(dfm, 0.005, seed=2)]
    )
    raw = frame_view(res, 0, "none")
    fix = frame_view(res, 0, "rigid")
    mesh = res.dvc_mesh
    m = raw.selection.mask
    gs = mesh.grid_shape
    X = mesh.coordinates
    lay = (np.arange(mesh.n_nodes) // (gs[1] * gs[2]) == gs[0] // 2) & m
    gain = 40.0
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.25), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    lo, hi = X[lay, :2].min() - 7, X[lay, :2].max() + 7
    for ax, view, title, color, scale in (
        (axes[0], raw, "(a) As measured", PINK, 1.0),
        (axes[1], fix, "(b) Rigid motion removed", ACCENT, gain),
    ):
        U = view.displacement()
        ax.quiver(
            X[lay, 0],
            X[lay, 1],
            U[lay, 0],
            U[lay, 1],
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1.0 / scale,
            width=0.007,
            headwidth=3.5,
            headlength=4,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(hi, lo)  # image convention: y down, as in the maps of the other figures
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [voxel]   (arrows to scale)" if scale == 1.0 else f"x [voxel]   (arrows ×{scale:.0f})")
    axes[1].set_yticklabels([])
    axes[0].set_ylabel("y [voxel]")
    comps = ("exx", "eyy", "ezz")
    truth = np.diag(E)
    meas = np.array([frame_stats(res, 0, comps, motion="none").stats[cc].mean for cc in comps])
    corr = np.array([frame_stats(res, 0, comps, motion="rigid").stats[cc].mean for cc in comps])
    xk = np.arange(3)
    ax = axes[2]
    w = 0.26
    ax.bar(xk - w, truth * 100, w, color=TEAL, label="imposed")
    ax.bar(xk, meas * 100, w, color=PINK, label="as measured", hatch="///", edgecolor=AX_BG, lw=0)
    ax.bar(xk + w, corr * 100, w, color=ACCENT, label="rigid motion removed")
    ax.axhline(0, color=TEXT_2, lw=0.8)
    ax.set_xticks(xk, [r"$\varepsilon_{xx}$", r"$\varepsilon_{yy}$", r"$\varepsilon_{zz}$"], fontsize=10.5)
    ax.set_ylabel("infinitesimal strain [%]")
    ax.set_title("(c) Mean normal strains")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8.5)
    info = save(fig, out, "rigid_removal.png")
    fit = fix.fit
    print(f"    rotation {fit.rotation_deg:.4f}, meas {np.round(meas, 5)}, corr {np.round(corr, 5)}")
    return {
        **info,
        "setup": {
            "volume_voxels": [n, n, n],
            "rotation_deg": angle,
            "rotation_axis": sig(axis, 3),
            "translation_voxel": list(shift),
            "imposed_strain_percent": sig(truth * 100, 3),
            "noise_std_both_volumes": 0.005,
            "subset_voxel": 16,
            "step_voxel": 8,
            "search_radius_voxel": 6,
            "strain": "infinitesimal, plane fit (defaults)",
            "shown": "middle node layer (z)",
            "arrow_gain_panel_b": gain,
        },
        "fitted_rotation_deg": sig(fit.rotation_deg, 5),
        "fitted_translation_voxel": sig(fit.translation, 4),
        "fit_residual_rms_voxel": sig(fit.residual_rms, 3),
        "strain_as_measured_percent": sig(meas * 100, 3),
        "strain_removed_percent": sig(corr * 100, 3),
        "strain_imposed_percent": sig(truth * 100, 3),
        "false_strain_from_rotation_percent": sig((meas - truth) * 100, 3),
        "n_nodes": int(m.sum()),
    }


# ------------------------------------------------------------------ 6. subset size and texture
TEXTURE_SEEDS = ((21, 3), (22, 4), (23, 5))  # (spheres, grain noise); the first one is drawn
SUBSETS = (8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32)


def _texture_case(seeds: tuple[int, int]) -> dict:
    # the recipe of make_texture_report.page_heuristic: Boolean spheres, one grain noise shared by both scans
    radius, phi = 5.0, 0.3
    shape = (96, 104, 112)
    vol, _ = boolean_spheres(shape, radius, phi, seed=seeds[0])
    ref = gaussian_filter(vol, 0.8) + 0.02 * np.random.default_rng(seeds[1]).normal(size=shape).astype(np.float32)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    F = np.array([[0.008, 0.002, 0.0], [0.0, -0.005, 0.001], [0.001, 0.0, 0.006]])
    disp = affine_displacement(F, (0.35, -0.25, 0.4), centre)
    dfm = warp_volume_lagrangian(ref, disp)
    tex = analyse_texture(ref)
    rows = []
    for ws in SUBSETS:
        res = run_aldvc(para(winsize=ws, winstepsize=8, search_radius=4), [ref, dfm], compute_strain=False)
        mesh, fr = res.dvc_mesh, res.result_disp[0]
        interior = ~np.isin(np.arange(mesh.n_nodes), mesh.boundary_nodes)
        e = np.linalg.norm(fr.U - evaluate_at_nodes(disp, mesh.coordinates), axis=1)
        ok = interior & np.isfinite(e)
        rows.append((float(np.sqrt(np.mean(e[ok] ** 2))), float(np.mean(fr.status[interior] == STATUS_CONVERGED))))
    return {"ref": ref, "L": float(tex.length("radial")), "rec": recommend_parameters(tex).subset, "rows": rows}


def fig_texture(out: Path) -> dict:
    cases = [_texture_case(s) for s in TEXTURE_SEEDS]
    errs = np.array([[r[0] for r in cc["rows"]] for cc in cases])
    conv = np.array([[r[1] for r in cc["rows"]] for cc in cases])
    Ls = np.array([cc["L"] for cc in cases])
    L0 = float(Ls.mean())
    rec = cases[0]["rec"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 3.4), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.75]})
    # (a) a slice of the texture with subsets of one, two and four correlation lengths about its centre
    ref, L1 = cases[0]["ref"], cases[0]["L"]
    k, cy, cx, half = ref.shape[0] // 2, ref.shape[1] // 2, ref.shape[2] // 2, 24
    a1.imshow(
        ref[k, cy - half : cy + half, cx - half : cx + half],
        cmap="gray",
        interpolation="nearest",
        extent=(-half - 0.5, half - 0.5, half - 0.5, -half - 0.5),
    )
    outline = [patheffects.withStroke(linewidth=4.2, foreground=AX_BG)]
    handles = []
    for mult, color, ls, label in ((1, TEAL, "-", "1 L"), (2, PINK, "--", "2 L"), (4, AMBER, "-", "4 L")):
        e = mult * L1
        a1.add_patch(Rectangle((-e / 2, -e / 2), e, e, fill=False, ec=color, ls=ls, lw=2.2, path_effects=outline))
        handles.append(Line2D([], [], color=color, ls=ls, lw=2.2, label=label))
    a1.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.9, handlelength=1.8, borderpad=0.4)
    x0, y0 = -half + 3, half - 4
    a1.plot([x0, x0 + L1], [y0, y0], color=TEXT, lw=3, solid_capstyle="butt", path_effects=outline)
    a1.text(
        x0 + L1 + 1.2,
        y0,
        f"L = {L1:.1f} voxel",
        color=TEXT,
        fontsize=9,
        va="center",
        bbox={"boxstyle": "round,pad=0.2", "fc": AX_BG, "ec": "none", "alpha": 0.85},
    )
    a1.set_xticks([])
    a1.set_yticks([])
    a1.set_title("(a) Texture and subsets")
    # (b) error against subset size, mean and spread over three textures
    ws = np.array(SUBSETS, dtype=float)
    mean, sd = errs.mean(axis=0), errs.std(axis=0, ddof=1)
    a2.fill_between(ws, mean - sd, mean + sd, color=ACCENT, alpha=0.25, lw=0, label=f"± 1 s.d., {len(cases)} textures")
    a2.plot(ws, mean, "o-", color=ACCENT, ms=5, lw=1.8, label="mean")
    a2.axvline(rec[0], color=AMBER, ls="--", lw=1.4, label=f"suggested: 4 L → {rec[0]}")
    a2.set_ylim(0, float((mean + sd).max()) * 1.12)
    a2.set_xlabel("subset [voxel]")
    a2.set_ylabel("RMS displacement error [voxel]")
    a2.set_title("(b) Error against subset size")
    a2.grid(True)
    a2.legend(loc="upper right")
    sec = a2.secondary_xaxis("top", functions=(lambda x: x / L0, lambda r: r * L0))
    sec.set_xlabel("subset / L", color=TEXT_2, fontsize=9.5)
    sec.tick_params(colors=TEXT_2, labelsize=9)
    # lift title (a) to the height of title (b), which sits above the second axis
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dy = a2.title.get_window_extent(renderer).y0 - a1.title.get_window_extent(renderer).y0
    a1.set_title(a1.get_title(), pad=plt.rcParams["axes.titlepad"] + dy * 72.0 / fig.dpi)
    info = save(fig, out, "texture_subset.png")
    print(
        f"    L {Ls}, rec {[cc['rec'] for cc in cases]}, mean err {np.round(mean, 4)}, converged {np.round(conv.mean(axis=0), 3)}"
    )
    i4 = int(np.argmin(np.abs(ws - rec[0])))
    return {
        **info,
        "setup": {
            "volume_voxels": [112, 104, 96],
            "texture": "Boolean spheres, radius 5 voxel, volume fraction 0.3, blurred",
            "noise": "grain noise std 0.02 shared by both scans (no independent noise)",
            "displacement": "affine, gradient up to 0.8 %, translation (0.35, -0.25, 0.4) voxel",
            "step_voxel": 8,
            "search_radius_voxel": 4,
            "evaluated": "interior nodes",
            "seed_sets": [list(s) for s in TEXTURE_SEEDS],
            "subsets_voxel": list(SUBSETS),
        },
        "correlation_length_1e_voxel": sig(Ls, 3),
        "suggested_subset_voxel": [list(map(int, cc["rec"])) for cc in cases],
        "rms_error_mean_voxel": sig(mean, 3),
        "rms_error_sd_voxel": sig(sd, 3),
        "converged_fraction_mean": sig(conv.mean(axis=0), 3),
        "error_at_suggested_subset_voxel": sig(mean[i4], 3),
        "error_at_smallest_subset_voxel": sig(mean[0], 3),
        "error_at_largest_subset_voxel": sig(mean[-1], 3),
    }


# ------------------------------------------------------------------ 7. the in-app texture guide, at 2x
def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        log.warning("imageio-ffmpeg not available (%s): the guide animations stay GIFs", exc)
        return None


def _gif_to_mp4(ff: str, gif: Path, mp4: Path, fps: int) -> None:
    # BT.709 matrix, tagged, so browsers convert back with the matrix it was encoded with
    vf = (
        f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos:out_color_matrix=bt709:out_range=tv,"
        "format=yuv420p,setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709"
    )
    cmd = [
        ff,
        "-y",
        "-v",
        "error",
        "-i",
        str(gif),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "20",
        "-tune",
        "animation",
        "-profile:v",
        "high",
        "-movflags",
        "+faststart",
        "-an",
        str(mp4),
    ]
    subprocess.run(cmd, check=True)


def _poster(gif: Path, png: Path) -> tuple[int, int]:
    """The last frame of ``gif`` (the finished state) as a PNG."""
    with Image.open(gif) as im:
        im.seek(im.n_frames - 1)
        frame = im.convert("RGB")
    frame.save(png, optimize=True)
    return frame.size


# the site figures' font; DejaVu Sans supplies the glyphs Arial lacks (the subscript of Σ₀). Matplotlib falls
# back glyph by glyph only along a list of families, not along the entries of the generic "sans-serif".
GUIDE_RC = {"font.family": ["Arial", "DejaVu Sans"]}


@dataclass(frozen=True)
class _GuideLength:
    """The members of a ``TextureResult`` that ``recommend_parameters()`` reads, for one length on every axis."""

    value: float
    periodicity: None = None
    noise_floor: float = float("nan")

    def length(self, axis: str, threshold: float | None = None) -> float:
        return self.value


def _notes_in_text_colour(mod) -> None:
    """Draw the guide's one-line notes above its image (axes y > 1) in the text colour, not in amber.

    The notes are made inside the module's functions; they all exist when the animation is set up, so
    ``FuncAnimation`` is wrapped to recolour them first. Amber stays on the thresholds it marks.
    """
    real = mod.FuncAnimation

    def animation(fig, func, *args, **kwargs):
        for ax in fig.axes:
            for text in ax.texts:
                if text.get_transform() is ax.transAxes and text.get_position()[1] > 1:
                    text.set_color(TEXT)
        return real(fig, func, *args, **kwargs)

    mod.FuncAnimation = animation


def _guide_subset(mod, tex: np.ndarray, path: Path) -> dict:
    """Step 3 of the guide: the 1/e length, and the subset and step ``recommend_parameters()`` makes of it.

    The in-app ``make_subset`` draws ``4 x L`` and half of it unrounded; the application suggests even
    edges, so the site draws and labels those (L = 4.18 gives 16.7 -> 18 voxels and a step of 10).
    """
    cx, cy = mod.NX // 2, mod.NY // 2
    edge = 96
    box = (cx - edge // 2, cx + edge // 2, cy - edge // 2, cy + edge // 2)
    lags = np.arange(0, edge // mod.LAG_FRACTION + 1)
    rho = np.array([mod.rho_bb(tex, box, int(h), 0, True) for h in lags])
    length = mod.length_1e(rho, lags.astype(float))
    if not np.isfinite(length):
        raise RuntimeError("the guide texture never reaches 1/e: no subset to draw")
    rec = recommend_parameters(_GuideLength(length))
    subset, step = rec.subset[0], rec.step[0]

    fig, (ax_img, ax_curve) = plt.subplots(
        1, 2, figsize=(7.4, 3.1), dpi=mod.DPI, gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.3}
    )
    fig.set_facecolor(mod.FACE)
    crop = 26
    ax_img.imshow(
        tex[cy - crop : cy + crop, cx - crop : cx + crop],
        cmap="gray",
        origin="lower",
        vmin=-2.5,
        vmax=2.5,
        extent=[-crop - 0.5, crop - 0.5, -crop - 0.5, crop - 0.5],
    )
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_visible(False)
    half = subset / 2
    ax_img.add_patch(Rectangle((-half, -half), subset, subset, fill=False, ec=mod.BOX, lw=2.0))
    ax_img.add_patch(Rectangle((-half + step, -half), subset, subset, fill=False, ec=mod.BOX, lw=1.2, ls="--"))
    ax_img.plot([-half, -half + length], [-half - 3, -half - 3], color=mod.THRESHOLD, lw=3, solid_capstyle="butt")
    ax_img.text(-half + length + 1.5, -half - 3, "L(1/e)", color=mod.THRESHOLD, fontsize=9, ha="left", va="center")
    ax_img.set_title(
        f"{rec.factor:g} × L = {rec.factor * length:.1f} → subset {subset} voxel,  step {step}", color=TEXT, fontsize=9
    )
    ax_img.set_xlim(-crop - 0.5, crop - 0.5)
    ax_img.set_ylim(-crop - 0.5, crop - 0.5)

    mod.style(ax_curve)
    ax_curve.plot(lags, rho, color=mod.CURVE, lw=1.8)
    ax_curve.axhline(mod.INV_E, color=mod.THRESHOLD, ls="--", lw=1.0)
    ax_curve.axhline(0.0, color=mod.GRID, lw=0.6)
    ax_curve.plot([length], [mod.INV_E], "o", color=mod.THRESHOLD, ms=6)
    ax_curve.annotate(
        f"L(1/e) = {length:.2f}",  # two decimals, so that 4 x L in the title adds up
        xy=(length, mod.INV_E),
        xytext=(length + 4, mod.INV_E + 0.25),
        color=mod.THRESHOLD,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": mod.THRESHOLD},
    )
    ax_curve.set_xlim(0, lags[-1])
    ax_curve.set_ylim(-0.25, 1.05)
    ax_curve.set_xlabel("lag h [voxel]", fontsize=9)
    ax_curve.set_ylabel("ρ(h)", fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.2)
    fig.savefig(path, dpi=mod.DPI, facecolor=mod.FACE)
    plt.close(fig)
    return {"length_1e_voxel": sig(length, 3), "subset_voxel": int(subset), "step_voxel": int(step)}


def fig_guide(out: Path) -> dict:
    spec = importlib.util.spec_from_file_location("make_guide_animations", ROOT / "scripts" / "make_guide_animations.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ff = _ffmpeg()
    files = {}
    # the guide's own layout and colours (matplotlib defaults otherwise), in the site figures' font
    with (
        tempfile.TemporaryDirectory(prefix="aldvc_guide_") as tmp,
        plt.style.context("default"),
        plt.rc_context(GUIDE_RC),
    ):
        # module globals, read at call time; main() is not called (it deletes files in OUT)
        mod.OUT, mod.DPI, mod.FACE = Path(tmp), GUIDE_DPI, BG
        _notes_in_text_colour(mod)
        mod.make_overlap_correction(mod.speckle(seed=4, sigma=6.0))
        mod.make_rve_sweep(mod.speckle())
        subset_numbers = _guide_subset(mod, mod.speckle(), Path(tmp) / "subset.png")
        for name in ("overlap_correction", "rve_sweep"):
            gif = Path(tmp) / f"{name}.gif"
            entry = {"gif_bytes": gif.stat().st_size}
            mp4, poster = out / f"guide_{name}.mp4", out / f"guide_{name}.png"
            if ff is not None:
                _gif_to_mp4(ff, gif, mp4, mod.FPS)
                if mp4.stat().st_size < gif.stat().st_size:
                    w, h = _poster(gif, poster)
                    entry.update(
                        {
                            "file": mp4.name,
                            "poster": poster.name,
                            "width_px": w,
                            "height_px": h,
                            "bytes": mp4.stat().st_size,
                            "poster_bytes": poster.stat().st_size,
                            "fps": mod.FPS,
                        }
                    )
                    print(
                        f"  wrote {mp4.name}: {w} x {h} px, {mp4.stat().st_size / 1024:.0f} kB "
                        f"(GIF {gif.stat().st_size / 1024:.0f} kB), poster {poster.stat().st_size / 1024:.0f} kB"
                    )
                    files[name] = entry
                    continue
                mp4.unlink()
            shutil.copyfile(gif, out / f"guide_{name}.gif")
            with Image.open(gif) as im:
                entry.update(
                    {"file": f"guide_{name}.gif", "width_px": im.width, "height_px": im.height, "bytes": gif.stat().st_size}
                )
            print(f"  wrote guide_{name}.gif: {gif.stat().st_size / 1024:.0f} kB")
            files[name] = entry
        png = out / "guide_subset.png"
        with Image.open(Path(tmp) / "subset.png") as im:
            rgb = im.convert("RGB")
        rgb.save(png, optimize=True)
        files["subset"] = {
            "file": png.name,
            "width_px": rgb.width,
            "height_px": rgb.height,
            "bytes": png.stat().st_size,
            **subset_numbers,
        }
        print(f"  wrote {png.name}: {rgb.width} x {rgb.height} px, {png.stat().st_size / 1024:.0f} kB")
    return {
        "dpi": GUIDE_DPI,
        "source": (
            "scripts/make_guide_animations.py (module globals OUT, DPI, FACE overridden; notes recoloured); "
            "step 3 drawn by make_site_figures.py with recommend_parameters()"
        ),
        **files,
    }


FIGURES = {
    "aldvc": ("aldvc_local_vs_global", fig_aldvc),
    "tracking": ("tracking_modes", fig_tracking),
    "noise": ("noise_floor", fig_noise_floor),
    "uncertainty": ("uncertainty_map", fig_uncertainty),
    "rigid": ("rigid_removal", fig_rigid),
    "texture": ("texture_subset", fig_texture),
    "guide": ("guide", fig_guide),
}


def write_numbers(out: Path, results: dict) -> None:
    """Merge ``results`` into ``out/numbers.json`` (a run with ``--only`` keeps the other entries)."""
    path = out / NUMBERS
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read %s (%s); it is rewritten", path, exc)
    data["_about"] = "Written by scripts/make_site_figures.py; regenerate, never edit. Lengths in voxels."
    data["al_dvc_version"] = __version__
    data["backend"] = BACKEND
    data["figure_dpi"] = DPI
    data.setdefault("figures", {}).update(results)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
    print(f"  wrote {path.name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default {DEFAULT_OUT})")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), help="make only these figures")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("al_dvc").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    style()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    t_all = time.perf_counter()
    for name in args.only or FIGURES:
        key, make = FIGURES[name]
        t0 = time.perf_counter()
        results[key] = make(out)
        print(f"  {name}: {time.perf_counter() - t0:.1f} s")
    write_numbers(out, results)
    print(f"done in {time.perf_counter() - t_all:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
