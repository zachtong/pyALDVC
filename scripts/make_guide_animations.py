"""Regenerate the demonstrations of the in-app texture analysis guide.

Output (``src/al_dvc/gui/assets/guide/``):

* ``region_window.gif``  -- one autocorrelation analysis: a window inside the region, its copy sliding
  by the shift h while the curve rho(h) is traced; the region edge stops the copy.
* ``rve_sweep.gif``      -- the RVE analysis: windows of growing size, all centred in the region and
  analysed with the same shifts, and the correlation length settling with the size.
* ``subset.png``         -- from the 1/e length to the subset: subset = 2.5 x L(1/e), step = subset / 2.

Everything is 2-D for legibility; the GUI does the same in 3-D. Run from the repository root::

    python scripts/make_guide_animations.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from al_dvc.texture.rve import DEFAULT_MIN_SPAN, DEFAULT_TOLERANCE_ABS, DEFAULT_TOLERANCE_REL, decide_plateau  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "src" / "al_dvc" / "gui" / "assets" / "guide"
FACE, TEXT, GRID = "#0c0d12", "#e2e8f0", "#4b5563"
REGION, WINDOW, COPY, CURVE, THRESHOLD = "#f97316", "#f8fafc", "#22d3ee", "#60a5fa", "#fbbf24"
NY, NX = 120, 160
SIGMA = 2.2
FPS = 5
DPI = 90
INV_E = float(np.exp(-1.0))


def speckle(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = gaussian_filter(rng.normal(size=(NY, NX)), SIGMA)
    return (tex - tex.mean()) / tex.std()


def rho_2d(tex: np.ndarray, region: tuple[int, int, int, int], window: tuple[int, int, int, int], dx: int, dy: int) -> float:
    """Sliding-window autocorrelation of a 2-D image: the window against its copy shifted by (dx, dy).

    ``region`` and ``window`` are ``(x0, x1, y0, y1)``; the mean is taken over the region.
    """
    rx0, rx1, ry0, ry1 = region
    wx0, wx1, wy0, wy1 = window
    u = tex - tex[ry0:ry1, rx0:rx1].mean()
    a = u[wy0:wy1, wx0:wx1]
    b = u[wy0 + dy : wy1 + dy, wx0 + dx : wx1 + dx]
    return float((a * b).sum() / (a * a).sum())


def length_1e(rho: np.ndarray, lags: np.ndarray) -> float:
    """Shift at which rho crosses 1/e (linear interpolation), NaN when it never does."""
    below = np.flatnonzero(rho < INV_E)
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = below[0]
    r0, r1 = rho[i - 1], rho[i]
    return float(lags[i - 1] + (r0 - INV_E) / (r0 - r1) * (lags[i] - lags[i - 1]))


def style(ax) -> None:
    ax.set_facecolor(FACE)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def image_axes(ax, tex: np.ndarray) -> None:
    ax.imshow(tex, cmap="gray", origin="lower", vmin=-2.5, vmax=2.5, extent=[-0.5, NX - 0.5, -0.5, NY - 0.5])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def box(ax, x0, x1, y0, y1, color, ls="-", lw=1.6):
    return ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=color, ls=ls, lw=lw))


# ---------------------------------------------------------------------- 1. region, window, shift
def make_region_window(tex: np.ndarray) -> None:
    region = (20, 140, 15, 105)  # 120 x 90
    w = 40
    cx, cy = (region[0] + region[1]) // 2, (region[2] + region[3]) // 2
    window = (cx - w // 2, cx + w // 2, cy - w // 2, cy + w // 2)
    reach = (region[1] - window[1], region[3] - window[3])  # how far the copy can move: 40 right, 25 up
    shifts_x = list(range(0, reach[0] + 1, 2))
    shifts_y = list(range(0, reach[1] + 1, 2))
    lags_x = np.array(shifts_x, dtype=float)
    rho_x = np.array([rho_2d(tex, region, window, h, 0) for h in shifts_x])
    rho_y = np.array([rho_2d(tex, region, window, 0, h) for h in shifts_y])
    frames = (
        [("x", i) for i in range(len(shifts_x))]
        + [("hold", 0)] * 4
        + [("y", i) for i in range(len(shifts_y))]
        + [("hold", 1)] * 6
    )

    fig, (ax_img, ax_curve) = plt.subplots(
        1, 2, figsize=(7.4, 3.1), dpi=DPI, gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.3}
    )
    fig.set_facecolor(FACE)
    image_axes(ax_img, tex)
    box(ax_img, *region, REGION, ls="--")
    box(ax_img, *window, WINDOW)
    copy = box(ax_img, *window, COPY)
    arrow = FancyArrowPatch((cx, cy), (cx, cy), color=COPY, lw=1.4, arrowstyle="->", mutation_scale=12)
    ax_img.add_patch(arrow)
    ax_img.text(region[0], region[3] + 3, "region", color=REGION, fontsize=8, va="bottom")
    ax_img.text(window[0], window[2] - 3, "window", color=WINDOW, fontsize=8, va="top")
    label = ax_img.text(0.5, -0.06, "", color=COPY, fontsize=9, ha="center", va="top", transform=ax_img.transAxes)
    note = ax_img.text(0.5, 1.10, "", color=REGION, fontsize=8, ha="center", va="bottom", transform=ax_img.transAxes)

    style(ax_curve)
    ax_curve.set_xlim(0, reach[0] + 1)
    ax_curve.set_ylim(-0.25, 1.05)
    ax_curve.axhline(INV_E, color=THRESHOLD, ls="--", lw=1.0)
    ax_curve.text(reach[0], INV_E + 0.03, "1/e", color=THRESHOLD, fontsize=8, ha="right")
    ax_curve.axhline(0.0, color=GRID, lw=0.6)
    ax_curve.set_xlabel("shift h [voxel]", fontsize=9)
    ax_curve.set_ylabel("ρ(h)", fontsize=9)
    (line_x,) = ax_curve.plot([], [], color=CURVE, lw=1.8, label="along x")
    (line_y,) = ax_curve.plot([], [], color="#f472b6", lw=1.8, ls="--", label="along y")
    (dot,) = ax_curve.plot([], [], "o", color=COPY, ms=6)
    ax_curve.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=TEXT)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.2)

    def draw(k):
        kind, i = frames[k]
        if kind == "x":
            dx, dy = shifts_x[i], 0
            line_x.set_data(lags_x[: i + 1], rho_x[: i + 1])
            dot.set_data([dx], [rho_x[i]])
            note.set_text("")
        elif kind == "y":
            dx, dy = 0, shifts_y[i]
            line_y.set_data(np.array(shifts_y[: i + 1], dtype=float), rho_y[: i + 1])
            dot.set_data([dy], [rho_y[i]])
            note.set_text("")
        else:
            dx, dy = (reach[0], 0) if i == 0 else (0, reach[1])
            dot.set_data([dx + dy], [rho_x[-1] if i == 0 else rho_y[-1]])
            note.set_text("the copy stops at the region edge: the region sets the largest shift")
        copy.set_xy((window[0] + dx - 0.5, window[2] + dy - 0.5))
        arrow.set_positions((cx, cy), (cx + dx, cy + dy))
        label.set_text(f"copy shifted by h = {dx + dy}   ρ = {rho_2d(tex, region, window, dx, dy):.2f}")
        return copy, arrow, label, line_x, line_y, dot, note

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(OUT / "region_window.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)


# ---------------------------------------------------------------------- 2. RVE: growing windows
def make_rve_sweep(tex: np.ndarray) -> None:
    region = (20, 140, 15, 105)
    cx, cy = (region[0] + region[1]) // 2, (region[2] + region[3]) // 2
    sizes = [12, 20, 28, 36, 44, 52, 60, 68]
    shifts = list(range(0, 13))  # the same shifts for every size (the smallest reach is 11 for size 68)
    demo_shifts = [0, 4, 8, 12]  # the copies shown while a size is analysed
    lengths = []
    for s in sizes:
        window = (cx - s // 2, cx + s // 2, cy - s // 2, cy + s // 2)
        rho = np.array([rho_2d(tex, region, window, h, 0) for h in shifts])
        lengths.append(length_1e(rho, np.array(shifts, dtype=float)))
    lengths = np.array(lengths)
    decision = decide_plateau(
        np.array(sizes, dtype=float),
        list(lengths),
        [float("nan")] * len(sizes),
        INV_E,
        DEFAULT_TOLERANCE_REL,
        DEFAULT_TOLERANCE_ABS,
        DEFAULT_MIN_SPAN,
    )
    stable_i = decision.start_index if decision.converged else None
    frames = [(i, j) for i in range(len(sizes)) for j in range(len(demo_shifts) + 1)] + [(-1, 0)] * 8

    fig, (ax_img, ax_len) = plt.subplots(
        1, 2, figsize=(7.4, 3.1), dpi=DPI, gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.3}
    )
    fig.set_facecolor(FACE)
    image_axes(ax_img, tex)
    box(ax_img, *region, REGION, ls="--")
    ax_img.text(region[0], region[3] + 3, "region", color=REGION, fontsize=8, va="bottom")
    win = box(ax_img, cx, cx, cy, cy, WINDOW)
    copy = box(ax_img, cx, cx, cy, cy, COPY)
    label = ax_img.text(0.5, -0.06, "", color=WINDOW, fontsize=9, ha="center", va="top", transform=ax_img.transAxes)
    note = ax_img.text(0.5, 1.10, "", color=REGION, fontsize=8, ha="center", va="bottom", transform=ax_img.transAxes)

    style(ax_len)
    ax_len.set_xlim(sizes[0] - 4, sizes[-1] + 4)
    lo, hi = np.nanmin(lengths), np.nanmax(lengths)
    ax_len.set_ylim(max(0.0, lo - 1.5), hi + 1.5)
    ax_len.set_xlabel("window edge [voxel]", fontsize=9)
    ax_len.set_ylabel("L(1/e) [voxel]", fontsize=9)
    ax_len.grid(color=GRID, alpha=0.5, lw=0.6)
    (pts,) = ax_len.plot([], [], "o-", color=CURVE, lw=1.6, ms=5)
    band = ax_len.add_patch(Rectangle((sizes[0] - 4, 0.0), sizes[-1] - sizes[0] + 8, 0.0, color=CURVE, alpha=0.0, lw=0))
    vline = ax_len.axvline(sizes[0], color=THRESHOLD, ls="--", lw=1.2, alpha=0.0)
    verdict = ax_len.text(0.03, 0.05, "", color=THRESHOLD, fontsize=8, transform=ax_len.transAxes, ha="left", va="bottom")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.2)

    def draw(k):
        i, j = frames[k]
        if i >= 0:
            s = sizes[i]
            x0, y0 = cx - s // 2, cy - s // 2
            win.set_xy((x0 - 0.5, y0 - 0.5))
            win.set_width(s)
            win.set_height(s)
            copy.set_width(s)
            copy.set_height(s)
            if j < len(demo_shifts):
                copy.set_xy((x0 + demo_shifts[j] - 0.5, y0 - 0.5))
                copy.set_alpha(1.0)
                label.set_text(f"window {s} x {s}, the same shifts as before")
                done = i
            else:
                copy.set_alpha(0.0)
                label.set_text(f"window {s} x {s}: L(1/e) = {lengths[i]:.2f}")
                done = i + 1
            pts.set_data(sizes[:done], lengths[:done])
            note.set_text("")
            band.set_alpha(0.0)
            vline.set_alpha(0.0)
            verdict.set_text("")
        else:
            copy.set_alpha(0.0)
            pts.set_data(sizes, lengths)
            if stable_i is not None:
                ref, tol = decision.reference, decision.tolerance
                band.set_xy((sizes[0] - 4, ref - tol))
                band.set_height(2 * tol)
                band.set_alpha(0.15)
                vline.set_xdata([sizes[stable_i], sizes[stable_i]])
                vline.set_alpha(1.0)
                verdict.set_text(f"stable from {sizes[stable_i]}: the window for step 3")
                note.set_text(f"the RVE is {sizes[stable_i]} voxel: smaller windows scatter, larger ones agree")
                label.set_text("")
        return win, copy, label, pts, band, vline, verdict, note

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(OUT / "rve_sweep.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)


# ---------------------------------------------------------------------- 3. from the length to the subset
def make_subset(tex: np.ndarray) -> None:
    region = (20, 140, 15, 105)
    w = 48
    cx, cy = (region[0] + region[1]) // 2, (region[2] + region[3]) // 2
    window = (cx - w // 2, cx + w // 2, cy - w // 2, cy + w // 2)
    shifts = np.arange(0, 25)
    rho = np.array([rho_2d(tex, region, window, int(h), 0) for h in shifts])
    L = length_1e(rho, shifts.astype(float))
    subset = 2.5 * L
    step = subset / 2

    fig, (ax_img, ax_curve) = plt.subplots(
        1, 2, figsize=(7.4, 3.1), dpi=DPI, gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.3}
    )
    fig.set_facecolor(FACE)
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
    ax_img.add_patch(Rectangle((-half, -half), subset, subset, fill=False, ec=WINDOW, lw=2.0))
    ax_img.add_patch(Rectangle((-half + step, -half), subset, subset, fill=False, ec=WINDOW, lw=1.2, ls="--"))
    ax_img.plot([-half, -half + L], [-half - 3, -half - 3], color=THRESHOLD, lw=3, solid_capstyle="butt")
    ax_img.text(-half + L + 1.5, -half - 3, "L(1/e)", color=THRESHOLD, fontsize=9, ha="left", va="center")
    ax_img.set_title(f"subset = 2.5 × L = {subset:.0f} voxel,  step = subset / 2 = {step:.0f}", color=WINDOW, fontsize=9)
    ax_img.set_xlim(-crop - 0.5, crop - 0.5)
    ax_img.set_ylim(-crop - 0.5, crop - 0.5)

    style(ax_curve)
    ax_curve.plot(shifts, rho, color=CURVE, lw=1.8)
    ax_curve.axhline(INV_E, color=THRESHOLD, ls="--", lw=1.0)
    ax_curve.axhline(0.0, color=GRID, lw=0.6)
    ax_curve.plot([L], [INV_E], "o", color=THRESHOLD, ms=6)
    ax_curve.annotate(
        f"L(1/e) = {L:.1f}",
        xy=(L, INV_E),
        xytext=(L + 4, INV_E + 0.25),
        color=THRESHOLD,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": THRESHOLD},
    )
    ax_curve.set_xlim(0, shifts[-1])
    ax_curve.set_ylim(-0.25, 1.05)
    ax_curve.set_xlabel("shift h [voxel]", fontsize=9)
    ax_curve.set_ylabel("ρ(h)", fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.2)
    fig.savefig(OUT / "subset.png", dpi=DPI, facecolor=FACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tex = speckle()
    make_region_window(tex)
    make_rve_sweep(tex)
    make_subset(tex)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
