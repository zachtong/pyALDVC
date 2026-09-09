"""Regenerate the demonstrations of the in-app texture analysis guide.

Output (``src/al_dvc/gui/assets/guide/``):

* ``overlap_correction.gif`` -- the estimator: a box and its copy shifted by h, the overlap shrinking
  with the shift, and the two curves it gives -- raw, pulled down by the shrinking overlap alone, and
  divided by the number of voxel pairs that still contribute.
* ``rve_sweep.gif``          -- the RVE analysis: concentric boxes about one centre, each measured on
  its own voxels alone, and the correlation length settling with the size.
* ``subset.png``             -- from the 1/e length to the subset: subset = factor x L(1/e), step = subset / 2.

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

from al_dvc.texture.recommend import DEFAULT_FACTOR  # noqa: E402
from al_dvc.texture.rve import DEFAULT_MIN_SPAN, DEFAULT_TOLERANCE_ABS, DEFAULT_TOLERANCE_REL, decide_plateau  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "src" / "al_dvc" / "gui" / "assets" / "guide"
FACE, TEXT, GRID = "#0c0d12", "#e2e8f0", "#4b5563"
REGION, BOX, COPY, CURVE, RAW, THRESHOLD = "#f97316", "#f8fafc", "#38bdf8", "#60a5fa", "#f472b6", "#fbbf24"
NY, NX = 120, 160
SIGMA = 2.2
LAG_FRACTION = 4  # the shipped convention: lags up to a quarter of the edge
FPS = 5
DPI = 90
INV_E = float(np.exp(-1.0))


def speckle(seed: int = 3, sigma: float = SIGMA) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = gaussian_filter(rng.normal(size=(NY, NX)), sigma)
    return (tex - tex.mean()) / tex.std()


def overlap(box: tuple[int, int, int, int], hx: int, hy: int) -> tuple[slice, slice, slice, slice]:
    """Row and column slices of the part of ``box`` (``(x0, x1, y0, y1)``) that its copy at ``(hx, hy)`` covers."""
    x0, x1, y0, y1 = box
    ax0, ax1 = max(x0, x0 - hx), min(x1, x1 - hx)
    ay0, ay1 = max(y0, y0 - hy), min(y1, y1 - hy)
    return slice(ay0, ay1), slice(ax0, ax1), slice(ay0 + hy, ay1 + hy), slice(ax0 + hx, ax1 + hx)


def rho_bb(tex: np.ndarray, box: tuple[int, int, int, int], hx: int, hy: int, corrected: bool = True) -> float:
    """The box against a copy of itself shifted by ``(hx, hy)``, using only the voxels of the box.

    With ``corrected`` every lag is divided by the number of voxel pairs that overlap, which is what
    removes the geometric decay; without it, the raw ratio of the original scripts is returned.
    """
    x0, x1, y0, y1 = box
    u = tex[y0:y1, x0:x1] - tex[y0:y1, x0:x1].mean()
    ry, rx, sy, sx = overlap(box, hx, hy)
    a = u[ry.start - y0 : ry.stop - y0, rx.start - x0 : rx.stop - x0]
    b = u[sy.start - y0 : sy.stop - y0, sx.start - x0 : sx.stop - x0]
    s, m = float((a * b).sum()), float(a.size)
    s0, m0 = float((u * u).sum()), float(u.size)
    if m < 1 or s0 <= 0:
        return float("nan")
    return (s / m) / (s0 / m0) if corrected else s / s0


def pair_fraction(box: tuple[int, int, int, int], hx: int, hy: int) -> float:
    """``M(h) / M(0)``: the share of the box's voxel pairs that a lag keeps."""
    x0, x1, y0, y1 = box
    return max(0.0, (x1 - x0 - abs(hx)) / (x1 - x0)) * max(0.0, (y1 - y0 - abs(hy)) / (y1 - y0))


def length_1e(rho: np.ndarray, lags: np.ndarray) -> float:
    """Lag at which rho crosses 1/e (linear interpolation), NaN when it never does."""
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


def draw_box(ax, x0, x1, y0, y1, color, ls="-", lw=1.6, **kw):
    return ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=color, ls=ls, lw=lw, **kw))


# ---------------------------------------------------------------------- 1. the estimator and its correction
def make_overlap_correction(tex: np.ndarray) -> None:
    edge, crop = 32, 36  # a small box on a coarse texture: the shrinking overlap is then plain to see
    cx, cy = NX // 2, NY // 2
    box = (cx - edge // 2, cx + edge // 2, cy - edge // 2, cy + edge // 2)
    lags = list(range(0, edge // LAG_FRACTION + 1))
    corrected = np.array([rho_bb(tex, box, h, 0, True) for h in lags])
    raw = np.array([rho_bb(tex, box, h, 0, False) for h in lags])
    frames = [("h", i) for i in range(len(lags))] + [("hold", 0)] * 8

    fig, (ax_img, ax_curve) = plt.subplots(
        1, 2, figsize=(7.4, 3.1), dpi=DPI, gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.3}
    )
    fig.set_facecolor(FACE)
    image_axes(ax_img, tex)
    ax_img.set_xlim(cx - crop - 0.5, cx + crop - 0.5)
    ax_img.set_ylim(cy - crop * 0.75 - 0.5, cy + crop * 0.75 - 0.5)
    draw_box(ax_img, *box, BOX)
    copy = draw_box(ax_img, *box, COPY, ls="--")
    shade = ax_img.add_patch(Rectangle((0, 0), 0, 0, color=COPY, alpha=0.28, lw=0))
    arrow = FancyArrowPatch((cx, cy), (cx, cy), color=COPY, lw=1.4, arrowstyle="->", mutation_scale=12)
    ax_img.add_patch(arrow)
    ax_img.text(box[0], box[3] + 3, "B", color=BOX, fontsize=10, va="bottom", fontweight="bold")
    ax_img.text(box[1] + edge // 2, box[3] + 3, "B'", color=COPY, fontsize=10, va="bottom", ha="center", fontweight="bold")
    label = ax_img.text(0.5, -0.06, "", color=TEXT, fontsize=9, ha="center", va="top", transform=ax_img.transAxes)
    note = ax_img.text(0.5, 1.10, "", color=THRESHOLD, fontsize=8, ha="center", va="bottom", transform=ax_img.transAxes)

    style(ax_curve)
    ax_curve.set_xlim(0, lags[-1])
    ax_curve.set_ylim(-0.05, 1.05)
    ax_curve.axhline(INV_E, color=THRESHOLD, ls="--", lw=1.0)
    ax_curve.text(0.2, INV_E + 0.04, "1/e", color=THRESHOLD, fontsize=8, ha="left")
    ax_curve.axhline(0.0, color=GRID, lw=0.6)
    ax_curve.set_xlabel("lag h [voxel]", fontsize=9)
    ax_curve.set_ylabel("ρ(h)", fontsize=9)
    (line_raw,) = ax_curve.plot([], [], color=RAW, lw=1.6, ls="--", label="raw: Σ / Σ₀")
    (line_ok,) = ax_curve.plot([], [], color=CURVE, lw=2.0, label="÷ M(h): corrected")
    (dot,) = ax_curve.plot([], [], "o", color=COPY, ms=6)
    ax_curve.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=TEXT)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.2)

    def draw(k):
        kind, i = frames[k]
        i = len(lags) - 1 if kind == "hold" else i
        h = lags[i]
        copy.set_xy((box[0] + h - 0.5, box[2] - 0.5))
        ry, rx, _sy, _sx = overlap(box, h, 0)
        shade.set_xy((rx.start + h - 0.5, ry.start - 0.5))
        shade.set_width(rx.stop - rx.start)
        shade.set_height(ry.stop - ry.start)
        arrow.set_positions((cx, cy), (cx + h, cy))
        line_raw.set_data(lags[: i + 1], raw[: i + 1])
        line_ok.set_data(lags[: i + 1], corrected[: i + 1])
        dot.set_data([h], [corrected[i]])
        label.set_text(f"h = {h}:  {100 * pair_fraction(box, h, 0):.0f} % of the voxel pairs are left")
        note.set_text("the raw curve falls with the overlap, not with the texture" if kind == "hold" else "")
        if kind == "hold":
            label.set_text(f"raw = corrected x M(h) / M(0):  {100 * pair_fraction(box, h, 0):.0f} % at h = {h}")
        return copy, shade, arrow, label, note, line_raw, line_ok, dot

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(OUT / "overlap_correction.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)


# ---------------------------------------------------------------------- 2. RVE: concentric boxes
def make_rve_sweep(tex: np.ndarray) -> None:
    region = (20, 140, 15, 105)
    cx, cy = (region[0] + region[1]) // 2, (region[2] + region[3]) // 2
    sizes = [16, 24, 32, 40, 48, 56, 64, 72, 80]
    lengths = []
    for s in sizes:
        b = (cx - s // 2, cx + s // 2, cy - s // 2, cy + s // 2)
        lags = np.arange(0, s // LAG_FRACTION + 1)
        rho = np.array([rho_bb(tex, b, int(h), 0, True) for h in lags])
        lengths.append(length_1e(rho, lags.astype(float)))
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
    frames = [(i, j) for i in range(len(sizes)) for j in range(2)] + [(-1, 0)] * 8

    fig, (ax_img, ax_len) = plt.subplots(
        1, 2, figsize=(7.4, 3.1), dpi=DPI, gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.3}
    )
    fig.set_facecolor(FACE)
    image_axes(ax_img, tex)
    draw_box(ax_img, *region, REGION, ls="--")
    ax_img.text(region[0], region[3] + 3, "region", color=REGION, fontsize=8, va="bottom")
    ghosts = [draw_box(ax_img, cx, cx, cy, cy, COPY, ls=(0, (4, 3)), lw=0.9, alpha=0.0) for _ in sizes]
    for g, s in zip(ghosts, sizes):
        g.set_xy((cx - s // 2 - 0.5, cy - s // 2 - 0.5))
        g.set_width(s)
        g.set_height(s)
    current = draw_box(ax_img, cx, cx, cy, cy, COPY, lw=2.2)
    ax_img.plot([cx], [cy], marker="+", color=COPY, ms=12, mew=2.0)
    label = ax_img.text(0.5, -0.06, "", color=TEXT, fontsize=9, ha="center", va="top", transform=ax_img.transAxes)
    note = ax_img.text(0.5, 1.10, "", color=THRESHOLD, fontsize=8, ha="center", va="bottom", transform=ax_img.transAxes)

    style(ax_len)
    ax_len.set_xlim(sizes[0] - 6, sizes[-1] + 6)
    lo, hi = np.nanmin(lengths), np.nanmax(lengths)
    ax_len.set_ylim(max(0.0, lo - 1.5), hi + 1.5)
    ax_len.set_xlabel("box edge [voxel]", fontsize=9)
    ax_len.set_ylabel("L(1/e) [voxel]", fontsize=9)
    ax_len.grid(color=GRID, alpha=0.5, lw=0.6)
    (pts,) = ax_len.plot([], [], "o-", color=CURVE, lw=1.6, ms=5)
    band = ax_len.add_patch(Rectangle((sizes[0] - 6, 0.0), sizes[-1] - sizes[0] + 12, 0.0, color=CURVE, alpha=0.0, lw=0))
    vline = ax_len.axvline(sizes[0], color=THRESHOLD, ls="--", lw=1.2, alpha=0.0)
    verdict = ax_len.text(0.03, 0.05, "", color=THRESHOLD, fontsize=8, transform=ax_len.transAxes, ha="left", va="bottom")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.2)

    def draw(k):
        i, j = frames[k]
        if i >= 0:
            s = sizes[i]
            current.set_xy((cx - s // 2 - 0.5, cy - s // 2 - 0.5))
            current.set_width(s)
            current.set_height(s)
            current.set_alpha(1.0)
            for g, done in zip(ghosts, range(len(sizes))):
                g.set_alpha(0.5 if done < i else 0.0)
            if j == 0:
                label.set_text(f"box {s} x {s}: lags up to {s // LAG_FRACTION}, its own voxels only")
                shown = i
            else:
                found = np.isfinite(lengths[i])
                label.set_text(f"box {s} x {s}: L(1/e) = {lengths[i]:.2f}" if found else f"box {s} x {s}: 1/e not reached")
                shown = i + 1
            pts.set_data(sizes[:shown], lengths[:shown])
            note.set_text("")
            band.set_alpha(0.0)
            vline.set_alpha(0.0)
            verdict.set_text("")
        else:
            current.set_alpha(0.0)
            for g in ghosts:
                g.set_alpha(0.5)
            pts.set_data(sizes, lengths)
            if stable_i is not None:
                ref, tol = decision.reference, decision.tolerance
                band.set_xy((sizes[0] - 6, ref - tol))
                band.set_height(2 * tol)
                band.set_alpha(0.15)
                vline.set_xdata([sizes[stable_i], sizes[stable_i]])
                vline.set_alpha(1.0)
                current.set_xy((cx - sizes[stable_i] // 2 - 0.5, cy - sizes[stable_i] // 2 - 0.5))
                current.set_width(sizes[stable_i])
                current.set_height(sizes[stable_i])
                current.set_alpha(1.0)
                verdict.set_text(f"stable from {sizes[stable_i]}: the cube for step 3")
                note.set_text(f"the RVE is {sizes[stable_i]} voxel: smaller boxes scatter, larger ones agree")
                label.set_text("")
        return (current, label, pts, band, vline, verdict, note, *ghosts)

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(OUT / "rve_sweep.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)


# ---------------------------------------------------------------------- 3. from the length to the subset
def make_subset(tex: np.ndarray) -> None:
    cx, cy = NX // 2, NY // 2
    edge = 96
    box = (cx - edge // 2, cx + edge // 2, cy - edge // 2, cy + edge // 2)
    lags = np.arange(0, edge // LAG_FRACTION + 1)
    rho = np.array([rho_bb(tex, box, int(h), 0, True) for h in lags])
    L = length_1e(rho, lags.astype(float))
    subset = DEFAULT_FACTOR * L
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
    ax_img.add_patch(Rectangle((-half, -half), subset, subset, fill=False, ec=BOX, lw=2.0))
    ax_img.add_patch(Rectangle((-half + step, -half), subset, subset, fill=False, ec=BOX, lw=1.2, ls="--"))
    ax_img.plot([-half, -half + L], [-half - 3, -half - 3], color=THRESHOLD, lw=3, solid_capstyle="butt")
    ax_img.text(-half + L + 1.5, -half - 3, "L(1/e)", color=THRESHOLD, fontsize=9, ha="left", va="center")
    ax_img.set_title(
        f"subset = {DEFAULT_FACTOR:g} × L = {subset:.0f} voxel,  step = subset / 2 = {step:.0f}", color=BOX, fontsize=9
    )
    ax_img.set_xlim(-crop - 0.5, crop - 0.5)
    ax_img.set_ylim(-crop - 0.5, crop - 0.5)

    style(ax_curve)
    ax_curve.plot(lags, rho, color=CURVE, lw=1.8)
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
    ax_curve.set_xlim(0, lags[-1])
    ax_curve.set_ylim(-0.25, 1.05)
    ax_curve.set_xlabel("lag h [voxel]", fontsize=9)
    ax_curve.set_ylabel("ρ(h)", fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.2)
    fig.savefig(OUT / "subset.png", dpi=DPI, facecolor=FACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stale = OUT / "region_window.gif"  # replaced by overlap_correction.gif
    if stale.is_file():
        stale.unlink()
    tex = speckle()
    make_overlap_correction(speckle(seed=4, sigma=6.0))
    make_rve_sweep(tex)
    make_subset(tex)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
