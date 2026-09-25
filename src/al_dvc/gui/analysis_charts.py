"""The charts of the Statistics tab, drawn into matplotlib figures in the colours of the theme in use.

Qt-free apart from the translated labels, so the report script can draw the same charts.
"""

from __future__ import annotations

import numpy as np

from .i18n import tr
from .names import field_name
from .theme import current_colors

FONT = 7


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "-"
    return f"{v:.{digits}g}"


def palette() -> tuple[str, ...]:
    """Line colours of the curves, one per field."""
    c = current_colors()
    return (c.ACCENT, c.SUCCESS, c.WARNING, c.DANGER, "#38bdf8", "#f472b6")


def style_axes(ax) -> None:
    c = current_colors()
    ax.set_facecolor(c.BG_CANVAS)
    ax.tick_params(colors=c.CANVAS_TEXT, labelsize=FONT)
    for spine in ax.spines.values():
        spine.set_color(c.CANVAS_SPINE)


def _labels(ax, xlabel: str, ylabel: str) -> None:
    c = current_colors()
    ax.set_xlabel(xlabel, color=c.CANVAS_TEXT, fontsize=FONT)
    ax.set_ylabel(ylabel, color=c.CANVAS_TEXT, fontsize=FONT)


def _legend(ax) -> None:
    if ax.get_legend_handles_labels()[0]:
        c = current_colors()
        ax.legend(fontsize=FONT, facecolor=c.BG_PANEL, edgecolor=c.BORDER, labelcolor=c.CANVAS_TEXT)


def draw_histograms(fig, stats: dict, histograms: dict) -> None:
    """One histogram per field (share of the nodes, %), with a normal curve of the same mean and std."""
    fig.clear()
    c = current_colors()
    names = [n for n in stats if histograms.get(n) is not None]
    if not names:
        return
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    for i, name in enumerate(names):
        ax = fig.add_subplot(rows, cols, i + 1)
        style_axes(ax)
        counts, edges = histograms[name]
        st = stats[name]
        width = np.diff(edges)
        share = 100.0 * counts / max(1, counts.sum())
        ax.bar(edges[:-1], share, width=width, align="edge", color=c.ACCENT, alpha=0.75)
        if np.isfinite(st.std) and st.std > 0:
            xs = np.linspace(edges[0], edges[-1], 200)
            pdf = np.exp(-0.5 * ((xs - st.mean) / st.std) ** 2) / (st.std * np.sqrt(2.0 * np.pi))
            ax.plot(xs, 100.0 * pdf * float(np.mean(width)), color=c.WARNING, lw=1.0)
        ax.axvline(st.mean, color=c.SUCCESS, lw=1.0)
        ax.axvline(st.median, color=c.TEXT_PRIMARY, lw=0.8, ls="--")
        ax.set_title(
            f"{field_name(name)}  " + tr("mean {m}, std {s}").format(m=fmt(st.mean), s=fmt(st.std)),
            color=c.CANVAS_TEXT,
            fontsize=FONT,
        )
        ax.set_ylabel("%", color=c.CANVAS_TEXT, fontsize=FONT)
    fig.tight_layout()


def series_curve(series, name: str, reduction: str):
    """``(frames, centre, band)`` of field ``name`` over a series of :class:`FrameStats`; ``band`` is ``(lo, hi)``
    or ``None``. A frame flagged as not ok is a gap (NaN), not a point."""
    frames = np.array([fs.frame + 1 for fs in series])
    ok = np.array([fs.flag == "ok" for fs in series])

    def pick(attr: str) -> np.ndarray:
        v = np.array([getattr(fs.stats[name], attr) if name in fs.stats else np.nan for fs in series], dtype=np.float64)
        v[~ok] = np.nan
        return v

    if reduction in ("mean_std", "median_robust", "mean_ci"):
        centre = pick("median" if reduction == "median_robust" else "mean")
        spread = pick({"mean_std": "std", "median_robust": "robust_std", "mean_ci": "ci95"}[reduction])
        return frames, centre, (centre - spread, centre + spread)
    return frames, pick(reduction), None


def draw_curves(fig, curves, xlabel: str, ylabel: str) -> None:
    """``curves``: ``(label, colour, x, y, band)`` with ``band`` ``(lo, hi)`` or ``None``."""
    fig.clear()
    if not curves:
        return
    ax = fig.add_subplot(1, 1, 1)
    style_axes(ax)
    for label, color, x, y, band in curves:
        ax.plot(x, y, "o-", color=color, lw=1.2, ms=3, label=label)
        if band is not None:
            ax.fill_between(x, band[0], band[1], color=color, alpha=0.18, lw=0)
    _labels(ax, xlabel, ylabel)
    _legend(ax)
    fig.tight_layout()


def draw_region_bars(fig, rows, ylabel: str) -> None:
    """``rows``: ``(label, colour, FieldStats)``; the mean with its 95 % confidence interval, the std as a thin bar."""
    fig.clear()
    c = current_colors()
    if not rows:
        return
    ax = fig.add_subplot(1, 1, 1)
    style_axes(ax)
    x = np.arange(len(rows))
    for i, (_label, color, st) in enumerate(rows):
        if not np.isfinite(st.mean):
            continue
        ax.bar(i, st.mean, width=0.6, color=color, alpha=0.55)
        if np.isfinite(st.std):
            ax.errorbar(i, st.mean, yerr=st.std, color=c.TEXT_SECONDARY, lw=0.8, capsize=2)
        if np.isfinite(st.ci95):
            ax.errorbar(i, st.mean, yerr=st.ci95, color=color, lw=2.2, capsize=5)
    ax.set_xticks(x, [label for label, _c, _s in rows], fontsize=FONT)
    ax.axhline(0.0, color=c.BORDER, lw=0.6)
    _labels(ax, "", ylabel)
    ax.set_title(tr("Mean; thick bar: 95 % confidence interval, thin bar: std"), color=c.CANVAS_TEXT, fontsize=FONT)
    fig.tight_layout()


def draw_profile(fig, current, frame: int, others, xlabel: str, ylabel: str) -> None:
    """The layer means of the current frame (``current``, an :class:`AxisProfile`) with a band of one std, and
    ``others`` -- ``(frame, AxisProfile)`` of every frame -- as thin lines coloured by frame."""
    fig.clear()
    c = current_colors()
    if current is None and not others:
        return
    ax = fig.add_subplot(1, 1, 1)
    style_axes(ax)
    if others:
        import matplotlib as mpl

        cmap = mpl.colormaps["viridis"]
        last = max(k for k, _p in others)
        for k, prof in others:
            ax.plot(prof.positions, prof.mean, color=cmap(k / max(1, last)), lw=0.9, alpha=0.9)
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(1, last + 1), cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label(tr("Frame"), color=c.CANVAS_TEXT, fontsize=FONT)
        cbar.ax.tick_params(colors=c.CANVAS_TEXT, labelsize=FONT)
    if current is not None:
        label = tr("Frame {k}").format(k=frame + 1)
        ax.plot(current.positions, current.mean, "o-", color=c.WARNING, lw=1.6, ms=3, label=label)
        lo, hi = current.mean - current.std, current.mean + current.std
        ax.fill_between(current.positions, lo, hi, color=c.WARNING, alpha=0.18, lw=0)
    _labels(ax, xlabel, ylabel)
    _legend(ax)
    fig.tight_layout()


def draw_line(fig, line, frame: int, series, xlabel: str, ylabel: str) -> None:
    """Left: ``line`` -- ``(distance, values)`` along the segment -- for the current frame, over the other frames in
    grey when ``series`` (a ``LineSeries``) is given. With ``series``, middle: the field at the two ends over the
    frames; right: the extensometer strain ``L / L0 - 1`` over the frames."""
    fig.clear()
    c = current_colors()
    if line is None and series is None:
        return
    cols = 3 if series is not None else 1
    ax = fig.add_subplot(1, cols, 1)
    style_axes(ax)
    if series is not None:
        for k, row in zip(series.frames, series.values):
            if k != frame:
                ax.plot(series.distance, row, "-", color=c.TEXT_MUTED, lw=0.7, alpha=0.6)
    if line is not None:
        distance, values = line
        ax.plot(distance, values, "-", color=c.ACCENT, lw=1.6, label=tr("Frame {k}").format(k=frame + 1))
    _labels(ax, xlabel, ylabel)
    _legend(ax)
    if series is not None:
        frames = series.frames + 1
        bx = fig.add_subplot(1, cols, 2)
        style_axes(bx)
        bx.plot(frames, series.ends[:, 0], "o-", color=c.ACCENT, lw=1.2, ms=3, label=tr("First point"))
        bx.plot(frames, series.ends[:, 1], "s-", color=c.WARNING, lw=1.2, ms=3, label=tr("Second point"))
        bx.axvline(frame + 1, color=c.BORDER, lw=0.6)
        _labels(bx, tr("Frame"), ylabel)
        _legend(bx)
        ext = series.extensometer
        cx = fig.add_subplot(1, cols, 3)
        style_axes(cx)
        cx.plot(ext.frames + 1, ext.strain, "o-", color=c.SUCCESS, lw=1.2, ms=3)
        cx.axhline(0.0, color=c.BORDER, lw=0.6)
        cx.axvline(frame + 1, color=c.BORDER, lw=0.6)
        _labels(cx, tr("Frame"), tr("Extensometer strain L/L0 - 1"))
        cx.set_title(tr("L0 = {L0}").format(L0=fmt(ext.L0)), color=c.CANVAS_TEXT, fontsize=FONT)
    fig.tight_layout()
