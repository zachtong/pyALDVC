"""Matplotlib drawing of a node-grid field on the three orthogonal planes.

Shared by the GUI canvases (slice viewer, strain window) and the image export, so a PNG written
from the export dialog looks like the screen. No Qt here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from al_dvc.core.data_structures import PipelineResult

from .export_utils import field_array

LAYOUTS = ("row", "column", "grid")  # three planes side by side, stacked, or XY / XZ left and YZ top-right
PLANES = ("xy", "xz", "yz")
DARK_BG = "#0c0d12"
DARK_FG = "#94a3b8"
DARK_BORDER = "#1e293b"

FIELD_LABELS = {
    "disp_u": "u",
    "disp_v": "v",
    "disp_w": "w",
    "disp_magnitude": "|u|",
    "disp_std_u": "std u",
    "disp_std_v": "std v",
    "disp_std_w": "std w",
    "disp_std": "std |u|",
    "exx": "exx",
    "eyy": "eyy",
    "ezz": "ezz",
    "exy": "exy",
    "exz": "exz",
    "eyz": "eyz",
    "e1": "e1 (max principal)",
    "e2": "e2",
    "e3": "e3 (min principal)",
    "max_shear": "max shear",
    "von_mises": "von Mises",
    "volumetric": "volumetric",
    "det_F": "det F",
    "rotation_deg": "rotation [deg]",
}
DISPLACEMENT_LIKE = ("disp_u", "disp_v", "disp_w", "disp_magnitude", "disp_std_u", "disp_std_v", "disp_std_w", "disp_std")


def field_label(name: str, units: str = "voxel") -> str:
    """Axis / colorbar label of a field: displacement-like fields carry the length unit."""
    base = FIELD_LABELS.get(name, name)
    if name in DISPLACEMENT_LIKE:
        return f"{base} [{units}]"
    if name == "rotation_deg":
        return base
    return base


@dataclass(frozen=True)
class PlaneStyle:
    """Colours of the figure (dark by default, matching the application theme)."""

    background: str = DARK_BG
    text: str = DARK_FG
    border: str = DARK_BORDER
    cursor: str = "#6366f1"


LIGHT_STYLE = PlaneStyle(background="white", text="#333333", border="#cccccc", cursor="#4f46e5")


def build_axes(fig, layout: str):
    """Three image axes (XY, XZ, YZ in that order) and one colorbar axes for ``layout``."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    fig.clear()
    if layout == "row":
        gs = fig.add_gridspec(1, 3, left=0.05, right=0.90, bottom=0.14, top=0.90, wspace=0.32)
        cells = [gs[0, 0], gs[0, 1], gs[0, 2]]
        cax_rect = (0.925, 0.16, 0.014, 0.68)
    elif layout == "column":
        gs = fig.add_gridspec(3, 1, left=0.14, right=0.86, bottom=0.05, top=0.96, hspace=0.42)
        cells = [gs[0, 0], gs[1, 0], gs[2, 0]]
        cax_rect = (0.90, 0.22, 0.02, 0.56)
    else:
        gs = fig.add_gridspec(2, 2, left=0.07, right=0.94, bottom=0.08, top=0.94, wspace=0.28, hspace=0.34)
        cells = [gs[0, 0], gs[1, 0], gs[0, 1]]
        cax_rect = (0.60, 0.10, 0.02, 0.32)
    axes = [fig.add_subplot(cell) for cell in cells]
    for ax in axes:
        ax.pyaldvc_cell = tuple(ax.get_position().bounds)  # remembered for apply_equal_scale / restore_cells
    cax = fig.add_axes(cax_rect)
    cax.set_visible(False)
    return axes, cax


def auto_range(values: NDArray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    """Percentile colour range of the finite values (``(0, 1)`` when nothing is finite)."""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.percentile(finite, low)), float(np.percentile(finite, high))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def slice_indices(shape: tuple[int, int, int], indices: dict[str, int | None] | None) -> tuple[int, int, int]:
    """``(iz, iy, ix)`` clipped to ``shape``; ``None`` entries fall back to the middle."""
    nz, ny, nx = shape
    idx = indices or {}

    def one(axis: str, n: int) -> int:
        v = idx.get(axis)
        return int(np.clip(n // 2 if v is None else v, 0, n - 1))

    return one("z", nz), one("y", ny), one("x", nx)


def apply_equal_scale(fig, axes, sizes) -> float:
    """Give the three panes the same voxels-per-pixel scale.

    ``sizes`` are the ``(width, height)`` voxel extents shown by each axes. The common scale is the
    largest at which every slice still fits the cell its layout gave it; each axes is then shrunk to
    exactly its slice at that scale and centred in its cell, so nothing is magnified and no pane
    shows empty axis range. Returns the scale in pixels per voxel. :func:`restore_cells` undoes it.
    """
    fw, fh = fig.get_size_inches() * fig.dpi
    cells = []
    for ax in axes:
        cell = getattr(ax, "pyaldvc_cell", None)
        if cell is None:
            cell = tuple(ax.get_position().bounds)
            ax.pyaldvc_cell = cell
        cells.append(cell)
    scale = min(min(c[2] * fw / w, c[3] * fh / h) for c, (w, h) in zip(cells, sizes) if w > 0 and h > 0)
    for ax, cell, (w, h) in zip(axes, cells, sizes):
        bw, bh = w * scale / fw, h * scale / fh  # figure fractions
        x = cell[0] + (cell[2] - bw) / 2.0
        y = cell[1] + (cell[3] - bh) / 2.0
        ax.set_position([x, y, bw, bh])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)
    return float(scale)


def restore_cells(axes) -> None:
    """Put axes back into the cells their layout gave them (after :func:`apply_equal_scale`)."""
    for ax in axes:
        cell = getattr(ax, "pyaldvc_cell", None)
        if cell is not None:
            ax.set_position(list(cell))
            ax.set_aspect("equal", adjustable="box")


def draw_field_planes(
    axes,
    cax,
    result: PipelineResult,
    frame: int,
    field: str,
    indices: dict[str, int | None] | None = None,
    cmap: str = "turbo",
    clim: tuple[float, float] | None = None,
    background: NDArray | None = None,
    alpha: float = 0.85,
    style: PlaneStyle = PlaneStyle(),
    volume_shape: tuple[int, int, int] | None = None,
    label_units: str | None = None,
    equal_scale: bool = False,
) -> dict:
    """Draw ``field`` of ``frame`` on the XY / XZ / YZ planes through ``indices`` (voxel positions).

    ``background`` is an optional volume drawn in grey under the field. Without one the panes
    still span the full volume (``volume_shape``, else the result's) so the field sits where it
    belongs in the scan. Returns ``{"clim", "indices", "mappable"}``.
    """
    mesh = result.dvc_mesh
    shape = tuple(int(s) for s in (volume_shape or result.volume_shape))
    if background is not None and tuple(background.shape) != shape:
        raise ValueError(f"background shape {background.shape} differs from the volume shape {shape}")
    iz, iy, ix = slice_indices(shape, indices)
    values = field_array(result, frame, field)
    grid = mesh.to_grid(values)
    lo, hi = clim if clim is not None else auto_range(grid)
    x0, y0, z0 = mesh.x0, mesh.y0, mesh.z0
    hx, hy, hz = mesh.spacing
    kz = int(np.argmin(np.abs(z0 - iz)))
    ky = int(np.argmin(np.abs(y0 - iy)))
    kx = int(np.argmin(np.abs(x0 - ix)))
    nz, ny, nx = shape
    panes = [
        (
            axes[0],
            grid[kz],
            [x0[0] - hx / 2, x0[-1] + hx / 2, y0[0] - hy / 2, y0[-1] + hy / 2],
            (nx, ny),
            "x",
            "y",
            f"XY  z = {iz}",
        ),
        (
            axes[1],
            grid[:, ky, :],
            [x0[0] - hx / 2, x0[-1] + hx / 2, z0[0] - hz / 2, z0[-1] + hz / 2],
            (nx, nz),
            "x",
            "z",
            f"XZ  y = {iy}",
        ),
        (
            axes[2],
            grid[:, :, kx],
            [y0[0] - hy / 2, y0[-1] + hy / 2, z0[0] - hz / 2, z0[-1] + hz / 2],
            (ny, nz),
            "y",
            "z",
            f"YZ  x = {ix}",
        ),
    ]
    bg_slices = None
    if background is not None:
        bg = np.asarray(background)
        finite = bg[np.isfinite(bg)] if bg.dtype.kind == "f" else bg.ravel()
        sample = finite.ravel()[:: max(1, finite.size // 200_000)] if finite.size else np.zeros(1)
        vmin, vmax = float(np.percentile(sample, 0.5)), float(np.percentile(sample, 99.5))
        bg_slices = [(bg[iz], vmin, vmax), (bg[:, iy, :], vmin, vmax), (bg[:, :, ix], vmin, vmax)]
    mappable = None
    for k, (ax, img, extent, (w, h), xl, yl, title) in enumerate(panes):
        ax.clear()
        ax.set_facecolor(style.background)
        ax.tick_params(colors=style.text, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(style.border)
        if bg_slices is not None:
            b, vmin, vmax = bg_slices[k]
            ax.imshow(b, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=[-0.5, w - 0.5, -0.5, h - 0.5])
        mappable = ax.imshow(
            np.ma.masked_invalid(img),
            cmap=cmap,
            origin="lower",
            vmin=lo,
            vmax=hi,
            alpha=alpha if bg_slices is not None else 1.0,
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(title, color=style.text, fontsize=8)
        ax.set_xlabel(xl, color=style.text, fontsize=7)
        ax.set_ylabel(yl, color=style.text, fontsize=7)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)
    if equal_scale:
        apply_equal_scale(axes[0].figure, axes, [size for _ax, _img, _ext, size, *_rest in panes])
    else:
        restore_cells(axes)
    axes[0].axhline(iy, color=style.cursor, lw=0.5, alpha=0.6)
    axes[0].axvline(ix, color=style.cursor, lw=0.5, alpha=0.6)
    axes[1].axhline(iz, color=style.cursor, lw=0.5, alpha=0.6)
    axes[1].axvline(ix, color=style.cursor, lw=0.5, alpha=0.6)
    axes[2].axhline(iz, color=style.cursor, lw=0.5, alpha=0.6)
    axes[2].axvline(iy, color=style.cursor, lw=0.5, alpha=0.6)
    cax.clear()
    cax.set_visible(True)
    fig = axes[0].figure
    cbar = fig.colorbar(mappable, cax=cax)
    units = label_units if label_units is not None else getattr(result.dvc_para, "units", "voxel")
    cbar.set_label(field_label(field, units), color=style.text, fontsize=8)
    cbar.ax.tick_params(colors=style.text, labelsize=7)
    return {"clim": (lo, hi), "indices": (iz, iy, ix), "mappable": mappable}


def export_field_images(
    result: PipelineResult,
    out_dir,
    fields: list[str],
    frames: list[int] | None = None,
    layout: str = "row",
    cmap: str = "turbo",
    clim: tuple[float, float] | None = None,
    indices: dict[str, int | None] | None = None,
    background: NDArray | None = None,
    dpi: int = 150,
    light: bool = True,
    progress_fn=None,
    equal_scale: bool = False,
) -> list:
    """Write one PNG per field and frame (``<field>_frame_<k>.png``) with the three planes.

    ``light`` uses white paper colours (for documents); otherwise the application's dark look.
    """
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(result.result_disp)
    frames = list(range(n)) if frames is None else [int(k) for k in frames if 0 <= int(k) < n]
    style = LIGHT_STYLE if light else PlaneStyle()
    written = []
    total = max(1, len(fields) * len(frames))
    done = 0
    for field in fields:
        for k in frames:
            fig = Figure(figsize=(12, 4.2) if layout == "row" else (7, 9), facecolor=style.background if not light else "white")
            axes, cax = build_axes(fig, layout)
            draw_field_planes(axes, cax, result, k, field, indices, cmap, clim, background, style=style, equal_scale=equal_scale)
            path = out / f"{field}_frame_{k + 1:03d}.png"
            fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
            written.append(path)
            done += 1
            if progress_fn is not None:
                progress_fn(done / total, f"{field} frame {k + 1}")
    return written
