"""Matplotlib orthogonal-slice views of node-grid fields."""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from ..core.data_structures import DVCMesh  # noqa: E402


def _extent(a: NDArray, b: NDArray, spacing_a: float, spacing_b: float):
    return [a[0] - spacing_a / 2, a[-1] + spacing_a / 2, b[-1] + spacing_b / 2, b[0] - spacing_b / 2]


def plot_field_slices(
    grid: NDArray[np.float64],
    mesh: DVCMesh,
    title: str = "",
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    slices: tuple[int, int, int] | None = None,
    fig=None,
    axes=None,
    units: str = "",
):
    """Plot the ``z``, ``y`` and ``x`` mid-slices of a ``(nz, ny, nx)`` grid field."""
    g = np.asarray(grid, dtype=np.float64)
    nz, ny, nx = g.shape
    if slices is None:
        slices = (nz // 2, ny // 2, nx // 2)
    kz, jy, ix = slices
    finite = g[np.isfinite(g)]
    if vmin is None or vmax is None:
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99])
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
            if vmax <= vmin:
                vmax = vmin + 1e-12
        else:
            vmin, vmax = 0.0, 1.0
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    hx, hy, hz = mesh.spacing
    panels = [
        (g[kz, :, :], f"z = {mesh.z0[kz]:.0f}", "x", "y", _extent(mesh.x0, mesh.y0, hx, hy)),
        (g[:, jy, :], f"y = {mesh.y0[jy]:.0f}", "x", "z", _extent(mesh.x0, mesh.z0, hx, hz)),
        (g[:, :, ix], f"x = {mesh.x0[ix]:.0f}", "y", "z", _extent(mesh.y0, mesh.z0, hy, hz)),
    ]
    im = None
    for ax, (img, sub, xl, yl, ext) in zip(axes, panels):
        im = ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, origin="upper",
                       interpolation="nearest", aspect="equal")
        ax.set_title(f"{title} @ {sub}", fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    if im is not None:
        cb = fig.colorbar(im, ax=list(axes), shrink=0.9, pad=0.02)
        if units:
            cb.set_label(units)
    return fig, axes


def plot_volume_slices(vol: NDArray, title: str = "", cmap: str = "gray", fig=None, axes=None):
    """Mid-slices of a raw ``(nz, ny, nx)`` volume."""
    v = np.asarray(vol)
    nz, ny, nx = v.shape
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (img, sub) in zip(axes, [(v[nz // 2], "z"), (v[:, ny // 2, :], "y"), (v[:, :, nx // 2], "x")]):
        ax.imshow(img, cmap=cmap, origin="upper", interpolation="nearest", aspect="equal")
        ax.set_title(f"{title} mid-{sub} slice", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


def plot_nodes_field(mesh: DVCMesh, values: NDArray, **kwargs):
    """Convenience: reshape a per-node array and call :func:`plot_field_slices`."""
    return plot_field_slices(mesh.to_grid(np.asarray(values, dtype=np.float64)), mesh, **kwargs)


def histogram_panel(ax, values: Sequence[float] | NDArray, title: str, bins: int = 60, gt: float | None = None):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.hist(v, bins=bins, color="#4c72b0", alpha=0.85)
    ax.set_title(f"{title}\nmean {v.mean():.4g}, sd {v.std():.3g}", fontsize=9)
    if gt is not None:
        ax.axvline(gt, color="k", ls="--", lw=1)
