"""Build pyvista scenes from DVC results (no Qt: usable for tests, reports and scripts).

The node grid of a result is a regular lattice, so it maps one-to-one onto a
``pyvista.ImageData`` with the node spacing as voxel spacing and the first
node as origin. Node ordering ``n = iz*ny*nx + iy*nx + ix`` is exactly VTK's
point ordering (x fastest), so per-node arrays attach without reordering.

``pyvista`` is imported lazily: the module can be imported (and
:func:`available` queried) on installations without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from al_dvc.core.data_structures import PipelineResult
from al_dvc.export.export_utils import displacement_physical, field_array

if TYPE_CHECKING:  # pragma: no cover
    import pyvista as pv

MODES = ("slices", "points", "surface", "warped")
DISPLACEMENT_ARRAY = "displacement"
MAX_ARROWS = 20_000
ARROW_SCALE_FRACTION = 0.6  # longest arrow = this fraction of the node spacing at unit arrow_scale
VOLUME_SLICE_MAX_PIXELS = 4_000_000  # subsample larger image slices before turning them into textures
CAMERAS = ("iso", "xy", "xz", "yz")
BACKGROUND = "#1b1d23"
BACKGROUNDS = {"dark": BACKGROUND, "black": "#000000", "grey": "#808080", "white": "#ffffff"}


def foreground_for(background: str) -> str:
    """White text on a dark background, black on a light one (scalar bar, axes triad)."""
    try:
        h = background.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return "white"
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "white"


def import_error() -> str | None:
    """Why pyvista cannot be imported (``None`` when it can): shown by the self-test and the panel."""
    try:
        import pyvista  # noqa: F401
    except Exception as exc:  # ImportError, or a broken VTK install raising something else
        return f"{type(exc).__name__}: {exc}"
    return None


def available() -> bool:
    """True when pyvista can be imported."""
    return import_error() is None


@dataclass(frozen=True)
class SceneOptions:
    """What to draw. Defaults match the slice viewer's overlay."""

    field: str = "disp_magnitude"
    frame: int = 0
    mode: str = "slices"
    colormap: str = "turbo"
    clim: tuple[float, float] | None = None
    opacity: float = 1.0
    warp_scale: float = 1.0
    show_arrows: bool = False
    arrow_stride: int = 2
    arrow_scale: float = 1.0
    show_outline: bool = True
    show_volume_slices: bool = False
    iso_fraction: float = 0.5
    slice_index: dict[str, int | None] = dc_field(default_factory=dict)
    background: str = BACKGROUND
    title: str | None = None  # scalar bar title; the field key when None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.arrow_stride < 1:
            raise ValueError("arrow_stride must be >= 1")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be within [0, 1]")
        if not 0.0 <= self.iso_fraction <= 1.0:
            raise ValueError("iso_fraction must be within [0, 1]")


@dataclass
class SceneInfo:
    """What :func:`build_scene` drew, for labels and tests."""

    field: str
    clim: tuple[float, float]
    n_nodes: int
    n_finite: int
    n_arrows: int = 0
    actors: dict[str, Any] = dc_field(default_factory=dict)


# ----------------------------------------------------------------------------- datasets
def node_grid(result: PipelineResult, frame: int, fields: tuple[str, ...] = ("disp_magnitude",)) -> "pv.ImageData":
    """``ImageData`` over the node lattice with the requested per-node fields and the displacement vectors."""
    import pyvista as pv

    mesh = result.dvc_mesh
    nz, ny, nx = mesh.grid_shape
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=tuple(float(s) for s in mesh.spacing),
        origin=(float(mesh.x0[0]), float(mesh.y0[0]), float(mesh.z0[0])),
    )
    if grid.n_points != mesh.n_nodes:
        raise ValueError(f"node grid {mesh.grid_shape} has {grid.n_points} lattice points but the mesh has {mesh.n_nodes} nodes")
    fr = result.result_disp[frame]
    for name in fields:
        grid.point_data[name] = np.asarray(field_array(result, frame, name), dtype=np.float64)
    U = np.asarray(displacement_physical(fr, result.dvc_para.voxel_size), dtype=np.float64)
    grid.point_data[DISPLACEMENT_ARRAY] = np.nan_to_num(U)
    return grid


def volume_slice_planes(
    volume: NDArray, slice_index: dict[str, int | None], voxel_size=(1.0, 1.0, 1.0)
) -> dict[str, "pv.ImageData"]:
    """The XY, XZ and YZ planes of an image volume as single-layer ``ImageData`` (intensity in ``point_data['intensity']``)."""
    import pyvista as pv

    vol = np.asarray(volume)
    nz, ny, nx = vol.shape
    sx, sy, sz = (float(v) for v in voxel_size)
    iz = int(np.clip(slice_index.get("z") if slice_index.get("z") is not None else nz // 2, 0, nz - 1))
    iy = int(np.clip(slice_index.get("y") if slice_index.get("y") is not None else ny // 2, 0, ny - 1))
    ix = int(np.clip(slice_index.get("x") if slice_index.get("x") is not None else nx // 2, 0, nx - 1))
    planes = {}
    specs = {
        "xy": (vol[iz], (nx, ny, 1), (sx, sy, sz), (0.0, 0.0, iz * sz)),
        "xz": (vol[:, iy, :], (nx, 1, nz), (sx, sy, sz), (0.0, iy * sy, 0.0)),
        "yz": (vol[:, :, ix], (1, ny, nz), (sx, sy, sz), (ix * sx, 0.0, 0.0)),
    }
    for key, (img, dims, spacing, origin) in specs.items():
        img = np.asarray(img, dtype=np.float32)
        step = int(np.ceil(np.sqrt(img.size / VOLUME_SLICE_MAX_PIXELS))) if img.size > VOLUME_SLICE_MAX_PIXELS else 1
        if step > 1:
            img = img[::step, ::step]
            dims = tuple(int(d) if d == 1 else int(np.ceil(d / step)) for d in dims)
            spacing = _scaled_spacing(key, (sx, sy, sz), step)
        plane = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
        plane.point_data["intensity"] = img.ravel()  # rows (slow axis) map onto the second lattice axis: x fastest
        planes[key] = plane
    return planes


def _scaled_spacing(key: str, spacing: tuple[float, float, float], step: int) -> tuple[float, float, float]:
    sx, sy, sz = spacing
    if key == "xy":
        return (sx * step, sy * step, sz)
    if key == "xz":
        return (sx * step, sy, sz * step)
    return (sx, sy * step, sz * step)


def auto_clim(values: NDArray) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


# ----------------------------------------------------------------------------- scene
def build_scene(plotter, result: PipelineResult, opts: SceneOptions, volume: NDArray | None = None) -> SceneInfo:
    """Clear ``plotter`` and draw the result according to ``opts``; returns what was drawn."""
    import pyvista as pv

    plotter.clear()
    frame = int(np.clip(opts.frame, 0, len(result.result_disp) - 1))
    grid = node_grid(result, frame, (opts.field,))
    values = np.asarray(grid.point_data[opts.field])
    finite = np.isfinite(values)
    clim = opts.clim if opts.clim is not None else auto_clim(values)
    info = SceneInfo(field=opts.field, clim=clim, n_nodes=int(values.size), n_finite=int(finite.sum()))
    common = dict(scalars=opts.field, cmap=opts.colormap, clim=clim, nan_opacity=0.0, show_scalar_bar=False)
    fg = foreground_for(opts.background)
    plotter.set_background(opts.background)
    scalar_bar = dict(  # a slim vertical bar centred on the right edge, plain sans-serif text
        title=(opts.title or opts.field) + chr(10),  # the newline keeps the title clear of the top label
        vertical=True,
        n_labels=5,
        fmt="%.3g",
        width=0.07,
        height=0.55,
        position_x=0.90,
        position_y=0.22,
        title_font_size=13,
        label_font_size=11,
        unconstrained_font_size=True,
        font_family="arial",
        color=fg,
        shadow=False,
        italic=False,
        bold=False,
    )

    if opts.mode == "slices":
        x, y, z = _slice_positions(result, opts.slice_index)
        sliced = grid.slice_orthogonal(x=x, y=y, z=z)
        info.actors["field"] = plotter.add_mesh(sliced, opacity=opts.opacity, **common)
    elif opts.mode == "points":
        idx = np.flatnonzero(finite)
        if idx.size:
            cloud = pv.PolyData(np.asarray(grid.points)[idx])
            cloud.point_data[opts.field] = values[idx]
            info.actors["field"] = plotter.add_mesh(
                cloud, style="points", point_size=8.0, render_points_as_spheres=True, opacity=opts.opacity, **common
            )
    elif opts.mode == "surface":
        lo, hi = clim
        level = lo + opts.iso_fraction * (hi - lo)
        filled = grid.copy()
        filled.point_data[opts.field] = np.where(finite, values, lo - 1.0)
        surf = filled.contour(isosurfaces=[level], scalars=opts.field)
        if surf.n_points:
            surf = surf.compute_normals(auto_orient_normals=True, consistent_normals=True)
            info.actors["field"] = plotter.add_mesh(surf, opacity=opts.opacity, smooth_shading=True, specular=0.3, **common)
        info.actors["iso_level"] = level
    else:  # warped: the lattice of the valid nodes moved by the displacement, drawn with its cell edges
        grid.point_data["_valid"] = finite.astype(np.float32)
        cells = grid.threshold(0.5, scalars="_valid", preference="point")  # cells whose 8 nodes are valid
        if cells.n_cells:
            warped = cells.warp_by_vector(DISPLACEMENT_ARRAY, factor=opts.warp_scale)
            info.actors["field"] = plotter.add_mesh(
                warped, opacity=opts.opacity, show_edges=True, edge_color=fg, line_width=1, **common
            )
        if opts.show_outline:
            info.actors["original"] = plotter.add_mesh(grid.outline(), color="gray", line_width=1)

    if opts.show_arrows:
        info.n_arrows = _add_arrows(plotter, grid, finite, result, opts, info)
    if opts.show_outline:
        info.actors["outline"] = plotter.add_mesh(_volume_outline(result), color=fg, line_width=1, opacity=0.6)
    if opts.show_volume_slices and volume is not None:
        planes = volume_slice_planes(volume, opts.slice_index, result.dvc_para.voxel_size)
        for key, plane in planes.items():
            info.actors[f"volume_{key}"] = plotter.add_mesh(
                plane, scalars="intensity", cmap="gray", show_scalar_bar=False, opacity=0.9
            )
    if "field" in info.actors:
        # bind the bar to the field's mapper explicitly: by default pyvista takes the last added mesh (outline, arrows)
        plotter.add_scalar_bar(mapper=info.actors["field"].mapper, **scalar_bar)
    plotter.add_axes(color=fg)
    return info


def _slice_positions(result: PipelineResult, slice_index: dict[str, int | None]) -> tuple[float, float, float]:
    """World coordinates of the three orthogonal slices (physical units via voxel_size)."""
    mesh = result.dvc_mesh
    nz, ny, nx = result.volume_shape
    vs = np.asarray(result.dvc_para.voxel_size, dtype=np.float64)
    ix = slice_index.get("x") if slice_index.get("x") is not None else nx // 2
    iy = slice_index.get("y") if slice_index.get("y") is not None else ny // 2
    iz = slice_index.get("z") if slice_index.get("z") is not None else nz // 2
    x = float(np.clip(ix, mesh.x0[0], mesh.x0[-1]))
    y = float(np.clip(iy, mesh.y0[0], mesh.y0[-1]))
    z = float(np.clip(iz, mesh.z0[0], mesh.z0[-1]))
    return x * vs[0], y * vs[1], z * vs[2]


def _volume_outline(result: PipelineResult) -> "pv.PolyData":
    import pyvista as pv

    nz, ny, nx = result.volume_shape
    sx, sy, sz = (float(v) for v in result.dvc_para.voxel_size)
    box = pv.ImageData(dimensions=(2, 2, 2), spacing=((nx - 1) * sx, (ny - 1) * sy, (nz - 1) * sz), origin=(0.0, 0.0, 0.0))
    return box.outline()


def _add_arrows(plotter, grid, finite: NDArray[np.bool_], result: PipelineResult, opts: SceneOptions, info: SceneInfo) -> int:
    import pyvista as pv

    nz, ny, nx = result.dvc_mesh.grid_shape
    keep = np.zeros((nz, ny, nx), dtype=bool)
    s = opts.arrow_stride
    keep[::s, ::s, ::s] = True
    idx = np.flatnonzero(keep.ravel() & finite)
    if idx.size > MAX_ARROWS:
        idx = idx[:: int(np.ceil(idx.size / MAX_ARROWS))]
    if idx.size == 0:
        return 0
    vec = np.asarray(grid.point_data[DISPLACEMENT_ARRAY])[idx]
    mag = np.linalg.norm(vec, axis=1)
    longest = float(mag.max()) if mag.size else 0.0
    if longest <= 0.0:
        return 0
    spacing = float(min(result.dvc_mesh.spacing)) * float(min(result.dvc_para.voxel_size))
    factor = ARROW_SCALE_FRACTION * spacing / longest * opts.arrow_scale
    cloud = pv.PolyData(np.asarray(grid.points)[idx])
    cloud.point_data["vec"] = vec
    cloud.point_data["mag"] = mag
    arrows = cloud.glyph(orient="vec", scale="mag", factor=factor)
    info.actors["arrows"] = plotter.add_mesh(arrows, scalars="mag", cmap=opts.colormap, show_scalar_bar=False)
    return int(idx.size)


def render_image(
    result: PipelineResult,
    opts: SceneOptions | None = None,
    volume: NDArray | None = None,
    window_size: tuple[int, int] = (900, 700),
    camera: str = "iso",
    path=None,
) -> tuple[NDArray[np.uint8], SceneInfo]:
    """Render a scene off-screen; returns ``(rgb image (h, w, 3), info)`` and writes ``path`` when given."""
    import pyvista as pv

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    opts = opts or SceneOptions()
    pl.set_background(opts.background)
    info = build_scene(pl, result, opts, volume)
    _apply_camera(pl, camera)
    img = np.asarray(pl.screenshot(str(path) if path is not None else None, return_img=True))
    pl.close()
    return img[..., :3].copy(), info


def render_png(
    result: PipelineResult,
    path,
    opts: SceneOptions | None = None,
    volume: NDArray | None = None,
    window_size: tuple[int, int] = (900, 700),
    camera: str = "iso",
) -> SceneInfo:
    """Render a scene off-screen to ``path``; the same code the panel's static backend uses."""
    return render_image(result, opts, volume, window_size, camera, path=path)[1]


def _apply_camera(plotter, camera: str) -> None:
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS}, got {camera!r}")
    if camera == "iso":
        plotter.view_isometric()
    else:
        getattr(plotter, f"view_{camera}")()
    plotter.reset_camera()
